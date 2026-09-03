"""Benchmark adapter for the nnPU (PU-BCE) failure discriminator.

Sits on top of :class:`PolicyBenchmarkDiscriminator` so frozen policy encoding
and the trajectory feature cache are reused unchanged. Replaces any per-task
scorer with a single shared :class:`PUBCEDiscriminator` trained with the
non-negative PU risk on (positives = pre-done success frames, unlabeled = WHOLE
failure trajectories), plus per-task success_percentile thresholds.

Success trajectories (train + eval) use only frames before task completion,
determined by the per-frame ``is_success`` label (``False`` = still executing;
first ``True`` marks done). Failure rollouts in the unlabeled pool are used whole.

**No GT failure timing is used in this branch.** Each failure-rollout trajectory
is pooled into the unlabeled set as a whole -- ``first_gt_failure_frame()`` is
never consulted.

Hard invariants:
  * No ``video_id`` in ``unlabeled_fail_trajectories`` may appear in the eval
    failure set passed to ``fit_on_benchmark``. Enforced by
    :meth:`_assert_disjoint_unlabeled_eval_fail` (hard assert).
  * Train success ``video_id``s must not appear in the eval success set when a
    separate train pool is supplied (:meth:`_assert_disjoint_train_eval_success`).
  * ``fit_on_benchmark`` does **not** call ``bench.evaluate(...)`` or compute
    any AUROC / metric over eval trajectories. Evaluation is the caller's job
    (see ``robosuite_pu_bce.py``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from benchmark.core import BenchmarkTrajectory, DiscriminatorOutput

from robosuite.discriminator.dyn_disc.adapters.single_bank import (
    PolicyBenchmarkDiscriminator,
    _pad_to_length,
)
from robosuite.discriminator.dyn_disc.detectors.pu_bce import PUBCEDiscriminator


def _pad_to_length_with_fill(
    values: np.ndarray,
    *,
    target_len: int,
    fill_value: float | int,
    dtype=np.float32,
) -> np.ndarray:
    """Pad a prefix-length array to ``target_len`` with a constant fill value."""
    arr = np.asarray(values, dtype=dtype).reshape(-1)
    n = int(arr.shape[0])
    T = int(target_len)
    if n >= T:
        return arr[:T].copy()
    if n == 0:
        return np.full((T,), fill_value, dtype=dtype)
    pad = np.full((T - n,), fill_value, dtype=dtype)
    return np.concatenate([arr, pad], axis=0)


class PUBCEBenchmarkDiscriminator(PolicyBenchmarkDiscriminator):
    """nnPU failure detector (no GT timing, success_percentile calib) over the benchmark API."""

    name = "policy_disc_pu_bce"

    def __init__(
        self,
        *,
        policy_ckpt: str,
        unlabeled_fail_trajectories: Sequence[BenchmarkTrajectory],
        pi_p: float = 0.5,
        head_hidden: int = 256,
        head_layers: int = 2,
        epochs: int = 20,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        batch_size: int = 512,
        loss_surrogate: str = "sigmoid",
        nn_correction: bool = True,
        beta: float = 0.0,
        save_ckpt_dir: Optional[str] = None,
        # forwarded to the shared parent for encoding / cache parity
        device: str = "cuda",
        encode_batch_size: int = 128,
        delta: float = 10.0,
        preload_workers: int = 4,
        prefetch_factor: int = 2,
        pin_memory: bool = True,
        feature_cache_dir: Optional[str] = None,
        reuse_feature_cache: bool = True,
        calib_fraction: float = 0.2,
        seed: int = 0,
        verbose_fit: bool = True,
    ) -> None:
        super().__init__(
            policy_ckpt=policy_ckpt,
            device=device,
            encode_batch_size=encode_batch_size,
            preload_workers=preload_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
            feature_cache_dir=feature_cache_dir,
            reuse_feature_cache=reuse_feature_cache,
            calib_fraction=calib_fraction,
            seed=seed,
            verbose_fit=verbose_fit,
        )
        self.unlabeled_fail_trajectories: List[BenchmarkTrajectory] = list(unlabeled_fail_trajectories)

        if not (0.0 < float(pi_p) < 1.0):
            raise ValueError(f"pi_p must be in (0, 1), got {pi_p}")
        self.pi_p = float(pi_p)
        self.head_hidden = int(head_hidden)
        self.head_layers = int(head_layers)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.loss_surrogate = str(loss_surrogate)
        self.nn_correction = bool(nn_correction)
        self.beta = float(beta)
        self.delta = float(delta)
        self.save_ckpt_dir = None if save_ckpt_dir is None else str(save_ckpt_dir)

        # Single shared detector across tasks (constructed in fit_on_benchmark
        # once the encoder's feature dim is known).
        self._shared_detector: Optional[PUBCEDiscriminator] = None
        self._detectors_per_task: Dict[str, PUBCEDiscriminator] = {}
        self._calibration_stats: Dict[str, dict] = {}
        self._global_stats: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _assert_disjoint_unlabeled_eval_fail(
        eval_trajs: Sequence[BenchmarkTrajectory],
        unlabeled_fail_trajs: Sequence[BenchmarkTrajectory],
    ) -> None:
        """Unlabeled failure pool must not overlap eval failure ``video_id``s."""
        eval_fail_keys = {
            str(t.video_id) for t in eval_trajs if bool(t.is_failure)
        }
        unlabeled_keys = {str(t.video_id) for t in unlabeled_fail_trajs}
        overlap = sorted(eval_fail_keys & unlabeled_keys)
        if overlap:
            raise RuntimeError(
                "PUBCEBenchmarkDiscriminator disjointness invariant violated:\n  - "
                f"eval_fail intersect unlabeled_fail = {overlap}"
            )

    @staticmethod
    def _assert_disjoint_train_eval_success(
        train_success_trajs: Sequence[BenchmarkTrajectory],
        eval_trajs: Sequence[BenchmarkTrajectory],
    ) -> None:
        """Train success pool must not overlap eval success ``video_id``s."""
        train_keys = {str(t.video_id) for t in train_success_trajs}
        eval_succ_keys = {
            str(t.video_id) for t in eval_trajs if not bool(t.is_failure)
        }
        overlap = sorted(train_keys & eval_succ_keys)
        if overlap:
            raise RuntimeError(
                "PUBCEBenchmarkDiscriminator disjointness invariant violated:\n  - "
                f"train_success intersect eval_success = {overlap}"
            )

    @staticmethod
    def _success_prefix_frame_end(trajectory: BenchmarkTrajectory) -> int:
        """Exclusive end index for PU positive/calib success frames."""
        prefix_fn = getattr(trajectory, "prefix_frames_before_done", None)
        if prefix_fn is not None:
            return int(prefix_fn())
        return int(trajectory.num_frames)

    def _encode_success_train_trajectories(
        self,
        trajectories: Sequence[BenchmarkTrajectory],
        *,
        desc: str = "encode",
    ) -> List[torch.Tensor]:
        """Encode success trajectories using only pre-done frames."""
        if not trajectories:
            return []

        frame_ends: List[int] = []
        for traj in trajectories:
            t_end = self._success_prefix_frame_end(traj)
            if t_end <= 0:
                raise ValueError(
                    f"Success trajectory has no pre-done frames: {traj.describe()}"
                )
            frame_ends.append(t_end)
        return self._encode_trajectories(
            trajectories, desc=desc, frame_ends=frame_ends
        )

    def _save_checkpoint(self) -> Optional[Path]:
        if self.save_ckpt_dir is None or self._shared_detector is None:
            return None
        out_dir = Path(self.save_ckpt_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "epoch": int(self.epochs),
            "in_dim": int(self._shared_detector.in_dim),
            "hidden": int(self._shared_detector.hidden),
            "num_layers": int(self._shared_detector.num_layers),
            "pu_bce_detector": self._shared_detector.state_dict(),
            **self.feature_metadata(),
            "pi_p": float(self.pi_p),
            "loss_surrogate": str(self.loss_surrogate),
            "nn_correction": bool(self.nn_correction),
            "beta": float(self.beta),
            "unlabeled_fail_video_ids": sorted(
                str(t.video_id) for t in self.unlabeled_fail_trajectories
            ),
        }
        fp = out_dir / "pu_bce_head.pth"
        torch.save(payload, fp)
        return fp

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def fit_on_benchmark(
        self,
        eval_trajectories: List[BenchmarkTrajectory],
        *,
        train_success_trajectories: Optional[List[BenchmarkTrajectory]] = None,
    ) -> None:
        """Train one shared nnPU head; calibrate per-task success_percentile thresholds.

        Does **not** call ``bench.evaluate(...)`` and does **not** compute any
        eval metric over ``eval_trajectories``. Positives + calibration success
        come from ``train_success_trajectories`` when provided; otherwise they
        are taken from success demos inside ``eval_trajectories`` (legacy).
        The unlabeled pool comes from ``self.unlabeled_fail_trajectories``.
        """
        self._assert_disjoint_unlabeled_eval_fail(
            eval_trajs=eval_trajectories,
            unlabeled_fail_trajs=self.unlabeled_fail_trajectories,
        )

        if train_success_trajectories is not None:
            self._assert_disjoint_train_eval_success(
                train_success_trajs=train_success_trajectories,
                eval_trajs=eval_trajectories,
            )
            task_to_success: Dict[str, List[BenchmarkTrajectory]] = {}
            for traj in train_success_trajectories:
                if bool(traj.is_failure):
                    continue
                task_to_success.setdefault(str(traj.task_name), []).append(traj)
        else:
            task_to_success = {}
            for traj in eval_trajectories:
                if bool(traj.is_failure):
                    continue
                task_to_success.setdefault(str(traj.task_name), []).append(traj)

        eval_tasks = sorted({
            str(t.task_name)
            for t in eval_trajectories
            if bool(t.is_failure)
        })
        if not eval_tasks:
            eval_tasks = sorted(task_to_success.keys())

        missing_tasks = sorted(set(eval_tasks) - set(task_to_success))
        if missing_tasks:
            raise RuntimeError(
                "PU-BCE requires train success trajectories for every eval task. "
                f"Missing: {missing_tasks}. Pass --success-train-split or check data."
            )

        if not task_to_success:
            raise RuntimeError(
                "PU-BCE requires success trajectories per task for train + calibration."
            )

        rng = np.random.default_rng(int(self.seed))
        train_success_per_task: Dict[str, List[BenchmarkTrajectory]] = {}
        calib_success_per_task: Dict[str, List[BenchmarkTrajectory]] = {}
        for task, succ_list in task_to_success.items():
            if len(succ_list) < 2:
                raise RuntimeError(
                    f"Task {task!r} has only {len(succ_list)} success trajectory; "
                    "need at least 2 for disjoint train + calib split."
                )
            perm = rng.permutation(len(succ_list))
            n_calib = int(round(self.calib_fraction * len(succ_list)))
            n_calib = max(1, min(len(succ_list) - 1, n_calib))
            calib_idx = set(perm[:n_calib].tolist())
            train_success_per_task[task] = [t for i, t in enumerate(succ_list) if i not in calib_idx]
            calib_success_per_task[task] = [t for i, t in enumerate(succ_list) if i in calib_idx]

        # ------ encode positives via the parent's _encode cache ------
        if self.verbose_fit:
            n_train = sum(len(v) for v in train_success_per_task.values())
            n_calib = sum(len(v) for v in calib_success_per_task.values())
            n_unlabeled = len(self.unlabeled_fail_trajectories)
            print(
                f"[pu_bce][encode] fitting pool: "
                f"success_train={n_train} success_calib={n_calib} "
                f"unlabeled_fail={n_unlabeled}",
                flush=True,
            )

        pooled_positive: List[torch.Tensor] = []
        pooled_positive_stats: Dict[str, int] = {}
        for task, trajs in train_success_per_task.items():
            task_total = 0
            for f in self._encode_success_train_trajectories(
                trajs,
                desc=f"[pu_bce][encode] success train pre-done ({task})",
            ):
                pooled_positive.append(f)
                task_total += int(f.shape[0])
            pooled_positive_stats[task] = task_total

        success_calib_per_task: Dict[str, List[torch.Tensor]] = {}
        success_calib_stats: Dict[str, int] = {}
        for task, trajs in calib_success_per_task.items():
            feats_list = self._encode_success_train_trajectories(
                trajs,
                desc=f"[pu_bce][encode] success calib pre-done ({task})",
            )
            tot = sum(int(f.shape[0]) for f in feats_list)
            success_calib_per_task[task] = feats_list
            success_calib_stats[task] = tot

        # ------ pool unlabeled failure frames: WHOLE trajectory (no GT split) ------
        pooled_unlabeled: List[torch.Tensor] = []
        unlabeled_stats_per_task: Dict[str, int] = {}
        unlabeled_by_task: Dict[str, List[BenchmarkTrajectory]] = {}
        for t in self.unlabeled_fail_trajectories:
            unlabeled_by_task.setdefault(str(t.task_name), []).append(t)
        for task, trajs in sorted(unlabeled_by_task.items()):
            for f in self._encode_trajectories(
                trajs,
                desc=f"[pu_bce][encode] unlabeled fail ({task})",
            ):
                pooled_unlabeled.append(f)
                unlabeled_stats_per_task[task] = (
                    unlabeled_stats_per_task.get(task, 0) + int(f.shape[0])
                )

        if not pooled_unlabeled:
            raise RuntimeError(
                "PU-BCE requires at least one unlabeled failure trajectory. "
                "Check the failure pool / --unlabeled-per-task."
            )

        # ------ infer in_dim, build shared detector, fit ------
        feat_dim = int(pooled_positive[0].shape[1])
        for tens in pooled_positive + pooled_unlabeled:
            if int(tens.shape[1]) != feat_dim:
                raise RuntimeError(
                    f"Feature dim mismatch across trajectories: "
                    f"expected {feat_dim}, got {int(tens.shape[1])}"
                )

        if self.verbose_fit:
            n_p = sum(int(t.shape[0]) for t in pooled_positive)
            n_u = sum(int(t.shape[0]) for t in pooled_unlabeled)
            n_c = sum(int(t.shape[0]) for ts in success_calib_per_task.values() for t in ts)
            print(
                f"[pu_bce][fit] tasks={sorted(task_to_success)} "
                f"feat_dim={feat_dim} "
                f"Np(success train)={n_p} Nu(unlabeled fail whole)={n_u} N_calib={n_c} "
                f"pi_p={self.pi_p} surrogate={self.loss_surrogate} "
                f"nn_correction={self.nn_correction} "
                f"latent={self.encoder.latent_name}",
                flush=True,
            )

        self._shared_detector = PUBCEDiscriminator(
            in_dim=feat_dim,
            hidden=self.head_hidden,
            num_layers=self.head_layers,
            device=self.device,
        )

        # NOTE: this is the only training call. It runs purely on the nnPU risk
        # over a fixed epoch budget. No eval, no AUROC, no validation pass.
        thresholds = self._shared_detector.fit(
            positive_features=pooled_positive,
            unlabeled_features=pooled_unlabeled,
            success_calib_per_task=success_calib_per_task,
            pi_p=self.pi_p,
            epochs=self.epochs,
            lr=self.lr,
            weight_decay=self.weight_decay,
            batch_size=self.batch_size,
            delta=self.delta,
            seed=self.seed,
            loss_surrogate=self.loss_surrogate,
            nn_correction=self.nn_correction,
            beta=self.beta,
            pin_memory=self.pin_memory,
            verbose=self.verbose_fit,
        )

        # Alias the same detector under every task name.
        self._detectors_per_task = {task: self._shared_detector for task in task_to_success}

        # Stats for calibration_summary().
        self._global_stats = {
            "feat_dim": int(feat_dim),
            "epochs": int(self.epochs),
            "lr": float(self.lr),
            "weight_decay": float(self.weight_decay),
            "batch_size": int(self.batch_size),
            "head_hidden": int(self.head_hidden),
            "head_layers": int(self.head_layers),
            "delta": float(self.delta),
            "calib_fraction": float(self.calib_fraction),
            "calib_mode": "success_percentile",
            "pi_p": float(self.pi_p),
            "loss_surrogate": str(self.loss_surrogate),
            "nn_correction": bool(self.nn_correction),
            "beta": float(self.beta),
            "seed": int(self.seed),
            "latent": self.encoder.latent_name,
            "policy_ckpt_hash": self.policy_ckpt_hash,
            "num_unlabeled_fail_trajectories": int(len(self.unlabeled_fail_trajectories)),
            "train_history": list(self._shared_detector._train_history),
        }
        for task, tau in thresholds.items():
            cs = self._shared_detector.calib_stats.get(task)
            self._calibration_stats[task] = {
                "threshold": float(tau),
                "calib_mode": "success_percentile",
                "num_success_trajectories": int(len(task_to_success[task])),
                "num_train_success_trajectories": int(len(train_success_per_task[task])),
                "num_calib_success_trajectories": int(len(calib_success_per_task[task])),
                "num_train_success_frames": int(pooled_positive_stats.get(task, 0)),
                "num_calib_success_frames": int(success_calib_stats.get(task, 0)),
                "num_unlabeled_fail_frames": int(unlabeled_stats_per_task.get(task, 0)),
                "calib_score_min": None if cs is None else float(cs.calib_score_min),
                "calib_score_max": None if cs is None else float(cs.calib_score_max),
                "calib_score_mean": None if cs is None else float(cs.calib_score_mean),
                "calib_score_std": None if cs is None else float(cs.calib_score_std),
                "calib_num_frames": None if cs is None else int(cs.num_calib_frames),
            }

        # Final one-shot checkpoint (no mid-training snapshots).
        ckpt_path = self._save_checkpoint()
        if ckpt_path is not None and self.verbose_fit:
            print(f"[pu_bce][ckpt] wrote {ckpt_path}", flush=True)

    def score_trajectory(self, trajectory: BenchmarkTrajectory) -> DiscriminatorOutput:
        task = str(trajectory.task_name)
        det = self._detectors_per_task.get(task, None)
        if det is None:
            raise KeyError(
                f"Task {task!r} is not calibrated. Available: "
                f"{sorted(self._detectors_per_task)}"
            )

        T = int(trajectory.num_frames)
        tau = float(det.thresholds.get(task, float("nan")))

        if bool(trajectory.is_failure):
            feat = self._encode(trajectory)
            result = det.score(feat, task=task)
            step_scores = _pad_to_length(result.step_scores, target_len=T, dtype=np.float32)
            thresholds = _pad_to_length(result.thresholds, target_len=T, dtype=np.float32)
            preds = _pad_to_length(result.preds, target_len=T, dtype=np.int64).astype(np.int64)
        else:
            t_end = self._success_prefix_frame_end(trajectory)
            if t_end <= 0:
                raise ValueError(
                    f"Success trajectory has no pre-done frames: {trajectory.describe()}"
                )
            feat = self._encode(trajectory, frame_end=t_end)
            result = det.score(feat, task=task)
            # Post-done padding keeps benchmark length alignment without scoring
            # idle frames. The fill MUST be a *low* failure score so the post-done
            # region never wins the max/topk trajectory aggregation: failure_score
            # = -g lives in a negative regime for success frames, so 0.0 is a high
            # spike that inverts trajectory-level AUROC. Use this trajectory's own
            # minimum real score (<= every scored frame, scale-free, no -inf).
            real_scores = np.asarray(result.step_scores, dtype=np.float32).reshape(-1)
            post_done_fill = float(real_scores.min()) if real_scores.size > 0 else 0.0
            step_scores = _pad_to_length_with_fill(
                result.step_scores, target_len=T, fill_value=post_done_fill, dtype=np.float32
            )
            thresholds = _pad_to_length_with_fill(
                result.thresholds, target_len=T, fill_value=tau, dtype=np.float32
            )
            preds = _pad_to_length_with_fill(
                result.preds, target_len=T, fill_value=0, dtype=np.int64
            ).astype(np.int64)

        positive = np.where(preds == 1)[0]
        first_failure_frame = int(positive[0]) if positive.size > 0 else None

        aux: Dict[str, Any] = {
            "task": task,
            "threshold": tau,
            "delta": float(self.delta),
            "thresholds": thresholds,
            "step_scores_raw": step_scores,
            "feature_len": int(feat.shape[0]),
            "latent": self.encoder.latent_name,
            "camera_names": list(self.camera_names),
        }
        if not bool(trajectory.is_failure):
            aux["success_prefix_frames"] = int(feat.shape[0])
        return DiscriminatorOutput(
            step_scores=step_scores,
            predictions=preds,
            first_failure_frame=first_failure_frame,
            aux=aux,
        )

    def calibration_summary(self) -> dict:
        return {
            "per_task": dict(self._calibration_stats),
            "global": dict(self._global_stats),
            **self.feature_metadata(),
            "delta": float(self.delta),
            "calib_fraction": float(self.calib_fraction),
            "calib_mode": "success_percentile",
            "pi_p": float(self.pi_p),
            "loss_surrogate": str(self.loss_surrogate),
            "encode_batch_size": int(self.encode_batch_size),
            "save_ckpt_dir": self.save_ckpt_dir,
        }
