"""Benchmark adapter for the BCE-WAM failure discriminator (GT failure split).

Sits on top of :class:`LPBV2BenchmarkDiscriminator` so all the WAM encoding +
trajectory feature cache is reused unchanged. Replaces the per-task KNN with a
single shared :class:`BCEDiscriminator` trained on pooled (D_e, D_o) frames,
plus per-task thresholds.

Training labels use **GT failure timing** only: each failure bank trajectory is
sliced at ``first_gt_failure_frame()`` — prefix ``[0, t*)`` joins ``D_e``,
suffix ``[t*, T)`` is ``D_o``.

Hard invariants:
  * No ``video_id`` in ``fail_bank_trajectories`` / ``fail_calib_trajectories``
    may appear in the eval set passed to ``fit_on_benchmark``. Enforced by a
    duplicate of :meth:`TwoBankBenchmarkDiscriminator._assert_disjoint`.
  * ``fit_on_benchmark`` does **not** call ``bench.evaluate(...)`` or compute
    any AUROC / metric over eval trajectories. Evaluation is the caller's job
    (see ``run_bce_benchmark.py``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from benchmark.core import BenchmarkTrajectory, DiscriminatorOutput

from .bce_discriminator import BCEDiscriminator
from .benchmark import LPBV2BenchmarkDiscriminator, _pad_to_length


class BCEBenchmarkDiscriminator(LPBV2BenchmarkDiscriminator):
    """BCE failure detector (GT-labeled prefix/suffix split) over the benchmark API."""

    name = "lpb_v2_bce"

    def __init__(
        self,
        *,
        model_ckpt: str,
        fail_bank_trajectories: Sequence[BenchmarkTrajectory],
        fail_calib_trajectories: Optional[Sequence[BenchmarkTrajectory]] = None,
        max_expert_other_ratio: Optional[float] = 1.0,
        head_hidden: int = 256,
        head_layers: int = 2,
        epochs: int = 20,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        batch_size: int = 512,
        save_ckpt_dir: Optional[str] = None,
        # forwarded to the single-bank parent for encoding / cache parity
        device: str = "cuda",
        encode_batch_size: int = 32,
        proprio_indices: Optional[Sequence[int]] = None,
        camera_to_view: Optional[Dict[str, str]] = None,
        visual_weight: float = 1.0,
        proprio_weight: float = 2.0,
        action_weight: float = 1.0,
        delta: float = 10.0,
        knn_chunk_size: int = 2048,
        feature_source: str = "transformer",
        transformer_layer: int = 1,
        calib_fraction: float = 0.2,
        seed: int = 0,
        verbose_fit: bool = True,
    ) -> None:
        super().__init__(
            model_ckpt=model_ckpt,
            device=device,
            encode_batch_size=encode_batch_size,
            proprio_indices=proprio_indices,
            camera_to_view=camera_to_view,
            visual_weight=visual_weight,
            proprio_weight=proprio_weight,
            action_weight=action_weight,
            delta=delta,
            knn_chunk_size=knn_chunk_size,
            feature_source=feature_source,
            transformer_layer=transformer_layer,
            calib_fraction=calib_fraction,
            seed=seed,
            verbose_fit=verbose_fit,
        )
        self.fail_bank_trajectories: List[BenchmarkTrajectory] = list(fail_bank_trajectories)
        self.fail_calib_trajectories: List[BenchmarkTrajectory] = (
            list(fail_calib_trajectories) if fail_calib_trajectories else []
        )

        self.max_expert_other_ratio: Optional[float] = (
            None if max_expert_other_ratio is None else float(max_expert_other_ratio)
        )

        self.head_hidden = int(head_hidden)
        self.head_layers = int(head_layers)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.save_ckpt_dir = None if save_ckpt_dir is None else str(save_ckpt_dir)

        # Single shared detector across tasks (constructed in fit_on_benchmark
        # once the encoder's feature dim is known).
        self._shared_detector: Optional[BCEDiscriminator] = None
        # Override parent's per-task storage to alias the shared detector for
        # every task. Keeps score_trajectory's lookup pattern unchanged.
        self._detectors_per_task: Dict[str, BCEDiscriminator] = {}
        self._calibration_stats: Dict[str, dict] = {}
        self._global_stats: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _assert_disjoint(
        eval_trajs: Sequence[BenchmarkTrajectory],
        fail_bank_trajs: Sequence[BenchmarkTrajectory],
        fail_calib_trajs: Sequence[BenchmarkTrajectory],
    ) -> None:
        """video_id disjointness invariant; mirrors benchmark_two_bank._assert_disjoint."""
        eval_keys = {str(t.video_id) for t in eval_trajs}
        bank_keys = {str(t.video_id) for t in fail_bank_trajs}
        calib_keys = {str(t.video_id) for t in fail_calib_trajs}
        overlap_bank = sorted(eval_keys & bank_keys)
        overlap_calib = sorted(eval_keys & calib_keys)
        bank_calib_overlap = sorted(bank_keys & calib_keys)
        problems = []
        if overlap_bank:
            problems.append(f"eval ∩ fail_bank = {overlap_bank}")
        if overlap_calib:
            problems.append(f"eval ∩ fail_calib = {overlap_calib}")
        if bank_calib_overlap:
            problems.append(f"fail_bank ∩ fail_calib = {bank_calib_overlap}")
        if problems:
            raise RuntimeError(
                "BCEBenchmarkDiscriminator disjointness invariant violated:\n  - "
                + "\n  - ".join(problems)
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
            "bce_detector": self._shared_detector.state_dict(),
            "feature_source": str(self.feature_source),
            "transformer_layer": int(self.transformer_layer),
            "model_ckpt": str(self.model_ckpt),
            "max_expert_other_ratio": self.max_expert_other_ratio,
            "fail_bank_video_ids": sorted(
                str(t.video_id) for t in self.fail_bank_trajectories
            ),
            "fail_calib_video_ids": sorted(
                str(t.video_id) for t in self.fail_calib_trajectories
            ),
        }
        fp = out_dir / "bce_head.pth"
        torch.save(payload, fp)
        return fp

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def fit_on_benchmark(self, trajectories: List[BenchmarkTrajectory]) -> None:
        """Train one shared BCE head; calibrate per-task thresholds.

        Does **not** call ``bench.evaluate(...)`` and does **not** compute any
        eval metric over ``trajectories``. ``trajectories`` is used only to
        gather the success-rollout pool (training + per-task calibration).
        """
        self._assert_disjoint(
            eval_trajs=trajectories,
            fail_bank_trajs=self.fail_bank_trajectories,
            fail_calib_trajs=self.fail_calib_trajectories,
        )

        # ------ split benchmark success demos into train / calib per-task ------
        task_to_success: Dict[str, List[BenchmarkTrajectory]] = {}
        for traj in trajectories:
            if bool(traj.is_failure):
                continue
            task_to_success.setdefault(str(traj.task_name), []).append(traj)

        if not task_to_success:
            raise RuntimeError(
                "BCE benchmark requires success trajectories per task in the eval set."
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

        # ------ encode everything via the parent's _encode cache ------
        pooled_expert: List[torch.Tensor] = []
        pooled_expert_stats: Dict[str, int] = {}
        for task, trajs in train_success_per_task.items():
            task_total = 0
            for t in trajs:
                f = self._encode(t)
                pooled_expert.append(f)
                task_total += int(f.shape[0])
            pooled_expert_stats[task] = task_total

        expert_calib_per_task: Dict[str, List[torch.Tensor]] = {}
        expert_calib_stats: Dict[str, int] = {}
        for task, trajs in calib_success_per_task.items():
            feats_list: List[torch.Tensor] = []
            tot = 0
            for t in trajs:
                f = self._encode(t)
                feats_list.append(f)
                tot += int(f.shape[0])
            expert_calib_per_task[task] = feats_list
            expert_calib_stats[task] = tot

        # Pool failure bank: prefix [0, t*) -> D_e, suffix [t*, T) -> D_o (GT).
        pooled_other: List[torch.Tensor] = []
        fail_other_stats_per_task: Dict[str, int] = {}
        fail_prefix_stats_per_task: Dict[str, int] = {}
        skipped_no_gt: List[str] = []
        for t in self.fail_bank_trajectories:
            f = self._encode(t)
            task_name = str(t.task_name)
            t_star = t.first_gt_failure_frame()
            T = int(f.shape[0])
            if t_star is None:
                skipped_no_gt.append(str(t.video_id))
                continue
            t_star_int = int(t_star)
            if t_star_int <= 0 or t_star_int >= T:
                skipped_no_gt.append(str(t.video_id))
                continue
            prefix = f[:t_star_int]
            suffix = f[t_star_int:]
            pooled_expert.append(prefix)
            pooled_other.append(suffix)
            fail_prefix_stats_per_task[task_name] = (
                fail_prefix_stats_per_task.get(task_name, 0) + int(prefix.shape[0])
            )
            fail_other_stats_per_task[task_name] = (
                fail_other_stats_per_task.get(task_name, 0) + int(suffix.shape[0])
            )

        if skipped_no_gt and self.verbose_fit:
            print(
                f"[bce][gt] WARNING: skipped {len(skipped_no_gt)} failure "
                f"trajectories without usable first_gt_failure_frame: "
                f"{skipped_no_gt[:8]}{'...' if len(skipped_no_gt) > 8 else ''}",
                flush=True,
            )
        if not pooled_other:
            raise RuntimeError(
                "BCE requires at least one failure suffix in fail_bank_trajectories. "
                "Check GT annotations."
            )

        # ------ infer in_dim, build shared detector, fit ------
        feat_dim = int(pooled_expert[0].shape[1])
        for tens in pooled_expert + pooled_other:
            if int(tens.shape[1]) != feat_dim:
                raise RuntimeError(
                    f"Feature dim mismatch across trajectories: "
                    f"expected {feat_dim}, got {int(tens.shape[1])}"
                )

        if self.verbose_fit:
            ne = sum(int(t.shape[0]) for t in pooled_expert)
            no = sum(int(t.shape[0]) for t in pooled_other)
            nc = sum(int(t.shape[0]) for ts in expert_calib_per_task.values() for t in ts)
            prefix_total = sum(fail_prefix_stats_per_task.values())
            print(
                f"[bce][fit] tasks={sorted(task_to_success)} "
                f"feat_dim={feat_dim} "
                f"Ne(train)={ne} (incl. fail_prefix={prefix_total}) "
                f"No(train)={no} N_calib={nc} "
                f"max_expert_other_ratio={self.max_expert_other_ratio} "
                f"feature_source={self.feature_source} layer={self.transformer_layer}",
                flush=True,
            )

        self._shared_detector = BCEDiscriminator(
            in_dim=feat_dim,
            hidden=self.head_hidden,
            num_layers=self.head_layers,
            device=self.device,
        )

        # NOTE: this is the only training call. It runs purely on BCE over a fixed
        # epoch budget. No eval, no AUROC, no validation pass.
        thresholds = self._shared_detector.fit(
            expert_features=pooled_expert,
            other_features=pooled_other,
            expert_calib_per_task=expert_calib_per_task,
            epochs=self.epochs,
            lr=self.lr,
            weight_decay=self.weight_decay,
            batch_size=self.batch_size,
            delta=self.delta,
            seed=self.seed,
            max_expert_other_ratio=self.max_expert_other_ratio,
            verbose=self.verbose_fit,
        )

        # Alias the same detector under every task name (so the parent's
        # `_detectors_per_task[task]` lookup pattern keeps working).
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
            "seed": int(self.seed),
            "feature_source": str(self.feature_source),
            "transformer_layer": int(self.transformer_layer),
            "max_expert_other_ratio": self.max_expert_other_ratio,
            "num_fail_bank_trajectories": int(len(self.fail_bank_trajectories)),
            "num_fail_calib_trajectories": int(len(self.fail_calib_trajectories)),
            "num_fail_bank_skipped_no_gt": int(len(skipped_no_gt)),
            "train_history": list(self._shared_detector._train_history),
        }
        for task, tau in thresholds.items():
            cs = self._shared_detector.calib_stats.get(task)
            self._calibration_stats[task] = {
                "threshold": float(tau),
                "num_success_trajectories": int(len(task_to_success[task])),
                "num_train_success_trajectories": int(len(train_success_per_task[task])),
                "num_calib_success_trajectories": int(len(calib_success_per_task[task])),
                "num_train_success_frames": int(pooled_expert_stats.get(task, 0)),
                "num_calib_success_frames": int(expert_calib_stats.get(task, 0)),
                "num_fail_other_frames": int(fail_other_stats_per_task.get(task, 0)),
                "num_fail_prefix_frames": int(fail_prefix_stats_per_task.get(task, 0)),
                "calib_score_min": None if cs is None else float(cs.calib_score_min),
                "calib_score_max": None if cs is None else float(cs.calib_score_max),
                "calib_score_mean": None if cs is None else float(cs.calib_score_mean),
                "calib_score_std": None if cs is None else float(cs.calib_score_std),
                "calib_num_frames": None if cs is None else int(cs.num_calib_frames),
            }

        # Final one-shot checkpoint (no mid-training snapshots).
        ckpt_path = self._save_checkpoint()
        if ckpt_path is not None and self.verbose_fit:
            print(f"[bce][ckpt] wrote {ckpt_path}", flush=True)

    def score_trajectory(self, trajectory: BenchmarkTrajectory) -> DiscriminatorOutput:
        task = str(trajectory.task_name)
        det = self._detectors_per_task.get(task, None)
        if det is None:
            raise KeyError(
                f"Task {task!r} is not calibrated. Available: "
                f"{sorted(self._detectors_per_task)}"
            )

        T = int(trajectory.num_frames)
        feat = self._encode(trajectory)
        result = det.score(feat, task=task)

        step_scores = _pad_to_length(result.step_scores, target_len=T, dtype=np.float32)
        thresholds = _pad_to_length(result.thresholds, target_len=T, dtype=np.float32)
        preds = _pad_to_length(result.preds, target_len=T, dtype=np.int64).astype(np.int64)

        positive = np.where(preds == 1)[0]
        first_failure_frame = int(positive[0]) if positive.size > 0 else None

        tau = float(det.thresholds.get(task, float("nan")))
        aux: Dict[str, Any] = {
            "task": task,
            "threshold": tau,
            "delta": float(self.delta),
            "thresholds": thresholds,
            "step_scores_raw": step_scores,
            "feature_len": int(feat.shape[0]),
            "feature_source": self.feature_source,
            "transformer_layer": int(self.transformer_layer),
            "view_names": list(self.encoder.view_names),
        }
        return DiscriminatorOutput(
            step_scores=step_scores,
            predictions=preds,
            first_failure_frame=first_failure_frame,
            aux=aux,
        )

    def calibration_summary(self) -> dict:
        base = {
            "per_task": dict(self._calibration_stats),
            "global": dict(self._global_stats),
            "model_ckpt": self.model_ckpt,
            "view_names": list(self.encoder.view_names),
            "camera_to_view": dict(self.camera_to_view),
            "delta": float(self.delta),
            "calib_fraction": float(self.calib_fraction),
            "encode_batch_size": int(self.encode_batch_size),
            "feature_source": self.feature_source,
            "transformer_layer": int(self.transformer_layer),
            "save_ckpt_dir": self.save_ckpt_dir,
        }
        return base
