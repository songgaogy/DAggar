"""Benchmark adapter for the two-bank KNN failure detector.

Drop-in extension of :class:`LPBV2BenchmarkDiscriminator` that adds a
**failure bank** alongside the existing success bank. Per-task workflow:

1. ``fit_on_benchmark(trajectories)`` -- as in the single-bank adapter, split
   per-task success demos into bank + disjoint calibration. Additionally pull
   per-task failure trajectories from the constructor-provided list, slice
   each to the last ``fail_bank_last_k`` frames at-or-after
   ``first_gt_failure_frame``, encode them, and pass everything into
   :class:`TwoBankKNN`.
2. ``score_trajectory(traj)`` -- encode frame-by-frame, score via the two-bank
   formula, threshold.

Hard invariant (defence-in-depth): no ``video_id`` appearing in
``fail_bank_trajectories`` or ``fail_calib_trajectories`` may also appear in
the eval set passed to ``fit_on_benchmark``. Discovery is the caller's job;
this class asserts the invariant on every construction-fit pair.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from benchmark.core import BenchmarkTrajectory, DiscriminatorOutput

from .benchmark import LPBV2BenchmarkDiscriminator, _pad_to_length
from .two_bank_knn import TwoBankKNN


def _fail_frame_range(num_frames: int, first_gt: Optional[int], last_k: int) -> Optional[range]:
    """Frame indices to keep for a failure-bank trajectory.

    If ``first_gt`` is None, the trajectory has no annotated failure start --
    skip it (returns ``None``); the caller emits a warning. Otherwise use
    ``[first_gt, min(num_frames, first_gt + last_k))``.
    """
    if first_gt is None:
        return None
    T = int(num_frames)
    g = int(first_gt)
    if g < 0 or g >= T:
        return None
    end = min(T, g + int(last_k))
    if end <= g:
        return None
    return range(g, end)


class TwoBankBenchmarkDiscriminator(LPBV2BenchmarkDiscriminator):
    """Two-bank (success + failure) KNN OOD detector.

    Composes :class:`LPBV2BenchmarkDiscriminator` for all encoding/caching
    plumbing, but swaps the per-task detector for a :class:`TwoBankKNN`.
    """

    name = "lpb_v2_two_bank_knn"

    def __init__(
        self,
        *,
        model_ckpt: str,
        fail_bank_trajectories: Sequence[BenchmarkTrajectory],
        fail_calib_trajectories: Optional[Sequence[BenchmarkTrajectory]] = None,
        fail_bank_last_k: int = 60,
        alpha: float = 1.0,
        score_mode: str = "difference",
        calib_mode: str = "success_percentile",
        # forwarded to the single-bank parent for parity with LPBV2BenchmarkDiscriminator
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
        self.fail_bank_last_k = int(fail_bank_last_k)
        self.alpha = float(alpha)
        self.score_mode = str(score_mode)
        self.calib_mode = str(calib_mode)

        # Override per-task storage to hold the two-bank detector.
        self._detectors_per_task: Dict[str, TwoBankKNN] = {}
        self._calibration_stats: Dict[str, dict] = {}
        self._fail_bank_index_ranges: Dict[str, Dict[str, List[int]]] = {}

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _encode_fail_subset(
        self,
        trajectory: BenchmarkTrajectory,
    ) -> Optional[torch.Tensor]:
        """Encode a failure trajectory and slice frames to the bank window."""
        idx_range = _fail_frame_range(
            num_frames=int(trajectory.num_frames),
            first_gt=trajectory.first_gt_failure_frame(),
            last_k=self.fail_bank_last_k,
        )
        if idx_range is None:
            print(
                f"[two_bank_knn] WARNING: skipping fail-bank trajectory with no "
                f"first_gt_failure_frame: video_id={trajectory.video_id}"
            )
            return None
        feats = self._encode(trajectory)
        T = int(feats.shape[0])
        # Re-clip in case feature-length is shorter than nominal num_frames.
        start = min(idx_range.start, T)
        end = min(idx_range.stop, T)
        if end <= start:
            print(
                f"[two_bank_knn] WARNING: empty fail-bank slice after re-clip "
                f"(video_id={trajectory.video_id}, T={T}, range={idx_range})"
            )
            return None
        sub = feats[start:end]
        # Record the actual indices used for the manifest.
        per_task = self._fail_bank_index_ranges.setdefault(str(trajectory.task_name), {})
        per_task[str(trajectory.video_id)] = [int(start), int(end)]
        return sub

    @staticmethod
    def _assert_disjoint(
        eval_trajs: Sequence[BenchmarkTrajectory],
        fail_bank_trajs: Sequence[BenchmarkTrajectory],
        fail_calib_trajs: Sequence[BenchmarkTrajectory],
    ) -> None:
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
                "TwoBankBenchmarkDiscriminator disjointness invariant violated:\n  - "
                + "\n  - ".join(problems)
            )

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def fit_on_benchmark(self, trajectories: List[BenchmarkTrajectory]) -> None:
        self._assert_disjoint(
            eval_trajs=trajectories,
            fail_bank_trajs=self.fail_bank_trajectories,
            fail_calib_trajs=self.fail_calib_trajectories,
        )

        task_to_success: Dict[str, List[BenchmarkTrajectory]] = {}
        for traj in trajectories:
            if bool(traj.is_failure):
                continue
            task_to_success.setdefault(str(traj.task_name), []).append(traj)

        if not task_to_success:
            raise RuntimeError(
                "two-bank KNN calibration requires success trajectories per task."
            )

        task_to_fail_bank: Dict[str, List[BenchmarkTrajectory]] = {}
        for t in self.fail_bank_trajectories:
            task_to_fail_bank.setdefault(str(t.task_name), []).append(t)

        task_to_fail_calib: Dict[str, List[BenchmarkTrajectory]] = {}
        for t in self.fail_calib_trajectories:
            task_to_fail_calib.setdefault(str(t.task_name), []).append(t)

        for task, succ_list in task_to_success.items():
            if len(succ_list) < 2:
                raise RuntimeError(
                    f"Task {task!r} has only {len(succ_list)} success trajectory; "
                    "need at least 2 for disjoint bank + calibration split."
                )
            fail_bank_list = task_to_fail_bank.get(task, [])
            if self.score_mode != "dsucc_only" and not fail_bank_list:
                raise RuntimeError(
                    f"Task {task!r} has no failure-bank trajectories but score_mode={self.score_mode!r} "
                    "needs them. Pass fail_bank_trajectories for every eval task."
                )

            rng = np.random.default_rng(int(self.seed))
            perm = rng.permutation(len(succ_list))
            n_calib = int(round(self.calib_fraction * len(succ_list)))
            n_calib = max(1, min(len(succ_list) - 1, n_calib))
            calib_idx = set(perm[:n_calib].tolist())
            bank_trajs = [t for i, t in enumerate(succ_list) if i not in calib_idx]
            calib_trajs = [t for i, t in enumerate(succ_list) if i in calib_idx]

            bank_feats: List[torch.Tensor] = []
            calib_feats: List[torch.Tensor] = []
            bank_total = 0
            calib_total = 0
            for t in bank_trajs:
                f = self._encode(t)
                bank_feats.append(f)
                bank_total += int(f.shape[0])
            for t in calib_trajs:
                f = self._encode(t)
                calib_feats.append(f)
                calib_total += int(f.shape[0])

            fail_bank_feats: List[torch.Tensor] = []
            fail_bank_total = 0
            for t in fail_bank_list:
                sub = self._encode_fail_subset(t)
                if sub is None:
                    continue
                fail_bank_feats.append(sub)
                fail_bank_total += int(sub.shape[0])
            if self.score_mode != "dsucc_only" and not fail_bank_feats:
                raise RuntimeError(
                    f"Task {task!r}: all failure-bank trajectories were skipped (no first_gt_failure_frame)."
                )

            fail_calib_feats: List[torch.Tensor] = []
            fail_calib_total = 0
            for t in task_to_fail_calib.get(task, []):
                sub = self._encode_fail_subset(t)
                if sub is None:
                    continue
                fail_calib_feats.append(sub)
                fail_calib_total += int(sub.shape[0])

            feat_dim = int(bank_feats[0].shape[1])
            if self.feature_source == "encoder":
                visual_dim = int(self.encoder.visual_emb_dim_total)
                proprio_dim = int(self.encoder.proprio_emb_dim)
                action_dim = int(self.encoder.action_emb_dim)
                visual_weight = float(self.visual_weight)
                proprio_weight = float(self.proprio_weight)
                action_weight = float(self.action_weight)
            else:
                visual_dim = int(feat_dim)
                proprio_dim = 0
                action_dim = 0
                visual_weight = 1.0
                proprio_weight = 1.0
                action_weight = 1.0

            if self.verbose_fit:
                print(
                    f"[two_bank_knn][fit] task={task} "
                    f"bank_trajs={len(bank_trajs)} ({bank_total} steps)  "
                    f"calib_trajs={len(calib_trajs)} ({calib_total} steps)  "
                    f"fail_bank_trajs={len(fail_bank_feats)} ({fail_bank_total} steps)  "
                    f"fail_calib_trajs={len(fail_calib_feats)} ({fail_calib_total} steps)  "
                    f"feature_source={self.feature_source} layer={self.transformer_layer}  "
                    f"feat_dim={feat_dim} score_mode={self.score_mode} calib_mode={self.calib_mode}"
                )

            det = TwoBankKNN(
                visual_dim=visual_dim,
                proprio_dim=proprio_dim,
                action_dim=action_dim,
                visual_weight=visual_weight,
                proprio_weight=proprio_weight,
                action_weight=action_weight,
                alpha=self.alpha,
                score_mode=self.score_mode,
                delta=self.delta,
                calib_mode=self.calib_mode,
                chunk_size=self.knn_chunk_size,
                device=self.device,
            )
            threshold = det.fit(
                expert_features=bank_feats,
                fail_features=fail_bank_feats,
                success_calib_features=calib_feats,
                fail_calib_features=fail_calib_feats if fail_calib_feats else None,
            )
            self._detectors_per_task[task] = det
            self._calibration_stats[task] = {
                "num_success_trajectories": int(len(succ_list)),
                "num_bank_trajectories": int(len(bank_trajs)),
                "num_calib_trajectories": int(len(calib_trajs)),
                "num_bank_steps": int(bank_total),
                "num_calib_steps": int(calib_total),
                "num_fail_bank_trajectories": int(len(fail_bank_feats)),
                "num_fail_bank_frames": int(fail_bank_total),
                "num_fail_calib_trajectories": int(len(fail_calib_feats)),
                "num_fail_calib_frames": int(fail_calib_total),
                "threshold_init": float(threshold),
                "delta_init": float(self.delta),
                "feat_dim": feat_dim,
                "visual_dim": visual_dim,
                "proprio_dim": proprio_dim,
                "action_dim": action_dim,
                "feature_source": self.feature_source,
                "transformer_layer": int(self.transformer_layer),
                "effective_visual_weight": float(visual_weight),
                "effective_proprio_weight": float(proprio_weight),
                "effective_action_weight": float(action_weight),
                "score_mode": self.score_mode,
                "calib_mode": self.calib_mode,
                "alpha": float(self.alpha),
                "fail_bank_last_k": int(self.fail_bank_last_k),
                "calib_summary": det.calib_summary(),
            }

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
        result = det.score(feat)

        step_scores = _pad_to_length(result.step_scores, target_len=T, dtype=np.float32)
        thresholds = _pad_to_length(result.thresholds, target_len=T, dtype=np.float32)
        preds = _pad_to_length(result.preds, target_len=T, dtype=np.int64).astype(np.int64)

        positive = np.where(preds == 1)[0]
        first_failure_frame = int(positive[0]) if positive.size > 0 else None

        aux: Dict[str, Any] = {
            "task": task,
            "threshold": float(det.threshold) if det.threshold is not None else float("nan"),
            "delta": float(det.delta),
            "thresholds": thresholds,
            "step_scores_raw": step_scores,
            "feature_len": int(feat.shape[0]),
            "feature_source": self.feature_source,
            "transformer_layer": int(self.transformer_layer),
            "score_mode": self.score_mode,
            "calib_mode": self.calib_mode,
            "alpha": float(self.alpha),
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
            "model_ckpt": self.model_ckpt,
            "view_names": list(self.encoder.view_names),
            "camera_to_view": dict(self.camera_to_view),
            "visual_weight": float(self.visual_weight),
            "proprio_weight": float(self.proprio_weight),
            "action_weight": float(self.action_weight),
            "transformer_metric": "uniform_l2" if self.feature_source == "transformer" else "block_weighted_l2",
            "feature_source": self.feature_source,
            "transformer_layer": int(self.transformer_layer),
            "delta": float(self.delta),
            "knn_chunk_size": int(self.knn_chunk_size),
            "calib_fraction": float(self.calib_fraction),
            "encode_batch_size": int(self.encode_batch_size),
            "score_mode": self.score_mode,
            "calib_mode": self.calib_mode,
            "alpha": float(self.alpha),
            "fail_bank_last_k": int(self.fail_bank_last_k),
            "fail_bank_index_ranges": dict(self._fail_bank_index_ranges),
        }
        return base
