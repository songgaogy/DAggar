"""LPB (latent-policy-bank) KNN discriminator plugged into the shared benchmark.

Each task gets its own expert KNN bank built from success-trajectory latent
features (o,a) produced by a pretrained LPB dynamics checkpoint. The lambda
threshold is calibrated per task on a disjoint subset of the success pool.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import torch

from data.utils.benchmark import BenchmarkTrajectory, DiscriminatorOutput

from robosuite.discriminator.float.float_data import PolicyTrajectory

from .knn_discriminator import AdaptiveKNNDiscriminator, LPBFeatureExtractor


def _pad_to_length(values: np.ndarray, target_len: int, dtype=np.float32) -> np.ndarray:
    """Left-align `values` against the original timeline and right-pad (edge)."""
    arr = np.asarray(values, dtype=dtype).reshape(-1)
    n = int(arr.shape[0])
    T = int(target_len)
    if n == T:
        return arr
    if n > T:
        return arr[:T].copy()
    if n == 0:
        return np.zeros((T,), dtype=dtype)
    pad = np.full((T - n,), arr[-1], dtype=dtype)
    return np.concatenate([arr, pad], axis=0)


class LPBBenchmarkDiscriminator:
    """LPB KNN detector exposed through the shared BenchmarkTrajectory API."""

    name = "lpb_knn"

    def __init__(
        self,
        *,
        checkpoint_path: str,
        device: str = "cuda",
        feature_batch_size: int = 256,
        action_horizon: int = -1,
        camera_name: str = "agentview",
        proprio_indices: Optional[Sequence[int]] = None,
        normalize_feature: bool = True,
        use_transition_error: bool = False,
        transition_proprio_error_weight: float = 0.1,
        delta: float = 10.0,
        delta_step: float = 1.0,
        knn_chunk_size: int = 8192,
        lambda_mode: str = "mean",
        lambda_window_size: int = -1,
        transition_aux_weight: float = 0.0,
        calib_fraction: float = 0.2,
        seed: int = 0,
        verbose_fit: bool = True,
    ) -> None:
        if not (0.0 < float(calib_fraction) < 1.0):
            raise ValueError(
                f"calib_fraction must be in (0, 1); train/calib must be disjoint. got {calib_fraction}"
            )
        if float(delta) < 0.0 or float(delta) > 100.0:
            raise ValueError(f"delta must be in [0, 100], got {delta}")

        self.checkpoint_path = str(checkpoint_path)
        self.device = str(device)
        self.camera_name = str(camera_name)
        self.use_transition_error = bool(use_transition_error)
        self.transition_proprio_error_weight = float(transition_proprio_error_weight)
        self.delta = float(delta)
        self.delta_step = float(delta_step)
        self.knn_chunk_size = int(knn_chunk_size)
        self.lambda_mode = str(lambda_mode)
        self.lambda_window_size = int(lambda_window_size)
        self.transition_aux_weight = float(transition_aux_weight)
        self.calib_fraction = float(calib_fraction)
        self.seed = int(seed)
        self.verbose_fit = bool(verbose_fit)

        self.extractor = LPBFeatureExtractor(
            checkpoint_path=self.checkpoint_path,
            device=self.device,
            batch_size=int(feature_batch_size),
            action_horizon=int(action_horizon),
            proprio_indices=None if proprio_indices is None else list(proprio_indices),
            normalize_feature=bool(normalize_feature),
        )

        # Per-task state.
        self._detectors_per_task: dict[str, AdaptiveKNNDiscriminator] = {}
        self._calibration_stats: dict[str, dict] = {}

        # Cache extracted features keyed by (file_path, demo_path).
        # For use_transition_error=False: stores (features, None).
        # For use_transition_error=True:  stores (features, aux_np).
        self._feature_cache: dict[tuple[str, str], tuple[torch.Tensor, Optional[np.ndarray]]] = {}

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _trajectory_key(self, trajectory: BenchmarkTrajectory) -> tuple[str, str]:
        return (str(trajectory.file_path), str(trajectory.demo_path))

    def _to_policy_trajectory(self, trajectory: BenchmarkTrajectory) -> PolicyTrajectory:
        images_by_cam = trajectory.load_images(cameras=[self.camera_name])
        images = np.asarray(images_by_cam[self.camera_name], dtype=np.uint8)
        states = np.asarray(trajectory.load_states(), dtype=np.float32)
        actions = np.asarray(trajectory.load_actions(), dtype=np.float32)

        t_len = min(int(images.shape[0]), int(states.shape[0]), int(actions.shape[0]))
        if t_len <= 0:
            raise ValueError(
                f"Benchmark trajectory has zero valid timesteps: {trajectory.describe()}"
            )

        meta = {
            "file_path": str(trajectory.file_path),
            "demo_key": str(trajectory.source_demo_key or trajectory.demo_path.split("/")[-1]),
            "successful": not bool(trajectory.is_failure),
            "length": int(t_len),
            "camera_name": self.camera_name,
        }
        return PolicyTrajectory(
            states=states[:t_len],
            images=images[:t_len],
            actions=actions[:t_len],
            meta=meta,
        )

    def _encode(self, trajectory: BenchmarkTrajectory) -> tuple[torch.Tensor, Optional[np.ndarray]]:
        key = self._trajectory_key(trajectory)
        cached = self._feature_cache.get(key, None)
        if cached is not None:
            return cached

        pt = self._to_policy_trajectory(trajectory)
        if self.use_transition_error:
            feat, err = self.extractor.encode_trajectory_with_transition_error(
                pt,
                proprio_error_weight=self.transition_proprio_error_weight,
            )
            aux_np = err.detach().cpu().numpy().astype(np.float32)
            out = (feat, aux_np)
        else:
            feat = self.extractor.encode_trajectory(pt)
            out = (feat, None)
        self._feature_cache[key] = out
        return out

    def _use_aux(self) -> bool:
        return self.use_transition_error and self.transition_aux_weight > 0.0

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def fit_on_benchmark(self, trajectories: list[BenchmarkTrajectory]) -> None:
        task_to_success: dict[str, list[BenchmarkTrajectory]] = {}
        for traj in trajectories:
            if bool(traj.is_failure):
                continue
            task_to_success.setdefault(str(traj.task_name), []).append(traj)

        if not task_to_success:
            raise RuntimeError(
                "LPB KNN calibration requires success trajectories per task. "
                "Provide --success-root/<task>/video_manifest.jsonl entries."
            )

        for task, succ_list in task_to_success.items():
            if len(succ_list) < 2:
                raise RuntimeError(
                    f"Task {task!r} has only {len(succ_list)} success trajectory; "
                    "need at least 2 for disjoint bank + calibration split."
                )

            rng = np.random.default_rng(int(self.seed))
            perm = rng.permutation(len(succ_list))
            n_calib = int(round(self.calib_fraction * len(succ_list)))
            n_calib = max(1, min(len(succ_list) - 1, n_calib))
            calib_idx = set(perm[:n_calib].tolist())
            bank_trajs = [t for i, t in enumerate(succ_list) if i not in calib_idx]
            calib_trajs = [t for i, t in enumerate(succ_list) if i in calib_idx]

            bank_features: list[torch.Tensor] = []
            calib_features: list[torch.Tensor] = []
            calib_aux: list[np.ndarray] = []
            bank_total = 0
            calib_total = 0
            for t in bank_trajs:
                f, _ = self._encode(t)
                bank_features.append(f)
                bank_total += int(f.shape[0])
            for t in calib_trajs:
                f, aux = self._encode(t)
                calib_features.append(f)
                if aux is not None:
                    calib_aux.append(aux)
                calib_total += int(f.shape[0])

            if self.verbose_fit:
                print(
                    f"[lpb_knn][fit] task={task} "
                    f"bank_trajs={len(bank_trajs)} ({bank_total} steps)  "
                    f"calib_trajs={len(calib_trajs)} ({calib_total} steps)  "
                    f"feat_dim={int(bank_features[0].shape[1])}  "
                    f"aux={self._use_aux()}"
                )

            detector = AdaptiveKNNDiscriminator(
                delta=self.delta,
                delta_step=self.delta_step,
                knn_chunk_size=self.knn_chunk_size,
                lambda_mode=self.lambda_mode,
                lambda_window_size=self.lambda_window_size,
                aux_weight=self.transition_aux_weight,
                device=self.device,
            )
            threshold = detector.fit(
                expert_sequences=bank_features,
                calibration_sequences=calib_features,
                calibration_aux=calib_aux if self._use_aux() else None,
            )

            self._detectors_per_task[task] = detector
            self._calibration_stats[task] = {
                "num_success_trajectories": int(len(succ_list)),
                "num_bank_trajectories": int(len(bank_trajs)),
                "num_calib_trajectories": int(len(calib_trajs)),
                "num_bank_steps": int(bank_total),
                "num_calib_steps": int(calib_total),
                "threshold_init": float(threshold),
                "delta_init": float(self.delta),
            }

    def score_trajectory(self, trajectory: BenchmarkTrajectory) -> DiscriminatorOutput:
        task = str(trajectory.task_name)
        detector = self._detectors_per_task.get(task, None)
        if detector is None:
            raise KeyError(
                f"Task {task!r} is not calibrated. Available: "
                f"{sorted(self._detectors_per_task)}"
            )

        T = int(trajectory.num_frames)
        feat, aux_np = self._encode(trajectory)
        aux_scores = aux_np if self._use_aux() else None

        result = detector.detect_sequence(
            features=feat,
            labels=None,
            adaptive_delta=False,
            aux_scores=aux_scores,
        )

        # Align back to trajectory timeline (may be shorter when use_transition_error).
        step_scores = _pad_to_length(result.lambda_values, target_len=T, dtype=np.float32)
        predictions = _pad_to_length(result.preds, target_len=T, dtype=np.int64).astype(np.int64)
        thresholds = _pad_to_length(result.thresholds, target_len=T, dtype=np.float32)
        raw_step = _pad_to_length(result.step_scores, target_len=T, dtype=np.float32)

        positive = np.where(predictions == 1)[0]
        first_failure_frame = int(positive[0]) if positive.size > 0 else None

        aux: dict[str, Any] = {
            "task": task,
            "threshold": float(detector.threshold) if detector.threshold is not None else float("nan"),
            "delta": float(detector.delta),
            "raw_step_scores": raw_step,
            "thresholds": thresholds,
            "lambda_mode": str(self.lambda_mode),
            "lambda_window_size": int(self.lambda_window_size),
            "use_transition_error": bool(self.use_transition_error),
            "transition_aux_weight": float(self.transition_aux_weight),
            "feature_len": int(feat.shape[0]),
        }
        return DiscriminatorOutput(
            step_scores=step_scores,
            predictions=predictions,
            first_failure_frame=first_failure_frame,
            aux=aux,
        )

    def calibration_summary(self) -> dict:
        return {
            "per_task": dict(self._calibration_stats),
            "checkpoint_path": self.checkpoint_path,
            "camera_name": self.camera_name,
            "delta": float(self.delta),
            "delta_step": float(self.delta_step),
            "knn_chunk_size": int(self.knn_chunk_size),
            "lambda_mode": str(self.lambda_mode),
            "lambda_window_size": int(self.lambda_window_size),
            "transition_aux_weight": float(self.transition_aux_weight),
            "use_transition_error": bool(self.use_transition_error),
            "transition_proprio_error_weight": float(self.transition_proprio_error_weight),
            "calib_fraction": float(self.calib_fraction),
            "action_horizon": int(self.extractor.action_horizon),
        }

    def close(self) -> None:
        # Release GPU memory held by the underlying dynamics model.
        self.extractor.model.to("cpu")
        del self.extractor.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
