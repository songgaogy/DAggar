"""FLOAT benchmark adapter using DINOv2 embeddings + per-task Sinkhorn OT.

Implements the paper spec from `prompt/float.md`:
    lambda_n(T_b) = sum_{ij} mu_{n,i,j}^* * c(phi(o_e,i), phi(o_b,j))
    lambda(T_b)  = min_n lambda_n(T_b)
    Lambda       = Percentile_{1-delta}({lambda(T_{b,k})} over success rollouts)
Threshold is calibrated per task with leave-one-out over the success bank.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from data.utils.benchmark import BenchmarkTrajectory, DiscriminatorOutput

from .float_core import FLOATComputer, IdentityEncoder, Trajectory
from .float_dino_encoder import DEFAULT_IMAGE_SIZE, DinoV2ImageEncoder


class FloatBenchmarkDiscriminator:
    """FLOAT detector plugged into the shared failure-detector benchmark."""

    name = "float_dinov2"

    def __init__(
        self,
        *,
        device: str = "cuda",
        image_size: int = DEFAULT_IMAGE_SIZE,
        encoder_batch_size: int = 64,
        camera_name: str = "agentview",
        sinkhorn_reg: float = 0.05,
        max_iter: int = 300,
        tol: float = 1e-5,
        delta: float = 10.0,
        step_stride: int = 8,
        use_similarity_cost: bool = False,
    ) -> None:
        if float(delta) < 0.0 or float(delta) > 100.0:
            raise ValueError(f"delta must be in [0, 100], got {delta}")
        if int(step_stride) <= 0:
            raise ValueError(f"step_stride must be >=1, got {step_stride}")

        self.encoder = DinoV2ImageEncoder(
            device=device,
            image_size=int(image_size),
            batch_size=int(encoder_batch_size),
        )
        self.camera_name = str(camera_name)
        self.sinkhorn_reg = float(sinkhorn_reg)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.delta = float(delta)
        self.step_stride = int(step_stride)
        self.use_similarity_cost = bool(use_similarity_cost)

        # Populated by fit_on_benchmark.
        self._computers_per_task: dict[str, FLOATComputer] = {}
        self._thresholds_per_task: dict[str, float] = {}
        self._expert_keys_per_task: dict[str, list[tuple[str, str]]] = {}
        self._calibration_stats: dict[str, dict] = {}

        # Embedding cache keyed by (file_path, demo_path).
        self._embedding_cache: dict[tuple[str, str], np.ndarray] = {}

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _trajectory_key(self, trajectory: BenchmarkTrajectory) -> tuple[str, str]:
        return (str(trajectory.file_path), str(trajectory.demo_path))

    def _encode_trajectory(self, trajectory: BenchmarkTrajectory) -> np.ndarray:
        key = self._trajectory_key(trajectory)
        cached = self._embedding_cache.get(key, None)
        if cached is not None:
            return cached

        images_by_cam = trajectory.load_images(cameras=[self.camera_name])
        frames = np.asarray(images_by_cam[self.camera_name], dtype=np.uint8)
        length = min(int(frames.shape[0]), int(trajectory.num_frames))
        if length <= 0:
            raise ValueError(
                f"Trajectory is empty after alignment: {trajectory.describe()}"
            )
        if length < int(frames.shape[0]):
            frames = frames[:length]

        emb = self.encoder.encode_images(frames).astype(np.float32)
        if emb.shape[0] != length:
            raise RuntimeError(
                f"Encoder returned {emb.shape[0]} embeddings for {length} frames"
            )
        self._embedding_cache[key] = emb
        return emb

    def _build_computer(self, expert_embeddings: list[np.ndarray]) -> FLOATComputer:
        experts = [Trajectory(obs=e) for e in expert_embeddings]
        return FLOATComputer(
            experts=experts,
            encoder=IdentityEncoder(),
            sinkhorn_reg=self.sinkhorn_reg,
            max_iter=self.max_iter,
            tol=self.tol,
            pad_to=None,
            use_similarity_cost=self.use_similarity_cost,
            pad_rollout_to_bound=False,
        )

    @staticmethod
    def _percentile_from_delta(values: np.ndarray, delta: float) -> float:
        q = 100.0 * (1.0 - float(delta) / 100.0)
        q = float(np.clip(q, 0.0, 100.0))
        return float(np.percentile(np.asarray(values, dtype=np.float64), q=q))

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def fit_on_benchmark(self, trajectories: list[BenchmarkTrajectory]) -> None:
        """Build per-task expert bank from success trajectories and calibrate Lambda."""
        task_to_success: dict[str, list[BenchmarkTrajectory]] = {}
        for traj in trajectories:
            if bool(traj.is_failure):
                continue
            task_to_success.setdefault(str(traj.task_name), []).append(traj)

        if not task_to_success:
            raise RuntimeError(
                "FLOAT calibration requires success trajectories per task. "
                "Provide --success-root/<task>/video_manifest.jsonl entries."
            )

        for task, succ_list in task_to_success.items():
            if len(succ_list) < 2:
                raise RuntimeError(
                    f"Task {task!r} has only {len(succ_list)} success trajectory; "
                    "leave-one-out calibration needs at least 2."
                )

            embeds = [self._encode_trajectory(t) for t in succ_list]
            computer = self._build_computer(embeds)

            lambdas: list[float] = []
            for k, emb in enumerate(embeds):
                per_expert = computer.lambda_per_expert(Trajectory(obs=emb))
                per_expert = np.delete(per_expert, k)
                lambdas.append(float(np.min(per_expert)))
            threshold = self._percentile_from_delta(
                np.asarray(lambdas, dtype=np.float64), self.delta
            )

            self._computers_per_task[task] = computer
            self._thresholds_per_task[task] = float(threshold)
            self._expert_keys_per_task[task] = [self._trajectory_key(t) for t in succ_list]
            self._calibration_stats[task] = {
                "num_experts": int(len(embeds)),
                "lambda_mean": float(np.mean(lambdas)),
                "lambda_std": float(np.std(lambdas)),
                "lambda_min": float(np.min(lambdas)),
                "lambda_max": float(np.max(lambdas)),
                "threshold": float(threshold),
                "delta": float(self.delta),
            }

    def score_trajectory(self, trajectory: BenchmarkTrajectory) -> DiscriminatorOutput:
        task = str(trajectory.task_name)
        computer = self._computers_per_task.get(task, None)
        if computer is None:
            raise KeyError(
                f"Task {task!r} is not calibrated. Available: "
                f"{sorted(self._computers_per_task)}"
            )
        threshold = float(self._thresholds_per_task[task])

        emb = self._encode_trajectory(trajectory)
        T = int(emb.shape[0])

        exclude_idx: Optional[int] = None
        key = self._trajectory_key(trajectory)
        expert_keys = self._expert_keys_per_task.get(task, [])
        if key in expert_keys:
            exclude_idx = int(expert_keys.index(key))

        rollout = Trajectory(obs=emb)
        step_scores = np.zeros(T, dtype=np.float32)
        last_lambda = 0.0

        for t0 in range(1, T + 1):
            # OT between length-1 rollout prefix and any expert is degenerate
            # (both marginals collapse). Use 0 score until we have >= 2 frames.
            if t0 < 2:
                step_scores[t0 - 1] = 0.0
                continue

            need_update = (t0 % self.step_stride == 0) or (t0 == T) or (last_lambda == 0.0)
            if need_update:
                per_expert = computer.lambda_per_expert(rollout, t0=t0)
                if exclude_idx is not None:
                    per_expert = np.delete(per_expert, exclude_idx)
                last_lambda = float(np.min(per_expert))
            step_scores[t0 - 1] = last_lambda

        predictions = (step_scores > threshold).astype(np.int64)
        positive = np.where(predictions == 1)[0]
        first_failure_frame = int(positive[0]) if positive.size > 0 else None

        aux = {
            "task": task,
            "threshold": float(threshold),
            "delta": float(self.delta),
            "step_stride": int(self.step_stride),
            "encoder": self.encoder.hub_model,
            "image_size": int(self.encoder.image_size),
            "embedding_dim": int(self.encoder.embedding_dim),
            "self_excluded_expert_idx": (
                int(exclude_idx) if exclude_idx is not None else -1
            ),
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
            "delta": float(self.delta),
            "step_stride": int(self.step_stride),
            "sinkhorn_reg": float(self.sinkhorn_reg),
            "max_iter": int(self.max_iter),
            "tol": float(self.tol),
            "camera_name": str(self.camera_name),
            "use_similarity_cost": bool(self.use_similarity_cost),
            "encoder": self.encoder.hub_model,
            "image_size": int(self.encoder.image_size),
            "embedding_dim": int(self.encoder.embedding_dim),
        }

    def close(self) -> None:
        self.encoder.close()
