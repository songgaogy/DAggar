"""logpZO (FAIL-Detect) discriminator plugged into the shared failure benchmark.

Each task gets its own normalizing flow trained on success-trajectory DINOv2
embeddings; a conformal-prediction threshold is calibrated on the same pool.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from benchmark.core import BenchmarkTrajectory, DiscriminatorOutput

from robosuite.discriminator.float.float_dino_encoder import (
    DEFAULT_IMAGE_SIZE,
    DinoV2ImageEncoder,
)

from .monitor import FAILDetectMonitor


class LogpZOBenchmarkDiscriminator:
    """FAIL-Detect / logpZO detector for the failure-detection benchmark."""

    name = "logpZO"

    def __init__(
        self,
        *,
        device: str = "cuda",
        image_size: int = DEFAULT_IMAGE_SIZE,
        encoder_batch_size: int = 64,
        camera_name: str = "agentview",
        num_layers: int = 8,
        hidden_dim: int = 512,
        scale_clamp: float = 3.0,
        epochs: int = 50,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        val_fraction: float = 0.1,
        early_stop_patience: int = 8,
        alpha: float = 0.1,
        calib_fraction: float = 0.2,
        seed: int = 0,
        verbose_fit: bool = True,
    ) -> None:
        if not (0.0 < float(alpha) < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if not (0.0 < float(calib_fraction) < 1.0):
            raise ValueError(
                f"calib_fraction must be in (0, 1); train/calib must be disjoint. got {calib_fraction}"
            )

        self.encoder = DinoV2ImageEncoder(
            device=device,
            image_size=int(image_size),
            batch_size=int(encoder_batch_size),
        )
        self.camera_name = str(camera_name)
        self.num_layers = int(num_layers)
        self.hidden_dim = int(hidden_dim)
        self.scale_clamp = float(scale_clamp)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.val_fraction = float(val_fraction)
        self.early_stop_patience = int(early_stop_patience)
        self.alpha = float(alpha)
        self.calib_fraction = float(calib_fraction)
        self.seed = int(seed)
        self.device = str(device)
        self.verbose_fit = bool(verbose_fit)

        self._monitors_per_task: dict[str, FAILDetectMonitor] = {}
        self._calibration_stats: dict[str, dict] = {}

        # Cache DINOv2 embeddings keyed by source/cache path to avoid
        # re-encoding success trajectories that are also scored later.
        self._embedding_cache: dict[tuple[str, str], np.ndarray] = {}

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _trajectory_key(self, trajectory: BenchmarkTrajectory) -> tuple[str, str]:
        group_key = getattr(trajectory, "demo_path", None)
        if group_key is None:
            group_key = getattr(trajectory, "episode_path", "")
        cache_key = getattr(trajectory, "cache_npz_path", "")
        return (str(getattr(trajectory, "file_path", "")), str(cache_key or group_key))

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

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def fit_on_benchmark(self, trajectories: list[BenchmarkTrajectory]) -> None:
        """Per-task flow training + CP threshold calibration.

        The success pool is split **at the trajectory level** into disjoint
        train and calibration subsets (seeded, fraction `self.calib_fraction`).
        This guarantees the CP quantile is computed on data the flow has never
        seen — a hard requirement for conformal validity.
        """
        task_to_success: dict[str, list[BenchmarkTrajectory]] = {}
        for traj in trajectories:
            if bool(traj.is_failure):
                continue
            task_to_success.setdefault(str(traj.task_name), []).append(traj)

        if not task_to_success:
            raise RuntimeError(
                "logpZO requires success trajectories per task. "
                "Provide success trajectories or a real-world cache with success entries."
            )

        for task, succ_list in task_to_success.items():
            if len(succ_list) < 2:
                raise RuntimeError(
                    f"Task {task!r} has only {len(succ_list)} success trajectory; "
                    "need at least 2 for disjoint train + calibration split."
                )

            # Trajectory-level seeded split (disjoint by construction).
            rng = np.random.default_rng(int(self.seed))
            perm = rng.permutation(len(succ_list))
            n_calib = int(round(self.calib_fraction * len(succ_list)))
            n_calib = max(1, min(len(succ_list) - 1, n_calib))
            calib_idx = set(perm[:n_calib].tolist())
            train_trajs = [t for i, t in enumerate(succ_list) if i not in calib_idx]
            calib_trajs = [t for i, t in enumerate(succ_list) if i in calib_idx]

            train_feats = np.concatenate(
                [self._encode_trajectory(t) for t in train_trajs], axis=0
            ).astype(np.float32)
            calib_feats_per_traj: list[np.ndarray] = [
                self._encode_trajectory(t).astype(np.float32) for t in calib_trajs
            ]
            num_calib_frames = int(sum(int(f.shape[0]) for f in calib_feats_per_traj))

            if self.verbose_fit:
                print(
                    f"[logpZO][fit] task={task} "
                    f"train_trajs={len(train_trajs)} ({int(train_feats.shape[0])} frames)  "
                    f"calib_trajs={len(calib_trajs)} ({num_calib_frames} frames)  "
                    f"dim={int(train_feats.shape[1])}"
                )

            monitor = FAILDetectMonitor(
                feature_dim=int(train_feats.shape[1]),
                num_layers=self.num_layers,
                hidden_dim=self.hidden_dim,
                scale_clamp=self.scale_clamp,
                device=self.device,
            )
            fit_stats = monitor.fit_score_model(
                train_feats,
                epochs=self.epochs,
                batch_size=self.batch_size,
                lr=self.lr,
                weight_decay=self.weight_decay,
                val_fraction=self.val_fraction,
                early_stop_patience=self.early_stop_patience,
                seed=self.seed,
                verbose=self.verbose_fit,
            )
            # Paper Sec. IV-B: time-varying CP band over success calibration trajectories.
            monitor.calibrate_functional_cp_band(calib_feats_per_traj, alpha=self.alpha)

            self._monitors_per_task[task] = monitor
            self._calibration_stats[task] = {
                "num_success_trajectories": int(len(succ_list)),
                "num_train_trajectories": int(len(train_trajs)),
                "num_calib_trajectories": int(len(calib_trajs)),
                "num_train_frames": int(train_feats.shape[0]),
                "num_calib_frames": num_calib_frames,
                "fit_epochs": int(fit_stats.epochs),
                "final_train_nll": float(fit_stats.final_train_nll),
                "best_val_nll": float(fit_stats.best_val_nll),
                **monitor.calibration_summary(),
            }

    def score_trajectory(self, trajectory: BenchmarkTrajectory) -> DiscriminatorOutput:
        task = str(trajectory.task_name)
        monitor = self._monitors_per_task.get(task, None)
        if monitor is None:
            raise KeyError(
                f"Task {task!r} is not calibrated. Available: "
                f"{sorted(self._monitors_per_task)}"
            )

        emb = self._encode_trajectory(trajectory)
        T = int(emb.shape[0])
        step_scores = monitor.score_features(emb).astype(np.float32)
        if step_scores.shape[0] != T:
            raise RuntimeError(
                f"score length mismatch: got {step_scores.shape[0]}, expected {T}"
            )

        # Paper Sec. IV-B: time-varying threshold eta_t = mu_t + eta * sigma_t.
        eta_t = monitor.threshold_per_step(T).astype(np.float32)
        predictions = (step_scores > eta_t).astype(np.int64)
        positive = np.where(predictions == 1)[0]
        first_failure_frame = int(positive[0]) if positive.size > 0 else None

        aux = {
            "task": task,
            "alpha": float(self.alpha),
            "threshold_per_step_mean": float(eta_t.mean()),
            "threshold_per_step_min": float(eta_t.min()),
            "threshold_per_step_max": float(eta_t.max()),
            "camera_name": str(self.camera_name),
            "encoder": self.encoder.hub_model,
            "image_size": int(self.encoder.image_size),
            "embedding_dim": int(self.encoder.embedding_dim),
            "flow_num_layers": int(self.num_layers),
            "flow_hidden_dim": int(self.hidden_dim),
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
            "alpha": float(self.alpha),
            "calib_fraction": float(self.calib_fraction),
            "camera_name": str(self.camera_name),
            "encoder": self.encoder.hub_model,
            "image_size": int(self.encoder.image_size),
            "embedding_dim": int(self.encoder.embedding_dim),
            "flow": {
                "num_layers": int(self.num_layers),
                "hidden_dim": int(self.hidden_dim),
                "scale_clamp": float(self.scale_clamp),
                "epochs": int(self.epochs),
                "batch_size": int(self.batch_size),
                "lr": float(self.lr),
                "weight_decay": float(self.weight_decay),
                "val_fraction": float(self.val_fraction),
                "early_stop_patience": int(self.early_stop_patience),
            },
        }

    def close(self) -> None:
        self.encoder.close()
