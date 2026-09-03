"""RPT trajectory-encoding backbone shared by discriminator adapters."""

from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

from benchmark.core import BenchmarkTrajectory, DiscriminatorOutput  # noqa: F401

from robosuite.discriminator.dyn_disc.core.rpt_encoder import RPTTrajectoryEncoder


def _pad_to_length(values: np.ndarray, target_len: int, dtype=np.float32) -> np.ndarray:
    arr = np.asarray(values, dtype=dtype).reshape(-1)
    target = int(target_len)
    if arr.shape[0] >= target:
        return arr[:target].copy()
    if arr.shape[0] == 0:
        return np.zeros((target,), dtype=dtype)
    return np.concatenate(
        [arr, np.full((target - arr.shape[0],), arr[-1], dtype=dtype)]
    )


class DynBenchmarkDiscriminator:
    """RPT-only encoder and trajectory feature cache.

    The historical class name is retained because it is part of the benchmark
    adapter API; dynamics-model behavior is intentionally no longer supported.
    """

    name = "rpt_disc"
    feature_source = "rpt_action_token"

    def __init__(
        self,
        *,
        model_ckpt: str,
        device: str = "cuda",
        encode_batch_size: int = 128,
        proprio_indices: Optional[Sequence[int]] = None,
        camera_to_view: Optional[Dict[str, str]] = None,
        delta: float = 10.0,
        calib_fraction: float = 0.2,
        seed: int = 0,
        verbose_fit: bool = True,
    ) -> None:
        if not (0.0 < float(calib_fraction) < 1.0):
            raise ValueError(f"calib_fraction must be in (0, 1), got {calib_fraction}")
        self.model_ckpt = str(model_ckpt)
        self.device = str(device)
        self.encode_batch_size = int(encode_batch_size)
        self.proprio_indices = (
            None
            if proprio_indices is None
            else np.asarray(proprio_indices, dtype=np.int64)
        )
        self.camera_to_view = dict(camera_to_view) if camera_to_view else None
        self.delta = float(delta)
        self.calib_fraction = float(calib_fraction)
        self.seed = int(seed)
        self.verbose_fit = bool(verbose_fit)

        self.encoder = RPTTrajectoryEncoder(
            model_ckpt=self.model_ckpt,
            device=self.device,
            image_batch_size=self.encode_batch_size,
        )
        self.proprio_map = self.encoder.proprio_map
        if self.camera_to_view is None:
            self.camera_to_view = {view: view for view in self.encoder.view_names}

        self._detectors_per_task: Dict[str, Any] = {}
        self._calibration_stats: Dict[str, dict] = {}
        self._feature_cache: Dict[tuple, torch.Tensor] = {}
        self.batch_infer_times: Optional[List[float]] = None

    @property
    def representation_fingerprint(self) -> str:
        return self.encoder.checkpoint_fingerprint

    def _trajectory_key(self, trajectory: BenchmarkTrajectory) -> tuple[str, str, str]:
        group_key = getattr(trajectory, "demo_path", None)
        if group_key is None:
            group_key = getattr(trajectory, "episode_path", "")
        cache_key = getattr(trajectory, "cache_npz_path", "")
        return (
            str(getattr(trajectory, "file_path", "")),
            str(cache_key or group_key),
            self.representation_fingerprint,
        )

    def _resolve_camera(self, view: str) -> str:
        for camera, view_name in self.camera_to_view.items():
            if view_name == view:
                return camera
        raise KeyError(f"No camera mapped to RPT view {view!r}: {self.camera_to_view}")

    def _slice_proprio(self, states: np.ndarray, task_name: str) -> np.ndarray:
        if self.proprio_indices is not None:
            states = states[:, self.proprio_indices]
        elif self.proprio_map:
            task_cfg = self.proprio_map.get(task_name, {})
            indices = task_cfg.get("indices") if isinstance(task_cfg, dict) else None
            if indices:
                states = states[:, np.asarray(indices, dtype=np.int64)]
        if states.shape[1] != self.encoder.proprio_dim:
            raise ValueError(
                f"RPT expects {self.encoder.proprio_dim} proprio values, got "
                f"{states.shape[1]} for task {task_name!r}"
            )
        return states.astype(np.float32, copy=False)

    def _prepare_trajectory_tensors(self, trajectory: BenchmarkTrajectory) -> Dict[str, Any]:
        cameras = [self._resolve_camera(view) for view in self.encoder.view_names]
        images_by_camera = trajectory.load_images(cameras=cameras)
        states = np.asarray(trajectory.load_states(), dtype=np.float32)
        actions = np.asarray(trajectory.load_actions(), dtype=np.float32)
        lengths = [states.shape[0], actions.shape[0]] + [
            images_by_camera[camera].shape[0] for camera in cameras
        ]
        t_len = int(min(lengths))
        if t_len <= 0:
            raise ValueError(f"Empty trajectory: {trajectory.describe()}")
        actions = actions[:t_len]
        if actions.ndim != 2 or actions.shape[1] != self.encoder.action_dim:
            raise ValueError(
                f"RPT expects actions shaped (T, {self.encoder.action_dim}), got {actions.shape}"
            )

        images: Dict[str, np.ndarray] = {}
        for view in self.encoder.view_names:
            array = np.asarray(images_by_camera[self._resolve_camera(view)][:t_len])
            if not np.issubdtype(array.dtype, np.integer):
                array = array.astype(np.float32, copy=False)
                if array.max(initial=0.0) > 1.5:
                    array = array / 255.0
            images[view] = np.transpose(array, (0, 3, 1, 2))
        return {
            "images": images,
            "proprio": self._slice_proprio(
                states[:t_len], task_name=str(trajectory.task_name)
            ),
            "actions": actions.astype(np.float32, copy=False),
            "t_len": t_len,
        }

    @staticmethod
    def _truncate_prepared(prepared: Dict[str, Any], frame_end: int) -> Dict[str, Any]:
        end = min(int(frame_end), int(prepared["t_len"]))
        if end <= 0:
            raise ValueError(f"frame_end must be positive, got {end}")
        return {
            "images": {key: value[:end] for key, value in prepared["images"].items()},
            "proprio": prepared["proprio"][:end],
            "actions": prepared["actions"][:end],
            "t_len": end,
        }

    def _encode_tensors(self, key: tuple, prepared: Dict[str, Any]) -> torch.Tensor:
        images = {
            view: torch.from_numpy(array) for view, array in prepared["images"].items()
        }
        if self.batch_infer_times is not None:
            torch.cuda.synchronize()
            start = time.perf_counter()
        features = self.encoder.encode_trajectory(
            images, prepared["proprio"], prepared["actions"]
        )
        if self.batch_infer_times is not None:
            torch.cuda.synchronize()
            self.batch_infer_times.append(time.perf_counter() - start)
        output = features.detach().cpu()
        self._feature_cache[key] = output
        return output

    def preload_trajectory(self, trajectory: BenchmarkTrajectory) -> Dict[str, Any]:
        return self._prepare_trajectory_tensors(trajectory)

    @torch.no_grad()
    def encode_preloaded(
        self, trajectory: BenchmarkTrajectory, prepared: Dict[str, Any]
    ) -> torch.Tensor:
        key = self._trajectory_key(trajectory)
        cached = self._feature_cache.get(key)
        return cached if cached is not None else self._encode_tensors(key, prepared)

    @torch.no_grad()
    def _encode(
        self, trajectory: BenchmarkTrajectory, *, frame_end: Optional[int] = None
    ) -> torch.Tensor:
        key: tuple = self._trajectory_key(trajectory)
        if frame_end is not None:
            key += (f"end{int(frame_end)}",)
        cached = self._feature_cache.get(key)
        if cached is not None:
            return cached
        prepared = self._prepare_trajectory_tensors(trajectory)
        if frame_end is not None:
            prepared = self._truncate_prepared(prepared, frame_end)
        return self._encode_tensors(key, prepared)

    def _encode_trajectories(
        self,
        trajectories: Sequence[BenchmarkTrajectory],
        *,
        desc: str = "encode",
    ) -> List[torch.Tensor]:
        iterable: Iterable[BenchmarkTrajectory] = trajectories
        if self.verbose_fit:
            try:
                from tqdm import tqdm

                iterable = tqdm(trajectories, desc=desc, unit="traj", dynamic_ncols=True)
            except ImportError:
                pass
        return [self._encode(trajectory) for trajectory in iterable]

    def fit_on_benchmark(self, trajectories: List[BenchmarkTrajectory]) -> None:
        raise NotImplementedError

    def score_trajectory(self, trajectory: BenchmarkTrajectory) -> DiscriminatorOutput:
        raise NotImplementedError

    def calibration_summary(self) -> dict:
        return {
            "per_task": dict(self._calibration_stats),
            "pretraining_method": "rpt",
            "representation_fingerprint": self.representation_fingerprint,
            "model_ckpt": self.model_ckpt,
            "view_names": list(self.encoder.view_names),
            "camera_to_view": dict(self.camera_to_view),
            "feature_source": self.feature_source,
            "delta": self.delta,
            "calib_fraction": self.calib_fraction,
            "encode_batch_size": self.encode_batch_size,
        }

    def close(self) -> None:
        self.encoder.close()


__all__ = ["DynBenchmarkDiscriminator", "_pad_to_length"]
