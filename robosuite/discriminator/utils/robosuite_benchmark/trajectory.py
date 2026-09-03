"""Robosuite benchmark trajectories for the new data/<task>/<split> layout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import h5py
import numpy as np

from benchmark.core.trajectory import BenchmarkTrajectory


@dataclass
class RobosuiteBenchmarkTrajectory(BenchmarkTrajectory):
    """BenchmarkTrajectory backed by one demo group in a robosuite HDF5 file."""

    file_path: str = ""
    demo_path: str = ""
    split: str = ""

    def _demo_group(self, handle: h5py.File):
        return handle[self.demo_path]

    def load_images(
        self,
        cameras: Optional[Sequence[str]] = None,
    ) -> dict[str, np.ndarray]:
        req = tuple(cameras) if cameras is not None else self.available_cameras
        missing = [c for c in req if c not in self.available_cameras]
        if missing:
            raise KeyError(
                f"cameras {missing} not available for {self.video_id}; "
                f"available: {self.available_cameras}"
            )

        out: dict[str, np.ndarray] = {}
        with h5py.File(self.file_path, "r") as f:
            obs = self._demo_group(f)["observations"]
            for cam in req:
                out[cam] = obs[cam]["images"][:]
        return out

    def load_states(self) -> np.ndarray:
        with h5py.File(self.file_path, "r") as f:
            return self._demo_group(f)["states"][:]

    def load_actions(self) -> np.ndarray:
        with h5py.File(self.file_path, "r") as f:
            return self._demo_group(f)["actions"][:]

    def load_model_inputs(
        self,
        cameras: Sequence[str],
        frame_end: Optional[int] = None,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Load all model inputs while opening the trajectory file only once."""
        req = tuple(cameras)
        missing = [camera for camera in req if camera not in self.available_cameras]
        if missing:
            raise KeyError(
                f"cameras {missing} not available for {self.video_id}; "
                f"available: {self.available_cameras}"
            )

        with h5py.File(self.file_path, "r") as handle:
            demo = self._demo_group(handle)
            observations = demo["observations"]
            selection = slice(None) if frame_end is None else slice(0, int(frame_end))
            images = {
                camera: observations[camera]["images"][selection]
                for camera in req
            }
            states = demo["states"][selection]
            actions = demo["actions"][selection]
        return images, states, actions

    def load_failure_mask(self) -> Optional[np.ndarray]:
        if not self.is_failure:
            return None
        with h5py.File(self.file_path, "r") as f:
            return self._demo_group(f)["annotations"]["failure_frame_mask"][:]

    def load_failure_segment_index(self) -> Optional[np.ndarray]:
        if not self.is_failure:
            return None
        with h5py.File(self.file_path, "r") as f:
            ann = self._demo_group(f)["annotations"]
            if "failure_segment_index" not in ann:
                return None
            return ann["failure_segment_index"][:]

    def load_is_success(self) -> Optional[np.ndarray]:
        """Per-frame flag: True if task success already occurred before this step."""
        with h5py.File(self.file_path, "r") as f:
            g = self._demo_group(f)
            if "is_success" not in g:
                return None
            return np.asarray(g["is_success"][:], dtype=bool)

    def prefix_frames_before_done(self) -> int:
        """Exclusive end index for pre-success frames (from ``is_success`` label)."""
        mask = self.load_is_success()
        if mask is None:
            return int(self.num_frames)
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        n = int(min(int(self.num_frames), int(mask.shape[0])))
        if n <= 0:
            return 0
        mask = mask[:n]
        if not mask.any():
            return n
        return int(np.argmax(mask))
