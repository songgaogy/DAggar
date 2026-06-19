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
