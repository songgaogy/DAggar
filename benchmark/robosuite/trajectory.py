"""Robosuite-schema concrete BenchmarkTrajectory.

Reads an HDF5 group laid out as:
    <demo_path>/observations/<cam>/images   (T, H, W, 3) uint8
    <demo_path>/states                      (T, S)       float64
    <demo_path>/actions                     (T, A)       float32
    <demo_path>/annotations/failure_frame_mask     (T,)  uint8   (failures only)
    <demo_path>/annotations/failure_segment_index  (T,)  int32   (failures only)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Sequence

import h5py
import numpy as np

from benchmark.core.trajectory import BenchmarkTrajectory


@dataclass
class RobosuiteBenchmarkTrajectory(BenchmarkTrajectory):
    """BenchmarkTrajectory backed by a robosuite-style HDF5 demo group."""

    file_path: str = ""
    demo_path: str = ""
    cache_npz_path: str = ""

    def _has_cache(self) -> bool:
        return bool(self.cache_npz_path)

    def _cache_array(self, name: str) -> np.ndarray:
        if os.path.isdir(self.cache_npz_path):
            return np.load(
                os.path.join(self.cache_npz_path, f"{name}.npy"),
                mmap_mode="r",
            )
        with np.load(self.cache_npz_path, allow_pickle=False) as data:
            return data[name][:]

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
        if self._has_cache():
            images_chw = self._cache_array("images_chw")
            cam_to_idx = {name: i for i, name in enumerate(self.available_cameras)}
            for cam in req:
                out[cam] = np.transpose(images_chw[:, cam_to_idx[cam]], (0, 2, 3, 1))
            return out

        with h5py.File(self.file_path, "r") as f:
            obs = f[self.demo_path]["observations"]
            for cam in req:
                out[cam] = obs[cam]["images"][:]
        return out

    def load_states(self) -> np.ndarray:
        if self._has_cache():
            return np.asarray(self._cache_array("proprio"), dtype=np.float32)
        with h5py.File(self.file_path, "r") as f:
            return f[self.demo_path]["states"][:]

    def load_actions(self) -> np.ndarray:
        if self._has_cache():
            return np.asarray(self._cache_array("actions"), dtype=np.float32)
        with h5py.File(self.file_path, "r") as f:
            return f[self.demo_path]["actions"][:]

    def load_failure_mask(self) -> Optional[np.ndarray]:
        if not self.is_failure:
            return None
        if self._has_cache():
            return np.asarray(self._cache_array("failure_mask"), dtype=np.uint8)
        with h5py.File(self.file_path, "r") as f:
            return f[self.demo_path]["annotations"]["failure_frame_mask"][:]

    def load_failure_segment_index(self) -> Optional[np.ndarray]:
        if not self.is_failure:
            return None
        if self._has_cache():
            return np.asarray(self._cache_array("failure_segment_index"), dtype=np.int32)
        with h5py.File(self.file_path, "r") as f:
            return f[self.demo_path]["annotations"]["failure_segment_index"][:]
