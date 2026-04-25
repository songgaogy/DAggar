"""Agilex-schema concrete BenchmarkTrajectory.

Two on-disk layouts are supported through a single class:

  Annotated failures (``failure_annotations/out_by_task/<task>/out.hdf5``)
      The episode group lives at ``episodes/<split>/<episode_N>``; pass that
      path via ``episode_path``. The group exposes:
          {episode_path}/action                           (T, 14) float32
          {episode_path}/observations/<proprio_field>     (T, ...)
          {episode_path}/observations/images/<cam>        (T, H, W, 3) uint8
          {episode_path}/annotations/failure_frame_mask   (T,)    uint8
          {episode_path}/annotations/failure_segment_index(T,)    int32

  Raw success episodes (``<task>/success_rollout/episode_<N>.hdf5``)
      Same fields live at the root of the file. Set ``episode_path = ""``.

Slicing knobs:
  * ``action_slice``   - subset of the (T, 14) action returned by ``load_actions``.
                         Default ``slice(7, 13)`` -> right arm 6-DoF (no gripper).
  * ``proprio_slice``  - subset of the proprio field returned by ``load_states``.
                         Default ``slice(7, 14)`` -> right arm 7-D.
  * ``proprio_field``  - which observation key to read for the state vector.
                         Default ``"qpos"``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import h5py
import numpy as np

from benchmark.core.trajectory import BenchmarkTrajectory


@dataclass
class AgilexBenchmarkTrajectory(BenchmarkTrajectory):
    file_path: str = ""
    episode_path: str = ""               # "" => fields at file root (raw success)
    proprio_field: str = "qpos"
    proprio_slice: slice = field(default_factory=lambda: slice(7, 14))
    action_slice: slice = field(default_factory=lambda: slice(7, 13))

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _episode_group(self, hf: h5py.File):
        """Return the h5py group that owns action/observations/annotations."""
        return hf if not self.episode_path else hf[self.episode_path]

    # ------------------------------------------------------------------ #
    # Lazy loaders                                                       #
    # ------------------------------------------------------------------ #

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
            g = self._episode_group(f)
            imgs = g["observations"]["images"]
            for cam in req:
                out[cam] = imgs[cam][:]
        return out

    def load_states(self) -> np.ndarray:
        with h5py.File(self.file_path, "r") as f:
            g = self._episode_group(f)
            arr = g["observations"][self.proprio_field][:]
        return arr[:, self.proprio_slice] if arr.ndim == 2 else arr

    def load_actions(self) -> np.ndarray:
        with h5py.File(self.file_path, "r") as f:
            g = self._episode_group(f)
            arr = g["action"][:]
        return arr[:, self.action_slice] if arr.ndim == 2 else arr

    def load_failure_mask(self) -> Optional[np.ndarray]:
        if not self.is_failure:
            return None
        with h5py.File(self.file_path, "r") as f:
            g = self._episode_group(f)
            return g["annotations"]["failure_frame_mask"][:]

    def load_failure_segment_index(self) -> Optional[np.ndarray]:
        if not self.is_failure:
            return None
        with h5py.File(self.file_path, "r") as f:
            g = self._episode_group(f)
            return g["annotations"]["failure_segment_index"][:]
