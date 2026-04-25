"""BenchmarkTrajectory abstract base.

Holds the metadata fields that the orchestrator (FailureBenchmark) and
metrics need, and declares the lazy data-loading methods that concrete
subclasses (robosuite / agilex / ...) must implement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


@dataclass
class BenchmarkTrajectory:
    """Per-trajectory descriptor; subclasses implement the load_* methods."""

    task_name: str
    num_frames: int
    is_failure: bool
    video_id: str
    fps: int = 20
    available_cameras: tuple[str, ...] = field(default_factory=tuple)
    failure_segments: list[dict] = field(default_factory=list)   # [{start,end,mode}]
    source_hdf5_path: str = ""
    source_demo_key: str = ""

    # ------------------------------------------------------------------ #
    # Lazy loaders (override in subclasses)                              #
    # ------------------------------------------------------------------ #

    def load_images(
        self,
        cameras: Optional[Sequence[str]] = None,
    ) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def load_states(self) -> np.ndarray:
        raise NotImplementedError

    def load_actions(self) -> np.ndarray:
        raise NotImplementedError

    def load_failure_mask(self) -> Optional[np.ndarray]:
        raise NotImplementedError

    def load_failure_segment_index(self) -> Optional[np.ndarray]:
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Generic helpers                                                    #
    # ------------------------------------------------------------------ #

    def first_gt_failure_frame(self) -> Optional[int]:
        if not self.is_failure or not self.failure_segments:
            return None
        return int(min(int(seg["start"]) for seg in self.failure_segments))

    def describe(self) -> str:
        tag = "FAIL" if self.is_failure else "SUCC"
        return (
            f"[{tag}] task={self.task_name} video_id={self.video_id} "
            f"T={self.num_frames} segs={len(self.failure_segments)}"
        )
