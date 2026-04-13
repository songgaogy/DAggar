from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib import colors as mcolors


@dataclass(frozen=True)
class SuboptimalDemoRef:
    task_name: str
    file_path: str
    demo_key: str
    sub_start: int
    sub_stop: int


@dataclass(frozen=True)
class FailureSegment:
    start: int
    end: int
    mode: str


@dataclass(frozen=True)
class AnnotatedFailureDemoRef:
    task_name: str
    file_path: str
    demo_key: str
    failure_mask: np.ndarray
    failure_segments: tuple[FailureSegment, ...]
    source_task_name: str


@dataclass(frozen=True)
class TrajectorySequence:
    latents: np.ndarray
    failure_labels: np.ndarray
    task_name: str
    task_index: int
    source_name: str
    split: str
    file_path: str
    demo_key: str
    rollout_id: str
    ood_start: int | None = None
    ood_stop: int | None = None
    failure_segments: tuple[FailureSegment, ...] = ()


SUCCESS_COLOR = "#B9D6F2"
FAILURE_NORMAL_COLOR = "#2563EB"
FAILURE_START_COLOR = "#D94841"
FAILURE_END_COLOR = "#F2C14E"
EXPERT_COLOR = SUCCESS_COLOR
SUBOPTIMAL_COLOR = FAILURE_NORMAL_COLOR
OOD_COLOR = FAILURE_START_COLOR
BACKGROUND_FACE_COLOR = "#FAFBFD"
GRID_COLOR = "#DCE3EC"
ANNOTATION_TEXT = "OOD segment fades red -> blue from sub_start to sub_stop."
ANNOTATED_FAILURE_TEXT = "All tasks and failure modes are pooled; failure frames fade red -> yellow."


def _blend_hex(start_hex: str, end_hex: str, weight: float) -> str:
    start_rgb = np.asarray(mcolors.to_rgb(start_hex), dtype=np.float32)
    end_rgb = np.asarray(mcolors.to_rgb(end_hex), dtype=np.float32)
    alpha = float(np.clip(weight, 0.0, 1.0))
    blended = (1.0 - alpha) * start_rgb + alpha * end_rgb
    return str(mcolors.to_hex(blended))
