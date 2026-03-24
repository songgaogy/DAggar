from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class DetectorCalibrationSummary:
    detector_name: str
    threshold: Optional[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajectoryDetectionResult:
    detector_name: str
    step_scores: np.ndarray
    aggregate_scores: np.ndarray
    thresholds: np.ndarray
    predictions: np.ndarray
    aux_scores: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VideoRenderRecord:
    trajectory_id: int
    source_file: str
    demo_key: str
    num_frames: int
    first_pred_failure_frame: Optional[int]
    pred_failure_frame_count: int
    gt_failure_frame_count: int
    video_path: str
    metadata: dict[str, Any] = field(default_factory=dict)
