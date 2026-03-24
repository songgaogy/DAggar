from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Optional, Sequence, TypeVar

import numpy as np

from .types import DetectorCalibrationSummary, TrajectoryDetectionResult


TTrajectory = TypeVar("TTrajectory")


class OfflineTrajectoryDiscriminator(ABC, Generic[TTrajectory]):
    """
    Base API for offline trajectory-level failure detectors.

    A discriminator implementing this API should:
    1. fit / calibrate on normal trajectories,
    2. detect failures on one trajectory,
    3. expose all step-wise outputs needed for metrics or visualization.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a short stable detector name."""

    @abstractmethod
    def fit(
        self,
        normal_bank_trajectories: Sequence[TTrajectory],
        calibration_trajectories: Optional[Sequence[TTrajectory]] = None,
    ) -> DetectorCalibrationSummary:
        """Fit bank / calibrator state and return a calibration summary."""

    @abstractmethod
    def detect_trajectory(
        self,
        trajectory: TTrajectory,
        *,
        labels: Optional[np.ndarray] = None,
        adaptive_threshold: bool = False,
        delta_min: float = 0.0,
        delta_max: float = 100.0,
        warmup_steps: int = 0,
        update_interval: int = 1,
    ) -> TrajectoryDetectionResult:
        """Run detection on one trajectory."""

    def close(self) -> None:
        """Release optional resources."""

