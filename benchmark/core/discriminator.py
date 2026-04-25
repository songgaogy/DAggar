"""Discriminator interface contract.

Any method plugged into FailureBenchmark must implement `score_trajectory`.
All outputs are over the original (unpadded) frame timeline of the
trajectory; alignment with GT frame mask must be 1-to-1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from .trajectory import BenchmarkTrajectory


@dataclass
class DiscriminatorOutput:
    """Per-trajectory scoring output.

    step_scores
        (T,) float. Higher value => more failure-like. Required.
    predictions
        (T,) int in {0, 1}. Optional.
    first_failure_frame
        Optional int. Derived from `predictions` if absent.
    aux
        Free-form dictionary for method-specific outputs.
    """

    step_scores: np.ndarray
    predictions: Optional[np.ndarray] = None
    first_failure_frame: Optional[int] = None
    aux: Optional[dict] = None

    def validate(self, expected_T: int) -> None:
        s = np.asarray(self.step_scores)
        if s.ndim != 1 or int(s.shape[0]) != int(expected_T):
            raise ValueError(
                f"step_scores must be shape ({expected_T},), got {s.shape}"
            )
        if self.predictions is not None:
            p = np.asarray(self.predictions)
            if p.ndim != 1 or int(p.shape[0]) != int(expected_T):
                raise ValueError(
                    f"predictions must be shape ({expected_T},), got {p.shape}"
                )


@runtime_checkable
class Discriminator(Protocol):
    """Protocol any discriminator must satisfy."""

    name: str

    def score_trajectory(
        self,
        trajectory: BenchmarkTrajectory,
    ) -> DiscriminatorOutput: ...
