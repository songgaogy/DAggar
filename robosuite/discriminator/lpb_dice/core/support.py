from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SupportPenaltyCalibration:
    mean: float
    std: float

    def normalize(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        scale = float(self.std) if float(self.std) > 1e-6 else 1.0
        normalized = (arr - float(self.mean)) / scale
        return np.maximum(normalized, 0.0).astype(np.float32, copy=False)


def calibrate_support_penalty(values: list[np.ndarray]) -> SupportPenaltyCalibration:
    if not values:
        return SupportPenaltyCalibration(mean=0.0, std=1.0)
    merged = np.concatenate([np.asarray(item, dtype=np.float32).reshape(-1) for item in values], axis=0)
    if merged.size <= 0:
        return SupportPenaltyCalibration(mean=0.0, std=1.0)
    return SupportPenaltyCalibration(
        mean=float(np.mean(merged)),
        std=float(max(np.std(merged), 1e-6)),
    )

