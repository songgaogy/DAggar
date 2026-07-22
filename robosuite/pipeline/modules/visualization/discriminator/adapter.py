"""Pipeline-owned adapter behavior for collected offline episodes."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from benchmark.core import BenchmarkTrajectory

from robosuite.discriminator.dyn_disc.adapters.pu_bce import (
    PUBCEBenchmarkDiscriminator,
)


class FinetunedPUBCEBenchmarkDiscriminator(PUBCEBenchmarkDiscriminator):
    """Accept already-selected policy proprio without changing dyn-disc adapters."""

    _preselected_proprio_active = False

    def _trajectory_key(self, trajectory: BenchmarkTrajectory) -> tuple:
        """Disambiguate the feature cache per trajectory.

        In-memory offline episodes share the same source file and expose no
        ``file_path``/``demo_path``/``cache_npz_path``, so the base key collides
        across episodes and the first episode's latents get reused for all of
        them. Append stable per-trajectory identifiers while preserving the base
        feature-contract components.
        """
        base = super()._trajectory_key(trajectory)
        identity = (
            str(getattr(trajectory, "video_id", "")),
            str(getattr(trajectory, "source_hdf5_path", "")),
            str(getattr(trajectory, "source_demo_key", "")),
        )
        return tuple(base) + identity

    def _prepare_trajectory_tensors(
        self,
        trajectory: BenchmarkTrajectory,
    ) -> dict[str, Any]:
        self._preselected_proprio_active = bool(
            getattr(trajectory, "states_are_preselected_proprio", False)
        )
        try:
            return super()._prepare_trajectory_tensors(trajectory)
        finally:
            self._preselected_proprio_active = False

    def _slice_proprio(
        self,
        states: np.ndarray,
        target_dim: Optional[int],
        task_name: Optional[str] = None,
    ) -> np.ndarray:
        if not self._preselected_proprio_active:
            return super()._slice_proprio(
                states,
                target_dim=target_dim,
                task_name=task_name,
            )

        proprio = np.asarray(states, dtype=np.float32)
        if proprio.ndim != 2:
            raise ValueError(
                f"Preselected trajectory proprio must be (T, D), got {proprio.shape}."
            )
        if target_dim is not None and int(proprio.shape[1]) != int(target_dim):
            raise ValueError(
                "Preselected trajectory proprio must exactly match the frozen "
                f"encoder input dim {target_dim}; got {proprio.shape[1]}."
            )
        return np.ascontiguousarray(proprio, dtype=np.float32)


__all__ = ["FinetunedPUBCEBenchmarkDiscriminator"]
