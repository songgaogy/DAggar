"""Chunk-centric replay sampler for IQL.

Wraps the same underlying `Transition` store used by `DipoleReplayBuffer`
to avoid duplicating frames. On sample it:
  1. Picks valid windows where all H steps share the same episode.
  2. Runs the frozen `SharedFrozenEncoder` on s and s'.
  3. Synthesizes `r_total = r_env + λ_disc · disc.intrinsic_reward(...)`
     using the CURRENT discriminator (re-evaluated per sample to avoid
     stale rewards).
  4. Returns `IQLStepBatch` / `IQLActorBatch`.

Mirror baseline awr/replay_buffer.py:295-510 for the windowing logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from .common import IQLActorBatch, IQLConfig, IQLStepBatch

if TYPE_CHECKING:
    from robosuite.pipeline.algorithms.discriminator.encoder import SharedFrozenEncoder
    from robosuite.pipeline.algorithms.discriminator.online_bce import (
        OnlineBCEDiscriminator,
    )


class IQLReplayBuffer:
    """Chunk-centric replay buffer.

    Args:
        base_buffer: the underlying transition store (typically the DIPOLE
            online buffer; passed by reference, not copied).
        cfg: IQLConfig (carries action_horizon, discount, disc_reward_coef).
    """

    def __init__(self, base_buffer: Any, cfg: IQLConfig) -> None:
        self._base = base_buffer
        self.cfg = cfg

    # ------------------------------------------------------------------ #
    # Sampling                                                            #
    # ------------------------------------------------------------------ #

    def sample_step_batch(
        self,
        batch_size: int,
        *,
        encoder: "SharedFrozenEncoder",
        discriminator: "OnlineBCEDiscriminator",
        device: str,
    ) -> IQLStepBatch:
        """Sample `batch_size` chunk windows, encode contexts under no_grad,
        and synthesize total rewards.

        Returns:
            IQLStepBatch with all tensors on `device`.
        """
        raise NotImplementedError

    def sample_actor_batch(
        self,
        batch_size: int,
        *,
        encoder: "SharedFrozenEncoder",
        device: str,
    ) -> IQLActorBatch:
        """Sample chunks for actor-side advantage scoring (no rewards needed)."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Capacity / readiness                                                #
    # ------------------------------------------------------------------ #

    def ready(self, batch_size: int) -> bool:
        """True iff enough valid windows exist to sample `batch_size`."""
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
