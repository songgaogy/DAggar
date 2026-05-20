"""Balanced replay buffer for online discriminator training.

Two pools:
    expert_pool : transitions tagged is_intervention=True (human takeovers)
                  + optional bootstrap from offline demos.
    policy_pool : transitions from the current policy rollout (non-intervention).

`sample()` returns a 50/50 (configurable via balance_ratio) mini-batch.
Expert pool is typically much smaller than policy pool — sample with
replacement from expert and without replacement from policy.

The buffer never stores the encoder latent; it stores raw transitions and
runs the frozen encoder at sample time. This keeps encoder weights and
buffer state decoupled (so re-binding cameras or switching encoders never
invalidates the buffer).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .base import DiscriminatorBatch
from .online_bce import DiscriminatorConfig

if TYPE_CHECKING:
    from robosuite.pipeline.common.types import Transition

    from .encoder import SharedFrozenEncoder


@dataclass
class _Pool:
    """Internal helper: ring buffer of transitions with capacity cap."""

    capacity: int

    def append(self, t: "Transition") -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class DiscriminatorReplayBuffer:
    """Balanced two-pool buffer for the BCE discriminator.

    Args:
        cfg: DiscriminatorConfig (uses batch_size, balance_ratio).
        expert_capacity: max expert-pool size.
        policy_capacity: max policy-pool size.
    """

    def __init__(
        self,
        cfg: DiscriminatorConfig,
        *,
        expert_capacity: int = 50_000,
        policy_capacity: int = 200_000,
    ) -> None:
        self.cfg = cfg
        self.expert_capacity = expert_capacity
        self.policy_capacity = policy_capacity

    def add_from_transition(self, t: "Transition") -> None:
        """Route a transition into expert_pool if `t.is_intervention` else
        policy_pool. Called by DipoleTrainer.record_transition."""
        raise NotImplementedError

    def bootstrap_from_demos(self, demos: list["Transition"]) -> None:
        """Optionally seed expert_pool from offline demonstrations."""
        raise NotImplementedError

    def sample(
        self,
        batch_size: int,
        *,
        encoder: "SharedFrozenEncoder",
        device: str,
    ) -> DiscriminatorBatch:
        """Return a balanced mini-batch with frozen latents pre-computed.

        Sampling protocol:
            n_expert = round(batch_size * balance_ratio / (1 + balance_ratio))
            n_policy = batch_size - n_expert
        Use replacement for expert when len(expert_pool) < n_expert.
        """
        raise NotImplementedError

    def ready(self, batch_size: int) -> bool:
        """At least 1 expert and `n_policy` policy samples available."""
        raise NotImplementedError
