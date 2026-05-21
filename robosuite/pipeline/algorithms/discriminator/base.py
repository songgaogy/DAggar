"""Discriminator abstract interface.

All online discriminators consumed by DIPOLE-RL must implement this ABC.

Sign convention (matches the warm-started lpb_v2 BCE head):
    higher logit  →  more failure-like (the policy is failing here;
                     human intervention would be appropriate)
    label = 1     →  intervention / failure chunk
    label = 0     →  expert demo or non-intervention on-policy chunk

Reward composition (consumed by IQL): with the default
`disc_reward_sign="negate_logit"`, `r_disc = -logit`, so that the agent
is *rewarded* for being non-failure-like and *penalized* for being
failure-like.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class DiscriminatorOutput:
    """Per-sample scoring output.

    Shapes:
        logit:         (B,) raw BCE logit; higher = more failure-like.
        prob_failure:  (B,) sigmoid(logit) ≈ P(intervention needed).
        decision:      (B,) bool — predicted "human should intervene".
        metadata:      dict — threshold, normalized margin, etc.
    """

    logit: torch.Tensor
    prob_failure: torch.Tensor
    decision: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DiscriminatorBatch:
    """Mini-batch consumed by `DiscriminatorBase.update`.

    Shapes:
        context:      (B, D_ctx) — frozen encoder latent.
        action_chunk: (B, H, D_a).
        label:        (B,) — 1 = intervention / failure chunk,
                              0 = expert demo or non-intervention on-policy.
    """

    context: torch.Tensor
    action_chunk: torch.Tensor
    label: torch.Tensor


class DiscriminatorBase(ABC):
    """ABC for online-trainable (or frozen) discriminators."""

    @abstractmethod
    def score(
        self,
        *,
        context: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> DiscriminatorOutput:
        """Inference. Must be safe to call under `no_grad`."""

    @abstractmethod
    def update(self, batch: DiscriminatorBatch) -> dict[str, float]:
        """One training step on the trainable head. Returns metrics."""

    @abstractmethod
    def intrinsic_reward(
        self,
        *,
        context: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        """Raw logit used by Q-learning (sign flip is applied at the IQL
        replay boundary based on `cfg.disc_reward_sign`).

        Returns:
            (B,) tensor; higher = more failure-like.
        """

    # Persistence: implementations should return / consume a dict that the
    # DipoleTrainer can checkpoint alongside the actor and Q/V state.

    @abstractmethod
    def state_dict(self) -> dict: ...

    @abstractmethod
    def load_state_dict(self, sd: dict, strict: bool = True) -> None: ...
