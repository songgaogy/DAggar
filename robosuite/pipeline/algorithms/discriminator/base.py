"""Discriminator abstract interface.

All online discriminators consumed by DIPOLE-RL must implement this ABC.
The sign convention is: `intrinsic_reward` returns higher values for more
expert-like (s, a) — i.e. r_disc = -logit when `disc_reward_sign` is the
default `negate_logit`.
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
        logit:        (B,) raw BCE logit; higher = more expert-like.
        prob_expert:  (B,) sigmoid(logit).
        decision:     (B,) bool — predicted "human should NOT intervene".
        metadata:     dict — threshold, normalized margin, etc.
    """

    logit: torch.Tensor
    prob_expert: torch.Tensor
    decision: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DiscriminatorBatch:
    """Mini-batch consumed by `DiscriminatorBase.update`.

    Shapes:
        context:      (B, D_ctx) — frozen encoder latent.
        action_chunk: (B, H, D_a).
        label:        (B,) — 1 = expert (intervention), 0 = current-policy
                      rollout (non-intervention).
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
        """Scalar reward used by Q-learning.

        Returns:
            (B,) tensor; higher = more expert-like.
        """

    # Persistence: implementations should return / consume a dict that the
    # DipoleTrainer can checkpoint alongside the actor and Q/V state.

    @abstractmethod
    def state_dict(self) -> dict: ...

    @abstractmethod
    def load_state_dict(self, sd: dict, strict: bool = True) -> None: ...
