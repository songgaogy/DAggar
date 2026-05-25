"""Discriminator abstract interface.

All online discriminators consumed by DIPOLE-RL must implement this ABC.

Sign convention (matches the warm-started lpb_v2 BCE head):
    higher logit  →  more failure-like (the policy is failing here;
                     human intervention would be appropriate)
    label = 1     →  intervention / failure frame
    label = 0     →  expert demo or non-intervention on-policy frame

Granularity: single-frame. The discriminator scores `context: (B, D_ctx)`
where each row is one frozen-encoder latent — no per-chunk action
flattening. This mirrors the lpb v2 reference BCE head, whose first
Linear is `(hidden, D_ctx)`.

Reward composition (consumed by IQL): `intrinsic_reward(...)` returns
`-sigmoid(logit - threshold)` ∈ (-1, 0). The IQL replay encodes every
frame of the H-step chunk and γ-aggregates the per-frame rewards via the
same `aggregate_chunk_reward` path that handles r_env.
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
        context:      (B, D_ctx) — frozen encoder latent (single frame).
        label:        (B,) — 1 = intervention / failure frame,
                              0 = expert demo or non-intervention on-policy.
    """

    context: torch.Tensor
    label: torch.Tensor


class DiscriminatorBase(ABC):
    """ABC for online-trainable (or frozen) discriminators."""

    @abstractmethod
    def score(
        self,
        *,
        context: torch.Tensor,
    ) -> DiscriminatorOutput:
        """Inference on per-frame latents. Must be safe to call under `no_grad`."""

    @abstractmethod
    def update(self, batch: DiscriminatorBatch) -> dict[str, float]:
        """One training step on the trainable head. Returns metrics."""

    @abstractmethod
    def intrinsic_reward(
        self,
        *,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Per-frame intrinsic reward used by Q-learning.

        Returns:
            (B,) tensor in (-1, 0); failure-like → ≈ -1, expert-like → ≈ 0.
            The IQL replay calls this once per frame in the H-step chunk and
            γ-aggregates with r_env (no further sign manipulation).
        """

    # Persistence: implementations should return / consume a dict that the
    # DipoleTrainer can checkpoint alongside the actor and Q/V state.

    @abstractmethod
    def state_dict(self) -> dict: ...

    @abstractmethod
    def load_state_dict(self, sd: dict, strict: bool = True) -> None: ...
