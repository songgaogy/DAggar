"""G provider that mixes Q-V advantage with the online BCE logit.

Replaces `LPBV2GProvider` when DipoleConfig.g_mode == "advantage".
Implements the exact same `compute_g_for_batch(batch) -> (B,)` contract so
`DipoleFlowPolicy.update()` is unchanged.

Math:
    A(s, a)        = min(Q1(s, a), Q2(s, a)) - V(s)
    A_norm         = batch_zscore(A)
    disc_logit     = OnlineBCEDiscriminator.score(s, a).logit
    disc_norm      = batch_zscore(disc_logit)
    G              = alpha * A_norm + beta * (-disc_norm)
                              ^                  ^
                              advantage          disc; sign-flip keeps
                                                 higher = more failure-like
                                                 to match DIPOLE's existing
                                                 sigmoid convention.

The flow policy then maps G -> w_pos = sigmoid(beta_policy * G + k).

This object owns nothing it doesn't construct: the IQL learner, the
discriminator, and the encoder are all injected.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from robosuite.pipeline.algorithms.discriminator.encoder import SharedFrozenEncoder
    from robosuite.pipeline.algorithms.discriminator.online_bce import (
        OnlineBCEDiscriminator,
    )
    from robosuite.pipeline.algorithms.q_learning.iql import IQLLearner


class AdvantageGProvider:
    """Mix advantage and discriminator logit into a single G tensor.

    Args:
        iql_learner:           the IQL learner; provides advantage.
        discriminator:         online BCE; provides disc_logit.
        encoder:               shared frozen encoder; used to build context
                               from `DipoleBatch.image_obs_raw` + `proprio_raw`.
        alpha, beta:           linear mixing coefficients.
        advantage_normalization, disc_normalization:
                               "batch_zscore" | "running_zscore" | "minmax" | "none".
                               Matches DipoleConfig.g_normalization options.
    """

    def __init__(
        self,
        *,
        iql_learner: "IQLLearner",
        discriminator: "OnlineBCEDiscriminator",
        encoder: "SharedFrozenEncoder",
        alpha: float,
        beta: float,
        advantage_normalization: str = "batch_zscore",
        disc_normalization: str = "batch_zscore",
    ) -> None:
        self.iql_learner = iql_learner
        self.discriminator = discriminator
        self.encoder = encoder
        self.alpha = alpha
        self.beta = beta
        self.advantage_normalization = advantage_normalization
        self.disc_normalization = disc_normalization

    # ------------------------------------------------------------------ #
    # G provider contract (matches LPBV2GProvider)                        #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def compute_g_for_batch(self, batch: Any) -> torch.Tensor:
        """Args:
            batch: `DipoleBatch`. We use `image_obs_raw`, `proprio_raw`,
                   `action_sequences_raw` to build the IQL/disc inputs.
        Returns:
            (B,) tensor of G values on the same device as the batch.
        """
        raise NotImplementedError

    @torch.no_grad()
    def compute_g_for_observation(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Single-observation variant for eval-time scoring (mirrors
        `LPBV2GProvider.compute_g_for_observation`). Optional — implement
        only if eval_dipole.py needs the same code path."""
        raise NotImplementedError

    def bind_policy_cameras(self, policy_cameras: list[str]) -> None:
        """Pass-through to the encoder (so the existing wiring in
        train_dipole.py:689-696 keeps working)."""
        self.encoder.bind_policy_cameras(policy_cameras)
