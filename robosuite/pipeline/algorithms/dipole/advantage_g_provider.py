"""G provider that mixes the VAST TD1 fallback with frozen nnPU rewards.

Replaces `NNPUGProvider` when DipoleConfig.g_mode == "advantage".

Online fallback math (no Q head):
    A(s, a)        = r + gamma^H * target_V(s') - V(s)      # TD residual
    r_disc         = FrozenNNPUDiscriminator.intrinsic_reward(z(s, a))
    G              = alpha * A + disc_weight * r_disc

The flow policy then maps G -> w_pos = sigmoid(beta_policy * (G + k)), so a
larger advantage raises the positive-branch weight.

STATUS: the online ``compute_g_for_batch`` is a stub. Computing the TD residual
requires the next state ``s'`` and the chunk reward ``r`` for each sampled
window, which the online :class:`DipoleBatch` does not currently carry (the
offline path precomputes the residual by start index instead — see
``offline/utils/advantage.py``, which can accumulate GAE from those residuals).
Wiring next-state/reward through the online
sampler is deferred; until then this provider raises ``NotImplementedError``.

This object owns nothing it doesn't construct: the VAST learner, the
discriminator, and the encoder are all injected.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from robosuite.pipeline.algorithms.discriminator.encoder import SharedDynamicsEncoder
    from robosuite.pipeline.algorithms.discriminator.nnpu import FrozenNNPUDiscriminator
    from robosuite.pipeline.algorithms.vast.vast import VASTLearner


class AdvantageGProvider:
    """Mix advantage and signed discriminator reward into one G tensor.

    Args:
        vast_learner:           the VAST learner; provides advantage.
        discriminator:         frozen nnPU head; provides intrinsic reward.
        encoder:               shared frozen encoder; used to build features
                               from `DipoleBatch.image_obs_raw` + `proprio_raw`.
        alpha, disc_weight:    linear mixing coefficients.
    """

    def __init__(
        self,
        *,
        vast_learner: "VASTLearner",
        discriminator: "FrozenNNPUDiscriminator",
        encoder: "SharedDynamicsEncoder",
        alpha: float,
        disc_weight: float,
    ) -> None:
        self.vast_learner = vast_learner
        self.discriminator = discriminator
        self.encoder = encoder
        self.alpha = float(alpha)
        self.disc_weight = float(disc_weight)

    # ------------------------------------------------------------------ #
    # G provider contract                                                  #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def compute_g_for_batch(self, batch: Any) -> torch.Tensor:
        """Online TD-advantage G — NOT YET WIRED.

        The online fallback advantage is the TD1 residual
        ``r + gamma^H * target_V(s') - V(s)``, which needs the next state
        ``s'`` and chunk reward ``r`` for each window. The online
        :class:`DipoleBatch` does not carry those (only ``s`` and the action
        chunk). Until next-state/reward is threaded through the online sampler,
        this method raises. The offline DIPOLE path uses
        :class:`OfflineAdvantageGProvider` (precomputed TD residual) instead.
        """
        raise NotImplementedError(
            "Online AdvantageGProvider.compute_g_for_batch is not implemented for "
            "the VAST TD1 fallback: the online DipoleBatch carries no next state "
            "or chunk reward. Use the offline OfflineAdvantageGProvider (precomputed "
            "TD1/GAE advantage), or thread next_obs/reward through the online sampler "
            "before enabling g_mode='advantage' online."
        )

    @torch.no_grad()
    def compute_g_for_observation(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Single-observation variant. Not supported by AdvantageGProvider:
        eval/UI scoring uses the background nnPU runtime instead."""
        raise NotImplementedError(
            "AdvantageGProvider does not support single-observation scoring; "
            "use compute_g_for_batch in training and the nnPU runtime for UI scoring."
        )

    def bind_policy_cameras(self, policy_cameras: list[str]) -> None:
        """Pass-through to the encoder (so the existing wiring in
        train_dipole.py:689-696 keeps working)."""
        self.encoder.bind_policy_cameras(policy_cameras)


__all__ = ["AdvantageGProvider"]
