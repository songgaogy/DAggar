"""G provider that mixes Q-V advantage with the online BCE logit.

Replaces `LPBV2GProvider` when DipoleConfig.g_mode == "advantage".
Implements the exact same `compute_g_for_batch(batch) -> (B,)` contract so
`DipoleFlowPolicy.update()` is unchanged.

Math:
    A(s, a)        = min(Q1(s, a), Q2(s, a)) - V(s)
    A_norm         = normalize(A,         mode=advantage_normalization)
    disc_logit     = OnlineBCEDiscriminator.score(s, a).logit
    disc_norm      = normalize(disc_logit, mode=disc_normalization)
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

from robosuite.pipeline.algorithms.q_learning.common import IQLActorBatch

if TYPE_CHECKING:
    from robosuite.pipeline.algorithms.discriminator.encoder import SharedFrozenEncoder
    from robosuite.pipeline.algorithms.discriminator.online_bce import (
        OnlineBCEDiscriminator,
    )
    from robosuite.pipeline.algorithms.q_learning.iql import IQLLearner


_NORMALIZATION_MODES = ("none", "batch_zscore", "running_zscore", "minmax")


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
        for mode in (advantage_normalization, disc_normalization):
            if str(mode).lower() not in _NORMALIZATION_MODES:
                raise ValueError(
                    f"Unknown normalization mode: {mode!r}. "
                    f"Expected one of {_NORMALIZATION_MODES}."
                )
        self.iql_learner = iql_learner
        self.discriminator = discriminator
        self.encoder = encoder
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.advantage_normalization = str(advantage_normalization).lower()
        self.disc_normalization = str(disc_normalization).lower()

        self._running_momentum = 0.1
        self._adv_running_mean = 0.0
        self._adv_running_var = 1.0
        self._adv_running_count = 0
        self._disc_running_mean = 0.0
        self._disc_running_var = 1.0
        self._disc_running_count = 0

    # ------------------------------------------------------------------ #
    # G provider contract (matches LPBV2GProvider)                        #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def compute_g_for_batch(self, batch: Any) -> torch.Tensor:
        """Args:
            batch: `DipoleBatch`. We use `image_obs_raw`, `proprio_raw`,
                   `action_sequences_raw` to build the IQL/disc inputs.
        Returns:
            (B,) tensor of G values on the same device as
            `batch.action_sequences_raw`.
        """
        context = self.encoder.encode(
            image_obs_raw=batch.image_obs_raw,
            proprio_raw=batch.proprio_raw,
        )

        action_chunk_raw = batch.action_sequences_raw.to(
            device=context.device, dtype=torch.float32
        )

        actor_batch = IQLActorBatch(
            context=context,
            action_chunk_raw=action_chunk_raw,
            metadata={},
        )

        advantage = self.iql_learner.compute_advantage_for_batch(actor_batch)
        advantage = advantage.reshape(-1)

        disc_out = self.discriminator.score(
            context=context, action_chunk=action_chunk_raw
        )
        disc_logit = disc_out.logit.reshape(-1)

        a_norm = self._normalize(advantage, mode=self.advantage_normalization, stats="advantage")
        d_norm = self._normalize(disc_logit, mode=self.disc_normalization, stats="disc")

        g = self.alpha * a_norm + self.beta * (-d_norm)
        return g.to(batch.action_sequences_raw.device).reshape(-1)

    @torch.no_grad()
    def compute_g_for_observation(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Single-observation variant. Not supported by AdvantageGProvider:
        eval/UI scoring uses the frozen `LPBV2GProvider` path instead."""
        raise NotImplementedError(
            "AdvantageGProvider does not support single-observation scoring; "
            "use compute_g_for_batch in training, and the frozen "
            "LPBV2GProvider path for eval-time UI scoring."
        )

    def bind_policy_cameras(self, policy_cameras: list[str]) -> None:
        """Pass-through to the encoder (so the existing wiring in
        train_dipole.py:689-696 keeps working)."""
        self.encoder.bind_policy_cameras(policy_cameras)

    # ------------------------------------------------------------------ #
    # Normalization (mirrors DipoleFlowPolicy._normalize_g)               #
    # ------------------------------------------------------------------ #

    def _normalize(self, x: torch.Tensor, *, mode: str, stats: str) -> torch.Tensor:
        """Normalize the (B,) tensor. Mirrors `DipoleFlowPolicy._normalize_g`
        but routes running-stats to a per-channel buffer (advantage vs disc).
        """
        mode_l = str(mode).lower()
        if mode_l == "none":
            return x
        if mode_l == "batch_zscore":
            return (x - x.mean()) / (x.std() + 1e-6)
        if mode_l == "running_zscore":
            return self._running_zscore(x, stats=stats)
        if mode_l == "minmax":
            lo = x.min()
            hi = x.max()
            span = (hi - lo).clamp_min(1e-6)
            return 2.0 * (x - lo) / span - 1.0
        raise ValueError(f"Unknown normalization mode: {mode!r}")

    def _running_zscore(self, x: torch.Tensor, *, stats: str) -> torch.Tensor:
        if stats == "advantage":
            mean_attr, var_attr, count_attr = (
                "_adv_running_mean", "_adv_running_var", "_adv_running_count",
            )
        elif stats == "disc":
            mean_attr, var_attr, count_attr = (
                "_disc_running_mean", "_disc_running_var", "_disc_running_count",
            )
        else:
            raise ValueError(f"Unknown running-stats channel: {stats!r}")

        batch_mean = float(x.mean().item())
        batch_var = float(x.var(unbiased=False).item())
        count = int(x.numel())

        cur_count = int(getattr(self, count_attr))
        if cur_count == 0:
            setattr(self, mean_attr, batch_mean)
            setattr(self, var_attr, batch_var if batch_var > 0 else 1.0)
        else:
            momentum = self._running_momentum
            new_mean = (1.0 - momentum) * float(getattr(self, mean_attr)) + momentum * batch_mean
            new_var = (1.0 - momentum) * float(getattr(self, var_attr)) + momentum * batch_var
            setattr(self, mean_attr, new_mean)
            setattr(self, var_attr, new_var)
        setattr(self, count_attr, cur_count + count)

        mean_t = torch.tensor(getattr(self, mean_attr), dtype=x.dtype, device=x.device)
        std_t = torch.tensor(getattr(self, var_attr), dtype=x.dtype, device=x.device).clamp_min(1e-6).sqrt()
        return (x - mean_t) / std_t


__all__ = ["AdvantageGProvider"]
