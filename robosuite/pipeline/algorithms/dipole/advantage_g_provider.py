"""G provider that mixes Q-V advantage with frozen nnPU failure scores.

Replaces `NNPUGProvider` when DipoleConfig.g_mode == "advantage".
Implements the exact same `compute_g_for_batch(batch) -> (B,)` contract so
`DipoleFlowPolicy.update()` is unchanged.

Math:
    A(s, a)        = mean(Q_1(s, a), ..., Q_K(s, a)) - V(s)
    failure        = FrozenNNPUDiscriminator.failure_score(z(s, a))
    G              = alpha * A - beta * failure

The flow policy then maps G -> w_pos = sigmoid(beta_policy * G + k), so a
larger advantage raises the positive-branch weight.

This object owns nothing it doesn't construct: the IQL learner, the
discriminator, and the encoder are all injected.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from robosuite.pipeline.algorithms.q_learning.common import IQLActorBatch

if TYPE_CHECKING:
    from robosuite.pipeline.algorithms.discriminator.encoder import SharedDynamicsEncoder
    from robosuite.pipeline.algorithms.discriminator.nnpu import FrozenNNPUDiscriminator
    from robosuite.pipeline.algorithms.q_learning.iql import IQLLearner


class AdvantageGProvider:
    """Mix advantage and discriminator logit into a single G tensor.

    Args:
        iql_learner:           the IQL learner; provides advantage.
        discriminator:         frozen nnPU head; provides failure score.
        encoder:               shared frozen encoder; used to build features
                               from `DipoleBatch.image_obs_raw` + `proprio_raw`.
        alpha, beta:           linear mixing coefficients.
    """

    def __init__(
        self,
        *,
        iql_learner: "IQLLearner",
        discriminator: "FrozenNNPUDiscriminator",
        encoder: "SharedDynamicsEncoder",
        alpha: float,
        beta: float,
    ) -> None:
        self.iql_learner = iql_learner
        self.discriminator = discriminator
        self.encoder = encoder
        self.alpha = float(alpha)
        self.beta = float(beta)

    # ------------------------------------------------------------------ #
    # G provider contract                                                  #
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
        action_chunk_raw = batch.action_sequences_raw.to(
            device=torch.device(self.encoder.device), dtype=torch.float32
        )
        state_feature, chunk_feature = self.encoder.encode_state_and_chunk(
            image_obs_raw=batch.image_obs_raw,
            proprio_raw=batch.proprio_raw,
            action_chunk=action_chunk_raw,
        )

        actor_batch = IQLActorBatch(
            q_chunk_feature=chunk_feature,
            v_state_feature=state_feature,
            action_chunk=action_chunk_raw,
            metadata={},
        )

        advantage = self.iql_learner.compute_advantage_for_batch(actor_batch)
        advantage = advantage.reshape(-1)

        failure_score = self.discriminator.failure_score(
            chunk_feature=chunk_feature
        ).reshape(-1)

        g = self.alpha * advantage - self.beta * failure_score
        return g.to(batch.action_sequences_raw.device).reshape(-1)

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
