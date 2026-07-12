"""Dataclasses shared across the V-only IQL module.

Interface contract (see the DIPOLE-RL section in ``pipeline/README.md``):
- `IQLStepBatch` is the unit consumed by the V-only TD backup; rewards are
  already aggregated as n-step discounted returns over the action chunk and
  include the discriminator intrinsic term r_disc. `q_chunk_feature` is kept
  because the frozen nnPU discriminator scores failure on the chunk feature.
- `IQLActorBatch` carries the state/chunk features used to produce the TD
  advantage signal that `AdvantageGProvider` mixes with the discriminator
  logit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class IQLConfig:
    """Configuration for the IQL learner.

    Notes:
        action_horizon must match DipoleConfig.action_horizon (the chunk
        bootstrap discounts by gamma^H over the executed chunk).
        r_disc is sourced from ``FrozenNNPUDiscriminator.intrinsic_reward``:
        ``-sigmoid(failure_score - tau)`` with ``failure_score = -head(z)``
        (nnPU convention). Per-frame values lie in (-1, 0); the replay
        γ-aggregates them with r_env via ``aggregate_chunk_reward``.
    """

    action_horizon: int = 8
    discount: float = 0.99
    # RESERVED optimism knob for the planned expectile-TD value fit
    # (L_V = expectile_tau(target - V)); the current path trains V by plain
    # MSE-TD and does not read this. See V_ONLY_ADVANTAGE_DESIGN.md.
    expectile_tau: float = 0.7

    v_lr: float = 3e-4
    target_polyak: float = 0.005

    n_step_aggregate: bool = True
    # Head width *after* the Token/Group projector (the projector — not the
    # head — is the anti-overfit lever, so this is now much smaller than the
    # legacy 512x512 that fed on the raw 12288-d feature).
    hidden_dims: tuple[int, ...] = (256, 256)
    # Token/Group dim-reduction projector inserted BEFORE the V head
    # (see networks.py). The frozen encoder state feature is 16 tokens x
    # per-token dim; state proj dim must be divisible by n_tokens (asserted in
    # the learner once n_tokens is known from the encoder).
    state_proj_dim: int = 256      # 12288 state visual tokens -> compressed V input
    proprio_proj_dim: int = 32     # state proprio block -> appended to V input
    proj_activation: str = "mish"  # mish | gelu | relu | silu
    grad_clip_norm: float = 1.0
    weight_decay: float = 1e-6
    device: str = "cuda:1"

    # Reward composition (r_total = r_env * output_reward_coef + disc_reward_coef * r_disc).
    reward_mode: str = "-1/0"   # "0/1" | "-1/0"
    output_reward_coef: float = 1.0
    disc_reward_coef: float = 1.0
    # Gradient steps per learner tick inside DipoleTrainer.train_step (each resamples).
    update_freq: int = 1

    def __post_init__(self) -> None:
        self.hidden_dims = tuple(int(h) for h in self.hidden_dims)
        self.state_proj_dim = int(self.state_proj_dim)
        self.proprio_proj_dim = int(self.proprio_proj_dim)
        for name in ("state_proj_dim", "proprio_proj_dim"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"IQLConfig.{name} must be >= 1, got {getattr(self, name)}.")


@dataclass
class IQLStepBatch:
    """Transition-centric batch for Q/V updates.

    Shapes:
        q_chunk_feature (B, D_chunk) — action-conditioned chunk latent for Q
        v_state_feature (B, D_state) — action-free state latent for V(s_t)
        next_v_state_feature
                        (B, D_state) — action-free state latent for V(s')
        action_chunk    (B, H, D_a) — raw chunk retained for metadata/debug
        rewards        (B, 1)       — n-step aggregated r_env + λ_disc·r_disc
        dones          (B, 1)       — 1 if any step in chunk terminated
        is_online      (B, 1)       — 1 if sampled from online buffer
        is_intervention(B, 1)       — 1 if first step is an intervention
        metadata       dict         — debug fields (episode ids, sources)
    """

    q_chunk_feature: torch.Tensor
    v_state_feature: torch.Tensor
    next_v_state_feature: torch.Tensor
    action_chunk: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    is_online: torch.Tensor
    is_intervention: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    def to(self, device: str | torch.device) -> "IQLStepBatch":
        return IQLStepBatch(
            q_chunk_feature=self.q_chunk_feature.to(device),
            v_state_feature=self.v_state_feature.to(device),
            next_v_state_feature=self.next_v_state_feature.to(device),
            action_chunk=self.action_chunk.to(device),
            rewards=self.rewards.to(device),
            dones=self.dones.to(device),
            is_online=self.is_online.to(device),
            is_intervention=self.is_intervention.to(device),
            metadata=self.metadata,
        )


@dataclass
class IQLActorBatch:
    """Actor-side batch consumed by AdvantageGProvider.

    Shapes:
        q_chunk_feature   (B, D_chunk) — action-conditioned chunk latent for Q
        v_state_feature   (B, D_state) — action-free state latent for V
        action_chunk      (B, H, D_a)  — raw action chunk re-injected into Q
        metadata          dict         — episode ids, mask flags
    """

    q_chunk_feature: torch.Tensor
    v_state_feature: torch.Tensor
    action_chunk: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    def to(self, device: str | torch.device) -> "IQLActorBatch":
        return IQLActorBatch(
            q_chunk_feature=self.q_chunk_feature.to(device),
            v_state_feature=self.v_state_feature.to(device),
            action_chunk=self.action_chunk.to(device),
            metadata=self.metadata,
        )
