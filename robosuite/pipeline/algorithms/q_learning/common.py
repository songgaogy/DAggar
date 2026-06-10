"""Dataclasses shared across the IQL Q-chunking module.

Interface contract (see docs/DIPOLE_RL.md §A):
- `IQLStepBatch` is the unit consumed by Bellman + expectile losses; rewards
  are already aggregated as n-step discounted returns over the action chunk
  and include the discriminator intrinsic term r_disc.
- `IQLActorBatch` is the unit consumed by `compute_advantage_for_batch` to
  produce the advantage signal that `AdvantageGProvider` mixes with the
  discriminator logit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class IQLConfig:
    """Configuration for the IQL learner.

    Notes:
        action_horizon must match DipoleConfig.action_horizon (Q-chunking
        assumes the critic sees the full execution chunk).
        r_disc is sourced from ``OnlineBCEDiscriminator.intrinsic_reward``:
        ``-sigmoid(failure_score - tau)`` with ``failure_score = -head(z)``
        (LPB convention). Per-frame values lie in (-1, 0); the replay
        γ-aggregates them with r_env via ``aggregate_chunk_reward``.
    """

    action_horizon: int = 8
    discount: float = 0.99
    expectile_tau: float = 0.7

    # Critic-target shaping for offline warmup.
    # mc_blend_lambda: y = (1 - lambda) * TD + lambda * MC_return_to_go.
    # terminal_undiscounted_reward: terminal chunk reward = undiscounted env
    #   reward sum (=1 for a 0/1 success chunk) instead of the in-chunk
    #   gamma^k discounted value, so terminal Q targets ~1.
    # terminal_loss_weight: per-sample loss weight on terminal (done==1) chunks.
    mc_blend_lambda: float = 0.5
    terminal_undiscounted_reward: bool = True
    terminal_loss_weight: float = 10.0

    q_lr: float = 3e-4
    v_lr: float = 3e-4
    target_polyak: float = 0.005

    n_step_aggregate: bool = True
    hidden_dims: tuple[int, ...] = (512, 512)
    q_ensemble_size: int = 5
    v_subset_size: int = 2
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
        self.q_ensemble_size = int(self.q_ensemble_size)
        self.v_subset_size = int(self.v_subset_size)
        if self.q_ensemble_size < 1:
            raise ValueError(
                f"IQLConfig.q_ensemble_size must be >= 1, got {self.q_ensemble_size}."
            )
        if self.v_subset_size < 1 or self.v_subset_size > self.q_ensemble_size:
            raise ValueError(
                "IQLConfig.v_subset_size must satisfy "
                f"1 <= v_subset_size <= q_ensemble_size; got "
                f"{self.v_subset_size} and {self.q_ensemble_size}."
            )
        self.mc_blend_lambda = float(self.mc_blend_lambda)
        if not (0.0 <= self.mc_blend_lambda <= 1.0):
            raise ValueError(
                f"IQLConfig.mc_blend_lambda must be in [0, 1]; got {self.mc_blend_lambda}."
            )
        self.terminal_loss_weight = float(self.terminal_loss_weight)
        if self.terminal_loss_weight < 1.0:
            raise ValueError(
                f"IQLConfig.terminal_loss_weight must be >= 1; got {self.terminal_loss_weight}."
            )


@dataclass
class IQLStepBatch:
    """Transition-centric batch for Q/V updates.

    Shapes:
        context        (B, D_ctx)   — frozen latent at chunk start s_t
                                      (Q/V Bellman state; disc also scores
                                      all H frames inside the replay sampler)
        next_context   (B, D_ctx)   — frozen encoder latent for s'
        action_chunk   (B, H, D_a)  — normalized chunk a_{t:t+H}
        rewards        (B, 1)       — n-step aggregated r_env + λ_disc·r_disc
        dones          (B, 1)       — 1 if any step in chunk terminated
        is_online      (B, 1)       — 1 if sampled from online buffer
        is_intervention(B, 1)       — 1 if first step is an intervention
        mc_return      (B, 1) | None— MC return-to-go target (terminal chunk = 1,
                                      gamma^(t_term-t) ramp); None => pure TD.
        metadata       dict         — debug fields (episode ids, sources)
    """

    context: torch.Tensor
    next_context: torch.Tensor
    action_chunk: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    is_online: torch.Tensor
    is_intervention: torch.Tensor
    mc_return: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to(self, device: str | torch.device) -> "IQLStepBatch":
        return IQLStepBatch(
            context=self.context.to(device),
            next_context=self.next_context.to(device),
            action_chunk=self.action_chunk.to(device),
            rewards=self.rewards.to(device),
            dones=self.dones.to(device),
            is_online=self.is_online.to(device),
            is_intervention=self.is_intervention.to(device),
            mc_return=None if self.mc_return is None else self.mc_return.to(device),
            metadata=self.metadata,
        )


@dataclass
class IQLActorBatch:
    """Actor-side batch consumed by AdvantageGProvider.

    Shapes:
        context           (B, D_ctx)   — frozen encoder latent for s
        action_chunk_raw  (B, H, D_a)  — UN-normalized chunk used to score Q
        metadata          dict         — episode ids, mask flags
    """

    context: torch.Tensor
    action_chunk_raw: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    def to(self, device: str | torch.device) -> "IQLActorBatch":
        return IQLActorBatch(
            context=self.context.to(device),
            action_chunk_raw=self.action_chunk_raw.to(device),
            metadata=self.metadata,
        )
