"""Dataclasses shared across the VAST value-stitching module.

Interface contract (see the DIPOLE-RL section in ``pipeline/README.md``):
- `VASTStepBatch` is the unit consumed by the joint G/V update; rewards are
  already aggregated as n-step discounted returns over the action chunk and
  include the discriminator intrinsic term r_disc. `chunk_feature` is kept
  because the frozen nnPU discriminator scores failure on the chunk feature.
- `VASTActorBatch` carries the state/chunk features used to produce the TD
  advantage signal that `AdvantageGProvider` mixes with the discriminator
  logit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class VASTConfig:
    """Configuration for the VAST learner.

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
    # Optimism knob for the expectile-TD value fit (L_V = expectile_tau(target
    # - V_k)). tau > 0.5 pulls V up toward the best locally-reachable
    # continuation, turning V^beta toward the optimistic reachable value.
    expectile_tau: float = 0.9

    # VAST value-stitching. Horizons are counted in action chunks, so k=1
    # spans ``action_horizon`` environment transitions and discounts by
    # gamma ** action_horizon. The sampling seed owns only k/j sampling; the
    # replay start-index RNG remains unchanged for reproducibility.
    vast_v_mode: str = "single_vast"  # single_vast | ensemble_lcb
    vast_max_k: int = 10
    vast_comp_coef: float = 0.5
    vast_sampling_seed: int = 0
    g_lr: float = 3e-4

    # V-ensemble soft-LCB (deadly-triad guardrail): V_lcb = mean_k V_k - beta *
    # std_k V_k over ``v_ensemble_size`` independent heads. Soft LCB (not hard
    # min) so the recoverable-state lift in the high-disagreement region is
    # kept, not suppressed. Heads are diversified by a per-head Bernoulli
    # bootstrap mask (keep-prob ``ensemble_bootstrap_prob``) so the std does not
    # collapse. v_ensemble_size == 1 recovers the single-head V (std == 0, beta
    # and bootstrap mask inert).
    v_ensemble_size: int = 2
    ensemble_lcb_beta: float = 0.5
    ensemble_bootstrap_prob: float = 0.5

    v_lr: float = 3e-4
    target_polyak: float = 0.005

    n_step_aggregate: bool = True
    # Head width *after* the Token/Group projector (the projector — not the
    # head — is the anti-overfit lever, so this is now much smaller than the
    # previous 512x512 head that fed on the raw 12288-d feature).
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
        self.vast_v_mode = str(self.vast_v_mode).strip().lower()
        if self.vast_v_mode not in {"single_vast", "ensemble_lcb"}:
            raise ValueError(
                "VASTConfig.vast_v_mode must be 'single_vast' or 'ensemble_lcb', "
                f"got {self.vast_v_mode!r}."
            )
        self.action_horizon = int(self.action_horizon)
        if self.action_horizon < 1:
            raise ValueError(
                f"VASTConfig.action_horizon must be >= 1, got {self.action_horizon}."
            )
        self.vast_max_k = int(self.vast_max_k)
        if self.vast_max_k < 1:
            raise ValueError(f"VASTConfig.vast_max_k must be >= 1, got {self.vast_max_k}.")
        self.vast_comp_coef = float(self.vast_comp_coef)
        if self.vast_comp_coef < 0.0:
            raise ValueError(
                f"VASTConfig.vast_comp_coef must be >= 0, got {self.vast_comp_coef}."
            )
        self.vast_sampling_seed = int(self.vast_sampling_seed)
        self.g_lr = float(self.g_lr)
        if self.g_lr <= 0.0:
            raise ValueError(f"VASTConfig.g_lr must be > 0, got {self.g_lr}.")
        self.hidden_dims = tuple(int(h) for h in self.hidden_dims)
        self.state_proj_dim = int(self.state_proj_dim)
        self.proprio_proj_dim = int(self.proprio_proj_dim)
        for name in ("state_proj_dim", "proprio_proj_dim"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"VASTConfig.{name} must be >= 1, got {getattr(self, name)}.")
        self.v_ensemble_size = int(self.v_ensemble_size)
        if self.v_ensemble_size < 1:
            raise ValueError(
                f"VASTConfig.v_ensemble_size must be >= 1, got {self.v_ensemble_size}."
            )
        self.ensemble_lcb_beta = float(self.ensemble_lcb_beta)
        if self.ensemble_lcb_beta < 0.0:
            raise ValueError(
                f"VASTConfig.ensemble_lcb_beta must be >= 0, got {self.ensemble_lcb_beta}."
            )
        self.ensemble_bootstrap_prob = float(self.ensemble_bootstrap_prob)
        if not 0.0 < self.ensemble_bootstrap_prob <= 1.0:
            raise ValueError(
                "VASTConfig.ensemble_bootstrap_prob must be in (0, 1], got "
                f"{self.ensemble_bootstrap_prob}."
            )
        self.expectile_tau = float(self.expectile_tau)
        if not 0.0 < self.expectile_tau < 1.0:
            raise ValueError(
                f"VASTConfig.expectile_tau must be in (0, 1), got {self.expectile_tau}."
            )


@dataclass
class VASTStepBatch:
    """Transition-centric batch for joint G/V updates.

    Shapes:
        chunk_feature (B, D_chunk) — action-conditioned chunk latent for nnPU
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

    chunk_feature: torch.Tensor
    v_state_feature: torch.Tensor
    next_v_state_feature: torch.Tensor
    action_chunk: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    is_online: torch.Tensor
    is_intervention: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)
    # VAST path fields remain optional so pre-encoded caches can be inspected
    # before path sampling is requested.
    future_v_state_feature: torch.Tensor | None = None
    intermediate_v_state_feature: torch.Tensor | None = None
    k: torch.Tensor | None = None
    j: torch.Tensor | None = None
    k_step_returns: torch.Tensor | None = None
    mc_mask: torch.Tensor | None = None
    future_dones: torch.Tensor | None = None

    def to(self, device: str | torch.device) -> "VASTStepBatch":
        return VASTStepBatch(
            chunk_feature=self.chunk_feature.to(device),
            v_state_feature=self.v_state_feature.to(device),
            next_v_state_feature=self.next_v_state_feature.to(device),
            action_chunk=self.action_chunk.to(device),
            rewards=self.rewards.to(device),
            dones=self.dones.to(device),
            is_online=self.is_online.to(device),
            is_intervention=self.is_intervention.to(device),
            metadata=self.metadata,
            future_v_state_feature=(
                None
                if self.future_v_state_feature is None
                else self.future_v_state_feature.to(device)
            ),
            intermediate_v_state_feature=(
                None
                if self.intermediate_v_state_feature is None
                else self.intermediate_v_state_feature.to(device)
            ),
            k=None if self.k is None else self.k.to(device),
            j=None if self.j is None else self.j.to(device),
            k_step_returns=(
                None if self.k_step_returns is None else self.k_step_returns.to(device)
            ),
            mc_mask=None if self.mc_mask is None else self.mc_mask.to(device),
            future_dones=(
                None if self.future_dones is None else self.future_dones.to(device)
            ),
        )


@dataclass
class VASTActorBatch:
    """Actor-side batch consumed by AdvantageGProvider.

    Shapes:
        chunk_feature   (B, D_chunk) — action-conditioned chunk latent for nnPU
        v_state_feature   (B, D_state) — action-free state latent for V
        action_chunk      (B, H, D_a)  — raw chunk retained for diagnostics
        metadata          dict         — episode ids, mask flags
    """

    chunk_feature: torch.Tensor
    v_state_feature: torch.Tensor
    action_chunk: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    def to(self, device: str | torch.device) -> "VASTActorBatch":
        return VASTActorBatch(
            chunk_feature=self.chunk_feature.to(device),
            v_state_feature=self.v_state_feature.to(device),
            action_chunk=self.action_chunk.to(device),
            metadata=self.metadata,
        )
