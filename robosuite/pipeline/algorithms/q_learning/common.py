"""Dataclasses shared across the ResNet-50 IQL Q-chunking module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class IQLConfig:
    """Configuration for the IQL learner."""

    action_horizon: int = 8
    discount: float = 0.99
    expectile_tau: float = 0.7

    q_lr: float = 3e-4
    v_lr: float = 3e-4
    target_polyak: float = 0.005

    n_step_aggregate: bool = True
    hidden_dims: tuple[int, ...] = (512, 512)
    state_feature_dim: int = 256
    action_feature_dim: int = 256
    resnet_pretrained_path: str = "data/pretrained/resnet50.pth"
    grad_clip_norm: float = 1.0
    weight_decay: float = 1e-6
    device: str = "cuda:1"

    # Reward composition (r_total = r_env * output_reward_coef + disc_reward_coef * r_disc).
    reward_mode: str = "-1/0"   # "0/1" | "-1/0"
    output_reward_coef: float = 1.0
    disc_reward_coef: float = 1.0
    # Gradient steps per learner tick inside DipoleTrainer.train_step (each resamples).
    update_freq: int = 1


@dataclass
class IQLStepBatch:
    """Transition-centric raw-observation batch for Q/V updates."""

    image_obs_raw: torch.Tensor
    proprio_raw: torch.Tensor
    next_image_obs_raw: torch.Tensor
    next_proprio_raw: torch.Tensor
    action_chunk: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    is_online: torch.Tensor
    is_intervention: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    def to(self, device: str | torch.device) -> "IQLStepBatch":
        return IQLStepBatch(
            image_obs_raw=self.image_obs_raw.to(device),
            proprio_raw=self.proprio_raw.to(device),
            next_image_obs_raw=self.next_image_obs_raw.to(device),
            next_proprio_raw=self.next_proprio_raw.to(device),
            action_chunk=self.action_chunk.to(device),
            rewards=self.rewards.to(device),
            dones=self.dones.to(device),
            is_online=self.is_online.to(device),
            is_intervention=self.is_intervention.to(device),
            metadata=self.metadata,
        )


@dataclass
class IQLActorBatch:
    """Actor-side raw-observation batch consumed by ``AdvantageGProvider``."""

    image_obs_raw: torch.Tensor
    proprio_raw: torch.Tensor
    action_chunk_raw: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    def to(self, device: str | torch.device) -> "IQLActorBatch":
        return IQLActorBatch(
            image_obs_raw=self.image_obs_raw.to(device),
            proprio_raw=self.proprio_raw.to(device),
            action_chunk_raw=self.action_chunk_raw.to(device),
            metadata=self.metadata,
        )
