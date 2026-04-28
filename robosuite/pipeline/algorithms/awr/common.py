from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import torch

from robosuite.pipeline.common.types import (
    EncoderConfig,
    ReplayBufferConfig,
    Transition,
)


@dataclass
class FlowAugmentationConfig:
    minimal_shift_pad: int = 2
    eye_in_hand_crop_scale: float = 0.88


@dataclass
class AWRConfig:
    action_dim: int
    proprio_dim: int
    action_horizon: int = 8
    execute_horizon: int = 8
    image_size: int = 128
    actor_learning_rate: float = 1e-4
    critic_learning_rate: float = 3e-4
    weight_decay: float = 1e-6
    grad_clip_norm: float = 1.0
    lambda_endpoint: float = 0.5
    lambda_smooth: float = 0.05
    beta: float = 3.0
    max_adv_weight: float = 100.0
    discount: float = 0.99
    expectile: float = 0.7
    critic_hidden_dims: Sequence[int] = field(default_factory=lambda: (512, 512))
    margin_scale_floor: float = 0.1
    success_reward_scale: float = 1.0
    discriminator_reward_scale: float = 1.0
    discriminator_reward_clip: float = 5.0
    n_ode_steps: int = 8
    reward_from_discriminator: bool = True
    device: str = "cpu"
    inference_device: Optional[str] = None
    task_name: Optional[str] = None
    language_instruction: Optional[str] = None
    augmentation: FlowAugmentationConfig = field(default_factory=FlowAugmentationConfig)


@dataclass
class TrainerConfig:
    batch_size: int = 64
    warmup_steps: int = 0
    updates_per_step: int = 1
    steps_per_update: int = 50
    random_steps: int = 0
    value_warmup_steps: int = 20_000


@dataclass
class AWRActorBatch:
    image_obs: torch.Tensor
    proprio: torch.Tensor
    action_sequences: torch.Tensor
    raw_action_sequences: torch.Tensor
    first_actions: torch.Tensor
    is_online: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    def to(self, device: torch.device | str) -> "AWRActorBatch":
        return AWRActorBatch(
            image_obs=self.image_obs.to(device),
            proprio=self.proprio.to(device),
            action_sequences=self.action_sequences.to(device),
            raw_action_sequences=self.raw_action_sequences.to(device),
            first_actions=self.first_actions.to(device),
            is_online=self.is_online.to(device),
            metadata=self.metadata,
        )

    @property
    def batch_size(self) -> int:
        return int(self.action_sequences.shape[0])

    @staticmethod
    def concat(batches: list["AWRActorBatch"]) -> "AWRActorBatch":
        valid_batches = [batch for batch in batches if batch.batch_size > 0]
        if len(valid_batches) == 0:
            raise ValueError("AWRActorBatch.concat requires at least one non-empty batch.")
        if len(valid_batches) == 1:
            return valid_batches[0]

        metadata: dict[str, Any] = {}
        for batch in valid_batches:
            for key, value in batch.metadata.items():
                metadata.setdefault(key, [])
                if isinstance(value, list):
                    metadata[key].extend(value)
                else:
                    metadata[key].append(value)
        return AWRActorBatch(
            image_obs=torch.cat([batch.image_obs for batch in valid_batches], dim=0),
            proprio=torch.cat([batch.proprio for batch in valid_batches], dim=0),
            action_sequences=torch.cat([batch.action_sequences for batch in valid_batches], dim=0),
            raw_action_sequences=torch.cat([batch.raw_action_sequences for batch in valid_batches], dim=0),
            first_actions=torch.cat([batch.first_actions for batch in valid_batches], dim=0),
            is_online=torch.cat([batch.is_online for batch in valid_batches], dim=0),
            metadata=metadata,
        )


@dataclass
class AWRStepBatch:
    image_obs: torch.Tensor
    proprio: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_image_obs: torch.Tensor
    next_proprio: torch.Tensor
    dones: torch.Tensor
    is_online: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    def to(self, device: torch.device | str) -> "AWRStepBatch":
        return AWRStepBatch(
            image_obs=self.image_obs.to(device),
            proprio=self.proprio.to(device),
            actions=self.actions.to(device),
            rewards=self.rewards.to(device),
            next_image_obs=self.next_image_obs.to(device),
            next_proprio=self.next_proprio.to(device),
            dones=self.dones.to(device),
            is_online=self.is_online.to(device),
            metadata=self.metadata,
        )

    @property
    def batch_size(self) -> int:
        return int(self.actions.shape[0])

    @staticmethod
    def concat(batches: list["AWRStepBatch"]) -> "AWRStepBatch":
        valid_batches = [batch for batch in batches if batch.batch_size > 0]
        if len(valid_batches) == 0:
            raise ValueError("AWRStepBatch.concat requires at least one non-empty batch.")
        if len(valid_batches) == 1:
            return valid_batches[0]

        metadata: dict[str, Any] = {}
        for batch in valid_batches:
            for key, value in batch.metadata.items():
                metadata.setdefault(key, [])
                if isinstance(value, list):
                    metadata[key].extend(value)
                else:
                    metadata[key].append(value)
        return AWRStepBatch(
            image_obs=torch.cat([batch.image_obs for batch in valid_batches], dim=0),
            proprio=torch.cat([batch.proprio for batch in valid_batches], dim=0),
            actions=torch.cat([batch.actions for batch in valid_batches], dim=0),
            rewards=torch.cat([batch.rewards for batch in valid_batches], dim=0),
            next_image_obs=torch.cat([batch.next_image_obs for batch in valid_batches], dim=0),
            next_proprio=torch.cat([batch.next_proprio for batch in valid_batches], dim=0),
            dones=torch.cat([batch.dones for batch in valid_batches], dim=0),
            is_online=torch.cat([batch.is_online for batch in valid_batches], dim=0),
            metadata=metadata,
        )


__all__ = [
    "AWRActorBatch",
    "AWRConfig",
    "AWRStepBatch",
    "EncoderConfig",
    "FlowAugmentationConfig",
    "ReplayBufferConfig",
    "TrainerConfig",
    "Transition",
]
