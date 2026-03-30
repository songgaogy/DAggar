from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

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
class DipoleConfig:
    action_dim: int
    proprio_dim: int
    action_horizon: int = 8
    execute_horizon: int = 1
    image_size: int = 128
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    grad_clip_norm: float = 1.0
    lambda_endpoint: float = 0.5
    lambda_smooth: float = 0.05
    beta: float = 2.0
    guidance_scale: float = 1.0
    positive_loss_scale: float = 1.0
    negative_loss_scale: float = 1.0
    n_ode_steps: int = 8
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
    pretrain_steps: int = 20_000
    online_fraction: float = 0.5


@dataclass
class DipoleBatch:
    image_obs: torch.Tensor
    proprio: torch.Tensor
    action_sequences: torch.Tensor
    lambda_values: torch.Tensor
    force_positive: torch.Tensor
    is_online: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    def to(self, device: torch.device | str) -> "DipoleBatch":
        return DipoleBatch(
            image_obs=self.image_obs.to(device),
            proprio=self.proprio.to(device),
            action_sequences=self.action_sequences.to(device),
            lambda_values=self.lambda_values.to(device),
            force_positive=self.force_positive.to(device),
            is_online=self.is_online.to(device),
            metadata=self.metadata,
        )

    @property
    def batch_size(self) -> int:
        return int(self.action_sequences.shape[0])

    @staticmethod
    def concat(batches: list["DipoleBatch"]) -> "DipoleBatch":
        valid_batches = [batch for batch in batches if batch.batch_size > 0]
        if len(valid_batches) == 0:
            raise ValueError("DipoleBatch.concat requires at least one non-empty batch.")
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
        return DipoleBatch(
            image_obs=torch.cat([batch.image_obs for batch in valid_batches], dim=0),
            proprio=torch.cat([batch.proprio for batch in valid_batches], dim=0),
            action_sequences=torch.cat([batch.action_sequences for batch in valid_batches], dim=0),
            lambda_values=torch.cat([batch.lambda_values for batch in valid_batches], dim=0),
            force_positive=torch.cat([batch.force_positive for batch in valid_batches], dim=0),
            is_online=torch.cat([batch.is_online for batch in valid_batches], dim=0),
            metadata=metadata,
        )


__all__ = [
    "DipoleBatch",
    "DipoleConfig",
    "EncoderConfig",
    "FlowAugmentationConfig",
    "ReplayBufferConfig",
    "TrainerConfig",
    "Transition",
]
