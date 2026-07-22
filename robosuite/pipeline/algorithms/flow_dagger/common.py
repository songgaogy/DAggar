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
class FlowDaggerConfig:
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
    n_ode_steps: int = 8
    device: str = "cuda:0"
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
    pretrain_steps: int = 20_000


@dataclass
class FlowDaggerBatch:
    image_obs: torch.Tensor
    proprio: torch.Tensor
    action_sequences: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    def to(self, device: torch.device | str) -> "FlowDaggerBatch":
        return FlowDaggerBatch(
            image_obs=self.image_obs.to(device),
            proprio=self.proprio.to(device),
            action_sequences=self.action_sequences.to(device),
            metadata=self.metadata,
        )

    @property
    def batch_size(self) -> int:
        return int(self.action_sequences.shape[0])


__all__ = [
    "EncoderConfig",
    "FlowAugmentationConfig",
    "FlowDaggerBatch",
    "FlowDaggerConfig",
    "ReplayBufferConfig",
    "TrainerConfig",
    "Transition",
]
