from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from robosuite.pipeline.common.types import (
    EncoderConfig,
    ReplayBatch,
    ReplayBufferConfig,
    Transition,
)


@dataclass
class BCConfig:
    action_dim: int
    hidden_dims: Sequence[int] = field(default_factory=lambda: (256, 256))
    learning_rate: float = 3e-4
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    tanh_squash_distribution: bool = True
    hybrid_gripper_head: bool = True
    num_gripper_actions: int = 3
    gripper_loss_weight: float = 1.0
    device: str = "cpu"
    inference_device: Optional[str] = None


@dataclass
class TrainerConfig:
    batch_size: int = 256
    warmup_steps: int = 0
    updates_per_step: int = 1
    steps_per_update: int = 50
    random_steps: int = 0
    pretrain_steps: int = 20_000


__all__ = [
    "BCConfig",
    "EncoderConfig",
    "ReplayBatch",
    "ReplayBufferConfig",
    "TrainerConfig",
    "Transition",
]
