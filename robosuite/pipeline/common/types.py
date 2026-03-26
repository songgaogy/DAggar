from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

import numpy as np
import torch


Observation = np.ndarray | Mapping[str, Any]
TensorObservation = torch.Tensor | dict[str, Any]


@dataclass
class Transition:
    obs: Observation
    action: Any
    reward: Optional[float]
    next_obs: Observation
    done: bool
    grasp_penalty: Optional[float] = None
    is_intervention: bool = False
    info: Optional[dict[str, Any]] = None
    reward_source: Optional[str] = None
    demo_source: Optional[str] = None


@dataclass
class ReplayBatch:
    obs: TensorObservation
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: TensorObservation
    dones: torch.Tensor
    grasp_penalty: torch.Tensor
    is_intervention: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    def to(self, device: torch.device | str) -> "ReplayBatch":
        return ReplayBatch(
            obs=_tree_to_device(self.obs, device),
            actions=self.actions.to(device),
            rewards=self.rewards.to(device),
            next_obs=_tree_to_device(self.next_obs, device),
            dones=self.dones.to(device),
            grasp_penalty=self.grasp_penalty.to(device),
            is_intervention=self.is_intervention.to(device),
            metadata=self.metadata,
        )

    @property
    def batch_size(self) -> int:
        return int(self.actions.shape[0])


def _tree_to_device(value: Any, device: torch.device | str) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _tree_to_device(item, device) for key, item in value.items()}
    return value


class RewardProvider(Protocol):
    def __call__(self, transition: Transition, env_reward: Optional[float] = None) -> float | tuple[float, str]:
        ...


BatchAugmentationFn = Callable[[ReplayBatch], ReplayBatch]


@dataclass
class EncoderConfig:
    encoder_type: str = "resnet-pretrained"
    image_keys: Sequence[str] = field(default_factory=tuple)
    proprio_keys: Sequence[str] = field(default_factory=tuple)
    feature_dim: int = 256
    image_size: int = 84
    cnn_channels: Sequence[int] = field(default_factory=lambda: (32, 64, 64, 64))
    use_layer_norm: bool = True
    resnet_name: str = "resnet18"
    pretrained: bool = True
    freeze_backbone: bool = True
    share_image_encoder: bool = False
    proprio_feature_dim: int = 64
    num_spatial_blocks: int = 8
    pretrained_path: Optional[str] = None


@dataclass
class ReplayBufferConfig:
    capacity: int = 200_000
    batch_size: int = 256


@dataclass
class SACConfig:
    action_dim: int
    actor_hidden_dims: Sequence[int] = field(default_factory=lambda: (256, 256))
    critic_hidden_dims: Sequence[int] = field(default_factory=lambda: (256, 256))
    discount: float = 0.97
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    init_temperature: float = 1e-2
    target_entropy: Optional[float] = None
    auto_entropy_tuning: bool = True
    backup_entropy: bool = False
    reward_bias: float = 0.0
    critic_ensemble_size: int = 2
    critic_subsample_size: Optional[int] = None
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    device: str = "cpu"
    inference_device: Optional[str] = None


@dataclass
class TrainerConfig:
    batch_size: int = 256
    cta_ratio: int = 2
    warmup_steps: int = 100
    updates_per_step: int = 1
    steps_per_update: int = 50
    random_steps: int = 0
    online_fraction: float = 0.5

    def split_batch_sizes(self) -> tuple[int, int]:
        online_batch = int(round(self.batch_size * self.online_fraction))
        online_batch = min(max(1, online_batch), self.batch_size - 1)
        demo_batch = self.batch_size - online_batch
        return online_batch, demo_batch
