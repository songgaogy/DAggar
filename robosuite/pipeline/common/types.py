from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np


Observation = np.ndarray | Mapping[str, Any]


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
