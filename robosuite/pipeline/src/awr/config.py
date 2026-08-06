from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


def cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def require_cuda_device(value: str, *, name: str) -> str:
    device = str(value)
    if device != "cuda" and not (
        device.startswith("cuda:") and device[5:].isdigit()
    ):
        raise ValueError(f"{name} must be a CUDA device, got {device!r}.")
    return device


@dataclass
class EncoderConfig:
    encoder_type: str = "flow-multi"
    image_keys: Sequence[str] = field(default_factory=tuple)
    proprio_keys: Sequence[str] = ("state",)
    image_size: int = 128


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
    critic_hidden_dims: Sequence[int] = (512, 512)
    n_ode_steps: int = 8
    device: str = "cuda:0"
    inference_device: str = "cuda:0"
    task_name: str = "PickPlaceCereal"
    language_instruction: str | None = None
    augmentation: FlowAugmentationConfig = field(default_factory=FlowAugmentationConfig)

    def __post_init__(self) -> None:
        self.device = require_cuda_device(self.device, name="awr.device")
        self.inference_device = require_cuda_device(
            self.inference_device, name="awr.inference_device"
        )


@dataclass
class TrainerConfig:
    value_batch_size: int = 256
    actor_batch_size: int = 128
    warmup_steps: int = 0
    episodes_per_train: int = 10
    updates_per_train: int = 2000
    inference_sync_interval: int = 50
    value_warmup_steps: int = 20_000


__all__ = [
    "AWRConfig",
    "EncoderConfig",
    "FlowAugmentationConfig",
    "TrainerConfig",
    "cfg_get",
    "require_cuda_device",
]
