from __future__ import annotations

import pickle
import sys
import types
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from robosuite.pipeline.utils.tensor import standardize_image_tensor


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


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_encoder(observation_example, config: EncoderConfig) -> "MultiModalObservationEncoder":
    return MultiModalObservationEncoder(observation_example=observation_example, config=config)


class MLP(nn.Module):
    def __init__(self, hidden_dims, activation=nn.Tanh, use_layer_norm: bool = True, activate_final: bool = True):
        super().__init__()
        layers = []
        dims = list(hidden_dims)
        for index in range(len(dims) - 1):
            layers.append(nn.Linear(dims[index], dims[index + 1]))
            if use_layer_norm:
                layers.append(nn.LayerNorm(dims[index + 1]))
            if activate_final or index < len(dims) - 2:
                layers.append(activation())
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class SimpleCNNEncoder(nn.Module):
    def __init__(self, in_channels: int, feature_dim: int, channels=(32, 64, 64, 64)):
        super().__init__()
        layers = []
        current_channels = in_channels
        for index, next_channels in enumerate(channels):
            stride = 2 if index < 3 else 1
            layers.append(nn.Conv2d(current_channels, next_channels, kernel_size=3, stride=stride, padding=1))
            layers.append(nn.ReLU(inplace=True))
            current_channels = next_channels
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.backbone = nn.Sequential(*layers)
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(current_channels, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )
        self.output_dim = int(feature_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.projection(self.backbone(image))


class SpatialLearnedEmbeddings(nn.Module):
    def __init__(self, height: int, width: int, channels: int, num_features: int = 8) -> None:
        super().__init__()
        self.kernel = nn.Parameter(torch.empty(height, width, channels, num_features))
        nn.init.kaiming_normal_(self.kernel)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError(f"Expected feature map with shape (B, C, H, W), got {tuple(features.shape)}.")
        pooled = torch.einsum("bchw,hwcf->bcf", features, self.kernel)
        return pooled.reshape(features.shape[0], -1)


class ResNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm1 = nn.GroupNorm(num_groups=4, num_channels=out_channels, eps=1e-5)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.norm2 = nn.GroupNorm(num_groups=4, num_channels=out_channels, eps=1e-5)
        self.proj = None
        self.proj_norm = None
        if stride != 1 or in_channels != out_channels:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            self.proj_norm = nn.GroupNorm(num_groups=4, num_channels=out_channels, eps=1e-5)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        outputs = torch.relu(self.norm1(self.conv1(inputs)))
        outputs = self.norm2(self.conv2(outputs))
        if self.proj is not None and self.proj_norm is not None:
            residual = self.proj_norm(self.proj(residual))
        return torch.relu(outputs + residual)


class OfficialResNet10Trunk(nn.Module):
    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.conv_init = nn.Conv2d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.norm_init = nn.GroupNorm(num_groups=4, num_channels=64, eps=1e-5)
        self.blocks = nn.ModuleList(
            [
                ResNetBlock(64, 64, stride=1),
                ResNetBlock(64, 128, stride=2),
                ResNetBlock(128, 256, stride=2),
                ResNetBlock(256, 512, stride=2),
            ]
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        outputs = self.conv_init(image)
        outputs = torch.relu(self.norm_init(outputs))
        outputs = F.max_pool2d(outputs, kernel_size=3, stride=2, padding=1)
        for block in self.blocks:
            outputs = block(outputs)
        return outputs


class OfficialPretrainedResNet10Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        feature_dim: int,
        num_spatial_blocks: int,
        image_size: int,
        pretrained_path: str | None = None,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.input_adapter = nn.Identity() if in_channels == 3 else nn.Conv2d(in_channels, 3, kernel_size=1)
        self.backbone = OfficialResNet10Trunk(in_channels=3)
        feature_height, feature_width, feature_channels = self._infer_backbone_shape()
        self.pool = SpatialLearnedEmbeddings(
            height=feature_height,
            width=feature_width,
            channels=feature_channels,
            num_features=int(num_spatial_blocks),
        )
        self.dropout = nn.Dropout(p=0.1)
        self.projection = nn.Sequential(
            nn.Linear(feature_channels * int(num_spatial_blocks), feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )
        self.output_dim = int(feature_dim)
        if pretrained_path:
            load_official_resnet10_weights(self.backbone, pretrained_path)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad_(False)

    def _infer_backbone_shape(self) -> tuple[int, int, int]:
        height = self.image_size
        width = self.image_size
        # conv_init, max-pool, and the final three residual stages each use stride 2.
        for _ in range(5):
            height = (height + 1) // 2
            width = (width + 1) // 2
        return height, width, 512

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = _normalize_resnet_image(image, image_size=self.image_size)
        image = self.input_adapter(image)
        with torch.set_grad_enabled(any(param.requires_grad for param in self.backbone.parameters())):
            feature_map = self.backbone(image)
        pooled = self.pool(feature_map)
        pooled = self.dropout(pooled)
        return self.projection(pooled)


class MultiModalObservationEncoder(nn.Module):
    def __init__(self, observation_example, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.image_keys = list(config.image_keys)
        self.proprio_keys = list(config.proprio_keys)

        if isinstance(observation_example, Mapping):
            example_dict = dict(observation_example)
        else:
            example_dict = {"state": observation_example}
            if not self.proprio_keys and not self.image_keys:
                self.proprio_keys = ["state"]

        if not self.image_keys:
            self.image_keys = [
                key for key, value in example_dict.items() if _looks_like_image(value)
            ]
        if not self.proprio_keys:
            self.proprio_keys = [key for key in example_dict.keys() if key not in self.image_keys]

        self.image_example_shapes = {
            key: tuple(np.asarray(example_dict[key]).shape) for key in self.image_keys
        }

        self.shared_image_encoder: nn.Module | None = None
        self.image_encoders = nn.ModuleDict()
        if self.image_keys:
            if config.share_image_encoder:
                self.shared_image_encoder = self._build_image_encoder(self.image_keys[0])
            else:
                for key in self.image_keys:
                    self.image_encoders[key] = self._build_image_encoder(key)

        self.state_projector = nn.Sequential(
            nn.LazyLinear(int(config.proprio_feature_dim)),
            nn.LayerNorm(int(config.proprio_feature_dim)),
            nn.Tanh(),
        )

        image_output_dim = 0
        if self.image_keys:
            if self.shared_image_encoder is not None:
                image_output_dim = len(self.image_keys) * int(self.shared_image_encoder.output_dim)
            else:
                image_output_dim = sum(int(self.image_encoders[key].output_dim) for key in self.image_keys)
        proprio_output_dim = int(config.proprio_feature_dim) if self.proprio_keys else 0
        self.output_dim = image_output_dim + proprio_output_dim

    def _build_image_encoder(self, key: str) -> nn.Module:
        shape = self.image_example_shapes[key]
        if len(shape) < 3:
            raise ValueError(f"Image observation {key!r} has invalid shape {shape}.")
        in_channels = int(shape[-1] if shape[-1] in (1, 3, 4) else shape[-3])
        if self.config.encoder_type in ("cnn", "simple_cnn"):
            return SimpleCNNEncoder(
                in_channels=in_channels,
                feature_dim=self.config.feature_dim,
                channels=tuple(self.config.cnn_channels),
            )
        if self.config.encoder_type in ("resnet", "resnet-pretrained"):
            pretrained_path = None
            freeze_backbone = False
            if self.config.encoder_type == "resnet-pretrained" or self.config.pretrained:
                pretrained_path = self.config.pretrained_path
                freeze_backbone = bool(self.config.freeze_backbone)
            return OfficialPretrainedResNet10Encoder(
                in_channels=in_channels,
                feature_dim=self.config.feature_dim,
                num_spatial_blocks=self.config.num_spatial_blocks,
                image_size=self.config.image_size,
                pretrained_path=pretrained_path,
                freeze_backbone=freeze_backbone,
            )
        raise ValueError(f"Unsupported encoder_type: {self.config.encoder_type}.")

    def _lookup_image_encoder(self, key: str) -> nn.Module:
        if self.shared_image_encoder is not None:
            return self.shared_image_encoder
        return self.image_encoders[key]

    def forward(self, obs, stop_gradient: bool = False) -> torch.Tensor:
        if not isinstance(obs, Mapping):
            obs = {"state": obs}

        features = []
        for key in self.image_keys:
            if key not in obs:
                continue
            image = standardize_image_tensor(torch.as_tensor(obs[key], device=self._device()))
            encoded = self._lookup_image_encoder(key)(image)
            features.append(encoded.detach() if stop_gradient else encoded)

        proprio_features = []
        for key in self.proprio_keys:
            if key not in obs:
                continue
            value = torch.as_tensor(obs[key], device=self._device()).float()
            if value.ndim == 1:
                value = value.unsqueeze(0)
            proprio_features.append(value.flatten(start_dim=1))
        if proprio_features:
            proprio = self.state_projector(torch.cat(proprio_features, dim=-1))
            features.append(proprio.detach() if stop_gradient else proprio)

        if not features:
            raise ValueError("No observation features were found for the configured encoder keys.")
        return torch.cat(features, dim=-1)

    def _device(self) -> torch.device:
        return next(self.parameters()).device


def load_official_resnet10_weights(backbone: OfficialResNet10Trunk, path: str | Path) -> None:
    params = _load_official_resnet10_params(path)
    with torch.no_grad():
        _load_conv2d(backbone.conv_init, params["conv_init"]["kernel"])
        _load_group_norm(backbone.norm_init, params["norm_init"])
        for block_index, block in enumerate(backbone.blocks):
            block_params = params[f"ResNetBlock_{block_index}"]
            _load_conv2d(block.conv1, block_params["Conv_0"]["kernel"])
            _load_conv2d(block.conv2, block_params["Conv_1"]["kernel"])
            _load_group_norm(block.norm1, block_params["MyGroupNorm_0"])
            _load_group_norm(block.norm2, block_params["MyGroupNorm_1"])
            if block.proj is not None and "conv_proj" in block_params:
                _load_conv2d(block.proj, block_params["conv_proj"]["kernel"])
            if block.proj_norm is not None and "norm_proj" in block_params:
                _load_group_norm(block.proj_norm, block_params["norm_proj"])


def _load_official_resnet10_params(path: str | Path) -> dict:
    jax_module = types.ModuleType("jax")
    jax_src_module = types.ModuleType("jax._src")
    jax_array_module = types.ModuleType("jax._src.array")

    def _reconstruct_array(numpy_reconstruct, numpy_args, state, _jax_state):
        array = numpy_reconstruct(*numpy_args)
        array.__setstate__(state)
        return array

    jax_array_module._reconstruct_array = _reconstruct_array
    restore_modules = {}
    for module_name, module in (
        ("jax", jax_module),
        ("jax._src", jax_src_module),
        ("jax._src.array", jax_array_module),
    ):
        restore_modules[module_name] = sys.modules.get(module_name)
        sys.modules[module_name] = module
    try:
        with Path(path).open("rb") as file_handle:
            return pickle.load(file_handle)
    finally:
        for module_name, module in restore_modules.items():
            if module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = module


def _load_conv2d(module: nn.Conv2d, kernel: np.ndarray) -> None:
    weight = torch.from_numpy(np.asarray(kernel).transpose(3, 2, 0, 1)).to(dtype=module.weight.dtype)
    module.weight.copy_(weight)


def _load_group_norm(module: nn.GroupNorm, params: Mapping[str, np.ndarray]) -> None:
    module.weight.copy_(torch.from_numpy(np.asarray(params["scale"])).to(dtype=module.weight.dtype))
    module.bias.copy_(torch.from_numpy(np.asarray(params["bias"])).to(dtype=module.bias.dtype))


def _normalize_resnet_image(image: torch.Tensor, image_size: int) -> torch.Tensor:
    if image.shape[-2:] != (image_size, image_size):
        image = F.interpolate(image, size=(image_size, image_size), mode="bilinear", align_corners=False)
    scale_uint8 = not image.is_floating_point()
    image = image.float()
    if scale_uint8:
        image = image / 255.0
    mean = torch.as_tensor(IMAGENET_MEAN, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    std = torch.as_tensor(IMAGENET_STD, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    return (image - mean) / std


def _looks_like_image(value) -> bool:
    shape = tuple(np.asarray(value).shape)
    if len(shape) < 3:
        return False
    spatial_shape = shape[-3:]
    return spatial_shape[-1] in (1, 3, 4) or spatial_shape[0] in (1, 3, 4)
