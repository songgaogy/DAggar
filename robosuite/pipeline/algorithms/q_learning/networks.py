"""ResNet-50 Q-chunk and V networks for DIPOLE-RL IQL.

The discriminator keeps its LPB ``SharedFrozenEncoder``. Q/V instead own
independent frozen torchvision ResNet-50 backbones initialized from the
configured local checkpoint.

Architecture:
    Q: concat(vis_encoder(images), state_encoder(state),
              action_encoder(flatten(action_chunk))) -> MLP -> (B, 1)
    V: concat(vis_encoder(images), state_encoder(state)) -> MLP -> (B, 1)
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision.models import resnet50


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
RESNET50_OUTPUT_DIM = 2048


def _build_mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = int(input_dim)
    for hidden_dim in hidden_dims:
        linear = nn.Linear(last_dim, int(hidden_dim))
        nn.init.kaiming_normal_(linear.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(linear.bias)
        layers.append(linear)
        layers.append(nn.LayerNorm(int(hidden_dim)))
        layers.append(nn.GELU())
        last_dim = int(hidden_dim)
    final = nn.Linear(last_dim, int(output_dim))
    nn.init.zeros_(final.weight)
    nn.init.zeros_(final.bias)
    layers.append(final)
    return nn.Sequential(*layers)


def _build_feature_encoder(input_dim: int, output_dim: int) -> nn.Sequential:
    linear = nn.Linear(int(input_dim), int(output_dim))
    nn.init.kaiming_normal_(linear.weight, mode="fan_in", nonlinearity="relu")
    nn.init.zeros_(linear.bias)
    return nn.Sequential(linear, nn.LayerNorm(int(output_dim)), nn.GELU())


class FrozenMultiViewResNet50Encoder(nn.Module):
    """Shared-across-views frozen ResNet-50 with ImageNet normalization."""

    def __init__(self, *, num_cameras: int, pretrained_path: str) -> None:
        super().__init__()
        if int(num_cameras) <= 0:
            raise ValueError(f"num_cameras must be positive, got {num_cameras}")
        path = Path(str(pretrained_path)).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"ResNet-50 checkpoint not found: {path}")

        backbone = resnet50(weights=None)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, Mapping) and "state_dict" in payload:
            payload = payload["state_dict"]
        if not isinstance(payload, Mapping):
            raise TypeError(
                f"ResNet-50 checkpoint must contain a state dict, got {type(payload).__name__}"
            )
        backbone.load_state_dict(payload, strict=True)
        backbone.fc = nn.Identity()
        for param in backbone.parameters():
            param.requires_grad_(False)
        backbone.eval()

        self.backbone = backbone
        self.num_cameras = int(num_cameras)
        self.output_dim = self.num_cameras * RESNET50_OUTPUT_DIM
        self.pretrained_path = str(path)
        self.register_buffer(
            "_image_mean",
            torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_image_std",
            torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def train(self, mode: bool = True) -> "FrozenMultiViewResNet50Encoder":
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, image_obs_raw: torch.Tensor) -> torch.Tensor:
        if image_obs_raw.dim() != 5:
            raise ValueError(
                "FrozenMultiViewResNet50Encoder expected images (B, V, 3, H, W); "
                f"got {tuple(image_obs_raw.shape)}"
            )
        B, V, C, H, W = image_obs_raw.shape
        if V != self.num_cameras or C != 3:
            raise ValueError(
                "FrozenMultiViewResNet50Encoder expected "
                f"(B, {self.num_cameras}, 3, H, W); got {tuple(image_obs_raw.shape)}"
            )
        images = image_obs_raw.to(dtype=torch.float32).reshape(B * V, C, H, W)
        images = (images - self._image_mean) / self._image_std
        with torch.no_grad():
            features = self.backbone(images)
        return features.reshape(B, V * RESNET50_OUTPUT_DIM)


class _CompactCheckpointMixin:
    """Exclude frozen ResNet tensors from Q/V checkpoints."""

    _BACKBONE_PREFIX = "vis_encoder.backbone."

    def compact_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            key: value
            for key, value in self.state_dict().items()
            if not key.startswith(self._BACKBONE_PREFIX)
        }

    def load_compact_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        expected = set(self.compact_state_dict())
        actual = set(state)
        if actual != expected:
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            raise ValueError(
                "Compact Q/V state mismatch: "
                f"missing={missing[:5]} unexpected={unexpected[:5]}"
            )
        incompatible = self.load_state_dict(dict(state), strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing = [
            key
            for key in incompatible.missing_keys
            if not key.startswith(self._BACKBONE_PREFIX)
        ]
        if missing or unexpected:
            raise ValueError(
                f"Compact Q/V state load failed: missing={missing} unexpected={unexpected}"
            )


class QChunkNetwork(_CompactCheckpointMixin, nn.Module):
    """Q(images, state, action_chunk) -> (B, 1)."""

    def __init__(
        self,
        *,
        num_cameras: int,
        proprio_dim: int,
        action_dim: int,
        action_horizon: int,
        resnet_pretrained_path: str,
        state_feature_dim: int = 256,
        action_feature_dim: int = 256,
        hidden_dims: tuple[int, ...] = (512, 512),
    ) -> None:
        super().__init__()
        self.num_cameras = int(num_cameras)
        self.proprio_dim = int(proprio_dim)
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.state_feature_dim = int(state_feature_dim)
        self.action_feature_dim = int(action_feature_dim)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.vis_encoder = FrozenMultiViewResNet50Encoder(
            num_cameras=self.num_cameras,
            pretrained_path=resnet_pretrained_path,
        )
        self.state_encoder = _build_feature_encoder(self.proprio_dim, self.state_feature_dim)
        self.action_encoder = _build_feature_encoder(
            self.action_dim * self.action_horizon,
            self.action_feature_dim,
        )
        input_dim = self.vis_encoder.output_dim + self.state_feature_dim + self.action_feature_dim
        self.net = _build_mlp(input_dim, self.hidden_dims, 1)

    def forward(
        self,
        image_obs_raw: torch.Tensor,
        proprio_raw: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        if proprio_raw.dim() != 2 or proprio_raw.shape[1] != self.proprio_dim:
            raise ValueError(
                f"QChunkNetwork expected proprio (B, {self.proprio_dim}); "
                f"got {tuple(proprio_raw.shape)}"
            )
        if action_chunk.dim() != 3 or action_chunk.shape[1:] != (
            self.action_horizon,
            self.action_dim,
        ):
            raise ValueError(
                "QChunkNetwork expected action_chunk "
                f"(B, {self.action_horizon}, {self.action_dim}); "
                f"got {tuple(action_chunk.shape)}"
            )
        B = int(proprio_raw.shape[0])
        vis = self.vis_encoder(image_obs_raw)
        state = self.state_encoder(proprio_raw)
        action = self.action_encoder(action_chunk.reshape(B, -1))
        return self.net(torch.cat([vis, state, action], dim=-1))


class VNetwork(_CompactCheckpointMixin, nn.Module):
    """V(images, state) -> (B, 1)."""

    def __init__(
        self,
        *,
        num_cameras: int,
        proprio_dim: int,
        resnet_pretrained_path: str,
        state_feature_dim: int = 256,
        hidden_dims: tuple[int, ...] = (512, 512),
    ) -> None:
        super().__init__()
        self.num_cameras = int(num_cameras)
        self.proprio_dim = int(proprio_dim)
        self.state_feature_dim = int(state_feature_dim)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.vis_encoder = FrozenMultiViewResNet50Encoder(
            num_cameras=self.num_cameras,
            pretrained_path=resnet_pretrained_path,
        )
        self.state_encoder = _build_feature_encoder(self.proprio_dim, self.state_feature_dim)
        input_dim = self.vis_encoder.output_dim + self.state_feature_dim
        self.net = _build_mlp(input_dim, self.hidden_dims, 1)

    def forward(self, image_obs_raw: torch.Tensor, proprio_raw: torch.Tensor) -> torch.Tensor:
        if proprio_raw.dim() != 2 or proprio_raw.shape[1] != self.proprio_dim:
            raise ValueError(
                f"VNetwork expected proprio (B, {self.proprio_dim}); "
                f"got {tuple(proprio_raw.shape)}"
            )
        vis = self.vis_encoder(image_obs_raw)
        state = self.state_encoder(proprio_raw)
        return self.net(torch.cat([vis, state], dim=-1))


def trainable_parameters(module: nn.Module) -> list[nn.Parameter]:
    return [param for param in module.parameters() if param.requires_grad]


def resolved_resnet_path(path: str) -> str:
    return str(Path(str(path)).expanduser().resolve())


__all__ = [
    "FrozenMultiViewResNet50Encoder",
    "QChunkNetwork",
    "VNetwork",
    "resolved_resnet_path",
    "trainable_parameters",
]
