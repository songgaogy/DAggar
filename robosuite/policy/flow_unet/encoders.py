import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


def _cfg_get(cfg: Any, key: str, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class SpatialSoftmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, channels, height, width = x.shape
        pos_y, pos_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=x.device, dtype=x.dtype),
            torch.linspace(-1.0, 1.0, width, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        pos_x = pos_x.reshape(height * width)
        pos_y = pos_y.reshape(height * width)

        attention = F.softmax(x.reshape(bsz, channels, height * width), dim=-1)
        expected_x = torch.sum(attention * pos_x.view(1, 1, -1), dim=-1)
        expected_y = torch.sum(attention * pos_y.view(1, 1, -1), dim=-1)
        return torch.cat([expected_x, expected_y], dim=-1)


class SharedResNetSpatialSoftmaxEncoder(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        pretrained_path: str,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        backbone = resnet18(weights=None)
        state_dict = torch.load(pretrained_path, map_location="cpu")
        backbone.load_state_dict(state_dict, strict=True)

        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self.spatial_softmax = SpatialSoftmax()
        self.proj = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, feature_dim),
            nn.Mish(),
            nn.Linear(feature_dim, feature_dim),
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        keypoints = self.spatial_softmax(features)
        return self.proj(keypoints)


class ProprioMLPProjector(nn.Module):
    def __init__(self, proprio_dim: int, hidden_dim: int, feature_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(proprio_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        return self.net(proprio)


def build_image_encoder(cfg: Any) -> nn.Module:
    encoder_type = _cfg_get(cfg, "type", "shared_resnet18_spatial_softmax")
    if encoder_type != "shared_resnet18_spatial_softmax":
        raise ValueError(f"Unsupported image encoder type: {encoder_type}")
    return SharedResNetSpatialSoftmaxEncoder(
        feature_dim=int(_cfg_get(cfg, "feature_dim")),
        pretrained_path=str(_cfg_get(cfg, "pretrained_path")),
        freeze_backbone=bool(_cfg_get(cfg, "freeze_backbone", False)),
    )


def build_proprio_projector(cfg: Any, proprio_dim: int, feature_dim: int) -> nn.Module:
    projector_type = _cfg_get(cfg, "type", "mlp")
    if projector_type != "mlp":
        raise ValueError(f"Unsupported proprio projector type: {projector_type}")
    hidden_dim = int(_cfg_get(cfg, "hidden_dim", feature_dim))
    return ProprioMLPProjector(
        proprio_dim=proprio_dim,
        hidden_dim=hidden_dim,
        feature_dim=feature_dim,
    )
