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
        include_spatial_softmax: bool = True,
        layer3_pool_size: int = 2,
        layer4_pool_size: int = 2,
    ):
        super().__init__()
        backbone = resnet18(weights=None)
        state_dict = torch.load(pretrained_path, map_location="cpu")
        backbone.load_state_dict(state_dict, strict=True)

        self.include_spatial_softmax = bool(include_spatial_softmax)
        self.layer3_pool_size = int(layer3_pool_size)
        self.layer4_pool_size = int(layer4_pool_size)
        self.backbone = nn.ModuleDict(
            {
                "stem": nn.Sequential(
                    backbone.conv1,
                    backbone.bn1,
                    backbone.relu,
                    backbone.maxpool,
                    backbone.layer1,
                    backbone.layer2,
                ),
                "layer3": backbone.layer3,
                "layer4": backbone.layer4,
            }
        )

        self.num_tokens = 0
        if self.include_spatial_softmax:
            self.spatial_softmax = SpatialSoftmax()
            self.spatial_proj = nn.Sequential(
                nn.LayerNorm(1024),
                nn.Linear(1024, feature_dim),
                nn.Mish(),
                nn.Linear(feature_dim, feature_dim),
            )
            self.spatial_token_embedding = nn.Parameter(torch.zeros(1, 1, feature_dim))
            self.num_tokens += 1

        if self.layer3_pool_size > 0:
            self.layer3_proj = nn.Sequential(
                nn.LayerNorm(256),
                nn.Linear(256, feature_dim),
                nn.Mish(),
                nn.Linear(feature_dim, feature_dim),
            )
            self.layer3_token_embedding = nn.Parameter(
                torch.zeros(1, self.layer3_pool_size * self.layer3_pool_size, feature_dim)
            )
            self.num_tokens += self.layer3_pool_size * self.layer3_pool_size

        if self.layer4_pool_size > 0:
            self.layer4_proj = nn.Sequential(
                nn.LayerNorm(512),
                nn.Linear(512, feature_dim),
                nn.Mish(),
                nn.Linear(feature_dim, feature_dim),
            )
            self.layer4_token_embedding = nn.Parameter(
                torch.zeros(1, self.layer4_pool_size * self.layer4_pool_size, feature_dim)
            )
            self.num_tokens += self.layer4_pool_size * self.layer4_pool_size

        if self.num_tokens <= 0:
            raise ValueError("Image encoder must output at least one token per image")

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        stem = self.backbone["stem"](images)
        layer3 = self.backbone["layer3"](stem)
        layer4 = self.backbone["layer4"](layer3)

        tokens = []
        if self.include_spatial_softmax:
            keypoints = self.spatial_softmax(layer4)
            spatial_token = self.spatial_proj(keypoints).unsqueeze(1)
            tokens.append(spatial_token + self.spatial_token_embedding)

        if self.layer3_pool_size > 0:
            pooled_layer3 = F.adaptive_avg_pool2d(layer3, output_size=(self.layer3_pool_size, self.layer3_pool_size))
            layer3_tokens = pooled_layer3.flatten(2).transpose(1, 2)
            layer3_tokens = self.layer3_proj(layer3_tokens)
            tokens.append(layer3_tokens + self.layer3_token_embedding)

        if self.layer4_pool_size > 0:
            pooled_layer4 = F.adaptive_avg_pool2d(layer4, output_size=(self.layer4_pool_size, self.layer4_pool_size))
            layer4_tokens = pooled_layer4.flatten(2).transpose(1, 2)
            layer4_tokens = self.layer4_proj(layer4_tokens)
            tokens.append(layer4_tokens + self.layer4_token_embedding)

        return torch.cat(tokens, dim=1)


class ProprioMLPTokenizer(nn.Module):
    def __init__(self, proprio_dim: int, hidden_dim: int, feature_dim: int, num_tokens: int):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.feature_dim = int(feature_dim)
        self.net = nn.Sequential(
            nn.Linear(proprio_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, self.num_tokens * self.feature_dim),
        )
        self.output_norm = nn.LayerNorm(self.feature_dim)

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        tokens = self.net(proprio).view(proprio.shape[0], self.num_tokens, self.feature_dim)
        return self.output_norm(tokens)


def build_image_encoder(cfg: Any) -> nn.Module:
    encoder_type = _cfg_get(cfg, "type", "shared_resnet18_spatial_softmax")
    if encoder_type != "shared_resnet18_spatial_softmax":
        raise ValueError(f"Unsupported image encoder type: {encoder_type}")
    return SharedResNetSpatialSoftmaxEncoder(
        feature_dim=int(_cfg_get(cfg, "feature_dim")),
        pretrained_path=str(_cfg_get(cfg, "pretrained_path")),
        freeze_backbone=bool(_cfg_get(cfg, "freeze_backbone", False)),
        include_spatial_softmax=bool(_cfg_get(cfg, "include_spatial_softmax", True)),
        layer3_pool_size=int(_cfg_get(cfg, "layer3_pool_size", 2)),
        layer4_pool_size=int(_cfg_get(cfg, "layer4_pool_size", 2)),
    )


def build_proprio_tokenizer(cfg: Any, proprio_dim: int, feature_dim: int) -> nn.Module:
    tokenizer_type = _cfg_get(cfg, "type", "mlp")
    if tokenizer_type != "mlp":
        raise ValueError(f"Unsupported proprio tokenizer type: {tokenizer_type}")
    hidden_dim = int(_cfg_get(cfg, "hidden_dim", feature_dim))
    return ProprioMLPTokenizer(
        proprio_dim=proprio_dim,
        hidden_dim=hidden_dim,
        feature_dim=feature_dim,
        num_tokens=int(_cfg_get(cfg, "num_tokens", 2)),
    )


def build_proprio_projector(cfg: Any, proprio_dim: int, feature_dim: int) -> nn.Module:
    return build_proprio_tokenizer(cfg, proprio_dim=proprio_dim, feature_dim=feature_dim)
