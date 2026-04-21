from typing import Any

import torch
import torch.nn as nn


def _cfg_get(cfg: Any, key: str, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class MultiModalTransformerFusion(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_modalities: int,
        depth: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_modalities = num_modalities

        self.cls_token = nn.Parameter(torch.zeros(1, 1, feature_dim))
        self.modality_embeddings = nn.Parameter(torch.zeros(1, num_modalities, feature_dim))
        self.cls_pos_embedding = nn.Parameter(torch.zeros(1, 1, feature_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.output_norm = nn.LayerNorm(feature_dim)

    def forward(self, image_tokens: torch.Tensor, proprio_token: torch.Tensor) -> torch.Tensor:
        batch_size, num_images, _ = image_tokens.shape
        if num_images + 1 != self.num_modalities:
            raise ValueError(
                f"Expected {self.num_modalities - 1} image tokens, got {num_images}"
            )

        modality_tokens = torch.cat([image_tokens, proprio_token.unsqueeze(1)], dim=1)
        modality_tokens = modality_tokens + self.modality_embeddings[:, : modality_tokens.shape[1], :]
        cls_token = self.cls_token.expand(batch_size, -1, -1) + self.cls_pos_embedding
        tokens = torch.cat([cls_token, modality_tokens], dim=1)
        fused = self.transformer(tokens)
        return self.output_norm(fused[:, 0])


def build_fusion_module(cfg: Any, num_modalities: int) -> nn.Module:
    fusion_type = _cfg_get(cfg, "type", "transformer")
    if fusion_type != "transformer":
        raise ValueError(f"Unsupported fusion type: {fusion_type}")
    return MultiModalTransformerFusion(
        feature_dim=int(_cfg_get(cfg, "feature_dim")),
        num_modalities=num_modalities,
        depth=int(_cfg_get(cfg, "depth", 3)),
        num_heads=int(_cfg_get(cfg, "num_heads", 8)),
        dropout=float(_cfg_get(cfg, "dropout", 0.0)),
    )
