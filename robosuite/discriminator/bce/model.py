from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _cfg_get(cfg: Any, key: str, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class ActionSequenceEncoder(nn.Module):
    def __init__(
        self,
        action_dim: int,
        model_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        max_horizon: int,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.model_dim = int(model_dim)
        self.max_horizon = int(max_horizon)
        self.action_proj = nn.Linear(self.action_dim, self.model_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.model_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.max_horizon + 1, self.model_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=int(num_heads),
            dim_feedforward=int(hidden_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        self.norm = nn.LayerNorm(self.model_dim)

    def forward(self, action_sequence: torch.Tensor) -> torch.Tensor:
        if action_sequence.ndim != 3:
            raise ValueError(f"Expected action_sequence to be (B,H,A), got {action_sequence.shape}")
        batch_size, horizon, _ = action_sequence.shape
        if horizon > self.max_horizon:
            raise ValueError(f"horizon {horizon} exceeds max_horizon={self.max_horizon}")
        action_tokens = self.action_proj(action_sequence)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_token, action_tokens], dim=1)
        tokens = tokens + self.pos_embedding[:, : tokens.shape[1], :]
        encoded = self.encoder(tokens)
        return self.norm(encoded[:, 0])


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_dim: int, mlp_hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(hidden_dim))
        self.fc_1 = nn.Linear(int(hidden_dim), int(mlp_hidden_dim) * 2)
        self.fc_2 = nn.Linear(int(mlp_hidden_dim), int(hidden_dim))
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        gate, value = self.fc_1(h).chunk(2, dim=-1)
        h = F.silu(gate) * value
        h = self.dropout(h)
        h = self.fc_2(h)
        h = self.dropout(h)
        return x + h


class TemporalPUDiscriminator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        num_tasks: int,
        latent_model_dim: int = 768,
        action_model_dim: int = 384,
        action_hidden_dim: int = 1536,
        action_num_layers: int = 4,
        action_num_heads: int = 8,
        trunk_hidden_dim: int = 1536,
        trunk_num_blocks: int = 6,
        head_hidden_dim: int = 1024,
        head_num_layers: int = 3,
        task_embed_dim: int = 128,
        max_action_horizon: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.num_tasks = int(num_tasks)
        self.max_action_horizon = int(max_action_horizon)

        self.latent_proj = nn.Sequential(
            nn.Linear(self.latent_dim, int(latent_model_dim)),
            nn.LayerNorm(int(latent_model_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.action_encoder = ActionSequenceEncoder(
            action_dim=self.action_dim,
            model_dim=int(action_model_dim),
            hidden_dim=int(action_hidden_dim),
            num_layers=int(action_num_layers),
            num_heads=int(action_num_heads),
            dropout=float(dropout),
            max_horizon=int(self.max_action_horizon),
        )
        self.task_embedding = nn.Embedding(int(self.num_tasks), int(task_embed_dim))
        fusion_input_dim = (
            int(self.latent_dim)
            + int(latent_model_dim)
            + int(action_model_dim)
            + int(task_embed_dim)
        )
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, int(trunk_hidden_dim)),
            nn.LayerNorm(int(trunk_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        block_hidden_dim = max(int(trunk_hidden_dim), int(2 * head_hidden_dim))
        self.backbone = nn.ModuleList(
            [
                ResidualMLPBlock(
                    hidden_dim=int(trunk_hidden_dim),
                    mlp_hidden_dim=int(block_hidden_dim),
                    dropout=float(dropout),
                )
                for _ in range(max(int(trunk_num_blocks), 1))
            ]
        )
        self.backbone_norm = nn.LayerNorm(int(trunk_hidden_dim))

        head_layers: list[nn.Module] = []
        in_dim = int(trunk_hidden_dim)
        for _ in range(max(int(head_num_layers) - 1, 0)):
            head_layers.extend(
                [
                    nn.Linear(in_dim, int(head_hidden_dim)),
                    nn.LayerNorm(int(head_hidden_dim)),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                ]
            )
            in_dim = int(head_hidden_dim)
        head_layers.append(nn.Linear(in_dim, 1))
        self.logit_head = nn.Sequential(*head_layers)

    def extract_feature(
        self,
        current_latent: torch.Tensor,
        action_sequence: torch.Tensor,
        task_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        latent_embed = self.latent_proj(current_latent)
        action_embed = self.action_encoder(action_sequence)
        if task_index is None:
            task_embed = torch.zeros(
                current_latent.shape[0],
                self.task_embedding.embedding_dim,
                dtype=current_latent.dtype,
                device=current_latent.device,
            )
        else:
            task_embed = self.task_embedding(task_index.long())
        fused = self.fusion(torch.cat([current_latent, latent_embed, action_embed, task_embed], dim=-1))
        for block in self.backbone:
            fused = block(fused)
        return self.backbone_norm(fused)

    def forward(
        self,
        current_latent: torch.Tensor,
        action_sequence: torch.Tensor,
        task_index: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        fused = self.extract_feature(
            current_latent=current_latent,
            action_sequence=action_sequence,
            task_index=task_index,
        )
        logits = self.logit_head(fused).squeeze(-1)
        probs = torch.sigmoid(logits)
        return {
            "logits": logits,
            "probs": probs,
            "features": fused,
        }


def build_temporal_pu_discriminator(
    *,
    latent_dim: int,
    action_dim: int,
    num_tasks: int,
    cfg_model: Any,
    transition_horizon: int,
) -> TemporalPUDiscriminator:
    max_action_horizon = max(
        int(_cfg_get(cfg_model, "max_action_horizon", transition_horizon)),
        int(transition_horizon),
    )
    return TemporalPUDiscriminator(
        latent_dim=int(latent_dim),
        action_dim=int(action_dim),
        num_tasks=int(num_tasks),
        latent_model_dim=int(_cfg_get(cfg_model, "latent_model_dim", 768)),
        action_model_dim=int(_cfg_get(cfg_model, "action_model_dim", 384)),
        action_hidden_dim=int(_cfg_get(cfg_model, "action_hidden_dim", 1536)),
        action_num_layers=int(_cfg_get(cfg_model, "action_num_layers", 4)),
        action_num_heads=int(_cfg_get(cfg_model, "action_num_heads", 8)),
        trunk_hidden_dim=int(_cfg_get(cfg_model, "trunk_hidden_dim", 1536)),
        trunk_num_blocks=int(_cfg_get(cfg_model, "trunk_num_blocks", 6)),
        head_hidden_dim=int(_cfg_get(cfg_model, "head_hidden_dim", 1024)),
        head_num_layers=int(_cfg_get(cfg_model, "head_num_layers", 3)),
        task_embed_dim=int(_cfg_get(cfg_model, "task_embed_dim", 128)),
        max_action_horizon=int(max_action_horizon),
        dropout=float(_cfg_get(cfg_model, "dropout", 0.1)),
    )


__all__ = [
    "TemporalPUDiscriminator",
    "build_temporal_pu_discriminator",
]
