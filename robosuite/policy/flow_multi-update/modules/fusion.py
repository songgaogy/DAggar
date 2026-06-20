from typing import Any

import math
import torch
import torch.nn as nn


def _cfg_get(cfg: Any, key: str, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class LanguageConditionedLayerNorm(nn.Module):
    def __init__(self, feature_dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.cond_proj = nn.Linear(cond_dim, feature_dim * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.cond_proj(cond).chunk(2, dim=-1)
        return self.norm(x) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class LanguageConditionedTransformerBlock(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_heads: int,
        dropout: float,
        use_language_adaln: bool,
    ):
        super().__init__()
        self.use_language_adaln = bool(use_language_adaln)
        if self.use_language_adaln:
            self.attn_norm = LanguageConditionedLayerNorm(feature_dim=feature_dim, cond_dim=feature_dim)
            self.ff_norm = LanguageConditionedLayerNorm(feature_dim=feature_dim, cond_dim=feature_dim)
        else:
            self.attn_norm = nn.LayerNorm(feature_dim)
            self.ff_norm = nn.LayerNorm(feature_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def _norm(self, norm_module, tokens: torch.Tensor, language_global: torch.Tensor) -> torch.Tensor:
        if self.use_language_adaln:
            return norm_module(tokens, language_global)
        return norm_module(tokens)

    def forward(
        self,
        tokens: torch.Tensor,
        language_global: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        attn_input = self._norm(self.attn_norm, tokens, language_global)
        attn_output, _ = self.attn(
            attn_input,
            attn_input,
            attn_input,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        tokens = tokens + self.dropout(attn_output)

        ff_input = self._norm(self.ff_norm, tokens, language_global)
        tokens = tokens + self.dropout(self.ff(ff_input))
        return tokens


class MultiModalTransformerFusion(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_image_tokens: int,
        num_proprio_tokens: int,
        max_language_tokens: int,
        depth: int,
        num_heads: int,
        dropout: float = 0.0,
        use_language_adaln: bool = True,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_image_tokens = int(num_image_tokens)
        self.num_proprio_tokens = int(num_proprio_tokens)
        self.max_language_tokens = int(max_language_tokens)
        self.max_total_tokens = self.max_language_tokens + self.num_proprio_tokens + self.num_image_tokens

        self.language_type_embedding = nn.Parameter(torch.zeros(1, 1, self.feature_dim))
        self.proprio_type_embedding = nn.Parameter(torch.zeros(1, 1, self.feature_dim))
        self.image_type_embedding = nn.Parameter(torch.zeros(1, 1, self.feature_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, self.max_total_tokens, self.feature_dim))
        self.blocks = nn.ModuleList(
            [
                LanguageConditionedTransformerBlock(
                    feature_dim=self.feature_dim,
                    num_heads=int(num_heads),
                    dropout=float(dropout),
                    use_language_adaln=bool(use_language_adaln),
                )
                for _ in range(int(depth))
            ]
        )
        self.output_norm = nn.LayerNorm(feature_dim)

    def forward(
        self,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        proprio_tokens: torch.Tensor,
        image_tokens: torch.Tensor,
        language_global: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if image_tokens.shape[1] != self.num_image_tokens:
            raise ValueError(f"Expected {self.num_image_tokens} image tokens, got {image_tokens.shape[1]}")
        if proprio_tokens.shape[1] != self.num_proprio_tokens:
            raise ValueError(f"Expected {self.num_proprio_tokens} proprio tokens, got {proprio_tokens.shape[1]}")
        if language_tokens.shape[1] > self.max_language_tokens:
            raise ValueError(
                f"Language sequence length {language_tokens.shape[1]} exceeds max_language_tokens={self.max_language_tokens}"
            )

        tokens = torch.cat(
            [
                language_tokens + self.language_type_embedding,
                proprio_tokens + self.proprio_type_embedding,
                image_tokens + self.image_type_embedding,
            ],
            dim=1,
        )
        tokens = tokens + self.position_embedding[:, : tokens.shape[1], :]

        batch_size = tokens.shape[0]
        proprio_mask = torch.zeros(
            (batch_size, self.num_proprio_tokens),
            device=tokens.device,
            dtype=torch.bool,
        )
        image_mask = torch.zeros(
            (batch_size, self.num_image_tokens),
            device=tokens.device,
            dtype=torch.bool,
        )
        key_padding_mask = torch.cat([~language_mask, proprio_mask, image_mask], dim=1)

        fused = tokens
        for block in self.blocks:
            fused = block(fused, language_global=language_global, key_padding_mask=key_padding_mask)
        return self.output_norm(fused), key_padding_mask


class AttentionConditionAggregator(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        self.output_dim = int(output_dim)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(feature_dim * 3),
            nn.Linear(feature_dim * 3, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, self.output_dim),
        )

    def forward(
        self,
        fused_tokens: torch.Tensor,
        token_padding_mask: torch.Tensor | None,
        language_global: torch.Tensor,
    ) -> torch.Tensor:
        query = self.query_proj(language_global).unsqueeze(1)
        keys = self.key_proj(fused_tokens)
        scores = torch.sum(query * keys, dim=-1) / math.sqrt(float(fused_tokens.shape[-1]))
        if token_padding_mask is not None:
            scores = scores.masked_fill(token_padding_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        pooled = torch.sum(weights * self.value_proj(fused_tokens), dim=1)

        if token_padding_mask is not None:
            valid_mask = (~token_padding_mask).to(dtype=fused_tokens.dtype).unsqueeze(-1)
            mean_tokens = torch.sum(fused_tokens * valid_mask, dim=1) / valid_mask.sum(dim=1).clamp_min(1.0)
        else:
            mean_tokens = fused_tokens.mean(dim=1)

        return self.output_proj(torch.cat([pooled, mean_tokens, language_global], dim=-1))


def build_fusion_module(cfg: Any, num_image_tokens: int, num_proprio_tokens: int) -> nn.Module:
    fusion_type = _cfg_get(cfg, "type", "transformer")
    if fusion_type != "transformer":
        raise ValueError(f"Unsupported fusion type: {fusion_type}")
    
    return MultiModalTransformerFusion(
        feature_dim=int(_cfg_get(cfg, "feature_dim")),
        num_image_tokens=int(num_image_tokens),
        num_proprio_tokens=int(num_proprio_tokens),
        max_language_tokens=int(_cfg_get(cfg, "max_language_tokens", 77)),
        depth=int(_cfg_get(cfg, "depth", 3)),
        num_heads=int(_cfg_get(cfg, "num_heads", 8)),
        dropout=float(_cfg_get(cfg, "dropout", 0.0)),
        use_language_adaln=bool(_cfg_get(cfg, "use_language_adaln", True)),
    )


def build_condition_aggregator(cfg: Any, feature_dim: int) -> nn.Module:
    aggregator_type = _cfg_get(cfg, "type", "attention_pool")
    if aggregator_type != "attention_pool":
        raise ValueError(f"Unsupported condition aggregator type: {aggregator_type}")
    
    return AttentionConditionAggregator(
        feature_dim=int(feature_dim),
        hidden_dim=int(_cfg_get(cfg, "hidden_dim", feature_dim * 2)),
        output_dim=int(_cfg_get(cfg, "output_dim", feature_dim)),
    )
