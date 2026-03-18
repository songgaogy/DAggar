from typing import Any

import torch
import torch.nn as nn


def _cfg_get(cfg: Any, key: str, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class LanguageGuidedTokenModulator(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, feature_dim * 3),
        )

    def forward(self, tokens: torch.Tensor, language_global: torch.Tensor) -> torch.Tensor:
        scale, shift, gate = self.cond_proj(language_global).chunk(3, dim=-1)
        modulated = self.norm(tokens)
        modulated = modulated * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return tokens + torch.tanh(gate).unsqueeze(1) * modulated


class LanguageGuidedModulation(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.visual_modulator = LanguageGuidedTokenModulator(feature_dim=feature_dim, hidden_dim=hidden_dim)
        self.proprio_modulator = LanguageGuidedTokenModulator(feature_dim=feature_dim, hidden_dim=hidden_dim)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        proprio_tokens: torch.Tensor,
        language_global: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.visual_modulator(visual_tokens, language_global),
            self.proprio_modulator(proprio_tokens, language_global),
        )


def build_language_guided_modulation(cfg: Any, feature_dim: int) -> nn.Module:
    modulation_type = _cfg_get(cfg, "type", "film")
    if modulation_type != "film":
        raise ValueError(f"Unsupported modulation type: {modulation_type}")
    return LanguageGuidedModulation(
        feature_dim=feature_dim,
        hidden_dim=int(_cfg_get(cfg, "hidden_dim", feature_dim * 2)),
    )
