from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def _cfg_get(cfg: Any, key: str, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class CLIPLanguageEncoder(nn.Module):
    def __init__(
        self,
        pretrained_name: str,
        feature_dim: int,
        max_length: int = 77,
        freeze_backbone: bool = True,
        cache_dir: str | None = None,
        local_files_only: bool = False,
    ):
        super().__init__()
        try:
            from transformers import AutoTokenizer, CLIPTextModel
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "transformers is required for flow_multi language encoding. "
                "Please install it before using the multimodal policy."
            ) from exc

        self.max_length = int(max_length)
        self.freeze_backbone = bool(freeze_backbone)
        load_kwargs = {
            "cache_dir": cache_dir,
            "local_files_only": bool(local_files_only),
        }
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name, **load_kwargs)
            self.backbone = CLIPTextModel.from_pretrained(pretrained_name, **load_kwargs)
        except OSError as exc:  # pragma: no cover
            raise OSError(
                f"Failed to load language encoder '{pretrained_name}'. "
                "Download the HuggingFace checkpoint first or set "
                "flow.language_encoder.pretrained_name to a local directory."
            ) from exc
        hidden_size = int(self.backbone.config.hidden_size)

        self.token_proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, feature_dim),
        )
        self.global_proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, feature_dim),
        )

        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _masked_mean(self, token_features: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        weights = attention_mask.to(dtype=token_features.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (token_features * weights).sum(dim=1) / denom

    def forward(self, texts: list[str] | tuple[str, ...] | str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(texts, str):
            texts = [texts]
        elif not isinstance(texts, (list, tuple)):
            raise TypeError(f"Unsupported language input type: {type(texts)!r}")

        device = next(self.parameters()).device
        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        if self.freeze_backbone:
            self.backbone.eval()
            with torch.no_grad():
                outputs = self.backbone(**encoded)
        else:
            outputs = self.backbone(**encoded)

        token_features = outputs.last_hidden_state
        pooled_output = getattr(outputs, "pooler_output", None)
        if pooled_output is None:
            pooled_output = self._masked_mean(token_features, encoded["attention_mask"])

        token_features = self.token_proj(token_features)
        pooled_output = self.global_proj(pooled_output)
        token_mask = encoded["attention_mask"].to(dtype=torch.bool)
        return token_features, pooled_output, token_mask


def build_language_encoder(cfg: Any, feature_dim: int) -> nn.Module:
    encoder_type = _cfg_get(cfg, "type", "clip_text")
    if encoder_type != "clip_text":
        raise ValueError(f"Unsupported language encoder type: {encoder_type}")
    return CLIPLanguageEncoder(
        pretrained_name=str(_cfg_get(cfg, "pretrained_name", "openai/clip-vit-base-patch32")),
        feature_dim=int(feature_dim),
        max_length=int(_cfg_get(cfg, "max_length", 77)),
        freeze_backbone=bool(_cfg_get(cfg, "freeze_backbone", True)),
        cache_dir=_cfg_get(cfg, "cache_dir", None),
        local_files_only=bool(_cfg_get(cfg, "local_files_only", False)),
    )
