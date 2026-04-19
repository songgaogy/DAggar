from __future__ import annotations

import os
from contextlib import nullcontext
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
            from transformers import AutoConfig, AutoTokenizer, CLIPTextModel
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "transformers is required for flow_multi language encoding. "
                "Please install it before using the multimodal policy."
            ) from exc

        self.max_length = int(max_length)
        self.freeze_backbone = bool(freeze_backbone)
        self._tokenized_text_cache: dict[str, dict[str, torch.Tensor]] = {}
        load_kwargs = {
            "cache_dir": cache_dir,
            "local_files_only": bool(local_files_only),
        }
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name, **load_kwargs)
            config = AutoConfig.from_pretrained(pretrained_name, **load_kwargs)
            if self._is_full_clip_config(config):
                self.backbone = self._load_text_backbone_from_full_clip_checkpoint(
                    pretrained_name=pretrained_name,
                    config=config,
                )
            else:
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

    def _is_full_clip_config(self, config: Any) -> bool:
        architectures = getattr(config, "architectures", None) or []
        if "CLIPModel" in architectures:
            return True
        return hasattr(config, "vision_config")

    def _load_text_backbone_from_full_clip_checkpoint(self, pretrained_name: str, config: Any) -> nn.Module:
        from transformers import CLIPTextModel

        text_model = CLIPTextModel(config.text_config)
        weights_path = os.path.join(pretrained_name, "pytorch_model.bin")
        if not os.path.exists(weights_path):
            raise OSError(
                f"Expected CLIP checkpoint weights at '{weights_path}', but the file was not found."
            )

        full_state_dict = torch.load(weights_path, map_location="cpu")
        text_state_dict = {}
        prefix = "text_model."
        for key, value in full_state_dict.items():
            if not key.startswith(prefix):
                continue
            if key.endswith("position_ids"):
                continue
            text_state_dict[key] = value

        incompatible = text_model.load_state_dict(text_state_dict, strict=False)
        unexpected_keys = [key for key in incompatible.unexpected_keys if not key.endswith("position_ids")]
        missing_keys = [key for key in incompatible.missing_keys if not key.endswith("position_ids")]
        if len(unexpected_keys) > 0 or len(missing_keys) > 0:
            raise OSError(
                "Failed to load CLIP text weights cleanly. "
                f"Missing keys: {missing_keys}. Unexpected keys: {unexpected_keys}."
            )
        return text_model

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _masked_mean(self, token_features: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        weights = attention_mask.to(dtype=token_features.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (token_features * weights).sum(dim=1) / denom

    def _tokenize_cached(self, texts: list[str]) -> dict[str, torch.Tensor]:
        unique_texts = list(dict.fromkeys(str(text) for text in texts))
        missing_texts = [text for text in unique_texts if text not in self._tokenized_text_cache]
        if missing_texts:
            encoded_missing = self.tokenizer(
                missing_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            for row, text in enumerate(missing_texts):
                self._tokenized_text_cache[text] = {
                    key: value[row].detach().cpu().clone()
                    for key, value in encoded_missing.items()
                }

        encoded_rows = [self._tokenized_text_cache[str(text)] for text in texts]
        keys = tuple(encoded_rows[0].keys())
        return {
            key: torch.stack([row[key] for row in encoded_rows], dim=0)
            for key in keys
        }

    def forward(
        self,
        texts: list[str] | tuple[str, ...] | str,
        profiler=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(texts, str):
            texts = [texts]
        elif not isinstance(texts, (list, tuple)):
            raise TypeError(f"Unsupported language input type: {type(texts)!r}")

        unique_texts: list[str] = []
        inverse_indices: list[int] = []
        text_to_unique_idx: dict[str, int] = {}
        for text in (str(item) for item in texts):
            unique_idx = text_to_unique_idx.get(text)
            if unique_idx is None:
                unique_idx = len(unique_texts)
                text_to_unique_idx[text] = unique_idx
                unique_texts.append(text)
            inverse_indices.append(unique_idx)

        # DataParallel replicas can make a generic `next(self.parameters())`
        # lookup brittle on some replicas. The projection layers are always
        # present, so use them directly to resolve the target device.
        device = self.token_proj[1].weight.device
        with (profiler.section("lang_tokenizer") if profiler is not None else nullcontext()):
            encoded = self._tokenize_cached(unique_texts)
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with (profiler.section("lang_backbone") if profiler is not None else nullcontext()):
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

        with (profiler.section("lang_proj") if profiler is not None else nullcontext()):
            token_features = self.token_proj(token_features)
            pooled_output = self.global_proj(pooled_output)
        token_mask = encoded["attention_mask"].to(dtype=torch.bool)
        if len(unique_texts) != len(texts):
            inverse_index = torch.tensor(inverse_indices, dtype=torch.long, device=token_features.device)
            token_features = token_features.index_select(0, inverse_index)
            pooled_output = pooled_output.index_select(0, inverse_index)
            token_mask = token_mask.index_select(0, inverse_index)
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
