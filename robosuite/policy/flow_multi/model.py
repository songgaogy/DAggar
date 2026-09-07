from typing import Any

import torch
import torch.nn as nn

from robosuite.policy.flow_multi.modules.encoders import build_image_encoder, build_proprio_tokenizer
from robosuite.policy.flow_multi.modules.fusion import (
    build_condition_aggregator,
    build_fusion_module,
)
from robosuite.policy.flow_multi.modules.heads import build_flow_head
from robosuite.policy.flow_multi.modules.language import build_language_encoder
from robosuite.policy.flow_multi.modules.modulation import build_language_guided_modulation


def _cfg_get(cfg: Any, key: str, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class MultiModalFlowPolicy(nn.Module):
    def __init__(
        self,
        camera_names: list[str],
        proprio_dim: int,
        action_dim: int,
        image_encoder_cfg: Any,
        proprio_encoder_cfg: Any,
        language_encoder_cfg: Any,
        modulation_cfg: Any,
        fusion_cfg: Any,
        aggregator_cfg: Any,
        head_cfg: Any,
    ):
        super().__init__()
        self.camera_names = list(camera_names)
        self.action_dim = int(action_dim)
        self.feature_dim = int(_cfg_get(fusion_cfg, "feature_dim"))
        image_feature_dim = int(_cfg_get(image_encoder_cfg, "feature_dim"))
        if image_feature_dim != self.feature_dim:
            raise ValueError(
                "image_encoder.feature_dim must match fusion.feature_dim, "
                f"got {image_feature_dim} vs {self.feature_dim}"
            )

        self.image_encoder = build_image_encoder(image_encoder_cfg)
        self.proprio_tokenizer = build_proprio_tokenizer(
            proprio_encoder_cfg,
            proprio_dim=proprio_dim,
            feature_dim=self.feature_dim,
        )
        self.language_encoder = build_language_encoder(language_encoder_cfg, feature_dim=self.feature_dim)

        # apply FiLM on vision and proprio, conditioned on language
        self.language_guided_modulation = build_language_guided_modulation(modulation_cfg, feature_dim=self.feature_dim)

        # transformer
        # detail: concat modalities (add learnable + pos_ebd) -> LanguageConditionedTransformerBlock(language conditioned LN)
        self.fusion = build_fusion_module(
            fusion_cfg,
            num_image_tokens=len(self.camera_names) * int(self.image_encoder.num_tokens),
            num_proprio_tokens=int(self.proprio_tokenizer.num_tokens),
        )

        # transformer: reduce dimension
        self.condition_aggregator = build_condition_aggregator(aggregator_cfg, feature_dim=self.feature_dim)

        # 1D flow policy head
        self.flow_head = build_flow_head(
            head_cfg,
            action_dim=self.action_dim,
            context_dim=int(self.condition_aggregator.output_dim),
            token_context_dim=self.feature_dim,
        )

    def encode_multimodal_context(
        self,
        images: torch.Tensor,
        proprio: torch.Tensor,
        language: list[str] | tuple[str, ...] | str,
    ) -> dict[str, torch.Tensor]:
        batch_size, num_cameras, channels, height, width = images.shape
        if num_cameras != len(self.camera_names):
            raise ValueError(f"Expected {len(self.camera_names)} cameras, got {num_cameras}")

        flat_images = images.reshape(batch_size * num_cameras, channels, height, width)
        if flat_images.device.type == "cuda":
            flat_images = flat_images.contiguous(memory_format=torch.channels_last)
        
        # pass through pretrained model
        image_tokens = self.image_encoder(flat_images)
        image_tokens = image_tokens.reshape(batch_size, num_cameras * image_tokens.shape[1], image_tokens.shape[2])
        base_image_tokens = image_tokens
        proprio_tokens = self.proprio_tokenizer(proprio)
        language_tokens, language_global, language_mask = self.language_encoder(language)

        # language guided pre-processing
        image_tokens, proprio_tokens = self.language_guided_modulation(
            visual_tokens=image_tokens,
            proprio_tokens=proprio_tokens,
            language_global=language_global,
        )

        # language guided transformer
        fused_tokens, token_padding_mask = self.fusion(
            language_tokens=language_tokens,
            language_mask=language_mask,
            proprio_tokens=proprio_tokens,
            image_tokens=image_tokens,
            language_global=language_global,
        )

        # condense all the information into one token for flow head
        task_scene_cond = self.condition_aggregator(
            fused_tokens=fused_tokens,
            token_padding_mask=token_padding_mask,
            language_global=language_global,
        )

        context_tokens = torch.cat([language_tokens, fused_tokens], dim=1)      # add language again
        context_padding_mask = torch.cat([~language_mask, token_padding_mask], dim=1)

        return {
            "task_scene_cond": task_scene_cond,
            "context_tokens": context_tokens,
            "context_padding_mask": context_padding_mask,
            "image_tokens": base_image_tokens,
        }

    def encode_context(
        self,
        images: torch.Tensor,
        proprio: torch.Tensor,
        language: list[str] | tuple[str, ...] | str,
    ) -> torch.Tensor:
        return self.encode_multimodal_context(images=images, proprio=proprio, language=language)["task_scene_cond"]

    def get_cond_features(
        self,
        images: torch.Tensor,
        proprio: torch.Tensor,
        language: list[str] | tuple[str, ...] | str,
    ) -> torch.Tensor:
        return self.encode_context(images=images, proprio=proprio, language=language)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        images: torch.Tensor,
        proprio: torch.Tensor,
        language: list[str] | tuple[str, ...] | str,
    ) -> torch.Tensor:
        context = self.encode_multimodal_context(images=images, proprio=proprio, language=language)
        return self.flow_head(
            x_t=x_t,
            timesteps=t,
            task_scene_cond=context["task_scene_cond"],
            context_tokens=context["context_tokens"],
            context_padding_mask=context["context_padding_mask"],
        )


def build_flow_policy(cfg: Any, proprio_dim: int, action_dim: int, camera_names: list[str]) -> MultiModalFlowPolicy:
    return MultiModalFlowPolicy(
        camera_names=camera_names,
        proprio_dim=proprio_dim,
        action_dim=action_dim,
        image_encoder_cfg=_cfg_get(cfg, "image_encoder"),
        proprio_encoder_cfg=_cfg_get(cfg, "proprio_encoder"),
        language_encoder_cfg=_cfg_get(cfg, "language_encoder"),
        modulation_cfg=_cfg_get(cfg, "modulation"),
        fusion_cfg=_cfg_get(cfg, "fusion"),
        aggregator_cfg=_cfg_get(cfg, "aggregator"),
        head_cfg=_cfg_get(cfg, "head"),
    )
