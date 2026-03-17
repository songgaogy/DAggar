from typing import Any

import torch
import torch.nn as nn

from robosuite.policy.flow_unet.encoders import build_image_encoder, build_proprio_projector
from robosuite.policy.flow_unet.fusion import build_fusion_module
from robosuite.policy.flow_unet.heads import build_flow_head


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
        fusion_cfg: Any,
        head_cfg: Any,
    ):
        super().__init__()
        self.camera_names = list(camera_names)
        self.action_dim = int(action_dim)
        self.feature_dim = int(_cfg_get(fusion_cfg, "feature_dim"))

        self.image_encoder = build_image_encoder(image_encoder_cfg)
        self.proprio_projector = build_proprio_projector(proprio_encoder_cfg, proprio_dim=proprio_dim, feature_dim=self.feature_dim)
        self.fusion = build_fusion_module(fusion_cfg, num_modalities=len(self.camera_names) + 1)
        self.flow_head = build_flow_head(head_cfg, action_dim=self.action_dim, context_dim=self.feature_dim)

    def encode_context(self, images: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        batch_size, num_cameras, channels, height, width = images.shape
        if num_cameras != len(self.camera_names):
            raise ValueError(f"Expected {len(self.camera_names)} cameras, got {num_cameras}")

        flat_images = images.reshape(batch_size * num_cameras, channels, height, width)
        if flat_images.device.type == "cuda":
            flat_images = flat_images.contiguous(memory_format=torch.channels_last)
        image_features = self.image_encoder(flat_images).reshape(batch_size, num_cameras, -1)
        proprio_feature = self.proprio_projector(proprio)
        return self.fusion(image_features, proprio_feature)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, images: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        fused_context = self.encode_context(images, proprio)
        return self.flow_head(x_t=x_t, timesteps=t, context=fused_context)


def build_flow_policy(cfg: Any, proprio_dim: int, action_dim: int, camera_names: list[str]) -> MultiModalFlowPolicy:
    return MultiModalFlowPolicy(
        camera_names=camera_names,
        proprio_dim=proprio_dim,
        action_dim=action_dim,
        image_encoder_cfg=_cfg_get(cfg, "image_encoder"),
        proprio_encoder_cfg=_cfg_get(cfg, "proprio_encoder"),
        fusion_cfg=_cfg_get(cfg, "fusion"),
        head_cfg=_cfg_get(cfg, "head"),
    )
