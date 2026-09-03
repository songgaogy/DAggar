from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class DINOv3Encoder(nn.Module):
    def __init__(
        self,
        model_path: str,
        view_names: Sequence[str],
        emb_dim: int = 384,
        pooled_grid_size: int = 4,
        freeze_backbone: bool = True,
        train_projection: bool = True,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        from transformers import AutoModel

        self.model_path = str(model_path)
        self.view_names = list(view_names)
        self.emb_dim = int(emb_dim)
        self.pooled_grid_size = int(pooled_grid_size)
        self.num_patches = self.pooled_grid_size * self.pooled_grid_size
        self.freeze_backbone = bool(freeze_backbone)
        self.train_projection = bool(train_projection)
        self.image_size = int(image_size)
        self.latent_ndim = 4
        self.name = "dinov3"
        self.normalizes_images = True

        path = Path(self.model_path)
        if not path.exists() and not path.is_absolute():
            for parent in [Path.cwd()] + list(Path.cwd().parents):
                candidate = parent / path
                if candidate.exists():
                    path = candidate
                    break
        if not path.exists():
            raise FileNotFoundError(f"DINOv3 checkpoint path not found: {path}")

        self.backbone = AutoModel.from_pretrained(str(path), local_files_only=True)
        hidden_size = int(self.backbone.config.hidden_size)
        self.patch_size = int(self.backbone.config.patch_size)
        self.num_register_tokens = int(getattr(self.backbone.config, "num_register_tokens", 0))
        self.proj = nn.Linear(hidden_size, self.emb_dim)

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("image_mean", mean, persistent=False)
        self.register_buffer("image_std", std, persistent=False)
        self.set_trainable(train_backbone=not self.freeze_backbone, train_projection=self.train_projection)

    def set_trainable(self, train_backbone: bool, train_projection: bool | None = None) -> None:
        if train_projection is None:
            train_projection = self.train_projection
        for p in self.backbone.parameters():
            p.requires_grad = bool(train_backbone)
        for p in self.proj.parameters():
            p.requires_grad = bool(train_projection)
        self.freeze_backbone = not bool(train_backbone)
        self.train_projection = bool(train_projection)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _preprocess(self, imgs: torch.Tensor) -> torch.Tensor:
        if imgs.shape[-2:] != (self.image_size, self.image_size):
            imgs = F.interpolate(
                imgs,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        imgs = imgs.clamp(0.0, 1.0)
        return (imgs - self.image_mean) / self.image_std

    def _encode_flat(self, imgs: torch.Tensor) -> torch.Tensor:
        imgs = self._preprocess(imgs)
        with torch.set_grad_enabled(not self.freeze_backbone):
            out = self.backbone(pixel_values=imgs)
        tokens = out.last_hidden_state
        patch_start = 1 + self.num_register_tokens
        patch_tokens = tokens[:, patch_start:, :]
        grid_size = int(round(patch_tokens.shape[1] ** 0.5))
        if grid_size * grid_size != int(patch_tokens.shape[1]):
            raise ValueError(f"DINOv3 patch token count is not square: {patch_tokens.shape[1]}")
        patch_grid = rearrange(patch_tokens, "b (h w) d -> b d h w", h=grid_size, w=grid_size)
        pooled = F.adaptive_avg_pool2d(patch_grid, (self.pooled_grid_size, self.pooled_grid_size))
        pooled = rearrange(pooled, "b d h w -> b (h w) d")
        return self.proj(pooled)

    def forward(self, x):
        view_embs = {}
        for view_name in self.view_names:
            imgs = x[view_name]
            b = imgs.shape[0]
            imgs = rearrange(imgs, "b t c h w -> (b t) c h w")
            imgs_emb = self._encode_flat(imgs)
            imgs_emb = rearrange(imgs_emb, "(b t) p d -> b t p d", b=b)
            view_embs[view_name] = imgs_emb
        return view_embs
