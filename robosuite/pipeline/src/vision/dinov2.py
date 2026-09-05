from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from robosuite.pipeline.utils.tensor import resolve_cuda_device


class DinoV2Encoder(nn.Module):
    """Frozen DINOv2 ViT-B/14 encoder returning one CLS token per camera."""

    feature_dim = 768
    image_size = 224
    patch_size = 14

    def __init__(
        self,
        weights_path: str | Path,
        device: str | torch.device,
        *,
        hub_dir: str | Path | None = None,
        backbone_factory: Callable[[], nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.device = resolve_cuda_device(device)
        weights_path = Path(weights_path).expanduser().resolve()
        if not weights_path.is_file():
            raise FileNotFoundError(f"DINOv2 weights not found: {weights_path}")

        if backbone_factory is None:
            if hub_dir is None:
                hub_dir = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
            hub_dir = Path(hub_dir).expanduser().resolve()
            if not (hub_dir / "hubconf.py").is_file():
                raise FileNotFoundError(f"Local DINOv2 hub repository not found: {hub_dir}")
            backbone = torch.hub.load(
                str(hub_dir), "dinov2_vitb14", source="local", pretrained=False
            )
        else:
            backbone = backbone_factory()

        register_count = int(
            getattr(backbone, "num_register_tokens", getattr(backbone, "n_storage_tokens", 0))
        )
        if register_count != 0:
            raise ValueError(f"DINOv2 backbone must not use register tokens, got {register_count}")
        embed_dim = int(getattr(backbone, "embed_dim", self.feature_dim))
        if embed_dim != self.feature_dim:
            raise ValueError(f"Expected ViT-B feature dimension {self.feature_dim}, got {embed_dim}")
        backbone_patch_size = getattr(backbone, "patch_size", self.patch_size)
        if isinstance(backbone_patch_size, (tuple, list)):
            patch_shape = tuple(int(value) for value in backbone_patch_size)
            valid_patch = patch_shape == (self.patch_size, self.patch_size)
        else:
            valid_patch = int(backbone_patch_size) == self.patch_size
        if not valid_patch:
            raise ValueError(f"Expected ViT-B/14 patch size {self.patch_size}, got {backbone_patch_size}")

        raw_state = torch.load(weights_path, map_location="cpu", weights_only=True)
        if not isinstance(raw_state, dict):
            raise TypeError("DINOv2 checkpoint must contain a state dictionary")
        state = raw_state.get("model", raw_state)
        if not isinstance(state, dict):
            raise TypeError("DINOv2 checkpoint 'model' entry must be a state dictionary")
        backbone.load_state_dict(state, strict=True)

        self.backbone = backbone.to(self.device).eval()
        self.backbone.requires_grad_(False)
        self.register_buffer(
            "pixel_mean",
            torch.tensor((0.485, 0.456, 0.406), device=self.device).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor((0.229, 0.224, 0.225), device=self.device).view(1, 3, 1, 1),
            persistent=False,
        )
        self.eval()

    def train(self, mode: bool = True) -> "DinoV2Encoder":
        super().train(False)
        self.backbone.eval()
        return self

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        if images.device.type != "cuda":
            raise ValueError("DINOv2 image tensors must be on CUDA")
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError(f"Expected images [B,Cameras,3,H,W], got {tuple(images.shape)}")
        flat = images.flatten(0, 1)
        if flat.dtype == torch.uint8:
            flat = flat.to(dtype=torch.float32).div_(255.0)
        elif not flat.is_floating_point():
            raise TypeError(f"Expected uint8 or floating-point images, got {flat.dtype}")
        else:
            flat = flat.to(dtype=torch.float32)
        if flat.shape[-2:] != (self.image_size, self.image_size):
            flat = F.interpolate(
                flat,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        return (flat - self.pixel_mean) / self.pixel_std

    @torch.inference_mode()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch_size, camera_count = images.shape[:2]
        normalized = self._preprocess(images).contiguous(memory_format=torch.channels_last)
        features = self.backbone.forward_features(normalized)
        if isinstance(features, dict):
            cls = features.get("x_norm_clstoken")
            if cls is None:
                raise KeyError("DINOv2 forward_features output lacks 'x_norm_clstoken'")
        elif torch.is_tensor(features):
            cls = features[:, 0] if features.ndim == 3 else features
        else:
            raise TypeError(f"Unsupported DINOv2 output type: {type(features).__name__}")
        expected = (batch_size * camera_count, self.feature_dim)
        if tuple(cls.shape) != expected:
            raise ValueError(f"Expected DINOv2 CLS shape {expected}, got {tuple(cls.shape)}")
        return cls.reshape(batch_size, camera_count, self.feature_dim)
