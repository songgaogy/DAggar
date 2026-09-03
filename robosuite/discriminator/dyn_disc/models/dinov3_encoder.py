"""Frozen DINOv3 image encoder used by RPT cache generation and inference."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOv3Encoder(nn.Module):
    """CUDA-only DINOv3 patch-token mean pooling.

    The returned features are always float32, even though the backbone runs
    under BF16 autocast. CLS and register tokens are excluded from the mean.
    """

    def __init__(
        self,
        model_path: str,
        view_names: Sequence[str] = ("agentview", "robot0_eye_in_hand"),
        image_size: int = 224,
        expected_latent_dim: int = 768,
    ) -> None:
        super().__init__()
        from transformers import AutoModel

        path = self._resolve_path(model_path)
        self.model_path = str(path)
        self.view_names = tuple(str(name) for name in view_names)
        self.image_size = int(image_size)
        self.expected_latent_dim = int(expected_latent_dim)
        self.backbone = AutoModel.from_pretrained(str(path), local_files_only=True)
        self.latent_dim = int(self.backbone.config.hidden_size)
        self.num_register_tokens = int(getattr(self.backbone.config, "num_register_tokens", 0))
        if self.latent_dim != self.expected_latent_dim:
            raise ValueError(
                f"DINOv3 hidden size {self.latent_dim} != expected {self.expected_latent_dim}"
            )
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("image_mean", mean, persistent=False)
        self.register_buffer("image_std", std, persistent=False)

    @staticmethod
    def _resolve_path(model_path: str) -> Path:
        path = Path(model_path).expanduser()
        if path.exists():
            return path.resolve()
        if not path.is_absolute():
            for parent in (Path.cwd(), *Path.cwd().parents):
                candidate = parent / path
                if candidate.exists():
                    return candidate.resolve()
        raise FileNotFoundError(f"DINOv3 checkpoint path not found: {model_path}")

    def train(self, mode: bool = True):
        super().train(False)
        self.backbone.eval()
        return self

    def _require_cuda(self, images: torch.Tensor) -> None:
        if not images.is_cuda:
            raise RuntimeError("DINOv3Encoder requires CUDA image tensors")
        if next(self.backbone.parameters()).device != images.device:
            raise RuntimeError("DINOv3 backbone and images must be on the same CUDA device")

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected NCHW RGB images, got {tuple(images.shape)}")
        if images.dtype == torch.uint8:
            images = images.to(dtype=torch.float32).div_(255.0)
        else:
            images = images.to(dtype=torch.float32)
        if images.shape[-2:] != (self.image_size, self.image_size):
            images = F.interpolate(
                images,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        return (images.clamp_(0.0, 1.0) - self.image_mean) / self.image_std

    @torch.inference_mode()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encode ``[N,3,H,W]`` images into float32 ``[N,768]`` features."""
        self._require_cuda(images)
        pixels = self._preprocess(images)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = self.backbone(pixel_values=pixels)
            tokens = output.last_hidden_state
            patch_start = 1 + self.num_register_tokens
            if tokens.shape[1] <= patch_start:
                raise ValueError(
                    f"DINOv3 output has no patch tokens: shape={tuple(tokens.shape)}, "
                    f"register_tokens={self.num_register_tokens}"
                )
            features = tokens[:, patch_start:, :].mean(dim=1)
        return features.float()

    @torch.inference_mode()
    def forward(self, images_by_view: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Return ``[B,T,V,768]`` features in configured view order."""
        encoded = []
        batch_time = None
        for view_name in self.view_names:
            images = images_by_view[view_name]
            if images.ndim != 5:
                raise ValueError(f"Expected [B,T,3,H,W] for {view_name}, got {tuple(images.shape)}")
            b, t = images.shape[:2]
            if batch_time is None:
                batch_time = (b, t)
            elif batch_time != (b, t):
                raise ValueError("All camera views must have the same batch and time dimensions")
            encoded.append(self.encode_images(images.flatten(0, 1)).view(b, t, self.latent_dim))
        return torch.stack(encoded, dim=2)

    def manifest_config(self) -> dict:
        model_root = Path(self.model_path)
        tracked_files = []
        if model_root.is_dir():
            candidates = [model_root / "config.json"]
            candidates.extend(sorted(model_root.glob("*.safetensors")))
            candidates.extend(sorted(model_root.glob("pytorch_model*.bin")))
            for path in dict.fromkeys(candidates):
                if not path.is_file():
                    continue
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                        digest.update(chunk)
                stat = path.stat()
                tracked_files.append(
                    {
                        "path": path.relative_to(model_root).as_posix(),
                        "size": int(stat.st_size),
                        "mtime_ns": int(stat.st_mtime_ns),
                        "sha256": digest.hexdigest(),
                    }
                )
        if not tracked_files:
            raise RuntimeError(
                f"No DINOv3 config/weight files found for cache fingerprinting: {model_root}"
            )
        return {
            "model_path": self.model_path,
            "model_files": tracked_files,
            "image_size": self.image_size,
            "latent_dim": self.latent_dim,
            "pooling": "mean_patch_tokens_excluding_cls_and_registers",
            "inference_dtype": "bfloat16",
            "output_dtype": "float32",
        }
