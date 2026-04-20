"""Pretrained DINOv2 ViT-B/14 per-frame image encoder for FLOAT."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torchvision.transforms import Normalize


DEFAULT_HUB_REPO = "facebookresearch/dinov2"
DEFAULT_HUB_MODEL = "dinov2_vitb14"
DEFAULT_IMAGE_SIZE = 224          # multiple of 14
DEFAULT_BATCH_SIZE = 64
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _center_crop_resize(img: np.ndarray, out_size: int) -> np.ndarray:
    """Center-crop to square then bilinear-ish resize to (out_size, out_size, 3)."""
    h, w = int(img.shape[0]), int(img.shape[1])
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    crop = img[y0 : y0 + side, x0 : x0 + side, :]
    if side == out_size:
        return crop
    ys = np.linspace(0, side - 1, out_size).astype(np.int32)
    xs = np.linspace(0, side - 1, out_size).astype(np.int32)
    return crop[ys][:, xs]


class DinoV2ImageEncoder:
    """DINOv2 ViT-B/14 image encoder returning per-frame CLS embeddings.

    Loads weights via ``torch.hub.load(DEFAULT_HUB_REPO, DEFAULT_HUB_MODEL)``
    (cached under ``~/.cache/torch/hub`` after first download).
    """

    def __init__(
        self,
        device: str = "cuda",
        image_size: int = DEFAULT_IMAGE_SIZE,
        batch_size: int = DEFAULT_BATCH_SIZE,
        hub_repo: str = DEFAULT_HUB_REPO,
        hub_model: str = DEFAULT_HUB_MODEL,
    ) -> None:
        if int(image_size) % 14 != 0:
            raise ValueError(
                f"DINOv2 ViT-B/14 requires image_size divisible by 14, got {image_size}"
            )
        if int(batch_size) <= 0:
            raise ValueError(f"batch_size must be >=1, got {batch_size}")

        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.batch_size = int(batch_size)
        self.hub_repo = str(hub_repo)
        self.hub_model = str(hub_model)

        # Trust repo download; cached across runs.
        self.model = torch.hub.load(self.hub_repo, self.hub_model, pretrained=True)
        self.model.to(self.device)
        self.model.eval()

        self.normalize = Normalize(mean=list(_IMAGENET_MEAN), std=list(_IMAGENET_STD))

        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.image_size, self.image_size, device=self.device)
            out = self.model(dummy)
        if out.ndim != 2:
            raise RuntimeError(
                f"Unexpected DINOv2 output rank {out.ndim}; expected 2 (B, E)."
            )
        self.embedding_dim = int(out.shape[1])

    def close(self) -> None:
        if getattr(self, "model", None) is not None:
            try:
                del self.model
            finally:
                self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @torch.no_grad()
    def encode_images(self, images_thwc: np.ndarray) -> np.ndarray:
        """Encode an image sequence into per-frame embeddings.

        Args:
            images_thwc: (T, H, W, 3) uint8 array. H/W can be any size.

        Returns:
            (T, embedding_dim) float32 array.
        """
        arr = np.asarray(images_thwc)
        if arr.ndim != 4 or arr.shape[-1] != 3:
            raise ValueError(
                f"images_thwc must have shape (T, H, W, 3), got {arr.shape}"
            )
        if arr.shape[0] == 0:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        t = int(arr.shape[0])
        # Preprocess: center crop + resize + to CHW float32/255.
        pre = np.empty((t, 3, self.image_size, self.image_size), dtype=np.float32)
        for i in range(t):
            resized = _center_crop_resize(arr[i], self.image_size)
            pre[i] = np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1))

        outputs: list[np.ndarray] = []
        for start in range(0, t, self.batch_size):
            stop = min(t, start + self.batch_size)
            batch = torch.from_numpy(pre[start:stop]).to(self.device)
            batch = self.normalize(batch)
            emb = self.model(batch)
            outputs.append(emb.detach().cpu().numpy().astype(np.float32))

        return np.concatenate(outputs, axis=0)
