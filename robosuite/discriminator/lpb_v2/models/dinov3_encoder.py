from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from einops import rearrange
from transformers import AutoModel


class DINOv3Encoder(torch.nn.Module):
    def __init__(
        self,
        policy_ckpt_path: Optional[str],
        view_names: Sequence[str],
        pretrained_path: str = "data/pretrained/dinov3-vitl16-pretrain-lvd1689m",
    ):
        super().__init__()
        self.policy_ckpt_path = policy_ckpt_path
        self.view_names = list(view_names)
        self.name = "dinov3"
        self.latent_ndim = 2

        if self.policy_ckpt_path:
            raise ValueError(
                "lpb_v2 does not support diffusion-policy policy_ckpt_path. "
                "Set policy_ckpt_path/env.policy_ckpt_path to null, or use lpb_original."
            )

        ckpt_dir = self._resolve_pretrained_path(pretrained_path)
        self.model = AutoModel.from_pretrained(str(ckpt_dir), local_files_only=True)
        self.emb_dim = int(self.model.config.hidden_size)
        self.image_size = int(getattr(self.model.config, "image_size", 224))

        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    @staticmethod
    def _resolve_pretrained_path(pretrained_path: str) -> Path:
        path = Path(pretrained_path).expanduser()
        if path.is_dir():
            return path
        here = Path(__file__).resolve()
        for parent in here.parents:
            candidate = parent / pretrained_path
            if candidate.is_dir():
                return candidate
        raise FileNotFoundError(f"DINO-v3 checkpoint directory not found: {pretrained_path}")

    def _preprocess(self, imgs: torch.Tensor) -> torch.Tensor:
        # LPB normalizer maps images from [0, 1] to [-1, 1]; DINO-v3 expects ImageNet-normalized [0, 1].
        imgs = (imgs + 1.0) * 0.5
        imgs = imgs.clamp(0.0, 1.0)
        if imgs.shape[-2:] != (self.image_size, self.image_size):
            imgs = F.interpolate(
                imgs,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        return (imgs - self.image_mean.to(dtype=imgs.dtype)) / self.image_std.to(dtype=imgs.dtype)

    def forward(self, x):
        view_embs = {}
        for view_name in self.view_names:
            imgs = x[view_name]
            b = imgs.shape[0]
            imgs = rearrange(imgs, "b t ... -> (b t) ...")
            imgs = self._preprocess(imgs)
            outputs = self.model(pixel_values=imgs)

            # NOTE: get CLS token
            imgs_emb = getattr(outputs, "pooler_output", None)
            if imgs_emb is None:
                imgs_emb = outputs.last_hidden_state[:, 0]
            imgs_emb = imgs_emb.unsqueeze(1)
            imgs_emb = rearrange(imgs_emb, "(b t) p d -> b t p d", b=b)
            view_embs[view_name] = imgs_emb
        return view_embs
