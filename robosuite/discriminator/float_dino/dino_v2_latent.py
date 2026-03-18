from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Normalize

from .float_data import PolicyTrajectory

_DIFFUSION_POLICY_ROOT = Path(__file__).resolve().parents[2] / "armada" / "diffusion_policy"
if str(_DIFFUSION_POLICY_ROOT) not in sys.path:
    sys.path.insert(0, str(_DIFFUSION_POLICY_ROOT))

from diffusion_policy.model.vision.dinov2_vit import vit_base, vit_large, vit_small


@dataclass
class DinoV2BuildResult:
    model: torch.nn.Module
    embed_dim: int
    loaded_params: int


def _torch_load_checkpoint(path: str, map_location: str = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _center_crop_resize(img: np.ndarray, out_size: int) -> np.ndarray:
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


def _is_tensor_dict(x: Any) -> bool:
    if not isinstance(x, dict) or len(x) == 0:
        return False
    return all(torch.is_tensor(v) for v in x.values())


def _iter_state_dict_candidates(state: Any) -> Iterable[dict[str, torch.Tensor]]:
    if _is_tensor_dict(state):
        yield dict(state)
    if isinstance(state, dict):
        for key in ("state_dict", "model", "teacher", "student", "backbone", "encoder"):
            sub = state.get(key)
            if _is_tensor_dict(sub):
                yield dict(sub)


def _strip_prefix(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            out[key[len(prefix) :]] = value
        else:
            out[key] = value
    return out


def _candidate_key_variants(state_dict: dict[str, torch.Tensor]) -> Iterable[dict[str, torch.Tensor]]:
    prefixes = ("module.", "model.", "backbone.", "encoder.", "teacher.", "student.")
    seen = set()
    queue = [dict(state_dict)]

    while queue:
        cur = queue.pop(0)
        sig = tuple(sorted(cur.keys()))
        if sig in seen:
            continue
        seen.add(sig)
        yield cur
        for prefix in prefixes:
            if any(key.startswith(prefix) for key in cur.keys()):
                queue.append(_strip_prefix(cur, prefix))


def build_dino_v2_backbone(
    pretrained_path: str,
    model_name: str,
    image_size: int,
    device: str,
    patch_size: int = 14,
    num_register_tokens: int = 4,
) -> DinoV2BuildResult:
    factories = {
        "vit_small": vit_small,
        "vit_base": vit_base,
        "vit_large": vit_large,
    }
    name = str(model_name).lower()
    if name not in factories:
        raise ValueError(f"Unsupported DINOv2 model_name='{model_name}'. Supported: {sorted(factories.keys())}")

    model = factories[name](
        img_size=int(image_size),
        patch_size=int(patch_size),
        init_values=1.0,
        ffn_layer="mlp",
        block_chunks=0,
        num_register_tokens=int(num_register_tokens),
        interpolate_antialias=True,
        interpolate_offset=0.0,
    )

    loaded_params = 0
    if pretrained_path:
        payload = _torch_load_checkpoint(pretrained_path, map_location="cpu")
        own = model.state_dict()
        best_filtered: Optional[dict[str, torch.Tensor]] = None

        for candidate in _iter_state_dict_candidates(payload):
            for variant in _candidate_key_variants(candidate):
                filtered = {k: v for k, v in variant.items() if k in own and own[k].shape == v.shape}
                if best_filtered is None or len(filtered) > len(best_filtered):
                    best_filtered = filtered

        if best_filtered is None or len(best_filtered) == 0:
            raise ValueError(
                f"Could not match any DINOv2 backbone params from pretrained_path={pretrained_path}"
            )
        own.update(best_filtered)
        model.load_state_dict(own, strict=False)
        loaded_params = len(best_filtered)
    else:
        print("Warning: policy.pretrained_path is empty. Using randomly initialized DINOv2 backbone.")

    model.to(device)
    model.eval()
    return DinoV2BuildResult(model=model, embed_dim=int(model.embed_dim), loaded_params=loaded_params)


class DinoV2ImageLatentExtractor:
    """
    Extract image-only DINOv2 embeddings for FLOAT.

    The embedding phi(o_t) is the normalized DINOv2 CLS token for each frame.
    """

    def __init__(
        self,
        camera_name: str,
        image_size: int,
        device: str,
        pretrained_path: str = "",
        model_name: str = "vit_base",
        batch_size: int = 32,
        patch_size: int = 14,
        num_register_tokens: int = 4,
        normalize_embedding: bool = True,
    ) -> None:
        self.camera_name = str(camera_name)
        self.image_size = int(image_size)
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.normalize_embedding = bool(normalize_embedding)

        build = build_dino_v2_backbone(
            pretrained_path=str(pretrained_path),
            model_name=str(model_name),
            image_size=self.image_size,
            device=device,
            patch_size=int(patch_size),
            num_register_tokens=int(num_register_tokens),
        )
        self.model = build.model
        self.embed_dim = int(build.embed_dim)
        self.loaded_params = int(build.loaded_params)
        self.img_normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def close(self) -> None:
        return None

    def _prepare_images(self, images_raw: np.ndarray) -> np.ndarray:
        images = np.asarray(images_raw, dtype=np.uint8)
        if images.ndim != 4:
            raise ValueError(f"Expected images shape (T,H,W,C), got {images.shape}")
        outputs = np.empty((images.shape[0], 3, self.image_size, self.image_size), dtype=np.float32)
        for i in range(images.shape[0]):
            img = _center_crop_resize(images[i], self.image_size)
            outputs[i] = np.transpose(img.astype(np.float32) / 255.0, (2, 0, 1))
        return outputs

    @torch.no_grad()
    def encode_trajectory_with_indices(self, traj: PolicyTrajectory) -> tuple[np.ndarray, np.ndarray]:
        images = self._prepare_images(np.asarray(traj.images, dtype=np.uint8))
        outputs: list[np.ndarray] = []

        for start in range(0, images.shape[0], self.batch_size):
            stop = min(images.shape[0], start + self.batch_size)
            batch = torch.from_numpy(images[start:stop]).to(self.device)
            batch = self.img_normalize(batch)
            emb = self.model(batch)
            if isinstance(emb, dict):
                emb = emb["x_norm_clstoken"]
            if self.normalize_embedding:
                emb = F.normalize(emb, p=2.0, dim=-1)
            outputs.append(emb.detach().cpu().numpy().astype(np.float32))

        indices = np.arange(images.shape[0], dtype=np.int64)
        return np.concatenate(outputs, axis=0), indices

    @torch.no_grad()
    def encode_trajectory(self, traj: PolicyTrajectory) -> np.ndarray:
        emb, _ = self.encode_trajectory_with_indices(traj)
        return emb
