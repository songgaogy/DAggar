from __future__ import annotations

from collections import OrderedDict
from typing import Any, Iterable, Optional

import torch
import torch.nn as nn
from torchvision import models


class Encoder(nn.Module):
    """
    ResNet-18 visual encoder h_theta that outputs a single latent token per image.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        pretrained: bool = True,
        freeze: bool = True,
        normalize_input: bool = True,
    ) -> None:
        super().__init__()
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet18(weights=weights)
        except Exception:
            resnet = models.resnet18(pretrained=pretrained)

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.flatten = nn.Flatten()
        self.latent_dim = 512
        self.normalize_input = bool(normalize_input)
        self.register_buffer(
            "rgb_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "rgb_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        if checkpoint_path:
            self._load_external_checkpoint(checkpoint_path)
        if freeze:
            self.freeze()

    def _load_external_checkpoint(self, checkpoint_path: str) -> None:
        state = torch.load(checkpoint_path, map_location="cpu")
        full = models.resnet18(weights=None)
        full_state_keys = set(full.state_dict().keys())
        backbone_keys = {k for k in full_state_keys if not k.startswith("fc.")}

        for candidate in self._iter_state_dict_candidates(state):
            for cleaned in self._candidate_key_variants(candidate):
                try:
                    incompatible = full.load_state_dict(cleaned, strict=False)
                except Exception:
                    continue

                missing = set(incompatible.missing_keys)
                unexpected = set(incompatible.unexpected_keys)
                if not (backbone_keys - missing) == backbone_keys:
                    continue
                if any(k not in {"fc.weight", "fc.bias"} for k in missing):
                    continue
                if len(unexpected) > 0:
                    continue

                self.backbone = nn.Sequential(*list(full.children())[:-1])
                return
        raise ValueError(f"Unsupported ResNet checkpoint format: {checkpoint_path}")

    @staticmethod
    def _is_tensor_dict(x: Any) -> bool:
        if not isinstance(x, (dict, OrderedDict)) or len(x) == 0:
            return False
        return all(torch.is_tensor(v) for v in x.values())

    @classmethod
    def _iter_state_dict_candidates(cls, state: Any) -> Iterable[dict]:
        if cls._is_tensor_dict(state):
            yield dict(state)
        if isinstance(state, dict):
            for key in ("state_dict", "model", "encoder", "backbone", "resnet", "net"):
                sub = state.get(key)
                if cls._is_tensor_dict(sub):
                    yield dict(sub)

    @staticmethod
    def _strip_prefix(state_dict: dict, prefix: str) -> dict:
        return {
            (k[len(prefix) :] if k.startswith(prefix) else k): v
            for k, v in state_dict.items()
        }

    @classmethod
    def _candidate_key_variants(cls, state_dict: dict) -> Iterable[dict]:
        prefixes = ("module.", "model.", "encoder.", "backbone.", "resnet.", "net.")
        seen = set()
        queue = [dict(state_dict)]

        while queue:
            cur = queue.pop(0)
            key_sig = tuple(sorted(cur.keys()))
            if key_sig in seen:
                continue
            seen.add(key_sig)
            yield cur
            for prefix in prefixes:
                if any(k.startswith(prefix) for k in cur.keys()):
                    queue.append(cls._strip_prefix(cur, prefix))

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError(f"Expected image shape (B,3,H,W), got {tuple(image.shape)}")
        x = image.float()
        if x.max() > 1.5:
            x = x / 255.0
        if self.normalize_input:
            x = (x - self.rgb_mean.to(x.device)) / self.rgb_std.to(x.device)
        z = self.backbone(x)
        return self.flatten(z)
