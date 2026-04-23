"""ModelEMA: parameter-level exponential moving average.

CFG inference on raw training weights is noisier than on EMA weights (standard
diffusion-training practice). EMA also dampens the Sweep-I / Sweep-II
oscillation between the critic and the advantage-gate during the bootstrap
phase. The shadow is kept in full precision and written back via ``apply_to``
for benchmark inference.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        if not (0.0 < float(decay) < 1.0):
            raise ValueError(f"decay must be in (0,1), got {decay}")
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        sd = model.state_dict()
        for k, v in sd.items():
            if k not in self.shadow:
                self.shadow[k] = v.detach().clone()
                continue
            if v.dtype.is_floating_point:
                shadow_k = self.shadow[k]
                if shadow_k.dtype != v.dtype or shadow_k.device != v.device:
                    shadow_k = shadow_k.to(device=v.device, dtype=v.dtype).clone()
                    self.shadow[k] = shadow_k
                shadow_k.mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

    def apply_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": dict(self.shadow)}

    def load_state_dict(self, state: dict) -> None:
        self.decay = float(state["decay"])
        self.shadow = {k: v.detach().clone() for k, v in state["shadow"].items()}
