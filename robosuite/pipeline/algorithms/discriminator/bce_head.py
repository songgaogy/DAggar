"""Trainable BCE head.

Structurally identical to `robosuite.discriminator.lpb_v2.detectors.bce`'s
`BCEHead` (LayerNorm + GELU MLP), but lives in this module so the head's
parameters can be `requires_grad=True` while the upstream frozen artifact
stays untouched.

`warm_start_from_lpb_bce_ckpt` copies weights from a pre-fitted ckpt
(e.g. `checkpoints/lpb_v2/bce_viz_robosuite/<run>/checkpoints/bce_head.pth`).
Only shape-matching tensors are copied; in particular the first Linear
is re-initialized because this head's in_dim is `D_ctx + H * D_a`
(action-conditioned) whereas the frozen lpb_v2 head was `D_ctx` only.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import torch
from torch import nn


class TrainableBCEHead(nn.Module):
    """MLP scalar-logit head: (B, in_dim) -> (B,)."""

    def __init__(
        self,
        in_dim: int,
        *,
        hidden: int = 256,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        if in_dim <= 0:
            raise ValueError(f"in_dim must be positive, got {in_dim}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self.num_layers = int(num_layers)

        layers: List[nn.Module] = []
        d = self.in_dim
        for _ in range(self.num_layers):
            layers.append(nn.Linear(d, self.hidden))
            layers.append(nn.LayerNorm(self.hidden))
            layers.append(nn.GELU())
            d = self.hidden
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        linears = [m for m in self.net if isinstance(m, nn.Linear)]
        for lin in linears[:-1]:
            nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)
        last = linears[-1]
        nn.init.zeros_(last.weight)
        if last.bias is not None:
            nn.init.zeros_(last.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() != 2:
            raise ValueError(f"expected (B, in_dim); got shape {tuple(z.shape)}")
        if z.shape[-1] != self.in_dim:
            raise ValueError(
                f"TrainableBCEHead in_dim mismatch: head expects {self.in_dim}, "
                f"got {z.shape[-1]}"
            )
        return self.net(z).squeeze(-1)

    def warm_start_from_lpb_bce_ckpt(self, ckpt_path: str) -> int:
        """Copy weights from a pre-fitted lpb_v2 BCE checkpoint into this head.

        Returns the number of parameter tensors actually updated. Only
        shape-matching keys are loaded; thresholds/calibration discarded.
        """
        path = Path(ckpt_path)
        if not path.exists():
            raise FileNotFoundError(f"BCE checkpoint not found: {path}")
        state = torch.load(str(path), map_location="cpu", weights_only=False)
        bce_detector = state.get("bce_detector", None)
        if not isinstance(bce_detector, dict):
            raise KeyError(
                f"BCE checkpoint {path} missing 'bce_detector' dict; got keys "
                f"{list(state.keys())}"
            )
        head_state = bce_detector.get("head", None)
        if not isinstance(head_state, dict):
            raise KeyError(
                f"BCE checkpoint {path} missing 'bce_detector.head'; got "
                f"keys {list(bce_detector.keys())}"
            )
        own = self.state_dict()
        own_keys_before = {k: v.detach().clone() for k, v in own.items()}
        compatible: dict[str, torch.Tensor] = {}
        for key, tensor in head_state.items():
            if not isinstance(tensor, torch.Tensor):
                continue
            if key in own and own[key].shape == tensor.shape:
                compatible[key] = tensor.to(own[key].dtype)
        if not compatible:
            raise RuntimeError(
                "warm_start_from_lpb_bce_ckpt: no shape-compatible tensors "
                "found between ckpt head and TrainableBCEHead. Ckpt keys: "
                f"{list(head_state.keys())}; head keys: {list(own.keys())}."
            )
        missing, unexpected = self.load_state_dict({**own, **compatible}, strict=False)
        # The above call replaced every key in own_keys_before, so count
        # how many actually differ as a sanity check (returns >= 1).
        updated = 0
        new_state = self.state_dict()
        for key, before in own_keys_before.items():
            if key in compatible and not torch.equal(before, new_state[key]):
                updated += 1
        if updated == 0:
            raise RuntimeError(
                "warm_start_from_lpb_bce_ckpt: matched compatible keys but no "
                f"tensor changed (compatible={list(compatible.keys())})."
            )
        return updated
