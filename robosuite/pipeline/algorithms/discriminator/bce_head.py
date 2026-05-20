"""Trainable BCE head.

Structurally identical to `robosuite.discriminator.lpb_v2.detectors.bce`'s
`BCEHead` (LayerNorm + GELU MLP), but lives in this module so the head's
parameters can be `requires_grad=True` while the upstream frozen artifact
stays untouched.

`warm_start_from_lpb_bce_ckpt` copies weights from the pre-fitted ckpt at
`checkpoints/lpb_v2/robosuite_ckpt/bce_head.pth`.
"""

from __future__ import annotations

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
        self.in_dim = in_dim
        self.hidden = hidden
        self.num_layers = num_layers
        # Build LayerNorm + GELU + Linear stack mirroring lpb_v2 BCEHead.

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Args:
            z: (B, in_dim) — concatenation of frozen latent and action features.
        Returns:
            (B,) raw logit.
        """
        raise NotImplementedError

    def warm_start_from_lpb_bce_ckpt(self, ckpt_path: str) -> None:
        """Copy weights from a pre-fitted lpb_v2 BCE checkpoint into this head.

        Args:
            ckpt_path: path to `bce_head.pth` produced by
                       `run_bce_robosuite_benchmark.sh`. Only the head
                       weights are copied; thresholds / calibration are
                       discarded (we re-estimate them online).
        """
        raise NotImplementedError
