"""Loss helpers for the online discriminator."""

from __future__ import annotations

import torch


def bce_with_logits_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    label_smoothing: float = 0.0,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """BCEWithLogits with optional label smoothing.

    Args:
        logits: (B,) raw head outputs.
        labels: (B,) in {0, 1}.
        label_smoothing: in [0, 0.5); applied as
            y' = y * (1 - 2s) + s.
        pos_weight: optional (1,) scalar for class-imbalance reweighting.
    Returns:
        scalar mean loss.
    """
    raise NotImplementedError
