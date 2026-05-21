"""Loss helpers for the online discriminator."""

from __future__ import annotations

import torch
import torch.nn.functional as F


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
    if not (0.0 <= label_smoothing < 0.5):
        raise ValueError(
            f"label_smoothing must be in [0, 0.5); got {label_smoothing}"
        )
    y = labels.to(dtype=logits.dtype).view(-1)
    if label_smoothing > 0:
        y = y * (1.0 - 2.0 * label_smoothing) + label_smoothing
    return F.binary_cross_entropy_with_logits(
        logits.view(-1),
        y,
        pos_weight=pos_weight,
        reduction="mean",
    )
