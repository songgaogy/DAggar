"""Loss helpers for VAST value stitching.

Implementation notes:
- V regresses onto the detached stitched target with expectile regression.
- `expectile_v_loss` mirrors baseline awr/models/flow.py:37-39: asymmetric
  L2 weighted by tau if diff > 0 else (1 - tau).
"""

from __future__ import annotations

import torch


def expectile_v_loss(
    diff: torch.Tensor,
    tau: float,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Asymmetric expectile regression loss.

    Args:
        diff: (B, 1) = stitched_target.detach() - v_pred.
        tau:  expectile in (0, 1); 0.9 is the VAST default.
        weights: optional (B, 1) per-sample loss weights.
    Returns:
        scalar mean loss.
    """
    weight = torch.where(
        diff > 0.0,
        torch.full_like(diff, float(tau)),
        torch.full_like(diff, 1.0 - float(tau)),
    )
    elementwise = weight * diff.square()
    if weights is None:
        return elementwise.mean()
    w = weights.to(elementwise)
    return (elementwise * w).sum() / w.sum()
