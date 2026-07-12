"""Loss helpers for the V-only IQL value backup.

Implementation notes:
- The value is learned V-only by a TD backup; there is no Q head (Q is
  redundant under deterministic dynamics + single-action coverage — see
  V_ONLY_ADVANTAGE_DESIGN.md). V currently regresses onto the bootstrap target
  by plain MSE (see `IQLLearner.update`).
- `expectile_v_loss` mirrors baseline awr/models/flow.py:37-39: asymmetric
  L2 weighted by tau if diff > 0 else (1 - tau). It is RESERVED for the
  planned optimism knob (`L_V = expectile_tau(target - V)`), not called by the
  current MSE-TD training path.
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
        diff: (B, 1) = q_min.detach() - v_pred, where v_pred requires grad.
        tau:  expectile in (0, 1); 0.7 is the IQL default.
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
