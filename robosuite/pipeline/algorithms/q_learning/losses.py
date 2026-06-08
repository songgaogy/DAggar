"""Loss helpers for IQL Q-chunking.

Implementation notes:
- `bellman_q_loss` is plain MSE between predicted Q and detached target.
- `expectile_v_loss` mirrors baseline awr/models/flow.py:37-39: asymmetric
  L2 weighted by tau if diff > 0 else (1 - tau).
- `compute_advantage` returns `min(q1, q2) - v` and is kept for legacy
  two-Q callers.
- `compute_ensemble_advantage` returns `mean(Q_1..Q_K) - V`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def bellman_q_loss(q_pred: torch.Tensor, target_q: torch.Tensor) -> torch.Tensor:
    """Args:
        q_pred:   (B, 1) or (K, B, 1) predicted Q values, requires grad.
        target_q: same shape as q_pred, or broadcast-compatible target.
    Returns:
        scalar MSE. For an ensemble, returns the sum of per-critic MSE terms
        so each critic keeps the same gradient scale as the two-Q v0 learner.
    """
    if q_pred.dim() == 3:
        loss = F.mse_loss(q_pred, target_q.expand_as(q_pred), reduction="none")
        return loss.mean(dim=(1, 2)).sum()
    return F.mse_loss(q_pred, target_q)


def expectile_v_loss(diff: torch.Tensor, tau: float) -> torch.Tensor:
    """Asymmetric expectile regression loss.

    Args:
        diff: (B, 1) = q_min.detach() - v_pred, where v_pred requires grad.
        tau:  expectile in (0, 1); 0.7 is the IQL default.
    Returns:
        scalar mean loss.
    """
    weight = torch.where(
        diff > 0.0,
        torch.full_like(diff, float(tau)),
        torch.full_like(diff, 1.0 - float(tau)),
    )
    return (weight * diff.square()).mean()


def compute_advantage(q1: torch.Tensor, q2: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Advantage A = min(Q1, Q2) - V.  All inputs must already be detached
    upstream; this is purely a numerical op.

    Args:
        q1, q2, v: shape (B, 1).
    Returns:
        (B,) flattened advantage.
    """
    return (torch.min(q1, q2) - v).squeeze(-1)


def compute_ensemble_advantage(q_values: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Advantage A = mean(Q_1..Q_K) - V for a Q ensemble.

    Args:
        q_values: shape (K, B, 1).
        v:        shape (B, 1).
    Returns:
        (B,) flattened advantage.
    """
    if q_values.dim() != 3:
        raise ValueError(f"q_values must be (K, B, 1); got {tuple(q_values.shape)}")
    if q_values.shape[-1] != 1:
        raise ValueError(f"q_values last dim must be 1; got {tuple(q_values.shape)}")
    return (q_values.mean(dim=0) - v).squeeze(-1)
