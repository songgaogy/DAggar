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


def bellman_q_loss(
    q_pred: torch.Tensor,
    target_q: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Args:
        q_pred:   (B, 1) or (K, B, 1) predicted Q values, requires grad.
        target_q: same shape as q_pred, or broadcast-compatible target.
        weights:  optional (B, 1) per-sample loss weights (e.g. terminal
                  up-weighting). When None, behaves as a plain mean (the
                  weighted mean with all-ones weights is identical).
    Returns:
        scalar MSE. For an ensemble, returns the sum of per-critic MSE terms
        so each critic keeps the same gradient scale as the two-Q v0 learner.
    """
    if q_pred.dim() == 3:
        loss = F.mse_loss(q_pred, target_q.expand_as(q_pred), reduction="none")
        if weights is None:
            return loss.mean(dim=(1, 2)).sum()
        w = weights.to(loss).unsqueeze(0)  # (1, B, 1) -> broadcast over K
        per_critic = (loss * w).sum(dim=(1, 2)) / w.sum()
        return per_critic.sum()
    if weights is None:
        return F.mse_loss(q_pred, target_q)
    loss = F.mse_loss(q_pred, target_q, reduction="none")
    w = weights.to(loss)
    return (loss * w).sum() / w.sum()


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
