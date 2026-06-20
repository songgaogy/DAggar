"""Utility helpers shared by replay sampling and warmup.

Contents:
    - `aggregate_chunk_reward(step_rewards, discount)`: returns the n-step
      discounted return Σ_{i=0}^{H-1} γ^i · r_{t+i}.
    - `chunk_done_mask(step_dones)`: True if any step in the chunk ended an
      episode; used to mask the bootstrap term γ^H · V(s').
"""

from __future__ import annotations

import torch


def aggregate_chunk_reward(
    step_rewards: torch.Tensor,
    discount: float,
) -> torch.Tensor:
    """Args:
        step_rewards: (B, H) per-step env-or-mixed rewards.
        discount:     scalar γ.
    Returns:
        (B, 1) n-step discounted return Σ_{i=0}^{H-1} γ^i · r_{t+i}.
    """
    if step_rewards.dim() != 2:
        raise ValueError(
            f"step_rewards expected (B, H); got shape {tuple(step_rewards.shape)}"
        )
    horizon = step_rewards.shape[1]
    powers = torch.pow(
        torch.full((horizon,), float(discount), device=step_rewards.device, dtype=step_rewards.dtype),
        torch.arange(horizon, device=step_rewards.device, dtype=step_rewards.dtype),
    )
    return (step_rewards * powers.view(1, horizon)).sum(dim=1, keepdim=True)


def chunk_done_mask(step_dones: torch.Tensor) -> torch.Tensor:
    """Args:
        step_dones: (B, H) bool/float.
    Returns:
        (B, 1) — 1 if any step in the chunk is done.
    """
    if step_dones.dim() != 2:
        raise ValueError(
            f"step_dones expected (B, H); got shape {tuple(step_dones.shape)}"
        )
    return step_dones.to(torch.bool).any(dim=1, keepdim=True).to(torch.float32)
