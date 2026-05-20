"""Utility helpers shared by replay sampling and warmup.

Contents (to be implemented):
    - `aggregate_chunk_reward(step_rewards, discount)`: returns the n-step
      discounted return Σ_{i=0}^{H-1} γ^i · r_{t+i}.
    - `chunk_done_mask(step_dones)`: True if any step in the chunk ended an
      episode; used to mask the bootstrap term γ^H · V(s').
    - `tile_proprio_as_action(proprio_raw, action_input_dim)`: implements
      the f(o, s) degradation by repeating proprio to fill the encoder's
      action input slot.
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
        (B, 1) n-step discounted return.
    """
    raise NotImplementedError


def chunk_done_mask(step_dones: torch.Tensor) -> torch.Tensor:
    """Args:
        step_dones: (B, H) bool/float.
    Returns:
        (B, 1) — 1 if any step in the chunk is done.
    """
    raise NotImplementedError


def tile_proprio_as_action(
    proprio_raw: torch.Tensor,
    action_input_dim: int,
) -> torch.Tensor:
    """Implements the f(o, s) := encode(o, s, a:=tile(s)) degradation.

    Args:
        proprio_raw:      (B, D_s).
        action_input_dim: encoder's expected action-input dim
                          (frameskip * action_dim_per_step).
    Returns:
        (B, action_input_dim).
    """
    raise NotImplementedError
