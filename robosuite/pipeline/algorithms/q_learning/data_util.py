"""Utility helpers shared by replay sampling and warmup.

Contents:
    - `aggregate_chunk_reward(step_rewards, discount)`: returns the n-step
      discounted return Σ_{i=0}^{H-1} γ^i · r_{t+i}.
    - `chunk_done_mask(step_dones)`: True if any step in the chunk ended an
      episode; used to mask the bootstrap term γ^H · V(s').
    - `tile_proprio_as_action(proprio_raw, action_input_dim)`: implements
      the f(o, s) degradation by repeating proprio to fill the encoder's
      action input slot.

NOTE: `SharedFrozenEncoder` (subagent-3) keeps a private `_tile_proprio_as_action`
in `discriminator/encoder.py`. The public version here is intended to be
the single source of truth once subagent-3 migrates over (see DIPOLE_RL.md §10).
"""

from __future__ import annotations

import numpy as np
import torch


def compute_action_quantile_stats(
    actions: np.ndarray | torch.Tensor,
    q_low: float = 0.01,
    q_high: float = 0.99,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-dim action quantiles for EXPO-style normalization (q_low/q_high -> [-1, 1]).

    Args:
        actions: (N, D_a) raw env actions gathered from the offline dataset.
        q_low, q_high: lower/upper quantiles (defaults 0.01 / 0.99).
    Returns:
        (q01, q99) float32 tensors of shape (D_a,). Degenerate dims (q99 == q01)
        are widened by +/-0.5 so the normalization span stays > 0.
    """
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"compute_action_quantile_stats expected (N, D_a); got {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError("compute_action_quantile_stats got zero action samples.")
    q01 = np.quantile(arr, float(q_low), axis=0).astype(np.float32)
    q99 = np.quantile(arr, float(q_high), axis=0).astype(np.float32)
    degenerate = (q99 - q01) < 1e-6
    q01 = np.where(degenerate, q01 - 0.5, q01).astype(np.float32)
    q99 = np.where(degenerate, q99 + 0.5, q99).astype(np.float32)
    return torch.from_numpy(q01), torch.from_numpy(q99)


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


def terminal_undiscounted_reward(
    step_rewards: torch.Tensor,
    output_reward_coef: float = 1.0,
) -> torch.Tensor:
    """Undiscounted env-reward sum over the chunk (no gamma^k discounting).

    For a 0/1 success chunk this equals ``output_reward_coef * 1.0``. Used to
    override the terminal chunk reward so terminal Q targets ~1 instead of the
    in-chunk-discounted ``gamma^k``.

    Args:
        step_rewards: (B, H) per-step env rewards.
        output_reward_coef: scaling applied to env reward.
    Returns:
        (B, 1) undiscounted reward sum.
    """
    if step_rewards.dim() != 2:
        raise ValueError(
            f"step_rewards expected (B, H); got shape {tuple(step_rewards.shape)}"
        )
    return float(output_reward_coef) * step_rewards.sum(dim=1, keepdim=True)


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
    if proprio_raw.dim() != 2:
        raise ValueError(
            f"proprio_raw expected (B, D_s); got shape {tuple(proprio_raw.shape)}"
        )
    B, D_s = proprio_raw.shape
    if D_s == 0:
        raise ValueError("proprio_raw has zero feature dim; cannot tile as action")
    reps = (action_input_dim + D_s - 1) // D_s
    flat = proprio_raw.repeat(1, reps)
    if flat.shape[1] >= action_input_dim:
        flat = flat[:, :action_input_dim]
    else:
        pad = torch.zeros(
            B,
            action_input_dim - flat.shape[1],
            device=flat.device,
            dtype=flat.dtype,
        )
        flat = torch.cat([flat, pad], dim=-1)
    return flat.contiguous()
