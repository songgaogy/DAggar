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

import torch


def disc_logit_to_intrinsic_reward(logit: torch.Tensor, sign_mode: str) -> torch.Tensor:
    """Map BCE head logits to IQL intrinsic reward on (-1, 0).

    Convention: higher logit = more failure-like.

    - ``negate_logit`` (default): ``-sigmoid(logit)`` — expert-like ~ 0, failure-like ~ -1.
    - ``raw``: passthrough logit (legacy / debugging).
    """
    mode = str(sign_mode).lower()
    if mode == "negate_logit":
        return -torch.sigmoid(logit)
    if mode == "raw":
        return logit
    raise ValueError(
        f"Unsupported disc_reward_sign={sign_mode!r}. Expected 'negate_logit' or 'raw'."
    )


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
