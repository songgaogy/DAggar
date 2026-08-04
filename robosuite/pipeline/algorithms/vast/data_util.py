"""Utility helpers shared by replay sampling and warmup.

Contents:
    - `aggregate_chunk_reward(step_rewards, discount)`: returns the n-step
      discounted return Σ_{i=0}^{H-1} γ^i · r_{t+i}.
    - `chunk_done_mask(step_dones)`: True if any step in the chunk ended an
      episode; used to mask the bootstrap term γ^H · V(s').
    - `freeze_post_success_tail(transitions)`: collapse recorded post-success
      drift into one absorbing success anchor without changing episode layout.
    - `clone_with_absorbing_success_tail(transitions, horizon)`: build a
      VAST-only copy whose successful episodes have enough absorbing tail for
      every chunk phase to receive terminal supervision.
    - `mask_absorbing_tail_rewards(...)`: zero mixed rewards after success.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch


def clone_with_absorbing_success_tail(
    transitions: list[Any],
    horizon: int,
) -> tuple[list[Any], int, int]:
    """Clone transitions and complete successful episodes with absorbing tail.

    A successful episode receives enough synthetic frames to leave at least
    ``horizon - 1`` frames after its first success. This makes every stride-1
    chunk phase reach a terminal window. Inputs and their ``info`` dicts are
    never mutated. Returns ``(copies, synthetic_count, padded_episode_count)``.
    """
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if not transitions:
        return [], 0, 0

    copies = [replace(item, info=dict(item.info or {})) for item in transitions]
    expanded: list[Any] = []
    synthetic_count = 0
    padded_episode_count = 0
    episode_start = 0
    for index, transition in enumerate(copies):
        if not (bool(transition.done) or index == len(copies) - 1):
            continue
        episode = copies[episode_start : index + 1]
        first_success = next(
            (
                offset
                for offset, item in enumerate(episode)
                if bool((item.info or {}).get("success", False))
            ),
            None,
        )
        if first_success is not None:
            existing_tail = len(episode) - first_success - 1
            missing_tail = max(0, horizon - 1 - existing_tail)
            if missing_tail:
                episode[-1].done = False
                anchor = episode[first_success]
                anchor_info = dict(anchor.info or {})
                last_step = int((episode[-1].info or {}).get("episode_step", len(episode) - 1))
                for offset in range(missing_tail):
                    info = dict(anchor_info)
                    info["episode_step"] = last_step + offset + 1
                    info["success"] = True
                    info["frozen_post_success"] = True
                    info["synthetic_vast_success_tail"] = True
                    episode.append(
                        replace(
                            anchor,
                            obs=anchor.obs,
                            next_obs=anchor.obs,
                            action=anchor.action,
                            reward=0.0,
                            done=offset == missing_tail - 1,
                            is_intervention=False,
                            info=info,
                            reward_source="vast_absorbing_success_tail",
                        )
                    )
                synthetic_count += missing_tail
                padded_episode_count += 1
        expanded.extend(episode)
        episode_start = index + 1

    freeze_post_success_tail(expanded)
    return expanded, synthetic_count, padded_episode_count


def freeze_post_success_tail(transitions: list[Any]) -> int:
    """Collapse each episode's post-success drift into one absorbing anchor.

    For every episode delimited by ``Transition.done``, frames after the first
    true ``info["success"]`` reuse that success frame's observation and action,
    receive zero reward, and stay marked successful. Episode boundaries and the
    number of valid replay windows remain unchanged. Episodes without success
    are untouched. Returns the number of frozen tail frames.
    """
    if not transitions:
        return 0
    frozen = 0
    episode_start = 0
    for index, transition in enumerate(transitions):
        if not (bool(transition.done) or index == len(transitions) - 1):
            continue
        episode = transitions[episode_start : index + 1]
        first_success = next(
            (
                offset
                for offset, item in enumerate(episode)
                if bool((item.info or {}).get("success", False))
            ),
            None,
        )
        if first_success is not None:
            anchor = episode[first_success]
            for tail in episode[first_success + 1 :]:
                tail.obs = anchor.obs
                tail.next_obs = anchor.obs
                tail.action = anchor.action
                tail.reward = 0.0
                info = dict(tail.info or {})
                info["success"] = True
                info["frozen_post_success"] = True
                tail.info = info
                frozen += 1
        episode_start = index + 1
    return frozen


def mask_absorbing_tail_rewards(
    step_rewards: torch.Tensor,
    step_success: torch.Tensor,
    step_post_success: torch.Tensor,
) -> torch.Tensor:
    """Zero rewards after a terminal while retaining the real terminal reward."""
    if step_rewards.dim() != 2:
        raise ValueError(
            f"step_rewards expected (B, H); got shape {tuple(step_rewards.shape)}"
        )
    if step_success.shape != step_rewards.shape:
        raise ValueError("step_success must match step_rewards shape")
    if step_post_success.shape != step_rewards.shape:
        raise ValueError("step_post_success must match step_rewards shape")
    success = step_success.to(torch.bool)
    success_before = success.to(torch.int64).cumsum(dim=1) - success.to(torch.int64)
    inactive = success_before.to(torch.bool) | step_post_success.to(torch.bool)
    return step_rewards * (~inactive).to(step_rewards.dtype)


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
