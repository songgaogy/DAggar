"""Discriminator-specific policy-segment and GT-window routing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from robosuite.pipeline.common.episodes import (
    OFFLINE_EPISODE_SCHEMA_VERSION,
    load_offline_episodes,
    validate_offline_payload,
)


@dataclass(frozen=True)
class PolicySegment:
    """One maximal non-intervention run inside a collected episode."""

    source_episode_index: int
    segment_index: int
    lo: int
    hi: int
    pool: str
    terminal_reason: str
    ended_by: str
    episode: Mapping[str, Any] = field(repr=False, compare=False)

    @property
    def num_frames(self) -> int:
        return int(self.hi - self.lo)

    @property
    def identifier(self) -> str:
        return (
            f"offline-episode-{self.source_episode_index:06d}"
            f"-segment-{self.segment_index:03d}"
        )


@dataclass(frozen=True)
class GTNegativeWindow:
    """One intervention-onset window encoded with counterfactual policy actions."""

    source_episode_index: int
    event_index: int
    intervention_lo: int
    intervention_hi: int
    lo: int
    hi: int
    frame_indices: tuple[int, ...]
    episode: Mapping[str, Any] = field(repr=False, compare=False)
    kind: str = "intervention"  # "intervention" or "pre_end"

    @property
    def onset(self) -> int:
        return int(self.intervention_lo)

    @property
    def num_frames(self) -> int:
        return len(self.frame_indices)

    @property
    def num_pre_frames(self) -> int:
        return sum(index < self.onset for index in self.frame_indices)

    @property
    def num_post_frames(self) -> int:
        return sum(index >= self.onset for index in self.frame_indices)

    @property
    def identifier(self) -> str:
        tag = "preend" if self.kind == "pre_end" else "intervention"
        return (
            f"offline-episode-{self.source_episode_index:06d}"
            f"-{tag}-{self.event_index:03d}"
        )


def split_policy_segments(
    payload: Mapping[str, Any],
) -> tuple[list[PolicySegment], dict[str, Any]]:
    """Split episodes at intervention boundaries and route policy runs to P/U."""
    episodes = list(payload.get("episodes", []))
    segments: list[PolicySegment] = []
    stats: dict[str, Any] = {
        "episodes": len(episodes),
        "policy_segments": 0,
        "positive_segments": 0,
        "unlabeled_segments": 0,
        "positive_frames": 0,
        "unlabeled_frames": 0,
        "excluded_human_frames": 0,
        "terminal_reasons": {},
    }
    for episode_index, episode in enumerate(episodes):
        flags = np.asarray(episode["is_intervention"], dtype=np.bool_)
        success = np.asarray(episode["success"], dtype=np.bool_)
        reason = str(episode["terminal_reason"])
        stats["excluded_human_frames"] += int(flags.sum())
        terminal_reasons = stats["terminal_reasons"]
        terminal_reasons[reason] = terminal_reasons.get(reason, 0) + 1
        lo = 0
        local_index = 0
        while lo < int(flags.shape[0]):
            current = bool(flags[lo])
            hi = lo + 1
            while hi < int(flags.shape[0]) and bool(flags[hi]) == current:
                hi += 1
            if not current:
                positive = (
                    hi == int(flags.shape[0])
                    and reason == "success"
                    and bool(success[hi - 1])
                )
                if positive:
                    pool = "positive"
                    ended_by = "success"
                else:
                    pool = "unlabeled"
                    ended_by = "intervention" if hi < int(flags.shape[0]) else reason
                segment = PolicySegment(
                    source_episode_index=int(episode_index),
                    segment_index=int(local_index),
                    lo=int(lo),
                    hi=int(hi),
                    pool=pool,
                    terminal_reason=reason,
                    ended_by=str(ended_by),
                    episode=episode,
                )
                segments.append(segment)
                stats["policy_segments"] += 1
                stats[f"{pool}_segments"] += 1
                stats[f"{pool}_frames"] += segment.num_frames
                local_index += 1
            lo = hi
    return segments, stats


def build_gt_negative_windows(
    payload: Mapping[str, Any],
    *,
    pre_intervention_chunks: int,
    post_intervention_chunks: int,
    frameskip: int,
    pre_end_chunks: int = 0,
) -> tuple[list[GTNegativeWindow], dict[str, Any]]:
    """Build event-local GT-negative windows without consulting stored labels.

    Each intervention window joins the contiguous policy-only prefix immediately
    before an intervention onset with the beginning of that same intervention
    block. When ``pre_end_chunks > 0``, the final ``pre_end_chunks * frameskip``
    frames of every episode that terminates for a non-``success`` reason are also
    routed to GT-negative, capturing policy behavior that drove the episode into a
    failed/aborted end. ``frame_indices`` is globally de-duplicated by
    episode/frame key across both sources.
    """
    if int(pre_intervention_chunks) < 0:
        raise ValueError(
            "pre_intervention_chunks must be non-negative, got "
            f"{pre_intervention_chunks}."
        )
    if int(post_intervention_chunks) <= 0:
        raise ValueError(
            "post_intervention_chunks must be positive, got "
            f"{post_intervention_chunks}."
        )
    if int(frameskip) <= 0:
        raise ValueError(f"frameskip must be positive, got {frameskip}.")
    if int(pre_end_chunks) < 0:
        raise ValueError(
            f"pre_end_chunks must be non-negative, got {pre_end_chunks}."
        )

    pre_limit = int(pre_intervention_chunks) * int(frameskip)
    post_limit = int(post_intervention_chunks) * int(frameskip)
    pre_end_limit = int(pre_end_chunks) * int(frameskip)
    episodes = list(payload.get("episodes", []))
    windows: list[GTNegativeWindow] = []
    seen_frames: set[tuple[int, int]] = set()
    duplicate_memberships = 0
    event_summaries: list[dict[str, Any]] = []
    pre_end_summaries: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "episodes": len(episodes),
        "intervention_events": 0,
        "gt_negative_windows": 0,
        "gt_negative_frames": 0,
        "pre_frames": 0,
        "post_frames": 0,
        "pre_truncated_events": 0,
        "post_truncated_events": 0,
        "non_success_episodes": 0,
        "pre_end_windows": 0,
        "pre_end_frames": 0,
        "pre_end_truncated_events": 0,
        "deduplicated_frames": 0,
        "selected_frame_keys": [],
        "action_source": "policy_action",
        "observation_source": "policy_prefix_and_human_intervention",
        "post_boundary": "intervention_block",
        "chunk_boundary": "continuous_window",
        "pre_end_chunks": int(pre_end_chunks),
        "stored_gt_fail_ignored": True,
        "theory_deviation": (
            "GT-negative windows include human-control observations and therefore "
            "are not strictly a subset of offline policy-only unlabeled data."
        ),
        "events": event_summaries,
        "pre_end_events": pre_end_summaries,
    }
    for episode_index, episode in enumerate(episodes):
        flags = np.asarray(episode["is_intervention"], dtype=np.bool_)
        event_index = 0
        cursor = 0
        while cursor < int(flags.shape[0]):
            if not bool(flags[cursor]):
                cursor += 1
                continue
            intervention_lo = cursor
            intervention_hi = intervention_lo + 1
            while (
                intervention_hi < int(flags.shape[0])
                and bool(flags[intervention_hi])
            ):
                intervention_hi += 1

            policy_prefix_lo = intervention_lo
            while policy_prefix_lo > 0 and not bool(flags[policy_prefix_lo - 1]):
                policy_prefix_lo -= 1
            lo = max(policy_prefix_lo, intervention_lo - pre_limit)
            hi = min(intervention_hi, intervention_lo + post_limit)
            candidate_indices = tuple(range(lo, hi))
            unique_indices: list[int] = []
            for frame_index in candidate_indices:
                key = (int(episode_index), int(frame_index))
                if key in seen_frames:
                    duplicate_memberships += 1
                    continue
                seen_frames.add(key)
                unique_indices.append(int(frame_index))

            available_pre = intervention_lo - policy_prefix_lo
            available_post = intervention_hi - intervention_lo
            pre_truncated = available_pre < pre_limit
            post_truncated = available_post < post_limit
            if pre_truncated:
                stats["pre_truncated_events"] += 1
            if post_truncated:
                stats["post_truncated_events"] += 1
            window = GTNegativeWindow(
                source_episode_index=int(episode_index),
                event_index=int(event_index),
                intervention_lo=int(intervention_lo),
                intervention_hi=int(intervention_hi),
                lo=int(lo),
                hi=int(hi),
                frame_indices=tuple(unique_indices),
                episode=episode,
            )
            if window.num_frames:
                windows.append(window)
                stats["gt_negative_windows"] += 1
                stats["gt_negative_frames"] += window.num_frames
                stats["pre_frames"] += window.num_pre_frames
                stats["post_frames"] += window.num_post_frames
            event_summaries.append(
                {
                    "source_episode_index": int(episode_index),
                    "event_index": int(event_index),
                    "intervention_start": int(intervention_lo),
                    "intervention_end": int(intervention_hi),
                    "window_start": int(lo),
                    "window_end": int(hi),
                    "pre_frames": int(window.num_pre_frames),
                    "post_frames": int(window.num_post_frames),
                    "pre_truncated": bool(pre_truncated),
                    "post_truncated": bool(post_truncated),
                    "deduplicated_frames": int(len(candidate_indices) - window.num_frames),
                }
            )
            stats["intervention_events"] += 1
            event_index += 1
            cursor = intervention_hi

        reason = str(episode.get("terminal_reason", ""))
        length = int(flags.shape[0])
        if pre_end_limit > 0 and reason != "success" and length > 0:
            stats["non_success_episodes"] += 1
            end_lo = max(0, length - pre_end_limit)
            end_unique: list[int] = []
            for frame_index in range(end_lo, length):
                key = (int(episode_index), int(frame_index))
                if key in seen_frames:
                    duplicate_memberships += 1
                    continue
                seen_frames.add(key)
                end_unique.append(int(frame_index))
            end_truncated = (length - end_lo) < pre_end_limit
            if end_truncated:
                stats["pre_end_truncated_events"] += 1
            # onset == hi so every selected frame is counted as a pre-end frame.
            end_window = GTNegativeWindow(
                source_episode_index=int(episode_index),
                event_index=0,
                intervention_lo=int(length),
                intervention_hi=int(length),
                lo=int(end_lo),
                hi=int(length),
                frame_indices=tuple(end_unique),
                episode=episode,
                kind="pre_end",
            )
            if end_window.num_frames:
                windows.append(end_window)
                stats["gt_negative_windows"] += 1
                stats["gt_negative_frames"] += end_window.num_frames
                stats["pre_end_windows"] += 1
                stats["pre_end_frames"] += end_window.num_frames
            pre_end_summaries.append(
                {
                    "source_episode_index": int(episode_index),
                    "terminal_reason": reason,
                    "window_start": int(end_lo),
                    "window_end": int(length),
                    "pre_end_frames": int(end_window.num_frames),
                    "pre_end_truncated": bool(end_truncated),
                    "deduplicated_frames": int(
                        (length - end_lo) - end_window.num_frames
                    ),
                }
            )
    stats["deduplicated_frames"] = int(duplicate_memberships)
    stats["selected_frame_keys"] = [
        [int(episode_index), int(frame_index)]
        for episode_index, frame_index in sorted(seen_frames)
    ]
    return windows, stats


__all__ = [
    "OFFLINE_EPISODE_SCHEMA_VERSION",
    "GTNegativeWindow",
    "PolicySegment",
    "build_gt_negative_windows",
    "load_offline_episodes",
    "split_policy_segments",
    "validate_offline_payload",
]
