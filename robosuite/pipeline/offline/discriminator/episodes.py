"""Offline episode validation and policy-segment routing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


OFFLINE_EPISODE_SCHEMA_VERSION = 1


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


def _as_array(value: Any, *, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    if ndim is not None and array.ndim != int(ndim):
        raise ValueError(f"{name} must have ndim={ndim}, got shape {array.shape}.")
    return array


def _camera_names(payload: Mapping[str, Any]) -> list[str]:
    names = payload.get("camera_names")
    if names:
        return [str(name) for name in names]
    episodes = list(payload.get("episodes", []))
    if not episodes:
        return []
    return [str(key) for key in episodes[0]["obs"] if str(key) != "state"]


def _validate_episode(
    episode: Mapping[str, Any],
    *,
    source_index: int,
    camera_names: Sequence[str],
) -> None:
    required = (
        "obs",
        "next_obs",
        "executed_action",
        "policy_action",
        "is_intervention",
        "success",
        "done",
        "terminal_reason",
    )
    missing = [key for key in required if key not in episode]
    if missing:
        raise KeyError(f"offline episode {source_index} is missing fields {missing}.")

    executed = _as_array(
        episode["executed_action"],
        name=f"episode[{source_index}].executed_action",
        ndim=2,
    )
    policy = _as_array(
        episode["policy_action"],
        name=f"episode[{source_index}].policy_action",
        ndim=2,
    )
    interventions = _as_array(
        episode["is_intervention"],
        name=f"episode[{source_index}].is_intervention",
        ndim=1,
    ).astype(np.bool_, copy=False)
    success = _as_array(
        episode["success"], name=f"episode[{source_index}].success", ndim=1
    ).astype(np.bool_, copy=False)
    done = _as_array(
        episode["done"], name=f"episode[{source_index}].done", ndim=1
    ).astype(np.bool_, copy=False)
    length = int(executed.shape[0])
    if length <= 0:
        raise ValueError(f"offline episode {source_index} is empty.")
    if policy.shape != executed.shape:
        raise ValueError(
            f"episode {source_index} policy/executed action shapes differ: "
            f"{policy.shape} vs {executed.shape}."
        )
    for name, array in (
        ("is_intervention", interventions),
        ("success", success),
        ("done", done),
    ):
        if int(array.shape[0]) != length:
            raise ValueError(
                f"episode {source_index} {name} length={array.shape[0]} != {length}."
            )

    for obs_key in ("obs", "next_obs"):
        observations = episode[obs_key]
        if not isinstance(observations, Mapping) or "state" not in observations:
            raise TypeError(f"episode {source_index} {obs_key} must contain state.")
        for name in ("state", *camera_names):
            if name not in observations:
                raise KeyError(
                    f"episode {source_index} {obs_key} is missing {name!r}."
                )
            if int(np.asarray(observations[name]).shape[0]) != length:
                raise ValueError(
                    f"episode {source_index} {obs_key}[{name!r}] has inconsistent length."
                )

    policy_mask = ~interventions
    if bool(policy_mask.any()) and not np.allclose(
        executed[policy_mask], policy[policy_mask], rtol=1e-5, atol=1e-6
    ):
        delta = float(np.max(np.abs(executed[policy_mask] - policy[policy_mask])))
        raise ValueError(
            f"episode {source_index} has non-intervention executed/policy action mismatch "
            f"(max_abs={delta:.3e})."
        )

    reason = str(episode["terminal_reason"])
    success_indices = np.flatnonzero(success)
    if reason == "success":
        if success_indices.tolist() != [length - 1]:
            raise ValueError(
                f"episode {source_index} terminal_reason='success' requires only the final "
                f"success flag, got indices {success_indices.tolist()}."
            )
    elif success_indices.size:
        raise ValueError(
            f"episode {source_index} terminal_reason={reason!r} contains success flags."
        )
    if not bool(done[-1]) or bool(done[:-1].any()):
        raise ValueError(
            f"episode {source_index} must have exactly one terminal done flag at the final frame."
        )


def validate_offline_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the NumPy episode schema without tensor computation."""
    if not isinstance(payload, dict):
        raise TypeError(
            f"offline episodes payload must be a dict, got {type(payload).__name__}."
        )
    schema = int(payload.get("schema_version", -1))
    if schema != OFFLINE_EPISODE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported offline episode schema {schema}; "
            f"expected {OFFLINE_EPISODE_SCHEMA_VERSION}."
        )
    episodes = list(payload.get("episodes", []))
    if not episodes:
        raise ValueError("offline episodes payload contains no episodes.")
    camera_names = _camera_names(payload)
    if not camera_names:
        raise ValueError("offline episodes contain no camera names.")
    for index, episode in enumerate(episodes):
        if not isinstance(episode, Mapping):
            raise TypeError(f"offline episode {index} must be a mapping.")
        _validate_episode(episode, source_index=index, camera_names=camera_names)
    validated = dict(payload)
    validated["episodes"] = episodes
    validated["camera_names"] = camera_names
    return validated


def load_offline_episodes(path: str | Path) -> dict[str, Any]:
    """Load and validate the collected ``offline_episodes.pt`` payload."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"offline episodes not found: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    validated = validate_offline_payload(payload)
    validated["_resolved_path"] = str(source)
    return validated


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


__all__ = [
    "OFFLINE_EPISODE_SCHEMA_VERSION",
    "PolicySegment",
    "load_offline_episodes",
    "split_policy_segments",
    "validate_offline_payload",
]
