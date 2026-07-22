"""Collected-episode schema validation and round-aware loading."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

OFFLINE_EPISODE_SCHEMA_VERSION = 1


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
    if int(policy.shape[1]) <= 0:
        raise ValueError(f"episode {source_index} policy_action has empty action dimension.")
    if not bool(np.isfinite(policy).all()):
        raise ValueError(f"episode {source_index} policy_action contains non-finite values.")
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


_CONSISTENT_FIELDS = ("task_name", "camera_names", "img_height", "img_width", "action_dim")


def merge_episode_payloads(
    payloads: Sequence[Mapping[str, Any]],
    *,
    source_rounds: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Merge validated round payloads and renumber episodes globally."""
    if not payloads:
        raise ValueError("At least one collected-episode payload is required.")
    rounds = list(range(len(payloads))) if source_rounds is None else [int(v) for v in source_rounds]
    if len(rounds) != len(payloads):
        raise ValueError("source_rounds must match the number of payloads.")
    if len(set(rounds)) != len(rounds):
        raise ValueError("source_rounds must be unique.")

    validated = [validate_offline_payload(dict(payload)) for payload in payloads]
    reference = validated[0]
    for index, payload in enumerate(validated[1:], start=1):
        for field in _CONSISTENT_FIELDS:
            left = reference.get(field)
            right = payload.get(field)
            if field == "camera_names":
                left = [str(value) for value in left or []]
                right = [str(value) for value in right or []]
            if left != right:
                raise ValueError(
                    f"Round payload {index} {field}={right!r} does not match {left!r}."
                )

    episodes: list[dict[str, Any]] = []
    round_stats: list[dict[str, int]] = []
    for round_index, payload in zip(rounds, validated):
        transition_count = 0
        for episode in payload["episodes"]:
            copied = dict(episode)
            copied["episode_index"] = len(episodes)
            copied["source_round"] = int(round_index)
            copied["source_episode_index"] = int(episode.get("episode_index", 0))
            length = int(np.asarray(copied["executed_action"]).shape[0])
            transition_count += length
            episodes.append(copied)
        round_stats.append(
            {
                "round": int(round_index),
                "episodes": int(len(payload["episodes"])),
                "transitions": int(transition_count),
            }
        )

    merged = dict(reference)
    merged.update(
        {
            "episodes": episodes,
            "num_episodes": len(episodes),
            "num_transitions": int(sum(item["transitions"] for item in round_stats)),
            "source_rounds": rounds,
            "round_stats": round_stats,
        }
    )
    return validate_offline_payload(merged)


@lru_cache(maxsize=2)
def _load_round_episode_payloads_cached(
    cache_key: tuple[tuple[str, int, int], ...],
) -> dict[str, Any]:
    resolved = [Path(path) for path, _size, _mtime_ns in cache_key]
    payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in resolved]
    merged = merge_episode_payloads(payloads, source_rounds=list(range(len(payloads))))
    merged["_resolved_paths"] = [str(path) for path in resolved]
    return merged


def load_round_episode_payloads(paths: Sequence[str | Path]) -> dict[str, Any]:
    """Load once per process and return one validated cumulative round view."""
    resolved = [Path(path).expanduser().resolve() for path in paths]
    missing = [str(path) for path in resolved if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Collected-episode files do not exist: {missing}")
    cache_key = tuple(
        (str(path), int(path.stat().st_size), int(path.stat().st_mtime_ns))
        for path in resolved
    )
    return _load_round_episode_payloads_cached(cache_key)
