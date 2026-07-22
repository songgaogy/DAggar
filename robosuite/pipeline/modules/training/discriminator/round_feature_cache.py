"""Serializable per-round frozen discriminator feature cache."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .pools import LatentTrajectory


ROUND_FEATURE_CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RoundFeatureCache:
    """Frozen encoder outputs for the two batch-online GT risks."""

    round_index: int
    cache_key: str
    offline_positive: tuple[LatentTrajectory, ...]
    offline_gt_negative: tuple[LatentTrajectory, ...]


def _validate_header(*, round_index: int, cache_key: str) -> None:
    if isinstance(round_index, bool) or not isinstance(round_index, int):
        raise TypeError("round_index must be an integer.")
    if round_index < 0:
        raise ValueError("round_index must be non-negative.")
    if not isinstance(cache_key, str) or not cache_key:
        raise ValueError("cache_key must be a non-empty string.")


def _trajectory_payload(
    trajectory: LatentTrajectory,
    *,
    expected_input_pool: str,
    serialized_pool: str,
) -> dict[str, Any]:
    if not isinstance(trajectory, LatentTrajectory):
        raise TypeError("Round feature pools must contain LatentTrajectory values.")
    accepted_input_pools = {expected_input_pool}
    if serialized_pool == "offline_positive":
        accepted_input_pools.add("offline_positive")
    if trajectory.pool not in accepted_input_pools:
        raise ValueError(
            f"Expected trajectory pool in {sorted(accepted_input_pools)!r}, "
            f"got {trajectory.pool!r}."
        )
    features = trajectory.features
    if not torch.is_tensor(features) or features.ndim != 2 or features.shape[0] == 0:
        raise ValueError(
            f"Trajectory {trajectory.identifier!r} features must have non-empty "
            "shape (N, D)."
        )
    return {
        "features": features,
        "pool": serialized_pool,
        "source": trajectory.source,
        "identifier": trajectory.identifier,
        "metadata": dict(trajectory.metadata),
    }


def save_round_feature_cache(
    path: str | Path,
    *,
    round_index: int,
    cache_key: str,
    offline_positive: Sequence[LatentTrajectory],
    offline_gt_negative: Sequence[LatentTrajectory],
) -> Path:
    """Atomically save frozen features; head outputs are intentionally excluded."""
    _validate_header(round_index=round_index, cache_key=cache_key)
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": ROUND_FEATURE_CACHE_SCHEMA_VERSION,
        "round_index": round_index,
        "cache_key": cache_key,
        "pools": {
            "offline_positive": [
                _trajectory_payload(
                    item,
                    expected_input_pool="positive",
                    serialized_pool="offline_positive",
                )
                for item in offline_positive
            ],
            "offline_gt_negative": [
                _trajectory_payload(
                    item,
                    expected_input_pool="offline_gt_negative",
                    serialized_pool="offline_gt_negative",
                )
                for item in offline_gt_negative
            ],
        },
    }
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _load_pool(
    raw_pool: Any, *, pool_name: str, device: torch.device
) -> tuple[LatentTrajectory, ...]:
    if not isinstance(raw_pool, list):
        raise TypeError(f"Cached pool {pool_name!r} must be a list.")
    trajectories: list[LatentTrajectory] = []
    for raw in raw_pool:
        if not isinstance(raw, Mapping):
            raise TypeError(f"Cached pool {pool_name!r} contains a non-mapping.")
        if set(raw) != {"features", "pool", "source", "identifier", "metadata"}:
            raise ValueError(f"Cached pool {pool_name!r} has an invalid schema.")
        features = raw["features"]
        if (
            not torch.is_tensor(features)
            or features.device != device
            or features.device.type != "cuda"
            or features.ndim != 2
            or features.shape[0] == 0
        ):
            raise ValueError(
                f"Cached {pool_name!r} features must be non-empty (N, D) on {device}."
            )
        if str(raw["pool"]) != pool_name:
            raise ValueError(f"Cached trajectory pool does not match {pool_name!r}.")
        metadata = raw["metadata"]
        if not isinstance(metadata, Mapping):
            raise TypeError("Cached trajectory metadata must be a mapping.")
        trajectories.append(
            LatentTrajectory(
                features=features,
                pool=pool_name,
                source=str(raw["source"]),
                identifier=str(raw["identifier"]),
                metadata=dict(metadata),
            )
        )
    return tuple(trajectories)


def load_round_feature_cache(
    path: str | Path,
    *,
    expected_round_index: int,
    expected_cache_key: str,
    device: str | torch.device,
) -> RoundFeatureCache | None:
    """Load a cache hit onto CUDA; return ``None`` for absence or key mismatch."""
    _validate_header(
        round_index=expected_round_index,
        cache_key=expected_cache_key,
    )
    resolved_device = torch.device(device)
    if resolved_device.type != "cuda":
        raise ValueError(f"Round feature cache loading requires CUDA, got {device!r}.")
    if not torch.cuda.is_available():
        raise RuntimeError("Round feature cache loading requires available CUDA.")
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        return None
    payload = torch.load(source, map_location=resolved_device, weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("Round feature cache payload must be a mapping.")
    if int(payload.get("schema_version", -1)) != ROUND_FEATURE_CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported round feature cache schema {payload.get('schema_version')}."
        )
    if int(payload.get("round_index", -1)) != expected_round_index:
        return None
    if str(payload.get("cache_key", "")) != expected_cache_key:
        return None
    if set(payload) != {"schema_version", "round_index", "cache_key", "pools"}:
        raise ValueError("Round feature cache contains unsupported fields.")
    pools = payload["pools"]
    if not isinstance(pools, Mapping) or set(pools) != {
        "offline_positive",
        "offline_gt_negative",
    }:
        raise ValueError("Round feature cache must contain exactly the two GT pools.")
    return RoundFeatureCache(
        round_index=expected_round_index,
        cache_key=expected_cache_key,
        offline_positive=_load_pool(
            pools["offline_positive"],
            pool_name="offline_positive",
            device=resolved_device,
        ),
        offline_gt_negative=_load_pool(
            pools["offline_gt_negative"],
            pool_name="offline_gt_negative",
            device=resolved_device,
        ),
    )


__all__ = [
    "ROUND_FEATURE_CACHE_SCHEMA_VERSION",
    "RoundFeatureCache",
    "load_round_feature_cache",
    "save_round_feature_cache",
]
