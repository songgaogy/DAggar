"""Latent shard loading and P/U/calibration pool composition."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


PRETRAIN_SCHEMA_VERSION = 1


@dataclass
class LatentTrajectory:
    """A trajectory-boundary-preserving latent shard."""

    features: torch.Tensor
    pool: str
    source: str
    identifier: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DiscriminatorPools:
    """Positive, unlabeled, and held-out calibration latent trajectories."""

    positive: list[LatentTrajectory] = field(default_factory=list)
    unlabeled: list[LatentTrajectory] = field(default_factory=list)
    calibration: list[LatentTrajectory] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


def pool_stats(pools: DiscriminatorPools) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    dimensions: set[int] = set()
    for name in ("positive", "unlabeled", "calibration"):
        trajectories = getattr(pools, name)
        for trajectory in trajectories:
            dimensions.add(int(trajectory.features.shape[1]))
        stats[f"{name}_trajectories"] = len(trajectories)
        stats[f"{name}_frames"] = int(
            sum(int(trajectory.features.shape[0]) for trajectory in trajectories)
        )
    if len(dimensions) > 1:
        raise ValueError(
            f"Discriminator latent dimensions do not match: {sorted(dimensions)}"
        )
    stats["latent_dim"] = 0 if not dimensions else int(next(iter(dimensions)))
    return stats


def _load_shard(
    root: Path, entry: Mapping[str, Any], *, pool: str
) -> LatentTrajectory:
    path = (root / str(entry["path"])).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise FileNotFoundError(f"Invalid or missing pretrain shard: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, Mapping)
        or int(payload.get("schema_version", -1)) != PRETRAIN_SCHEMA_VERSION
    ):
        raise ValueError(f"Unsupported pretrain shard schema: {path}")
    latent = payload.get("latent")
    if not torch.is_tensor(latent) or latent.ndim != 2 or latent.dtype != torch.float32:
        raise TypeError(f"Shard {path} latent must be float32 (T, D).")
    if (
        int(latent.shape[0]) != int(entry["num_frames"])
        or int(latent.shape[1]) != int(entry["latent_dim"])
    ):
        raise ValueError(f"Shard {path} shape does not match manifest entry.")
    frame_indices = np.asarray(payload.get("frame_indices"))
    expected_indices = np.arange(int(latent.shape[0]), dtype=np.int64)
    if frame_indices.ndim != 1 or not np.array_equal(frame_indices, expected_indices):
        raise ValueError(
            f"Shard {path} frame_indices must be contiguous [0, num_frames)."
        )
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise TypeError(f"Shard {path} provenance must be a mapping.")
    if str(provenance.get("video_id", entry["video_id"])) != str(entry["video_id"]):
        raise ValueError(f"Shard {path} video_id does not match its manifest entry.")
    return LatentTrajectory(
        features=latent,
        pool=pool,
        source="pretrain",
        identifier=str(entry["video_id"]),
        metadata=dict(provenance),
    )


def load_pretrain_pools(
    path: str | Path,
) -> tuple[DiscriminatorPools, dict[str, Any]]:
    """Load extracted pretrain shards without merging trajectory boundaries."""
    raw = Path(path).expanduser().resolve()
    root = raw if raw.is_dir() else raw.parent
    manifest_path = root / "manifest.json" if raw.is_dir() else raw
    if not manifest_path.is_file():
        raise FileNotFoundError(f"pretrain manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if int(manifest.get("schema_version", -1)) != PRETRAIN_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported pretrain manifest schema {manifest.get('schema_version')}."
        )
    split_map = {
        "positive_train": "positive",
        "unlabeled_train": "unlabeled",
        "positive_calib": "calibration",
    }
    split_entries: dict[str, list[Mapping[str, Any]]] = {}
    seen_paths: set[str] = set()
    split_ids: dict[str, set[str]] = {}
    for split_name in split_map:
        entries = manifest.get("splits", {}).get(split_name)
        if not isinstance(entries, list) or not entries:
            raise ValueError(
                f"pretrain manifest split {split_name!r} is empty or invalid."
            )
        if any(not isinstance(entry, Mapping) for entry in entries):
            raise TypeError(
                f"pretrain manifest split {split_name!r} contains a non-mapping entry."
            )
        ids = [str(entry["video_id"]) for entry in entries]
        paths = [str(entry["path"]) for entry in entries]
        if len(ids) != len(set(ids)):
            raise ValueError(
                f"pretrain manifest split {split_name!r} contains duplicate video IDs."
            )
        path_overlap = sorted(set(paths) & seen_paths)
        if len(paths) != len(set(paths)) or path_overlap:
            raise ValueError(
                f"pretrain manifest contains duplicate shard paths in {split_name!r}: "
                f"{path_overlap or paths}."
            )
        seen_paths.update(paths)
        split_ids[split_name] = set(ids)
        split_entries[split_name] = entries
    split_names = list(split_map)
    for left_index, left_name in enumerate(split_names):
        for right_name in split_names[left_index + 1 :]:
            overlap = sorted(split_ids[left_name] & split_ids[right_name])
            if overlap:
                raise ValueError(
                    f"pretrain manifest splits {left_name!r} and {right_name!r} "
                    f"overlap in video IDs: {overlap}."
                )
    pools = DiscriminatorPools()
    for split_name, pool_name in split_map.items():
        entries = split_entries[split_name]
        target = getattr(pools, pool_name)
        target.extend(
            _load_shard(root, entry, pool=pool_name) for entry in entries
        )
    pools.stats = pool_stats(pools)
    return pools, manifest


def combine_pools(
    pretrain: DiscriminatorPools,
    online: DiscriminatorPools,
    *,
    use_only_offline: bool = False,
) -> DiscriminatorPools:
    """Naturally mix frames within P/U and retain pretrain-only calibration."""
    if online.calibration:
        raise ValueError("Online calibration is disabled; use pretrain positive_calib only.")
    combined = DiscriminatorPools(
        positive=[*pretrain.positive, *online.positive],
        unlabeled=list(online.unlabeled)
        if use_only_offline
        else [*pretrain.unlabeled, *online.unlabeled],
        calibration=list(pretrain.calibration),
    )
    if not combined.positive or not combined.unlabeled or not combined.calibration:
        raise ValueError("Combined discriminator pools require non-empty P, U, and calibration.")
    combined.stats = {
        **pool_stats(combined),
        "pretrain": dict(pretrain.stats),
        "online": dict(online.stats),
        "use_only_offline": bool(use_only_offline),
        "mixing": "natural_uniform_frames_within_balanced_pu",
        "calibration_source": "pretrain_positive_calib_only",
    }
    return combined


def feature_tensors(
    trajectories: Sequence[LatentTrajectory],
) -> list[torch.Tensor]:
    return [trajectory.features for trajectory in trajectories]


__all__ = [
    "PRETRAIN_SCHEMA_VERSION",
    "DiscriminatorPools",
    "LatentTrajectory",
    "combine_pools",
    "feature_tensors",
    "load_pretrain_pools",
    "pool_stats",
]
