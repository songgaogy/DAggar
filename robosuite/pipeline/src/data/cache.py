from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .legacy import LegacyBuffer


CACHE_SCHEMA_VERSION = 1
FEATURE_KEYS = (
    "dino_cls",
    "proprio",
    "task_scene_cond",
    "context_tokens",
    "context_padding_mask",
)
TRANSITION_ARRAYS = (
    "state_index",
    "next_state_index",
    "actions",
    "rewards",
    "dones",
    "executed_length",
)


class CacheValidationError(RuntimeError):
    pass


def _json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_identity(path: str | Path, *, include_digest: bool) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    stat = source.stat()
    identity: dict[str, Any] = {
        "path": os.fspath(source),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if include_digest:
        digest = hashlib.sha256()
        with source.open("rb") as handle:
            while block := handle.read(8 << 20):
                digest.update(block)
        identity["sha256"] = digest.hexdigest()
    return identity


def build_cache_fingerprint(
    *,
    source_path: str | Path,
    metadata_path: str | Path,
    dino_weights: str | Path,
    flow_checkpoint: str | Path,
    camera_names: Sequence[str],
    image_size: int,
    normalizer: Mapping[str, Any],
    prompt: str,
    action_horizon: int,
    ode_steps: int,
) -> tuple[str, dict[str, Any]]:
    """Return a stable fingerprint without hashing the multi-gigabyte replay itself."""

    meta_path = Path(metadata_path).expanduser().resolve()
    ingredients = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "source": _file_identity(source_path, include_digest=False),
        "metadata": {
            **_file_identity(meta_path, include_digest=False),
            "sha256": hashlib.sha256(meta_path.read_bytes()).hexdigest(),
        },
        "dino_weights": _file_identity(dino_weights, include_digest=True),
        "flow_checkpoint": _file_identity(flow_checkpoint, include_digest=True),
        "camera_names": list(camera_names),
        "image_size": int(image_size),
        "normalizer": normalizer,
        "prompt": str(prompt),
        "action_horizon": int(action_horizon),
        "ode_steps": int(ode_steps),
    }
    return _json_hash(ingredients), ingredients


def _field(item: Any, name: str, default: Any = None) -> Any:
    return item.get(name, default) if isinstance(item, Mapping) else getattr(item, name, default)


def _transition_info(item: Any) -> Mapping[str, Any]:
    info = _field(item, "info", None)
    return info if isinstance(info, Mapping) else {}


def _starts_new_fragment(previous: Any, current: Any) -> bool:
    if bool(_field(previous, "done", False)):
        return True
    previous_info = _transition_info(previous)
    current_info = _transition_info(current)
    keys_present = all(
        key in info
        for info in (previous_info, current_info)
        for key in ("episode_index", "episode_step")
    )
    if not keys_present:
        return False
    return (
        current_info["episode_index"] != previous_info["episode_index"]
        or int(current_info["episode_step"]) != int(previous_info["episode_step"]) + 1
    )


def iter_fragments(storage: Sequence[Any]) -> Iterator[tuple[int, int]]:
    if not storage:
        return
    start = 0
    for index in range(1, len(storage)):
        if _starts_new_fragment(storage[index - 1], storage[index]):
            yield start, index
            start = index
    yield start, len(storage)


def _feature_observations(
    storage: Sequence[Any], fragments: Sequence[tuple[int, int]]
) -> Iterator[Any]:
    for start, end in fragments:
        for index in range(start, end):
            observation = _field(storage[index], "obs")
            if observation is None:
                raise ValueError(f"Transition {index} is missing obs.")
            yield observation
        final_observation = _field(storage[end - 1], "next_obs")
        if final_observation is None:
            raise ValueError(f"Transition {end - 1} is missing next_obs.")
        yield final_observation


def _batches(values: Iterator[Any], batch_size: int) -> Iterator[list[Any]]:
    batch: list[Any] = []
    for value in values:
        batch.append(value)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _build_into(
    destination: Path,
    legacy: LegacyBuffer,
    *,
    fingerprint: str,
    fingerprint_ingredients: Mapping[str, Any],
    action_horizon: int,
    feature_batch_size: int,
    feature_extractor: Callable[[Sequence[Any]], Mapping[str, np.ndarray]],
    action_normalizer: Callable[[np.ndarray], np.ndarray],
    action_batch_size: int,
) -> None:
    storage = legacy.storage
    fragments = tuple(iter_fragments(storage))
    state_count = sum(end - start + 1 for start, end in fragments)
    macro_count = sum(max(0, end - start - action_horizon + 1) for start, end in fragments)
    if macro_count == 0:
        raise ValueError("Legacy replay contains no complete macro transitions.")
    declared_macro_count = legacy.metadata.get("num_valid_sequences")
    if declared_macro_count is not None and int(declared_macro_count) != macro_count:
        raise ValueError(
            "Legacy valid sequence count mismatch: "
            f"metadata={declared_macro_count}, computed={macro_count}."
        )

    feature_maps: dict[str, np.memmap] = {}
    write_offset = 0
    for observations in _batches(
        _feature_observations(storage, fragments), feature_batch_size
    ):
        features = feature_extractor(observations)
        missing = set(FEATURE_KEYS).difference(features)
        if missing:
            raise KeyError(f"Feature extractor omitted keys: {sorted(missing)}")
        for key in FEATURE_KEYS:
            value = np.asarray(features[key])
            if value.shape[0] != len(observations):
                raise ValueError(
                    f"Feature '{key}' batch mismatch: expected {len(observations)}, got {value.shape[0]}."
                )
            if key not in feature_maps:
                feature_maps[key] = np.lib.format.open_memmap(
                    destination / f"{key}.npy",
                    mode="w+",
                    dtype=value.dtype,
                    shape=(state_count, *value.shape[1:]),
                )
            elif feature_maps[key].shape[1:] != value.shape[1:]:
                raise ValueError(f"Feature '{key}' shape changed between batches.")
            feature_maps[key][write_offset : write_offset + len(observations)] = value
        write_offset += len(observations)
    if write_offset != state_count:
        raise RuntimeError(f"State count mismatch: expected {state_count}, wrote {write_offset}.")

    first_action = np.asarray(_field(storage[0], "action"), dtype=np.float32).reshape(-1)
    action_dim = int(first_action.shape[0])
    arrays = {
        "state_index": np.lib.format.open_memmap(
            destination / "state_index.npy", mode="w+", dtype=np.int64, shape=(macro_count,)
        ),
        "next_state_index": np.lib.format.open_memmap(
            destination / "next_state_index.npy", mode="w+", dtype=np.int64, shape=(macro_count,)
        ),
        "actions": np.lib.format.open_memmap(
            destination / "actions.npy",
            mode="w+",
            dtype=np.float32,
            shape=(macro_count, action_horizon, action_dim),
        ),
        "rewards": np.lib.format.open_memmap(
            destination / "rewards.npy", mode="w+", dtype=np.float32, shape=(macro_count, 1)
        ),
        "dones": np.lib.format.open_memmap(
            destination / "dones.npy", mode="w+", dtype=np.bool_, shape=(macro_count, 1)
        ),
        "executed_length": np.lib.format.open_memmap(
            destination / "executed_length.npy", mode="w+", dtype=np.uint8, shape=(macro_count, 1)
        ),
    }
    macro_index = 0
    state_base = 0
    for fragment_start, fragment_end in fragments:
        fragment_length = fragment_end - fragment_start
        for local_start in range(max(0, fragment_length - action_horizon + 1)):
            source_start = fragment_start + local_start
            window = range(source_start, source_start + action_horizon)
            actions = []
            rewards = []
            for index in window:
                action = np.asarray(_field(storage[index], "action"), dtype=np.float32).reshape(-1)
                if action.shape != (action_dim,):
                    raise ValueError(f"Transition {index} action shape changed to {action.shape}.")
                actions.append(action)
                reward = _field(storage[index], "reward", None)
                if reward is None:
                    raise ValueError(f"Transition {index} is missing reward.")
                rewards.append(float(reward))
            arrays["state_index"][macro_index] = state_base + local_start
            arrays["next_state_index"][macro_index] = state_base + local_start + action_horizon
            arrays["actions"][macro_index] = np.stack(actions, axis=0)
            arrays["rewards"][macro_index, 0] = math.fsum(rewards)
            arrays["dones"][macro_index, 0] = bool(
                _field(storage[source_start + action_horizon - 1], "done", False)
            )
            arrays["executed_length"][macro_index, 0] = action_horizon
            macro_index += 1
        state_base += fragment_length + 1

    for start in range(0, macro_count, action_batch_size):
        end = min(start + action_batch_size, macro_count)
        normalized = np.asarray(action_normalizer(np.asarray(arrays["actions"][start:end])))
        if normalized.shape != arrays["actions"][start:end].shape:
            raise ValueError(
                "Action normalizer shape mismatch: "
                f"expected {arrays['actions'][start:end].shape}, got {normalized.shape}."
            )
        arrays["actions"][start:end] = normalized.astype(np.float32, copy=False)

    all_arrays = {**feature_maps, **arrays}
    for array in all_arrays.values():
        array.flush()
    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "fingerprint_ingredients": fingerprint_ingredients,
        "state_count": state_count,
        "transition_count": macro_count,
        "action_horizon": int(action_horizon),
        "actions_normalized": True,
        "arrays": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in all_arrays.items()
        },
    }
    _write_manifest(destination / "manifest.json", manifest)


def build_feature_cache(
    cache_root: str | Path,
    legacy: LegacyBuffer,
    *,
    fingerprint: str,
    fingerprint_ingredients: Mapping[str, Any],
    action_horizon: int,
    feature_extractor: Callable[[Sequence[Any]], Mapping[str, np.ndarray]],
    action_normalizer: Callable[[np.ndarray], np.ndarray],
    feature_batch_size: int = 64,
    action_batch_size: int = 1024,
) -> "CacheDataset":
    """Atomically build or reuse a fingerprinted memory-mapped cache.

    Feature and action callbacks must perform tensor computation on CUDA and return
    host arrays only after their device work is complete. This function performs
    indexing, scalar reward aggregation, and memory-mapped I/O only.
    """

    if action_horizon <= 0 or action_horizon > np.iinfo(np.uint8).max:
        raise ValueError("action_horizon must be in [1, 255].")
    if feature_batch_size <= 0:
        raise ValueError("feature_batch_size must be positive.")
    if action_batch_size <= 0:
        raise ValueError("action_batch_size must be positive.")
    legacy_horizon = legacy.payload.get(
        "action_horizon", legacy.metadata.get("action_horizon")
    )
    if legacy_horizon is not None and int(legacy_horizon) != int(action_horizon):
        raise ValueError(
            f"Legacy action horizon is {legacy_horizon}, requested {action_horizon}."
        )
    root = Path(cache_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / fingerprint
    if destination.exists():
        try:
            return CacheDataset(destination, expected_fingerprint=fingerprint)
        except CacheValidationError:
            pass

    temporary = root / f".{fingerprint}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        _build_into(
            temporary,
            legacy,
            fingerprint=fingerprint,
            fingerprint_ingredients=fingerprint_ingredients,
            action_horizon=action_horizon,
            feature_batch_size=feature_batch_size,
            feature_extractor=feature_extractor,
            action_normalizer=action_normalizer,
            action_batch_size=action_batch_size,
        )
        CacheDataset(temporary, expected_fingerprint=fingerprint)
        stale = None
        if destination.exists():
            stale = root / f".{fingerprint}.stale-{uuid.uuid4().hex}"
            destination.replace(stale)
        temporary.replace(destination)
        if stale is not None:
            shutil.rmtree(stale)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return CacheDataset(destination, expected_fingerprint=fingerprint)


class CacheDataset:
    def __init__(self, path: str | Path, *, expected_fingerprint: str | None = None) -> None:
        self.path = Path(path).expanduser().resolve()
        manifest_path = self.path / "manifest.json"
        if not manifest_path.is_file():
            raise CacheValidationError(f"Cache manifest is missing: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
            raise CacheValidationError("Cache schema version mismatch.")
        if expected_fingerprint is not None and self.manifest.get("fingerprint") != expected_fingerprint:
            raise CacheValidationError("Cache fingerprint mismatch.")
        if self.manifest.get("actions_normalized") is not True:
            raise CacheValidationError("Cache actions are not checkpoint-normalized.")
        self.arrays: dict[str, np.ndarray] = {}
        for name, spec in self.manifest.get("arrays", {}).items():
            array_path = self.path / f"{name}.npy"
            if not array_path.is_file():
                raise CacheValidationError(f"Cache array is missing: {array_path}")
            value = np.load(array_path, mmap_mode="r", allow_pickle=False)
            if list(value.shape) != spec["shape"] or str(value.dtype) != spec["dtype"]:
                raise CacheValidationError(f"Cache array metadata mismatch: {name}")
            self.arrays[name] = value
        required = set(FEATURE_KEYS).union(TRANSITION_ARRAYS)
        if missing := required.difference(self.arrays):
            raise CacheValidationError(f"Cache arrays are missing: {sorted(missing)}")

    def __len__(self) -> int:
        return int(self.manifest["transition_count"])

    def gather(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        indexes = np.asarray(indices, dtype=np.int64).reshape(-1)
        if indexes.size and (int(indexes.min()) < 0 or int(indexes.max()) >= len(self)):
            raise IndexError("Cache sample index is out of range.")
        current = self.arrays["state_index"][indexes]
        following = self.arrays["next_state_index"][indexes]
        batch: dict[str, np.ndarray] = {}
        for key in FEATURE_KEYS:
            public_key = "dino_features" if key == "dino_cls" else key
            batch[public_key] = np.asarray(self.arrays[key][current])
            batch[f"next_{public_key}"] = np.asarray(self.arrays[key][following])
        for key in ("actions", "rewards", "dones", "executed_length"):
            batch[key] = np.asarray(self.arrays[key][indexes])
        return batch
