from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np


WARMUP_SCHEMA_VERSION = 1
WARMUP_ARRAYS = (
    "visual_features",
    "proprio",
    "next_visual_features",
    "next_proprio",
    "actions",
    "rewards",
    "dones",
    "executed_length",
)


class WarmupValidationError(RuntimeError):
    pass


def _json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_identity(path: str | Path) -> dict[str, Any]:
    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Base-policy checkpoint does not exist: {checkpoint}")
    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    stat = checkpoint.stat()
    return {
        "path": os.fspath(checkpoint),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def build_warmup_fingerprint(
    *,
    task_name: str,
    warmup_seed: int,
    base_checkpoint: str | Path,
    env_metadata: Mapping[str, Any],
    camera_names: Sequence[str],
    action_horizon: int,
    ode_config: Mapping[str, Any],
    feature_schema: Mapping[str, Any],
    reward_schema: str | Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Build the identity shared by warmup collection and training runs."""

    if action_horizon <= 0 or action_horizon > np.iinfo(np.uint8).max:
        raise ValueError("action_horizon must be in [1, 255].")
    ingredients = {
        "schema_version": WARMUP_SCHEMA_VERSION,
        "task_name": str(task_name),
        "warmup_seed": int(warmup_seed),
        "base_checkpoint": _checkpoint_identity(base_checkpoint),
        "env_metadata": dict(env_metadata),
        "camera_names": list(camera_names),
        "action_horizon": int(action_horizon),
        "ode_config": dict(ode_config),
        "feature_schema": dict(feature_schema),
        "reward_schema": reward_schema,
    }
    try:
        fingerprint = _json_hash(ingredients)
    except (TypeError, ValueError) as error:
        raise TypeError("Warmup fingerprint ingredients must be JSON serializable.") from error
    return fingerprint, ingredients


def _validated_arrays(arrays: Mapping[str, np.ndarray]) -> tuple[dict[str, np.ndarray], int]:
    if set(arrays) != set(WARMUP_ARRAYS):
        missing = sorted(set(WARMUP_ARRAYS).difference(arrays))
        extra = sorted(set(arrays).difference(WARMUP_ARRAYS))
        raise WarmupValidationError(f"Warmup array schema mismatch; missing={missing}, extra={extra}.")
    values = {key: np.asarray(arrays[key]) for key in WARMUP_ARRAYS}
    transition_count = len(values[WARMUP_ARRAYS[0]])
    if transition_count == 0:
        raise WarmupValidationError("Warmup replay must contain at least one transition.")
    if any(len(value) != transition_count for value in values.values()):
        raise WarmupValidationError("Warmup arrays have inconsistent transition counts.")
    for key in ("rewards", "dones", "executed_length"):
        if values[key].shape != (transition_count, 1):
            raise WarmupValidationError(f"Warmup array '{key}' must have shape [N, 1].")
    if values["actions"].ndim != 3:
        raise WarmupValidationError("Warmup actions must have shape [N, horizon, action_dim].")
    for key in ("proprio", "next_proprio", "actions", "rewards"):
        if values[key].dtype != np.float32:
            raise WarmupValidationError(f"Warmup array '{key}' must have float32 dtype.")
    if values["visual_features"].dtype != values["next_visual_features"].dtype:
        raise WarmupValidationError("Current and next visual feature dtypes differ.")
    if not np.all(np.logical_or(values["rewards"] == 0, values["rewards"] == 1)):
        raise WarmupValidationError("Warmup rewards must use the binary 0/1 success schema.")
    if not np.issubdtype(values["dones"].dtype, np.bool_):
        raise WarmupValidationError("Warmup dones must have boolean dtype.")
    executed = values["executed_length"]
    if not np.issubdtype(executed.dtype, np.integer) or np.any(executed <= 0):
        raise WarmupValidationError("Warmup executed_length must contain positive integers.")
    if np.any(executed > values["actions"].shape[1]):
        raise WarmupValidationError("Warmup executed_length exceeds the latent action horizon.")
    return values, transition_count


def _validated_boundaries(boundaries: Sequence[int], transition_count: int) -> list[int]:
    result = [int(value) for value in boundaries]
    if len(result) < 2 or result[0] != 0 or result[-1] != transition_count:
        raise WarmupValidationError("Episode boundaries must start at 0 and end at transition_count.")
    if any(left >= right for left, right in zip(result, result[1:], strict=False)):
        raise WarmupValidationError("Episode boundaries must be strictly increasing.")
    return result


class WarmupReplay:
    """Immutable, memory-mapped replay collected before online SAC updates."""

    def __init__(self, path: str | Path, *, expected_fingerprint: str | None = None) -> None:
        self.path = Path(path).expanduser().resolve()
        manifest_path = self.path / "manifest.json"
        if not manifest_path.is_file():
            raise WarmupValidationError(f"Warmup manifest is missing: {manifest_path}")
        try:
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise WarmupValidationError(f"Warmup manifest is invalid: {manifest_path}") from error
        if self.manifest.get("schema_version") != WARMUP_SCHEMA_VERSION:
            raise WarmupValidationError("Warmup schema version mismatch.")
        fingerprint = self.manifest.get("fingerprint")
        ingredients = self.manifest.get("fingerprint_ingredients")
        if not isinstance(fingerprint, str) or not isinstance(ingredients, Mapping):
            raise WarmupValidationError("Warmup fingerprint metadata is missing.")
        if _json_hash(ingredients) != fingerprint:
            raise WarmupValidationError("Warmup fingerprint ingredients do not match the fingerprint.")
        if expected_fingerprint is not None and fingerprint != expected_fingerprint:
            raise WarmupValidationError("Warmup fingerprint mismatch.")
        transition_count = self.manifest.get("transition_count")
        if not isinstance(transition_count, int) or transition_count <= 0:
            raise WarmupValidationError("Warmup transition_count is invalid.")
        boundaries = _validated_boundaries(
            self.manifest.get("episode_boundaries", ()), transition_count
        )
        if self.manifest.get("episode_count") != len(boundaries) - 1:
            raise WarmupValidationError("Warmup episode_count does not match episode boundaries.")
        completed = self.manifest.get("completed_by_worker")
        if (
            not isinstance(completed, list)
            or any(not isinstance(value, int) or value < 0 for value in completed)
            or sum(completed) != self.manifest["episode_count"]
        ):
            raise WarmupValidationError("Warmup completed_by_worker is invalid.")
        specs = self.manifest.get("arrays")
        if not isinstance(specs, Mapping) or set(specs) != set(WARMUP_ARRAYS):
            raise WarmupValidationError("Warmup manifest array schema mismatch.")
        loaded: dict[str, np.ndarray] = {}
        for name in WARMUP_ARRAYS:
            array_path = self.path / f"{name}.npy"
            if not array_path.is_file():
                raise WarmupValidationError(f"Warmup array is missing: {array_path}")
            try:
                value = np.load(array_path, mmap_mode="r", allow_pickle=False)
            except (OSError, ValueError) as error:
                raise WarmupValidationError(f"Warmup array is invalid: {array_path}") from error
            spec = specs[name]
            if not isinstance(spec, Mapping):
                raise WarmupValidationError(f"Warmup array metadata is invalid: {name}")
            if list(value.shape) != spec.get("shape") or str(value.dtype) != spec.get("dtype"):
                raise WarmupValidationError(f"Warmup array metadata mismatch: {name}")
            loaded[name] = value
        _validated_arrays(loaded)
        horizon = ingredients.get("action_horizon")
        if horizon != loaded["actions"].shape[1]:
            raise WarmupValidationError("Warmup latent action horizon does not match fingerprint.")
        if self.manifest.get("macro_steps") != transition_count:
            raise WarmupValidationError("Warmup macro_steps does not match transition_count.")
        expected_primitive_steps = int(np.asarray(loaded["executed_length"]).sum())
        if self.manifest.get("primitive_steps") != expected_primitive_steps:
            raise WarmupValidationError("Warmup primitive_steps does not match executed_length.")
        vector_steps = self.manifest.get("vector_steps")
        if not isinstance(vector_steps, int) or vector_steps <= 0:
            raise WarmupValidationError("Warmup vector_steps is invalid.")
        self.arrays = MappingProxyType(loaded)
        self.episode_boundaries = tuple(boundaries)

    def __len__(self) -> int:
        return int(self.manifest["transition_count"])

    def gather(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        indexes = np.asarray(indices, dtype=np.int64).reshape(-1)
        if indexes.size and (int(indexes.min()) < 0 or int(indexes.max()) >= len(self)):
            raise IndexError("Warmup replay sample index is out of range.")
        return {key: np.asarray(value[indexes]) for key, value in self.arrays.items()}


def save_warmup_cache(
    cache_root: str | Path,
    *,
    fingerprint: str,
    fingerprint_ingredients: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    episode_boundaries: Sequence[int],
    metadata: Mapping[str, Any] | None = None,
) -> WarmupReplay:
    """Atomically publish a fingerprinted warmup cache, or reuse an exact match."""

    if _json_hash(fingerprint_ingredients) != fingerprint:
        raise WarmupValidationError("Fingerprint ingredients do not match fingerprint.")
    root = Path(cache_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / fingerprint
    if destination.exists():
        return WarmupReplay(destination, expected_fingerprint=fingerprint)

    values, transition_count = _validated_arrays(arrays)
    boundaries = _validated_boundaries(episode_boundaries, transition_count)
    extra_metadata = dict(metadata or {})
    reserved = {
        "schema_version",
        "fingerprint",
        "fingerprint_ingredients",
        "transition_count",
        "episode_count",
        "episode_boundaries",
        "arrays",
    }
    if conflict := reserved.intersection(extra_metadata):
        raise WarmupValidationError(f"Warmup metadata uses reserved keys: {sorted(conflict)}")
    required_metadata = {"completed_by_worker", "primitive_steps", "macro_steps", "vector_steps"}
    if missing := required_metadata.difference(extra_metadata):
        raise WarmupValidationError(f"Warmup metadata is missing keys: {sorted(missing)}")
    completed = extra_metadata["completed_by_worker"]
    if (
        not isinstance(completed, list)
        or any(not isinstance(value, int) or value < 0 for value in completed)
        or sum(completed) != len(boundaries) - 1
    ):
        raise WarmupValidationError("Warmup completed_by_worker is invalid.")
    if extra_metadata["macro_steps"] != transition_count:
        raise WarmupValidationError("Warmup macro_steps must equal transition_count.")
    if extra_metadata["primitive_steps"] != int(values["executed_length"].sum()):
        raise WarmupValidationError("Warmup primitive_steps must equal summed executed_length.")
    if not isinstance(extra_metadata["vector_steps"], int) or extra_metadata["vector_steps"] <= 0:
        raise WarmupValidationError("Warmup vector_steps must be a positive integer.")
    try:
        json.dumps(extra_metadata)
    except (TypeError, ValueError) as error:
        raise TypeError("Warmup metadata must be JSON serializable.") from error

    temporary = root / f".{fingerprint}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        for name, value in values.items():
            np.save(temporary / f"{name}.npy", value, allow_pickle=False)
        manifest = {
            "schema_version": WARMUP_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "fingerprint_ingredients": dict(fingerprint_ingredients),
            "transition_count": transition_count,
            "episode_count": len(boundaries) - 1,
            "episode_boundaries": boundaries,
            "arrays": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in values.items()
            },
            **extra_metadata,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        WarmupReplay(temporary, expected_fingerprint=fingerprint)
        try:
            temporary.replace(destination)
        except FileExistsError:
            shutil.rmtree(temporary)
        return WarmupReplay(destination, expected_fingerprint=fingerprint)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_warmup_cache(cache_root: str | Path, fingerprint: str) -> WarmupReplay:
    """Load one shared cache by fingerprint with full manifest validation."""

    return WarmupReplay(Path(cache_root).expanduser().resolve() / fingerprint, expected_fingerprint=fingerprint)
