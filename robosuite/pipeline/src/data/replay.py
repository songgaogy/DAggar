from __future__ import annotations

import json
import shutil
import threading
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np


ONLINE_REPLAY_SCHEMA_VERSION = 1
ONLINE_ARRAYS = (
    "visual_features",
    "proprio",
    "next_visual_features",
    "next_proprio",
    "actions",
    "rewards",
    "dones",
    "executed_length",
)


class ReplaySource(Protocol):
    def __len__(self) -> int: ...

    def gather(self, indices: np.ndarray) -> dict[str, np.ndarray]: ...


@dataclass(frozen=True)
class CompactTransition:
    """One DSRL-SAC macro transition; actions are policy latent noise."""

    visual_features: np.ndarray
    proprio: np.ndarray
    next_visual_features: np.ndarray
    next_proprio: np.ndarray
    actions: np.ndarray
    reward: float
    done: bool
    executed_length: int


class OnlineReplay:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("Online replay capacity must be positive.")
        self.capacity = int(capacity)
        self.size = 0
        self.position = 0
        self.arrays: dict[str, np.ndarray] = {}
        self._lock = threading.RLock()

    def __len__(self) -> int:
        return self.size

    def _as_values(self, transition: CompactTransition | Mapping[str, Any]) -> dict[str, np.ndarray]:
        get = transition.get if isinstance(transition, Mapping) else lambda key: getattr(transition, key)
        executed_length = int(get("executed_length"))
        actions = np.asarray(get("actions"), dtype=np.float32)
        reward = float(get("reward"))
        if reward not in (0.0, 1.0):
            raise ValueError("Online reward must use the binary 0/1 success schema.")
        if actions.ndim != 2:
            raise ValueError("Online latent actions must have shape [horizon, action_dim].")
        if not 0 < executed_length <= actions.shape[0]:
            raise ValueError("Online executed_length must be within the latent action horizon.")
        return {
            "visual_features": np.asarray(get("visual_features")),
            "proprio": np.asarray(get("proprio"), dtype=np.float32),
            "next_visual_features": np.asarray(get("next_visual_features")),
            "next_proprio": np.asarray(get("next_proprio"), dtype=np.float32),
            "actions": actions,
            "rewards": np.asarray([reward], dtype=np.float32),
            "dones": np.asarray([get("done")], dtype=np.bool_),
            "executed_length": np.asarray([executed_length], dtype=np.uint8),
        }

    def add(self, transition: CompactTransition | Mapping[str, Any]) -> None:
        values = self._as_values(transition)
        with self._lock:
            if not self.arrays:
                self.arrays = {
                    key: np.empty((self.capacity, *value.shape), dtype=value.dtype)
                    for key, value in values.items()
                }
            for key, value in values.items():
                target = self.arrays.get(key)
                if target is None or target.shape[1:] != value.shape or target.dtype != value.dtype:
                    raise ValueError(f"Online transition field '{key}' is incompatible with replay schema.")
                target[self.position] = value
            self.position = (self.position + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def gather(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        indexes = np.asarray(indices, dtype=np.int64).reshape(-1)
        with self._lock:
            if indexes.size and (int(indexes.min()) < 0 or int(indexes.max()) >= self.size):
                raise IndexError("Online replay sample index is out of range.")
            return {key: np.asarray(value[indexes]) for key, value in self.arrays.items()}

    def snapshot(self, path: str | Path) -> dict[str, Any]:
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
        temporary.mkdir()
        with self._lock:
            for key, value in self.arrays.items():
                np.save(temporary / f"{key}.npy", value[: self.size], allow_pickle=False)
            manifest = {
                "schema_version": ONLINE_REPLAY_SCHEMA_VERSION,
                "capacity": self.capacity,
                "size": self.size,
                "position": self.position,
                "arrays": {
                    key: {"shape": list(value[: self.size].shape), "dtype": str(value.dtype)}
                    for key, value in self.arrays.items()
                },
            }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        stale = None
        if destination.exists():
            stale = destination.parent / f".{destination.name}.stale-{uuid.uuid4().hex}"
            destination.replace(stale)
        temporary.replace(destination)
        if stale is not None:
            shutil.rmtree(stale)
        return manifest

    @classmethod
    def load_snapshot(cls, path: str | Path) -> "OnlineReplay":
        source = Path(path).expanduser().resolve()
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema_version") != ONLINE_REPLAY_SCHEMA_VERSION:
            raise ValueError("Online replay snapshot schema version mismatch.")
        replay = cls(int(manifest["capacity"]))
        replay.size = int(manifest["size"])
        replay.position = int(manifest["position"])
        if not 0 <= replay.size <= replay.capacity:
            raise ValueError("Online replay snapshot has invalid size.")
        if not 0 <= replay.position < replay.capacity:
            raise ValueError("Online replay snapshot has invalid position.")
        if replay.size < replay.capacity and replay.position != replay.size:
            raise ValueError("Online replay snapshot position is inconsistent with its size.")
        array_specs = manifest.get("arrays")
        if not isinstance(array_specs, Mapping):
            raise ValueError("Online replay snapshot array manifest is invalid.")
        if array_specs and set(array_specs) != set(ONLINE_ARRAYS):
            raise ValueError("Online replay snapshot array schema mismatch.")
        for key, spec in array_specs.items():
            compact = np.load(source / f"{key}.npy", mmap_mode="r", allow_pickle=False)
            if list(compact.shape) != spec.get("shape") or str(compact.dtype) != spec.get("dtype"):
                raise ValueError(f"Online replay snapshot field '{key}' metadata mismatch.")
            if len(compact) != replay.size:
                raise ValueError(f"Online replay snapshot field '{key}' has invalid length.")
            target = np.empty((replay.capacity, *compact.shape[1:]), dtype=compact.dtype)
            target[: replay.size] = compact
            replay.arrays[key] = target
        return replay


class UniformReplay:
    """Uniformly sample the union of immutable warmup and online transitions."""

    def __init__(
        self,
        warmup: ReplaySource,
        online: OnlineReplay,
        *,
        seed: int,
    ) -> None:
        self.warmup = warmup
        self.online = online
        self.rng = np.random.default_rng(int(seed))
        self._lock = threading.RLock()

    def __len__(self) -> int:
        return len(self.warmup) + len(self.online)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        total = len(self)
        if total == 0:
            raise ValueError("Cannot sample an empty replay.")
        with self._lock:
            combined = self.rng.integers(0, total, size=int(batch_size), dtype=np.int64)
        warmup_mask = combined < len(self.warmup)
        warmup_positions = np.flatnonzero(warmup_mask)
        online_positions = np.flatnonzero(~warmup_mask)
        parts: list[tuple[np.ndarray, dict[str, np.ndarray]]] = []
        if warmup_positions.size:
            parts.append((warmup_positions, self.warmup.gather(combined[warmup_positions])))
        if online_positions.size:
            parts.append(
                (online_positions, self.online.gather(combined[online_positions] - len(self.warmup)))
            )
        keys = parts[0][1].keys()
        batch = {
            key: np.empty((batch_size, *parts[0][1][key].shape[1:]), dtype=parts[0][1][key].dtype)
            for key in keys
        }
        for positions, part in parts:
            if part.keys() != keys:
                raise ValueError("Warmup and online replay schemas differ.")
            for key in keys:
                if (
                    part[key].shape[1:] != batch[key].shape[1:]
                    or part[key].dtype != batch[key].dtype
                ):
                    raise ValueError(f"Warmup and online field '{key}' schemas differ.")
                batch[key][positions] = part[key]
        return batch

    def state_dict(self) -> dict[str, Any]:
        with self._lock:
            return {"bit_generator_state": self.rng.bit_generator.state}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        with self._lock:
            self.rng.bit_generator.state = state["bit_generator_state"]
