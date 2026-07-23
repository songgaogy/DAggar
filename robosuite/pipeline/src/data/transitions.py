from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


Observation = np.ndarray | Mapping[str, Any]


@dataclass
class Transition:
    obs: Observation
    action: Any
    reward: float
    next_obs: Observation
    done: bool
    grasp_penalty: float | None = None
    is_intervention: bool = False
    info: dict[str, Any] | None = None
    reward_source: str | None = None
    demo_source: str | None = None


@dataclass
class ReplayBufferConfig:
    capacity: int = 200_000
    batch_size: int = 64


def clone_array_tree(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): clone_array_tree(item) for key, item in value.items()}
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().copy()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, (np.generic, float, int, bool)):
        return np.asarray(value).copy()
    return copy.deepcopy(value)


def to_numpy(value: Any, dtype: np.dtype | None = None) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    return array.astype(dtype, copy=False) if dtype is not None else array


def tree_shapes(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: tree_shapes(item) for key, item in value.items()}
    if torch.is_tensor(value) or isinstance(value, np.ndarray):
        return tuple(value.shape)
    return ()


def assert_same_structure(reference: Any, value: Any, path: str = "root") -> None:
    if isinstance(reference, Mapping):
        if not isinstance(value, Mapping):
            raise ValueError(f"{path} expected mapping, got {type(value)!r}.")
        if set(reference) != set(value):
            raise ValueError(
                f"{path} keys mismatch. Expected {sorted(reference)}, got {sorted(value)}."
            )
        for key in reference:
            assert_same_structure(reference[key], value[key], f"{path}.{key}")
        return
    if tree_shapes(reference) != tree_shapes(value):
        raise ValueError(
            f"{path} shape mismatch. Expected {tree_shapes(reference)}, got {tree_shapes(value)}."
        )


class TransitionChunkWriter:
    """Synchronously persist online and intervention transitions in bounded chunks."""

    def __init__(self, root: str | Path, *, chunk_size: int = 1_000) -> None:
        if int(chunk_size) <= 0:
            raise ValueError("chunk_size must be positive.")
        self.root = Path(root)
        self.chunk_size = int(chunk_size)
        self._pending: dict[str, list[Transition]] = {"online": [], "demo": []}
        self._indices = {"online": 0, "demo": 0}
        for role in self._pending:
            directory = self.root / f"{role}_chunks"
            directory.mkdir(parents=True, exist_ok=True)
            existing = sorted(directory.glob("chunk_*.pt"))
            if existing:
                self._indices[role] = int(existing[-1].stem.rsplit("_", 1)[1]) + 1

    def append(self, transition: Transition) -> None:
        self._append_role("online", transition)
        if transition.is_intervention:
            self._append_role("demo", transition)

    def _append_role(self, role: str, transition: Transition) -> None:
        self._pending[role].append(copy.deepcopy(transition))
        if len(self._pending[role]) >= self.chunk_size:
            self.flush(role)

    def flush(self, role: str | None = None) -> None:
        roles = tuple(self._pending) if role is None else (str(role),)
        for current_role in roles:
            if current_role not in self._pending:
                raise ValueError(f"Unknown transition role: {current_role}")
            transitions = self._pending[current_role]
            if not transitions:
                continue
            index = self._indices[current_role]
            target = (
                self.root
                / f"{current_role}_chunks"
                / f"chunk_{index:06d}.pt"
            )
            temporary = target.with_name(f".{target.name}.tmp")
            torch.save({"version": 1, "role": current_role, "transitions": transitions}, temporary)
            temporary.replace(target)
            self._pending[current_role] = []
            self._indices[current_role] += 1

    def close(self) -> None:
        self.flush()

    def __enter__(self) -> "TransitionChunkWriter":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


__all__ = [
    "Observation",
    "ReplayBufferConfig",
    "Transition",
    "TransitionChunkWriter",
    "assert_same_structure",
    "clone_array_tree",
    "to_numpy",
]
