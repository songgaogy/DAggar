from __future__ import annotations

import copy
import queue
import threading
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
    """Persist online and intervention transitions in bounded chunks."""

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
        self._commands: queue.Queue[tuple[str, Any]] = queue.Queue(
            maxsize=self.chunk_size
        )
        self._worker_error: BaseException | None = None
        self._closed = False
        self._worker = threading.Thread(
            target=self._run_worker,
            name="transition-chunk-writer",
            daemon=True,
        )
        self._worker.start()

    def append(self, transition: Transition) -> None:
        if self._closed:
            raise RuntimeError("Cannot append to a closed TransitionChunkWriter.")
        self._enqueue(("append", copy.deepcopy(transition)))

    def _append_role(self, role: str, transition: Transition) -> None:
        self._pending[role].append(transition)
        if len(self._pending[role]) >= self.chunk_size:
            self._flush_role(role)

    def flush(self, role: str | None = None) -> None:
        roles = tuple(self._pending) if role is None else (str(role),)
        for current_role in roles:
            if current_role not in self._pending:
                raise ValueError(f"Unknown transition role: {current_role}")
        if self._closed:
            self._raise_worker_error()
            return
        completed = threading.Event()
        self._enqueue(("flush", (roles, completed)))
        self._wait_for(completed)

    def close(self) -> None:
        if self._closed:
            self._raise_worker_error()
            return
        self._closed = True
        completed = threading.Event()
        try:
            self._enqueue(("close", completed))
            self._wait_for(completed)
        finally:
            self._worker.join()
        self._raise_worker_error()

    def _enqueue(self, command: tuple[str, Any]) -> None:
        while True:
            self._raise_worker_error()
            try:
                self._commands.put(command, timeout=0.05)
                return
            except queue.Full:
                continue

    def _wait_for(self, completed: threading.Event) -> None:
        while not completed.wait(timeout=0.05):
            self._raise_worker_error()
        self._raise_worker_error()

    def _raise_worker_error(self) -> None:
        if self._worker_error is not None:
            raise RuntimeError("Transition chunk writer failed.") from self._worker_error

    def _run_worker(self) -> None:
        while True:
            command, payload = self._commands.get()
            completed: threading.Event | None = None
            try:
                if command == "append":
                    transition = payload
                    self._append_role("online", transition)
                    if transition.is_intervention:
                        self._append_role("demo", copy.deepcopy(transition))
                elif command == "flush":
                    roles, completed = payload
                    for role in roles:
                        self._flush_role(role)
                elif command == "close":
                    completed = payload
                    for role in self._pending:
                        self._flush_role(role)
                    completed.set()
                    return
            except BaseException as error:
                self._worker_error = error
                if completed is not None:
                    completed.set()
                self._release_waiters()
                return
            finally:
                self._commands.task_done()
            if completed is not None:
                completed.set()

    def _flush_role(self, role: str) -> None:
        transitions = self._pending[role]
        if not transitions:
            return
        index = self._indices[role]
        target = self.root / f"{role}_chunks" / f"chunk_{index:06d}.pt"
        temporary = target.with_name(f".{target.name}.tmp")
        torch.save(
            {"version": 1, "role": role, "transitions": transitions},
            temporary,
        )
        temporary.replace(target)
        self._pending[role] = []
        self._indices[role] += 1

    def _release_waiters(self) -> None:
        while True:
            try:
                command, payload = self._commands.get_nowait()
            except queue.Empty:
                return
            if command == "flush":
                payload[1].set()
            elif command == "close":
                payload.set()
            self._commands.task_done()

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
