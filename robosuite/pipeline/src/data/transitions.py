from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import torch

from ...utils.tensor import clone_array_tree


Observation = np.ndarray | Mapping[str, Any]


@dataclass
class Transition:
    obs: Observation
    action: Any
    reward: Optional[float]
    next_obs: Observation
    done: bool
    grasp_penalty: Optional[float] = None
    is_intervention: bool = False
    info: Optional[dict[str, Any]] = None
    reward_source: Optional[str] = None
    demo_source: Optional[str] = None
    terminated: Optional[bool] = None
    truncated: bool = False


def serialize_transition(transition: Transition) -> dict[str, Any]:
    return {
        "obs": clone_array_tree(transition.obs),
        "action": clone_array_tree(transition.action),
        "reward": None if transition.reward is None else float(transition.reward),
        "next_obs": clone_array_tree(transition.next_obs),
        "done": bool(transition.done),
        "grasp_penalty": None if transition.grasp_penalty is None else float(transition.grasp_penalty),
        "is_intervention": bool(transition.is_intervention),
        "info": dict(transition.info) if transition.info is not None else None,
        "reward_source": transition.reward_source,
        "demo_source": transition.demo_source,
        "terminated": transition.terminated,
        "truncated": bool(transition.truncated),
    }


def deserialize_transition(payload: dict[str, Any]) -> Transition:
    return Transition(
        obs=clone_array_tree(payload["obs"]),
        action=clone_array_tree(payload["action"]),
        reward=payload["reward"],
        next_obs=clone_array_tree(payload["next_obs"]),
        done=bool(payload["done"]),
        grasp_penalty=payload.get("grasp_penalty"),
        is_intervention=bool(payload.get("is_intervention", False)),
        info=dict(payload["info"]) if payload.get("info") is not None else None,
        reward_source=payload.get("reward_source"),
        demo_source=payload.get("demo_source"),
        terminated=payload.get("terminated"),
        truncated=bool(payload.get("truncated", False)),
    )


def save_transition_shard(path: str | Path, transitions: Sequence[Transition]) -> None:
    path = Path(path)
    payload = [serialize_transition(transition) for transition in transitions]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def load_transition_shard(path: str | Path) -> list[Transition]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    return [deserialize_transition(item) for item in payload]


class AsyncTransitionChunkWriter:
    def __init__(
        self,
        output_dir: Path,
        *,
        chunk_size: int,
        event_logger: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.chunk_size = max(1, int(chunk_size))
        self.event_logger = event_logger
        self._condition = threading.Condition()
        self._online: deque[Transition] = deque()
        self._demo: deque[Transition] = deque()
        self._thread: threading.Thread | None = None
        self._stop = False
        self._flush = False
        self._busy = False
        self._error: BaseException | None = None
        self._indices = {"online": 0, "demo": 0}

    def start(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with self._condition:
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run,
                    name="hil_serl_buffer_writer",
                    daemon=True,
                )
                self._thread.start()

    def request_transition(
        self,
        *,
        online_transition: Transition,
        demo_transition: Transition | None = None,
    ) -> None:
        self._raise_if_failed()
        with self._condition:
            self._online.append(online_transition)
            if demo_transition is not None:
                self._demo.append(demo_transition)
            self._condition.notify_all()

    def flush(self, timeout: float | None = None) -> None:
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        with self._condition:
            self._flush = True
            self._condition.notify_all()
            while self._online or self._demo or self._busy or self._flush:
                self._raise_if_failed()
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("Timed out while flushing buffer chunks.")
                self._condition.wait(timeout=0.1)

    def close(self) -> None:
        with self._condition:
            thread = self._thread
            if thread is None:
                return
            self._stop = True
            self._flush = True
            self._condition.notify_all()
        thread.join(timeout=30)
        if thread.is_alive():
            raise RuntimeError("Buffer writer did not stop.")
        self._raise_if_failed()

    def _run(self) -> None:
        pending: dict[str, list[Transition]] = {"online": [], "demo": []}
        try:
            while True:
                with self._condition:
                    while not self._online and not self._demo and not self._flush and not self._stop:
                        self._condition.wait(timeout=0.1)
                    pending["online"].extend(self._online)
                    pending["demo"].extend(self._demo)
                    self._online.clear()
                    self._demo.clear()
                    force = self._flush or self._stop
                    self._busy = True
                for stream in ("online", "demo"):
                    while len(pending[stream]) >= self.chunk_size or (force and pending[stream]):
                        chunk = pending[stream][: self.chunk_size]
                        del pending[stream][: len(chunk)]
                        self._write(stream, chunk)
                with self._condition:
                    self._busy = False
                    if force and not pending["online"] and not pending["demo"]:
                        self._flush = False
                    self._condition.notify_all()
                    if self._stop and not pending["online"] and not pending["demo"]:
                        return
        except BaseException as exc:
            with self._condition:
                self._error = exc
                self._busy = False
                self._condition.notify_all()

    def _write(self, stream: str, transitions: list[Transition]) -> None:
        directory = self.output_dir / f"{stream}_chunks"
        directory.mkdir(parents=True, exist_ok=True)
        index = self._indices[stream]
        path = directory / f"chunk_{index:08d}.pt"
        tmp_path = path.with_name(f".{path.name}.tmp")
        torch.save({"chunk_index": index, "transitions": transitions}, tmp_path)
        tmp_path.replace(path)
        self._indices[stream] += 1
        if self.event_logger is not None:
            self.event_logger(
                {
                    "event": "buffer_chunk_written",
                    "stream": stream,
                    "chunk_index": index,
                    "transition_count": len(transitions),
                }
            )

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"Buffer writer failed: {self._error}") from self._error
