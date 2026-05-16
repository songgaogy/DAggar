from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from robosuite.pipeline.base import OfflinePrefixOnlineDiscriminator, OnlineDiscriminatorDecision
from robosuite.pipeline.factory import build_online_discriminator


@dataclass
class AWRDiscriminatorLabel:
    global_step: int
    episode_index: int
    labeled_episode_step: int
    decision: OnlineDiscriminatorDecision


class AWRPolicyWorker:
    def __init__(self, agent) -> None:
        self.agent = agent
        self._condition = threading.Condition()
        self._thread: threading.Thread | None = None
        self._pending_commands: deque[dict[str, Any]] = deque()
        self._results: dict[int, np.ndarray] = {}
        self._stop_requested = False
        self._busy = False
        self._error: BaseException | None = None
        self._next_request_id = 1

    def start(self) -> None:
        with self._condition:
            if self._thread is not None:
                return
            self._stop_requested = False
            self._thread = threading.Thread(target=self._worker_loop, name="awr_policy_worker", daemon=True)
            self._thread.start()

    def select_action(self, obs, deterministic: bool = False, timeout: float | None = None) -> np.ndarray:
        self._raise_error()
        request_id = self._enqueue_command(
            {
                "type": "select_action",
                "request_id": self._next_request_id,
                "obs": self.agent.clone_observation(obs),
                "deterministic": bool(deterministic),
            }
        )
        with self._condition:
            end_time = None if timeout is None else time.monotonic() + float(timeout)
            while request_id not in self._results:
                self._raise_error()
                if end_time is not None:
                    remaining = end_time - time.monotonic()
                    if remaining <= 0.0:
                        raise TimeoutError("Timed out while waiting for AWR policy worker action.")
                    self._condition.wait(timeout=min(0.1, remaining))
                else:
                    self._condition.wait(timeout=0.1)
            action = self._results.pop(request_id)
        self._raise_error()
        return np.asarray(action, dtype=np.float32)

    def reset_policy_state(self) -> None:
        self._enqueue_command({"type": "reset_policy_state"})

    def notify_intervention(self) -> None:
        self._enqueue_command({"type": "notify_intervention"})

    def flush(self, timeout: float | None = None) -> None:
        self._raise_error()
        with self._condition:
            end_time = None if timeout is None else time.monotonic() + float(timeout)
            while self._pending_commands or self._busy:
                self._raise_error()
                if end_time is not None:
                    remaining = end_time - time.monotonic()
                    if remaining <= 0.0:
                        raise TimeoutError("Timed out while waiting for AWR policy worker.")
                    self._condition.wait(timeout=min(0.1, remaining))
                else:
                    self._condition.wait(timeout=0.1)
        self._raise_error()

    def close(self) -> None:
        self.flush()
        with self._condition:
            thread = self._thread
            if thread is None:
                return
            self._stop_requested = True
            self._condition.notify_all()
        thread.join(timeout=5.0)
        if thread.is_alive():
            raise RuntimeError("AWR policy worker did not stop cleanly.")
        with self._condition:
            self._thread = None
        self._raise_error()

    def _enqueue_command(self, command: dict[str, Any]) -> int | None:
        with self._condition:
            request_id = None
            if command["type"] == "select_action":
                request_id = int(command["request_id"])
                self._next_request_id = request_id + 1
            self._pending_commands.append(command)
            self._condition.notify_all()
            return request_id

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                while not self._pending_commands and not self._stop_requested:
                    self._condition.wait(timeout=0.1)
                if self._stop_requested and not self._pending_commands:
                    return
                command = self._pending_commands.popleft()
                self._busy = True
            try:
                command_type = str(command["type"])
                if command_type == "select_action":
                    action = self.agent.select_action(
                        command["obs"],
                        deterministic=bool(command["deterministic"]),
                    )
                    with self._condition:
                        self._results[int(command["request_id"])] = np.asarray(action, dtype=np.float32)
                        self._condition.notify_all()
                elif command_type == "reset_policy_state":
                    self.agent.reset_policy_state()
                elif command_type == "notify_intervention":
                    self.agent.notify_intervention()
                else:
                    raise KeyError(f"Unsupported AWRPolicyWorker command: {command_type}")
            except BaseException as exc:
                with self._condition:
                    self._error = exc
                    self._stop_requested = True
                    self._condition.notify_all()
                return
            finally:
                with self._condition:
                    self._busy = False
                    self._condition.notify_all()

    def _raise_error(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"AWR policy worker failed: {self._error}") from self._error


class AWRDiscriminatorWorker:
    def __init__(self, cfg: Any, *, task_name: str) -> None:
        self.cfg = cfg
        self.task_name = str(task_name)
        self.runtime = build_online_discriminator(cfg)
        if not isinstance(self.runtime, OfflinePrefixOnlineDiscriminator):
            raise TypeError(
                "AWRDiscriminatorWorker currently requires an OfflinePrefixOnlineDiscriminator "
                f"instance, got {type(self.runtime).__name__}."
            )
        self._condition = threading.Condition()
        self._thread: threading.Thread | None = None
        self._pending_commands: deque[dict[str, Any]] = deque()
        self._results: deque[AWRDiscriminatorLabel] = deque()
        self._stop_requested = False
        self._busy = False
        self._error: BaseException | None = None
        self._episode_index = -1
        self._last_available_steps = 0

    def start(self) -> None:
        with self._condition:
            if self._thread is not None:
                return
            self._stop_requested = False
            self._thread = threading.Thread(target=self._worker_loop, name="awr_discriminator_worker", daemon=True)
            self._thread.start()

    def reset(
        self,
        *,
        task_name: str,
        initial_state: np.ndarray,
        initial_images: dict[str, np.ndarray],
        episode_index: int,
    ) -> None:
        self._enqueue_command(
            {
                "type": "reset",
                "task_name": str(task_name),
                "initial_state": np.asarray(initial_state, dtype=np.float32).reshape(-1),
                "initial_images": {
                    key: np.asarray(value, dtype=np.uint8)
                    for key, value in initial_images.items()
                },
                "episode_index": int(episode_index),
            }
        )

    def record_step(
        self,
        *,
        action: np.ndarray,
        next_state: np.ndarray,
        next_images: dict[str, np.ndarray],
        episode_index: int,
        global_step: int,
    ) -> None:
        self._enqueue_command(
            {
                "type": "record_step",
                "action": np.asarray(action, dtype=np.float32).reshape(-1),
                "next_state": np.asarray(next_state, dtype=np.float32).reshape(-1),
                "next_images": {
                    key: np.asarray(value, dtype=np.uint8)
                    for key, value in next_images.items()
                },
                "episode_index": int(episode_index),
                "global_step": int(global_step),
            }
        )

    def drain_results(self) -> list[AWRDiscriminatorLabel]:
        self._raise_error()
        with self._condition:
            results = list(self._results)
            self._results.clear()
        return results

    def flush(self, timeout: float | None = None) -> None:
        self._raise_error()
        with self._condition:
            end_time = None if timeout is None else time.monotonic() + float(timeout)
            while self._pending_commands or self._busy:
                self._raise_error()
                if end_time is not None:
                    remaining = end_time - time.monotonic()
                    if remaining <= 0.0:
                        raise TimeoutError("Timed out while waiting for AWR discriminator worker.")
                    self._condition.wait(timeout=min(0.1, remaining))
                else:
                    self._condition.wait(timeout=0.1)
        self._raise_error()

    def close(self) -> None:
        self.flush()
        with self._condition:
            thread = self._thread
            if thread is None:
                return
            self._stop_requested = True
            self._condition.notify_all()
        thread.join(timeout=10.0)
        if thread.is_alive():
            raise RuntimeError("AWR discriminator worker did not stop cleanly.")
        with self._condition:
            self._thread = None
        try:
            self.runtime.close()
        finally:
            self._raise_error()

    def _enqueue_command(self, command: dict[str, Any]) -> None:
        self._raise_error()
        with self._condition:
            self._pending_commands.append(command)
            self._condition.notify_all()

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                while not self._pending_commands and not self._stop_requested:
                    self._condition.wait(timeout=0.1)
                if self._stop_requested and not self._pending_commands:
                    return
                command = self._pending_commands.popleft()
                batched_record_steps: list[dict[str, Any]] | None = None
                if str(command["type"]) == "record_step":
                    batched_record_steps = [command]
                    while self._pending_commands:
                        next_command = self._pending_commands[0]
                        if (
                            str(next_command["type"]) != "record_step"
                            or int(next_command["episode_index"]) != int(command["episode_index"])
                        ):
                            break
                        batched_record_steps.append(self._pending_commands.popleft())
                self._busy = True
            try:
                command_type = str(command["type"])
                if command_type == "reset":
                    self.runtime.reset(
                        task_name=str(command["task_name"]),
                        initial_state=command["initial_state"],
                        initial_images=command["initial_images"],
                    )
                    self._episode_index = int(command["episode_index"])
                    self._last_available_steps = 0
                elif command_type == "record_step":
                    self._handle_record_steps(batched_record_steps or [command])
                else:
                    raise KeyError(f"Unsupported AWRDiscriminatorWorker command: {command_type}")
            except BaseException as exc:
                with self._condition:
                    self._error = exc
                    self._stop_requested = True
                    self._condition.notify_all()
                return
            finally:
                with self._condition:
                    self._busy = False
                    self._condition.notify_all()

    def _handle_record_steps(self, commands: list[dict[str, Any]]) -> None:
        if not commands:
            return
        final_command = commands[-1]
        if int(final_command["episode_index"]) != self._episode_index:
            return

        for command in commands:
            self.runtime.action_history.append(command["action"])
            self.runtime.state_history.append(command["next_state"])
            for camera_name in self.runtime.camera_names:
                self.runtime.image_history[camera_name].append(command["next_images"][camera_name])

        trajectory = self.runtime._build_prefix_trajectory()
        if trajectory is None:
            return
        result = self.runtime._detect_prefix(trajectory)
        available_steps = int(result.aggregate_scores.shape[0])
        if available_steps <= self._last_available_steps:
            return

        new_labels: list[AWRDiscriminatorLabel] = []
        for labeled_index in range(self._last_available_steps, available_steps):
            decision = OnlineDiscriminatorDecision(
                evaluated=True,
                score=float(result.aggregate_scores[labeled_index]),
                threshold=float(result.thresholds[labeled_index]),
                prediction=int(result.predictions[labeled_index]),
                available_steps=available_steps,
                raw_step_score=float(result.step_scores[labeled_index]),
                metadata={
                    "detector_name": str(result.detector_name),
                    "task_name": str(self.runtime.task_name),
                    "threshold_final": float(result.metadata.get("threshold_final", result.thresholds[labeled_index])),
                    "delta_final": float(result.metadata.get("delta_final", np.nan)),
                },
            )
            new_labels.append(
                AWRDiscriminatorLabel(
                    global_step=int(final_command["global_step"]),
                    episode_index=int(final_command["episode_index"]),
                    labeled_episode_step=int(labeled_index),
                    decision=decision,
                )
            )
        self.runtime.last_decision = new_labels[-1].decision
        self.runtime.last_eval_action_count = len(self.runtime.action_history)
        self._last_available_steps = available_steps
        with self._condition:
            self._results.extend(new_labels)

    def _raise_error(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"AWR discriminator worker failed: {self._error}") from self._error
