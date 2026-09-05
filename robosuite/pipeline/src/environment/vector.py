"""Spawn-based vector runtime for batched DSRL macro actions."""

from __future__ import annotations

import multiprocessing as mp
import os
import traceback
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class MacroStepResult:
    worker_id: int
    observation: dict[str, np.ndarray]
    reward: float
    done: bool
    success: bool
    executed_length: int
    primitive_steps: int
    reason: str


def _success(env, info: dict[str, Any]) -> bool:
    if bool(info.get("success", info.get("is_success", False))):
        return True
    check = getattr(env, "_check_success", None)
    return bool(check()) if callable(check) else False


def _worker_main(
    worker_id: int,
    connection: Connection,
    env_metadata: dict[str, Any],
    camera_names: tuple[str, ...],
    image_height: int,
    image_width: int,
    control_frequency: int,
    horizon: int,
    seed: int,
) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    env = None
    extractor = None
    try:
        np.random.seed(seed + worker_id)
        from robosuite.pipeline.src.environment.observations import (
            bind_proprio_extractor,
            build_policy_observation,
            reset_policy_observation,
        )
        from robosuite.pipeline.src.environment.robosuite import (
            build_robosuite_env,
            build_runtime_config,
        )

        runtime_config = build_runtime_config(
            env_metadata,
            camera_names=camera_names,
            image_height=image_height,
            image_width=image_width,
            control_freq=control_frequency,
            horizon=horizon,
            interactive=False,
        )
        env = build_robosuite_env(runtime_config)
        extractor = bind_proprio_extractor(env, env_metadata)
        episode_steps = 0

        def reset() -> dict[str, np.ndarray]:
            nonlocal episode_steps
            observation, _ = reset_policy_observation(
                env,
                preserve_mjviewer=False,
                extractor=extractor,
                camera_names=camera_names,
                camera_aliases={},
                image_height=image_height,
                image_width=image_width,
            )
            episode_steps = 0
            return observation

        connection.send(("ready", None))
        while True:
            command, payload = connection.recv()
            if command == "close":
                break
            if command == "reset":
                connection.send(("ok", reset()))
                continue
            if command != "step_chunk":
                raise ValueError(f"Unknown vector-worker command: {command}")

            action_chunk = np.asarray(payload, dtype=np.float32)
            macro_reward = 0.0
            executed = 0
            success = False
            done = False
            reason = "chunk"
            info: dict[str, Any] = {}
            for action in action_chunk:
                result = env.step(action)
                if len(result) == 5:
                    raw_observation, _, terminated, truncated, raw_info = result
                else:
                    raw_observation, _, terminated, raw_info = result
                    truncated = False
                info = dict(raw_info) if isinstance(raw_info, dict) else {"raw_info": raw_info}
                episode_steps += 1
                executed += 1
                success = _success(env, info)
                macro_reward += 0.0 if success else -1.0
                horizon_done = episode_steps >= horizon
                done = bool(success or terminated or truncated or horizon_done)
                if done:
                    if success:
                        reason = "success"
                    elif horizon_done or truncated:
                        reason = "horizon"
                    else:
                        reason = "environment"
                    break

            observation = build_policy_observation(
                env,
                extractor=extractor,
                camera_names=camera_names,
                camera_aliases={},
                image_height=image_height,
                image_width=image_width,
                raw_observation=raw_observation,
            )
            connection.send(
                (
                    "ok",
                    MacroStepResult(
                        worker_id=worker_id,
                        observation=observation,
                        reward=float(macro_reward),
                        done=done,
                        success=success,
                        executed_length=executed,
                        primitive_steps=episode_steps,
                        reason=reason,
                    ),
                )
            )
    except BaseException as error:
        try:
            connection.send(("error", (repr(error), traceback.format_exc())))
        except BaseException:
            pass
    finally:
        if extractor is not None:
            extractor.close()
        if env is not None:
            env.close()
        connection.close()


class RobosuiteVectorRuntime:
    """Run one robosuite environment per spawned process."""

    def __init__(
        self,
        *,
        env_metadata: dict[str, Any],
        camera_names: Sequence[str],
        image_height: int,
        image_width: int,
        control_frequency: int,
        horizon: int,
        num_envs: int,
        seed: int,
    ) -> None:
        if int(num_envs) <= 0:
            raise ValueError("num_envs must be positive.")
        context = mp.get_context("spawn")
        self._parents: list[Connection] = []
        self._processes: list[mp.Process] = []
        for worker_id in range(int(num_envs)):
            parent, child = context.Pipe()
            process = context.Process(
                target=_worker_main,
                args=(
                    worker_id,
                    child,
                    dict(env_metadata),
                    tuple(camera_names),
                    int(image_height),
                    int(image_width),
                    int(control_frequency),
                    int(horizon),
                    int(seed),
                ),
                name=f"dsrl-env-{worker_id}",
            )
            process.start()
            child.close()
            self._parents.append(parent)
            self._processes.append(process)
        for worker_id, parent in enumerate(self._parents):
            status, payload = parent.recv()
            self._check(status, payload, worker_id)

    @property
    def num_envs(self) -> int:
        return len(self._parents)

    @staticmethod
    def _check(status: str, payload: Any, worker_id: int) -> Any:
        if status == "ok" or status == "ready":
            return payload
        if status == "error":
            message, trace = payload
            raise RuntimeError(f"Environment worker {worker_id} failed: {message}\n{trace}")
        raise RuntimeError(f"Environment worker {worker_id} returned invalid status {status!r}.")

    def reset(self, worker_ids: Sequence[int] | None = None) -> dict[int, dict[str, np.ndarray]]:
        selected = list(range(self.num_envs)) if worker_ids is None else [int(i) for i in worker_ids]
        for worker_id in selected:
            self._parents[worker_id].send(("reset", None))
        return {
            worker_id: self._check(*self._parents[worker_id].recv(), worker_id)
            for worker_id in selected
        }

    def step(
        self, worker_ids: Sequence[int], action_chunks: np.ndarray
    ) -> dict[int, MacroStepResult]:
        selected = [int(i) for i in worker_ids]
        chunks = np.asarray(action_chunks, dtype=np.float32)
        if chunks.shape[0] != len(selected):
            raise ValueError("The action batch must match worker_ids.")
        for batch_index, worker_id in enumerate(selected):
            self._parents[worker_id].send(("step_chunk", chunks[batch_index]))
        return {
            worker_id: self._check(*self._parents[worker_id].recv(), worker_id)
            for worker_id in selected
        }

    def close(self) -> None:
        for parent in self._parents:
            try:
                parent.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process in self._processes:
            process.join(timeout=10.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
        for parent in self._parents:
            parent.close()
        self._parents.clear()
        self._processes.clear()

    def __enter__(self) -> "RobosuiteVectorRuntime":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


__all__ = ["MacroStepResult", "RobosuiteVectorRuntime"]
