from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PolicyDecision:
    environment_actions: np.ndarray
    replay_actions: np.ndarray


@dataclass(frozen=True)
class CollectionSummary:
    completed_episodes: int
    completed_by_worker: tuple[int, ...]
    primitive_steps: int
    macro_steps: int
    online_vector_steps: int


def distribute_episode_quota(total_episodes: int, num_envs: int) -> tuple[int, ...]:
    if total_episodes < 0 or num_envs <= 0:
        raise ValueError("total_episodes must be non-negative and num_envs must be positive.")
    base, remainder = divmod(int(total_episodes), int(num_envs))
    return tuple(base + int(worker < remainder) for worker in range(num_envs))


def run_episode_collection(
    *,
    runtime: Any,
    target_by_worker: Sequence[int],
    encode: Callable[[Mapping[int, Mapping[str, np.ndarray]]], dict[int, Any]],
    act: Callable[[Sequence[int], Mapping[int, Any]], PolicyDecision],
    add_transition: Callable[[int, Any, Any, np.ndarray, Any], None],
    on_episode: Callable[[dict[str, Any]], None],
    update: Callable[[], Mapping[str, float]] | None = None,
    sync_inference: Callable[[], None] | None = None,
    on_update: Callable[[Mapping[str, float], int], None] | None = None,
    update_interval_vector_steps: int = 1,
    on_checkpoint: Callable[[CollectionSummary], None] | None = None,
    checkpoint_interval_episodes: int | None = None,
    initial_completed_by_worker: Sequence[int] | None = None,
    initial_primitive_steps: int = 0,
    initial_macro_steps: int = 0,
    initial_online_vector_steps: int = 0,
) -> CollectionSummary:
    """Collect per-worker quotas with exact global episode checkpoints."""

    targets = [int(value) for value in target_by_worker]
    if not targets or any(value < 0 for value in targets):
        raise ValueError("Per-worker episode targets must be non-negative.")
    num_envs = len(targets)
    completed = (
        [0] * num_envs
        if initial_completed_by_worker is None
        else [int(value) for value in initial_completed_by_worker]
    )
    if len(completed) != num_envs or any(
        not 0 <= value <= targets[index] for index, value in enumerate(completed)
    ):
        raise ValueError("Initial per-worker episode counters are invalid.")
    if update is not None and update_interval_vector_steps <= 0:
        raise ValueError("update_interval_vector_steps must be positive.")

    returns = [0.0] * num_envs
    macro_lengths = [0] * num_envs
    episode_started = [time.monotonic()] * num_envs
    primitive_total = int(initial_primitive_steps)
    macro_total = int(initial_macro_steps)
    vector_steps = int(initial_online_vector_steps)
    completed_total = sum(completed)
    target_total = sum(targets)
    interval = None if checkpoint_interval_episodes is None else int(checkpoint_interval_episodes)
    next_checkpoint = (
        None
        if interval is None or interval <= 0
        else (completed_total // interval + 1) * interval
    )

    def stage_targets() -> list[int]:
        if next_checkpoint is None:
            return list(targets)
        stage_total = min(next_checkpoint, target_total)
        staged = list(completed)
        remaining = stage_total - completed_total
        while remaining > 0:
            progressed = False
            for worker_id in range(num_envs):
                if staged[worker_id] >= targets[worker_id]:
                    continue
                staged[worker_id] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
            if not progressed:
                raise RuntimeError("Cannot allocate the next episode checkpoint stage.")
        return staged

    def summary() -> CollectionSummary:
        return CollectionSummary(
            completed_episodes=completed_total,
            completed_by_worker=tuple(completed),
            primitive_steps=primitive_total,
            macro_steps=macro_total,
            online_vector_steps=vector_steps,
        )

    initial_stage = stage_targets()
    active_at_start = [
        index for index, count in enumerate(completed) if count < initial_stage[index]
    ]
    observations = runtime.reset(active_at_start) if active_at_start else {}
    states = encode(observations) if observations else {}

    while completed_total < target_total:
        current_stage = stage_targets()
        worker_ids = [
            index for index, count in enumerate(completed) if count < current_stage[index]
        ]
        missing_states = [worker for worker in worker_ids if worker not in states]
        if missing_states:
            states.update(encode(runtime.reset(missing_states)))

        decision = act(worker_ids, states)
        results = runtime.step(worker_ids, decision.environment_actions)
        next_observations = {worker: results[worker].observation for worker in worker_ids}
        next_states = encode(next_observations)
        for row, worker_id in enumerate(worker_ids):
            result = results[worker_id]
            add_transition(
                worker_id,
                states[worker_id],
                next_states[worker_id],
                decision.replay_actions[row],
                result,
            )
            returns[worker_id] += float(result.reward)
            macro_lengths[worker_id] += 1
            primitive_total += int(result.executed_length)
            macro_total += 1

        vector_steps += 1
        if update is not None and vector_steps % int(update_interval_vector_steps) == 0:
            metrics = update()
            if sync_inference is not None:
                sync_inference()
            if on_update is not None:
                on_update(metrics, vector_steps)

        reset_ids: list[int] = []
        for worker_id in worker_ids:
            result = results[worker_id]
            if not result.done:
                states[worker_id] = next_states[worker_id]
                continue
            completed[worker_id] += 1
            completed_total += 1
            elapsed = max(time.monotonic() - episode_started[worker_id], 1e-9)
            on_episode(
                {
                    "episode": completed_total,
                    "worker": worker_id,
                    "worker_episode": completed[worker_id],
                    "success": bool(result.success),
                    "return": returns[worker_id],
                    "primitive_length": int(result.primitive_steps),
                    "macro_length": macro_lengths[worker_id],
                    "termination_reason": str(result.reason),
                    "fps": float(result.primitive_steps) / elapsed,
                }
            )
            returns[worker_id] = 0.0
            macro_lengths[worker_id] = 0
            if completed[worker_id] < current_stage[worker_id]:
                reset_ids.append(worker_id)
                episode_started[worker_id] = time.monotonic()
            else:
                states.pop(worker_id, None)
        if reset_ids:
            states.update(encode(runtime.reset(reset_ids)))
        if (
            on_checkpoint is not None
            and next_checkpoint is not None
            and completed_total == next_checkpoint
        ):
            on_checkpoint(summary())
            next_checkpoint += interval

    return summary()


__all__ = [
    "CollectionSummary",
    "PolicyDecision",
    "distribute_episode_quota",
    "run_episode_collection",
]
