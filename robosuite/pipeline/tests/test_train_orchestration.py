from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robosuite.pipeline.train import (
    PolicyDecision,
    _canonical_observation,
    run_online_collection,
)


@dataclass(frozen=True)
class Result:
    worker_id: int
    observation: dict[str, np.ndarray]
    reward: float
    done: bool
    success: bool
    executed_length: int
    primitive_steps: int
    reason: str


class MockRuntime:
    def __init__(self, num_envs: int, macro_steps_per_episode: int = 2) -> None:
        self.num_envs = num_envs
        self.macro_steps_per_episode = macro_steps_per_episode
        self.episode_macro = [0] * num_envs
        self.completed = [0] * num_envs
        self.reset_calls: list[tuple[int, ...]] = []
        self.step_calls: list[tuple[int, ...]] = []

    @staticmethod
    def observation(worker: int, marker: int) -> dict[str, np.ndarray]:
        return {"marker": np.asarray([worker, marker], dtype=np.int64)}

    def reset(self, worker_ids=None):
        selected = list(range(self.num_envs)) if worker_ids is None else list(worker_ids)
        self.reset_calls.append(tuple(selected))
        for worker in selected:
            self.episode_macro[worker] = 0
        return {worker: self.observation(worker, self.completed[worker]) for worker in selected}

    def step(self, worker_ids, action_chunks):
        selected = list(worker_ids)
        self.step_calls.append(tuple(selected))
        assert action_chunks.shape == (len(selected), 8, 7)
        results = {}
        for worker in selected:
            self.episode_macro[worker] += 1
            done = self.episode_macro[worker] == self.macro_steps_per_episode
            if done:
                self.completed[worker] += 1
            results[worker] = Result(
                worker_id=worker,
                observation=self.observation(worker, 100 + self.episode_macro[worker]),
                reward=-3.0,
                done=done,
                success=done,
                executed_length=3 if done else 8,
                primitive_steps=11 if done else 8,
                reason="success" if done else "chunk",
            )
        return results


def test_legacy_bare_camera_keys_are_canonicalized_without_copying_arrays() -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    observation = {"agentview": image, "state": np.zeros(14, dtype=np.float32)}
    canonical = _canonical_observation(observation, ["agentview"])
    assert canonical["agentview_image"] is image
    assert canonical["state"] is observation["state"]


def test_collection_enforces_per_worker_quota_and_updates_before_episode_callback() -> None:
    runtime = MockRuntime(num_envs=4)
    transitions = []
    ordering = []
    learner_steps = 0

    def encode(observations):
        return {worker: tuple(value["marker"].tolist()) for worker, value in observations.items()}

    def act(worker_ids, states):
        assert all(worker in states for worker in worker_ids)
        shape = (len(worker_ids), 8, 7)
        return PolicyDecision(np.zeros(shape, dtype=np.float32), np.ones(shape, dtype=np.float32))

    def add_transition(worker, state, next_state, actions, result):
        transitions.append((worker, state, next_state, actions.shape, result.executed_length))

    def update():
        nonlocal learner_steps
        learner_steps += 1
        ordering.append(("update", learner_steps))
        return {"qa_loss": 1.0}

    def episode(metrics):
        ordering.append(("episode", metrics["episode"]))
        assert ordering[-2][0] in {"update", "episode"}

    summary = run_online_collection(
        runtime=runtime,
        num_envs=4,
        episodes_per_env=5,
        encode=encode,
        act=act,
        add_transition=add_transition,
        update=update,
        sync_inference=lambda: None,
        on_episode=episode,
        on_update=lambda metrics, step: None,
    )

    assert summary.completed_episodes == 20
    assert summary.completed_by_worker == (5, 5, 5, 5)
    assert runtime.completed == [5, 5, 5, 5]
    assert summary.macro_steps == 40
    assert summary.primitive_steps == 4 * 5 * (8 + 3)
    assert learner_steps == 10
    assert len(transitions) == 40
    assert runtime.reset_calls == [(0, 1, 2, 3)] + [(0, 1, 2, 3)] * 4
    assert runtime.step_calls == [(0, 1, 2, 3)] * 10
    assert ordering[-1] == ("episode", 20)


def test_finished_worker_is_never_reset_or_stepped_again() -> None:
    class UnevenRuntime(MockRuntime):
        def step(self, worker_ids, action_chunks):
            results = super().step(worker_ids, action_chunks)
            for worker in worker_ids:
                target = 1 if worker == 0 else 2
                result = results[worker]
                done = self.episode_macro[worker] >= target
                if worker == 0 and done and not result.done:
                    self.completed[worker] += 1
                results[worker] = Result(
                    **{**result.__dict__, "done": done, "success": done, "reason": "success" if done else "chunk"}
                )
            return results

    runtime = UnevenRuntime(num_envs=2)
    summary = run_online_collection(
        runtime=runtime,
        num_envs=2,
        episodes_per_env=1,
        encode=lambda observations: dict(observations),
        act=lambda workers, states: PolicyDecision(
            np.zeros((len(workers), 8, 7), dtype=np.float32),
            np.zeros((len(workers), 8, 7), dtype=np.float32),
        ),
        add_transition=lambda *args: None,
        update=lambda: {},
        sync_inference=lambda: None,
        on_episode=lambda metrics: None,
        on_update=lambda metrics, step: None,
    )
    assert summary.completed_by_worker == (1, 1)
    assert runtime.step_calls == [(0, 1), (1,)]
    assert runtime.reset_calls == [(0, 1)]


def test_collection_resume_only_runs_remaining_worker_quota() -> None:
    runtime = MockRuntime(num_envs=2, macro_steps_per_episode=1)
    summary = run_online_collection(
        runtime=runtime,
        num_envs=2,
        episodes_per_env=2,
        encode=lambda observations: dict(observations),
        act=lambda workers, states: PolicyDecision(
            np.zeros((len(workers), 8, 7), dtype=np.float32),
            np.zeros((len(workers), 8, 7), dtype=np.float32),
        ),
        add_transition=lambda *args: None,
        update=lambda: {},
        sync_inference=lambda: None,
        on_episode=lambda metrics: None,
        on_update=lambda metrics, step: None,
        initial_completed_by_worker=(2, 1),
        initial_primitive_steps=30,
        initial_macro_steps=4,
    )
    assert summary.completed_by_worker == (2, 2)
    assert summary.completed_episodes == 4
    assert summary.primitive_steps == 33
    assert summary.macro_steps == 5
    assert runtime.reset_calls == [(1,)]
    assert runtime.step_calls == [(1,)]


def test_collection_checkpoints_after_interval_update() -> None:
    runtime = MockRuntime(num_envs=2, macro_steps_per_episode=1)
    checkpoints = []
    summary = run_online_collection(
        runtime=runtime,
        num_envs=2,
        episodes_per_env=2,
        encode=lambda observations: dict(observations),
        act=lambda workers, states: PolicyDecision(
            np.zeros((len(workers), 8, 7), dtype=np.float32),
            np.zeros((len(workers), 8, 7), dtype=np.float32),
        ),
        add_transition=lambda *args: None,
        update=lambda: {},
        sync_inference=lambda: None,
        on_episode=lambda metrics: None,
        on_update=lambda metrics, step: None,
        on_checkpoint=checkpoints.append,
        checkpoint_interval_episodes=2,
    )
    assert [item.completed_episodes for item in checkpoints] == [2, 4]
    assert checkpoints[-1] == summary


def test_collection_does_not_skip_crossed_checkpoint_threshold() -> None:
    runtime = MockRuntime(num_envs=3, macro_steps_per_episode=1)
    checkpoints = []
    run_online_collection(
        runtime=runtime,
        num_envs=3,
        episodes_per_env=1,
        encode=lambda observations: dict(observations),
        act=lambda workers, states: PolicyDecision(
            np.zeros((len(workers), 8, 7), dtype=np.float32),
            np.zeros((len(workers), 8, 7), dtype=np.float32),
        ),
        add_transition=lambda *args: None,
        update=lambda: {},
        sync_inference=lambda: None,
        on_episode=lambda metrics: None,
        on_update=lambda metrics, step: None,
        on_checkpoint=checkpoints.append,
        checkpoint_interval_episodes=2,
    )
    assert [item.completed_episodes for item in checkpoints] == [3]
