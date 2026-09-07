from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robosuite.pipeline.orchestration import (
    PolicyDecision,
    distribute_episode_quota,
    run_episode_collection,
)
from robosuite.pipeline.train import _canonical_observation, _resume_signature


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
        self.action_chunks: list[np.ndarray] = []

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
        self.action_chunks.append(np.asarray(action_chunks).copy())
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
                reward=float(done),
                done=done,
                success=done,
                executed_length=3 if done else 8,
                primitive_steps=11 if done else 8,
                reason="success" if done else "chunk",
            )
        return results


def _decision(workers, _states):
    shape = (len(workers), 8, 7)
    return PolicyDecision(np.zeros(shape, dtype=np.float32), np.ones(shape, dtype=np.float32))


def test_legacy_bare_camera_keys_are_canonicalized_without_copying_arrays() -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    observation = {"agentview": image, "state": np.zeros(14, dtype=np.float32)}
    canonical = _canonical_observation(observation, ["agentview"])
    assert canonical["agentview_image"] is image
    assert canonical["state"] is observation["state"]


def test_episode_quota_distribution_is_global() -> None:
    assert distribute_episode_quota(20, 4) == (5, 5, 5, 5)
    assert distribute_episode_quota(7, 3) == (3, 2, 2)


def test_collection_updates_every_second_vector_step() -> None:
    runtime = MockRuntime(num_envs=4)
    transitions = []
    update_steps = []

    summary = run_episode_collection(
        runtime=runtime,
        target_by_worker=(2, 2, 2, 2),
        encode=lambda observations: dict(observations),
        act=_decision,
        add_transition=lambda *args: transitions.append(args),
        update=lambda: {"critic_loss": 1.0},
        sync_inference=lambda: None,
        on_episode=lambda metrics: None,
        on_update=lambda metrics, step: update_steps.append(step),
        update_interval_vector_steps=2,
    )

    assert summary.completed_episodes == 8
    assert summary.completed_by_worker == (2, 2, 2, 2)
    assert summary.online_vector_steps == 4
    assert summary.macro_steps == len(transitions) == 16
    assert update_steps == [2, 4]


def test_finished_worker_is_never_reset_or_stepped_again() -> None:
    runtime = MockRuntime(num_envs=2, macro_steps_per_episode=1)
    summary = run_episode_collection(
        runtime=runtime,
        target_by_worker=(0, 1),
        initial_completed_by_worker=(0, 0),
        encode=lambda observations: dict(observations),
        act=_decision,
        add_transition=lambda *args: None,
        on_episode=lambda metrics: None,
    )
    assert summary.completed_by_worker == (0, 1)
    assert runtime.reset_calls == [(1,)]
    assert runtime.step_calls == [(1,)]


def test_resume_preserves_update_cadence() -> None:
    runtime = MockRuntime(num_envs=2, macro_steps_per_episode=1)
    update_steps = []
    summary = run_episode_collection(
        runtime=runtime,
        target_by_worker=(2, 2),
        initial_completed_by_worker=(2, 1),
        initial_primitive_steps=30,
        initial_macro_steps=4,
        initial_online_vector_steps=3,
        encode=lambda observations: dict(observations),
        act=_decision,
        add_transition=lambda *args: None,
        update=lambda: {},
        on_update=lambda metrics, step: update_steps.append(step),
        update_interval_vector_steps=2,
        on_episode=lambda metrics: None,
    )
    assert summary.completed_by_worker == (2, 2)
    assert summary.primitive_steps == 33
    assert summary.macro_steps == 5
    assert summary.online_vector_steps == 4
    assert update_steps == [4]


def test_parallel_collection_hits_checkpoint_threshold_exactly() -> None:
    runtime = MockRuntime(num_envs=3, macro_steps_per_episode=1)
    checkpoints = []
    summary = run_episode_collection(
        runtime=runtime,
        target_by_worker=(1, 1, 1),
        encode=lambda observations: dict(observations),
        act=_decision,
        add_transition=lambda *args: None,
        on_episode=lambda metrics: None,
        on_checkpoint=checkpoints.append,
        checkpoint_interval_episodes=2,
    )
    assert [item.completed_episodes for item in checkpoints] == [2]
    assert runtime.reset_calls == [(0, 1), (2,)]
    assert runtime.step_calls == [(0, 1), (2,)]
    assert summary.completed_episodes == 3


def test_four_worker_warmup_collects_exactly_twenty_latent_episodes() -> None:
    runtime = MockRuntime(num_envs=4, macro_steps_per_episode=1)
    replay_actions = []

    def add_transition(_worker, _state, _next_state, replay_action, _result):
        replay_actions.append(replay_action.copy())

    summary = run_episode_collection(
        runtime=runtime,
        target_by_worker=distribute_episode_quota(20, 4),
        encode=lambda observations: dict(observations),
        act=_decision,
        add_transition=add_transition,
        on_episode=lambda metrics: None,
    )

    assert summary.completed_episodes == 20
    assert summary.completed_by_worker == (5, 5, 5, 5)
    assert len(replay_actions) == 20
    assert all(np.all(action == 1.0) for action in replay_actions)
    assert all(np.all(action_chunk == 0.0) for action_chunk in runtime.action_chunks)


def test_resume_signature_allows_only_resume_location_changes() -> None:
    stored = {
        "seed": 42,
        "task": {"name": "PickPlaceCereal"},
        "environment": {"horizon": 500},
        "inputs": {"base_policy_checkpoint": "flow.pt"},
        "vision": {"cache_dtype": "float16"},
        "flow": {"action_horizon": 8},
        "algorithm": {"utd": 30},
        "warmup": {"episodes": 20},
        "runtime": {"num_envs": 4, "resume": False, "checkpoint": None},
        "storage": {"checkpoint_interval_episodes": 20},
    }
    resumed = {
        **stored,
        "runtime": {
            **stored["runtime"],
            "resume": True,
            "checkpoint": "episode_00000040.pt",
        },
    }
    assert _resume_signature(stored) == _resume_signature(resumed)
    changed = {**resumed, "seed": 7}
    assert _resume_signature(stored) != _resume_signature(changed)
