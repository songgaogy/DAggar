from pathlib import Path
from types import SimpleNamespace

import numpy as np

from robosuite.pipeline.src.data import Transition, TransitionChunkWriter
from robosuite.pipeline.src.awr import AWRTrainer, TrainerConfig
from robosuite.pipeline.src.environment import sparse_success_reward
from robosuite.pipeline.train import (
    _checkpoint_due,
    _format_episode_summary,
    _save_checkpoint,
)


class _SuccessEnv:
    def __init__(self, success: bool) -> None:
        self.success = bool(success)

    def _check_success(self) -> bool:
        return self.success


def _transition(*, intervention: bool) -> Transition:
    return Transition(
        obs={"state": np.zeros(2, dtype=np.float32)},
        action=np.zeros(7, dtype=np.float32),
        reward=-1.0,
        next_obs={"state": np.ones(2, dtype=np.float32)},
        done=False,
        is_intervention=intervention,
    )


def test_sparse_reward_is_minus_one_or_zero() -> None:
    assert sparse_success_reward(_SuccessEnv(False)) == (-1.0, False)
    assert sparse_success_reward(_SuccessEnv(True)) == (0.0, True)
    assert sparse_success_reward(_SuccessEnv(False), {"success": True}) == (0.0, True)


def test_intervention_transition_is_persisted_to_both_roles(tmp_path: Path) -> None:
    with TransitionChunkWriter(tmp_path, chunk_size=1) as writer:
        writer.append(_transition(intervention=True))

    assert len(list((tmp_path / "online_chunks").glob("chunk_*.pt"))) == 1
    assert len(list((tmp_path / "demo_chunks").glob("chunk_*.pt"))) == 1


def test_policy_transition_is_not_persisted_as_demo(tmp_path: Path) -> None:
    with TransitionChunkWriter(tmp_path, chunk_size=1) as writer:
        writer.append(_transition(intervention=False))

    assert len(list((tmp_path / "online_chunks").glob("chunk_*.pt"))) == 1
    assert list((tmp_path / "demo_chunks").glob("chunk_*.pt")) == []


class _ReadyAgent:
    def __init__(self) -> None:
        self.trainer_config = TrainerConfig(
            updates_per_train=100,
            inference_sync_interval=50,
        )
        self.updates = 0
        self.syncs = 0

    def ready_for_update(self) -> bool:
        return True

    def update(self, _batch_size=None) -> dict[str, float]:
        self.updates += 1
        return {"loss": float(self.updates)}

    def sync_inference_policy(self) -> None:
        self.syncs += 1


def test_episode_boundary_runs_exactly_one_hundred_updates() -> None:
    agent = _ReadyAgent()
    trainer = AWRTrainer(agent)
    trainer.total_env_steps = 1

    metrics = trainer.train_episode()

    assert len(metrics) == 100
    assert trainer.total_actor_updates == 100
    assert trainer.total_value_updates == 100
    assert trainer.total_inference_syncs == 2
    assert agent.syncs == 2


def test_episode_progress_preserves_update_and_episode_counts() -> None:
    agent = _ReadyAgent()
    trainer = AWRTrainer(agent)
    trainer.total_env_steps = 1

    metrics = trainer.train_episode(updates=3, show_progress=True)

    assert len(metrics) == 3
    assert trainer.total_episodes == 1
    assert trainer.total_actor_updates == 3
    assert agent.updates == 3


def test_episode_summary_contains_runtime_and_intervention_details() -> None:
    summary = _format_episode_summary(
        4,
        "environment",
        {
            "success": 1.0,
            "return": 0.0,
            "length": 25.0,
            "env_fps": 19.5,
            "intervention_steps": 5.0,
            "intervention_rate": 0.2,
            "learner_updates": 100.0,
            "learner_seconds": 3.0,
            "reset_seconds": 0.4,
            "checkpoint_seconds": 0.0,
            "pause_seconds": 0.0,
            "boundary_seconds": 3.4,
        },
    )

    for field in (
        "index=4",
        "reason=environment",
        "success=1",
        "return=0.00",
        "length=25",
        "fps=19.5",
        "intervention_steps=5",
        "intervention_rate=0.200",
        "updates=100",
        "learner_sec=3.00",
        "reset_sec=0.40",
        "checkpoint_sec=0.00",
        "pause_sec=0.00",
        "boundary_sec=3.40",
    ):
        assert field in summary


def test_checkpoint_due_uses_completed_episode_count() -> None:
    assert not _checkpoint_due(19, 20)
    assert _checkpoint_due(20, 20)
    assert not _checkpoint_due(21, 20)
    assert _checkpoint_due(40, 20)


class _CheckpointAgent:
    def __init__(self) -> None:
        self.calls = []

    def save_checkpoint(self, path, *, include_buffers, trainer_state) -> None:
        self.calls.append((path, include_buffers, trainer_state))


class _CheckpointTrainer:
    total_env_steps = 1234
    total_episodes = 20

    def state_dict(self):
        return {
            "total_env_steps": self.total_env_steps,
            "total_episodes": self.total_episodes,
        }


def test_checkpoint_filename_and_resume_state_use_episode_count(tmp_path: Path) -> None:
    agent = _CheckpointAgent()
    trainer = _CheckpointTrainer()
    cfg = SimpleNamespace(
        checkpoint=SimpleNamespace(directory="checkpoints"),
    )

    saved = _save_checkpoint(
        agent,
        trainer,
        tmp_path,
        cfg,
        episode_index=20,
        success_count=7,
    )

    assert saved.name == "episode_00000020.pt"
    assert [call[0].name for call in agent.calls] == [
        "episode_00000020.pt",
        "latest.pt",
    ]
    assert [call[1] for call in agent.calls] == [False, True]
    trainer_state = agent.calls[0][2]
    assert trainer_state["total_env_steps"] == 1234
    assert trainer_state["total_episodes"] == 20
    assert trainer_state["episode_index"] == 20
    assert trainer_state["success_count"] == 7
