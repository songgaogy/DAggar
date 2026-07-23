from pathlib import Path

import numpy as np

from robosuite.pipeline.src.data import Transition, TransitionChunkWriter
from robosuite.pipeline.src.awr import AWRTrainer, TrainerConfig
from robosuite.pipeline.src.environment import sparse_success_reward


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
            updates_per_episode=100,
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
