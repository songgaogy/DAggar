import threading
import time

import numpy as np

from robosuite.pipeline.src.data.transitions import Transition
from robosuite.pipeline.src.environment import RobosuiteObservationAdapter, unpack_robosuite_step
from robosuite.pipeline.src.hil_serl import HILSERLTrainer
from robosuite.pipeline.src.hil_serl.agent import TrainerConfig
from robosuite.pipeline.train import _episode_checkpoint_due


class FakeAgent:
    def __init__(self, config: TrainerConfig) -> None:
        self.trainer_config = config
        self.online = []
        self.demo = []
        self.update_modes = []
        self.publish_count = 0

    def store_demo_transition(self, transition: Transition) -> None:
        self.demo.append(transition)

    def store_transition(self, transition: Transition) -> None:
        self.online.append(transition)
        if transition.is_intervention:
            self.demo.append(transition)

    def ready_for_update(self, batch_size=None) -> bool:
        return True

    def sample_mixed_batch(self, batch_size=None):
        return object()

    def update(self, *, batch, critic_only: bool):
        self.update_modes.append(bool(critic_only))
        return {} if critic_only else {"alpha_loss": 0.0}

    def sync_inference_policy(self) -> None:
        self.publish_count += 1


def _transition(**overrides) -> Transition:
    values = {
        "obs": {"state": np.zeros(2, dtype=np.float32)},
        "action": np.zeros(7, dtype=np.float32),
        "reward": None,
        "next_obs": {"state": np.ones(2, dtype=np.float32)},
        "done": False,
    }
    values.update(overrides)
    return Transition(**values)


def test_success_reward_terminal_and_timeout_bootstrap() -> None:
    agent = FakeAgent(TrainerConfig(batch_size=2))
    trainer = HILSERLTrainer(agent)

    timeout = trainer.record_transition(
        obs=_transition().obs,
        action=np.zeros(7, dtype=np.float32),
        next_obs=_transition().next_obs,
        terminated=False,
        truncated=True,
        is_success=False,
    )
    success = trainer.record_transition(
        obs=_transition().obs,
        action=np.zeros(7, dtype=np.float32),
        next_obs=_transition().next_obs,
        terminated=False,
        truncated=False,
        is_success=True,
    )

    assert timeout.reward == 0.0
    assert timeout.done is False
    assert timeout.truncated is True
    assert success.reward == 1.0
    assert success.done is True


def test_legacy_robosuite_done_is_time_limit() -> None:
    _, _, terminated, truncated, _ = unpack_robosuite_step(("obs", 0.0, True, {}))
    assert terminated is False
    assert truncated is True


def test_observation_adapter_distinguishes_omitted_empty_and_explicit_proprio() -> None:
    raw_obs = {
        "robot-state": np.array([1.0, 2.0], dtype=np.float32),
        "object-state": np.array([3.0], dtype=np.float32),
    }
    images = {"camera": np.zeros((4, 4, 3), dtype=np.uint8)}

    inferred = RobosuiteObservationAdapter(
        object(), camera_names=["camera"], img_height=4, img_width=4
    ).transform(raw_obs, images=images)
    image_only = RobosuiteObservationAdapter(
        object(), camera_names=["camera"], img_height=4, img_width=4, proprio_keys=[]
    ).transform(raw_obs, images=images)
    explicit = RobosuiteObservationAdapter(
        object(),
        camera_names=["camera"],
        img_height=4,
        img_width=4,
        proprio_keys=["robot-state"],
    ).transform(raw_obs, images=images)

    np.testing.assert_array_equal(inferred["state"], [1.0, 2.0, 3.0])
    assert set(image_only) == {"camera"}
    np.testing.assert_array_equal(explicit["state"], [1.0, 2.0])


def test_intervention_transition_is_written_to_both_replays() -> None:
    agent = FakeAgent(TrainerConfig(batch_size=2))
    trainer = HILSERLTrainer(agent)
    transition = trainer.record_transition(
        obs=_transition().obs,
        action=np.ones(7, dtype=np.float32),
        next_obs=_transition().next_obs,
        terminated=False,
        is_success=False,
        is_intervention=True,
    )

    assert agent.online == [transition]
    assert agent.demo == [transition]


def test_cta_counts_outer_learner_steps_and_publishes() -> None:
    config = TrainerConfig(batch_size=2, cta_ratio=3, steps_per_update=2)
    agent = FakeAgent(config)
    trainer = HILSERLTrainer(agent)

    trainer.train_step()
    assert agent.update_modes == [True, True, False]
    assert trainer.total_updates == 1
    assert trainer.total_critic_updates == 3
    assert trainer.total_actor_updates == 1
    assert trainer.total_temperature_updates == 1

    trainer.train_step()
    assert trainer._maybe_publish_inference_policy() is True
    assert trainer.total_updates == 2
    assert agent.publish_count == 1


def test_continuous_learner_runs_without_update_queue() -> None:
    config = TrainerConfig(
        batch_size=2,
        cta_ratio=2,
        steps_per_update=2,
        max_learner_steps=4,
    )
    agent = FakeAgent(config)
    trainer = HILSERLTrainer(agent)
    trainer.start_async_worker()
    deadline = time.monotonic() + 2.0
    while not trainer.learner_finished and time.monotonic() < deadline:
        time.sleep(0.01)
    trainer.close_async_worker()

    assert trainer.total_updates == 4
    assert trainer.dropped_async_updates() == 0
    assert agent.publish_count == 2


def test_episode_checkpoint_due_uses_completed_online_episodes() -> None:
    assert _episode_checkpoint_due(20, 20) is True
    assert _episode_checkpoint_due(40, 20) is True
    assert _episode_checkpoint_due(0, 20) is False
    assert _episode_checkpoint_due(19, 20) is False
    assert _episode_checkpoint_due(21, 20) is False
    assert _episode_checkpoint_due(20, 0) is False
