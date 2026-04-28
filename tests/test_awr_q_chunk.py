from __future__ import annotations

import numpy as np
import pytest

from robosuite.pipeline.algorithms.awr.replay_buffer import AWRReplayBuffer
from robosuite.pipeline.common.types import ReplayBufferConfig, Transition


def _obs(step: int) -> dict[str, np.ndarray]:
    return {
        "state": np.asarray([float(step)], dtype=np.float32),
        "agentview": np.full((4, 4, 3), step, dtype=np.uint8),
    }


def _transition(*, episode: int, step: int, reward: float, done: bool) -> Transition:
    return Transition(
        obs=_obs(step),
        action=np.asarray([float(step), float(step) + 0.5], dtype=np.float32),
        reward=float(reward),
        next_obs=_obs(step + 1),
        done=bool(done),
        info={
            "episode_namespace": "test",
            "episode_index": int(episode),
            "episode_step": int(step),
        },
        reward_source="test",
        demo_source="test",
    )


def _buffer(horizon: int = 3) -> AWRReplayBuffer:
    return AWRReplayBuffer(
        ReplayBufferConfig(capacity=32, batch_size=4),
        name="test_buffer",
        camera_names=["agentview"],
        action_horizon=horizon,
        image_size=4,
        await_discriminator_labels=False,
    )


def test_q_chunk_batch_uses_discounted_rewards_and_terminal_mask() -> None:
    buffer = _buffer(horizon=3)
    buffer.extend(
        [
            _transition(episode=0, step=0, reward=1.0, done=False),
            _transition(episode=0, step=1, reward=2.0, done=False),
            _transition(episode=0, step=2, reward=3.0, done=True),
        ]
    )

    batch = buffer.sample_step_batch(batch_size=1, discount=0.5, augment=False)

    np.testing.assert_allclose(batch.rewards.numpy(), np.asarray([[2.75]], dtype=np.float32))
    np.testing.assert_allclose(batch.dones.numpy(), np.asarray([[1.0]], dtype=np.float32))
    assert tuple(batch.actions.shape) == (1, 3, 2)
    np.testing.assert_allclose(batch.next_proprio.numpy(), np.asarray([[3.0]], dtype=np.float32))


def test_q_chunk_sampling_rejects_short_episode_tail() -> None:
    buffer = _buffer(horizon=3)
    buffer.extend(
        [
            _transition(episode=0, step=0, reward=1.0, done=False),
            _transition(episode=0, step=1, reward=1.0, done=True),
        ]
    )

    assert buffer.num_valid_sequences() == 0
    assert buffer.num_ready_steps() == 0
    with pytest.raises(ValueError, match="ready critic sequences"):
        buffer.sample_step_batch(batch_size=1, discount=0.99, augment=False)


def test_q_chunk_sampling_rejects_done_inside_chunk() -> None:
    buffer = _buffer(horizon=3)
    buffer.extend(
        [
            _transition(episode=0, step=0, reward=1.0, done=False),
            _transition(episode=0, step=1, reward=1.0, done=True),
            _transition(episode=1, step=0, reward=1.0, done=False),
            _transition(episode=1, step=1, reward=1.0, done=False),
            _transition(episode=1, step=2, reward=1.0, done=True),
        ]
    )

    assert buffer.num_valid_sequences() == 1
    batch = buffer.sample_step_batch(batch_size=1, discount=1.0, augment=False)
    assert batch.metadata["episode_ids"] == [1]
