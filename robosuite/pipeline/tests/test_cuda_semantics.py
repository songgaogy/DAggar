import numpy as np
import pytest
import torch

from robosuite.pipeline.src.awr.config import (
    AWRConfig,
    FlowAugmentationConfig,
    require_cuda_device,
)
from robosuite.pipeline.src.awr.replay_buffer import AWRReplayBuffer
from robosuite.pipeline.src.data import ReplayBufferConfig, Transition
from robosuite.pipeline.utils.tensor import resolve_cuda_device


def _chunk_transition(index: int, *, reward: float, done: bool = False) -> Transition:
    obs = {
        "camera": np.full((8, 8, 3), index, dtype=np.uint8),
        "state": np.full(2, index, dtype=np.float32),
    }
    next_obs = {
        "camera": np.full((8, 8, 3), index + 1, dtype=np.uint8),
        "state": np.full(2, index + 1, dtype=np.float32),
    }
    return Transition(
        obs=obs,
        action=np.full(2, index, dtype=np.float32),
        reward=reward,
        next_obs=next_obs,
        done=done,
        info={"episode_index": 1, "episode_step": index},
    )


def _replay(action_horizon: int) -> AWRReplayBuffer:
    return AWRReplayBuffer(
        ReplayBufferConfig(capacity=16, batch_size=1),
        name="test",
        camera_names=["camera"],
        action_horizon=action_horizon,
        image_size=8,
        augmentation_config=FlowAugmentationConfig(
            minimal_shift_pad=0,
            eye_in_hand_crop_scale=1.0,
        ),
    )


def test_cpu_device_is_rejected_by_awr_config() -> None:
    with pytest.raises(ValueError, match="must be a CUDA device"):
        require_cuda_device("cpu", name="test device")
    with pytest.raises(ValueError, match="must be a CUDA device"):
        AWRConfig(action_dim=7, proprio_dim=9, device="cpu")


def test_short_episode_does_not_form_q_chunk() -> None:
    replay = _replay(action_horizon=3)
    replay.add(_chunk_transition(0, reward=-1.0))
    replay.add(_chunk_transition(1, reward=0.0, done=True))
    assert replay.num_valid_sequences() == 0


def test_resolved_cuda_device_never_falls_back_to_cpu() -> None:
    device = resolve_cuda_device("cuda:0")
    assert device.type == "cuda"


def test_q_chunk_discounted_return_and_terminal_mask_stay_on_cuda() -> None:
    replay = _replay(action_horizon=3)
    replay.add(_chunk_transition(0, reward=-1.0))
    replay.add(_chunk_transition(1, reward=-1.0))
    replay.add(_chunk_transition(2, reward=0.0, done=True))

    batch = replay.sample_step_batch(
        1,
        discount=0.5,
        device="cuda:0",
        augment=False,
    )

    assert batch.rewards.is_cuda
    assert batch.dones.is_cuda
    assert batch.actions.is_cuda
    assert batch.rewards.item() == pytest.approx(-1.5)
    assert batch.dones.item() == 1.0
