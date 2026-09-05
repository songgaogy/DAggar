from __future__ import annotations

import numpy as np
import pytest
import torch

from robosuite.pipeline.src.environment.observation import observation_batch_to_cuda
from robosuite.pipeline.src.environment.runtime import (
    build_policy_observation,
    build_runtime_config,
    reset_policy_observation,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


class FakeExtractor:
    def extract(self) -> np.ndarray:
        return np.array([1.0, 2.0], dtype=np.float32)


class FakeEnv:
    def reset(self):
        return {
            "front_image": np.zeros((8, 10, 3), dtype=np.uint8),
            "wrist_image": np.ones((8, 10, 3), dtype=np.uint8),
        }


def test_observation_collection_and_cuda_batch() -> None:
    env = FakeEnv()
    extractor = FakeExtractor()
    raw = env.reset()
    observation = build_policy_observation(
        env,
        extractor,
        camera_names=("a", "b"),
        camera_aliases={"a": "front", "b": "wrist"},
        image_height=8,
        image_width=10,
        raw_observation=raw,
    )
    images, proprio = observation_batch_to_cuda(
        observation,
        camera_names=("a", "b"),
        proprio_key="state",
        device="cuda:0",
    )
    assert images.shape == (1, 2, 3, 8, 10)
    assert images.dtype == torch.uint8
    assert proprio.shape == (1, 2)
    assert images.device.type == proprio.device.type == "cuda"


def test_legacy_bare_camera_keys_and_state_proprio_are_supported() -> None:
    observation = {
        "agentview": np.zeros((8, 10, 3), dtype=np.uint8),
        "robot0_robotview": np.ones((8, 10, 3), dtype=np.uint8),
        "robot0_eye_in_hand": np.full((8, 10, 3), 2, dtype=np.uint8),
        "state": np.asarray([3.0, 4.0], dtype=np.float32),
    }
    images, proprio = observation_batch_to_cuda(
        observation,
        camera_names=("agentview", "robot0_robotview", "robot0_eye_in_hand"),
        proprio_key="state",
        device="cuda:0",
    )
    assert images.shape == (1, 3, 3, 8, 10)
    assert torch.equal(images[:, 0], torch.zeros_like(images[:, 0]))
    assert torch.equal(images[:, 1], torch.ones_like(images[:, 1]))
    assert torch.equal(images[:, 2], torch.full_like(images[:, 2], 2))
    assert torch.equal(proprio, torch.tensor([[3.0, 4.0]], device="cuda:0"))


def test_live_camera_key_takes_precedence_over_legacy_alias() -> None:
    observation = {
        "agentview_image": np.full((4, 4, 3), 7, dtype=np.uint8),
        "agentview": np.full((4, 4, 3), 9, dtype=np.uint8),
        "state": np.asarray([1.0], dtype=np.float32),
    }
    images, _ = observation_batch_to_cuda(
        observation,
        camera_names=("agentview",),
        proprio_key="state",
        device="cuda:0",
    )
    assert torch.equal(images, torch.full_like(images, 7))


def test_runtime_config_and_reset_contract() -> None:
    config = build_runtime_config(
        {"env_name": "Fake", "robots": ["Panda"]},
        camera_names=("front", "wrist"),
        image_height=8,
        image_width=10,
        control_freq=20,
        horizon=500,
        interactive=False,
    )
    assert config.camera_names == ("front", "wrist")
    assert config.horizon == 500

    observation, info = reset_policy_observation(
        FakeEnv(),
        preserve_mjviewer=False,
        extractor=FakeExtractor(),
        camera_names=("front", "wrist"),
        camera_aliases={},
        image_height=8,
        image_width=10,
    )
    assert set(observation) == {"front_image", "wrist_image", "state"}
    assert info == {}
