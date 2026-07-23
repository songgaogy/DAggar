import numpy as np
import pytest
import torch

from robosuite.pipeline.src.data.transitions import Transition
from robosuite.pipeline.src.hil_serl import HILSERLAgent, HILSERLTrainer
from robosuite.pipeline.src.hil_serl.sac import GaussianPolicy
from robosuite.pipeline.utils.tensor import random_crop_observations, require_cuda_device


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def test_random_crop_stays_on_cuda_and_preserves_shape() -> None:
    images = torch.arange(2 * 8 * 8 * 3, device="cuda:0").reshape(2, 8, 8, 3)
    cropped = random_crop_observations({"camera": images}, ["camera"], padding=4)["camera"]
    assert cropped.is_cuda
    assert cropped.shape == images.shape


def test_policy_sampling_uses_cuda_temperature() -> None:
    policy = GaussianPolicy(
        input_dim=4,
        hidden_dims=[8, 8],
        action_dim=2,
        std_min=1e-5,
        std_max=5.0,
        action_low=-np.ones(2, dtype=np.float32),
        action_high=np.ones(2, dtype=np.float32),
    ).to("cuda:0")
    features = torch.zeros(3, 4, device="cuda:0")
    action, log_prob = policy.sample(
        features,
        temperature=torch.tensor([1e-2], device="cuda:0"),
    )
    assert action.is_cuda
    assert log_prob is not None and log_prob.is_cuda
    assert action.shape == (3, 2)
    assert log_prob.shape == (3, 1)


def test_cpu_device_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be a CUDA device"):
        require_cuda_device("cpu")


def test_full_hybrid_update_and_checkpoint_round_trip_on_cuda(tmp_path) -> None:
    observation = {
        "camera": np.zeros((16, 16, 3), dtype=np.uint8),
        "state": np.zeros(4, dtype=np.float32),
    }
    config = {
        "encoder": {
            "encoder_type": "simple_cnn",
            "image_keys": ["camera"],
            "proprio_keys": ["state"],
            "feature_dim": 16,
            "proprio_feature_dim": 8,
            "image_size": 16,
            "cnn_channels": [8, 8],
            "pretrained": False,
        },
        "sac": {
            "actor_hidden_dims": [16, 16],
            "critic_hidden_dims": [16, 16],
            "device": "cuda:0",
            "inference_device": "cuda:0",
            "augmentation_padding": 4,
        },
        "online_buffer": {"capacity": 16},
        "demo_buffer": {"capacity": 16},
        "trainer": {
            "batch_size": 4,
            "cta_ratio": 2,
            "warmup_steps": 1,
            "steps_per_update": 50,
            "online_fraction": 0.5,
            "max_learner_steps": 2,
        },
    }
    agent = HILSERLAgent.from_config(
        config,
        observation_example=observation,
        action_low=-np.ones(7, dtype=np.float32),
        action_high=np.ones(7, dtype=np.float32),
    )
    for index in range(2):
        transition = Transition(
            obs={key: value.copy() for key, value in observation.items()},
            action=np.zeros(7, dtype=np.float32),
            reward=float(index == 1),
            next_obs={key: value.copy() for key, value in observation.items()},
            done=bool(index == 1),
            grasp_penalty=0.0,
            is_intervention=False,
        )
        agent.store_online_transition(transition)
        transition.is_intervention = True
        agent.store_demo_transition(transition)

    mixed = agent.sample_mixed_batch()
    assert mixed.actions.is_cuda
    assert mixed.batch_size == 4
    assert mixed.is_intervention[:2].sum().item() == 0
    assert mixed.is_intervention[2:].sum().item() == 2

    trainer = HILSERLTrainer(agent)
    metrics = trainer.train_step()
    assert np.isfinite(metrics["critic_loss"])
    assert np.isfinite(metrics["actor_loss"])
    assert np.isfinite(metrics["alpha_loss"])

    checkpoint = tmp_path / "checkpoint.pt"
    agent.save_checkpoint(
        checkpoint,
        include_buffers=True,
        extra={"trainer_state": trainer.state_dict()},
    )
    restored = HILSERLAgent.from_config(
        config,
        observation_example=observation,
        action_low=-np.ones(7, dtype=np.float32),
        action_high=np.ones(7, dtype=np.float32),
    )
    extra = restored.load_checkpoint(checkpoint, load_buffers=True)
    assert extra["trainer_state"]["total_updates"] == 1
    assert len(restored.online_buffer) == len(agent.online_buffer)
    assert len(restored.demo_buffer) == len(agent.demo_buffer)
    np.testing.assert_allclose(
        restored.select_action(observation, deterministic=True),
        agent.select_action(observation, deterministic=True),
        atol=1e-6,
    )
