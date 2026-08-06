from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


CONFIG_DIR = Path(__file__).parents[1] / "config"


def compose_config():
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR.resolve())):
        return compose(config_name="overall", overrides=["task=PickPlaceCereal"])


def test_pick_place_cereal_config_resolves() -> None:
    resolved = OmegaConf.to_container(compose_config(), resolve=True)

    assert resolved["task"]["name"] == "PickPlaceCereal"
    assert resolved["env"]["environment"] == "PickPlaceCereal"
    assert resolved["env"]["horizon"] == 500
    assert resolved["awr"]["discount"] == 0.97
    assert resolved["awr"]["task_name"] == "PickPlaceCereal"
    assert (
        resolved["data"]["expert_dir"]
        == "./data/PickPlaceCereal/pretrain_data-20260615_174814"
    )
    assert resolved["data"]["success_dir"] == "./data/PickPlaceCereal/success_rollout"
    assert resolved["data"]["fail_dir"] == "./data/PickPlaceCereal/fail_rollout"
    assert resolved["trainer"]["episodes_per_train"] == 10
    assert resolved["trainer"]["updates_per_train"] == 2000
    assert "updates_per_episode" not in resolved["trainer"]
    assert resolved["runtime"]["learner_device"] == "cuda:0"
    assert resolved["runtime"]["inference_device"] == "cuda:1"
    assert resolved["logging"]["output_root"] == "./outputs/baseline/awr"
    assert resolved["checkpoint"]["interval_episodes"] == 20
    assert "interval_env_steps" not in resolved["checkpoint"]
    assert resolved["runtime"]["visualize_gripper_markers"] is True
    assert resolved["runtime"]["episode_pause_sec"] == 0.0


def test_config_exposes_only_awr_training_semantics() -> None:
    resolved = OmegaConf.to_container(compose_config(), resolve=True)
    serialized = OmegaConf.to_yaml(OmegaConf.create(resolved)).lower()

    assert "algorithm" not in resolved
    assert "discriminator" not in resolved
    assert "wandb" not in serialized
    assert "async" not in serialized
    assert "updates_per_step" not in serialized
    assert resolved["runtime"]["interactive"] is True
    assert resolved["env"]["renderer"] == "mjviewer"
    assert resolved["evaluation"]["viewer_enabled"] is False
    assert resolved["evaluation"]["deterministic"] is True


def test_every_tensor_device_is_explicitly_cuda() -> None:
    resolved = OmegaConf.to_container(compose_config(), resolve=True)
    configured_devices = [
        resolved["runtime"]["learner_device"],
        resolved["runtime"]["inference_device"],
        resolved["awr"]["device"],
        resolved["awr"]["inference_device"],
    ]

    assert all(str(device).startswith("cuda") for device in configured_devices)
