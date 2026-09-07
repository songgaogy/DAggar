from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


CONFIG_DIR = Path(__file__).parents[1] / "config"


def compose_config():
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR.resolve())):
        return compose(config_name="overall", overrides=["task=PickPlaceCereal"])


def test_pick_place_cereal_config_is_decision_complete() -> None:
    cfg = OmegaConf.to_container(compose_config(), resolve=True)
    assert cfg["task"] == {"name": "PickPlaceCereal", "horizon": 500}
    assert cfg["runtime"]["num_envs"] == 4
    assert cfg["runtime"]["total_episodes"] == 200
    assert cfg["warmup"] == {"episodes": 20, "seed": 42}
    assert cfg["flow"]["action_horizon"] == 8
    assert cfg["flow"]["action_dim"] == 7
    assert cfg["algorithm"]["hidden_dims"] == [1024, 1024, 1024]
    assert cfg["algorithm"]["utd"] == 30
    assert cfg["algorithm"]["update_interval_vector_steps"] == 2
    assert cfg["algorithm"]["gamma"] == 0.97
    assert cfg["algorithm"]["latent_action_bound"] == 2.0
    assert cfg["storage"]["checkpoint_interval_episodes"] == 20
    assert cfg["storage"]["output_root"] == "./outputs/baseline/dsrl"


def test_config_is_cuda_only_and_has_no_human_or_wandb_paths() -> None:
    cfg = OmegaConf.to_container(compose_config(), resolve=True)
    serialized = OmegaConf.to_yaml(OmegaConf.create(cfg)).lower()
    assert cfg["runtime"]["learner_device"].startswith("cuda")
    assert cfg["runtime"]["inference_device"].startswith("cuda")
    assert cfg["runtime"]["headless"] is True
    assert cfg["runtime"]["viewer_enabled"] is False
    assert "wandb" not in serialized
    assert "intervention" not in serialized
    assert "spacemouse" not in serialized
    assert "offline_data-vast" not in serialized
    assert "dinov2" not in serialized
