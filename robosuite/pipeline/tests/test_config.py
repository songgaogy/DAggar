from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from robosuite.pipeline.utils.logging import TensorBoardLogger


CONFIG_DIR = Path(__file__).parents[1] / "config"


def _compose_config():
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR.resolve())):
        return compose(config_name="overall", overrides=["task=PickPlaceCereal"])


def test_pick_place_cereal_config_resolves() -> None:
    cfg = _compose_config()
    resolved = OmegaConf.to_container(cfg, resolve=True)

    assert resolved["env"]["environment"] == "PickPlaceCereal"
    assert resolved["env"]["horizon"] == 500
    assert resolved["env"]["renderer"] == "mujoco"
    assert resolved["env"]["viewer_width"] == 1280
    assert resolved["env"]["viewer_height"] == 800
    assert resolved["env"]["use_object_obs"] is False
    assert resolved["env"]["proprio_keys"] == []
    assert resolved["algorithm"]["encoder"]["proprio_keys"] == []
    assert resolved["data"]["demo_path"] == "./data/PickPlaceCereal/expert"
    assert resolved["data"]["num_trajectories"] == 20
    assert resolved["data"]["random_sample"] is True
    assert resolved["data"]["random_seed"] == 42
    assert resolved["data"]["selection_manifest_filename"] == "demo_selection.json"
    assert resolved["runtime"]["learner_device"] == "cuda:0"
    assert resolved["runtime"]["inference_device"] == "cuda:1"
    assert resolved["algorithm"]["sac"]["discount"] == 0.97
    assert resolved["algorithm"]["trainer"]["online_fraction"] == 0.5
    assert resolved["algorithm"]["grasp_penalty"]["penalty"] == -0.02
    assert resolved["checkpoint"]["interval_online_episodes"] == 20
    assert "interval_learner_steps" not in resolved["checkpoint"]
    assert resolved["logging"]["output_root"] == "./outputs/baseline/hil-serl"


def test_tensorboard_logger_writes_required_metric_groups(tmp_path: Path) -> None:
    log_dir = tmp_path / "tensorboard"
    with TensorBoardLogger(log_dir, flush_secs=1) as logger:
        train_count = logger.log(
            {"critic_loss": 1.25, "invalid": float("nan"), "text": "skip"},
            step=7,
            prefix="train",
        )
        logger.log({"learner_updates_per_second": 3.0, "intervention_step_ratio": 0.25}, step=7, prefix="runtime")
        logger.log({"success": 1.0, "intervention_segments": 2.0}, step=1, prefix="episode")
        logger.flush()

    assert train_count == 1
    accumulator = EventAccumulator(str(log_dir))
    accumulator.Reload()
    tags = set(accumulator.Tags()["scalars"])
    assert {
        "train/critic_loss",
        "runtime/learner_updates_per_second",
        "runtime/intervention_step_ratio",
        "episode/success",
        "episode/intervention_segments",
    } <= tags
    event = accumulator.Scalars("train/critic_loss")[0]
    assert event.step == 7
    assert event.value == 1.25


def test_disabled_tensorboard_logger_has_no_side_effects(tmp_path: Path) -> None:
    log_dir = tmp_path / "tensorboard"
    with TensorBoardLogger(log_dir, enabled=False) as logger:
        assert logger.log({"loss": 1.0}, step=1) == 0

    assert not log_dir.exists()
