from pathlib import Path
import pytest
import torch
from hydra import compose, initialize_config_dir

from robosuite.pipeline.src.environment import RobosuiteObservationAdapter, build_robosuite_env
from robosuite.pipeline.src.hil_serl import HILSERLAgent
from robosuite.pipeline.train import _algorithm_config
from robosuite.pipeline.utils.runtime import (
    build_runtime_cfg,
    reset_observation_adapter,
    resolve_camera_names,
    set_seed,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
CONFIG_DIR = Path(__file__).parents[1] / "config"


def test_headless_environment_and_cuda_policy_single_step() -> None:
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR.resolve())):
        cfg = compose(
            config_name="overall",
            overrides=[
                "task=PickPlaceCereal",
                "runtime.learner_device=cuda:0",
                "runtime.inference_device=cuda:0",
            ],
        )
    set_seed(int(cfg.seed))
    cameras = resolve_camera_names(cfg)
    env = build_robosuite_env(
        build_runtime_cfg(
            cfg,
            cameras,
            has_renderer=False,
            has_offscreen_renderer=False,
        )
    )
    render_env = build_robosuite_env(
        build_runtime_cfg(
            cfg,
            cameras,
            has_renderer=False,
            has_offscreen_renderer=True,
        )
    )
    try:
        render_env.reset()
        adapter = RobosuiteObservationAdapter(
            env,
            render_env=render_env,
            camera_names=cameras,
            img_height=int(cfg.env.img_height),
            img_width=int(cfg.env.img_width),
            proprio_keys=tuple(cfg.env.proprio_keys or []),
            image_obs_fps=float(cfg.runtime.image_obs_fps),
        )
        observation, _ = reset_observation_adapter(adapter, preserve_mjviewer=False)
        low, high = adapter.action_spec()
        agent = HILSERLAgent.from_config(
            _algorithm_config(cfg),
            observation_example=observation,
            action_low=low,
            action_high=high,
        )
        action = agent.select_action(observation, deterministic=False)
        step_output = env.step(action)
        assert len(step_output) in (4, 5)
        assert action.shape == low.shape
    finally:
        env.close()
        render_env.close()
