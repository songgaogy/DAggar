from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

from robosuite.pipeline.src.environment import (
    RobosuiteObservationAdapter,
    build_robosuite_env,
    compute_grasp_penalty,
    estimate_gripper_openness,
)
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


def test_headless_environment_and_cuda_policy_single_step(tmp_path: Path, monkeypatch) -> None:
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
        raw_observation = env.reset()
        assert env.use_object_obs is False
        assert "object-state" not in raw_observation
        assert not any(key.startswith("Cereal") for key in raw_observation)
        render_env.reset()
        render_context = render_env.sim._render_context_offscreen
        make_current_calls = 0
        original_make_current = render_context.gl_ctx.make_current

        def tracked_make_current() -> None:
            nonlocal make_current_calls
            make_current_calls += 1
            original_make_current()

        monkeypatch.setattr(render_context.gl_ctx, "make_current", tracked_make_current)
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
        assert make_current_calls >= len(cameras)
        assert set(observation) == set(cameras)
        assert "state" not in observation
        low, high = adapter.action_spec()
        agent = HILSERLAgent.from_config(
            _algorithm_config(cfg),
            observation_example=observation,
            action_low=low,
            action_high=high,
        )
        assert agent.core.encoder.state_projector is None
        agent.save_checkpoint(tmp_path / "image_only.pt", include_buffers=False)
        action = agent.select_action(observation, deterministic=False)
        step_output = env.step(action)
        assert len(step_output) in (4, 5)
        assert action.shape == low.shape

        gripper_action = np.zeros_like(low, dtype=np.float32)
        gripper_action[-1] = -1.0
        for _ in range(20):
            env.step(gripper_action)
        assert estimate_gripper_openness(env) > 0.9
        assert compute_grasp_penalty(env, gripper_action) == pytest.approx(-0.02)

        gripper_action[-1] = 1.0
        for _ in range(20):
            env.step(gripper_action)
        assert estimate_gripper_openness(env) < 0.1
        assert compute_grasp_penalty(env, gripper_action) == pytest.approx(-0.02)
    finally:
        env.close()
        render_env.close()
