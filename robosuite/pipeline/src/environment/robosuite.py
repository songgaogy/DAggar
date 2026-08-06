from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import robosuite as suite
from robosuite.controllers import load_composite_controller_config


@dataclass(frozen=True)
class RobosuiteRuntimeConfig:
    env_name: str
    robots: tuple[str, ...]
    camera_names: tuple[str, ...]
    image_height: int
    image_width: int
    control_freq: int
    horizon: int
    controller: str | None = None
    controller_configs: dict[str, Any] | None = None
    env_configuration: str | None = None
    renderer: str | None = None
    render_camera: str | None = None
    has_renderer: bool = False


def build_runtime_config(
    env_metadata: dict[str, Any],
    *,
    camera_names: Sequence[str],
    image_height: int,
    image_width: int,
    control_freq: int,
    horizon: int,
    interactive: bool,
) -> RobosuiteRuntimeConfig:
    if not env_metadata:
        raise ValueError("Flow checkpoint environment metadata is required.")
    cameras = tuple(str(name) for name in camera_names)
    if not cameras:
        raise ValueError("AWR requires at least one policy camera.")
    return RobosuiteRuntimeConfig(
        env_name=str(env_metadata["env_name"]),
        robots=tuple(str(name) for name in env_metadata["robots"]),
        camera_names=cameras,
        image_height=int(image_height),
        image_width=int(image_width),
        control_freq=int(control_freq),
        horizon=int(horizon),
        controller=env_metadata.get("controller"),
        controller_configs=env_metadata.get("controller_configs"),
        env_configuration=env_metadata.get("env_configuration", env_metadata.get("config")),
        renderer="mjviewer" if interactive else None,
        render_camera=cameras[0] if interactive else None,
        has_renderer=bool(interactive),
    )


def build_robosuite_env(config: RobosuiteRuntimeConfig):
    controller_configs = config.controller_configs
    if controller_configs is None:
        controller_configs = load_composite_controller_config(
            controller=config.controller,
            robot=config.robots[0],
        )
    if controller_configs["type"] == "WHOLE_BODY_MINK_IK":
        from robosuite.examples.third_party_controller.mink_controller import WholeBodyMinkIK  # noqa: F401

    kwargs: dict[str, Any] = {
        "env_name": config.env_name,
        "robots": list(config.robots),
        "controller_configs": controller_configs,
        "has_renderer": config.has_renderer,
        "has_offscreen_renderer": True,
        "ignore_done": False,
        "use_camera_obs": True,
        "camera_names": list(config.camera_names),
        "camera_heights": config.image_height,
        "camera_widths": config.image_width,
        "reward_shaping": False,
        "control_freq": config.control_freq,
        "horizon": config.horizon,
    }
    if config.renderer is not None:
        kwargs["renderer"] = config.renderer
    if config.render_camera is not None:
        kwargs["render_camera"] = config.render_camera
    if config.env_configuration and "TwoArm" in config.env_name:
        kwargs["env_configuration"] = config.env_configuration
    return suite.make(**kwargs)


def reset_robosuite_env(env, *, preserve_mjviewer: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    if not preserve_mjviewer:
        result = env.reset()
        return result if isinstance(result, tuple) else (result, {})
    if str(getattr(env, "renderer", "")).lower() != "mjviewer":
        raise ValueError("preserve_mjviewer requires an interactive mjviewer environment.")

    original_hard_reset = bool(getattr(env, "hard_reset", False))
    env.hard_reset = False
    try:
        env.sim.reset()
        env._reset_internal()
        env.sim.forward()
        env._obs_cache = {}
        env._reset_observables()
        env.visualize(vis_settings={name: False for name in env._visualizations})
        env.update_state()
        if getattr(env, "viewer", None) is not None and hasattr(env.viewer, "reset"):
            env.viewer.reset()
        if env.viewer_get_obs:
            return env.viewer._get_observations(force_update=True), {}
        return env._get_observations(force_update=True), {}
    finally:
        env.hard_reset = original_hard_reset


def render_mjviewer(env, *, visualize_gripper_markers: bool = False) -> None:
    if str(getattr(env, "renderer", "")).lower() != "mjviewer":
        raise ValueError("Interactive training requires renderer='mjviewer'.")
    if env.viewer is None:
        env.initialize_renderer()
        env.viewer_get_obs = hasattr(env.viewer, "_get_observations")
    if not visualize_gripper_markers:
        env.viewer.update()
        return

    hidden_visuals = {name: False for name in env._visualizations}
    viewer_visuals = {name: name == "grippers" for name in env._visualizations}
    try:
        env.visualize(vis_settings=viewer_visuals)
        env.viewer.update()
    finally:
        env.visualize(vis_settings=hidden_visuals)
