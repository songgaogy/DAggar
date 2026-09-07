from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import mujoco
import numpy as np
import robosuite as suite

from .observation import camera_obs_key


@dataclass(frozen=True)
class RobosuiteRuntimeConfig:
    env_name: str
    robots: tuple[str, ...]
    controller_configs: Mapping[str, Any] | None
    env_configuration: str | None
    camera_names: tuple[str, ...]
    image_height: int
    image_width: int
    horizon: int
    control_freq: int = 20
    interactive: bool = False
    seed: int | None = None


def build_runtime_config(
    env_metadata: Mapping[str, Any],
    camera_names: Sequence[str],
    image_height: int,
    image_width: int,
    control_freq: int = 20,
    horizon: int = 500,
    interactive: bool = False,
    seed: int | None = None,
) -> RobosuiteRuntimeConfig:
    missing = [key for key in ("env_name", "robots") if key not in env_metadata]
    if missing:
        raise KeyError(f"Environment metadata is missing required keys: {missing}")
    cameras = tuple(str(name) for name in camera_names)
    if not cameras:
        raise ValueError("At least one policy camera is required")
    if int(image_height) <= 0 or int(image_width) <= 0 or int(horizon) <= 0:
        raise ValueError("image dimensions and horizon must be positive")
    return RobosuiteRuntimeConfig(
        env_name=str(env_metadata["env_name"]),
        robots=tuple(str(name) for name in env_metadata["robots"]),
        controller_configs=env_metadata.get("controller_configs"),
        env_configuration=env_metadata.get("env_configuration", env_metadata.get("config")),
        camera_names=cameras,
        image_height=int(image_height),
        image_width=int(image_width),
        horizon=int(horizon),
        control_freq=int(control_freq),
        interactive=bool(interactive),
        seed=None if seed is None else int(seed),
    )


def build_robosuite_env(config: RobosuiteRuntimeConfig):
    kwargs: dict[str, Any] = {
        "env_name": config.env_name,
        "robots": list(config.robots),
        "controller_configs": config.controller_configs,
        "has_renderer": config.interactive,
        "has_offscreen_renderer": True,
        "ignore_done": False,
        "use_camera_obs": True,
        "reward_shaping": False,
        "control_freq": config.control_freq,
        "camera_names": list(config.camera_names),
        "camera_heights": config.image_height,
        "camera_widths": config.image_width,
        "horizon": config.horizon,
        "seed": config.seed,
    }
    if config.env_configuration is not None and "TwoArm" in config.env_name:
        kwargs["env_configuration"] = config.env_configuration
    return suite.make(**kwargs)


class RobosuiteProprioExtractor:
    """Extract the checkpoint-compatible robot qpos/qvel vector from an existing env."""

    def __init__(self, env: Any) -> None:
        self.env = env
        model = env.sim.model
        robot_joints = [str(name) for name in model.joint_names if str(name).startswith("robot0_")]
        if not robot_joints:
            raise RuntimeError("No robot0_ joints found in the environment")
        qpos_indices: list[int] = []
        qvel_indices: list[int] = []
        for joint_name in robot_joints:
            joint_id = model.joint_name2id(joint_name)
            qpos_count, qvel_count = self._joint_dims(int(model.jnt_type[joint_id]))
            qpos_start = int(model.jnt_qposadr[joint_id])
            qvel_start = int(model.jnt_dofadr[joint_id])
            qpos_indices.extend(range(qpos_start, qpos_start + qpos_count))
            qvel_indices.extend(range(qvel_start, qvel_start + qvel_count))
        self.qpos_indices = np.asarray(qpos_indices, dtype=np.int64)
        self.qvel_indices = np.asarray(qvel_indices, dtype=np.int64)

    @staticmethod
    def _joint_dims(joint_type: int) -> tuple[int, int]:
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            return 7, 6
        if joint_type == mujoco.mjtJoint.mjJNT_BALL:
            return 4, 3
        if joint_type in (mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE):
            return 1, 1
        raise RuntimeError(f"Unknown joint type: {joint_type}")

    def extract(self) -> np.ndarray:
        data = self.env.sim.data
        return np.concatenate(
            (np.asarray(data.qpos)[self.qpos_indices], np.asarray(data.qvel)[self.qvel_indices])
        ).astype(np.float32)

    def close(self) -> None:
        return None


def bind_proprio_extractor(
    env: Any, env_metadata: Mapping[str, Any] | None = None
) -> RobosuiteProprioExtractor:
    del env_metadata
    return RobosuiteProprioExtractor(env)


def build_policy_observation(
    env: Any,
    extractor: RobosuiteProprioExtractor,
    camera_names: Sequence[str],
    camera_aliases: Mapping[str, str] | None = None,
    image_height: int | None = None,
    image_width: int | None = None,
    raw_observation: Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    if raw_observation is None:
        raw_observation = getattr(env, "_observables", None)
        raise ValueError("raw_observation is required after an environment step")
    aliases = {} if camera_aliases is None else dict(camera_aliases)
    observation: dict[str, np.ndarray] = {}
    for camera_name in camera_names:
        source_name = aliases.get(str(camera_name), str(camera_name))
        source_key = camera_obs_key(source_name)
        output_key = camera_obs_key(str(camera_name))
        if source_key not in raw_observation:
            raise KeyError(f"Robosuite observation is missing camera key '{source_key}'")
        image = np.asarray(raw_observation[source_key], dtype=np.uint8)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Camera '{source_key}' must be HWC RGB, got {image.shape}")
        if image_height is not None and image.shape[0] != int(image_height):
            raise ValueError(f"Camera '{source_key}' height is {image.shape[0]}, expected {image_height}")
        if image_width is not None and image.shape[1] != int(image_width):
            raise ValueError(f"Camera '{source_key}' width is {image.shape[1]}, expected {image_width}")
        observation[output_key] = image
    observation["state"] = extractor.extract()
    return observation


def reset_policy_observation(
    env: Any,
    preserve_mjviewer: bool,
    extractor: RobosuiteProprioExtractor,
    camera_names: Sequence[str],
    camera_aliases: Mapping[str, str] | None = None,
    image_height: int | None = None,
    image_width: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    del preserve_mjviewer
    reset_output = env.reset()
    if isinstance(reset_output, tuple):
        raw_obs, info = reset_output
    else:
        raw_obs, info = reset_output, {}
    observation = build_policy_observation(
        env,
        extractor,
        camera_names,
        camera_aliases,
        image_height,
        image_width,
        raw_observation=raw_obs,
    )
    return observation, dict(info)
