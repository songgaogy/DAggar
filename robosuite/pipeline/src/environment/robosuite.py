from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
from PIL import Image

import robosuite as suite
from robosuite.controllers import load_composite_controller_config


@dataclass
class RobosuiteRuntimeConfig:
    env_name: str
    robots: Sequence[str]
    env_configuration: Optional[str] = None
    controller: Optional[str] = None
    controller_configs: Optional[dict[str, Any]] = None
    renderer: str = "mjviewer"
    render_camera: Optional[str] = None
    camera_names: Sequence[str] = ()
    img_height: int = 128
    img_width: int = 128
    reward_shaping: bool = False
    control_freq: int = 20
    has_renderer: bool = True
    has_offscreen_renderer: bool = True
    ignore_done: bool = False
    use_camera_obs: bool = False
    use_object_obs: bool = True
    proprio_keys: Optional[Sequence[str]] = None
    horizon: Optional[int] = None


@dataclass
class RobosuiteViewerSnapshot:
    state: np.ndarray
    timestep: int
    cur_time: float
    done: bool


def build_robosuite_env(runtime_cfg: RobosuiteRuntimeConfig):
    controller_config = runtime_cfg.controller_configs
    if controller_config is None:
        controller_config = load_composite_controller_config(
            controller=runtime_cfg.controller,
            robot=runtime_cfg.robots[0],
        )
    if controller_config["type"] == "WHOLE_BODY_MINK_IK":
        from robosuite.examples.third_party_controller.mink_controller import WholeBodyMinkIK  # noqa: F401

    env_kwargs = {
        "env_name": runtime_cfg.env_name,
        "robots": list(runtime_cfg.robots),
        "controller_configs": controller_config,
        "has_renderer": bool(runtime_cfg.has_renderer),
        "renderer": runtime_cfg.renderer,
        "has_offscreen_renderer": bool(runtime_cfg.has_offscreen_renderer),
        "ignore_done": bool(runtime_cfg.ignore_done),
        "use_camera_obs": bool(runtime_cfg.use_camera_obs),
        "use_object_obs": bool(runtime_cfg.use_object_obs),
        "reward_shaping": bool(runtime_cfg.reward_shaping),
        "control_freq": int(runtime_cfg.control_freq),
    }
    render_camera = runtime_cfg.render_camera
    if render_camera is None and runtime_cfg.camera_names:
        render_camera = str(runtime_cfg.camera_names[0])
    if render_camera is not None:
        env_kwargs["render_camera"] = str(render_camera)
    if runtime_cfg.use_camera_obs and runtime_cfg.camera_names:
        env_kwargs["camera_names"] = list(runtime_cfg.camera_names)
        env_kwargs["camera_heights"] = int(runtime_cfg.img_height)
        env_kwargs["camera_widths"] = int(runtime_cfg.img_width)
    if runtime_cfg.env_configuration and "TwoArm" in runtime_cfg.env_name:
        env_kwargs["env_configuration"] = runtime_cfg.env_configuration
    if runtime_cfg.horizon is not None:
        env_kwargs["horizon"] = int(runtime_cfg.horizon)
    return suite.make(**env_kwargs)


def build_runtime_config_from_env_info(
    env_info: dict[str, Any],
    *,
    camera_names: Sequence[str],
    img_height: int,
    img_width: int,
    proprio_keys: Sequence[str],
    has_renderer: bool,
    renderer: str,
    reward_shaping: bool,
    control_freq: int,
    use_object_obs: bool = False,
) -> RobosuiteRuntimeConfig:
    return RobosuiteRuntimeConfig(
        env_name=str(env_info["env_name"]),
        robots=[str(name) for name in env_info["robots"]],
        env_configuration=env_info.get("env_configuration", env_info.get("config")),
        controller=env_info.get("controller"),
        controller_configs=env_info.get("controller_configs"),
        renderer=renderer,
        render_camera=str(camera_names[0]) if len(camera_names) > 0 else None,
        camera_names=tuple(camera_names),
        img_height=int(img_height),
        img_width=int(img_width),
        reward_shaping=bool(reward_shaping),
        control_freq=int(control_freq),
        has_renderer=bool(has_renderer),
        has_offscreen_renderer=bool(len(camera_names) > 0),
        ignore_done=False,
        use_camera_obs=False,
        use_object_obs=bool(use_object_obs),
        proprio_keys=tuple(proprio_keys),
    )


class RobosuiteObservationAdapter:
    def __init__(
        self,
        env,
        *,
        render_env=None,
        camera_names: Sequence[str],
        img_height: int,
        img_width: int,
        proprio_keys: Optional[Sequence[str]] = None,
        image_obs_fps: float | None = None,
    ) -> None:
        self.env = env
        self.render_env = render_env if render_env is not None else env
        self.camera_names = [str(name) for name in camera_names]
        self.img_height = int(img_height)
        self.img_width = int(img_width)
        self.proprio_keys = None if proprio_keys is None else [str(key) for key in proprio_keys]
        self.image_obs_fps = None if image_obs_fps is None else float(image_obs_fps)
        self.image_obs_period = None
        if self.image_obs_fps is not None and self.image_obs_fps > 0.0:
            self.image_obs_period = 1.0 / self.image_obs_fps
        self._cached_images: Optional[dict[str, np.ndarray]] = None
        self._last_image_time: float | None = None

    def reset(self):
        reset_output = self.env.reset()
        if isinstance(reset_output, tuple):
            raw_obs, info = reset_output
        else:
            raw_obs, info = reset_output, {}
        self._cached_images = None
        self._last_image_time = None
        return self.transform(raw_obs, force_render=True), info

    def transform(
        self,
        raw_obs: dict[str, Any],
        images: Optional[dict[str, np.ndarray]] = None,
        *,
        force_render: bool = False,
        now: float | None = None,
    ) -> dict[str, np.ndarray]:
        if images is None:
            images = self.render_images(force_render=force_render, now=now)
        policy_obs = {key: np.asarray(image, dtype=np.uint8) for key, image in images.items()}
        if self.proprio_keys is None or self.proprio_keys:
            policy_obs["state"] = self.flatten_proprio(raw_obs)
        return policy_obs

    def flatten_proprio(self, raw_obs: dict[str, Any]) -> np.ndarray:
        values = []
        keys = (
            self.proprio_keys
            if self.proprio_keys is not None
            else [key for key in raw_obs if _is_numeric_observation(raw_obs[key])]
        )
        for key in keys:
            if key not in raw_obs:
                continue
            array = np.asarray(raw_obs[key], dtype=np.float32)
            values.append(array.reshape(-1))
        if not values:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(values, axis=0).astype(np.float32)

    def render_images(
        self,
        *,
        force_render: bool = False,
        now: float | None = None,
    ) -> dict[str, np.ndarray]:
        current_time = time.monotonic() if now is None else float(now)
        if not force_render and self._should_reuse_cached_images(current_time):
            assert self._cached_images is not None
            return self._cached_images
        self._sync_render_env()
        images = {}
        for camera_name in self.camera_names:
            frame = self.render_env.sim.render(
                height=self.img_height,
                width=self.img_width,
                camera_name=camera_name,
            )
            images[camera_name] = np.asarray(frame, dtype=np.uint8)
        self._cached_images = images
        self._last_image_time = current_time
        return images

    def action_spec(self) -> tuple[np.ndarray, np.ndarray]:
        low, high = self.env.action_spec
        return np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32)

    def _sync_render_env(self) -> None:
        if self.render_env is self.env:
            return
        apply_snapshot_to_env(self.render_env, snapshot_env_state(self.env))

    def _should_reuse_cached_images(self, now: float) -> bool:
        if self._cached_images is None or self.image_obs_period is None:
            return False
        if self._last_image_time is None:
            return False
        return (now - self._last_image_time) < self.image_obs_period


def estimate_gripper_openness(env) -> float | None:
    openness_values: list[float] = []
    for robot in getattr(env, "robots", []):
        for arm in robot.arms:
            if not robot.has_gripper.get(arm, False):
                continue
            qpos_indexes = robot._ref_gripper_joint_pos_indexes.get(arm)
            joint_ids = robot._ref_joints_indexes_dict.get(robot.get_gripper_name(arm))
            if not qpos_indexes or not joint_ids:
                continue

            joint_qpos = np.asarray([env.sim.data.qpos[index] for index in qpos_indexes], dtype=np.float32)
            joint_ranges = np.asarray(
                [env.sim.model.jnt_range[joint_id] for joint_id in joint_ids],
                dtype=np.float32,
            )
            if joint_qpos.shape[0] != joint_ranges.shape[0]:
                continue

            # Panda finger joints move symmetrically away from zero as the gripper opens.
            magnitude_ranges = np.sort(np.abs(joint_ranges), axis=1)
            span = magnitude_ranges[:, 1] - magnitude_ranges[:, 0]
            valid = span > 1e-6
            if not np.any(valid):
                continue

            normalized = np.zeros_like(joint_qpos, dtype=np.float32)
            normalized[valid] = np.clip(
                (np.abs(joint_qpos[valid]) - magnitude_ranges[valid, 0]) / span[valid],
                0.0,
                1.0,
            )
            openness_values.append(float(normalized[valid].mean()))

    if not openness_values:
        return None
    return float(np.mean(openness_values))


def compute_grasp_penalty(
    env,
    action: np.ndarray,
    *,
    penalty: float = -0.02,
    command_threshold: float = 0.5,
    open_threshold: float = 0.9,
    closed_threshold: float = 0.1,
) -> float | None:
    action_array = np.asarray(action, dtype=np.float32).reshape(-1)
    if action_array.size == 0:
        return None

    openness = estimate_gripper_openness(env)
    if openness is None:
        return None

    gripper_action = float(action_array[-1])
    if gripper_action >= float(command_threshold) and openness <= float(closed_threshold):
        return float(penalty)
    if gripper_action <= -float(command_threshold) and openness >= float(open_threshold):
        return float(penalty)
    return 0.0


def sparse_success_reward(env, info: Optional[dict[str, Any]] = None) -> tuple[float, bool]:
    """Return the HIL-SERL sparse success reward."""
    success = False
    if isinstance(info, dict):
        if "is_success" in info:
            success = bool(info["is_success"])
        elif "success" in info:
            success = bool(info["success"])
    if not success and hasattr(env, "_check_success"):
        success = bool(env._check_success())
    return (1.0 if success else 0.0), success


def unpack_robosuite_step(step_output):
    """Normalize Gym and Gymnasium outputs while treating legacy `done` as a time limit."""
    if len(step_output) == 5:
        observation, reward, terminated, truncated, info = step_output
        return observation, reward, bool(terminated), bool(truncated), info
    if len(step_output) == 4:
        observation, reward, done, info = step_output
        return observation, reward, False, bool(done), info
    raise ValueError(f"Unexpected robosuite step output length: {len(step_output)}")


def snapshot_env_state(env) -> RobosuiteViewerSnapshot:
    return RobosuiteViewerSnapshot(
        state=np.asarray(env.sim.get_state().flatten(), dtype=np.float64).copy(),
        timestep=int(getattr(env, "timestep", 0)),
        cur_time=float(getattr(env, "cur_time", 0.0)),
        done=bool(getattr(env, "done", False)),
    )


def apply_snapshot_to_env(env, snapshot: RobosuiteViewerSnapshot) -> None:
    env.sim.set_state_from_flattened(snapshot.state)
    env.sim.forward()
    env.done = bool(snapshot.done)
    env.timestep = int(snapshot.timestep)
    env.cur_time = float(snapshot.cur_time)


def resolve_demo_images(
    demo_images: dict[str, np.ndarray],
    adapter: RobosuiteObservationAdapter,
    step_idx: int,
) -> dict[str, np.ndarray]:
    current_images = {}
    fallback_images = None
    for camera_name in adapter.camera_names:
        images = demo_images.get(camera_name)
        if images is not None and images.ndim == 4 and step_idx < images.shape[0]:
            current_images[camera_name] = resize_demo_image(
                np.asarray(images[step_idx], dtype=np.uint8),
                height=adapter.img_height,
                width=adapter.img_width,
            )
        else:
            if fallback_images is None:
                fallback_images = adapter.render_images()
            current_images[camera_name] = fallback_images[camera_name]
    return current_images


def reset_env_from_demo_xml(env, model_xml: str) -> None:
    xml = env.edit_model_xml(model_xml)
    env.reset_from_xml_string(xml)
    env.sim.reset()
    env.sim.forward()
    env.done = False
    env.timestep = 0
    env.cur_time = 0.0


def resize_demo_image(image: np.ndarray, *, height: int, width: int) -> np.ndarray:
    if image.ndim != 3:
        return image
    if image.shape[0] == height and image.shape[1] == width:
        return image
    resized = Image.fromarray(image).resize((width, height), resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def _is_numeric_observation(value: Any) -> bool:
    array = np.asarray(value)
    return array.dtype.kind in {"b", "i", "u", "f"} and array.ndim >= 1
