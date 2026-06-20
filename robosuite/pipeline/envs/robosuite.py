from __future__ import annotations

import threading
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.controllers.composite.composite_controller import WholeBody

from ..common.types import Transition


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
    proprio_keys: Sequence[str] = ()
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


def choose_viewer_backend(renderer: str, camera_names: Sequence[str], requested_backend: str = "auto") -> str:
    normalized_renderer = str(renderer).lower()
    normalized_requested = str(requested_backend).lower()
    valid_backends = {"auto", "opencv", "mjviewer"}
    if normalized_requested not in valid_backends:
        raise ValueError(f"Unsupported viewer backend '{requested_backend}'. Expected one of {sorted(valid_backends)}.")
    if normalized_requested == "auto":
        if normalized_renderer == "mjviewer" and len(camera_names) > 0:
            return "opencv"
        return normalized_renderer
    return normalized_requested


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
        proprio_keys: Sequence[str] = (),
        image_obs_fps: float | None = None,
    ) -> None:
        self.env = env
        self.render_env = render_env if render_env is not None else env
        self.camera_names = [str(name) for name in camera_names]
        self.img_height = int(img_height)
        self.img_width = int(img_width)
        self.proprio_keys = [str(key) for key in proprio_keys]
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
        policy_obs = {
            key: np.asarray(image, dtype=np.uint8)
            for key, image in images.items()
        }
        policy_obs["state"] = self.flatten_proprio(raw_obs)
        return policy_obs

    def flatten_proprio(self, raw_obs: dict[str, Any]) -> np.ndarray:
        values = []
        keys = self.proprio_keys or [key for key in raw_obs.keys() if _is_numeric_observation(raw_obs[key])]
        for key in keys:
            if key not in raw_obs:
                continue
            array = np.asarray(raw_obs[key], dtype=np.float32)
            values.append(array.reshape(-1))
        if not values:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(values, axis=0).astype(np.float32)

    def render_images(self, *, force_render: bool = False, now: float | None = None) -> dict[str, np.ndarray]:
        current_time = time.monotonic() if now is None else float(now)
        if (not force_render) and self._should_reuse_cached_images(current_time):
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
            images[str(camera_name)] = np.asarray(frame, dtype=np.uint8)
        self._cached_images = images
        self._last_image_time = current_time
        return images

    def action_spec(self) -> tuple[np.ndarray, np.ndarray]:
        low, high = self.env.action_spec
        return np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32)

    def _sync_render_env(self) -> None:
        if self.render_env is self.env:
            return
        snapshot = snapshot_env_state(self.env)
        apply_snapshot_to_env(self.render_env, snapshot)

    def _should_reuse_cached_images(self, now: float) -> bool:
        if self._cached_images is None or self.image_obs_period is None:
            return False
        if self._last_image_time is None:
            return False
        return (now - self._last_image_time) < self.image_obs_period


class RobosuiteViewerRuntime:
    def __init__(
        self,
        viewer_env,
        *,
        render_fps: float = 10.0,
        async_mode: bool = True,
        preview_camera: str | None = None,
        backend: str = "mjviewer",
    ) -> None:
        self.viewer_env = viewer_env
        self.render_fps = max(1.0, float(render_fps))
        self.async_mode = bool(async_mode)
        self.backend = str(backend)
        self.preview_camera = None if preview_camera is None else str(preview_camera)
        self._latest_snapshot: Optional[RobosuiteViewerSnapshot] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_render_time = 0.0
        self._render_count = 0
        self._render_failed = False
        if self.backend == "opencv":
            self.async_mode = False

    def start(self) -> None:
        if self.backend != "mjviewer" or not self.async_mode or self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._render_loop, name="hil_serl_gui", daemon=True)
        self._thread.start()

    def publish_from_env(self, env) -> None:
        self.publish(snapshot_env_state(env))

    def publish(self, snapshot: RobosuiteViewerSnapshot) -> None:
        with self._lock:
            self._latest_snapshot = snapshot

    def render_if_due(self) -> bool:
        if self.async_mode or self._render_failed:
            return False
        return self._maybe_render_sync()

    def consume_render_count(self) -> int:
        with self._lock:
            render_count = int(self._render_count)
            self._render_count = 0
        return render_count

    def close(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self.viewer_env.close()

    def reset_preview(self, snapshot: RobosuiteViewerSnapshot, warmup_frames: int = 0) -> None:
        self.publish(snapshot)
        if self.backend != "opencv" or self._render_failed:
            return
        self.viewer_env.reset()
        frame_count = max(1, int(warmup_frames))
        for _ in range(frame_count):
            self._render_snapshot(snapshot)

    def _render_loop(self) -> None:
        period = 1.0 / self.render_fps
        while not self._stop_event.is_set():
            started = time.monotonic()
            snapshot = None
            with self._lock:
                snapshot = self._latest_snapshot
            if snapshot is not None and not self._render_failed:
                self._render_snapshot(snapshot)
            elapsed = time.monotonic() - started
            sleep_time = max(0.0, period - elapsed)
            self._stop_event.wait(sleep_time)

    def _maybe_render_sync(self) -> bool:
        if self._render_failed:
            return False
        now = time.monotonic()
        if (now - self._last_render_time) < (1.0 / self.render_fps):
            return False
        with self._lock:
            snapshot = self._latest_snapshot
        if snapshot is None:
            return False
        self._render_snapshot(snapshot)
        self._last_render_time = now
        return True

    def _render_snapshot(self, snapshot: RobosuiteViewerSnapshot) -> None:
        if self._render_failed:
            return
        apply_snapshot_to_env(self.viewer_env, snapshot)
        try:
            render_viewer_env(self.viewer_env)
        except Exception as exc:
            if self.backend == "opencv":
                self._disable_opencv_preview(exc)
                return
            raise
        with self._lock:
            self._render_count += 1

    def _disable_opencv_preview(self, exc: Exception) -> None:
        if self._render_failed:
            return
        self._render_failed = True
        try:
            if getattr(self.viewer_env, "viewer", None) is not None:
                self.viewer_env.close_renderer()
        except Exception:
            pass
        print(f"[WARN] Disabling OpenCV preview after render failure: {exc}")


class RobosuiteInterventionRuntime:
    def __init__(self, env, device, goal_update_mode: str = "target") -> None:
        self.env = env
        self.device = device
        self.goal_update_mode = str(goal_update_mode)
        self.all_prev_gripper_actions = []
        self._episode_active = False
        self._last_grasp_states: list[list[bool]] = []

    def start_episode(self) -> None:
        self.device.start_control()
        self.all_prev_gripper_actions = [
            {
                f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
                for robot_arm in robot.arms
                if robot.gripper[robot_arm].dof > 0
            }
            for robot in self.env.robots
        ]
        self._last_grasp_states = _snapshot_device_grasp_states(self.device)
        self._episode_active = True

    def stop_episode(self) -> None:
        self._episode_active = False

    def maybe_override_action(self, policy_action: np.ndarray) -> tuple[np.ndarray | None, bool, bool]:
        if not self._episode_active:
            self.start_episode()

        input_ac_dict = self.device.input2action(goal_update_mode=self.goal_update_mode)
        if input_ac_dict is None:
            return None, False, True

        grasp_state_changed = self._consume_grasp_state_change()
        if not (check_intervention(self.device, input_ac_dict) or grasp_state_changed):
            return np.asarray(policy_action, dtype=np.float32), False, False

        active_robot = self.env.robots[self.device.active_robot]
        action_dict = deepcopy(input_ac_dict)
        for arm_name in active_robot.arms:
            if isinstance(active_robot.composite_controller, WholeBody):
                controller_input_type = active_robot.composite_controller.joint_action_policy.input_type
            else:
                controller_input_type = active_robot.part_controllers[arm_name].input_type

            if controller_input_type == "delta":
                action_dict[arm_name] = input_ac_dict[f"{arm_name}_delta"]
            elif controller_input_type == "absolute":
                action_dict[arm_name] = input_ac_dict[f"{arm_name}_abs"]
            else:
                raise ValueError(f"Unsupported controller input type: {controller_input_type}")

        env_action_list = [
            robot.create_action_vector(self.all_prev_gripper_actions[idx])
            for idx, robot in enumerate(self.env.robots)
        ]
        env_action_list[self.device.active_robot] = active_robot.create_action_vector(action_dict)
        env_action = np.concatenate(env_action_list).astype(np.float32)

        for gripper_action in self.all_prev_gripper_actions[self.device.active_robot]:
            self.all_prev_gripper_actions[self.device.active_robot][gripper_action] = action_dict[gripper_action]

        return env_action, True, False

    def close(self) -> None:
        try:
            if hasattr(self.device, "stop_control"):
                self.device.stop_control()
        except Exception:
            pass

    def _consume_grasp_state_change(self) -> bool:
        current_grasp_states = _snapshot_device_grasp_states(self.device)
        changed = current_grasp_states != self._last_grasp_states
        self._last_grasp_states = current_grasp_states
        return changed


def check_intervention(device, input_ac_dict) -> bool:
    if input_ac_dict is None:
        return True
    control_gripper = getattr(device, "control_gripper", None)
    if control_gripper is not None and abs(float(control_gripper)) > 1e-3:
        return True
    threshold = 0.1
    total_mag = 0.0
    for key, value in input_ac_dict.items():
        if not isinstance(value, np.ndarray):
            continue
        if "delta" in key:
            total_mag += np.linalg.norm(value)
    return total_mag > threshold


def _snapshot_device_grasp_states(device) -> list[list[bool]]:
    grasp_states = getattr(device, "grasp_states", [])
    return [[bool(value) for value in robot_states] for robot_states in grasp_states]


def build_device(env, device_cfg):
    device_type = str(getattr(device_cfg, "device", None) or device_cfg["device"])
    if device_type == "keyboard":
        from robosuite.devices import Keyboard

        return Keyboard(
            env=env,
            pos_sensitivity=float(getattr(device_cfg, "pos_sensitivity", 1.0)),
            rot_sensitivity=float(getattr(device_cfg, "rot_sensitivity", 1.0)),
        )
    if device_type == "spacemouse":
        from robosuite.devices import SpaceMouse

        return SpaceMouse(
            env=env,
            vendor_id=0x256F,
            product_id=0xC635,
            pos_sensitivity=float(getattr(device_cfg, "pos_sensitivity", 1.0)),
            rot_sensitivity=float(getattr(device_cfg, "rot_sensitivity", 1.0)),
        )
    if device_type == "dualsense":
        from robosuite.devices import DualSense

        return DualSense(
            env=env,
            pos_sensitivity=float(getattr(device_cfg, "pos_sensitivity", 1.0)),
            rot_sensitivity=float(getattr(device_cfg, "rot_sensitivity", 1.0)),
            reverse_xy=bool(getattr(device_cfg, "reverse_xy", False)),
        )
    if device_type == "mjgui":
        assert str(getattr(device_cfg, "renderer", "mjviewer")) == "mjviewer"
        from robosuite.devices.mjgui import MJGUI

        return MJGUI(env=env)
    raise ValueError(f"Invalid device choice: {device_type}")


def sparse_success_reward(env, info: Optional[dict[str, Any]] = None) -> tuple[float, bool]:
    """NOTE: here we use -1/0 reward"""
    success = False
    if isinstance(info, dict) and "success" in info:
        success = bool(info["success"])
    if not success and hasattr(env, "_check_success"):
        success = bool(env._check_success())
    return (0.0 if success else -1.0), success


def make_checkpoint_directory(root_dir: str | Path, run_name: str) -> Path:
    checkpoint_dir = Path(root_dir) / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "checkpoints").mkdir(exist_ok=True)
    (checkpoint_dir / "buffer_cache").mkdir(exist_ok=True)
    (checkpoint_dir / "demo_cache").mkdir(exist_ok=True)
    return checkpoint_dir


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


def render_viewer_env(env) -> None:
    renderer = str(getattr(env, "renderer", "")).lower()
    if renderer == "mjviewer":
        if env.viewer is None:
            env.initialize_renderer()
            env.viewer_get_obs = hasattr(env.viewer, "_get_observations")
        env.viewer.update()
        return
    env.render()



def _is_numeric_observation(value: Any) -> bool:
    array = np.asarray(value)
    return array.dtype.kind in {"b", "i", "u", "f"} and array.ndim >= 1
