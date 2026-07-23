from __future__ import annotations

from copy import deepcopy

import numpy as np

from robosuite.controllers.composite.composite_controller import WholeBody


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
        if isinstance(value, np.ndarray) and "delta" in key:
            total_mag += np.linalg.norm(value)
    return total_mag > threshold


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


def _snapshot_device_grasp_states(device) -> list[list[bool]]:
    grasp_states = getattr(device, "grasp_states", [])
    return [[bool(value) for value in robot_states] for robot_states in grasp_states]
