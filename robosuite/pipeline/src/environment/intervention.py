from __future__ import annotations

from copy import deepcopy

import numpy as np
from robosuite.controllers.composite.composite_controller import WholeBody


class InterventionRuntime:
    def __init__(self, env, device, *, goal_update_mode: str = "target") -> None:
        self.env = env
        self.device = device
        self.goal_update_mode = str(goal_update_mode)
        self.previous_gripper_actions: list[dict[str, np.ndarray]] = []
        self.previous_grasp_states: list[list[bool]] = []
        self.episode_active = False

    def start_episode(self) -> None:
        self.device.start_control()
        self.previous_gripper_actions = [
            {
                f"{arm}_gripper": np.zeros(robot.gripper[arm].dof)
                for arm in robot.arms
                if robot.gripper[arm].dof > 0
            }
            for robot in self.env.robots
        ]
        self.previous_grasp_states = _grasp_states(self.device)
        self.episode_active = True

    def stop_episode(self) -> None:
        self.episode_active = False

    def override(self, policy_action: np.ndarray) -> tuple[np.ndarray | None, bool, bool]:
        if not self.episode_active:
            self.start_episode()
        device_action = self.device.input2action(goal_update_mode=self.goal_update_mode)
        if device_action is None:
            return None, False, True

        grasp_states = _grasp_states(self.device)
        grasp_changed = grasp_states != self.previous_grasp_states
        self.previous_grasp_states = grasp_states
        if not (_has_intervention(self.device, device_action) or grasp_changed):
            return np.asarray(policy_action, dtype=np.float32), False, False

        active_index = self.device.active_robot
        active_robot = self.env.robots[active_index]
        action_dict = deepcopy(device_action)
        for arm in active_robot.arms:
            controller = active_robot.composite_controller
            input_type = (
                controller.joint_action_policy.input_type
                if isinstance(controller, WholeBody)
                else active_robot.part_controllers[arm].input_type
            )
            if input_type not in {"delta", "absolute"}:
                raise ValueError(f"Unsupported controller input type: {input_type}")
            suffix = "delta" if input_type == "delta" else "abs"
            action_dict[arm] = device_action[f"{arm}_{suffix}"]

        actions = [
            robot.create_action_vector(self.previous_gripper_actions[index])
            for index, robot in enumerate(self.env.robots)
        ]
        actions[active_index] = active_robot.create_action_vector(action_dict)
        for key in self.previous_gripper_actions[active_index]:
            self.previous_gripper_actions[active_index][key] = action_dict[key]
        return np.concatenate(actions).astype(np.float32), True, False

    def close(self) -> None:
        if hasattr(self.device, "stop_control"):
            self.device.stop_control()


def build_spacemouse(env, *, pos_sensitivity: float = 1.0, rot_sensitivity: float = 1.0):
    from robosuite.devices import SpaceMouse

    return SpaceMouse(
        env=env,
        vendor_id=0x256F,
        product_id=0xC635,
        pos_sensitivity=float(pos_sensitivity),
        rot_sensitivity=float(rot_sensitivity),
    )


def sparse_success_reward(env, info: dict | None = None) -> tuple[float, bool]:
    success = bool(info.get("success", False)) if isinstance(info, dict) else False
    if not success and hasattr(env, "_check_success"):
        success = bool(env._check_success())
    return (0.0 if success else -1.0), success


def compute_grasp_penalty(
    env,
    action: np.ndarray,
    *,
    penalty: float = -0.02,
    command_threshold: float = 0.5,
    open_threshold: float = 0.9,
    closed_threshold: float = 0.1,
) -> float | None:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    openness = _gripper_openness(env)
    if action.size == 0 or openness is None:
        return None
    command = float(action[-1])
    if command >= command_threshold and openness <= closed_threshold:
        return float(penalty)
    if command <= -command_threshold and openness >= open_threshold:
        return float(penalty)
    return 0.0


def _has_intervention(device, action: dict) -> bool:
    gripper = getattr(device, "control_gripper", None)
    if gripper is not None and abs(float(gripper)) > 1e-3:
        return True
    return sum(
        np.linalg.norm(value)
        for key, value in action.items()
        if isinstance(value, np.ndarray) and "delta" in key
    ) > 0.1


def _grasp_states(device) -> list[list[bool]]:
    return [[bool(value) for value in states] for states in getattr(device, "grasp_states", [])]


def _gripper_openness(env) -> float | None:
    values: list[float] = []
    for robot in getattr(env, "robots", []):
        for arm in robot.arms:
            if not robot.has_gripper.get(arm, False):
                continue
            qpos_indices = robot._ref_gripper_joint_pos_indexes.get(arm)
            joint_ids = robot._ref_joints_indexes_dict.get(robot.get_gripper_name(arm))
            if not qpos_indices or not joint_ids:
                continue
            qpos = np.asarray([env.sim.data.qpos[index] for index in qpos_indices])
            ranges = np.asarray([env.sim.model.jnt_range[index] for index in joint_ids])
            spans = ranges[:, 1] - ranges[:, 0]
            valid = spans > 1e-6
            if qpos.shape[0] == ranges.shape[0] and np.any(valid):
                normalized = np.clip((qpos[valid] - ranges[valid, 0]) / spans[valid], 0.0, 1.0)
                values.append(float(normalized.mean()))
    return None if not values else float(np.mean(values))
