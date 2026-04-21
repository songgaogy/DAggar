import json
from typing import Any

import mujoco
import numpy as np
import robosuite as suite


def decode_if_bytes(value: Any):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.decode("utf-8")
    return value


def parse_env_info(raw_env_info: Any) -> dict:
    raw_env_info = decode_if_bytes(raw_env_info)
    if isinstance(raw_env_info, str):
        return json.loads(raw_env_info)
    return dict(raw_env_info)


def camera_obs_key(camera_name: str) -> str:
    return f"{camera_name}_image"


class RobosuiteProprioExtractor:
    def __init__(
        self,
        env_kwargs: dict,
        has_renderer: bool = False,
        has_offscreen_renderer: bool = False,
        use_camera_obs: bool = False,
        camera_names: list[str] | None = None,
        reward_shaping: bool = False,
    ):
        env_kwargs = dict(env_kwargs)
        env_kwargs.update(
            dict(
                has_renderer=has_renderer,
                has_offscreen_renderer=has_offscreen_renderer,
                use_camera_obs=use_camera_obs,
                camera_names=camera_names,
                reward_shaping=reward_shaping,
            )
        )
        self.env = suite.make(**env_kwargs)
        self.sim = self.env.sim
        self._build_robot_joint_indices()

    def _joint_dims(self, joint_type: int):
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            return 7, 6
        if joint_type == mujoco.mjtJoint.mjJNT_BALL:
            return 4, 3
        if joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
            return 1, 1
        if joint_type == mujoco.mjtJoint.mjJNT_HINGE:
            return 1, 1
        raise RuntimeError(f"Unknown joint type: {joint_type}")

    def _build_robot_joint_indices(self):
        model = self.sim.model
        joint_names = list(model.joint_names)
        robot_joints = [name for name in joint_names if name.startswith("robot0_")]
        if len(robot_joints) == 0:
            raise RuntimeError("No robot0_ joints found in the environment.")

        qpos_indices = []
        qvel_indices = []
        for joint_name in robot_joints:
            joint_id = model.joint_name2id(joint_name)
            qpos_start = int(model.jnt_qposadr[joint_id])
            qvel_start = int(model.jnt_dofadr[joint_id])
            n_qpos, n_qvel = self._joint_dims(int(model.jnt_type[joint_id]))
            qpos_indices.extend(range(qpos_start, qpos_start + n_qpos))
            qvel_indices.extend(range(qvel_start, qvel_start + n_qvel))

        self.qpos_indices = np.asarray(qpos_indices, dtype=np.int64)
        self.qvel_indices = np.asarray(qvel_indices, dtype=np.int64)

    def extract(self, flattened_state: np.ndarray) -> np.ndarray:
        flattened_state = np.asarray(flattened_state).reshape(-1)
        nq = int(self.sim.model.nq)
        nv = int(self.sim.model.nv)
        na = int(getattr(self.sim.model, "na", 0))

        core0 = nq + nv
        core1 = nq + nv + na
        size = int(flattened_state.shape[0])

        if size == core0:
            base = flattened_state
        elif size == core1:
            base = flattened_state[:core0]
        elif size == core0 + 1:
            base = flattened_state[1 : core0 + 1]
        elif size == core1 + 1:
            base = flattened_state[1 : core0 + 1]
        else:
            raise ValueError(f"Unexpected flattened state length: {size}")

        qpos = base[:nq]
        qvel = base[nq : nq + nv]
        robot_qpos = qpos[self.qpos_indices]
        robot_qvel = qvel[self.qvel_indices]
        return np.concatenate([robot_qpos, robot_qvel], axis=0).astype(np.float32)

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass
