import numpy as np
import robosuite as suite
import mujoco


class PandaLiftProprioExtractor:
    def __init__(
        self,
        robots="Panda",
        env_name="Lift",
        controller_config=None,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        camera_names=None,
        reward_shaping=False,
    ):
        env_kwargs = dict(
            env_name=env_name,
            robots=robots,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            use_camera_obs=use_camera_obs,
            camera_names=camera_names,
            reward_shaping=reward_shaping,
        )
        if controller_config is not None:
            env_kwargs["controller_configs"] = controller_config

        self.env = suite.make(**env_kwargs)
        self.sim = self.env.sim
        print("nq nv na:", int(self.sim.model.nq), int(self.sim.model.nv), int(getattr(self.sim.model, "na", 0)))

        self._build_robot_joint_indices()

    def _joint_dims(self, j_type: int):
        if j_type == mujoco.mjtJoint.mjJNT_FREE:
            return 7, 6
        if j_type == mujoco.mjtJoint.mjJNT_BALL:
            return 4, 3
        if j_type == mujoco.mjtJoint.mjJNT_SLIDE:
            return 1, 1
        if j_type == mujoco.mjtJoint.mjJNT_HINGE:
            return 1, 1
        raise RuntimeError(f"Unknown joint type: {j_type}")

    def _build_robot_joint_indices(self):
        model = self.sim.model
        joint_names = list(model.joint_names)

        robot_joints = [jn for jn in joint_names if jn.startswith("robot0_")]
        if len(robot_joints) == 0:
            raise RuntimeError("No robot0_ joints found. Check robot naming in XML.")

        qpos_inds = []
        qvel_inds = []

        for jn in robot_joints:
            j_id = model.joint_name2id(jn)

            qpos_adr = int(model.jnt_qposadr[j_id])
            dof_adr = int(model.jnt_dofadr[j_id])
            j_type = int(model.jnt_type[j_id])

            n_qpos, n_dof = self._joint_dims(j_type)

            qpos_inds.extend(range(qpos_adr, qpos_adr + n_qpos))
            qvel_inds.extend(range(dof_adr, dof_adr + n_dof))

        self.qpos_inds = np.array(qpos_inds, dtype=np.int64)
        self.qvel_inds = np.array(qvel_inds, dtype=np.int64)
        self.nq = int(model.nq)
        self.nv = int(model.nv)

    def extract(self, flattened_state):
        flattened_state = np.asarray(flattened_state).reshape(-1)

        nq = int(self.sim.model.nq)
        nv = int(self.sim.model.nv)
        na = int(getattr(self.sim.model, "na", 0))

        core0 = nq + nv
        core1 = nq + nv + na

        n = int(flattened_state.shape[0])

        if n == core0:
            base = flattened_state
        elif n == core1:
            base = flattened_state[:core0]
        elif n == 1 + core0:
            base = flattened_state[1:1 + core0]
        elif n == 1 + core1:
            base = flattened_state[1:1 + core0]
        else:
            raise ValueError(
                f"Unexpected flattened state length {n} != {core0}, {core1}, {1+core0}, or {1+core1}"
            )

        qpos = base[:nq]
        qvel = base[nq:nq + nv]

        robot_qpos = qpos[self.qpos_inds]
        robot_qvel = qvel[self.qvel_inds]
        proprio = np.concatenate([robot_qpos, robot_qvel], axis=0).astype(np.float32)
        return proprio

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass