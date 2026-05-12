import numpy as np
import robosuite as suite


def get_joint_width(joint_type):
    """
    MuJoCo joint type convention:
    0: free joint, 1: ball joint, 2: slide joint, 3: hinge joint
    """
    if joint_type == 0:
        return 7, 6
    if joint_type == 1:
        return 4, 3
    if joint_type in [2, 3]:
        return 1, 1
    raise ValueError(f"Unknown joint type: {joint_type}")


def classify_joint_name(name, robot_prefixes=("robot0", "gripper0")):
    if name is None:
        return "unknown"

    for prefix in robot_prefixes:
        if name.startswith(prefix):
            return "robot"

    return "env"


def analyze_robosuite_state(env, robot_prefixes=("robot0", "gripper0"), print_values=False):
    sim = env.sim
    model = sim.model

    flat_state = sim.get_state().flatten()

    nq = model.nq
    nv = model.nv
    njnt = model.njnt

    time_idx = 0
    qpos_start = 1
    qpos_end = 1 + nq
    qvel_start = 1 + nq
    qvel_end = 1 + nq + nv

    print("=" * 100)
    print("Flattened simulator state layout")
    print("=" * 100)
    print(f"flat_state.shape: {flat_state.shape}")
    print(f"nq: {nq}")
    print(f"nv: {nv}")
    print(f"njnt: {njnt}")
    print()
    print(f"time: state[{time_idx}]")
    print(f"qpos: state[{qpos_start}:{qpos_end}]")
    print(f"qvel: state[{qvel_start}:{qvel_end}]")
    print()

    robot_qpos_indices = []
    robot_qvel_indices = []
    env_qpos_indices = []
    env_qvel_indices = []

    rows = []

    for joint_id in range(njnt):
        name = model.joint_id2name(joint_id)
        joint_type = int(model.jnt_type[joint_id])
        qpos_adr = int(model.jnt_qposadr[joint_id])
        qvel_adr = int(model.jnt_dofadr[joint_id])

        qpos_width, qvel_width = get_joint_width(joint_type)

        flat_qpos_start = qpos_start + qpos_adr
        flat_qpos_end = flat_qpos_start + qpos_width

        flat_qvel_start = qvel_start + qvel_adr
        flat_qvel_end = flat_qvel_start + qvel_width

        group = classify_joint_name(name, robot_prefixes=robot_prefixes)

        qpos_indices = list(range(flat_qpos_start, flat_qpos_end))
        qvel_indices = list(range(flat_qvel_start, flat_qvel_end))

        if group == "robot":
            robot_qpos_indices.extend(qpos_indices)
            robot_qvel_indices.extend(qvel_indices)
        elif group == "env":
            env_qpos_indices.extend(qpos_indices)
            env_qvel_indices.extend(qvel_indices)

        rows.append({
            "joint_id": joint_id,
            "name": name,
            "group": group,
            "type": joint_type,
            "qpos_adr": qpos_adr,
            "qvel_adr": qvel_adr,
            "qpos_width": qpos_width,
            "qvel_width": qvel_width,
            "flat_qpos_slice": f"state[{flat_qpos_start}:{flat_qpos_end}]",
            "flat_qvel_slice": f"state[{flat_qvel_start}:{flat_qvel_end}]",
        })

    print("=" * 100)
    print("Joint-level state mapping")
    print("=" * 100)

    header = (
        f"{'id':>3} | "
        f"{'group':>6} | "
        f"{'type':>4} | "
        f"{'qpos_adr':>8} | "
        f"{'qvel_adr':>8} | "
        f"{'qpos_w':>6} | "
        f"{'qvel_w':>6} | "
        f"{'flat qpos':>18} | "
        f"{'flat qvel':>18} | "
        f"name"
    )
    print(header)
    print("-" * len(header))

    for row in rows:
        print(
            f"{row['joint_id']:>3} | "
            f"{row['group']:>6} | "
            f"{row['type']:>4} | "
            f"{row['qpos_adr']:>8} | "
            f"{row['qvel_adr']:>8} | "
            f"{row['qpos_width']:>6} | "
            f"{row['qvel_width']:>6} | "
            f"{row['flat_qpos_slice']:>18} | "
            f"{row['flat_qvel_slice']:>18} | "
            f"{row['name']}"
        )

    robot_qpos_indices = np.array(sorted(robot_qpos_indices), dtype=np.int64)
    robot_qvel_indices = np.array(sorted(robot_qvel_indices), dtype=np.int64)
    env_qpos_indices = np.array(sorted(env_qpos_indices), dtype=np.int64)
    env_qvel_indices = np.array(sorted(env_qvel_indices), dtype=np.int64)

    robot_indices = np.concatenate([robot_qpos_indices, robot_qvel_indices])
    env_indices = np.concatenate([env_qpos_indices, env_qvel_indices])

    print()
    print("=" * 100)
    print("Index summary")
    print("=" * 100)
    print(f"robot_qpos_indices: {robot_qpos_indices.tolist()}")
    print(f"robot_qvel_indices: {robot_qvel_indices.tolist()}")
    print(f"robot_indices: {robot_indices.tolist()}")
    print()
    print(f"env_qpos_indices: {env_qpos_indices.tolist()}")
    print(f"env_qvel_indices: {env_qvel_indices.tolist()}")
    print(f"env_indices: {env_indices.tolist()}")

    if print_values:
        print()
        print("=" * 100)
        print("State values")
        print("=" * 100)
        print(f"time value: {flat_state[time_idx]}")
        print(f"robot qpos values: {flat_state[robot_qpos_indices]}")
        print(f"robot qvel values: {flat_state[robot_qvel_indices]}")
        print(f"env qpos values: {flat_state[env_qpos_indices]}")
        print(f"env qvel values: {flat_state[env_qvel_indices]}")

    return {
        "flat_state": flat_state,
        "nq": nq,
        "nv": nv,
        "time_index": time_idx,
        "qpos_slice": slice(qpos_start, qpos_end),
        "qvel_slice": slice(qvel_start, qvel_end),
        "robot_qpos_indices": robot_qpos_indices,
        "robot_qvel_indices": robot_qvel_indices,
        "robot_indices": robot_indices,
        "env_qpos_indices": env_qpos_indices,
        "env_qvel_indices": env_qvel_indices,
        "env_indices": env_indices,
        "joint_rows": rows,
    }

if __name__ == "__main__":
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )
    env.reset()

    info = analyze_robosuite_state(
        env,
        robot_prefixes=("robot0", "gripper0"),
        print_values=True,
    )
    state = env.sim.get_state().flatten()
    print(f"robot_state: {state[info['robot_indices']]}")
    print(f"robot_qpos: {state[info['robot_qpos_indices']]}")
    print(f"robot_qvel: {state[info['robot_qvel_indices']]}")