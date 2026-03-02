"""
A script to collect a batch of human interventions over a base policy.
"""

import argparse
import datetime
import json
import os
import shutil
import time
from glob import glob

import h5py
import numpy as np

import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.controllers.composite.composite_controller import WholeBody
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper


class DummyPolicy:
    def __init__(self, action_dim):
        self.action_dim = action_dim

    def get_action(self, obs):
        return np.random.normal(0, 0.02, size=self.action_dim)


def check_intervention(device, input_ac_dict):
    if input_ac_dict is None:
        return True 
    
    threshold = 0.1
    total_mag = 0.0
    for key, value in input_ac_dict.items():
        if ("delta" in key) and isinstance(value, np.ndarray):
            total_mag += np.linalg.norm(value)

    return total_mag > threshold


def now_readable(ts: datetime.datetime | None = None) -> str:
    if ts is None:
        ts = datetime.datetime.now()
    return ts.strftime("%Y-%m-%d_%H-%M-%S")


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def list_ep_dirs(tmp_root: str) -> list[str]:
    if not os.path.isdir(tmp_root):
        return []
    ep_dirs = [os.path.join(tmp_root, d) for d in os.listdir(tmp_root) if d.startswith("ep_")]
    ep_dirs = [d for d in ep_dirs if os.path.isdir(d)]
    ep_dirs.sort(key=lambda p: os.path.getmtime(p))
    return ep_dirs


def get_current_ep_dir(env_wrapped, tmp_root: str) -> str | None:
    ep_dir = getattr(env_wrapped, "ep_directory", None)
    if isinstance(ep_dir, str) and len(ep_dir) > 0:
        return ep_dir
    eps = list_ep_dirs(tmp_root)
    return eps[-1] if len(eps) > 0 else None


def read_single_demo_from_ep(ep_dir: str):
    state_paths = os.path.join(ep_dir, "state_*.npz")
    states = []
    actions = []
    success = False
    env_name = None

    for state_file in sorted(glob(state_paths)):
        dic = np.load(state_file, allow_pickle=True)
        env_name = str(dic["env"])

        states.extend(dic["states"])
        for ai in dic["action_infos"]:
            actions.append(ai["actions"])
        success = success or dic["successful"]

    if len(states) == 0:
        return None

    # Delete the last state to align with action length.
    del states[-1]

    states = np.array(states)
    actions = np.array(actions)

    return {
        "env_name": env_name,
        "states": states,
        "actions": actions,
        "success": bool(success),
        "ep_dir": ep_dir,
    }


def append_demo_to_hdf5(
    hdf5_path: str,
    demo_id: int,
    demo_payload: dict,
    step_labels: list[str],
    env_info: str,
    camera_names: list[str],
    images_dict: dict,
):
    f = h5py.File(hdf5_path, "a")

    if "demos" not in f:
        demos_grp = f.create_group("demos")
        f.attrs["created_at"] = now_readable()
        f.attrs["repository_version"] = suite.__version__
        f.attrs["env"] = demo_payload.get("env_name", "")
        f.attrs["env_info"] = env_info
        f.attrs["camera_names"] = json.dumps(camera_names)
    else:
        demos_grp = f["demos"]

    demo_grp = demos_grp.create_group(f"demo_{demo_id:06d}")

    # Store model xml
    xml_path = os.path.join(demo_payload["ep_dir"], "model.xml")
    if os.path.isfile(xml_path):
        with open(xml_path, "r") as fx:
            demo_grp.attrs["model_file"] = fx.read()

    states = demo_payload["states"]
    actions = demo_payload["actions"]

    # Align lengths across states/actions/images/labels
    T = min(states.shape[0], actions.shape[0], len(step_labels))
    for cam in camera_names:
        imgs = images_dict.get(cam, None)
        if isinstance(imgs, np.ndarray) and imgs.ndim == 4:
            T = min(T, imgs.shape[0])

    # Resolve fake cooldowns
    resolved_labels = list(step_labels[:T])
    n_labels = len(resolved_labels)
    
    for i in range(n_labels):
        if resolved_labels[i] == "COOLDOWN":
            # Look ahead to see what this cooldown resolves to
            resolves_to = "END"
            for j in range(i + 1, n_labels):
                if resolved_labels[j] != "COOLDOWN":
                    resolves_to = resolved_labels[j]
                    break
            
            # If it resolves to anything other than POLICY (e.g., back to INTERVENING or hits END), 
            # it is a fake cooldown. Keep it as INTERVENING.
            if resolves_to != "POLICY":
                resolved_labels[i] = "INTERVENING"

    # Filter out true COOLDOWN data that transitioned to POLICY
    keep_indices = [i for i in range(T) if resolved_labels[i] != "COOLDOWN"]
    
    # Generate aligned states, actions, and binary labels (0: policy, 1: intervention)
    states_filtered = states[keep_indices]
    actions_filtered = actions[keep_indices]
    labels_binary = np.array([1 if resolved_labels[i] == "INTERVENING" else 0 for i in keep_indices], dtype=np.uint8)

    # Calculate intervention segment indices [start, end]
    segments = []
    in_segment = False
    start_idx = 0
    for i, val in enumerate(labels_binary):
        if val == 1 and not in_segment:
            start_idx = i
            in_segment = True
        elif val == 0 and in_segment:
            segments.append([start_idx, i - 1])
            in_segment = False
    if in_segment:
        segments.append([start_idx, len(labels_binary) - 1])

    T_final = len(keep_indices)
    demo_grp.attrs["length"] = int(T_final)
    demo_grp.attrs["successful"] = bool(demo_payload.get("success", False))
    demo_grp.attrs["intervention_segments"] = json.dumps(segments)

    demo_grp.create_dataset("states", data=states_filtered)
    demo_grp.create_dataset("actions", data=actions_filtered)
    demo_grp.create_dataset("intervention_labels", data=labels_binary)

    obs_grp = demo_grp.create_group("observations")

    for cam in camera_names:
        cam_grp = obs_grp.create_group(cam)
        imgs = images_dict.get(cam, None)
        if imgs is None or not isinstance(imgs, np.ndarray) or imgs.ndim != 4 or imgs.shape[0] == 0:
            cam_grp.create_dataset("images", data=np.zeros((0,), dtype=np.uint8))
            continue

        imgs_filtered = imgs[keep_indices].astype(np.uint8)

        cam_grp.create_dataset(
            "images",
            data=imgs_filtered,
            dtype=np.uint8,
            compression="gzip",
            compression_opts=4,
            chunks=True,
        )

    f.close()


def collect_intervention_trajectory(env, device, policy, arm, max_fr, goal_update_mode, camera_names, img_height, img_width, cool_down):
    env.render()
    task_completion_hold_count = -1  
    device.start_control()

    for robot in env.robots:
        robot.print_action_info_dict()

    all_prev_gripper_actions = [
        {
            f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
            for robot_arm in robot.arms
            if robot.gripper[robot_arm].dof > 0
        }
        for robot in env.robots
    ]

    images = {cam: [] for cam in camera_names}
    step_labels = []

    # State machine parameters
    state_mode = "POLICY"
    prev_state_mode = "POLICY"
    COOLDOWN_STEPS = int(20 * cool_down)  # assuming 20Hz control frequency
    cooldown_counter = 0

    obs = env.unwrapped._get_observations()
    print("\n[INFO] Loop started. Default: Policy. Intervene with Device.")

    while True:
        start = time.time()

        active_robot = env.robots[device.active_robot]
        input_ac_dict = device.input2action(goal_update_mode=goal_update_mode)

        if input_ac_dict is None:
            break

        is_intervening_now = check_intervention(device, input_ac_dict)

        # Update state machine
        if is_intervening_now:
            state_mode = "INTERVENING"
            cooldown_counter = 0
        else:
            if state_mode == "INTERVENING":
                state_mode = "COOLDOWN"
                cooldown_counter = COOLDOWN_STEPS
            
            if state_mode == "COOLDOWN":
                cooldown_counter -= 1
                if cooldown_counter <= 0:
                    state_mode = "POLICY"

        # Print state transitions to terminal
        if state_mode != prev_state_mode:
            if state_mode in ["INTERVENING", "POLICY"]:
                print(f"[State] {state_mode}")
            prev_state_mode = state_mode

        # Record the current frame's state
        step_labels.append(state_mode)

        if state_mode in ["INTERVENING", "COOLDOWN"]:
            # Human Control / Brake Buffer
            from copy import deepcopy
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
                    raise ValueError

            env_action_list = [robot.create_action_vector(all_prev_gripper_actions[i]) for i, robot in enumerate(env.robots)]
            env_action_list[device.active_robot] = active_robot.create_action_vector(action_dict)
            env_action = np.concatenate(env_action_list)
            
            for gripper_ac in all_prev_gripper_actions[device.active_robot]:
                all_prev_gripper_actions[device.active_robot][gripper_ac] = action_dict[gripper_ac]

        else:
            # Policy Control
            env_action = policy.get_action(obs)

        # Step environment
        obs, reward, done, info = env.step(env_action)

        # Capture multiview images
        for cam in camera_names:
            img = env.sim.render(height=img_height, width=img_width, camera_name=cam)
            images[cam].append(img)

        env.render()

        if task_completion_hold_count == 0:
            break

        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1 
            else:
                task_completion_hold_count = 10 
        else:
            task_completion_hold_count = -1 

        if max_fr is not None:
            elapsed = time.time() - start
            diff = 1 / max_fr - elapsed
            if diff > 0:
                time.sleep(diff)

    images_np = {}
    for cam in camera_names:
        if len(images[cam]) == 0:
            images_np[cam] = np.zeros((0,), dtype=np.uint8)
        else:
            images_np[cam] = np.stack(images[cam], axis=0).astype(np.uint8)

    return images_np, step_labels


def prompt_user_action() -> str:
    while True:
        ans = input("Action for this demo: [s]ave / [d]elete / [q]uit / [f]inish ? ").strip().lower()
        if ans in ("s", "d", "q", "f"):
            return ans
        print("Invalid input. Please enter s, d, f or q.")


def build_env(env_config: dict, args):
    env = suite.make(
        **env_config,
        has_renderer=True,
        renderer=args.renderer,
        has_offscreen_renderer=True,
        render_camera=args.camera[0] if len(args.camera) > 0 else "agentview",
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )
    env = VisualizationWrapper(env)
    return env


def build_device(env, args):
    if args.device == "keyboard":
        from robosuite.devices import Keyboard
        device = Keyboard(env=env, pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse
        device = SpaceMouse(
            env=env,
            vendor_id=0x256F,
            product_id=0xC635,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        )
    elif args.device == "dualsense":
        from robosuite.devices import DualSense
        device = DualSense(
            env=env,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
            reverse_xy=args.reverse_xy,
        )
    elif args.device == "mjgui":
        assert args.renderer == "mjviewer", "Mocap is only supported with the mjviewer renderer"
        from robosuite.devices.mjgui import MJGUI
        device = MJGUI(env=env)
    else:
        raise Exception(f"Invalid device choice: {args.device}")
    return device


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, default=os.path.join(suite.models.assets_root, "demonstrations_private"))
    parser.add_argument("--environment", type=str, default="Lift")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda")
    parser.add_argument("--config", type=str, default="default")
    parser.add_argument("--arm", type=str, default="right")
    parser.add_argument("--camera", nargs="*", type=str, default=["agentview"])
    parser.add_argument("--controller", type=str, default=None)
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument("--pos-sensitivity", type=float, default=1.0)
    parser.add_argument("--rot-sensitivity", type=float, default=1.0)
    parser.add_argument("--renderer", type=str, default="mjviewer")
    parser.add_argument("--max_fr", default=20, type=int)
    parser.add_argument("--reverse_xy", type=bool, default=False)
    parser.add_argument("--goal_update_mode", type=str, default="target", choices=["target", "achieved"])
    parser.add_argument("--img_height", type=int, default=256)
    parser.add_argument("--img_width", type=int, default=256)
    parser.add_argument("--cool_down_second", type=float, default=1)
    args = parser.parse_args()

    controller_config = load_composite_controller_config(
        controller=args.controller,
        robot=args.robots[0],
    )

    if controller_config["type"] == "WHOLE_BODY_MINK_IK":
        from robosuite.examples.third_party_controller.mink_controller import WholeBodyMinkIK  # noqa: F401

    if controller_config["type"] == "WHOLE_BODY_IK":
        assert len(args.robots) == 1, "Whole Body IK only supports one robot"

    env_config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    if "TwoArm" in args.environment:
        env_config["env_configuration"] = args.config

    env_info = json.dumps(env_config)

    safe_mkdir(args.directory)
    base_time = now_readable()
    tmp_root = os.path.join("/tmp", f"robosuite_intervention_{base_time}")
    safe_mkdir(tmp_root)

    tmp_hdf5_path = os.path.join(args.directory, f"intervention_{base_time}_0.hdf5")
    print(f"Output HDF5: {tmp_hdf5_path}")

    env = build_env(env_config, args)
    device = build_device(env, args)

    # Instantiate the base policy to be intervened
    action_dim = env.action_spec[0].shape[0]
    policy = DummyPolicy(action_dim)

    # Wrap the environment with data collection wrapper
    env_wrapped = DataCollectionWrapper(env, tmp_root)
    saved_count = 0
    env_wrapped.reset()

    try:
        while True:
            images_dict, step_labels = collect_intervention_trajectory(
                env=env_wrapped,
                device=device,
                policy=policy,
                arm=args.arm,
                max_fr=args.max_fr,
                goal_update_mode=args.goal_update_mode,
                camera_names=args.camera,
                img_height=args.img_height,
                img_width=args.img_width,
                cool_down=args.cool_down_second,
            )

            ep_dir = get_current_ep_dir(env_wrapped, tmp_root)
            env_wrapped.reset()

            if ep_dir is None or not os.path.isdir(ep_dir):
                print("Empty demo. Discarded.")
                continue

            demo_payload = read_single_demo_from_ep(ep_dir)
            if demo_payload is None:
                shutil.rmtree(ep_dir, ignore_errors=True)
                print("Empty demo. Discarded.")
                continue

            action = prompt_user_action()

            if action == "s":
                saved_count += 1
                append_demo_to_hdf5(
                    hdf5_path=tmp_hdf5_path,
                    demo_id=saved_count,
                    demo_payload=demo_payload,
                    step_labels=step_labels,
                    env_info=env_info,
                    camera_names=args.camera,
                    images_dict=images_dict,
                )
                print(f"Saved demo_{saved_count:06d}.")
                shutil.rmtree(ep_dir, ignore_errors=True)
                continue

            if action == "d":
                shutil.rmtree(ep_dir, ignore_errors=True)
                print("Deleted current demo.")
                continue

            if action == "q":
                shutil.rmtree(ep_dir, ignore_errors=True)
                print("Quit requested.")
                break

            if action == "f":
                saved_count += 1
                append_demo_to_hdf5(
                    hdf5_path=tmp_hdf5_path,
                    demo_id=saved_count,
                    demo_payload=demo_payload,
                    step_labels=step_labels,
                    env_info=env_info,
                    camera_names=args.camera,
                    images_dict=images_dict,
                )
                print(f"Saved demo_{saved_count:06d}.")
                shutil.rmtree(ep_dir, ignore_errors=True)
                print("Save and quit")
                break

    finally:
        try:
            if hasattr(device, "stop_control"):
                device.stop_control()
        except Exception:
            pass

        try:
            env.close()
        except Exception:
            pass

    final_hdf5_path = os.path.join(args.directory, f"intervention_{base_time}_{saved_count}.hdf5")
    print(f"\nFinal episodes: {saved_count}")
    try:
        if os.path.isfile(tmp_hdf5_path):
            os.replace(tmp_hdf5_path, final_hdf5_path)
            print(f"Final HDF5: {final_hdf5_path}")
        else:
            print("No HDF5 file created.")
    except Exception as e:
        print(f"Failed to rename HDF5: {e}")
        print(f"Kept temporary file: {tmp_hdf5_path}")