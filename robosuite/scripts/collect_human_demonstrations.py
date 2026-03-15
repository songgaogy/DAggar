"""
A script to collect a batch of human demonstrations.

The demonstrations can be played back using the `playback_demonstrations_from_hdf5.py` script.
"""

import argparse
import datetime
import json
import os
import shutil
import sys
import time
from glob import glob

import h5py
import numpy as np

import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.controllers.composite.composite_controller import WholeBody
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper

try:
    import termios
    import tty
except ImportError:
    termios = None
    tty = None


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
    # DataCollectionWrapper typically exposes ep_directory; fallback to scanning tmp_root.
    ep_dir = getattr(env_wrapped, "ep_directory", None)
    if isinstance(ep_dir, str) and len(ep_dir) > 0:
        return ep_dir
    eps = list_ep_dirs(tmp_root)
    return eps[-1] if len(eps) > 0 else None


def read_single_demo_from_ep(ep_dir: str):
    """
    Read a single episode worth of DataCollectionWrapper outputs from @ep_dir.
    """
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

    # Delete the last state. This is because when the DataCollector wrapper
    # recorded the states and actions, the states were recorded AFTER playing that action,
    # so we end up with an extra state at the end.
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
    env_info: str,
    camera_names: list[str],
    images_dict: dict,
):
    """
    Append one demonstration into a single hdf5 file.

    The strucure of the hdf5 file is as follows.

    demos (group)
        demo_000001 (group)
            model_file (attribute) - model xml string for demonstration
            states (dataset) - flattened mujoco states
            actions (dataset) - actions applied during demonstration
            observations (group)
                <camera_name> (group)
                    images (dataset) - uint8 images, shape (T, H, W, 3)
    """
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

    # store model xml as an attribute
    xml_path = os.path.join(demo_payload["ep_dir"], "model.xml")
    if os.path.isfile(xml_path):
        with open(xml_path, "r") as fx:
            xml_str = fx.read()
        demo_grp.attrs["model_file"] = xml_str

    states = demo_payload["states"]
    actions = demo_payload["actions"]

    # Align lengths across states/actions/images
    T = min(states.shape[0], actions.shape[0])
    for cam in camera_names:
        imgs = images_dict.get(cam, None)
        if isinstance(imgs, np.ndarray) and imgs.ndim == 4:
            T = min(T, imgs.shape[0])

    states = states[:T]
    actions = actions[:T]

    demo_grp.attrs["length"] = int(T)
    demo_grp.attrs["successful"] = bool(demo_payload.get("success", False))

    demo_grp.create_dataset("states", data=states)
    demo_grp.create_dataset("actions", data=actions)

    obs_grp = demo_grp.create_group("observations")

    for cam in camera_names:
        cam_grp = obs_grp.create_group(cam)
        imgs = images_dict.get(cam, None)
        if imgs is None or not isinstance(imgs, np.ndarray) or imgs.ndim != 4 or imgs.shape[0] == 0:
            cam_grp.create_dataset("images", data=np.zeros((0,), dtype=np.uint8))
            continue

        imgs = imgs[:T].astype(np.uint8)

        cam_grp.create_dataset(
            "images",
            data=imgs,
            dtype=np.uint8,
            compression="gzip",
            compression_opts=4,
            chunks=True,
        )

    f.close()


def collect_human_trajectory(env, device, arm, max_fr, goal_update_mode, camera_names, img_height, img_width):
    """
    Use the device (keyboard or SpaceNav 3D mouse) to collect a demonstration.
    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env (MujocoEnv): environment to control
        device (Device): to receive controls from the device
        arms (str): which arm to control (eg bimanual) 'right' or 'left'
        max_fr (int): if specified, pause the simulation whenever simulation runs faster than max_fr
    """

    # NOTE:
    # We do not call env.reset() here. DataCollectionWrapper flushes to disk on reset(),
    # and deleting ep directories before flush can cause FileNotFoundError.
    # The caller should manage env.reset() boundaries.

    env.render()

    task_completion_hold_count = -1  # counter to collect 10 timesteps after reaching goal
    device.start_control()

    for robot in env.robots:
        robot.print_action_info_dict()

    # Keep track of prev gripper actions when using since they are position-based and must be maintained when arms switched
    all_prev_gripper_actions = [
        {
            f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
            for robot_arm in robot.arms
            if robot.gripper[robot_arm].dof > 0
        }
        for robot in env.robots
    ]

    images = {cam: [] for cam in camera_names}

    # Loop until we get a reset from the input or the task completes
    while True:
        start = time.time()

        # Set active robot
        active_robot = env.robots[device.active_robot]

        # Get the newest action
        input_ac_dict = device.input2action(goal_update_mode=goal_update_mode)

        # If action is none, then this a reset so we should break
        if input_ac_dict is None:
            break

        from copy import deepcopy

        action_dict = deepcopy(input_ac_dict)  # {}
        # set arm actions
        for arm_name in active_robot.arms:
            if isinstance(active_robot.composite_controller, WholeBody):  # input type passed to joint_action_policy
                controller_input_type = active_robot.composite_controller.joint_action_policy.input_type
            else:
                controller_input_type = active_robot.part_controllers[arm_name].input_type

            if controller_input_type == "delta":
                action_dict[arm_name] = input_ac_dict[f"{arm_name}_delta"]
            elif controller_input_type == "absolute":
                action_dict[arm_name] = input_ac_dict[f"{arm_name}_abs"]
            else:
                raise ValueError

        # Maintain gripper state for each robot but only update the active robot with action
        env_action = [robot.create_action_vector(all_prev_gripper_actions[i]) for i, robot in enumerate(env.robots)]
        env_action[device.active_robot] = active_robot.create_action_vector(action_dict)
        env_action = np.concatenate(env_action)
        for gripper_ac in all_prev_gripper_actions[device.active_robot]:
            all_prev_gripper_actions[device.active_robot][gripper_ac] = action_dict[gripper_ac]

        env.step(env_action)

        # Capture multiview images after env.step so that images align with post-action states.
        for cam in camera_names:
            img = env.sim.render(height=img_height, width=img_width, camera_name=cam)
            images[cam].append(img)

        env.render()

        # Also break if we complete the task
        if task_completion_hold_count == 0:
            break

        # state machine to check for having a success for 10 consecutive timesteps
        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1  # latched state, decrement count
            else:
                task_completion_hold_count = 10  # reset count on first success timestep
        else:
            task_completion_hold_count = -1  # null the counter if there's no success

        # limit frame rate if necessary
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

    return images_np


def prompt_user_action() -> str:
    while True:
        prompt = "Action for this demo: [s]ave / [d]elete / [q]uit / [f]inish ? "
        if termios is not None and tty is not None and sys.stdin.isatty():
            print(prompt, end="", flush=True)
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                ch = sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            if ch == "\x03":
                raise KeyboardInterrupt
            ans = ch.strip().lower()
            print(ans)
        else:
            ans = input(prompt).strip().lower()
        if ans in ("s", "d", "q", "f"):
            return ans
        print("Invalid input. Please press s, d, f or q.")


def build_env(env_config: dict, args):
    requested_camera_names = args.camera if args.camera else ["agentview"]
    env = suite.make(
        **env_config,
        has_renderer=True,
        renderer=args.renderer,
        has_offscreen_renderer=True,
        # `render_camera` controls only the on-screen viewer camera.
        render_camera=requested_camera_names[0],
        # Configure all requested cameras in the env so multiview capture can use all of them.
        camera_names=requested_camera_names,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )
    env = VisualizationWrapper(env)
    if args.camera:
        camera_names = list(args.camera)
    else:
        camera_names = [str(cam_name) for cam_name in env.sim.model.camera_names]
        if len(camera_names) == 0:
            camera_names = ["agentview"]
        print(f"No --camera provided. Saving all cameras by default: {camera_names}")
    return env, camera_names


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
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default=os.path.join(suite.models.assets_root, "demonstrations_private"),
    )
    parser.add_argument("--environment", type=str, default="Lift")
    parser.add_argument(
        "--robots",
        nargs="+",
        type=str,
        default="Panda",
        help="Which robot(s) to use in the env",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Specified environment configuration if necessary",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="right",
        help="Which arm to control (eg bimanual) 'right' or 'left'",
    )
    parser.add_argument(
        "--camera",
        nargs="*",
        type=str,
        default=[],
        help="List of camera names to save. Pass multiple names to enable multiple views. Note: the `mujoco` renderer must be enabled when using multiple views; `mjviewer` is not supported.",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default=None,
        help="Choice of controller. Can be generic (eg. 'BASIC' or 'WHOLE_BODY_MINK_IK') or json file (see robosuite/controllers/config for examples)",
    )
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument(
        "--pos-sensitivity",
        type=float,
        default=1.0,
        help="How much to scale position user inputs",
    )
    parser.add_argument(
        "--rot-sensitivity",
        type=float,
        default=1.0,
        help="How much to scale rotation user inputs",
    )
    parser.add_argument(
        "--renderer",
        type=str,
        default="mjviewer",
        help="Use Mujoco's builtin interactive viewer (mjviewer) or OpenCV viewer (mujoco)",
    )
    parser.add_argument(
        "--max_fr",
        default=20,
        type=int,
        help="Sleep when simluation runs faster than specified frame rate; 20 fps is real time.",
    )
    parser.add_argument(
        "--reverse_xy",
        type=bool,
        default=False,
        help="(DualSense Only)Reverse the effect of the x and y axes of the joystick.It is used to handle the case that the left/right and front/back sides of the view are opposite to the LX and LY of the joystick(Push LX up but the robot move left in your view)",
    )
    parser.add_argument(
        "--goal_update_mode",
        type=str,
        default="target",
        choices=["target", "achieved"],
        help="Used by the device to get the arm's actions. The mode to update the goal in. Can be 'target' or 'achieved'. If 'target', the goal is updated based on the current target pose. "
        "If 'achieved', the goal is updated based on the current achieved state. "
        "We recommend using 'achieved' (and input_ref_frame='base') if collecting demonstrations with a mobile base robot.",
    )
    parser.add_argument("--img_height", type=int, default=256)
    parser.add_argument("--img_width", type=int, default=256)
    args = parser.parse_args()

    # Get controller config
    controller_config = load_composite_controller_config(
        controller=args.controller,
        robot=args.robots[0],
    )

    if controller_config["type"] == "WHOLE_BODY_MINK_IK":
        # mink-speicific import. requires installing mink
        from robosuite.examples.third_party_controller.mink_controller import WholeBodyMinkIK  # noqa: F401

    # if WHOLE BODY IK; assert only one robot
    if controller_config["type"] == "WHOLE_BODY_IK":
        assert len(args.robots) == 1, "Whole Body IK only supports one robot"

    # Create argument configuration
    env_config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    # Check if we're using a multi-armed environment and use env_configuration argument if so
    if "TwoArm" in args.environment:
        env_config["env_configuration"] = args.config

    env_info = json.dumps(env_config)

    safe_mkdir(args.directory)

    base_time = now_readable()

    tmp_root = os.path.join("/tmp", f"robosuite_collect_{base_time}")
    safe_mkdir(tmp_root)

    tmp_hdf5_path = os.path.join(args.directory, f"{base_time}_0.hdf5")
    print(f"Output HDF5: {tmp_hdf5_path}")

    env, camera_names = build_env(env_config, args)
    device = build_device(env, args)

    # wrap the environment with data collection wrapper
    env_wrapped = DataCollectionWrapper(env, tmp_root)

    saved_count = 0

    # Start the first episode directory
    env_wrapped.reset()

    try:
        while True:
            # Collect one demo within the current episode directory
            images_dict = collect_human_trajectory(
                env=env_wrapped,
                device=device,
                arm=args.arm,
                max_fr=args.max_fr,
                goal_update_mode=args.goal_update_mode,
                camera_names=camera_names,
                img_height=args.img_height,
                img_width=args.img_width,
            )

            # Identify the episode directory that just finished
            ep_dir = get_current_ep_dir(env_wrapped, tmp_root)

            # Trigger flush-to-disk for the finished episode by starting a new episode.
            # DataCollectionWrapper writes state_*.npz during _flush() called inside reset().
            env_wrapped.reset()

            # Now ep_dir should contain flushed state_*.npz files and can be safely read/deleted.
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
                    env_info=env_info,
                    camera_names=camera_names,
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
                    env_info=env_info,
                    camera_names=camera_names,
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

    final_hdf5_path = os.path.join(args.directory, f"{base_time}_{saved_count}.hdf5")
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
