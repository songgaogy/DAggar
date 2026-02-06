"""
NOTE(gaoyuan): not official implementation; toy-case only
"""

import argparse
import datetime
import json
import os
import time
from glob import glob
from copy import deepcopy

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
    
    # NOTE(gaoyuan) 
    threshold = 0.1
    total_mag = 0.0
    for key, value in input_ac_dict.items():
        if ("delta" in key) and isinstance(value, np.ndarray):
            total_mag += np.linalg.norm(value)

    print(f"current magnitude: {total_mag}")
    return total_mag > threshold


def collect_intervention_trajectory(env, device, policy, arm, max_fr, goal_update_mode):
    obs = env.reset()
    env.render()

    task_completion_hold_count = -1 
    device.start_control()

    for robot in env.robots:
        robot.print_action_info_dict()

    # Track previous gripper actions
    all_prev_gripper_actions = [
        {
            f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
            for robot_arm in robot.arms
            if robot.gripper[robot_arm].dof > 0
        }
        for robot in env.robots
    ]

    print("\n[INFO] Loop started. Default: Policy. Intervene with Device.")

    while True:
        start = time.time()
        input_ac_dict = device.input2action(goal_update_mode=goal_update_mode)

        if input_ac_dict is None:
            break

        is_intervening = check_intervention(device, input_ac_dict)
        env_action = None

        if is_intervening:
            # Human Control
            active_robot = env.robots[device.active_robot]
            action_dict = deepcopy(input_ac_dict)

            for arm in active_robot.arms:
                if isinstance(active_robot.composite_controller, WholeBody):
                    input_type = active_robot.composite_controller.joint_action_policy.input_type
                else:
                    input_type = active_robot.part_controllers[arm].input_type

                if input_type == "delta":
                    action_dict[arm] = input_ac_dict[f"{arm}_delta"]
                elif input_type == "absolute":
                    action_dict[arm] = input_ac_dict[f"{arm}_abs"]

            env_action_list = [robot.create_action_vector(all_prev_gripper_actions[i]) for i, robot in enumerate(env.robots)]
            env_action_list[device.active_robot] = active_robot.create_action_vector(action_dict)
            env_action = np.concatenate(env_action_list)

            # Update gripper state
            for gripper_ac in all_prev_gripper_actions[device.active_robot]:
                all_prev_gripper_actions[device.active_robot][gripper_ac] = action_dict[gripper_ac]

        else:
            # Policy Control: random action
            env_action = policy.get_action(obs)

        obs, reward, done, info = env.step(env_action)
        env.render()

        if task_completion_hold_count == 0:
            break

        if env._check_success():
            task_completion_hold_count = task_completion_hold_count - 1 if task_completion_hold_count > 0 else 10
        else:
            task_completion_hold_count = -1

        if max_fr is not None:
            elapsed = time.time() - start
            diff = 1 / max_fr - elapsed
            if diff > 0:
                time.sleep(diff)

    env.close()


def gather_demonstrations_as_hdf5(directory, out_dir, env_info):
    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    f = h5py.File(hdf5_path, "w")
    grp = f.create_group("data")

    num_eps = 0
    env_name = None

    for ep_directory in os.listdir(directory):
        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        actions = []
        success = False

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])
            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])
            success = success or dic["successful"]

        if len(states) == 0:
            continue

        if success:
            print("Demonstration successful. Saving...")
            del states[-1]
            assert len(states) == len(actions)

            num_eps += 1
            ep_data_grp = grp.create_group(f"demo_{num_eps}")

            xml_path = os.path.join(directory, ep_directory, "model.xml")
            with open(xml_path, "r") as f_xml:
                ep_data_grp.attrs["model_file"] = f_xml.read()

            ep_data_grp.create_dataset("states", data=np.array(states))
            ep_data_grp.create_dataset("actions", data=np.array(actions))
        else:
            print("Demonstration failed. Discarding.")

    now = datetime.datetime.now()
    grp.attrs["date"] = f"{now.month}-{now.day}-{now.year}"
    grp.attrs["time"] = f"{now.hour}:{now.minute}:{now.second}"
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, default=os.path.join(suite.models.assets_root, "demonstrations_private"))
    parser.add_argument("--environment", type=str, default="Lift")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda")
    parser.add_argument("--config", type=str, default="default")
    parser.add_argument("--arm", type=str, default="right")
    parser.add_argument("--camera", nargs="*", type=str, default="agentview")
    parser.add_argument("--controller", type=str, default=None)
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument("--pos-sensitivity", type=float, default=1.0)
    parser.add_argument("--rot-sensitivity", type=float, default=1.0)
    parser.add_argument("--renderer", type=str, default="mjviewer")
    parser.add_argument("--max_fr", default=20, type=int)
    parser.add_argument("--reverse_xy", type=bool, default=False)
    parser.add_argument("--goal_update_mode", type=str, default="target", choices=["target", "achieved"])
    args = parser.parse_args()

    controller_config = load_composite_controller_config(controller=args.controller, robot=args.robots[0])

    if controller_config["type"] == "WHOLE_BODY_MINK_IK":
        from robosuite.examples.third_party_controller.mink_controller import WholeBodyMinkIK
    if controller_config["type"] == "WHOLE_BODY_IK":
        assert len(args.robots) == 1, "Whole Body IK only supports one robot"

    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }
    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config

    env = suite.make(
        **config,
        has_renderer=True,
        renderer=args.renderer,
        has_offscreen_renderer=False,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

    action_dim = env.action_spec[0].shape[0]
    policy = DummyPolicy(action_dim)
    env = VisualizationWrapper(env)
    env_info = json.dumps(config)

    tmp_directory = f"/tmp/{str(time.time()).replace('.', '_')}"
    env = DataCollectionWrapper(env, tmp_directory)

    if args.device == "keyboard":
        from robosuite.devices import Keyboard
        device = Keyboard(env=env, pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse
        device = SpaceMouse(env=env, pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    elif args.device == "dualsense":
        from robosuite.devices import DualSense
        device = DualSense(env=env, pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity, reverse_xy=args.reverse_xy)
    elif args.device == "mjgui":
        assert args.renderer == "mjviewer", "Mocap only supported with mjviewer"
        from robosuite.devices.mjgui import MJGUI
        device = MJGUI(env=env)
    else:
        raise Exception("Invalid device choice.")

    t1, t2 = str(time.time()).split(".")
    new_dir = os.path.join(args.directory, f"{t1}_{t2}")
    os.makedirs(new_dir)

    print(f"Collecting to: {new_dir}")
    print("Press Ctrl+C to stop if needed.")

    while True:
        collect_intervention_trajectory(env, device, policy, args.arm, args.max_fr, args.goal_update_mode)
        gather_demonstrations_as_hdf5(tmp_directory, new_dir, env_info)