"""
A script to collect a batch of human demonstrations and save them to a Zarr ReplayBuffer
for Diffusion Policy training.

Based on robosuite's `collect_human_demonstrations.py`
"""

import argparse
import datetime
import json
import os
import time
import numpy as np
from glob import glob
import hydra
from omegaconf import OmegaConf, DictConfig
from copy import deepcopy

import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.controllers.composite.composite_controller import WholeBody
from robosuite.wrappers import VisualizationWrapper
from diffusion_policy.diffusion_policy.common.replay_buffer import ReplayBuffer


def collect_human_trajectory(env, device, arm, max_fr, goal_update_mode):
    """
    Use the device (keyboard or SpaceNav 3D mouse) to collect a demonstration.
    Returns the collected episode data dictionary or None if discarded.
    """
    obs = env.reset()
    env.render()

    task_completion_hold_count = -1
    device.start_control()

    episode_data = {
        'wrist_cam': [],
        'side_cam': [],
        'tcp_pose': [],
        'joint_pos': [],
        'action': []
    }

    all_prev_gripper_actions = [
        {
            f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
            for robot_arm in robot.arms
            if robot.gripper[robot_arm].dof > 0
        }
        for robot in env.robots
    ]

    print("Collection started. Press device reset button (or ESC/Space on keyboard) to stop.")
    while True:
        start = time.time()

        active_robot = env.robots[device.active_robot]
        input_ac_dict = device.input2action(goal_update_mode=goal_update_mode)

        if input_ac_dict is None:
            return None

        action_dict = deepcopy(input_ac_dict)
        for arm in active_robot.arms:
            if isinstance(active_robot.composite_controller, WholeBody):
                controller_input_type = active_robot.composite_controller.joint_action_policy.input_type
            else:
                controller_input_type = active_robot.part_controllers[arm].input_type

            if controller_input_type == "delta":
                action_dict[arm] = input_ac_dict[f"{arm}_delta"]
            elif controller_input_type == "absolute":
                action_dict[arm] = input_ac_dict[f"{arm}_abs"]
            else:
                raise ValueError

        # maintaining gripper state
        env_action = [robot.create_action_vector(all_prev_gripper_actions[i]) for i, robot in enumerate(env.robots)]
        env_action[device.active_robot] = active_robot.create_action_vector(action_dict)
        env_action = np.concatenate(env_action)
        for gripper_ac in all_prev_gripper_actions[device.active_robot]:
            all_prev_gripper_actions[device.active_robot][gripper_ac] = action_dict[gripper_ac]

        obs, reward, done, info = env.step(env_action)
        env.render()

        if 'robot0_eye_in_hand_image' in obs:
            episode_data['wrist_cam'].append(obs['robot0_eye_in_hand_image'])
        if 'agentview_image' in obs:
            episode_data['side_cam'].append(obs['agentview_image'])
        if 'robot0_eef_pos' in obs and 'robot0_eef_quat' in obs:
            tcp = np.concatenate([obs['robot0_eef_pos'], obs['robot0_eef_quat']])
            episode_data['tcp_pose'].append(tcp)
        if 'robot0_joint_pos' in obs:
            episode_data['joint_pos'].append(obs['robot0_joint_pos'])

        episode_data['action'].append(env_action)

        if task_completion_hold_count == 0:
            break

        # state machine to check for having a success for 10 consecutive timesteps
        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1
            else:
                task_completion_hold_count = 10
        else:
            task_completion_hold_count = -1

        # limit frame rate
        if max_fr is not None:
            elapsed = time.time() - start
            diff = 1 / max_fr - elapsed
            if diff > 0:
                time.sleep(diff)

    result = dict()
    if len(episode_data['action']) == 0:
        return None

    # stack lists into numpy arrays
    for key in episode_data:
        if len(episode_data[key]) > 0:
            result[key] = np.stack(episode_data[key], axis=0)
        else:
            pass 
            
    return result


@hydra.main(version_base=None, config_path="config/task", config_name="robosuite_lift_image")
def main(cfg: DictConfig):
    if 'collect' not in cfg:
        raise ValueError("Config file must contain a 'collect' section!")

    args = cfg.collect
    
    resolution = list(args.resolution)
    if len(resolution) == 1:
        resolution = [resolution[0], resolution[0]]

    output_dir = args.output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zarr_path = os.path.join(output_dir, f"replay_buffer_{timestamp}.zarr")
    replay_buffer = ReplayBuffer.create_from_path(zarr_path, mode='a')
    print(f"[INFO] Data will be saved to: {zarr_path}")

    robots = list(args.robots) if OmegaConf.is_list(args.robots) else [args.robots]
    controller_config = load_composite_controller_config(
        controller=args.controller,
        robot=robots[0],
    )

    env_config = {
        "env_name": args.environment,
        "robots": robots,
        "controller_configs": controller_config,
    }

    if "TwoArm" in args.environment:
        env_config["env_configuration"] = args.config

    print(f"Initializing environment: {args.environment}")
    
    camera_names = list(args.camera) if OmegaConf.is_list(args.camera) else [args.camera]
    env = suite.make(
        **env_config,
        has_renderer=True,
        renderer=args.renderer,
        has_offscreen_renderer=True, # Needed for camera recording
        render_camera=camera_names[0],
        ignore_done=True,
        use_camera_obs=True,
        camera_names=camera_names,
        camera_heights=resolution[0],
        camera_widths=resolution[1],
        reward_shaping=True,
        control_freq=args.max_fr,
    )
    env = VisualizationWrapper(env)

    print("Warming up renderer...")
    env.reset()
    env.render() 

    # Initialize Device
    print(f"Initializing device: {args.device}")
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
        from robosuite.devices.mjgui import MJGUI
        device = MJGUI(env=env)
    else:
        raise ValueError(f"Unknown device: {args.device}")

    try:
        while True:
            print(f"\n--- Ready to record Episode {replay_buffer.n_episodes} ---")
            episode_data = collect_human_trajectory(env, device, args.arm, args.max_fr, args.goal_update_mode)
            
            if episode_data is not None:
                print("Saving episode...")
                replay_buffer.add_episode(episode_data, compressors='disk')
                print(f"Saved episode {replay_buffer.n_episodes - 1}. Steps: {len(episode_data['action'])}")
            else:
                print("Episode discarded.")

            print("waiting to resume in 3s...")
            time.sleep(3)
            print("You may start again!")

    except KeyboardInterrupt:
        print("\n\n!!! Keyboard Interrupt Detected !!!")
        print("Stopping recording gracefully...")
        print(f"Total episodes successfully saved: {replay_buffer.n_episodes - 1}")
        print(f"Data location: {zarr_path}")
        print("Exiting.")

    finally:
        env.close()


if __name__ == "__main__":
    main()