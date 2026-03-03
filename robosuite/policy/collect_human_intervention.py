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
import torch
from torchvision.transforms import Normalize
from collections import deque

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.controllers.composite.composite_controller import WholeBody
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper

from utils.env_util import PandaLiftProprioExtractor
from flow import FlowPolicy
from eval_flow import center_crop_resize, sample_action_fast


class BasePolicy:
    def reset(self, initial_obs, env):
        pass
        
    def update_history(self, obs, env):
        pass
        
    def notify_intervention(self):
        """Called when human takes over. Invalidate current action chunks."""
        pass
        
    def get_action(self):
        raise NotImplementedError


class DummyPolicy(BasePolicy):
    def __init__(self, action_dim):
        self.action_dim = action_dim

    def get_action(self):
        return np.random.normal(0, 0.02, size=self.action_dim)


class FlowMatchWrapper(BasePolicy):
    def __init__(self, cfg: DictConfig, env):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        if isinstance(cfg.camera, str):
            self.camera_names_list = [cfg.camera]
        else:
            self.camera_names_list = list(cfg.camera)
        
        # Extractor for proprio
        self.extractor = PandaLiftProprioExtractor(
            robots="Panda",
            env_name="Lift",
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=self.camera_names_list,
            reward_shaping=False,
        )
        # Bind extractor to environment's sim
        self.extractor.env = env
        
        # Load Checkpoint
        ckpt_path = to_absolute_path(cfg.ckpt)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        
        def _to_numpy(x):
            if x is None: return None
            if torch.is_tensor(x): return x.detach().cpu().numpy()
            return x

        self.act_mean = _to_numpy(ckpt.get("act_mean"))
        self.act_std = _to_numpy(ckpt.get("act_std"))
        self.prop_mean = _to_numpy(ckpt.get("prop_mean"))
        self.prop_std = _to_numpy(ckpt.get("prop_std"))
        
        state_dict = ckpt.get("ema_model", ckpt["model"])
        
        if self.act_mean is not None:
            act_dim = int(np.prod(self.act_mean.shape))
        else:
            out_w = next((state_dict[k] for k in ["out.weight", "head.weight", "vel_head.weight"] if k in state_dict), None)
            act_dim = int(out_w.shape[0])

        if self.prop_mean is not None:
            prop_dim = int(np.prod(self.prop_mean.shape))
        else:
            prop_w = next((state_dict[k] for k in ["prop_enc.0.weight", "proprio_enc.0.weight", "prop_encoder.0.weight"] if k in state_dict), None)
            prop_dim = int(prop_w.shape[1])

        self.base_act_dim = act_dim // cfg.chunk_size

        self.model = FlowPolicy(
            act_dim=act_dim,
            proprio_in_dim=prop_dim,
            img_dim=cfg.flow.image_dim,
            prop_dim=cfg.flow.propior_dim,
            time_dim=cfg.flow.time_dim,
            token_dim=cfg.flow.token_dim,
            pretrained_resnet=True,
            freeze_resnet=True,
            temporal_layers=cfg.flow.temporal_layers,
            temporal_heads=cfg.flow.temporal_heads,
            vel_hidden=cfg.flow.vel_hidden,
            vel_layers=cfg.flow.vel_layers,
            history_len=cfg.history_len,
            action_chunk_size=cfg.chunk_size,
            action_temporal_layers=cfg.flow.action_temporal_layers,
            action_temporal_heads=cfg.flow.action_temporal_heads,
        ).to(self.device)

        try:
            if "ema_model" in ckpt:
                self.model.load_state_dict(ckpt["ema_model"], strict=True)
                print("Successfully loaded EMA model weights for Flow Policy.")
            else:
                self.model.load_state_dict(ckpt["model"], strict=True)
                print("Loaded standard model weights for Flow Policy.")
        except RuntimeError as e:
            raise RuntimeError(
                "Checkpoint is incompatible with current FlowPolicy architecture. "
                "Please retrain with the updated model or switch to a matching code version."
            ) from e
            
        self.model.eval()
        self.img_normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        self.img_history = deque(maxlen=cfg.history_len)
        self.prop_history = deque(maxlen=cfg.history_len)
        
        self.current_chunk = None
        self.step_in_chunk = 0

    def reset(self, initial_obs, initial_images, env):
        self.img_history.clear()
        self.prop_history.clear()
        
        prop = self.extractor.extract(env.sim.get_state().flatten()).reshape(-1)
        
        cam_name = self.camera_names_list[0]
        raw_img = initial_images[cam_name]
        
        img = center_crop_resize(raw_img, self.cfg.image_size)
        img_processed = np.transpose(img.astype(np.float32) / 255.0, (2, 0, 1))

        for _ in range(self.cfg.history_len):
            self.prop_history.append(prop)
            self.img_history.append(img_processed)
            
        self.current_chunk = None
        self.step_in_chunk = 0

    def update_history(self, obs, step_images, env):
        prop = self.extractor.extract(env.sim.get_state().flatten()).reshape(-1)
        self.prop_history.append(prop)
        
        cam_name = self.camera_names_list[0]
        raw_img = step_images[cam_name]
        
        img = center_crop_resize(raw_img, self.cfg.image_size)
        img_processed = np.transpose(img.astype(np.float32) / 255.0, (2, 0, 1))
        self.img_history.append(img_processed)

    def notify_intervention(self):
        """Invalidate the current chunk so it replans upon return to policy mode."""
        self.current_chunk = None
        self.step_in_chunk = 0

    def get_action(self):
        execute_steps = min(self.cfg.action_horizon, self.cfg.chunk_size)
        
        if self.current_chunk is None or self.step_in_chunk >= execute_steps:
            # Need to sample a new action chunk
            imgs_stacked = np.stack(self.img_history, axis=0)
            images = torch.from_numpy(imgs_stacked).unsqueeze(0).to(self.device)
            B, K, C, H, W = images.shape
            images_flat = images.view(B * K, C, H, W)
            images_norm = self.img_normalize(images_flat).view(B, K, C, H, W)
            
            prop_stacked = np.concatenate(self.prop_history, axis=0).astype(np.float32)
            if self.prop_mean is not None:
                prop_stacked = (prop_stacked - self.prop_mean) / self.prop_std
            proprio = torch.from_numpy(prop_stacked).unsqueeze(0).to(self.device)

            a_norm = sample_action_fast(self.model, images_norm, proprio, n_steps=self.cfg.n_ode_steps).cpu().numpy().reshape(-1)
            
            if self.act_mean is not None:
                a_flat = a_norm * self.act_std + self.act_mean
            else:
                a_flat = a_norm

            self.current_chunk = a_flat.reshape(self.cfg.chunk_size, self.base_act_dim)
            self.step_in_chunk = 0

        action_step = self.current_chunk[self.step_in_chunk]
        self.step_in_chunk += 1
        return action_step


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
        f.attrs["camera_names"] = json.dumps(list(camera_names))
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


def collect_intervention_trajectory(env, device, policy, arm, max_fr, goal_update_mode, 
                                    camera_names, img_height, img_width, cool_down):
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

    if isinstance(camera_names, str):
        camera_names = [camera_names]

    images = {cam: [] for cam in camera_names}
    step_labels = []

    # State machine parameters
    state_mode = "POLICY"
    prev_state_mode = "POLICY"
    COOLDOWN_STEPS = int(20 * cool_down)  # assuming 20Hz control frequency
    cooldown_counter = 0

    obs = env.unwrapped._get_observations()
    initial_images = {}
    for cam in camera_names:
        img = env.sim.render(height=img_height, width=img_width, camera_name=cam)
        initial_images[cam] = img
        images[cam].append(img)
        
    policy.reset(obs, initial_images, env.unwrapped)
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
            env_action = policy.get_action()

        # Step environment
        obs, reward, done, info = env.step(env_action)

        step_images = {}
        for cam in camera_names:
            img = env.sim.render(height=img_height, width=img_width, camera_name=cam)
            step_images[cam] = img
            images[cam].append(img)
            
        # Update policy history with the new observations and images
        policy.update_history(obs, step_images, env.unwrapped)

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


def build_env(env_config: dict, cfg):
    camera_names = cfg.camera if len(cfg.camera) > 0 else ["agentview"]
    env = suite.make(
        **env_config,
        has_renderer=True,
        renderer=cfg.renderer,
        has_offscreen_renderer=True,
        # `render_camera` is the interactive viewer camera only.
        render_camera=camera_names[0],
        # Configure all requested cameras for multi-view image capture.
        camera_names=camera_names,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )
    env = VisualizationWrapper(env)
    return env


def build_device(env, cfg):
    if cfg.device == "keyboard":
        from robosuite.devices import Keyboard
        device = Keyboard(env=env, pos_sensitivity=cfg.pos_sensitivity, rot_sensitivity=cfg.rot_sensitivity)
    elif cfg.device == "spacemouse":
        from robosuite.devices import SpaceMouse
        device = SpaceMouse(
            env=env,
            vendor_id=0x256F,
            product_id=0xC635,
            pos_sensitivity=cfg.pos_sensitivity,
            rot_sensitivity=cfg.rot_sensitivity,
        )
    elif cfg.device == "dualsense":
        from robosuite.devices import DualSense
        device = DualSense(
            env=env,
            pos_sensitivity=cfg.pos_sensitivity,
            rot_sensitivity=cfg.rot_sensitivity,
            reverse_xy=cfg.reverse_xy,
        )
    elif cfg.device == "mjgui":
        assert cfg.renderer == "mjviewer", "Mocap is only supported with the mjviewer renderer"
        from robosuite.devices.mjgui import MJGUI
        device = MJGUI(env=env)
    else:
        raise Exception(f"Invalid device choice: {cfg.device}")
    return device


@hydra.main(version_base="1.2", config_path="./config", config_name="collect_intervention")
def main(cfg: DictConfig):
    if isinstance(cfg.camera, str):
        cfg.camera = [cfg.camera]
    else:
        cfg.camera = list(cfg.camera)
    if len(cfg.camera) == 0:
        cfg.camera = ["agentview"]

    icfg = cfg.intervention

    controller_config = load_composite_controller_config(
        controller=icfg.controller,
        robot=icfg.robots[0],
    )

    if controller_config["type"] == "WHOLE_BODY_MINK_IK":
        from robosuite.examples.third_party_controller.mink_controller import WholeBodyMinkIK  # noqa: F401

    if controller_config["type"] == "WHOLE_BODY_IK":
        assert len(icfg.robots) == 1, "Whole Body IK only supports one robot"

    env_config = {
        "env_name": icfg.environment,
        "robots": list(icfg.robots),
        "controller_configs": controller_config,
    }

    if "TwoArm" in icfg.environment:
        env_config["env_configuration"] = icfg.config

    env_info = json.dumps(env_config)

    safe_mkdir(icfg.directory)
    base_time = now_readable()
    tmp_root = os.path.join("/tmp", f"robosuite_intervention_{base_time}")
    safe_mkdir(tmp_root)

    tmp_hdf5_path = os.path.join(icfg.directory, f"intervention_{base_time}_0.hdf5")
    print(f"Output HDF5: {tmp_hdf5_path}")

    # Create dummy args mapping for compatibility with build_env / build_device
    class DummyArgs: pass
    args = DummyArgs()
    for k, v in icfg.items(): setattr(args, k, v)
    
    env = build_env(env_config, args)
    device = build_device(env, args)

    # policy
    if cfg.policy_type == "flow":
        print("Using FlowMatching as the base policy.")
        policy = FlowMatchWrapper(cfg, env.unwrapped)
    elif cfg.policy_type == "dummy":
        print("Using Dummy Policy (Random Noise).")
        action_dim = env.action_spec[0].shape[0]
        policy = DummyPolicy(action_dim)
    else:
        raise ValueError(f"Unknown policy_type: {cfg.policy_type}")

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
                arm=icfg.arm,
                max_fr=icfg.max_fr,
                goal_update_mode=icfg.goal_update_mode,
                camera_names=cfg.camera,
                img_height=icfg.img_height,
                img_width=icfg.img_width,
                cool_down=icfg.cool_down_second,
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
                    camera_names=cfg.camera,
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
                    camera_names=cfg.camera,
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

    final_hdf5_path = os.path.join(icfg.directory, f"intervention_{base_time}_{saved_count}.hdf5")
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


if __name__ == "__main__":
    main()
