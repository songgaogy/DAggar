import os
import sys
import time
import hydra
import numpy as np
import torch
import dill
import cv2
import robosuite as suite
from copy import deepcopy
from typing import Dict, List, Any, Optional
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.transform import Rotation as R

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ARMADA_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_DIFFUSION_POLICY_DIR = os.path.join(_ARMADA_DIR, "diffusion_policy")
if _DIFFUSION_POLICY_DIR not in sys.path:
    sys.path.insert(0, _DIFFUSION_POLICY_DIR)

from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import VisualizationWrapper

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from utils.episode_manager import EpisodeManager
from .base_env_runner import BaseEnvRunner
from .utils import AsyncKeyHandler
from utils.macros import HUMAN, INTV_END, INTV, ROBOT


class RobosuiteRunner(BaseEnvRunner):
    """
    Environment runner for Robosuite simulation.
    Adapts the RealEnvRunner logic to interact with a gym-like simulation environment.
    """
    def __init__(
            self, 
            cfg: DictConfig, 
            rank: int, 
            device_ids: List[int]
        ):
        super().__init__(cfg, rank, device_ids)

        # set random seed
        self.seed = cfg.seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # init components
        self.keyboard = AsyncKeyHandler()
        self._load_policy()
        self._setup_transformers()
        self._initialize_sim_env()
        self._initialize_input_device()
        self._initialize_episode_manager()
        self._initialize_replay_buffer()
        self.episode_idx = 0
        self.saved_episode_idx = 0

        if self.cfg.train.task.env_runner.max_steps is not None:
            self.max_episode_length = self.cfg.train.task.env_runner.max_steps
        else:
            self.max_episode_length = self._calculate_max_episode_length()
        
        # initialize failure detection module
        self.failure_detection_module = None
        if hasattr(cfg, 'failure_detection'):
            self._initialize_failure_detection_module()
        
        # output
        self.output_dir = cfg.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _load_policy(self):
        """
        Load policy from checkpoint
        """
        print(f"Loading policy from: {self.cfg.checkpoint_path}")
        payload = torch.load(open(self.cfg.checkpoint_path, 'rb'), pickle_module=dill)
        
        # Re-instantiate workspace to load weights
        cls = hydra.utils.get_class(self.cfg.train._target_)
        workspace = cls(self.cfg.train, self.rank, self.world_size, self.device_id, self.device)
        workspace.load_payload(payload, exclude_keys=("optimizer",), include_keys=None)
        
        # Extract model
        self.policy = getattr(workspace.model, "module", workspace.model)
        if self.cfg.train.training.use_ema:
            self.policy = getattr(workspace.ema_model, "module", workspace.ema_model)
        
        self.policy.to(self.device)
        self.policy.eval()
        
        # Extract metadata
        self.To = self.cfg.train.n_obs_steps
        self.Ta = self.cfg.Ta
        self.obs_feature_dim = self.policy.obs_feature_dim
        self.img_shape = self.cfg.train.task.image_shape  # [3, H, W]
    
    def _setup_transformers(self):
        """
        Setup rotation transformers
        """
        self.action_dim = self.cfg.train.task.shape_meta.action.shape[0]
        self.action_rot_transformer = None
        self.obs_rot_transformer = None
        
        # action rotation transformer
        # Robosuite usually uses axis-angle (3) or quat (4) depending on controller
        # TODO(gaoyuan): check this!
        if 'rotation_rep' in self.cfg.train.task.shape_meta.action:
            self.action_rot_transformer = RotationTransformer(
                from_rep='axis_angle',  
                to_rep=self.cfg.train.task.shape_meta.action.rotation_rep
            )

        # observation rotation transformer
        if 'ee_pose' in self.cfg.train.task.shape_meta.obs:
            self.ee_pose_dim = self.cfg.train.task.shape_meta.obs.ee_pose.shape[0]
            self.state_type = 'ee_pose'
            self.state_shape = self.cfg.train.task.shape_meta.obs.ee_pose.shape

            # Assuming Robosuite returns quaternion in obs, check your task wrapper
            # TODO(gaoyuan): check this!
            if 'rotation_rep' in self.cfg.train.task.shape_meta.obs.ee_pose:
                self.obs_rot_transformer = RotationTransformer(
                    from_rep='quaternion', 
                    to_rep=self.cfg.train.task.shape_meta.obs.ee_pose.rotation_rep
                )
        else:
            self.ee_pose_dim = self.cfg.train.task.shape_meta.obs.qpos.shape[0]
            self.state_type = 'qpos'
            self.state_shape = self.cfg.train.task.shape_meta.obs.qpos.shape

    def _initialize_sim_env(self):
        """
        Initialize Robosuite environment
        """
        # load controller config
        controller_config = load_composite_controller_config(
            controller=self.cfg.train.task.env.controller,
            robot=self.cfg.train.task.env.robots[0],
        ) 

        # create environment
        camera_names = list(self.cfg.train.task.env.camera) if OmegaConf.is_list(self.cfg.train.task.env.camera) \
                                                            else [self.cfg.train.task.env.camera]
        robots = list(self.cfg.train.task.env.robots) if OmegaConf.is_list(self.cfg.train.task.env.robots) \
                                                      else [self.cfg.train.task.env.robots]
        self.env = suite.make(
            env_name=self.cfg.train.task.env.environment,       # e.g. "Lift"
            robots=robots,                                      # e.g. "Panda"
            controller_configs=controller_config,
            has_renderer=True,
            renderer=self.cfg.train.task.env.renderer,
            has_offscreen_renderer=True,
            render_camera=self.cfg.train.task.env.camera[0],
            ignore_done=True,
            use_camera_obs=True,
            camera_names=camera_names,
            camera_heights=self.cfg.train.task.env.resolution[0],
            camera_widths=self.cfg.train.task.env.resolution[1],
            reward_shaping=True,
            control_freq=self.cfg.train.task.env.max_fr,
        )
        self.env = VisualizationWrapper(self.env)
        print("[INFO] Robosuite environment initialized.")

    def _initialize_input_device(self):
        """
        Initialize input device for human intervention.
        """
        self.input_device = self.cfg.train.task.env.device
        args = self.cfg.train.task.env
        print(f"Initializing device: {self.input_device}")

        if self.input_device == "keyboard":
            from robosuite.devices import Keyboard
            self.input_device = Keyboard(
                env=self.env, pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity
            )
        elif self.input_device == "spacemouse":
            from robosuite.devices import SpaceMouse
            self.input_device = SpaceMouse(
                env=self.env, pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity
            )
        elif self.input_device == "dualsense":
            from robosuite.devices import DualSense
            self.input_device = DualSense(
                env=self.env, pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity, reverse_xy=args.reverse_xy
            )
        elif self.input_device == "mjgui":
            from robosuite.devices.mjgui import MJGUI
            self.input_device = MJGUI(env=self.env)
        else:
            raise ValueError(f"Unknown device: {self.input_device}")

    def _initialize_episode_manager(self):
        """
        Initialize episode manager to handle observation buffer
        """
        self.episode_manager = EpisodeManager(
            obs_rot_transformer=self.obs_rot_transformer,
            action_rot_transformer=self.action_rot_transformer,
            obs_feature_dim=self.obs_feature_dim,
            img_shape=self.img_shape,
            state_type=self.state_type,
            state_shape=self.state_shape,
            action_dim=self.action_dim,
            To=self.To,
            Ta=self.Ta,
            device=self.device,
            num_samples=self.cfg.failure_detection.num_samples if hasattr(self.cfg, 'failure_detection') else 1
        )

    def _initialize_replay_buffer(self):
        """
        Initialize replay buffer for data collection (Aligned with original implementation)
        """
        base_zarr_path = self.cfg.train.dataset_path
        if not base_zarr_path.endswith('.zarr'):
            base_zarr_path = os.path.join(base_zarr_path, 'replay_buffer.zarr')

        print(f"[INFO] Copying structure from previous training set: {base_zarr_path}")
        self.replay_buffer = ReplayBuffer.copy_from_path(base_zarr_path, keys=None)

        # if the data is not initialized/formatted
        if 'action_mode' not in self.replay_buffer.keys():
            self.replay_buffer.data['action_mode'] = np.full((self.replay_buffer.n_steps, ), HUMAN)
        if 'failure_indices' not in self.replay_buffer.keys():
            self.replay_buffer.data['failure_indices'] = np.zeros((self.replay_buffer.n_steps, ), dtype=np.bool_)

    def _initialize_failure_detection_module(self):
        """
        Initialize failure detection module
        """
        self.failure_detection_module = hydra.utils.instantiate(self.cfg.failure_detection)
        self.failure_detection_module.runtime_initialize(
            device=self.device,
            policy=self.policy,
            replay_buffer=self.replay_buffer,
            episode_manager=self.episode_manager,
            max_episode_length=self.max_episode_length
        )

    def _calculate_max_episode_length(self) -> int:
        """
        Calculate maximum episode length based on expert demonstrations
        """
        human_demo_indices = []
        for i in range(self.replay_buffer.n_episodes):
            episode_start = self.replay_buffer.episode_ends[i-1] if i > 0 else 0
            if np.any(self.replay_buffer.data['action_mode'][episode_start: self.replay_buffer.episode_ends[i]] == HUMAN):
                human_demo_indices.append(i)
        
        human_eps_len = []
        for i in human_demo_indices:
            human_episode = self.replay_buffer.get_episode(i)
            human_eps_len.append(human_episode['side_cam'].shape[0])

        return int(torch.max(torch.tensor(human_eps_len)) // self.Ta * self.Ta)

    def _get_sim_observation(self, obs_dict):
        """
        Process raw robosuite observation
        """
        side_img = obs_dict['agentview_image']
        side_img = torch.from_numpy(side_img).float() / 255.0
        side_img = side_img.permute(2, 0, 1).contiguous()

        wrist_img = obs_dict['robot0_eye_in_hand_image']
        wrist_img = torch.from_numpy(wrist_img).float() / 255.0
        wrist_img = wrist_img.permute(2, 0, 1).contiguous()

        if self.state_type == 'ee_pose':
            tcp_pos = obs_dict['robot0_eef_pos']
            tcp_quat = obs_dict['robot0_eef_quat']
            state = np.concatenate([tcp_pos, tcp_quat])
        else:
            state = obs_dict['robot0_joint_pos']

        return {
            'policy_side_img': side_img,
            'policy_wrist_img': wrist_img,
            'state': torch.from_numpy(state).float(),
            'raw_obs': obs_dict
        }

    def _save_sim_state(self):
        return self.env.sim.get_state().flatten()

    def run_rollout(self):
        """
        Main rollout loop
        """
        try:
            while True:
                print(f"Rollout episode: {self.episode_idx}")
                
                # Run single episode
                episode_data = self._run_single_episode()
                
                # Save if data was collected (and not discarded)
                if episode_data is not None:
                    self.replay_buffer.add_episode(episode_data, compressors='disk')
                    self.saved_episode_idx = self.replay_buffer.n_episodes - 1
                    print(f'Saved episode {self.saved_episode_idx}')
                
                self.episode_idx += 1
                
                # Break condition for simulation (e.g. max episodes)
                if self.episode_idx >= self.cfg.train.task.env_runner.max_steps:
                    break
        finally:
            self._cleanup()

    def _run_single_episode(self) -> Optional[Dict[str, Any]]:
        """
        main loop
        """
        # reset env
        obs_dict = self.env.reset()
        self.env.render()
        assert hasattr(self, 'input_device')
        self.input_device.start_control()
        print("\n=== finish initialize env and input devices ===\n")
        
        self.sim_states = []    # for rewind
        self.episode_buffers = {    # buffers
            'action': [],
            'wrist_cam': [],
            'side_cam': [],
            'state': [],
            'tcp_pose': [],
            'joint_pos': [],
            'action_mode': []
        }

        # init episode manager
        self.episode_manager.reset_observation_history()
        sim_obs = self._get_sim_observation(obs_dict)
        for _ in range(self.To):
            self.episode_manager.update_observation(
                sim_obs['policy_side_img'],
                sim_obs['policy_wrist_img'],
                sim_obs['state']
            )
            self.sim_states.append(self._save_sim_state())

        # initialize failure detection
        if self.failure_detection_module:
            init_policy_obs = self.episode_manager.get_policy_observation()
            for key, value in init_policy_obs.items():
                init_policy_obs[key] = value[0:1]
            with torch.no_grad():
                init_latent = self.policy.extract_latent(init_policy_obs).reshape(-1)
            self.failure_detection_module.process_step({
                'step_type': 'episode_start',
                'episode_idx': self.episode_idx,
                'rollout_init_latent': init_latent.unsqueeze(0)
            })

        self.j = 0 # Episode timestep
        
        while True:
            if self.j >= self.max_episode_length:
                print("\n[INFO] Maximum episode length reached.")
                print("Options: [F]inish Success, [D]iscard, [H]uman Intervention (to fix end state)")

                self.keyboard.reset()
                while not self.keyboard.discard and not self.keyboard.finish and not self.keyboard.help and not self.keyboard.rst:
                    time.sleep(0.1)
                    self._safe_render()
                    if self.keyboard.discard: break

                if self.keyboard.rst: return None
                if self.keyboard.discard: 
                    print("[INFO] Discarding episode.")
                    return None
                if self.keyboard.finish:
                    print("[INFO] Finishing episode.")
                    break
            
            if not self.keyboard.help:
                self._run_policy_inference_loop()

            if self.keyboard.help:
                self._run_human_intervention()
                if self.keyboard.back:
                    self.keyboard.back = False
            
            if self.keyboard.discard:
                print("[INFO] Discarding episode.")
                return None
            
            if self.keyboard.finish:
                print("[INFO] Finishing episode early.")
                break
                
        # finalize
        if self.keyboard.finish:
            return self._finalize_episode_data()
            
        return None
    
    def _run_policy_inference_loop(self):
        """
        Run policy inference loop
        """
        print("=========== Policy inference ============")
        
        while not self.keyboard.finish and not self.keyboard.discard and not self.keyboard.help:
            policy_obs = self.episode_manager.get_policy_observation()
            
            with torch.no_grad():
                if self.failure_detection_module:
                    curr_action, curr_latent = self.policy.predict_action(policy_obs, return_latent=True)
                else:
                    curr_action = self.policy.predict_action(policy_obs)
                    curr_latent = None
            
            np_action_dict = dict_apply(curr_action, lambda x: x.detach().to('cpu').numpy())
            action_seq = np_action_dict['action'][0] # [Ta, ActionDim]
            
            sim_obs = None
            for step in range(self.Ta):
                self.sim_states.append(self._save_sim_state())

                raw_action = action_seq[step]
                
                # step
                obs_dict, reward, done, info = self.env.step(raw_action)
                self.env.render()
                sim_obs = self._get_sim_observation(obs_dict)
                
                # update manager & buffers
                self.episode_manager.update_observation(
                    sim_obs['policy_side_img'],
                    sim_obs['policy_wrist_img'],
                    sim_obs['state']
                )
                self._append_to_buffer(raw_action, sim_obs, ROBOT)
                self.j += 1

                # Check Pause
                if self.keyboard.pause:
                    print("\n[PAUSED] Press [P] again to resume...")
                    self.keyboard.pause = False
                    time.sleep(0.3)
                    while not self.keyboard.pause and not self.keyboard.rst and not self.keyboard.help:
                        self.env.render()
                        time.sleep(0.05)
                    
                    if self.keyboard.pause:
                        print("[RESUMED]")
                        self.keyboard.pause = False
                        time.sleep(0.3)
                    elif self.keyboard.help or self.keyboard.rst:
                        break 

                # Proactive Human Intervention
                if self.keyboard.help or self.keyboard.rst or self.keyboard.finish or self.keyboard.discard:
                    break
                
                if done or self.j >= self.max_episode_length:
                    break

            if self.keyboard.help or self.keyboard.rst or self.keyboard.finish or self.keyboard.discard:
                break

            if self.keyboard.pause:
                print("Pause. Press [P] again to continue")
                while True:
                    if not self.keyboard.pause:
                        break

            # failure detection
            if self.failure_detection_module:
                step_data = {
                    'step_type': 'policy_step',
                    'curr_latent': curr_latent,
                    'timestep': self.j,
                    'robot_state': sim_obs 
                }
                self.failure_detection_module.process_step(step_data)
                
                # call background thread, submit task, get results from queue, which is fast
                failure_flag, failure_reason, _ = self.failure_detection_module.detect_failure(
                    timestep=self.j,
                    max_episode_length=self.max_episode_length
                )
                
                if failure_flag or self.j >= self.max_episode_length:
                    if failure_flag:
                        print(f"[FAILURE] Step {self.j}: {failure_reason}")
                    else:
                        print("[INFO] Max length reached.")
                        
                    print("Options: [H]elp, [D]iscard, [F]inish, [R]eset")
                    self.keyboard.reset() # clear stale flags
                    while not self.keyboard.ctn and not self.keyboard.discard and \
                        not self.keyboard.help and not self.keyboard.finish and not self.keyboard.rst and not self.keyboard.pause:
                        time.sleep(0.1)
                        self.env.render()
                    
                    if self.keyboard.ctn:
                        print("Continuing policy...")
                        self.keyboard.ctn = False  # Reset continue flag
                        if self.j >= self.max_episode_length:
                            print("Max length reached, forcing help.")
                            self.keyboard.help = True

            elif self.j >= self.max_episode_length:
                print("Max length reached.")
                self.keyboard.help = True
                break

    def _run_human_intervention(self):
        """
        main logic for human intervention
        """

        def is_intervening(ac_dict, threshold=0.10):
            if ac_dict is None:
                return False
            total = 0.0
            for k, v in ac_dict.items():
                if isinstance(v, np.ndarray) and ("delta" in k):
                    total += float(np.linalg.norm(v))
            return total > float(threshold)

        def get_input_type(robot, arm):
            try:
                from robosuite.controllers.composite.composite_controller import WholeBody
                if hasattr(robot, "composite_controller") and isinstance(robot.composite_controller, WholeBody):
                    return robot.composite_controller.joint_action_policy.input_type
            except Exception:
                pass
            try:
                return robot.part_controllers[arm].input_type
            except Exception:
                return "delta"

        def build_env_action(active_robot_idx, action_dict_for_active, all_prev_gripper_actions):
            env_action_list = []
            for i, robot in enumerate(self.env.robots):
                if i == active_robot_idx:
                    env_action_list.append(robot.create_action_vector(action_dict_for_active))
                else:
                    env_action_list.append(robot.create_action_vector(all_prev_gripper_actions[i]))
            return np.concatenate(env_action_list)

        print("============ Human intervention =============")
        self.keyboard.help = False

        # Rewind first
        self._rewind_robot()

        # Start / reset input device if available (critical for stable teleop)
        if hasattr(self.input_device, "start_control"):
            try:
                self.input_device.start_control()
            except Exception:
                pass
        if hasattr(self.input_device, "reset"):
            try:
                self.input_device.reset()
            except Exception:
                pass

        # Maintain gripper hold state across robots
        all_prev_gripper_actions = []
        for robot in self.env.robots:
            gdict = {}
            for arm in robot.arms:
                gkey = f"{arm}_gripper"
                gdict[gkey] = 0.0
            all_prev_gripper_actions.append(gdict)

        # If we have previous action in buffer, reuse gripper values for active robot
        if len(self.episode_buffers.get("action", [])) > 0:
            last_action = self.episode_buffers["action"][-1]
            idx = 0
            for r_i, robot in enumerate(self.env.robots):
                rdim = robot.action_dim
                ra = last_action[idx:idx + rdim]
                idx += rdim
                try:
                    ra_dict = robot.create_action_dict(ra)
                    for arm in robot.arms:
                        gkey = f"{arm}_gripper"
                        if gkey in ra_dict:
                            all_prev_gripper_actions[r_i][gkey] = float(ra_dict[gkey])
                except Exception:
                    pass

        print("Controls: Press [B]ack to finish/resume policy")

        resume_policy = False
        last_hold_action_dict = None  # for absolute controllers hold

        # human intervention loop
        while True:
            if self.keyboard.back:
                break
            if self.keyboard.discard:
                break

            active_idx = int(self.input_device.active_robot)
            active_robot = self.env.robots[active_idx]

            # Read device
            input_ac_dict = self.input_device.input2action(goal_update_mode="target")

            # Finish signal: we may optionally pad to chunk boundary
            if self.keyboard.finish:
                resume_policy = True

            # If no device output: do not step, just render and (if requested) pad later
            if input_ac_dict is None:
                # Optional padding to chunk boundary after finish
                if resume_policy and (self.j % self.Ta != 0) and (last_hold_action_dict is not None):
                    # Build a hold action that is safe:
                    hold_dict = deepcopy(last_hold_action_dict)
                    for arm in active_robot.arms:
                        itype = get_input_type(active_robot, arm)
                        if itype == "delta":
                            if isinstance(hold_dict.get(arm, None), np.ndarray):
                                hold_dict[arm] = np.zeros_like(hold_dict[arm])
                            else:
                                hold_dict[arm] = np.zeros(6, dtype=np.float32)
                    env_action = build_env_action(active_idx, hold_dict, all_prev_gripper_actions)

                    self.sim_states.append(self._save_sim_state())
                    obs_dict, reward, done, info = self.env.step(env_action)
                    self.env.render()

                    sim_obs = self._get_sim_observation(obs_dict)
                    self.episode_manager.update_observation(
                        sim_obs["policy_side_img"],
                        sim_obs["policy_wrist_img"],
                        sim_obs["state"],
                    )
                    self._append_to_buffer(env_action, sim_obs, HUMAN)
                    self.j += 1

                    if done:
                        self.keyboard.finish = True
                        break
                    continue

                self.env.render()
                if resume_policy and (self.j % self.Ta == 0):
                    break
                continue

            # Only step when user is actually intervening (prevents drift / noise causing motion)
            if not is_intervening(input_ac_dict, threshold=0.10):
                self.env.render()
                if resume_policy and (self.j % self.Ta == 0):
                    break
                continue

            # Build action_dict for active robot following demo logic
            action_dict = deepcopy(input_ac_dict)

            for arm in active_robot.arms:
                itype = get_input_type(active_robot, arm)
                if itype == "delta":
                    if f"{arm}_delta" in input_ac_dict:
                        action_dict[arm] = input_ac_dict[f"{arm}_delta"]
                    elif arm in input_ac_dict:
                        action_dict[arm] = input_ac_dict[arm]
                elif itype == "absolute":
                    if f"{arm}_abs" in input_ac_dict:
                        action_dict[arm] = input_ac_dict[f"{arm}_abs"]
                    elif arm in input_ac_dict:
                        action_dict[arm] = input_ac_dict[arm]

            # Update gripper hold state for active robot
            for arm in active_robot.arms:
                gkey = f"{arm}_gripper"
                if gkey in action_dict:
                    all_prev_gripper_actions[active_idx][gkey] = float(action_dict[gkey])

            # Cache last action_dict for safe holding / padding later
            last_hold_action_dict = deepcopy(action_dict)

            env_action = build_env_action(active_idx, action_dict, all_prev_gripper_actions)

            # Save state exactly once per step (align sim_states with step count)
            self.sim_states.append(self._save_sim_state())

            obs_dict, reward, done, info = self.env.step(env_action)
            self.env.render()

            sim_obs = self._get_sim_observation(obs_dict)

            self.episode_manager.update_observation(
                sim_obs["policy_side_img"],
                sim_obs["policy_wrist_img"],
                sim_obs["state"],
            )

            self._append_to_buffer(env_action, sim_obs, HUMAN)
            self.j += 1

            if done:
                self.keyboard.finish = True
                break

            # If user pressed finish and we're on chunk boundary, exit intervention
            if resume_policy and (self.j % self.Ta == 0):
                break
    
    def _rewind_robot(self):
        """
        Rewind robot state
        """
        print("[INFO] Rewinding...")
        
        # Decide how many steps
        steps_to_rewind = 0
        if self.failure_detection_module and hasattr(self.failure_detection_module, 'should_stop_rewinding'):
            curr_timestep = self.j
            for _ in range(curr_timestep):
                # check if should stop rewinding
                if not self.failure_detection_module.rewind_step(self.j, self.episode_buffers, curr_timestep):
                    break
                self._rewind_single_step()
        else:
            # simple rewind (default 3 chunks)
            steps_to_rewind = 3 * self.Ta
            self._rewind_simple(steps_to_rewind)
            
        print(f"Rewound to step {self.j}")

    def _rewind_simple(self, max_steps):
        """
        Simple rewind implementation
        """
        count = 0
        while self.j > 0 and count < max_steps:
             self._rewind_single_step()
             count += 1

    def _rewind_single_step(self):
        """
        Rewind one step using Sim State
        """
        if self.j <= 0: return

        # pop buffers (remove data for step j-1)
        for key in self.episode_buffers:
            self.episode_buffers[key].pop()
        
        target_state = self.sim_states.pop()
        
        self.j -= 1
        
        # Restore Sim
        self.env.sim.set_state_from_flattened(target_state)
        self.env.sim.forward()

        self.env.sim.data.qvel[:] = 0
        self.env.sim.forward()
        self.env.render()
        
        self.episode_manager.reset_observation_history()
        obs_dict = self.env._get_observations()
        sim_obs = self._get_sim_observation(obs_dict)
        for _ in range(self.To):
             self.episode_manager.update_observation(
                sim_obs['policy_side_img'],
                sim_obs['policy_wrist_img'],
                sim_obs['state']
            )

    def _append_to_buffer(self, action, sim_obs, mode):
        """
        Helper to append data to buffers
        """
        self.episode_buffers['action'].append(action)
        self.episode_buffers['wrist_cam'].append(sim_obs['raw_obs']['robot0_eye_in_hand_image'])
        self.episode_buffers['side_cam'].append(sim_obs['raw_obs']['agentview_image'])
        self.episode_buffers['tcp_pose'].append(sim_obs['raw_obs']['robot0_eef_pos'])
        self.episode_buffers['joint_pos'].append(sim_obs['raw_obs']['robot0_joint_pos'])
        self.episode_buffers['state'].append(sim_obs['state'].cpu().numpy())
        self.episode_buffers['action_mode'].append(mode)

    def _finalize_episode_data(self):
        """
        Package buffers into episode dict
        """
        episode = dict()
        episode['wrist_cam'] = np.stack(self.episode_buffers['wrist_cam'], axis=0)
        episode['side_cam'] = np.stack(self.episode_buffers['side_cam'], axis=0)
        episode['tcp_pose'] = np.stack(self.episode_buffers['tcp_pose'], axis=0)
        episode['joint_pos'] = np.stack(self.episode_buffers['joint_pos'], axis=0)
        episode['action'] = np.stack(self.episode_buffers['action'], axis=0)
        episode['action_mode'] = np.array(self.episode_buffers['action_mode'])
        
        assert episode['action_mode'].shape[0] % self.Ta == 0, "A Ta-step chunking is required"

        if self.failure_detection_module:
            failure_episode_data = self.failure_detection_module.finalize_episode(episode)
            episode.update(failure_episode_data)
        else:
            episode['failure_indices'] = np.zeros((episode['action_mode'].shape[0],), dtype=np.bool_)
            
        return episode

    def _cleanup(self):
        """
        Cleanup resources
        """
        save_zarr_path = os.path.join(self.cfg.save_buffer_path, 'rollout_buffer.zarr')
        if hasattr(self, 'replay_buffer'):
            self.replay_buffer.save_to_path(save_zarr_path)
            print(f"Saved replay buffer to {save_zarr_path}")
        
        if self.failure_detection_module and hasattr(self.failure_detection_module, 'cleanup'):
            self.failure_detection_module.cleanup()
        
        if hasattr(self, 'env'):
            self.env.close()