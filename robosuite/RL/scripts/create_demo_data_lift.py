import os
import time
import argparse
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from gymnasium import spaces
import gymnasium as gym

LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env_obs_dim, env_action_dim, env_action_space):
        super().__init__()
        self.input_dim = env_obs_dim
        
        self.fc1 = nn.Linear(self.input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, env_action_dim)
        self.fc_logstd = nn.Linear(256, env_action_dim)
        self.ln1 = nn.LayerNorm(256)
        self.ln2 = nn.LayerNorm(256)
        
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env_action_space.high - env_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env_action_space.high + env_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.gelu(self.ln1(self.fc1(x)))
        x = F.gelu(self.ln2(self.fc2(x)))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)

        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_eval_action(self, x):
        mean, _ = self.forward(x)
        action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action


class DataCollectionEnv(gym.Env):
    def __init__(self, env_id, img_size=224, render=False):
        self.max_episode_steps = 300 # Keep consistent with eval config
        self.step_count = 0
        self.image_size = img_size
        
        robot = "Panda"
        task = "Lift"
        if "Nut" in env_id: task = "NutAssembly"

        # Config for offscreen rendering (for data collection)
        self.env = suite.make(
            env_name=task,
            robots=robot,
            controller_configs=load_composite_controller_config(controller=None, robot=robot),
            has_renderer=render,          # On-screen rendering (optional)
            has_offscreen_renderer=True,  # Crucial for saving images
            renderer="mjviewer",
            use_camera_obs=True,          # Enable cameras to get images
            use_object_obs=True,
            camera_names=["agentview", "robot0_eye_in_hand"],
            camera_heights=img_size,
            camera_widths=img_size,
            reward_shaping=True,
            control_freq=20,
            horizon=self.max_episode_steps,
        )
        
        low, high = self.env.action_spec
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Calculate observation dim for Actor (State only)
        dummy_obs = self.env.reset()
        state_vec = self._process_state_for_actor(dummy_obs)
        self.obs_dim = state_vec.shape[0]
        self.action_dim = self.action_space.shape[0]

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.step_count = 0
        obs_dict = self.env.reset()
        return obs_dict

    def step(self, action):
        obs_dict, reward, done, info = self.env.step(action)
        self.step_count += 1
        
        success = False
        try:
            if self.env._check_success():
                success = True
        except:
            pass
        info["success"] = success
        
        # Standard gym termination
        terminated = False 
        truncated = (self.step_count >= self.max_episode_steps)
        
        return obs_dict, reward, terminated, truncated, info

    def _process_state_for_actor(self, obs_dict):
        """
        Extract ONLY the proprioception and object state keys to match 
        the trained Actor's input format. Ignores images.
        """
        values = []
        robot_keys = [
            "robot0_joint_pos_cos", "robot0_joint_pos_sin", "robot0_joint_vel", 
            "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "robot0_gripper_qvel"
        ]
        object_keys = ["cube_pos", "cube_quat", "gripper_to_cube_pos"]

        for k in robot_keys:
            if k in obs_dict:
                values.append(obs_dict[k])
            else:
                if k == "robot0_joint_pos_cos" and "robot0_joint_pos" in obs_dict:
                     values.append(np.cos(obs_dict["robot0_joint_pos"]))
                elif k == "robot0_joint_pos_sin" and "robot0_joint_pos" in obs_dict:
                     values.append(np.sin(obs_dict["robot0_joint_pos"]))

        for k in object_keys:
            if k in obs_dict:
                values.append(obs_dict[k])

        # Fallback if specific keys missing (safety)
        if len(values) == 0:
             for k, v in obs_dict.items():
                 # Only take 1D arrays (states), ignore 3D arrays (images)
                 if isinstance(v, (np.ndarray, list)) and len(np.array(v).shape) == 1:
                     values.append(v)

        return np.concatenate(values).astype(np.float32)

    def get_images(self, obs_dict):
        """Extract images from obs_dict based on camera names"""
        # Robosuite images are flipped upside down by default in raw obs
        # Usually need to flip them for normal viewing, but standard datasets often keep raw.
        # Here we flip them to be human-readable/standard CV2 format.
        
        agentview = obs_dict["agentview_image"]
        eye_in_hand = obs_dict["robot0_eye_in_hand_image"]
        
        # Robosuite returns images as (H, W, C)
        # We ensure they are uint8 [0, 255]
        agentview = np.flipud(agentview)
        eye_in_hand = np.flipud(eye_in_hand)
        
        return agentview, eye_in_hand


def collect_data(args):
    env = DataCollectionEnv(
        env_id="PandaLift", 
        img_size=args.img_size,
        render=args.render_onscreen
    )
    
    # Setup Device & Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    actor = Actor(env.obs_dim, env.action_dim, env.action_space).to(device)
    
    # Load Checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}")
    
    print(f"Loading actor from {args.checkpoint}...")
    state_dict = torch.load(args.checkpoint, map_location=device)
    actor.load_state_dict(state_dict)
    actor.eval() # Set to eval mode

    # HDF5 File Setup
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    f = h5py.File(args.output, "w")
    grp = f.create_group("data")
    
    # Metadata
    f.attrs["env_name"] = "PandaLift"
    f.attrs["total_episodes"] = 0
    
    collected_episodes = 0
    total_attempts = 0

    print(f"Starting data collection. Target: {args.num_episodes} successful episodes.")

    while collected_episodes < args.num_episodes:
        total_attempts += 1
        obs_dict = env.reset(seed=total_attempts * 100) # Varied seeds
        done = False
        
        # Temporary buffers for the current episode
        ep_states = []
        ep_actions = []
        ep_rewards = []
        ep_dones = []
        ep_img_agent = []
        ep_img_wrist = []
        
        success = False
        
        while not done:
            # Prepare state for Actor
            state_vec = env._process_state_for_actor(obs_dict)
            state_tensor = torch.tensor(state_vec, device=device).unsqueeze(0) # Batch dim
            
            # Get Action (Deterministic)
            with torch.no_grad():
                action = actor.get_eval_action(state_tensor)
                action = action.cpu().numpy()[0]
            
            # Step Env
            next_obs_dict, reward, terminated, truncated, info = env.step(action)
            
            # Extract Images
            img_agent, img_wrist = env.get_images(obs_dict)
            
            # Store Step Data
            ep_states.append(state_vec)
            ep_actions.append(action)
            ep_rewards.append(reward)
            ep_dones.append(terminated or truncated)
            ep_img_agent.append(img_agent)
            ep_img_wrist.append(img_wrist)
            
            obs_dict = next_obs_dict
            
            if terminated or truncated:
                done = True
                if info.get("success", False):
                    success = True

        # Check success logic
        if success:
            print(f"Episode {total_attempts} Success! ({collected_episodes + 1}/{args.num_episodes}) len: {len(ep_states)}")
            
            # Create a group for this demo
            demo_grp = grp.create_group(f"demo_{collected_episodes}")
            
            # Save Datasets
            demo_grp.create_dataset("states", data=np.array(ep_states, dtype=np.float32))
            demo_grp.create_dataset("actions", data=np.array(ep_actions, dtype=np.float32))
            demo_grp.create_dataset("rewards", data=np.array(ep_rewards, dtype=np.float32))
            demo_grp.create_dataset("dones", data=np.array(ep_dones, dtype=bool))
            
            # Compress images to save space (lzf is fast, gzip is small)
            demo_grp.create_dataset("agentview_image", data=np.array(ep_img_agent, dtype=np.uint8), compression="lzf")
            demo_grp.create_dataset("robot0_eye_in_hand_image", data=np.array(ep_img_wrist, dtype=np.uint8), compression="lzf")
            
            # Add some attributes
            demo_grp.attrs["num_samples"] = len(ep_states)
            
            collected_episodes += 1
        else:
            print(f"Episode {total_attempts} Failed. Discarding.")

    f.attrs["total_episodes"] = collected_episodes
    f.close()
    print(f"Data collection complete. Saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .pth actor file")
    parser.add_argument("--output", type=str, default="data/lift_success_data.hdf5", help="Output HDF5 file path")
    parser.add_argument("--num_episodes", type=int, default=50, help="Number of successful episodes to collect")
    parser.add_argument("--img_size", type=int, default=224, help="Size of images to render")
    parser.add_argument("--render_onscreen", action="store_true", help="Visualize execution on screen (slower)")
    
    args = parser.parse_args()
    
    collect_data(args)