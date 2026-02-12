import os
import random
import time
import datetime
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from gymnasium import spaces
from gymnasium.vector import AsyncVectorEnv
from utils import print_args

# NOTE: Removed imports from 'utils' as we implemented a simple ReplayBuffer locally
# from utils import DictReplayBuffer, MultimodalEncoder

LOG_STD_MAX = 2
LOG_STD_MIN = -5


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = False
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "Robosuite-SAC-State"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    wandb_mode: str = "offline"
    """wandb mode: online, offline, or disabled"""
    capture_video: bool = False
    """whether to capture videos of the agent performances"""
    video_freq: int = 50
    """frequency of episodes to record video"""
    eval_freq: int = 5000
    """frequency of steps to eval"""
    max_episode_steps: int = 300
    """max steps in one episode"""
    success_terminate: bool = False
    """whether terminate this episode when success"""

    # Algorithm specific arguments
    env_id: str = "PandaLift"
    """the environment id (custom name for robosuite config)"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5000
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    ent_scale: float = 1.0
    """Scales the target entropy. < 1.0 encourages more exploration."""
    
    # Vision specific arguments (Removed or unused for state-based)
    renderer: str = "mjviewer"
    """default renderer in Robosuite"""


class ReplayBuffer:
    """A simple ReplayBuffer for flat state observations."""
    def __init__(self, buffer_size, obs_shape, action_shape, device):
        self.buffer_size = buffer_size
        self.ptr = 0
        self.size = 0
        
        self.obs = np.zeros((buffer_size, *obs_shape), dtype=np.float32)
        self.next_obs = np.zeros((buffer_size, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((buffer_size, *action_shape), dtype=np.float32)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.dones = np.zeros((buffer_size,), dtype=np.float32)
        
        self.device = device

    def add(self, obs, next_obs, action, reward, done, info):
        n_samples = obs.shape[0]
        
        indices = np.arange(self.ptr, self.ptr + n_samples) % self.buffer_size
        
        self.obs[indices] = obs
        self.next_obs[indices] = next_obs
        self.actions[indices] = action
        self.rewards[indices] = reward
        self.dones[indices] = done
        
        self.ptr = (self.ptr + n_samples) % self.buffer_size
        self.size = min(self.size + n_samples, self.buffer_size)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        
        return (
            torch.tensor(self.obs[idx], device=self.device),
            torch.tensor(self.actions[idx], device=self.device),
            torch.tensor(self.rewards[idx], device=self.device),
            torch.tensor(self.next_obs[idx], device=self.device),
            torch.tensor(self.dones[idx], device=self.device),
        )


class SingleRobosuiteEnv(gym.Env):
    def __init__(self, env_id, seed, max_episode_steps: int = 500, 
                 renderer: str = "mjviewer", img_size: int = 512, success_terminate: bool = True):
        self.max_episode_steps = max_episode_steps
        self.step_count = 0
        self.success_terminate = success_terminate

        robot = "Panda"
        task = "Lift"
        if "Nut" in env_id: task = "NutAssembly"
        
        self.env = suite.make(
            env_name=task,
            robots=robot,
            controller_configs=load_composite_controller_config(controller=None, robot=robot),
            has_renderer=False,
            has_offscreen_renderer=True,
            renderer=renderer,
            use_camera_obs=False, 
            use_object_obs=True,
            camera_names="agentview", 
            camera_heights=img_size,
            camera_widths=img_size,
            reward_shaping=True,
            control_freq=20,
            horizon=max_episode_steps,
        )
        self.image_size = img_size
        
        low, high = self.env.action_spec
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        dummy_obs = self.env.reset()
        obs_dim = self._process_obs(dummy_obs).shape[0]
        
        # Define a flat observation space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        self.metadata = {"render_modes": ["rgb_array"], "render_fps": 20}
        self.render_mode = "rgb_array"

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        
        self.step_count = 0
        obs_dict = self.env.reset()
        obs = self._process_obs(obs_dict)
        return obs, {}

    def step(self, action):
        obs_dict, reward, done, info = self.env.step(action)
        self.step_count += 1
        success = False

        if self.success_terminate:
            try:
                if self.env._check_success():
                    success = True
                    reward += 10.0
            except Exception:
                pass
            
            info["success"] = success
            terminated = success
            truncated = (self.step_count >= self.max_episode_steps) or (done and not success)

            obs = self._process_obs(obs_dict)
            return obs, reward, terminated, truncated, info

        else:
            try:
                if self.env._check_success():
                    success = True
            except:
                pass
            
            terminated = False
            truncated = (self.step_count >= self.max_episode_steps)
            
            info["success"] = success
            obs = self._process_obs(obs_dict)
            return obs, reward, terminated, truncated, info

    
    def _process_obs(self, obs_dict):
        """
        Flatten robot proprioception and object state into a single vector.
        Common keys for Lift: 
        - robot0_joint_pos_cos, robot0_joint_pos_sin, robot0_joint_vel
        - robot0_eef_pos, robot0_eef_quat, robot0_gripper_qpos, robot0_gripper_qvel
        - cube_pos, cube_quat
        """
        values = []
        robot_keys = [
            "robot0_joint_pos_cos", 
            "robot0_joint_pos_sin", 
            "robot0_joint_vel", 
            "robot0_eef_pos", 
            "robot0_eef_quat", 
            "robot0_gripper_qpos", 
            "robot0_gripper_qvel"
        ]
        object_keys = ["cube_pos", "cube_quat", "gripper_to_cube_pos"]

        for k in robot_keys:
            if k in obs_dict:
                values.append(obs_dict[k])
            else:
                # Fallback for sin/cos if raw pos is provided (though suite usually handles this)
                if k == "robot0_joint_pos_cos" and "robot0_joint_pos" in obs_dict:
                     values.append(np.cos(obs_dict["robot0_joint_pos"]))
                elif k == "robot0_joint_pos_sin" and "robot0_joint_pos" in obs_dict:
                     values.append(np.sin(obs_dict["robot0_joint_pos"]))

        for k in object_keys:
            if k in obs_dict:
                values.append(obs_dict[k])

        if len(values) == 0:
             for k, v in obs_dict.items():
                 if isinstance(v, (np.ndarray, list)) and len(np.array(v).shape) == 1:
                     values.append(v)

        return np.concatenate(values)
    
    def render(self):
        return np.flipud(self.env.sim.render(
            camera_name="agentview",
            height=self.image_size,
            width=self.image_size,
            depth=False
        ))

    def close(self):
        self.env.close()


def make_env(env_id, seed, idx, capture_video, run_name, args: Args, output_path: str):
    def thunk():
        env = SingleRobosuiteEnv(env_id=env_id, seed=seed+idx, max_episode_steps=args.max_episode_steps, 
                                 renderer=args.renderer, success_terminate=args.success_terminate)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(
                env,
                f"{output_path}/videos", 
                episode_trigger=lambda x: x % args.video_freq == 0
            )
            
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        # Input dim: State dim + Action dim
        self.input_dim = np.prod(env.single_observation_space.shape) + np.prod(env.single_action_space.shape)

        self.fc1 = nn.Linear(self.input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.input_dim = np.prod(env.single_observation_space.shape)
        
        self.fc1 = nn.Linear(self.input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.ln1 = nn.LayerNorm(256)
        self.ln2 = nn.LayerNorm(256)
        
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
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

    def get_action(self, x):
        mean, log_std = self.forward(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)

        x_t = normal.rsample()  # reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)

        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action
    
    def get_eval_action(self, x):
        mean, _ = self.forward(x)
        action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action


if __name__ == "__main__":
    args = tyro.cli(Args)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.env_id}_{args.exp_name}_{args.seed}_{timestamp}"
    print_args(args)

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
            mode=args.wandb_mode,
        )

    output_path = f"outputs/{run_name}"
    os.makedirs(output_path, exist_ok=True)
    writer = SummaryWriter(output_path)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.backends.cudnn.benchmark = True

    # GPU setup
    training_gpu_id = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    # State-based training usually doesn't strictly need AMP, but keeping it is fine.
    scaler = torch.amp.GradScaler(enabled=(training_gpu_id.type == "cuda"))

    # Env setup
    envs = AsyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name, args, output_path) 
         for i in range(args.num_envs)],
        context="spawn",
        shared_memory=True
    )
    eval_envs = gym.vector.SyncVectorEnv([
        make_env(args.env_id, args.seed + 1000, 0, args.capture_video, f"{run_name}/eval", args, output_path)
    ])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(envs).to(training_gpu_id)
    qf1 = SoftQNetwork(envs).to(training_gpu_id)
    qf2 = SoftQNetwork(envs).to(training_gpu_id)
    qf1_target = SoftQNetwork(envs).to(training_gpu_id)
    qf2_target = SoftQNetwork(envs).to(training_gpu_id)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        entropy_scale = args.ent_scale
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(training_gpu_id)).item() * entropy_scale
        log_alpha = torch.zeros(1, requires_grad=True, device=training_gpu_id)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    # Simple Replay Buffer
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space.shape,
        envs.single_action_space.shape,
        training_gpu_id
    )
    
    start_time = time.time()

    # Start the game
    obs, _ = envs.reset(seed=args.seed)
    
    for global_step in range(args.total_timesteps):
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            obs_tensor = torch.tensor(obs).to(training_gpu_id)
            with torch.no_grad():
                actions, _, _ = actor.get_action(obs_tensor)
            actions = actions.cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # Logging logic
        if "episode" in infos:
            env_indices = []
            if "_episode" in infos:
                if isinstance(infos["_episode"], (np.ndarray, list)):
                    env_indices = [i for i, x in enumerate(infos["_episode"]) if x]
                elif infos["_episode"]: 
                    env_indices = [0]
            
            for i in env_indices:
                if isinstance(infos["episode"], dict):
                    ep_return = infos["episode"]["r"][i]
                    ep_length = infos["episode"]["l"][i]
                elif isinstance(infos["episode"], (list, np.ndarray)):
                    ep_return = infos["episode"][i]["r"]
                    ep_length = infos["episode"][i]["l"]
                else:
                    continue
                
                writer.add_scalar("charts/episodic_return", ep_return, global_step)
                writer.add_scalar("charts/episodic_length", ep_length, global_step)

                if "success" in infos:
                    s_val = 0.0
                    if isinstance(infos["success"], (list, np.ndarray)):
                        s_val = float(infos["success"][i])
                    else:
                        s_val = float(infos["success"])
                    writer.add_scalar("charts/success_rate", s_val, global_step)

        # Handle terminal observations
        real_next_obs = next_obs.copy()
        if "final_observation" in infos:
            for idx, final_obs in enumerate(infos["final_observation"]):
                if final_obs is not None:
                    real_next_obs[idx] = final_obs

        # Add to buffer
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        obs = next_obs

        # Training
        if global_step > args.learning_starts:
            obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = rb.sample(args.batch_size)

            with torch.amp.autocast(device_type=training_gpu_id.type):
                with torch.no_grad():
                    next_state_actions, next_state_log_pi, _ = actor.get_action(next_obs_batch)
                    qf1_next_target = qf1_target(next_obs_batch, next_state_actions)
                    qf2_next_target = qf2_target(next_obs_batch, next_state_actions)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                    next_q_value = rew_batch.view(-1) + (1 - done_batch.view(-1)) * args.gamma * (min_qf_next_target).view(-1)

                qf1_a_values = qf1(obs_batch, act_batch).view(-1)
                qf2_a_values = qf2(obs_batch, act_batch).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            scaler.scale(qf_loss).backward()
            scaler.step(q_optimizer)
            scaler.update()

            if global_step % args.policy_frequency == 0:
                with torch.amp.autocast(device_type=training_gpu_id.type):
                    pi, log_pi, _ = actor.get_action(obs_batch)
                    qf1_pi = qf1(obs_batch, pi)
                    qf2_pi = qf2(obs_batch, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                actor_optimizer.zero_grad()
                scaler.scale(actor_loss).backward()
                scaler.step(actor_optimizer)
                scaler.update()

                if args.autotune:
                    with torch.no_grad():
                        _, log_pi, _ = actor.get_action(obs_batch)
                    alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                sps = int(global_step / (time.time() - start_time))
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                writer.add_scalar("charts/SPS", sps, global_step)
                
                print(f"[TRAIN] Step: {global_step} | SPS: {sps} | "
                      f"Q-Loss: {qf_loss.item():.3f} | Actor-Loss: {actor_loss.item():.3f} | "
                      f"Alpha: {alpha:.3f}")

            # Evaluation
            if global_step > 0 and global_step % args.eval_freq == 0:
                print(f"\n[EVAL] Starting Evaluation at Step {global_step}")
                actor.eval()
                eval_episodes = 20
                eval_returns = []
                eval_successes = []
                
                # Simple sequential eval loop for clarity
                for _ in range(eval_episodes):
                    eval_obs, _ = eval_envs.reset()
                    done = False
                    while not done:
                        with torch.no_grad():
                            eval_obs_tensor = torch.tensor(eval_obs).to(training_gpu_id)
                            eval_action = actor.get_eval_action(eval_obs_tensor)
                            eval_action = eval_action.cpu().numpy()
                        
                        eval_obs, _, eval_terminateds, eval_truncateds, eval_infos = eval_envs.step(eval_action)
                        
                        if "_episode" in eval_infos and eval_infos["_episode"][0]:
                            r = eval_infos["episode"]["r"][0]
                            s = eval_infos.get("success", [0])[0]
                            print(f"  > Episode finished. Return: {float(r):.2f} | Success: {bool(s)}")
                            eval_returns.append(r)
                            eval_successes.append(s)
                            done = True
                
                avg_ret = np.mean(eval_returns) if eval_returns else 0.0
                avg_suc = np.mean(eval_successes) if eval_successes else 0.0
                print(f"[EVAL] Mean Return: {avg_ret:.2f} | Success Rate: {avg_suc:.2f}\n")
                writer.add_scalar("charts/eval_return", avg_ret, global_step)
                writer.add_scalar("charts/eval_success_rate", avg_suc, global_step)

                torch.save(actor.state_dict(), f"{output_path}/actor_{global_step}.pth")
                actor.train()

        elif global_step % 500 == 0:
            print(f"[WARMUP] Collecting Data and warmup: {global_step}/{args.learning_starts} steps")

    envs.close()
    writer.close()