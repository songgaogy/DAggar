# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
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
import torchvision.models as models
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.buffers import ReplayBuffer

import robosuite as suite
from robosuite.wrappers.gym_wrapper import GymWrapper
from robosuite.controllers import load_composite_controller_config
from gymnasium import spaces
from gymnasium.vector import AsyncVectorEnv


def get_render_gpu_device_id(prefer_last=True):
    """
    Automatically choose a valid render GPU id
    after CUDA_VISIBLE_DEVICES remapping.
    """
    if not torch.cuda.is_available():
        return None

    visible_count = torch.cuda.device_count()
    assert visible_count > 0

    if prefer_last:
        return visible_count - 1   # e.g. use cuda:1 if CVD=2,3
    else:
        return 0                   # always use cuda:0


class DictReplayBuffer:
    def __init__(self, buffer_size, observation_space, action_space, device, n_envs=1):
        self.buffer_size = buffer_size
        self.n_envs = n_envs
        self.device = device
        
        self.keys = observation_space.keys()
        self.obs_shapes = {k: observation_space[k].shape for k in self.keys}

        self.observations = {k: np.zeros((buffer_size, n_envs) + shape, dtype=observation_space[k].dtype) 
                             for k, shape in self.obs_shapes.items()}
        self.next_observations = {k: np.zeros((buffer_size, n_envs) + shape, dtype=observation_space[k].dtype) 
                                  for k, shape in self.obs_shapes.items()}
        
        self.actions = np.zeros((buffer_size, n_envs) + action_space.shape, dtype=action_space.dtype)
        self.rewards = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.dones = np.zeros((buffer_size, n_envs), dtype=np.float32)
        
        self.pos = 0
        self.full = False

    def add(self, obs, next_obs, actions, rewards, dones, infos):
        for k in self.keys:
            self.observations[k][self.pos] = obs[k]
            self.next_observations[k][self.pos] = next_obs[k]
            
        self.actions[self.pos] = actions
        self.rewards[self.pos] = rewards
        self.dones[self.pos] = dones
        
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size):
        idx = np.random.randint(0, self.buffer_size if self.full else self.pos, batch_size)
        env_indices = np.random.randint(0, self.n_envs, batch_size)

        obs_batch = {}
        next_obs_batch = {}
        for k in self.keys:
            obs_batch[k] = torch.tensor(self.observations[k][idx, env_indices], device=self.device)
            next_obs_batch[k] = torch.tensor(self.next_observations[k][idx, env_indices], device=self.device)

        actions = torch.tensor(self.actions[idx, env_indices], device=self.device)
        rewards = torch.tensor(self.rewards[idx, env_indices], device=self.device)
        dones = torch.tensor(self.dones[idx, env_indices], device=self.device)

        return obs_batch, actions, rewards, next_obs_batch, dones


class SingleRobosuiteEnv(gym.Env):
    def __init__(self, env_id, seed, image_size: int = 224, max_episode_steps: int = 500):
        self.image_size = image_size
        self.max_episode_steps = max_episode_steps
        self.step_count = 0

        robot = "Panda"
        task = "Lift"
        if "Nut" in env_id: task = "NutAssembly"
        
        self.env = suite.make(
            env_name=task,
            robots=robot,
            controller_configs=load_composite_controller_config(controller=None, robot=robot),
            has_renderer=False,
            has_offscreen_renderer=True,
            renderer=args.renderer,
            use_camera_obs=True,
            use_object_obs=False,
            camera_names=["agentview", "robot0_eye_in_hand"],
            camera_heights=args.image_size,
            camera_widths=args.image_size,
            reward_shaping=True,  # Ensure dense rewards
            control_freq=20,
            horizon=max_episode_steps,
        )
        
        low, high = self.env.action_spec
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        dummy_obs = self.env.reset()
        proprio_dim = self._extract_proprio(dummy_obs).shape[0]
        
        self.observation_space = spaces.Dict({
            "visual": spaces.Box(low=0, high=255, shape=(6, image_size, image_size), dtype=np.uint8),
            "proprio": spaces.Box(low=-np.inf, high=np.inf, shape=(proprio_dim,), dtype=np.float32)
        })
        
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

        try:
            # Robosuite tasks usually have this method
            if self.env._check_success():
                success = True
        except Exception:
            pass
        
        info["success"] = success
        terminated = success
        truncated = (self.step_count >= self.max_episode_steps) or (done and not success)

        obs = self._process_obs(obs_dict)
        return obs, reward, terminated, truncated, info
    
    def _extract_proprio(self, obs_dict):
        """extract all proprior from Robosuite Dict than concat"""
        keys = [
            "robot0_joint_pos_cos", 
            "robot0_joint_pos_sin", 
            "robot0_joint_vel", 
            "robot0_eef_pos", 
            "robot0_eef_quat", 
            "robot0_gripper_qpos", 
            "robot0_gripper_qvel"
        ]
        values = []
        for k in keys:
            if k in obs_dict:
                values.append(obs_dict[k])
            else:
                if k == "robot0_joint_pos_cos" and "robot0_joint_pos" in obs_dict:
                     values.append(np.cos(obs_dict["robot0_joint_pos"]))
                elif k == "robot0_joint_pos_sin" and "robot0_joint_pos" in obs_dict:
                     values.append(np.sin(obs_dict["robot0_joint_pos"]))
        
        return np.concatenate(values)

    def _process_obs(self, obs_dict):
        img_global = obs_dict.get("agentview_image")
        img_wrist = obs_dict.get("robot0_eye_in_hand_image")

        if img_global is None: img_global = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        if img_wrist is None: img_wrist = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        t_global = np.transpose(img_global, (2, 0, 1))
        t_wrist = np.transpose(img_wrist, (2, 0, 1))
        visual = np.concatenate([t_global, t_wrist], axis=0)

        proprio = self._extract_proprio(obs_dict)

        return {"visual": visual, "proprio": proprio}

    def render(self):
        return np.flipud(self.env.sim.render(
            camera_name="agentview",
            height=self.image_size,
            width=self.image_size,
            depth=False
        ))

    def close(self):
        self.env.close()


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 42
    """seed of the experiment"""
    torch_deterministic: bool = False
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "Robosuite-SAC"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    wandb_mode: str = "offline"
    """wandb mode: online, offline, or disabled"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    video_freq: int = 10
    """frequency of episodes to record video"""
    eval_freq: int = 100
    """frequency of steps to eval"""
    max_episode_steps: int = 250
    """max steps in one episode"""

    # Algorithm specific arguments
    env_id: str = "PandaLift"
    """the environment id (custom name for robosuite config)"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(1e5)
    """the replay memory buffer size (reduced slightly for image memory safety)"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5000
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    
    # Vision specific arguments
    renderer: str = "mjviewer"
    """default renderer in Robosuite"""
    image_size: int = 224
    """Input image size (H, W)"""
    pretrained_path: Optional[str] = None
    """Path to the manually downloaded resnet weights (e.g., ./resnet18.pth)"""


def make_env(env_id, seed, idx, capture_video, run_name, args: Args, output_path: str):
    def thunk():
        env = SingleRobosuiteEnv(env_id=env_id, seed=seed+idx, image_size=args.image_size, 
                                 max_episode_steps=args.max_episode_steps)
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


class MultimodalEncoder(nn.Module):
    """
    Encodes a (B, 6, H, W) image by splitting it into two (B, 3, H, W) images,
    passing both through a shared ResNet backbone, and concatenating the features.
    """
    def __init__(self, observation_space, pretrained_path=None, img_num: int = 2):
        super().__init__()
        # augmentation
        self.aug = RandomShiftsAug(pad=4)
        # pretrained encoder
        self.backbone = models.resnet18(weights=None)
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"Loading pretrained weights from {pretrained_path}")
            state_dict = torch.load(pretrained_path)
            self.backbone.load_state_dict(state_dict, strict=False)
        else:
            print("No pretrained path provided or file not found. Using Random/Default initialization.")

        self.features = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4, 
        )

        self.out_channels = 512
        self.spatial_softmax = SpatialSoftmax(self.out_channels)
        self.visual_dim = self.out_channels * 2 * img_num

        self.proprio_dim = observation_space["proprio"].shape[0]

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.repr_dim = self.visual_dim + self.proprio_dim

    def forward(self, x_dict):
        x_vis = x_dict["visual"] / 255.0
        
        global_view, wrist_view = x_vis[:, 0:3], x_vis[:, 3:6]
        combined = torch.cat([global_view, wrist_view], dim=0)
        combined = self.aug(combined)
        combined = (combined - self.mean) / self.std
        
        feat_map = self.features(combined)
        feat_points = self.spatial_softmax(feat_map)
        
        batch_size = x_vis.shape[0]
        visual_feat = torch.cat([feat_points[:batch_size], feat_points[batch_size:]], dim=1)
        
        proprio_feat = x_dict["proprio"]    # (B, Dim)
        return torch.cat([visual_feat, proprio_feat], dim=1)


class RandomShiftsAug(nn.Module):
    def __init__(self, pad=4):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        # x: (B, C, H, W)
        if not self.training:
            return x
            
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)


class SpatialSoftmax(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features

    def forward(self, x):
        # x: (B, C, H, W)
        N, C, H, W = x.shape
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1., 1., H, device=x.device),
            torch.linspace(-1., 1., W, device=x.device),
            indexing='ij'
        )
        pos_x = pos_x.reshape(H * W)
        pos_y = pos_y.reshape(H * W)

        x = x.reshape(N, C, H * W)
        softmax_attention = F.softmax(x, dim=-1) # (N, C, H*W)

        expected_x = torch.sum(pos_x * softmax_attention, dim=2, keepdim=True)
        expected_y = torch.sum(pos_y * softmax_attention, dim=2, keepdim=True)
        
        expected_xy = torch.cat([expected_x, expected_y], dim=2)
        return expected_xy.reshape(N, C * 2)


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env, pretrained_path=None, ebd_size: int = 512, img_nums: int = 2):
        super().__init__()
        self.encoder = MultimodalEncoder(env.single_observation_space, pretrained_path=pretrained_path, img_num=img_nums)
        self.input_dim = self.encoder.repr_dim + np.prod(env.single_action_space.shape)

        self.fc1 = nn.Linear(self.input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = self.encoder(x)
        x = torch.cat([x, a], 1)
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env, pretrained_path=None, ebd_size: int = 512, img_nums: int = 2):
        super().__init__()
        self.encoder = MultimodalEncoder(env.single_observation_space, pretrained_path=pretrained_path, img_num=img_nums)
        self.input_dim = self.encoder.repr_dim
        
        self.fc1 = nn.Linear(self.input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        
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
        x = self.encoder(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)

        LOG_STD_MAX = 2
        LOG_STD_MIN = -5
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)

        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)

        # enforcing action bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
    
    def get_eval_action(self, x):
        mean, _ = self(x)
        action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action


if __name__ == "__main__":
    args = tyro.cli(Args)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.env_id}_{args.exp_name}_{args.seed}_{timestamp}"
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
    writer = SummaryWriter(output_path)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.backends.cudnn.benchmark = True

    # gpu setup
    training_gpu_id = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # envs setup
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

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs, pretrained_path=args.pretrained_path).to(training_gpu_id)
    qf1 = SoftQNetwork(envs, pretrained_path=args.pretrained_path).to(training_gpu_id)
    qf2 = SoftQNetwork(envs, pretrained_path=args.pretrained_path).to(training_gpu_id)
    qf1_target = SoftQNetwork(envs, pretrained_path=args.pretrained_path).to(training_gpu_id)
    qf2_target = SoftQNetwork(envs, pretrained_path=args.pretrained_path).to(training_gpu_id)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(training_gpu_id)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=training_gpu_id)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32

    # NOTE(gaoyuan): changed
    rb = DictReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        training_gpu_id,
        n_envs=args.num_envs,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            # Dict[numpy.ndarray] -> Dict[torch.Tensor]
            obs_tensor = {k: torch.tensor(v).to(training_gpu_id) for k, v in obs.items()}
            with torch.no_grad():
                actions, _, _ = actor.get_action(obs_tensor)
            actions = actions.cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # NOTE(gaoyuan) modified
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

        # NOTE(gaoyuan) modified
        real_next_obs = {k: v.copy() for k, v in next_obs.items()}
        
        for idx, trunc in enumerate(truncations):
            if trunc:
                if "final_observation" in infos:
                    final_obs = infos["final_observation"]
                    target_obs = None

                    if isinstance(final_obs, (list, np.ndarray)):
                        target_obs = final_obs[idx]
                    elif idx == 0:
                        target_obs = final_obs
                    
                    if target_obs is not None:
                        real_next_obs["visual"][idx] = target_obs["visual"]
                        real_next_obs["proprio"][idx] = target_obs["proprio"]
                        
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = rb.sample(args.batch_size)

            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(next_obs_batch)
                qf1_next_target = qf1_target(next_obs_batch, next_state_actions)
                qf2_next_target = qf2_target(next_obs_batch, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = rew_batch.flatten() + (1 - done_batch.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(obs_batch, act_batch).view(-1)
            qf2_a_values = qf2(obs_batch, act_batch).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(obs_batch)
                    qf1_pi = qf1(obs_batch, pi)
                    qf2_pi = qf2(obs_batch, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(obs_batch)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                sps = int(global_step / (time.time() - start_time))
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
                
                print(f"[TRAIN] Step: {global_step} | SPS: {sps} | "
                      f"Q-Loss: {qf_loss.item():.3f} | Actor-Loss: {actor_loss.item():.3f} | "
                      f"Alpha: {alpha:.3f}")
                
            if global_step > 0 and global_step % args.eval_freq == 0:
                print(f"\n[EVAL] Starting Evaluation at Step {global_step}")
                actor.eval()
                
                eval_episodes = 10
                eval_returns = []
                eval_successes = []
                
                for _ in range(eval_episodes):
                    eval_obs, _ = eval_envs.reset()
                    eval_done = False
                    
                    while not eval_done:
                        with torch.no_grad():
                            eval_obs_tensor = {k: torch.tensor(v).to(training_gpu_id) for k, v in eval_obs.items()}
                            eval_action = actor.get_eval_action(eval_obs_tensor)
                            eval_action = eval_action.cpu().numpy()
                        
                        eval_obs, _, eval_terminateds, eval_truncateds, eval_infos = eval_envs.step(eval_action)

                        if "episode" in eval_infos:
                            ep_data = eval_infos["episode"]
                            is_list_structure = isinstance(ep_data, (list, np.ndarray))

                            if "_episode" in eval_infos:
                                mask = eval_infos["_episode"]
                                done_idx = -1
                                if isinstance(mask, (list, np.ndarray)):
                                    for i, m in enumerate(mask):
                                        if m: 
                                            done_idx = i
                                            break
                                elif mask:
                                    done_idx = 0

                                if done_idx != -1:
                                    if is_list_structure:
                                        if ep_data[done_idx] is not None:
                                            r = ep_data[done_idx]["r"]
                                        else:
                                            continue
                                    else:
                                        r = ep_data["r"]

                                    s = 0.0
                                    if "success" in eval_infos:
                                        s_data = eval_infos["success"]
                                        if isinstance(s_data, (list, np.ndarray)):
                                            s = float(s_data[done_idx])
                                        else:
                                            s = float(s_data)
                                    
                                    print(f"  > Episode finished. Return: {float(r):.2f} | Success: {bool(s)}")
                                    eval_returns.append(r)
                                    eval_successes.append(s)
                                    eval_done = True

                if len(eval_returns) > 0:
                    avg_return = np.mean(eval_returns)
                    avg_success = np.mean(eval_successes)
                    
                    print(f"[EVAL] Step: {global_step} | Mean Return: {avg_return:.2f} | Success Rate: {avg_success:.2f}\n")
                    writer.add_scalar("charts/eval_return", avg_return, global_step)
                    writer.add_scalar("charts/eval_success_rate", avg_success, global_step)

                torch.save(actor.state_dict(), f"{output_path}/actor_{global_step}.pth")
                actor.train()
                
        elif global_step % 500 == 0:
            print(f"[WARMUP] Collecting Data and warmup: {global_step}/{args.learning_starts} steps")

    envs.close()
    writer.close()
