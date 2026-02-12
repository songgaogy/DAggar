# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
import h5py
import glob
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
from utils import DictReplayBuffer, MultimodalEncoder, load_demonstrations

LOG_STD_MAX = 2
LOG_STD_MIN = -5


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
            if self.env._check_success():
                success = True
        except:
            pass
        
        # NOTE: we do not ternimate a episode when success
        terminated = False
        truncated = (self.step_count >= self.max_episode_steps)
        
        info["success"] = success
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
    max_episode_steps: int = 300
    """max steps in one episode"""
    pretraining_steps: int = 20000
    """using BC and offline SAC to pretrain"""

    # Algorithm specific arguments
    env_id: str = "PandaLift"
    """the environment id (custom name for robosuite config)"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(5e5)
    """the replay memory buffer size (reduced slightly for image memory safety)"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 1024
    """the batch size of sample from the reply memory"""
    learning_starts: int = 0
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
    ent_scale: float = 0.5
    """Scales the target entropy. < 1.0 encourages more exploration."""
    
    # Vision specific arguments
    renderer: str = "mjviewer"
    """default renderer in Robosuite"""
    image_size: int = 128
    """Input image size (H, W)"""

    # paths
    pretrained_path: Optional[str] = None
    """Path to the manually downloaded resnet weights (e.g., ./resnet18.pth)"""
    demo_data_path: str = None
    """Path to expert successful demonstrations (e.g., ./data | should be a directory)"""



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


def compute_target_q(actor, qf1_target, qf2_target, next_obs_batch, alpha, gamma, rew_batch, done_batch):
    with torch.no_grad():
        # DrQ style: Average over M augmentations (M=2 is standard)
        # Augmentation 1
        next_actions_1, next_log_pi_1, _ = actor.get_action(next_obs_batch, with_aug=True, detach_encoder=True)
        qf1_next_target_1 = qf1_target(next_obs_batch, next_actions_1)
        qf2_next_target_1 = qf2_target(next_obs_batch, next_actions_1)
        min_qf_next_target_1 = torch.min(qf1_next_target_1, qf2_next_target_1) - alpha * next_log_pi_1
        
        # Augmentation 2
        next_actions_2, next_log_pi_2, _ = actor.get_action(next_obs_batch, with_aug=True, detach_encoder=True)
        qf1_next_target_2 = qf1_target(next_obs_batch, next_actions_2)
        qf2_next_target_2 = qf2_target(next_obs_batch, next_actions_2)
        min_qf_next_target_2 = torch.min(qf1_next_target_2, qf2_next_target_2) - alpha * next_log_pi_2
        
        # Average the targets
        min_qf_next_target = 0.5 * (min_qf_next_target_1 + min_qf_next_target_2)
        
        next_q_value = rew_batch.flatten() + (1 - done_batch.flatten()) * gamma * (min_qf_next_target).view(-1)
        return next_q_value


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

    def forward(self, x, detach_encoder: bool = False):
        feat = self.encoder(x)
        if detach_encoder:
            feat = feat.detach()
        x = F.relu(self.fc1(feat))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)

        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std


    def get_action(self, x, with_aug: bool = True, detach_encoder: bool = False):
        enc_was_training = self.encoder.training
        if not with_aug:
            self.encoder.eval()
        mean, log_std = self.forward(x, detach_encoder=detach_encoder)
        if not with_aug:
            self.encoder.train(enc_was_training)

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
        mean, _ = self.forward(x, detach_encoder=True)
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
    scaler = torch.amp.GradScaler(enabled=(training_gpu_id.type == "cuda"))

    # env setup
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

    # DrQ-v2 style: share a single encoder across actor and critics; update encoder via critic losses only
    shared_encoder = qf1.encoder
    qf2.encoder = shared_encoder
    actor.encoder = shared_encoder

    def _unique_params(modules):
        params = []
        seen = set()
        for m in modules:
            for p in m.parameters():
                if id(p) in seen:
                    continue
                params.append(p)
                seen.add(id(p))
        return params

    q_optimizer = optim.Adam(_unique_params([qf1, qf2]), lr=args.q_lr)
    actor_optimizer = optim.Adam([p for n, p in actor.named_parameters() if not n.startswith('encoder.')], lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        entropy_scale = args.ent_scale
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(training_gpu_id)).item() * entropy_scale
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

    # load success demos
    expert_steps = load_demonstrations(rb, data_dir=args.demo_data_path, target_image_size=args.image_size)
    assert expert_steps > args.batch_size
    args.learning_starts = 0

    # ADD PRE-TRAINING STEPS
    print(f"[Pre-Training] Starting {args.pretraining_steps} steps of offline training...")
    for i in range(args.pretraining_steps):
        obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = rb.sample(args.batch_size)

        with torch.amp.autocast(device_type=training_gpu_id.type):
            next_q_value = compute_target_q(
                actor, qf1_target, qf2_target, next_obs_batch, 
                alpha, args.gamma, rew_batch, done_batch
            )
            qf1_a_values = qf1(obs_batch, act_batch).view(-1)
            qf2_a_values = qf2(obs_batch, act_batch).view(-1)
            qf_loss = F.mse_loss(qf1_a_values, next_q_value) + F.mse_loss(qf2_a_values, next_q_value)

        q_optimizer.zero_grad()
        scaler.scale(qf_loss).backward()
        scaler.step(q_optimizer)
        scaler.update()

        # update actor (SAC + BC)
        if i % args.policy_frequency == 0:
            with torch.amp.autocast(device_type=training_gpu_id.type):
                pi, log_pi, _ = actor.get_action(obs_batch, with_aug=True, detach_encoder=True)
                qf1_pi = qf1(obs_batch, pi)
                qf2_pi = qf2(obs_batch, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                
                sac_loss = ((alpha * log_pi) - min_qf_pi).mean()
                bc_loss = F.mse_loss(pi, act_batch)
                actor_loss = sac_loss + (1.0 * bc_loss)

            actor_optimizer.zero_grad()
            scaler.scale(actor_loss).backward()
            scaler.step(actor_optimizer)
            scaler.update()
            
            if args.autotune:
                 with torch.no_grad():
                    _, log_pi, _ = actor.get_action(obs_batch, with_aug=True, detach_encoder=True)
                 alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()
                 a_optimizer.zero_grad()
                 alpha_loss.backward()
                 a_optimizer.step()
                 alpha = log_alpha.exp().item()

        if i % args.target_network_frequency == 0:
            for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

        if i % 100 == 0:
            print(f"[Pre-Train] Step {i}/{args.pretraining_steps} | Q-Loss: {qf_loss.item():.3f} | Actor-Loss: {actor_loss.item():.3f}")

    print("[Pre-Training] Finished. Starting Online Interaction...\n")

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            obs_tensor = {k: torch.tensor(v).to(training_gpu_id) for k, v in obs.items()}
            with torch.no_grad():
                actions, _, _ = actor.get_action(obs_tensor, with_aug=False, detach_encoder=True)
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
        
        if "final_observation" in infos:
            for idx, final_obs in enumerate(infos["final_observation"]):
                # If final_obs is not None, it means this environment just reset
                if final_obs is not None:
                    real_next_obs["visual"][idx] = final_obs["visual"]
                    real_next_obs["proprio"][idx] = final_obs["proprio"]
        
        # Add to buffer
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = rb.sample(args.batch_size)

            # NOTE(gaoyuan): changed; use the averaged target calculation
            with torch.amp.autocast(device_type=training_gpu_id.type):
                next_q_value = compute_target_q(
                    actor, qf1_target, qf2_target, next_obs_batch, 
                    alpha, args.gamma, rew_batch, done_batch
                )
                qf1_a_values = qf1(obs_batch, act_batch).view(-1)
                qf2_a_values = qf2(obs_batch, act_batch).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            scaler.scale(qf_loss).backward()
            scaler.step(q_optimizer)
            scaler.update()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    with torch.amp.autocast(device_type=training_gpu_id.type):
                        pi, log_pi, _ = actor.get_action(obs_batch, with_aug=True, detach_encoder=True)
                        qf1_pi = qf1(obs_batch, pi)
                        qf2_pi = qf2(obs_batch, pi)
                        min_qf_pi = torch.min(qf1_pi, qf2_pi)

                        sac_loss = ((alpha * log_pi) - min_qf_pi).mean()
                        bc_loss = F.mse_loss(pi, act_batch)
                        bc_weight = max(0, 1.0 - global_step / 200000.0)
                        actor_loss = sac_loss + (bc_weight * bc_loss)

                    actor_optimizer.zero_grad()
                    scaler.scale(actor_loss).backward()
                    scaler.step(actor_optimizer)
                    scaler.update()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(obs_batch, with_aug=True, detach_encoder=True)
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
