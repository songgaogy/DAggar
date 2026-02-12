import os
import h5py
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def print_args(args):
    print("=" * 40)
    print(f"Experiment: {args.env_id}")
    print("Hyperparameters:")
    for k, v in vars(args).items():
        print(f"  {k:25}: {v}")
    print("=" * 40)


def load_demonstrations(buffer, data_dir, target_image_size):
    """
    Load HDF5 demonstrations into the ReplayBuffer.
    """
    hdf5_files = glob.glob(os.path.join(data_dir, "*.hdf5"))
    print(f"[Data Loading] Found {len(hdf5_files)} files in {data_dir}")
    
    total_steps = 0
    episodes = 0
    
    ROBOT_STATE_DIM = 32 
    
    for filepath in hdf5_files:
        try:
            with h5py.File(filepath, "r") as f:
                for demo_key in f["data"].keys():
                    demo = f["data"][demo_key]
                    full_states = demo["states"][:] 
                    
                    if full_states.shape[1] >= ROBOT_STATE_DIM:
                        states = full_states[:, :ROBOT_STATE_DIM]
                    else:
                        print(f"Warning: State dim {full_states.shape[1]} is smaller than expected {ROBOT_STATE_DIM}!")
                        continue

                    actions = demo["actions"][:]
                    rewards = demo["rewards"][:]
                    dones = demo["dones"][:]
                    
                    img_agent = demo["agentview_image"][:]
                    img_wrist = demo["robot0_eye_in_hand_image"][:]
                    img_agent = np.transpose(img_agent, (0, 3, 1, 2))
                    img_wrist = np.transpose(img_wrist, (0, 3, 1, 2))
                    visual_obs = np.concatenate([img_agent, img_wrist], axis=1)
                    
                    num_samples = states.shape[0]
                    
                    for t in range(num_samples - 1):
                        obs = {
                            "visual": visual_obs[t],
                            "proprio": states[t]
                        }
                        
                        next_obs = {
                            "visual": visual_obs[t+1],
                            "proprio": states[t+1]
                        }
                        
                        action = actions[t]
                        reward = rewards[t]
                        done = dones[t]
                        info = {"success": True if (t == num_samples - 2) else False}
                        
                        buffer.add(obs, next_obs, action, reward, done, info)
                        total_steps += 1
                    
                    episodes += 1
                    
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            
    print(f"[Data Loading] Loaded {episodes} episodes, {total_steps} steps into ReplayBuffer.")
    return total_steps


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
