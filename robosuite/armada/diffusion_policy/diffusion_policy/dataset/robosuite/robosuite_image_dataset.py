import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from typing import Dict
import torch
import torchvision.transforms as transforms
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply, dict_apply_with_key
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask
)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from scipy.spatial.transform import Rotation as R
import kornia.augmentation as K


class RobosuiteImageDataset(BaseImageDataset):
    def __init__(
            self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            n_obs_steps=1,
            shape_meta=None,
            random_crop=False,
            color_jitter=False,
            image_shape=(3, 240, 320),
        ):
        
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['wrist_cam', 'side_cam', 'joint_pos', 'action', 'tcp_pose'])
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps
        self.shape_meta = shape_meta
        self.action_dim = self.shape_meta['action']['shape'][0]

        # auto-detect shape
        example_action = self.replay_buffer['action'][0]
        self.input_action_dim = example_action.shape[0]
        
        self.action_rot_transformer = None
        self.obs_rot_transformer = None

        if 'rotation_rep' in shape_meta['action']:
            # NOTE: Robosuite default is 7D (3 pos + 3 axis-angle + 1 gripper)
            if self.input_action_dim == 7:
                print("[MyDataset] Detected 7D action (Robosuite style). Using 'axis_angle' as source.")
                from_rep = 'axis_angle'
            else:
                # Default fallback (e.g. for 8D actions with quaternion)
                from_rep = 'quaternion'
                
            self.action_rot_transformer = RotationTransformer(
                from_rep=from_rep, 
                to_rep=shape_meta['action']['rotation_rep']
            )

        if 'ee_pose' in shape_meta['obs']:
            self.ee_pose_dim = self.shape_meta['obs']['ee_pose']['shape'][0]
            if 'rotation_rep' in shape_meta['obs']['ee_pose']:
                self.obs_rot_transformer = RotationTransformer(
                    from_rep='quaternion', 
                    to_rep=shape_meta['obs']['ee_pose']['rotation_rep']
                )
                
        side_img_processor = []
        wrist_img_processor = []

        if random_crop:
            side_img_processor.append(transforms.Resize((image_shape[1]+8, image_shape[2]+8), interpolation=transforms.InterpolationMode.BICUBIC))
            side_img_processor.append(transforms.RandomCrop((image_shape[1], image_shape[2])))
            wrist_img_processor.append(transforms.Resize((image_shape[1]+8, image_shape[2]+8), interpolation=transforms.InterpolationMode.BICUBIC))
            wrist_img_processor.append(transforms.RandomCrop((image_shape[1], image_shape[2])))
        
        # GPU-based color augmentation using Kornia
        self.color_jitter = None
        if color_jitter:
            self.color_jitter = K.ColorJitter(
                brightness=0.4, 
                contrast=0.4, 
                saturation=0.4, 
                hue=0.1
            )

        self.side_img_processor = transforms.Compose(side_img_processor) if len(side_img_processor) > 0 else None
        self.wrist_img_processor = transforms.Compose(wrist_img_processor) if len(wrist_img_processor) > 0 else None

        assert self.n_obs_steps > 1, "Currently we only support multiple observation steps"

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        action_sample = self.replay_buffer['action']
        
        if self.input_action_dim == 7:  # Robosuite: [pos(3), axis_angle(3), gripper(1)]
            action_pos = action_sample[:, :3]
            action_rot = action_sample[:, 3:6]
            action_gripper = action_sample[:, 6]
        else:
            # Fallback (original): [pos(3), quat(4), gripper(1)]
            action_pos = action_sample[:, :3]
            action_rot = action_sample[:, 3:7]
            action_gripper = action_sample[:, 7]

        action_processed = np.zeros((action_sample.shape[0], self.action_dim))
        action_processed[:, :3] = action_pos
        
        if self.action_rot_transformer is not None:
            action_processed[:, 3:self.action_dim-1] = self.action_rot_transformer.forward(action_rot)
        else:
            action_processed[:, 3:3+action_rot.shape[1]] = action_rot
            
        action_processed[:, -1] = action_gripper

        if 'ee_pose' in self.shape_meta['obs']:
            tcp_pose_sample = self.replay_buffer['tcp_pose']
            ee_pose = np.zeros((tcp_pose_sample.shape[0], self.ee_pose_dim))
            ee_pose[:, :3] = tcp_pose_sample[:, :3]
            ee_rot = tcp_pose_sample[:, 3:7] 
            
            if self.obs_rot_transformer is not None:
                ee_pose[:, 3:self.ee_pose_dim] = self.obs_rot_transformer.forward(ee_rot)
            else:
                ee_pose[:, 3:7] = ee_rot
        
        data = {
            'action': action_processed,
            'ee_pose': ee_pose
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['wrist_img'] = get_image_range_normalizer()
        normalizer['side_img'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        wrist_img = np.moveaxis(sample['wrist_cam'],-1,1)/255
        side_img = np.moveaxis(sample['side_cam'],-1,1)/255
        action_sample = sample['action'].copy()
        
        if self.input_action_dim == 7:  # Robosuite: [x, y, z, ax, ay, az, gripper]
            action_pos = action_sample[:, :3]
            action_rot = action_sample[:, 3:6]
            action_gripper = action_sample[:, 6]
        else:
            # Original: [x, y, z, qx, qy, qz, qw, gripper]
            action_pos = action_sample[:, :3]
            action_rot = action_sample[:, 3:7]
            action_gripper = action_sample[:, 7]

        action_processed = np.zeros((action_sample.shape[0], self.action_dim))
        action_processed[:, :3] = action_pos
        
        if self.action_rot_transformer is not None:
            action_processed[:, 3:self.action_dim-1] = self.action_rot_transformer.forward(action_rot)
        else:
            action_processed[:, 3:3+action_rot.shape[1]] = action_rot
            
        action_processed[:, -1] = action_gripper

        data = {
            'obs': {
                'wrist_img': wrist_img, 
                'side_img': side_img, 
            },
            'action': action_processed.astype(np.float32), 
        }

        if 'ee_pose' in self.shape_meta['obs']:
            tcp_pose_sample = sample['tcp_pose'].copy()
            ee_pose = np.zeros((tcp_pose_sample.shape[0], self.ee_pose_dim))
            ee_pose[:, :3] = tcp_pose_sample[:, :3]
            rel_ee_rot = tcp_pose_sample[:, 3:7]    # tcp_pose is always Quat in record.py
            
            if self.obs_rot_transformer is not None:
                ee_pose[:, 3:self.ee_pose_dim] = self.obs_rot_transformer.forward(rel_ee_rot)
            else:
                ee_pose[:, 3:7] = rel_ee_rot
            
            data['obs']['ee_pose'] = ee_pose

        return data
    
    def side_image_postprocess(self, img):
        img = self.side_img_processor(img) if self.side_img_processor is not None else img
        return img
    
    def wrist_image_postprocess(self, img):
        img = self.wrist_img_processor(img) if self.wrist_img_processor is not None else img
        return img
    
    def gpu_color_augment(self, data):
        """
        Apply GPU-based color augmentation
        """
        if self.color_jitter is not None:
            batch_size = data['obs']['side_img'].shape[0]
            horizon = data['obs']['side_img'].shape[1]
            if 'side_img' in data['obs']:
                jittered_side_img = self.color_jitter(data['obs']['side_img'].view(-1, *data['obs']['side_img'].shape[-3:]))
                data['obs']['side_img'] = jittered_side_img.view(batch_size, horizon, *data['obs']['side_img'].shape[-3:])
            if 'wrist_img' in data['obs']:
                jittered_wrist_img = self.color_jitter(data['obs']['wrist_img'].view(-1, *data['obs']['wrist_img'].shape[-3:]))
                data['obs']['wrist_img'] = jittered_wrist_img.view(batch_size, horizon, *data['obs']['wrist_img'].shape[-3:])
        
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        torch_data = dict_apply_with_key(torch_data, self.side_image_postprocess, ['side_img'])
        torch_data = dict_apply_with_key(torch_data, self.wrist_image_postprocess, ['wrist_img'])
        return torch_data


def test():
    raise NotImplementedError

if __name__ == '__main__':
    test()