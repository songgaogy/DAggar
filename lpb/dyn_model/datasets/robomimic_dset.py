from diffusion_policy.common.normalize_util import get_identity_normalizer_from_stat, get_image_range_normalizer, get_range_normalizer_from_stat, robomimic_abs_action_only_normalizer_from_stat
from diffusion_policy.dataset.robomimic_replay_image_dataset import _convert_robomimic_to_replay
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from dyn_model.datasets.img_transforms import default_transform, get_train_crop_transform_resnet, get_eval_crop_transform_resnet
from diffusion_policy.common.replay_buffer import ReplayBuffer
import torch
from filelock import FileLock
import os
import shutil
from torch.utils.data import Dataset
import numpy as np
from omegaconf import ListConfig

import zarr


def _normalize_dataset_paths(dataset_path):
    if isinstance(dataset_path, ListConfig):
        dataset_path = list(dataset_path)
    if isinstance(dataset_path, (list, tuple)):
        return [os.path.expanduser(str(path)) for path in dataset_path]
    return [os.path.expanduser(str(dataset_path))]


def _ensure_replay_cache(dataset_path, shape_meta, abs_action, rotation_transformer):
    cache_zarr_path = dataset_path + '.zarr'
    cache_lock_path = cache_zarr_path + '.lock'
    print(f'Acquiring lock on cache: {dataset_path}')
    with FileLock(cache_lock_path):
        if not os.path.exists(cache_zarr_path):
            try:
                print(f'Cache does not exist. Creating: {dataset_path}')
                replay_buffer = _convert_robomimic_to_replay(
                    store=zarr.MemoryStore(),
                    shape_meta=shape_meta,
                    dataset_path=dataset_path,
                    abs_action=abs_action,
                    rotation_transformer=rotation_transformer)
                print(f'Saving cache to disk: {cache_zarr_path}')
                replay_buffer.save_to_path(cache_zarr_path)
            except Exception as e:
                if os.path.exists(cache_zarr_path):
                    if os.path.isdir(cache_zarr_path):
                        shutil.rmtree(cache_zarr_path)
                    else:
                        os.remove(cache_zarr_path)
                raise e
    return cache_zarr_path


def _open_replay_buffer(cache_zarr_path):
    return ReplayBuffer.create_from_path(cache_zarr_path, mode='r')


def _update_running_stats(running_stats, chunk):
    chunk = np.asarray(chunk, dtype=np.float64)
    if chunk.ndim == 1:
        chunk = chunk[:, None]
    if chunk.shape[0] == 0:
        return running_stats

    chunk_min = np.min(chunk, axis=0)
    chunk_max = np.max(chunk, axis=0)
    chunk_sum = np.sum(chunk, axis=0)
    chunk_sumsq = np.sum(np.square(chunk), axis=0)

    if running_stats is None:
        return {
            'min': chunk_min,
            'max': chunk_max,
            'sum': chunk_sum,
            'sumsq': chunk_sumsq,
            'count': int(chunk.shape[0]),
        }

    running_stats['min'] = np.minimum(running_stats['min'], chunk_min)
    running_stats['max'] = np.maximum(running_stats['max'], chunk_max)
    running_stats['sum'] += chunk_sum
    running_stats['sumsq'] += chunk_sumsq
    running_stats['count'] += int(chunk.shape[0])
    return running_stats


def _finalize_running_stats(running_stats, dim):
    if running_stats is None:
        zeros = np.zeros((dim,), dtype=np.float32)
        return {
            'min': zeros.copy(),
            'max': zeros.copy(),
            'mean': zeros.copy(),
            'std': zeros.copy(),
        }

    mean = running_stats['sum'] / running_stats['count']
    var = running_stats['sumsq'] / running_stats['count'] - np.square(mean)
    var = np.maximum(var, 0.0)
    return {
        'min': running_stats['min'].astype(np.float32),
        'max': running_stats['max'].astype(np.float32),
        'mean': mean.astype(np.float32),
        'std': np.sqrt(var).astype(np.float32),
    }


class RobomimicImageDynamicsModelDataset(Dataset):
    def __init__(self, 
                 zarr_path, 
                 num_hist=1, 
                 num_pred=1, 
                 frameskip=8,
                 view_names=['agentview', 'robot0_eye_in_hand'],
                 abs_action=False,
                 use_crop=False,
                 train=True,
                 shape_obs=None,
                 original_img_size=140,
                 cropped_img_size=128,
                 action_dim=10):
        """
        Initializes the dataset by loading data from a Zarr file and precomputing valid anchor indices.
        
        Args:
            zarr_path (str): Path to the Zarr dataset.
            horizon (int): Number of steps for history and future.
            val_ratio (float): Fraction of episodes to use for validation.
            n_neg (int): Number of negative samples (unused in this implementation).
        """
        self.abs_action = abs_action
        self.original_img_size = original_img_size
        self.cropped_img_size = cropped_img_size
        
        # Use action_dim from config
        self.original_action_dim = action_dim
        
        # Build shape_meta from provided shape_obs
        shape_meta = {
            'obs': shape_obs,
            'action': {'shape': [action_dim]}
        }
        self.shape_obs = shape_obs
        rotation_transformer = RotationTransformer(
            from_rep='axis_angle', to_rep='rotation_6d')

        self.dataset_paths = _normalize_dataset_paths(zarr_path)
        self.view_names = view_names
        self.state_keys = [
            key for key, meta in shape_obs.items()
            if not (meta.get('type') == 'rgb' or 'image' in key)
        ]
        self.action_key = 'abs_action' if self.abs_action else 'action'
        self.states_dim = int(sum(np.prod(shape_obs[key]['shape']) for key in self.state_keys))
        self.proprio_dim = self.states_dim
        self.action_dim = self.original_action_dim * frameskip

        self.num_hist = num_hist
        self.num_pred = num_pred
        self.frameskip = frameskip
        self.num_frames = num_hist + num_pred
        self.use_crop = use_crop
        self.train = train

        self._replay_buffer_cache = dict()
        self.dataset_infos = []
        self._action_stats = None
        self._state_stats = None

        for dataset_path in self.dataset_paths:
            cache_zarr_path = _ensure_replay_cache(
                dataset_path=dataset_path,
                shape_meta=shape_meta,
                abs_action=abs_action,
                rotation_transformer=rotation_transformer,
            )
            replay_buffer = _open_replay_buffer(cache_zarr_path)
            episode_ends = replay_buffer.episode_ends[:]
            episode_starts = np.concatenate(([0], episode_ends[:-1]))
            episode_end_indices = episode_ends - 1
            valid_anchor_indices = []
            for start, end in zip(episode_starts, episode_end_indices):
                anchor_start = start
                anchor_end = end - num_pred * self.frameskip
                if anchor_end >= anchor_start:
                    valid_anchor_indices.extend(np.arange(anchor_start, anchor_end, dtype=np.int64))
            valid_anchor_indices = np.asarray(valid_anchor_indices, dtype=np.int64)

            action_array = replay_buffer[self.action_key]
            action_chunk_size = action_array.chunks[0] if hasattr(action_array, 'chunks') else 2048
            for start in range(0, action_array.shape[0], action_chunk_size):
                end = min(action_array.shape[0], start + action_chunk_size)
                self._action_stats = _update_running_stats(
                    self._action_stats,
                    action_array[start:end],
                )

            if len(self.state_keys) > 0:
                state_chunk_size = min(
                    replay_buffer[self.state_keys[0]].chunks[0]
                    if hasattr(replay_buffer[self.state_keys[0]], 'chunks')
                    else 2048,
                    action_chunk_size,
                )
                for start in range(0, action_array.shape[0], state_chunk_size):
                    end = min(action_array.shape[0], start + state_chunk_size)
                    state_chunk = np.concatenate(
                        [np.asarray(replay_buffer[key][start:end]) for key in self.state_keys],
                        axis=1,
                    )
                    self._state_stats = _update_running_stats(self._state_stats, state_chunk)

            self.dataset_infos.append({
                'cache_zarr_path': cache_zarr_path,
                'valid_anchor_indices': valid_anchor_indices,
            })

        self._action_stats = _finalize_running_stats(self._action_stats, self.original_action_dim)
        self._state_stats = _finalize_running_stats(self._state_stats, self.states_dim)
        self.valid_counts = np.asarray(
            [len(info['valid_anchor_indices']) for info in self.dataset_infos],
            dtype=np.int64,
        )
        self.valid_offsets = np.cumsum(self.valid_counts)
        self.num_valid = int(self.valid_offsets[-1]) if len(self.valid_offsets) > 0 else 0

        self.transform = default_transform()
        if self.use_crop:
            if self.train:
                self.transform = get_train_crop_transform_resnet(original_img_size, cropped_img_size)
            else:
                self.transform = get_eval_crop_transform_resnet(original_img_size, cropped_img_size)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_replay_buffer_cache'] = dict()
        return state

    def _close_replay_buffers(self):
        self._replay_buffer_cache = dict()

    def _get_replay_buffer(self, dataset_idx):
        if dataset_idx not in self._replay_buffer_cache:
            replay_buffer = _open_replay_buffer(
                self.dataset_infos[dataset_idx]['cache_zarr_path']
            )
            self._replay_buffer_cache[dataset_idx] = replay_buffer
        return self._replay_buffer_cache[dataset_idx]
        
    def __len__(self):
        """
        Returns the number of valid anchor samples.
        """
        return self.num_valid
    
    def __getitem__(self, idx):
        dataset_idx = int(np.searchsorted(self.valid_offsets, idx, side='right'))
        dataset_start = 0 if dataset_idx == 0 else int(self.valid_offsets[dataset_idx - 1])
        local_idx = int(idx - dataset_start)
        dataset_info = self.dataset_infos[dataset_idx]
        replay_buffer = self._get_replay_buffer(dataset_idx)

        start = int(dataset_info['valid_anchor_indices'][local_idx])
        end = start + (self.num_frames) * self.frameskip
        obs_indices = list(range(start, end, self.frameskip))
        action_indices = list(range(start, end))
        action_indices[-self.frameskip:] = [obs_indices[-1] - 1] * self.frameskip
        obs = {}
        obs['visual'] = {}
        for view_name in self.view_names:
            obs['visual'][view_name] = np.asarray(replay_buffer[view_name][obs_indices])
            obs['visual'][view_name] = np.moveaxis(obs['visual'][view_name],-1,1).astype(np.float32)/255
            obs['visual'][view_name] = torch.from_numpy(obs['visual'][view_name])

        if len(self.state_keys) > 0:
            state = np.concatenate(
                [np.asarray(replay_buffer[key][obs_indices]) for key in self.state_keys],
                axis=1,
            )
        else:
            state = np.zeros((len(obs_indices), 0), dtype=np.float32)

        obs['proprio'] = state
        obs['proprio'] = torch.from_numpy(obs['proprio'].astype(np.float32))
        act = np.asarray(replay_buffer[self.action_key][action_indices])
        act = torch.from_numpy(act.astype(np.float32))
        state = torch.from_numpy(state.astype(np.float32))

        return tuple([obs, act, state])

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        # action
        act_stat = self._action_stats

        if self.abs_action:
            act_normalizer = robomimic_abs_action_only_normalizer_from_stat(act_stat)
        else:
            # already normalized
            act_normalizer = get_identity_normalizer_from_stat(act_stat)
        normalizer['act'] = act_normalizer
        # state
        state_stat = self._state_stats
        normalizer['state'] = get_range_normalizer_from_stat(state_stat)
        for view_name in self.view_names:
            normalizer[view_name] = get_image_range_normalizer()
        return normalizer

    def __del__(self):
        self._close_replay_buffers()
