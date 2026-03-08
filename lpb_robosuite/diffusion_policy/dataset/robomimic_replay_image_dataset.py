from typing import Dict, List
import torch
import numpy as np
import h5py
from tqdm import tqdm
import zarr
import os
import shutil
import copy
import json
import hashlib
import glob
from contextlib import contextmanager
from filelock import FileLock
import multiprocessing
from omegaconf import OmegaConf
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset, LinearNormalizer
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.common.normalize_util import (
    robomimic_abs_action_only_normalizer_from_stat,
    robomimic_abs_action_only_dual_arm_normalizer_from_stat,
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats
)
register_codecs()

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    @contextmanager
    def threadpool_limits(*args, **kwargs):
        yield


def _to_builtin(obj):
    if OmegaConf.is_config(obj):
        obj = OmegaConf.to_container(obj, resolve=True)
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    return obj


def _normalize_dataset_paths(dataset_path) -> List[str]:
    dataset_path = _to_builtin(dataset_path)
    if isinstance(dataset_path, str):
        return [os.path.expanduser(dataset_path)]
    if isinstance(dataset_path, (list, tuple)):
        paths = []
        for p in dataset_path:
            if isinstance(p, str):
                paths.append(os.path.expanduser(p))
            else:
                raise TypeError(
                    f"Unsupported dataset path entry type: {type(p)}. "
                    "Expected each entry to be a string path."
                )
        if len(paths) == 0:
            raise ValueError("dataset_path list is empty")
        return paths
    raise TypeError(
        f"Unsupported dataset_path type: {type(dataset_path)}. "
        "Expected string or list/tuple of strings."
    )


def _get_replay_cache_path(dataset_path, shape_meta: dict, abs_action: bool) -> str:
    # Include shape_meta and abs_action in the cache identity so different configs
    # can safely share one source dataset path (single path or path list).
    dataset_paths = _normalize_dataset_paths(dataset_path)

    shape_meta = _to_builtin(shape_meta)

    cache_key = json.dumps(
        {
            "dataset_paths": dataset_paths,
            "shape_meta": shape_meta,
            "abs_action": bool(abs_action),
        },
        sort_keys=True,
    )
    digest = hashlib.md5(cache_key.encode("utf-8")).hexdigest()[:10]
    if len(dataset_paths) == 1:
        cache_root = dataset_paths[0].rstrip("/")
    else:
        # Keep cache colocated with the first dataset path.
        cache_root = dataset_paths[0].rstrip("/") + ".multi"
    return cache_root + f".{digest}.zarr.zip"

class RobomimicReplayImageDataset(BaseImageDataset):
    def __init__(self,
            shape_meta: dict,
            dataset_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            abs_action=False,
            rotation_rep='rotation_6d', # ignored when abs_action=False
            use_legacy_normalizer=False,
            use_cache=False,
            seed=42,
            val_ratio=0.0
        ):
        rotation_transformer = RotationTransformer(
            from_rep='axis_angle', to_rep=rotation_rep)

        replay_buffer = None
        if use_cache:
            cache_zarr_path = _get_replay_cache_path(
                dataset_path=dataset_path,
                shape_meta=shape_meta,
                abs_action=abs_action,
            )
            cache_lock_path = cache_zarr_path + '.lock'
            print('Acquiring lock on cache.')
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    # cache does not exists
                    try:
                        print('Cache does not exist. Creating!')
                        # store = zarr.DirectoryStore(cache_zarr_path)
                        replay_buffer = _convert_robomimic_to_replay(
                            store=zarr.MemoryStore(), 
                            shape_meta=shape_meta, 
                            dataset_path=dataset_path, 
                            abs_action=abs_action, 
                            rotation_transformer=rotation_transformer)
                        print('Saving cache to disk.')
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(
                                store=zip_store
                            )
                    except Exception as e:
                        if os.path.isdir(cache_zarr_path):
                            shutil.rmtree(cache_zarr_path)
                        elif os.path.exists(cache_zarr_path):
                            os.remove(cache_zarr_path)
                        raise e
                else:
                    print('Loading cached ReplayBuffer from Disk.')
                    print('cache_zarr_path ', cache_zarr_path)
                    with zarr.ZipStore(cache_zarr_path, mode='r') as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore())
                    print('Loaded!')
        else:
            replay_buffer = _convert_robomimic_to_replay(
                store=zarr.MemoryStore(), 
                shape_meta=shape_meta, 
                dataset_path=dataset_path, 
                abs_action=abs_action, 
                rotation_transformer=rotation_transformer)

        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)
        
        # for key in rgb_keys:
        #     replay_buffer[key].compressor.numthreads=1

        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        sampler = SequenceSampler(
            replay_buffer=replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k)
        
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.abs_action = abs_action
        self.n_obs_steps = n_obs_steps
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.use_legacy_normalizer = use_legacy_normalizer
        print('episode ends ', replay_buffer.episode_ends[:])
        # print('agentview_image', replay_buffer['agentview_image'].shape)
        # print('robot0_eef_pos ', replay_buffer['robot0_eef_pos'].shape)
        print('action ', replay_buffer['action'].shape)
        print('abs_action ', replay_buffer['abs_action'].shape)

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

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action
        if self.abs_action:
            stat = array_to_stats(self.replay_buffer['abs_action'])
        else:
            stat = array_to_stats(self.replay_buffer['action'])

        if self.abs_action:
            if stat['mean'].shape[-1] > 10:
                # dual arm
                this_normalizer = robomimic_abs_action_only_dual_arm_normalizer_from_stat(stat)
            else:
                this_normalizer = robomimic_abs_action_only_normalizer_from_stat(stat)
            
            if self.use_legacy_normalizer:
                this_normalizer = normalizer_from_stat(stat)
        else:
            # already normalized
            this_normalizer = get_identity_normalizer_from_stat(stat)
        normalizer['action'] = this_normalizer

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])

            if key.endswith('pos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('quat'):
                # quaternion is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('qpos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                raise RuntimeError('unsupported')
            normalizer[key] = this_normalizer

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)
        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        T_slice = slice(self.n_obs_steps)

        obs_dict = dict()
        for key in self.rgb_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = np.moveaxis(data[key][T_slice],-1,1
                ).astype(np.float32) / 255.
            # T,C,H,W
            del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]
        if self.abs_action:
            torch_data = {
                'obs': dict_apply(obs_dict, torch.from_numpy),
                'action': torch.from_numpy(data['abs_action'].astype(np.float32))
            }
        else:
            torch_data = {
                'obs': dict_apply(obs_dict, torch.from_numpy),
                'action': torch.from_numpy(data['action'].astype(np.float32))
            }
        return torch_data


def _convert_actions(raw_actions, abs_action, rotation_transformer):
    actions = raw_actions
    if abs_action:
        is_dual_arm = False
        if raw_actions.shape[-1] == 14:
            # dual arm
            raw_actions = raw_actions.reshape(-1,2,7)
            is_dual_arm = True

        pos = raw_actions[...,:3]
        rot = raw_actions[...,3:6]
        gripper = raw_actions[...,6:]
        rot = rotation_transformer.forward(rot)
        raw_actions = np.concatenate([
            pos, rot, gripper
        ], axis=-1).astype(np.float32)
    
        if is_dual_arm:
            raw_actions = raw_actions.reshape(-1,20)
        actions = raw_actions
    return actions

def undo_transform_action(action, rotation_transformer):
    raw_shape = action.shape
    if raw_shape[-1] == 20:
        # dual arm
        action = action.reshape(-1,2,10)

    d_rot = action.shape[-1] - 4
    pos = action[...,:3]
    rot = action[...,3:3+d_rot]
    gripper = action[...,[-1]]
    rot = rotation_transformer.inverse(rot)
    uaction = np.concatenate([
        pos, rot, gripper
    ], axis=-1)

    if raw_shape[-1] == 20:
        # dual arm
        uaction = uaction.reshape(*raw_shape[:-1], 14)

    return uaction
    
def _resolve_dataset_files(dataset_path) -> List[str]:
    paths = _normalize_dataset_paths(dataset_path)
    all_files = []
    for path in paths:
        if os.path.isfile(path):
            all_files.append(path)
            continue
        if os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, "**", "*.hdf5"), recursive=True))
            if len(files) == 0:
                raise FileNotFoundError(f"No .hdf5 files found under directory: {path}")
            all_files.extend(files)
            continue
        if any(ch in path for ch in ["*", "?", "["]):
            files = sorted(glob.glob(path))
            if len(files) == 0:
                raise FileNotFoundError(f"No .hdf5 files matched pattern: {path}")
            all_files.extend(files)
            continue
        raise FileNotFoundError(f"Dataset path is neither a file nor a directory: {path}")

    return sorted(set(all_files))


def _sorted_demo_keys(demos_group: h5py.Group) -> List[str]:
    def key_fn(name: str):
        suffix = name.split("_")[-1]
        if suffix.isdigit():
            return (0, int(suffix))
        return (1, suffix)
    return sorted(list(demos_group.keys()), key=key_fn)


def _get_data_group(file: h5py.File):
    if "data" in file:
        return "robomimic", file["data"]
    if "demos" in file:
        return "robosuite", file["demos"]
    raise KeyError(
        "Unsupported dataset format. Expected root group `data` (robomimic) or `demos` "
        "(robosuite collect_human_demonstrations format)."
    )


def _try_get_lowdim_dataset(demo: h5py.Group, dataset_format: str, key: str):
    if "obs" in demo and key in demo["obs"]:
        return demo["obs"][key]
    if key in demo:
        return demo[key]
    if dataset_format == "robosuite" and "observations" in demo and key in demo["observations"]:
        return demo["observations"][key]
    return None


def _try_get_image_dataset(demo: h5py.Group, dataset_format: str, key: str):
    if "obs" in demo and key in demo["obs"]:
        return demo["obs"][key]
    if dataset_format == "robosuite" and "observations" in demo:
        obs_group = demo["observations"]
        if key in obs_group and "images" in obs_group[key]:
            return obs_group[key]["images"]
        camera_name = key
        if key.endswith("_image"):
            camera_name = key[: -len("_image")]
        elif key.endswith("_rgb"):
            camera_name = key[: -len("_rgb")]
        if camera_name in obs_group and "images" in obs_group[camera_name]:
            return obs_group[camera_name]["images"]
    return None


def _extract_env_info_string(file: h5py.File):
    if "env_info" in file.attrs:
        env_info = file.attrs["env_info"]
        if isinstance(env_info, bytes):
            env_info = env_info.decode("utf-8")
        return str(env_info)
    if "data" in file and "env_info" in file["data"].attrs:
        env_info = file["data"].attrs["env_info"]
        if isinstance(env_info, bytes):
            env_info = env_info.decode("utf-8")
        return str(env_info)
    return None


class _RobosuiteStateObsExtractor:
    def __init__(self, env_info_json: str, model_xml: str = None):
        import mujoco

        self._mujoco = mujoco
        self._env_info_json = env_info_json
        self.env = None
        self.sim = None

        self.nq = None
        self.nv = None
        self.na = None
        self.robot_qpos_inds = None
        self.robot_qvel_inds = None

        # Prefer model xml from dataset so we don't depend on controller configs.
        if model_xml is not None:
            try:
                model = mujoco.MjModel.from_xml_string(model_xml)
                self._set_dims_and_indices_from_mj_model(model)
            except Exception:
                # fall back to runtime env creation below
                pass

        if self.robot_qpos_inds is None:
            self._init_runtime_env()

    def _joint_dims(self, joint_type: int):
        mujoco = self._mujoco
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            return 7, 6
        if joint_type == mujoco.mjtJoint.mjJNT_BALL:
            return 4, 3
        if joint_type in (mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE):
            return 1, 1
        raise RuntimeError(f"Unsupported joint type: {joint_type}")

    def _set_dims_and_indices_from_mj_model(self, model):
        mujoco = self._mujoco
        self.nq = int(model.nq)
        self.nv = int(model.nv)
        self.na = int(getattr(model, "na", 0))
        model_for_name_lookup = getattr(model, "_model", model)

        qpos_inds = []
        qvel_inds = []
        for joint_id in range(int(model.njnt)):
            if hasattr(model, "joint_names") and (model.joint_names is not None):
                joint_name = model.joint_names[joint_id]
            else:
                joint_name = mujoco.mj_id2name(model_for_name_lookup, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if (joint_name is None) or (not str(joint_name).startswith("robot0_")):
                continue

            qpos_adr = int(model.jnt_qposadr[joint_id])
            qvel_adr = int(model.jnt_dofadr[joint_id])
            joint_type = int(model.jnt_type[joint_id])
            n_qpos, n_dof = self._joint_dims(joint_type)

            qpos_inds.extend(range(qpos_adr, qpos_adr + n_qpos))
            qvel_inds.extend(range(qvel_adr, qvel_adr + n_dof))

        if len(qpos_inds) == 0 or len(qvel_inds) == 0:
            raise RuntimeError("No robot0_ joints found when parsing Mujoco model.")

        self.robot_qpos_inds = np.array(qpos_inds, dtype=np.int64)
        self.robot_qvel_inds = np.array(qvel_inds, dtype=np.int64)

    def _init_runtime_env(self):
        import robosuite as suite

        env_kwargs = json.loads(self._env_info_json)
        env_kwargs = dict(env_kwargs)
        # Collected controller configs can be version-specific and may miss keys
        # (e.g. "interpolation") in another robosuite version. They are not needed
        # for decoding flattened states, so rely on the environment defaults.
        env_kwargs.pop("controller_configs", None)
        env_kwargs.pop("controller_config", None)
        env_kwargs.update(
            has_renderer=False,
            has_offscreen_renderer=False,
            ignore_done=True,
            use_camera_obs=False,
            reward_shaping=False,
        )

        self.env = suite.make(**env_kwargs)
        self.sim = self.env.sim
        self._set_dims_and_indices_from_mj_model(self.sim.model)

    def close(self):
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass

    def _split_state(self, state: np.ndarray):
        state = np.asarray(state).reshape(-1)
        core0 = self.nq + self.nv
        core1 = self.nq + self.nv + self.na
        n = int(state.shape[0])

        if n == core0:
            t = 0.0
            base = state
        elif n == core1:
            t = 0.0
            base = state[:core0]
        elif n == 1 + core0:
            t = float(state[0])
            base = state[1 : 1 + core0]
        elif n == 1 + core1:
            t = float(state[0])
            base = state[1 : 1 + core0]
        else:
            raise ValueError(
                f"Unexpected flattened state length {n}. Expected one of "
                f"{core0}, {core1}, {1 + core0}, {1 + core1}."
            )

        qpos = base[: self.nq]
        qvel = base[self.nq : self.nq + self.nv]
        return t, qpos, qvel

    def extract_keys_from_states(self, states: np.ndarray, keys: List[str]):
        keys = list(dict.fromkeys(keys))
        out = {k: [] for k in keys}

        # keys that can be computed directly from qpos / qvel without sim query
        direct_keys = {
            "robot0_joint_qpos",
            "robot0_jointvel_qpos",
            "robot0_joint_vel_qpos",
        }
        sim_keys = [k for k in keys if k not in direct_keys]

        for state in states:
            t, qpos, qvel = self._split_state(state)
            robot_qpos = qpos[self.robot_qpos_inds].astype(np.float32)
            robot_qvel = qvel[self.robot_qvel_inds].astype(np.float32)

            if "robot0_joint_qpos" in out:
                out["robot0_joint_qpos"].append(robot_qpos)
            if "robot0_jointvel_qpos" in out:
                out["robot0_jointvel_qpos"].append(robot_qvel)
            if "robot0_joint_vel_qpos" in out:
                out["robot0_joint_vel_qpos"].append(robot_qvel)

            if len(sim_keys) > 0:
                if self.env is None:
                    self._init_runtime_env()
                sim_state = np.concatenate([[t], qpos, qvel], axis=0)
                self.sim.set_state_from_flattened(sim_state)
                self.sim.forward()
                obs = self.env._get_observations(force_update=True)
                for key in sim_keys:
                    if key not in obs:
                        raise KeyError(
                            f"Cannot derive observation key `{key}` from stored states. "
                            "Use keys available in env observations or qpos-based keys."
                        )
                    out[key].append(np.asarray(obs[key], dtype=np.float32))

        return {k: np.stack(v, axis=0).astype(np.float32) for k, v in out.items()}


def _convert_robomimic_to_replay(
    store,
    shape_meta,
    dataset_path,
    abs_action,
    rotation_transformer,
    n_workers=None,
    max_inflight_tasks=None,
):
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5
    _ = max_inflight_tasks  # kept for backward-compatible signature

    rgb_keys = []
    lowdim_keys = []
    obs_shape_meta = shape_meta["obs"]
    for key, attr in obs_shape_meta.items():
        obs_type = attr.get("type", "low_dim")
        if obs_type == "rgb":
            rgb_keys.append(key)
        elif obs_type == "low_dim":
            lowdim_keys.append(key)

    dataset_files = _resolve_dataset_files(dataset_path)
    print(f"Found {len(dataset_files)} dataset file(s) for {dataset_path}")

    # Pass 1: build episode index and compute total steps.
    episode_records = []
    episode_ends = []
    prev_end = 0
    for file_path in dataset_files:
        with h5py.File(file_path, "r") as file:
            dataset_format, demos_group = _get_data_group(file)
            for demo_key in _sorted_demo_keys(demos_group):
                demo = demos_group[demo_key]
                if "actions" not in demo:
                    continue

                episode_len = int(demo["actions"].shape[0])

                if abs_action and "abs_actions" not in demo:
                    raise KeyError(
                        f"Dataset requires `abs_actions` because abs_action=True, "
                        f"but `{file_path}:{demos_group.name}/{demo_key}` does not contain it."
                    )

                for lowdim_key in lowdim_keys:
                    lowdim_ds = _try_get_lowdim_dataset(demo, dataset_format, lowdim_key)
                    if lowdim_ds is not None:
                        episode_len = min(episode_len, int(lowdim_ds.shape[0]))
                    elif "states" in demo:
                        episode_len = min(episode_len, int(demo["states"].shape[0]))
                    else:
                        raise KeyError(
                            f"Missing lowdim key `{lowdim_key}` and no `states` dataset available at "
                            f"{file_path}:{demos_group.name}/{demo_key}"
                        )

                for rgb_key in rgb_keys:
                    img_ds = _try_get_image_dataset(demo, dataset_format, rgb_key)
                    if img_ds is None:
                        raise KeyError(
                            f"Missing image key `{rgb_key}` in {file_path}:{demos_group.name}/{demo_key}. "
                            "For robosuite-format files, expected `observations/<camera>/images` where "
                            "<camera> is either `<key>` or `<key>` without `_image` suffix."
                        )
                    episode_len = min(episode_len, int(img_ds.shape[0]))

                if episode_len <= 0:
                    continue

                prev_end += episode_len
                episode_ends.append(prev_end)
                episode_records.append(
                    {
                        "file_path": file_path,
                        "dataset_format": dataset_format,
                        "demo_key": demo_key,
                        "episode_len": episode_len,
                    }
                )

    if len(episode_records) == 0:
        raise RuntimeError(f"No valid episodes found in dataset path: {dataset_path}")

    n_steps = episode_ends[-1]
    print(f"Prepared {len(episode_records)} episodes, total steps: {n_steps}")

    root = zarr.group(store)
    data_group = root.require_group("data", overwrite=True)
    meta_group = root.require_group("meta", overwrite=True)
    _ = meta_group.array("episode_ends", episode_ends, dtype=np.int64, compressor=None, overwrite=True)

    # Pre-create image arrays in zarr.
    image_arrays = {}
    for rgb_key in rgb_keys:
        c, h, w = tuple(shape_meta["obs"][rgb_key]["shape"])
        image_arrays[rgb_key] = data_group.require_dataset(
            name=rgb_key,
            shape=(n_steps, h, w, c),
            chunks=(1, h, w, c),
            compressor=None,
            dtype=np.uint8,
        )

    action_chunks = []
    abs_action_chunks = []
    lowdim_chunks = {k: [] for k in lowdim_keys}
    state_extractors = {}
    offset = 0

    with tqdm(total=len(episode_records), desc="Converting episodes", mininterval=1.0) as pbar:
        for record in episode_records:
            file_path = record["file_path"]
            dataset_format = record["dataset_format"]
            demo_key = record["demo_key"]
            episode_len = record["episode_len"]

            with h5py.File(file_path, "r") as file:
                _, demos_group = _get_data_group(file)
                demo = demos_group[demo_key]
                sl = slice(0, episode_len)

                # actions
                raw_actions = demo["actions"][sl].astype(np.float32)
                action_chunks.append(
                    _convert_actions(
                        raw_actions=raw_actions,
                        abs_action=False,
                        rotation_transformer=rotation_transformer,
                    )
                )

                if "abs_actions" in demo:
                    raw_abs_actions = demo["abs_actions"][sl].astype(np.float32)
                    abs_action_chunks.append(
                        _convert_actions(
                            raw_actions=raw_abs_actions,
                            abs_action=True,
                            rotation_transformer=rotation_transformer,
                        )
                    )
                else:
                    # keep key present even for datasets without absolute actions
                    abs_action_chunks.append(action_chunks[-1].copy())

                # lowdim observations
                missing_lowdim = []
                per_demo_lowdim = {}
                for lowdim_key in lowdim_keys:
                    lowdim_ds = _try_get_lowdim_dataset(demo, dataset_format, lowdim_key)
                    if lowdim_ds is None:
                        missing_lowdim.append(lowdim_key)
                        continue
                    per_demo_lowdim[lowdim_key] = lowdim_ds[sl].astype(np.float32)

                if len(missing_lowdim) > 0:
                    if "states" not in demo:
                        raise KeyError(
                            f"Cannot infer lowdim keys {missing_lowdim} because `states` is missing at "
                            f"{file_path}:{demos_group.name}/{demo_key}"
                        )
                    if file_path not in state_extractors:
                        env_info = _extract_env_info_string(file)
                        if env_info is None:
                            raise KeyError(
                                f"`env_info` not found in {file_path}. Needed to reconstruct lowdim "
                                "observations from flattened states."
                            )
                        model_xml = demo.attrs.get("model_file", None)
                        if isinstance(model_xml, bytes):
                            model_xml = model_xml.decode("utf-8")
                        state_extractors[file_path] = _RobosuiteStateObsExtractor(
                            env_info_json=env_info,
                            model_xml=model_xml,
                        )

                    states = demo["states"][sl]
                    extracted = state_extractors[file_path].extract_keys_from_states(
                        states=states, keys=missing_lowdim
                    )
                    per_demo_lowdim.update(extracted)

                for lowdim_key in lowdim_keys:
                    arr = per_demo_lowdim[lowdim_key]
                    expected_shape = tuple(shape_meta["obs"][lowdim_key]["shape"])
                    if arr.shape != (episode_len,) + expected_shape:
                        raise ValueError(
                            f"Lowdim key `{lowdim_key}` has shape {arr.shape}, expected "
                            f"{(episode_len,) + expected_shape} for "
                            f"{file_path}:{demos_group.name}/{demo_key}"
                        )
                    lowdim_chunks[lowdim_key].append(arr.astype(np.float32))

                # images
                for rgb_key in rgb_keys:
                    img_ds = _try_get_image_dataset(demo, dataset_format, rgb_key)
                    imgs = img_ds[sl]
                    expected_hwc = tuple(shape_meta["obs"][rgb_key]["shape"])[1:] + (
                        tuple(shape_meta["obs"][rgb_key]["shape"])[0],
                    )
                    if imgs.shape[1:] != expected_hwc:
                        raise ValueError(
                            f"Image key `{rgb_key}` has shape {imgs.shape[1:]}, expected {expected_hwc}. "
                            f"Set `shape_meta.obs.{rgb_key}.shape` to match your dataset."
                        )
                    image_arrays[rgb_key][offset : offset + episode_len] = imgs.astype(np.uint8)

                offset += episode_len
                pbar.update(1)

    for extractor in state_extractors.values():
        extractor.close()

    action_data = np.concatenate(action_chunks, axis=0).astype(np.float32)
    abs_action_data = np.concatenate(abs_action_chunks, axis=0).astype(np.float32)
    _ = data_group.array(
        name="action",
        data=action_data,
        shape=action_data.shape,
        chunks=action_data.shape,
        compressor=None,
        dtype=action_data.dtype,
    )
    _ = data_group.array(
        name="abs_action",
        data=abs_action_data,
        shape=abs_action_data.shape,
        chunks=abs_action_data.shape,
        compressor=None,
        dtype=abs_action_data.dtype,
    )

    for lowdim_key, chunk_list in lowdim_chunks.items():
        lowdim_data = np.concatenate(chunk_list, axis=0).astype(np.float32)
        _ = data_group.array(
            name=lowdim_key,
            data=lowdim_data,
            shape=lowdim_data.shape,
            chunks=lowdim_data.shape,
            compressor=None,
            dtype=lowdim_data.dtype,
        )

    replay_buffer = ReplayBuffer(root)
    return replay_buffer

def normalizer_from_stat(stat):
    max_abs = np.maximum(stat['max'].max(), np.abs(stat['min']).max())
    scale = np.full_like(stat['max'], fill_value=1/max_abs)
    offset = np.zeros_like(stat['max'])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )
