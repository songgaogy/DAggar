import glob
import os
import re
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from robosuite.policy.flow_multi.utils.env_util import RobosuiteProprioExtractor, parse_env_info


DEFAULT_TASK_PROMPTS = {
    "Lift": [
        "lift the red cube",
        "pick up the red cube",
        "grasp the red cube and raise it",
        "lift the red block off the table",
        "pick up the red cube from the surface",
    ],
    "Stack": [
        "stack the red block on the green block",
        "place the red cube on top of the green cube",
        "pick up the red block and put it on the green block",
        "build a stack by placing the red block onto the green one",
        "grasp the red cube and set it on top of the green cube",
    ],
    "PickPlaceCan": [
        "pick up the red can and place it into the target bin far from the robot",
        "grasp the can and move it to the designated bin away from the initial platform",
        "pick the red can and put it into the goal bin on the far side of the workspace",
        "lift the can and place it into the bin located away from the arm and objects",
        "take the red can and move it away from the pickup area into the target bin",
    ],
    "PickPlaceBread": [
        "pick up the brown bread and place it into the target bin far from the robot but close to inital platform",
        "grasp the small suquare bread and move it to the designated bin close to initial platform",
        "lift the bread and place it into one of the four bins which is close to inital platform while far from the robot base",
        "take the brown bread and move it away from the large light-colored area into the target bin",
    ],
    "PickPlaceMilk": [
        "pick up the white milk carton and place it into the target bin close to the robot and the initial platform",
        "grasp the milk and move it to the designated bin near robot arm and close to the initial platform",
        "pick the large tall milk carton and put it into the goal bin located nearby and close to the robot base",
        "lift the milk and place it into one of the four bins that near the arm and also near the initial platform",
    ],
    "PickPlaceCereal": [
        "pick up the tall thin cereal box and place it into the target bin close to the robot and far from the initial platform",
        "grasp the red cereal and move it to the designated bin on the far side away from both the robot and the pickup area",
        "pick the cereal box and put it into the goal bin located far from the initial platform and robot base",
        "lift the red cereal and place it into the bin that is far away from the arm and other objects",
        "take the tall thin cereal box and move it across the workspace into the distant target bin",
    ],
}


def _normalize_prompt_value(value: Any) -> list[str]:
    if isinstance(value, str):
        prompts = [value]
    elif isinstance(value, (list, tuple)):
        prompts = [str(item).strip() for item in value if str(item).strip()]
    else:
        prompts = [str(value).strip()]
    prompts = [prompt for prompt in prompts if len(prompt) > 0]
    if len(prompts) == 0:
        raise ValueError("Task prompt list must contain at least one non-empty prompt")
    return prompts


def _list_hdf5_files(data_dir: str):
    files = sorted(glob.glob(os.path.join(data_dir, "*.hdf5")))
    if len(files) == 0:
        raise FileNotFoundError(f"No .hdf5 files found in {data_dir}")
    return files


def _cfg_mapping_to_dict(cfg: Any) -> dict:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return dict(cfg)
    try:
        return {key: cfg[key] for key in cfg.keys()}
    except Exception:
        return dict(cfg)


def _humanize_task_name(task_name: str) -> str:
    words = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(task_name)).replace("_", " ").strip().lower()
    if words.startswith("panda "):
        words = words[len("panda ") :]
    return words


class RobosuiteMultiViewFlowDataset(Dataset):
    def __init__(
        self,
        data_dir: str | None = None,
        data_dirs: list[str] | None = None,
        camera_names: list[str] | None = None,
        action_horizon: int = 8,
        image_size: int = 128,
        stride: int = 1,
        normalize: bool = True,
        max_demos_per_file: int | None = None,
        num_train_traj: int | None = None,
        cache_proprio: bool = True,
        use_demo_cache: bool = True,
        max_cached_demos_per_worker: int = 2,
        preload_all_demos_to_ram: bool = False,
        use_disk_cache: bool = True,
        task_prompt_map: dict | None = None,
    ):
        if data_dirs is None:
            if data_dir is None:
                raise ValueError("Either data_dir or data_dirs must be provided")
            data_dirs = [data_dir]
        self.data_dirs = [str(Path(path)) for path in data_dirs]
        self.files_by_data_dir = {root: _list_hdf5_files(root) for root in self.data_dirs}
        self.camera_names = list(camera_names or [])
        self.action_horizon = int(action_horizon)
        self.image_size = int(image_size)
        self.stride = int(stride)
        self.normalize = bool(normalize)
        self.max_demos_per_file = max_demos_per_file
        self.num_train_traj = None if num_train_traj is None else int(num_train_traj)
        self.cache_proprio = bool(cache_proprio)
        self.use_demo_cache = bool(use_demo_cache)
        self.max_cached_demos_per_worker = int(max_cached_demos_per_worker)
        self.preload_all_demos_to_ram = bool(preload_all_demos_to_ram)
        self.use_disk_cache = bool(use_disk_cache)

        self._handles = {}
        self._demo_meta = []
        self._proprio_cache = {}
        self._resize_index_cache = {}
        self._demo_cache = {}
        self._demo_cache_order = []
        self.index = []
        self.sample_indices_by_meta_id = {}

        self._proprio_extractors = {}
        self.task_prompt_map = {}
        self.task_metadata_map = {}
        self.task_roots = {}
        self.cache_dirs = {}

        cfg_prompt_map = _cfg_mapping_to_dict(task_prompt_map)
        for key, value in DEFAULT_TASK_PROMPTS.items():
            self.task_prompt_map[str(key)] = _normalize_prompt_value(value)
        for key, value in cfg_prompt_map.items():
            self.task_prompt_map[str(key)] = _normalize_prompt_value(value)

        self._build_index()
        if len(self.task_metadata_map) == 0:
            raise RuntimeError("No valid task metadata found while building dataset")
        first_task_name = next(iter(self.task_metadata_map.keys()))
        self.env_metadata = dict(self.task_metadata_map[first_task_name])
        if self.use_disk_cache:
            for root in self.data_dirs:
                cache_dir = os.path.join(root, ".flow_multi_cache")
                os.makedirs(cache_dir, exist_ok=True)
                self.cache_dirs[root] = cache_dir

        if self.preload_all_demos_to_ram:
            self._preload_all_demos()
        self._fit_normalizers()
        if self.preload_all_demos_to_ram:
            self.close()

    def _load_file_metadata(self, file_path: str) -> tuple[str, dict]:
        with h5py.File(file_path, "r") as handle:
            if "env_info" not in handle.attrs:
                raise KeyError(f"Missing env_info in {file_path}")
            env_metadata = parse_env_info(handle.attrs["env_info"])
            env_name = handle.attrs.get("env", env_metadata.get("env_name", Path(file_path).parent.name))
        return str(env_name), dict(env_metadata)

    def _resolve_prompt(self, task_name: str, data_dir: str) -> list[str]:
        candidates = [
            str(task_name),
            Path(data_dir).name,
            Path(data_dir).parent.name,
        ]
        for candidate in candidates:
            if candidate in self.task_prompt_map:
                return self.task_prompt_map[candidate]
        return [_humanize_task_name(task_name)]

    def _build_index(self):
        for data_dir in self.data_dirs:
            files = self.files_by_data_dir[data_dir]
            task_name, env_metadata = self._load_file_metadata(files[0])
            self.task_metadata_map[task_name] = dict(env_metadata)
            self.task_roots[task_name] = data_dir
            self.task_prompt_map[task_name] = self._resolve_prompt(task_name=task_name, data_dir=data_dir)

            num_loaded_traj = 0
            desc = f"Building flow_multi index [{task_name}]"
            for file_path in tqdm(files, desc=desc):
                if self.num_train_traj is not None and num_loaded_traj >= self.num_train_traj:
                    break
                with h5py.File(file_path, "r") as handle:
                    demo_root = handle["demos"]
                    demo_keys = sorted(list(demo_root.keys()))
                    if self.max_demos_per_file is not None:
                        demo_keys = demo_keys[: self.max_demos_per_file]

                    for demo_key in demo_keys:
                        if self.num_train_traj is not None and num_loaded_traj >= self.num_train_traj:
                            break
                        demo_group = demo_root[demo_key]
                        obs_group = demo_group["observations"]
                        missing = [name for name in self.camera_names if name not in obs_group]
                        if len(missing) > 0:
                            raise KeyError(f"Missing cameras {missing} in {file_path}:{demo_key}")

                        num_steps = demo_group["actions"].shape[0]
                        if num_steps < self.action_horizon:
                            continue

                        meta_id = len(self._demo_meta)
                        self._demo_meta.append(
                            {
                                "file_path": file_path,
                                "demo_key": demo_key,
                                "num_steps": int(num_steps),
                                "task_name": task_name,
                                "data_dir": data_dir,
                            }
                        )
                        self.sample_indices_by_meta_id[meta_id] = []
                        max_start = num_steps - self.action_horizon + 1
                        for t0 in range(0, max_start, self.stride):
                            sample_idx = len(self.index)
                            self.index.append((meta_id, t0))
                            self.sample_indices_by_meta_id[meta_id].append(sample_idx)
                        num_loaded_traj += 1

        if len(self.index) == 0:
            raise RuntimeError(f"No valid samples found across data_dirs={self.data_dirs}")
        print(f"flow_multi trajectories used for training: {len(self._demo_meta)}")
        for task_name in sorted(self.task_metadata_map.keys()):
            prompt = self.task_prompt_map[task_name]
            print(f"task={task_name} num_prompts={len(prompt)}")

    def _get_handle(self, file_path: str):
        pid = os.getpid()
        if pid not in self._handles:
            self._handles[pid] = {}
        if file_path not in self._handles[pid]:
            self._handles[pid][file_path] = h5py.File(file_path, "r", libver="latest", swmr=True)
        return self._handles[pid][file_path]

    def _get_demo_meta(self, meta_id: int) -> dict:
        return self._demo_meta[meta_id]

    def _get_proprio_extractor(self, task_name: str) -> RobosuiteProprioExtractor:
        if task_name not in self._proprio_extractors:
            self._proprio_extractors[task_name] = RobosuiteProprioExtractor(
                env_kwargs=self.task_metadata_map[task_name]
            )
        return self._proprio_extractors[task_name]

    def _get_demo_group(self, meta_id: int):
        meta = self._get_demo_meta(meta_id)
        handle = self._get_handle(meta["file_path"])
        return handle["demos"][meta["demo_key"]]

    def _load_state(self, meta_id: int, timestep: int):
        return self._get_demo_group(meta_id)["states"][timestep]

    def _load_actions(self, meta_id: int, start: int, end: int):
        return self._get_demo_group(meta_id)["actions"][start:end]

    def _load_images(self, meta_id: int, timestep: int):
        demo_group = self._get_demo_group(meta_id)
        images = []
        for camera_name in self.camera_names:
            images.append(demo_group["observations"][camera_name]["images"][timestep])
        return images

    def _get_proprio(self, meta_id: int, timestep: int):
        cache_key = (meta_id, timestep)
        if self.cache_proprio and cache_key in self._proprio_cache:
            return self._proprio_cache[cache_key]
        task_name = self._get_demo_meta(meta_id)["task_name"]
        proprio = self._get_proprio_extractor(task_name).extract(self._load_state(meta_id, timestep))
        if self.cache_proprio:
            self._proprio_cache[cache_key] = proprio
        return proprio

    def _resize_center_crop(self, img: np.ndarray):
        height, width = img.shape[:2]
        crop_size = min(height, width)
        y0 = (height - crop_size) // 2
        x0 = (width - crop_size) // 2
        crop = img[y0 : y0 + crop_size, x0 : x0 + crop_size]
        if crop_size == self.image_size:
            return crop
        if crop_size not in self._resize_index_cache:
            ys = np.linspace(0, crop_size - 1, self.image_size).astype(np.int32)
            xs = np.linspace(0, crop_size - 1, self.image_size).astype(np.int32)
            self._resize_index_cache[crop_size] = (ys, xs)
        ys, xs = self._resize_index_cache[crop_size]
        return crop[ys][:, xs]

    def _resize_center_crop_video(self, images: np.ndarray):
        height, width = images.shape[1:3]
        crop_size = min(height, width)
        y0 = (height - crop_size) // 2
        x0 = (width - crop_size) // 2
        crop = images[:, y0 : y0 + crop_size, x0 : x0 + crop_size]
        if crop_size == self.image_size:
            return crop
        if crop_size not in self._resize_index_cache:
            ys = np.linspace(0, crop_size - 1, self.image_size).astype(np.int32)
            xs = np.linspace(0, crop_size - 1, self.image_size).astype(np.int32)
            self._resize_index_cache[crop_size] = (ys, xs)
        ys, xs = self._resize_index_cache[crop_size]
        return crop[:, ys][:, :, xs]

    def _add_demo_to_cache(self, meta_id: int, demo_data: dict):
        self._demo_cache[meta_id] = demo_data
        if meta_id in self._demo_cache_order:
            self._demo_cache_order.remove(meta_id)
        self._demo_cache_order.append(meta_id)
        if self.preload_all_demos_to_ram:
            return
        while len(self._demo_cache_order) > self.max_cached_demos_per_worker:
            evicted_meta_id = self._demo_cache_order.pop(0)
            if evicted_meta_id in self._demo_cache:
                del self._demo_cache[evicted_meta_id]

    def _demo_cache_path(self, meta_id: int):
        meta = self._get_demo_meta(meta_id)
        camera_tag = "--".join(self.camera_names)
        base_name = f"{Path(meta['file_path']).stem}__{meta['demo_key']}__i{self.image_size}__{camera_tag}.npz"
        return os.path.join(self.cache_dirs[meta["data_dir"]], base_name)

    def _is_cache_valid(self, meta_id: int, cache_path: str):
        if not os.path.exists(cache_path):
            return False
        meta = self._get_demo_meta(meta_id)
        return os.path.getmtime(cache_path) >= os.path.getmtime(meta["file_path"])

    def _materialize_demo(self, meta_id: int):
        cache_path = self._demo_cache_path(meta_id) if self.use_disk_cache else None
        if cache_path is not None and self._is_cache_valid(meta_id, cache_path):
            with np.load(cache_path) as cached:
                return {
                    "images": cached["images"],
                    "actions": cached["actions"],
                    "proprio": cached["proprio"],
                }

        demo_group = self._get_demo_group(meta_id)
        states = demo_group["states"][:]
        actions = demo_group["actions"][:].astype(np.float32)
        extractor = self._get_proprio_extractor(self._get_demo_meta(meta_id)["task_name"])
        proprio = np.stack([extractor.extract(state) for state in states], axis=0).astype(np.float32)

        image_views = []
        for camera_name in self.camera_names:
            images = demo_group["observations"][camera_name]["images"][:]
            images = self._resize_center_crop_video(images)
            images = np.transpose(images, (0, 3, 1, 2))
            image_views.append(images)
        images = np.ascontiguousarray(np.stack(image_views, axis=1))

        demo_data = {
            "images": images,
            "actions": actions,
            "proprio": proprio,
        }
        if cache_path is not None:
            np.savez(cache_path, **demo_data)
        return demo_data

    def _load_demo_cache(self, meta_id: int):
        if meta_id in self._demo_cache:
            self._demo_cache_order.remove(meta_id)
            self._demo_cache_order.append(meta_id)
            return self._demo_cache[meta_id]

        demo_data = self._materialize_demo(meta_id)
        self._add_demo_to_cache(meta_id, demo_data)
        return demo_data

    def _preload_all_demos(self):
        t0 = time.time()
        total_bytes = 0
        print("preloading all demos to RAM...")
        for meta_id in tqdm(range(len(self._demo_meta)), desc="Preloading demos"):
            demo_data = self._materialize_demo(meta_id)
            total_bytes += (
                demo_data["images"].nbytes
                + demo_data["actions"].nbytes
                + demo_data["proprio"].nbytes
            )
            self._add_demo_to_cache(meta_id, demo_data)
        total_gb = total_bytes / float(1024 ** 3)
        elapsed = time.time() - t0
        print(f"preloaded {len(self._demo_cache)} demos into RAM ({total_gb:.2f} GB) in {elapsed:.1f}s")

    def _fit_normalizers(self):
        if not self.normalize:
            self.act_mean = None
            self.act_std = None
            self.prop_mean = None
            self.prop_std = None
            return

        num_samples = min(len(self.index), 5000)
        selected = np.random.choice(len(self.index), size=num_samples, replace=False)

        action_list = []
        proprio_list = []
        for sample_idx in tqdm(selected, desc="Fitting flow_multi normalizers"):
            meta_id, t0 = self.index[int(sample_idx)]
            if self.preload_all_demos_to_ram:
                demo_data = self._demo_cache[meta_id]
                action_seq = demo_data["actions"][t0 : t0 + self.action_horizon].astype(np.float32)
                proprio = demo_data["proprio"][t0].astype(np.float32)
            else:
                action_seq = self._load_actions(meta_id, t0, t0 + self.action_horizon).astype(np.float32)
                proprio = self._get_proprio(meta_id, t0).astype(np.float32)
            action_list.append(action_seq)
            proprio_list.append(proprio)

        action_array = np.stack(action_list, axis=0)
        proprio_array = np.stack(proprio_list, axis=0)

        self.act_mean = action_array.mean(axis=0)
        self.act_std = action_array.std(axis=0) + 1e-6
        self.prop_mean = proprio_array.mean(axis=0)
        self.prop_std = proprio_array.std(axis=0) + 1e-6

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        meta_id, t0 = self.index[idx]
        meta = self._get_demo_meta(meta_id)
        task_name = meta["task_name"]
        prompt_candidates = self.task_prompt_map[task_name]
        prompt_idx = int(np.random.randint(len(prompt_candidates)))
        language_instruction = prompt_candidates[prompt_idx]

        if self.preload_all_demos_to_ram or self.use_demo_cache:
            demo_data = self._load_demo_cache(meta_id)
            image_array = np.ascontiguousarray(demo_data["images"][t0])
            action_seq = demo_data["actions"][t0 : t0 + self.action_horizon].astype(np.float32)
            proprio = demo_data["proprio"][t0].astype(np.float32)
        else:
            images = self._load_images(meta_id, t0)
            action_seq = self._load_actions(meta_id, t0, t0 + self.action_horizon).astype(np.float32)
            proprio = self._get_proprio(meta_id, t0).astype(np.float32)

            image_tensors = []
            for image in images:
                image = self._resize_center_crop(image)
                image_tensors.append(np.transpose(image, (2, 0, 1)))
            image_array = np.ascontiguousarray(np.stack(image_tensors, axis=0))

        if self.normalize:
            action_seq = (action_seq - self.act_mean) / self.act_std
            proprio = (proprio - self.prop_mean) / self.prop_std

        return {
            "images": torch.from_numpy(image_array),
            "proprio": torch.from_numpy(proprio),
            "actions": torch.from_numpy(action_seq),
            "task_name": task_name,
            "language": language_instruction,
        }

    def close(self):
        for handle_map in self._handles.values():
            for handle in handle_map.values():
                try:
                    handle.close()
                except Exception:
                    pass
        for extractor in self._proprio_extractors.values():
            extractor.close()
        self._proprio_extractors.clear()
