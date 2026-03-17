import glob
import os
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from robosuite.policy.flow_unet.utils.env_util import RobosuiteProprioExtractor, parse_env_info


def _list_hdf5_files(data_dir: str):
    files = sorted(glob.glob(os.path.join(data_dir, "*.hdf5")))
    if len(files) == 0:
        raise FileNotFoundError(f"No .hdf5 files found in {data_dir}")
    return files


class RobosuiteMultiViewFlowDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        camera_names: list[str],
        action_horizon: int,
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
    ):
        self.data_dir = data_dir
        self.files = _list_hdf5_files(data_dir)
        self.camera_names = list(camera_names)
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
        self.cache_dir = os.path.join(self.data_dir, ".flow_unet_cache")

        self.env_metadata = self._load_env_metadata(self.files[0])
        self.proprio_extractor = RobosuiteProprioExtractor(env_kwargs=self.env_metadata)
        if self.use_disk_cache:
            os.makedirs(self.cache_dir, exist_ok=True)

        self._build_index()
        if self.preload_all_demos_to_ram:
            self._preload_all_demos()
        self._fit_normalizers()
        if self.preload_all_demos_to_ram:
            self.proprio_extractor.close()

    def _load_env_metadata(self, file_path: str) -> dict:
        with h5py.File(file_path, "r") as handle:
            if "env_info" not in handle.attrs:
                raise KeyError(f"Missing env_info in {file_path}")
            return parse_env_info(handle.attrs["env_info"])

    def _build_index(self):
        num_loaded_traj = 0
        for file_path in tqdm(self.files, desc="Building flow_unet index"):
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
                    self._demo_meta.append((file_path, demo_key, num_steps))
                    self.sample_indices_by_meta_id[meta_id] = []
                    max_start = num_steps - self.action_horizon + 1
                    for t0 in range(0, max_start, self.stride):
                        sample_idx = len(self.index)
                        self.index.append((meta_id, t0))
                        self.sample_indices_by_meta_id[meta_id].append(sample_idx)
                    num_loaded_traj += 1

        if len(self.index) == 0:
            raise RuntimeError(f"No valid samples found in {self.data_dir}")
        print(f"flow_unet trajectories used for training: {len(self._demo_meta)}")

    def _get_handle(self, file_path: str):
        pid = os.getpid()
        if pid not in self._handles:
            self._handles[pid] = {}
        if file_path not in self._handles[pid]:
            self._handles[pid][file_path] = h5py.File(file_path, "r", libver="latest", swmr=True)
        return self._handles[pid][file_path]

    def _get_demo_group(self, meta_id: int):
        file_path, demo_key, _ = self._demo_meta[meta_id]
        handle = self._get_handle(file_path)
        return handle["demos"][demo_key]

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
        proprio = self.proprio_extractor.extract(self._load_state(meta_id, timestep))
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

    def _materialize_demo(self, meta_id: int):
        cache_path = self._demo_cache_path(meta_id)
        if self.use_disk_cache and self._is_cache_valid(meta_id, cache_path):
            with np.load(cache_path) as cached:
                return {
                    "images": cached["images"],
                    "actions": cached["actions"],
                    "proprio": cached["proprio"],
                }

        demo_group = self._get_demo_group(meta_id)
        states = demo_group["states"][:]
        actions = demo_group["actions"][:].astype(np.float32)
        proprio = np.stack([self.proprio_extractor.extract(state) for state in states], axis=0).astype(np.float32)

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
        if self.use_disk_cache:
            np.savez(cache_path, **demo_data)
        return demo_data

    def _demo_cache_path(self, meta_id: int):
        file_path, demo_key, _ = self._demo_meta[meta_id]
        camera_tag = "--".join(self.camera_names)
        base_name = f"{Path(file_path).stem}__{demo_key}__i{self.image_size}__{camera_tag}.npz"
        return os.path.join(self.cache_dir, base_name)

    def _is_cache_valid(self, meta_id: int, cache_path: str):
        if not os.path.exists(cache_path):
            return False
        file_path, _, _ = self._demo_meta[meta_id]
        return os.path.getmtime(cache_path) >= os.path.getmtime(file_path)

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
        for sample_idx in tqdm(selected, desc="Fitting flow_unet normalizers"):
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
        }

    def close(self):
        for handle_map in self._handles.values():
            for handle in handle_map.values():
                try:
                    handle.close()
                except Exception:
                    pass
        self.proprio_extractor.close()
