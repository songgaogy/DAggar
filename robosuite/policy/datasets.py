import os
import glob
import h5py
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset


def _list_hdf5_files(data_dir: str):
    files = sorted(glob.glob(os.path.join(data_dir, "*.hdf5")))
    if len(files) == 0:
        raise FileNotFoundError(f"No .hdf5 files found in {data_dir}")
    return files


class PandaLiftFlowDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        proprio_extractor,
        camera_name: str = "agentview",
        history_len: int = 1,
        horizon: int = 1,
        stride: int = 1,
        image_size: int = 128,
        normalize: bool = True,
        max_demos_per_file: int | None = None,
        cache_proprio: bool = True,
    ):
        self.data_dir = data_dir
        self.files = _list_hdf5_files(data_dir)
        self.camera_name = camera_name
        self.history_len = int(history_len)
        self.horizon = int(horizon)
        self.stride = int(stride)
        self.image_size = int(image_size)
        self.normalize = bool(normalize)
        self.proprio_extractor = proprio_extractor
        self.max_demos_per_file = max_demos_per_file
        self.cache_proprio = cache_proprio

        self.index = []
        self._demo_meta = []
        self._handles = {}
        self._proprio_cache = {}

        self._build_index()
        print("finish building index!")
        self._fit_normalizers()
        print("finish fitting normalizers!")

    def _build_index(self):
        for fp in tqdm(self.files, desc="Building index (files)..."):
            with h5py.File(fp, "r") as f:
                demos_group = f["demos"]
                demo_keys = sorted(list(demos_group.keys()))
                if self.max_demos_per_file is not None:
                    demo_keys = demo_keys[: self.max_demos_per_file]

                for dk in tqdm(demo_keys, desc="  demos", leave=False):
                    demo = demos_group[dk]
                    states = demo["states"]
                    actions = demo["actions"]
                    images = demo["observations"][self.camera_name]["images"]

                    T = min(states.shape[0], actions.shape[0], images.shape[0])
                    if T < (self.history_len + self.horizon):
                        continue

                    meta_id = len(self._demo_meta)
                    self._demo_meta.append((fp, dk, T))

                    max_t = T - (self.history_len + self.horizon) + 1
                    for t0 in range(0, max_t, self.stride):
                        self.index.append((meta_id, t0))

    def _get_handle(self, fp):
        pid = os.getpid()
        if pid not in self._handles:
            self._handles[pid] = {}
        if fp not in self._handles[pid]:
            self._handles[pid][fp] = h5py.File(fp, "r", libver="latest", swmr=True)
        return self._handles[pid][fp]

    def _load_demo_arrays(self, meta_id):
        fp, dk, T = self._demo_meta[meta_id]
        f = self._get_handle(fp)
        demo = f["demos"][dk]
        states = demo["states"][:T]
        actions = demo["actions"][:T]
        images = demo["observations"][self.camera_name]["images"][:T]
        return states, actions, images

    def _get_proprio_seq(self, meta_id: int, states: np.ndarray):
        if not self.cache_proprio:
            proprio = np.stack([self.proprio_extractor.extract(s) for s in states], axis=0)
            return proprio

        key = (self._demo_meta[meta_id][0], self._demo_meta[meta_id][1])
        if key in self._proprio_cache:
            return self._proprio_cache[key]

        proprio = np.stack([self.proprio_extractor.extract(s) for s in states], axis=0)
        self._proprio_cache[key] = proprio
        return proprio

    def _resize_center_crop(self, img: np.ndarray):
        # img: HWC uint8
        H, W = img.shape[0], img.shape[1]
        s = min(H, W)
        y0 = (H - s) // 2
        x0 = (W - s) // 2
        crop = img[y0 : y0 + s, x0 : x0 + s, :]
        if s == self.image_size:
            return crop
        # simple nearest resize to avoid extra deps
        ys = np.linspace(0, s - 1, self.image_size).astype(np.int32)
        xs = np.linspace(0, s - 1, self.image_size).astype(np.int32)
        out = crop[ys][:, xs]
        return out

    def _fit_normalizers(self):
        if not self.normalize:
            self.act_mean = None
            self.act_std = None
            self.prop_mean = None
            self.prop_std = None
            return

        act_list = []
        prop_list = []

        n_samples = min(len(self.index), 500)
        sel = np.random.choice(len(self.index), size=n_samples, replace=False)

        for idx in tqdm(sel, desc="fit normalizers..."):
            meta_id, t0 = self.index[int(idx)]
            states, actions, images = self._load_demo_arrays(meta_id)
            try:
                prop = self._get_proprio_seq(meta_id, states)
                a = actions[t0 : t0 + self.horizon].reshape(-1).astype(np.float32)
                p = prop[t0 : t0 + self.history_len].reshape(-1).astype(np.float32)
                act_list.append(a)
                prop_list.append(p)
            except:
                raise Exception

        act_arr = np.stack(act_list, axis=0)
        prop_arr = np.stack(prop_list, axis=0)

        self.act_mean = act_arr.mean(axis=0)
        self.act_std = act_arr.std(axis=0) + 1e-6
        self.prop_mean = prop_arr.mean(axis=0)
        self.prop_std = prop_arr.std(axis=0) + 1e-6

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i: int):
        meta_id, t0 = self.index[i]
        states, actions, images = self._load_demo_arrays(meta_id)
        try:
            prop = self._get_proprio_seq(meta_id, states)

            img_seq = images[t0 : t0 + self.history_len]
            prop_seq = prop[t0 : t0 + self.history_len]
            act_seq = actions[t0 : t0 + self.horizon].reshape(-1).astype(np.float32)

            img_list = []
            for k in range(self.history_len):
                img = self._resize_center_crop(img_seq[k])
                img = img.astype(np.float32) / 255.0

                img = np.transpose(img, (2, 0, 1))  # CHW
                img_list.append(img)
            img_stack = np.stack(img_list, axis=0)  # K, C, H, W

            prop_flat = prop_seq.reshape(-1).astype(np.float32)

            if self.normalize:
                act_seq = (act_seq - self.act_mean) / self.act_std
                prop_flat = (prop_flat - self.prop_mean) / self.prop_std

            sample = dict(
                images=torch.from_numpy(img_stack),
                proprio=torch.from_numpy(prop_flat),
                actions=torch.from_numpy(act_seq),
            )
            return sample
        except:
            raise Exception