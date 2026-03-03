import glob
import os
from dataclasses import dataclass
from typing import Callable

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DemoInfo:
    file_path: str
    demo_key: str
    length: int


def center_crop_resize(img: np.ndarray, out_size: int, resize_cache: dict[int, tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    h, w = int(img.shape[0]), int(img.shape[1])
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    crop = img[y0 : y0 + side, x0 : x0 + side, :]
    if side == out_size:
        return crop

    if side not in resize_cache:
        ys = np.linspace(0, side - 1, out_size).astype(np.int32)
        xs = np.linspace(0, side - 1, out_size).astype(np.int32)
        resize_cache[side] = (ys, xs)
    ys, xs = resize_cache[side]
    return crop[ys][:, xs]


def list_hdf5_files(data_dir: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(data_dir, "*.hdf5")))
    if not files:
        raise FileNotFoundError(f"No .hdf5 files found in {data_dir}")
    return files


def scan_demos(
    data_dir: str,
    camera_name: str,
    history_len: int,
    max_trajectories: int | None = None,
) -> list[DemoInfo]:
    demos: list[DemoInfo] = []
    for file_path in list_hdf5_files(data_dir):
        with h5py.File(file_path, "r") as f:
            demo_group = f["demos"]
            demo_keys = sorted(demo_group.keys())
            for demo_key in demo_keys:
                demo = demo_group[demo_key]
                if "actions" not in demo or "states" not in demo:
                    continue
                if "observations" not in demo or camera_name not in demo["observations"]:
                    continue
                if "images" not in demo["observations"][camera_name]:
                    continue

                t = min(
                    int(demo["actions"].shape[0]),
                    int(demo["states"].shape[0]),
                    int(demo["observations"][camera_name]["images"].shape[0]),
                )
                if t < history_len:
                    continue
                demos.append(DemoInfo(file_path=file_path, demo_key=demo_key, length=t))
                if max_trajectories is not None and len(demos) >= int(max_trajectories):
                    return demos
    return demos


def split_demo_infos(
    demo_infos: list[DemoInfo],
    eval_ratio: float,
    seed: int,
) -> tuple[list[DemoInfo], list[DemoInfo]]:
    if not demo_infos:
        return [], []

    rng = np.random.default_rng(seed)
    order = np.arange(len(demo_infos))
    rng.shuffle(order)

    n_eval = int(round(len(demo_infos) * float(eval_ratio)))
    if len(demo_infos) >= 2:
        n_eval = min(max(n_eval, 1), len(demo_infos) - 1)
    else:
        n_eval = 0

    eval_ids = set(order[:n_eval].tolist())
    train_infos = [d for i, d in enumerate(demo_infos) if i not in eval_ids]
    eval_infos = [d for i, d in enumerate(demo_infos) if i in eval_ids]
    return train_infos, eval_infos


class StateActionHdf5Dataset(Dataset):
    def __init__(
        self,
        demo_infos: list[DemoInfo],
        proprio_extractor: Callable[[np.ndarray], np.ndarray],
        camera_name: str,
        history_len: int,
        image_size: int,
        label: float,
        normalize: bool = True,
        prop_mean: np.ndarray | None = None,
        prop_std: np.ndarray | None = None,
        act_mean: np.ndarray | None = None,
        act_std: np.ndarray | None = None,
    ):
        self.demo_infos = list(demo_infos)
        self.camera_name = camera_name
        self.history_len = int(history_len)
        self.image_size = int(image_size)
        self.label = float(label)
        self.normalize = bool(normalize)
        self.proprio_extractor = proprio_extractor
        self.prop_mean = prop_mean
        self.prop_std = prop_std
        self.act_mean = act_mean
        self.act_std = act_std
        self._resolved_act_mean = None
        self._resolved_act_std = None

        self.index: list[tuple[int, int]] = []
        self._handles: dict[int, dict[str, h5py.File]] = {}
        self._resize_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        for demo_id, info in enumerate(self.demo_infos):
            for t in range(self.history_len - 1, info.length):
                self.index.append((demo_id, t))

        self._resolve_action_stats()

    def __len__(self) -> int:
        return len(self.index)

    def _get_handle(self, file_path: str) -> h5py.File:
        pid = os.getpid()
        if pid not in self._handles:
            self._handles[pid] = {}
        if file_path not in self._handles[pid]:
            self._handles[pid][file_path] = h5py.File(file_path, "r", libver="latest", swmr=True)
        return self._handles[pid][file_path]

    def _infer_action_dim(self) -> int | None:
        if not self.demo_infos:
            return None
        info = self.demo_infos[0]
        with h5py.File(info.file_path, "r") as f:
            action_ds = f["demos"][info.demo_key]["actions"]
            if action_ds.ndim < 2:
                return None
            return int(action_ds.shape[-1])

    def _resolve_action_stats(self) -> None:
        if self.act_mean is None or self.act_std is None:
            self._resolved_act_mean = None
            self._resolved_act_std = None
            return

        action_dim = self._infer_action_dim()
        if action_dim is None:
            self._resolved_act_mean = None
            self._resolved_act_std = None
            return

        mean = np.asarray(self.act_mean, dtype=np.float32).reshape(-1)
        std = np.asarray(self.act_std, dtype=np.float32).reshape(-1)
        if mean.shape != std.shape:
            self._resolved_act_mean = None
            self._resolved_act_std = None
            return

        if mean.size == action_dim:
            self._resolved_act_mean = mean
            self._resolved_act_std = std
            return

        if mean.size % action_dim == 0:
            chunks = mean.size // action_dim
            mean_2d = mean.reshape(chunks, action_dim)
            std_2d = std.reshape(chunks, action_dim)
            merged_mean = mean_2d.mean(axis=0)
            second_moment = (std_2d**2 + mean_2d**2).mean(axis=0)
            merged_std = np.sqrt(np.maximum(second_moment - merged_mean**2, 1e-12))
            self._resolved_act_mean = merged_mean
            self._resolved_act_std = merged_std
            return

        self._resolved_act_mean = None
        self._resolved_act_std = None

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        demo_id, t = self.index[idx]
        info = self.demo_infos[demo_id]
        f = self._get_handle(info.file_path)
        demo = f["demos"][info.demo_key]

        t0 = t - self.history_len + 1
        t1 = t + 1

        image_seq = demo["observations"][self.camera_name]["images"][t0:t1]
        state_seq = demo["states"][t0:t1]
        action = demo["actions"][t].astype(np.float32)

        img_list = []
        for k in range(self.history_len):
            img = center_crop_resize(image_seq[k], self.image_size, self._resize_cache)
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            img_list.append(img)
        images = np.stack(img_list, axis=0).astype(np.float32)

        prop_seq = np.stack([self.proprio_extractor(s).astype(np.float32) for s in state_seq], axis=0)
        proprio = prop_seq.reshape(-1).astype(np.float32)

        if self.normalize:
            if self.prop_mean is not None and self.prop_std is not None:
                proprio = (proprio - self.prop_mean) / (self.prop_std + 1e-6)
            if self._resolved_act_mean is not None and self._resolved_act_std is not None:
                action = (action - self._resolved_act_mean) / (self._resolved_act_std + 1e-6)

        return {
            "images": torch.from_numpy(images),
            "proprio": torch.from_numpy(proprio),
            "actions": torch.from_numpy(action.astype(np.float32)),
            "label": torch.tensor(self.label, dtype=torch.float32),
        }
