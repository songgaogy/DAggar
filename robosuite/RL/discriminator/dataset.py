import os
import glob
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def _list_hdf5_files(
    root_dir: str,
    subdirs: Sequence[str],
    step_dirs: Optional[Sequence[str]] = None,
) -> List[str]:
    files: List[str] = []
    for sd in subdirs:
        base = os.path.join(root_dir, sd)

        if step_dirs is None:
            patterns = [
                os.path.join(base, "*.hdf5"),
                os.path.join(base, "*.h5"),
                os.path.join(base, "**", "*.hdf5"),
                os.path.join(base, "**", "*.h5"),
            ]
        else:
            patterns = []
            for st in step_dirs:
                patterns.extend(
                    [
                        os.path.join(base, st, "*.hdf5"),
                        os.path.join(base, st, "*.h5"),
                        os.path.join(base, st, "**", "*.hdf5"),
                        os.path.join(base, st, "**", "*.h5"),
                    ]
                )

        for pat in patterns:
            files.extend(glob.glob(pat, recursive=True))

    return sorted(list(set(files)))


@dataclass(frozen=True)
class H5IndexItem:
    path: str
    demo_key: str
    step_idx: int
    label: int


class PandaLiftH5StepsDataset(Dataset):
    """
    Loads per-step samples from HDF5 files produced by collect_pure_fail_lift.py.

    Expected structure (per file):
      /data/demo_0/{states, actions, rewards, dones, agentview_image, robot0_eye_in_hand_image}
      /data/demo_1/...
    """

    def __init__(
        self,
        root_dir: str,
        success_subdirs: Sequence[str] = ("success",),
        fail_subdirs: Sequence[str] = ("fail", "pure_fail"),
        use_images: bool = True,
        use_state_action: bool = True,
        max_steps_per_demo: Optional[int] = None,
        image_key_agent: str = "agentview_image",
        image_key_wrist: str = "robot0_eye_in_hand_image",
        seed: int = 0,
        cache_index_path: Optional[str] = None,
        validate_cache_files: bool = True,
        success_step_dirs: Optional[Sequence[str]] = None,
        fail_step_dirs: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        assert use_images or use_state_action, "At least one of use_images/use_state_action must be True."

        self.root_dir = root_dir
        self.use_images = use_images
        self.use_state_action = use_state_action
        self.max_steps_per_demo = max_steps_per_demo
        self.image_key_agent = image_key_agent
        self.image_key_wrist = image_key_wrist
        self.rng = random.Random(seed)

        self.success_files = _list_hdf5_files(root_dir, success_subdirs, step_dirs=success_step_dirs)
        self.fail_files = _list_hdf5_files(root_dir, fail_subdirs, step_dirs=fail_step_dirs)

        if len(self.success_files) == 0:
            raise FileNotFoundError(f"No success hdf5 files found under: {root_dir} subdirs={success_subdirs}")
        if len(self.fail_files) == 0:
            raise FileNotFoundError(f"No fail hdf5 files found under: {root_dir} subdirs={fail_subdirs}")

        self.items: List[H5IndexItem] = []
        self._cache_validate = validate_cache_files

        if cache_index_path is not None and os.path.exists(cache_index_path):
            self._load_index(cache_index_path)
        else:
            self._build_index()
            if cache_index_path is not None:
                self._save_index(cache_index_path)

    def _save_index(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)

        items_arr = np.array(
            [(it.path, it.demo_key, it.step_idx, it.label) for it in self.items],
            dtype=object,
        )
        success_arr = np.array(self.success_files, dtype=object)
        fail_arr = np.array(self.fail_files, dtype=object)

        # Use .npz to store metadata safely.
        # If user passes a .npy path, we still write an .npz next to it.
        base, ext = os.path.splitext(path)
        out_path = path if ext.lower() == ".npz" else (base + ".npz")

        np.savez(out_path, items=items_arr, success_files=success_arr, fail_files=fail_arr)

    def _load_index(self, path: str) -> None:
        base, ext = os.path.splitext(path)
        load_path = path if ext.lower() == ".npz" else (base + ".npz")
        if not os.path.exists(load_path):
            # Backward-compat: old .npy without metadata
            arr = np.load(path, allow_pickle=True)
            self.items = [H5IndexItem(str(p), str(k), int(i), int(y)) for (p, k, i, y) in arr.tolist()]
            return

        data = np.load(load_path, allow_pickle=True)
        items_arr = data["items"]

        if self._cache_validate:
            cached_success = set([str(x) for x in data["success_files"].tolist()])
            cached_fail = set([str(x) for x in data["fail_files"].tolist()])
            cur_success = set(self.success_files)
            cur_fail = set(self.fail_files)
            if cached_success != cur_success or cached_fail != cur_fail:
                raise RuntimeError(
                    "Index cache does not match current file list. "
                    "Delete the cache or use a different cache_index_path."
                )

        self.items = [H5IndexItem(str(p), str(k), int(i), int(y)) for (p, k, i, y) in items_arr.tolist()]

    def _build_index(self) -> None:
        def add_file(path: str, label: int) -> None:
            with h5py.File(path, "r") as f:
                if "data" not in f:
                    return
                grp = f["data"]
                for demo_key in grp.keys():
                    demo = grp[demo_key]
                    if "states" not in demo or "actions" not in demo:
                        continue
                    n = int(demo["states"].shape[0])
                    if self.max_steps_per_demo is not None:
                        n = min(n, int(self.max_steps_per_demo))
                    for i in range(n):
                        self.items.append(H5IndexItem(path=path, demo_key=demo_key, step_idx=i, label=label))

        for p in self.success_files:
            add_file(p, 1)
        for p in self.fail_files:
            add_file(p, 0)

        if len(self.items) == 0:
            raise RuntimeError("No samples were indexed from the provided HDF5 files.")

        self.rng.shuffle(self.items)

    def __len__(self) -> int:
        return len(self.items)

    @staticmethod
    def _to_chw_uint8(img_hwc: np.ndarray) -> np.ndarray:
        assert img_hwc.ndim == 3 and img_hwc.shape[2] in (1, 3, 4)
        if img_hwc.shape[2] == 4:
            img_hwc = img_hwc[:, :, :3]
        img = img_hwc.astype(np.uint8)
        img = np.transpose(img, (2, 0, 1))
        return img

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self.items[idx]
        with h5py.File(it.path, "r") as f:
            demo = f["data"][it.demo_key]

            out: Dict[str, Any] = {}
            out["label"] = np.int64(it.label)

            if self.use_state_action:
                states = demo["states"][it.step_idx].astype(np.float32)
                actions = demo["actions"][it.step_idx].astype(np.float32)
                out["state"] = states
                out["action"] = actions

            if self.use_images:
                img_a = demo[self.image_key_agent][it.step_idx]
                img_w = demo[self.image_key_wrist][it.step_idx]
                out["img_agent"] = self._to_chw_uint8(img_a)
                out["img_wrist"] = self._to_chw_uint8(img_w)

            return out


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    keys = batch[0].keys()
    out: Dict[str, torch.Tensor] = {}
    for k in keys:
        if k == "label":
            out[k] = torch.tensor([b[k] for b in batch], dtype=torch.long)
        elif k in ("state", "action"):
            out[k] = torch.tensor(np.stack([b[k] for b in batch], axis=0), dtype=torch.float32)
        elif k in ("img_agent", "img_wrist"):
            out[k] = torch.tensor(np.stack([b[k] for b in batch], axis=0), dtype=torch.uint8)
        else:
            raise KeyError(f"Unknown key in batch: {k}")
    return out