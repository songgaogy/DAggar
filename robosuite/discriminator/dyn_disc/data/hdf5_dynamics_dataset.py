"""HDF5 dataset adapter for TACO representation pretraining.

API contract matches `dyn_model.datasets.robomimic_dset.RobomimicImageDynamicsModelDataset`
so the original LPB train.py can swap zarr for HDF5 with a config-only change.

Expected HDF5 layout per file (matches robosuite/discriminator/lpb/dataset.py):
    demos/<demo_key>/states                       (T, state_dim)
    demos/<demo_key>/actions                      (T, action_dim)
    demos/<demo_key>/observations/<view>/images   (T, H, W, 3)

The dataset returns 3-tuples (obs, act, state) where:
    obs['visual'][view] : (view_frames, 3, H, W) float in [0, 1], or uint8
                          when ``return_uint8_images=True``. ``view_frames``
                          defaults to ``num_frames`` and can be configured per view.
    obs['proprio']      : (num_frames, proprio_dim) float
    act                 : (num_frames * frameskip, action_dim) float
    state               : (num_frames, state_dim) float
with `num_frames = num_hist + num_pred`. Indexing follows the upstream pattern
(obs sampled every `frameskip` steps; the trailing `frameskip` actions are
clipped to the last sampled index minus one — bug-for-bug compatible).
"""

from __future__ import annotations

import glob
import os
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from robosuite.discriminator.dyn_disc.utils.normalize_util import (
    array_to_stats,
    get_identity_normalizer_from_stat,
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
)
from robosuite.discriminator.dyn_disc.utils.normalizer import LinearNormalizer

from robosuite.discriminator.dyn_disc.data.img_transforms import (
    default_transform,
    get_eval_crop_transform_resnet,
    get_train_crop_transform_resnet,
)


def _expand_hdf5_inputs(paths: Optional[Sequence[str]]) -> List[str]:
    if not paths:
        return []
    if isinstance(paths, str):
        paths = [paths]
    out: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            out.extend(sorted(glob.glob(os.path.join(p, "*.hdf5"))))
        elif os.path.isfile(p) and p.endswith(".hdf5"):
            out.append(p)
    return sorted(set(out))


def _infer_task_name(path: str, task_names: Sequence[str]) -> Optional[str]:
    parts = set(Path(path).parts)
    for name in task_names:
        if str(name) in parts:
            return str(name)
    return None


class HDF5DynamicsModelDataset(Dataset):
    """HDF5 backend exposing the same interface as the upstream zarr datasets."""

    def __init__(
        self,
        zarr_path,                # accept hdf5 path(s) here for config-compat
        num_hist: int = 1,
        num_pred: int = 1,
        frameskip: int = 8,
        view_names: Sequence[str] = ("agentview",),
        abs_action: bool = False,
        use_crop: bool = False,
        use_cache: bool = False,
        cache_dir: Optional[str] = None,
        train: bool = True,
        shape_obs: Optional[dict] = None,
        original_img_size: int = 140,
        cropped_img_size: int = 128,
        action_dim: int = 7,
        proprio_indices: Optional[Sequence[int]] = None,
        proprio_map: Optional[Mapping[str, Mapping[str, Sequence[int]]]] = None,
        max_trajectories: Optional[int] = None,
        causal_action_chunks: bool = False,
        return_uint8_images: bool = False,
        view_frame_counts: Optional[Mapping[str, int]] = None,
    ) -> None:
        super().__init__()

        self.abs_action = bool(abs_action)
        self.original_img_size = int(original_img_size)
        self.cropped_img_size = int(cropped_img_size)
        self.original_action_dim = int(action_dim)
        self.view_names = list(view_names)
        self.num_hist = int(num_hist)
        self.num_pred = int(num_pred)
        self.frameskip = int(frameskip)
        self.num_frames = self.num_hist + self.num_pred
        self.use_crop = bool(use_crop)
        self.train = bool(train)
        self.action_dim = self.original_action_dim * self.frameskip
        self.causal_action_chunks = bool(causal_action_chunks)
        self.return_uint8_images = bool(return_uint8_images)
        self.view_frame_counts = (
            None
            if view_frame_counts is None
            else {str(view): int(count) for view, count in view_frame_counts.items()}
        )
        self._proprio_map = dict(proprio_map) if proprio_map is not None else None

        if self.view_frame_counts is not None:
            unknown_views = set(self.view_frame_counts) - set(self.view_names)
            if unknown_views:
                raise ValueError(f"view_frame_counts contains unknown views: {sorted(unknown_views)}")
            invalid_counts = {
                view: count
                for view, count in self.view_frame_counts.items()
                if count < 1 or count > self.num_frames
            }
            if invalid_counts:
                raise ValueError(
                    f"view_frame_counts must be in [1, {self.num_frames}], got {invalid_counts}"
                )

        if isinstance(zarr_path, (list, tuple)):
            files = _expand_hdf5_inputs(list(zarr_path))
        else:
            files = _expand_hdf5_inputs([str(zarr_path)])
        if not files:
            raise FileNotFoundError(f"No .hdf5 files found under {zarr_path}")

        cache_path: Optional[Path] = None
        if bool(use_cache):
            base = Path(cache_dir) if cache_dir else Path("data/.lpb_score_preprocessed_cache")
            if not base.is_absolute():
                base = Path(os.getcwd()) / base
            # Match d4disc convention: cache_root/<namespace>/<key>.*
            base = base / "lpb_original_hdf5_dyn"
            base.mkdir(parents=True, exist_ok=True)

            # d4disc-style global token for invalidation across format changes.
            global_token = "|".join(
                [
                    "lpb_original_hdf5_dyn_v1",
                    str(int(original_img_size)),
                    str(int(cropped_img_size)),
                    str(int(num_hist)),
                    str(int(num_pred)),
                    str(int(frameskip)),
                    ",".join(list(view_names)),
                    str(int(self.return_uint8_images)),
                    json.dumps(self.view_frame_counts, sort_keys=True),
                ]
            )
            meta = {
                "files": [
                    {
                        "path": str(Path(fp).resolve()),
                        "mtime": os.path.getmtime(fp),
                        "size": os.path.getsize(fp),
                    }
                    for fp in files
                ],
                "num_hist": int(num_hist),
                "num_pred": int(num_pred),
                "frameskip": int(frameskip),
                "view_names": list(view_names),
                "abs_action": bool(abs_action),
                "use_crop": bool(use_crop),
                "original_img_size": int(original_img_size),
                "cropped_img_size": int(cropped_img_size),
                "action_dim": int(action_dim),
                "proprio_indices": None if proprio_indices is None else list(map(int, proprio_indices)),
                "proprio_map": self._proprio_map,
                "max_trajectories": None if max_trajectories is None else int(max_trajectories),
                "causal_action_chunks": self.causal_action_chunks,
                "return_uint8_images": self.return_uint8_images,
                "view_frame_counts": self.view_frame_counts,
                "train": bool(train),
            }
            key = "|".join(
                [
                    global_token,
                    json.dumps(meta, sort_keys=True),
                ]
            )
            digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
            cache_path = base / f"{digest}.pt"
            self.cache_path = str(cache_path)
            if cache_path.is_file():
                try:
                    payload = torch.load(str(cache_path), map_location="cpu", weights_only=False)
                except TypeError:
                    payload = torch.load(str(cache_path), map_location="cpu")
                self._proprio_indices = None if payload.get("proprio_indices") is None else np.asarray(
                    payload["proprio_indices"], dtype=np.int64
                )
                self.states = payload["states"]
                self.actions = payload["actions"]
                self.imgs = payload["imgs"]
                self.episode_ends = payload["episode_ends"]
                self.episode_start_indices = np.concatenate(([0], self.episode_ends[:-1]))
                self.episode_end_indices = self.episode_ends - 1
                self.states_dim = int(self.states.shape[1])
                self.proprio_dim = int(self.states.shape[1])
                self.valid_anchor_indices = payload["valid_anchor_indices"]
                self.num_valid = int(self.valid_anchor_indices.shape[0])
                self.original_action_dim = int(payload["original_action_dim"])
                self.action_dim = int(payload["action_dim"])
                self.view_names = list(payload["view_names"])
                self.num_hist = int(payload["num_hist"])
                self.num_pred = int(payload["num_pred"])
                self.frameskip = int(payload["frameskip"])
                self.num_frames = int(payload["num_frames"])
                self.original_img_size = int(payload["original_img_size"])
                self.cropped_img_size = int(payload["cropped_img_size"])
                self.abs_action = bool(payload["abs_action"])
                self.use_crop = bool(payload["use_crop"])
                self.train = bool(payload["train"])
                self.causal_action_chunks = bool(payload.get("causal_action_chunks", False))
                self.return_uint8_images = bool(payload.get("return_uint8_images", False))
                self.view_frame_counts = payload.get("view_frame_counts", None)
                self._proprio_map = payload.get("proprio_map", None)

                if self.use_crop:
                    self.transform = (
                        get_train_crop_transform_resnet(self.original_img_size, self.cropped_img_size)
                        if self.train else
                        get_eval_crop_transform_resnet(self.original_img_size, self.cropped_img_size)
                    )
                else:
                    self.transform = default_transform()
                return

        self._proprio_indices = (
            None if proprio_indices is None else np.asarray(proprio_indices, dtype=np.int64)
        )
        proprio_map_indices = None
        if self._proprio_map is not None:
            proprio_map_indices = {
                str(task): np.asarray(spec["indices"], dtype=np.int64)
                for task, spec in self._proprio_map.items()
                if spec is not None and "indices" in spec
            }

        # Materialize all demos into in-memory arrays. For large datasets the user
        # can switch to a streaming version later; this matches the upstream
        # zarr dataset which also materializes everything.
        states_list: List[np.ndarray] = []
        actions_list: List[np.ndarray] = []
        imgs_per_view: Dict[str, List[np.ndarray]] = {v: [] for v in self.view_names}
        episode_ends: List[int] = []
        running = 0
        traj_count = 0

        for fp in files:
            task_name = _infer_task_name(fp, list(proprio_map_indices.keys())) if proprio_map_indices else None
            task_indices = proprio_map_indices.get(task_name) if task_name is not None else None
            if proprio_map_indices is not None and task_indices is None:
                raise ValueError(
                    f"Could not infer task for HDF5 path {fp}; known proprio_map tasks: "
                    f"{sorted(proprio_map_indices)}"
                )
            with h5py.File(fp, "r") as f:
                if "demos" not in f:
                    continue
                for demo_key in sorted(f["demos"].keys()):
                    demo = f["demos"][demo_key]
                    if "states" not in demo or "actions" not in demo or "observations" not in demo:
                        continue
                    obs_group = demo["observations"]
                    if any(v not in obs_group for v in self.view_names):
                        continue
                    cam_lengths = []
                    for v in self.view_names:
                        view = obs_group[v]
                        if "images" not in view:
                            cam_lengths.append(0)
                            break
                        cam_lengths.append(int(view["images"].shape[0]))
                    if any(L == 0 for L in cam_lengths):
                        continue
                    T = int(min(int(demo["states"].shape[0]),
                                int(demo["actions"].shape[0]),
                                *cam_lengths))
                    if T <= 0:
                        continue
                    states = np.asarray(demo["states"][:T], dtype=np.float32)
                    if task_indices is not None:
                        if int(task_indices.max()) >= int(states.shape[1]):
                            raise ValueError(
                                f"proprio_map indices out of range for task={task_name} state_dim={states.shape[1]}"
                            )
                        states = states[:, task_indices].astype(np.float32, copy=False)
                    states_list.append(states)
                    actions_list.append(np.asarray(demo["actions"][:T], dtype=np.float32))
                    for v in self.view_names:
                        imgs_per_view[v].append(np.asarray(obs_group[v]["images"][:T]))
                    running += T
                    episode_ends.append(running)
                    traj_count += 1
                    if max_trajectories is not None and max_trajectories > 0 and traj_count >= int(max_trajectories):
                        break
            if max_trajectories is not None and max_trajectories > 0 and traj_count >= int(max_trajectories):
                break

        if not states_list:
            raise RuntimeError(f"No valid demos in HDF5 inputs: {files}")

        self.states = np.concatenate(states_list, axis=0)            # (N, state_dim)
        if self._proprio_indices is not None:
            if int(self._proprio_indices.max()) >= int(self.states.shape[1]):
                raise ValueError(
                    f"proprio_indices out of range for state_dim={self.states.shape[1]}"
                )
            self.states = self.states[:, self._proprio_indices].astype(np.float32, copy=False)

        self.actions = np.concatenate(actions_list, axis=0)
        if self.actions.shape[1] < self.original_action_dim:
            pad = np.zeros((self.actions.shape[0], self.original_action_dim - self.actions.shape[1]), dtype=np.float32)
            self.actions = np.concatenate([self.actions, pad], axis=1)
        elif self.actions.shape[1] > self.original_action_dim:
            self.actions = self.actions[:, : self.original_action_dim]
        self.actions = self.actions.astype(np.float32, copy=False)

        self.imgs: Dict[str, np.ndarray] = {
            v: np.concatenate(imgs_per_view[v], axis=0) for v in self.view_names
        }
        for v in self.view_names:
            arr = self.imgs[v]
            if arr.shape[1] != self.original_img_size or arr.shape[2] != self.original_img_size:
                raise ValueError(
                    f"View {v} image shape {arr.shape[1:]} != original_img_size={self.original_img_size}. "
                    "Resize your HDF5 data or reduce original_img_size in config."
                )

        self.episode_ends = np.asarray(episode_ends, dtype=np.int64)
        self.episode_start_indices = np.concatenate(([0], self.episode_ends[:-1]))
        self.episode_end_indices = self.episode_ends - 1

        self.states_dim = int(self.states.shape[1])
        self.proprio_dim = int(self.states.shape[1])

        # Precompute valid anchor indices (mirrors RobomimicImageDynamicsModelDataset).
        valid: List[int] = []
        for start, end in zip(self.episode_start_indices, self.episode_end_indices):
            anchor_start = int(start)
            if self.causal_action_chunks:
                max_start = int(end) + 1 - self.num_frames * self.frameskip
            else:
                max_start = int(end) - self.num_pred * self.frameskip - 1
            if max_start >= anchor_start:
                valid.extend(range(anchor_start, max_start + 1))
        self.valid_anchor_indices = np.asarray(valid, dtype=np.int64)
        self.num_valid = int(self.valid_anchor_indices.shape[0])

        if self.use_crop:
            self.transform = (
                get_train_crop_transform_resnet(self.original_img_size, self.cropped_img_size)
                if self.train else
                get_eval_crop_transform_resnet(self.original_img_size, self.cropped_img_size)
            )
        else:
            self.transform = default_transform()

        if cache_path is not None:
            payload = {
                "cache_version": "lpb_original_hdf5_dyn_v1",
                "states": self.states,
                "actions": self.actions,
                "imgs": self.imgs,
                "episode_ends": self.episode_ends,
                "valid_anchor_indices": self.valid_anchor_indices,
                "view_names": self.view_names,
                "num_hist": self.num_hist,
                "num_pred": self.num_pred,
                "frameskip": self.frameskip,
                "num_frames": self.num_frames,
                "abs_action": self.abs_action,
                "use_crop": self.use_crop,
                "train": self.train,
                "original_img_size": self.original_img_size,
                "cropped_img_size": self.cropped_img_size,
                "original_action_dim": self.original_action_dim,
                "action_dim": self.action_dim,
                "proprio_indices": None if self._proprio_indices is None else self._proprio_indices.tolist(),
                "proprio_map": self._proprio_map,
                "causal_action_chunks": self.causal_action_chunks,
                "return_uint8_images": self.return_uint8_images,
                "view_frame_counts": self.view_frame_counts,
                "files": [str(Path(fp).resolve()) for fp in files],
            }
            try:
                torch.save(payload, str(cache_path))
            except Exception:
                # Cache is a best-effort optimization; ignore failures.
                pass

    def __len__(self) -> int:
        return self.num_valid

    def __getitem__(self, idx: int) -> Tuple[Dict, torch.Tensor, torch.Tensor]:
        start = int(self.valid_anchor_indices[idx])
        end = start + self.num_frames * self.frameskip
        obs_indices = list(range(start, end, self.frameskip))
        action_indices = list(range(start, end))
        if not self.causal_action_chunks:
            # Bug-for-bug compat with upstream: clip the trailing `frameskip` actions
            # back to obs_indices[-1] - 1.
            action_indices[-self.frameskip:] = [obs_indices[-1] - 1] * self.frameskip

        obs: Dict[str, Dict] = {"visual": {}}
        for v in self.view_names:
            view_obs_indices = obs_indices
            if self.view_frame_counts is not None:
                view_obs_indices = obs_indices[: self.view_frame_counts.get(v, self.num_frames)]
            arr = self.imgs[v][view_obs_indices]                # (F, H, W, 3) uint8
            arr = np.moveaxis(arr, -1, 1)
            if not self.return_uint8_images:
                arr = arr.astype(np.float32) / 255.0
            obs["visual"][v] = torch.from_numpy(arr)

        prop = self.states[obs_indices].astype(np.float32, copy=False)
        obs["proprio"] = torch.from_numpy(prop)

        act = torch.from_numpy(self.actions[action_indices].astype(np.float32, copy=False))
        state = torch.from_numpy(prop)
        return obs, act, state

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        act_stat = array_to_stats(self.actions)
        if self.abs_action:
            # No specific abs-action shaping for our HDF5 layout; identity-normalize
            # and let the user re-fit if needed.
            act_normalizer = get_identity_normalizer_from_stat(act_stat)
        else:
            act_normalizer = get_identity_normalizer_from_stat(act_stat)
        normalizer["act"] = act_normalizer
        state_stat = array_to_stats(self.states)
        normalizer["state"] = get_range_normalizer_from_stat(state_stat)
        for v in self.view_names:
            normalizer[v] = get_image_range_normalizer()
        return normalizer
