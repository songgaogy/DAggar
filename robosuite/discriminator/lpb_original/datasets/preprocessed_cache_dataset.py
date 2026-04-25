"""Dataset adapter that reads preprocessed LPB-score cache `.npz` files.

This dataset is meant for original-LPB dynamics training without relying on
raw HDF5 demos. It consumes the existing on-disk cache produced by the
LPB-score preprocessing pipeline:

    data/.lpb_score_preprocessed_cache/<task>/<sha1>.npz

Each `.npz` stores:
    - images_chw: (T, V, 3, H, W) uint8
    - proprio:    (T, P) float32
    - actions:    (T, A) float32

The API matches what `robosuite.discriminator.lpb_original.train` expects:
`__getitem__` returns (obs, act, state) where obs['visual'][view] has shape
(F, 3, H, W) and obs['proprio'] has shape (F, P). `F = num_hist + num_pred`.
"""

from __future__ import annotations

import glob
import hashlib
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from diffusion_policy.common.normalize_util import (
    array_to_stats,
    get_identity_normalizer_from_stat,
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
)
from diffusion_policy.model.common.normalizer import LinearNormalizer

from dyn_model.datasets.img_transforms import (
    default_transform,
    get_eval_crop_transform_resnet,
    get_train_crop_transform_resnet,
)


_LEGACY_CAMERA_NAMES: Tuple[str, ...] = (
    "agentview",
    "robot0_robotview",
    "robot0_eye_in_hand",
)


@dataclass(frozen=True)
class _EpisodeRef:
    task: str
    npz_path: str
    length: int
    data_type: Optional[str] = None


@dataclass(frozen=True)
class _AnchorRef:
    ep_idx: int
    start_t: int


def _expand_npz_paths(cache_root: str, tasks: Sequence[str]) -> List[str]:
    root = Path(cache_root)
    out: List[str] = []
    for task in tasks:
        task_dir = root / str(task)
        if not task_dir.is_dir():
            continue
        out.extend(glob.glob(str(task_dir / "*.npz")))
    return sorted(set(out))


def _action_fingerprint(actions: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(actions, dtype=np.float32))
    h = hashlib.sha1()
    h.update(str(tuple(arr.shape)).encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()


def _infer_data_type(file_path: str) -> Optional[str]:
    path = str(file_path)
    if "/expert/" in path:
        return "expert"
    if "/success_rollout/" in path:
        return "success"
    if "/fail_rollout/" in path:
        return "fail"
    return None


def _is_enabled_cap(value: Optional[int]) -> bool:
    return value is not None and int(value) >= 0


def _build_metadata_type_lookup(metadata_cache_root: str, tasks: Sequence[str]) -> Dict[str, Dict[str, str]]:
    root = Path(metadata_cache_root)
    out: Dict[str, Dict[str, str]] = {str(task): {} for task in tasks}
    for task in tasks:
        task_name = str(task)
        task_dir = root / task_name
        if not task_dir.is_dir():
            continue
        for npz_path in sorted(task_dir.glob("*.npz")):
            try:
                with np.load(str(npz_path), allow_pickle=True) as data:
                    if "actions" not in data or "file_path" not in data:
                        continue
                    data_type = _infer_data_type(str(data["file_path"].item()))
                    if data_type is None:
                        continue
                    out[task_name][_action_fingerprint(data["actions"])] = data_type
            except Exception:
                continue
    return out


def _select_episodes_by_type_caps(
    episodes: Sequence[_EpisodeRef],
    *,
    num_expert: int,
    num_success: int,
    num_fail: int,
) -> List[_EpisodeRef]:
    caps = {
        "expert": None if int(num_expert) < 0 else int(num_expert),
        "success": None if int(num_success) < 0 else int(num_success),
        "fail": None if int(num_fail) < 0 else int(num_fail),
    }
    counts: Dict[Tuple[str, str], int] = {}
    selected: List[_EpisodeRef] = []
    missing_type = 0
    for ep in episodes:
        if ep.data_type is None:
            missing_type += 1
            continue
        cap = caps.get(ep.data_type)
        key = (ep.task, ep.data_type)
        used = counts.get(key, 0)
        if cap is not None and used >= cap:
            continue
        selected.append(ep)
        counts[key] = used + 1

    if missing_type > 0:
        raise RuntimeError(
            f"Could not recover expert/success/fail type for {missing_type} cached episodes. "
            "Check env.metadata_cache_root or rebuild the preprocessed cache with metadata."
        )
    return selected


class PreprocessedCacheDynamicsModelDataset(Dataset):
    """Transition dataset backed by preprocessed `.npz` episodes."""

    def __init__(
        self,
        zarr_path=None,  # ignored; kept for config-compat
        *,
        cache_root: str = "data/.lpb_score_preprocessed_cache",
        tasks: Sequence[str] = ("PickPlaceBread",),
        camera_to_view: Optional[Dict[str, int]] = None,
        num_hist: int = 1,
        num_pred: int = 1,
        frameskip: int = 1,
        view_names: Sequence[str] = ("agentview",),
        abs_action: bool = False,
        use_crop: bool = False,
        train: bool = True,
        shape_obs: Optional[dict] = None,
        original_img_size: int = 128,
        cropped_img_size: int = 128,
        action_dim: int = 7,
        proprio_indices: Optional[Sequence[int]] = None,
        max_trajectories: Optional[int] = None,
        max_cached_episodes: int = 2,
        load_all_into_ram: bool = False,
        metadata_cache_root: str = "data/.lpb_score_cache",
        num_expert: int = -1,
        num_success: int = -1,
        num_fail: int = -1,
    ) -> None:
        super().__init__()
        self.cache_root = str(Path(cache_root).resolve())
        self.tasks = list(tasks)
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

        cam_map = dict((n, i) for i, n in enumerate(_LEGACY_CAMERA_NAMES))
        if camera_to_view is not None:
            cam_map.update({str(k): int(v) for k, v in camera_to_view.items()})
        self._camera_to_view = cam_map

        self._proprio_indices = (
            None if proprio_indices is None else np.asarray(proprio_indices, dtype=np.int64)
        )

        npz_paths = _expand_npz_paths(self.cache_root, self.tasks)
        if not npz_paths:
            raise FileNotFoundError(
                f"No *.npz found under cache_root={self.cache_root} for tasks={self.tasks}"
            )

        use_type_caps = any(
            _is_enabled_cap(v) for v in (num_expert, num_success, num_fail)
        )
        metadata_type_lookup = (
            _build_metadata_type_lookup(str(Path(metadata_cache_root).resolve()), self.tasks)
            if use_type_caps
            else {}
        )

        # Build episodes list with optional cap (treated as per-task cap if provided).
        task_counts: Dict[str, int] = {}
        episodes: List[_EpisodeRef] = []
        for npz_path in npz_paths:
            task = Path(npz_path).parent.name
            if not use_type_caps and max_trajectories is not None and int(max_trajectories) > 0:
                if task_counts.get(task, 0) >= int(max_trajectories):
                    continue
            with np.load(npz_path) as data:
                images = data["images_chw"]
                proprio = data["proprio"]
                actions = data["actions"]
                length = int(
                    min(
                        int(images.shape[0]),
                        int(proprio.shape[0]),
                        int(actions.shape[0]),
                    )
                )
            if length <= (self.num_pred * self.frameskip):
                continue
            data_type = None
            if use_type_caps:
                data_type = metadata_type_lookup.get(str(task), {}).get(_action_fingerprint(actions))
            episodes.append(
                _EpisodeRef(
                    task=str(task),
                    npz_path=str(npz_path),
                    length=length,
                    data_type=data_type,
                )
            )
            task_counts[task] = task_counts.get(task, 0) + 1

        if use_type_caps:
            episodes = _select_episodes_by_type_caps(
                episodes,
                num_expert=int(num_expert),
                num_success=int(num_success),
                num_fail=int(num_fail),
            )
            if max_trajectories is not None and int(max_trajectories) > 0:
                capped: List[_EpisodeRef] = []
                task_counts = {}
                for ep in episodes:
                    if task_counts.get(ep.task, 0) >= int(max_trajectories):
                        continue
                    capped.append(ep)
                    task_counts[ep.task] = task_counts.get(ep.task, 0) + 1
                episodes = capped

        if not episodes:
            raise RuntimeError("No usable cached episodes found after filtering.")
        self._episodes = episodes

        # Enumerate valid anchor indices over all episodes.
        anchors: List[_AnchorRef] = []
        for ep_idx, ep in enumerate(self._episodes):
            # Need (num_frames-1) future obs frames at stride frameskip.
            max_start = int(ep.length) - self.num_pred * self.frameskip - 1
            if max_start >= 0:
                anchors.extend(_AnchorRef(ep_idx=ep_idx, start_t=t) for t in range(0, max_start))
        if not anchors:
            raise RuntimeError("No valid anchors found in cached episodes.")
        self._anchors = anchors

        self.load_all_into_ram = bool(load_all_into_ram)
        # Small LRU cache for decoded episodes to reduce disk I/O.
        # IMPORTANT: this is bounded by default, otherwise each DataLoader worker
        # can keep a full dataset copy in RAM.
        self._max_cached_episodes = 0 if self.load_all_into_ram else int(max_cached_episodes)
        self._memo: "OrderedDict[str, Dict[str, np.ndarray]]" = OrderedDict()

        if self.use_crop:
            self.transform = (
                get_train_crop_transform_resnet(self.original_img_size, self.cropped_img_size)
                if self.train
                else get_eval_crop_transform_resnet(self.original_img_size, self.cropped_img_size)
            )
        else:
            self.transform = default_transform()

        # Precompute stats for normalizer from a light subsample (actions + proprio).
        # Images are normalized by range normalizer; no need to materialize all frames here.
        # Stratify over tasks so multi-task runs don't bias stats toward whichever task
        # sorts first alphabetically.
        by_task: Dict[str, List[_EpisodeRef]] = {}
        for ep in self._episodes:
            by_task.setdefault(ep.task, []).append(ep)
        per_task_budget = max(1, 64 // max(1, len(by_task)))
        sampled_eps: List[_EpisodeRef] = []
        for eps in by_task.values():
            sampled_eps.extend(eps[:per_task_budget])

        props: List[np.ndarray] = []
        acts: List[np.ndarray] = []
        for ep in sampled_eps:
            demo = self._load_npz(ep.npz_path)
            props.append(demo["proprio"])
            acts.append(demo["actions"])
        self._stats_proprio = array_to_stats(np.concatenate(props, axis=0))
        self._stats_action = array_to_stats(np.concatenate(acts, axis=0))

        self.states_dim = int(props[0].shape[1])
        self.proprio_dim = int(
            self._proprio_indices.shape[0] if self._proprio_indices is not None else props[0].shape[1]
        )

        if self.load_all_into_ram:
            for ep in self._episodes:
                if ep.npz_path not in self._memo:
                    self._memo[ep.npz_path] = self._read_npz(ep.npz_path)

    @staticmethod
    def _read_npz(path: str) -> Dict[str, np.ndarray]:
        with np.load(path) as data:
            images = np.asarray(data["images_chw"])
            proprio = np.asarray(data["proprio"], dtype=np.float32)
            actions = np.asarray(data["actions"], dtype=np.float32)
        return {"images_chw": images, "proprio": proprio, "actions": actions}

    def _load_npz(self, path: str) -> Dict[str, np.ndarray]:
        if self.load_all_into_ram:
            cached = self._memo.get(path)
            if cached is not None:
                return cached
            payload = self._read_npz(path)
            self._memo[path] = payload
            return payload
        if self._max_cached_episodes > 0:
            cached = self._memo.get(path)
            if cached is not None:
                # refresh LRU position
                self._memo.move_to_end(path)
                return cached
        payload = self._read_npz(path)
        if self._max_cached_episodes > 0:
            self._memo[path] = payload
            self._memo.move_to_end(path)
            while len(self._memo) > self._max_cached_episodes:
                self._memo.popitem(last=False)
        return payload

    def __len__(self) -> int:
        return int(len(self._anchors))

    def __getitem__(self, idx: int):
        ref = self._anchors[int(idx)]
        ep = self._episodes[ref.ep_idx]
        demo = self._load_npz(ep.npz_path)

        start = int(ref.start_t)
        end = start + self.num_frames * self.frameskip
        obs_indices = list(range(start, end, self.frameskip))
        action_indices = list(range(start, end))
        action_indices[-self.frameskip :] = [obs_indices[-1] - 1] * self.frameskip

        obs: Dict[str, Dict] = {"visual": {}}
        images = demo["images_chw"]  # (T, V, 3, H, W)
        proprio = demo["proprio"]
        actions = demo["actions"]

        length = min(int(images.shape[0]), int(proprio.shape[0]), int(actions.shape[0]))
        obs_indices = [i for i in obs_indices if 0 <= i < length]
        action_indices = [min(max(i, 0), length - 1) for i in action_indices]

        for vname in self.view_names:
            cam_idx = self._camera_to_view.get(str(vname), None)
            if cam_idx is None:
                raise KeyError(f"Unknown view name {vname!r}. Known: {sorted(self._camera_to_view)}")
            arr = images[obs_indices, cam_idx]  # (F, 3, H, W) uint8
            if arr.shape[-2] != self.original_img_size or arr.shape[-1] != self.original_img_size:
                raise ValueError(
                    f"Cached images for view={vname} have shape {arr.shape[-2:]} "
                    f"!= original_img_size={self.original_img_size}"
                )
            obs["visual"][vname] = torch.from_numpy(arr.astype(np.float32) / 255.0)

        prop = proprio[obs_indices].astype(np.float32, copy=False)
        if self._proprio_indices is not None:
            if int(self._proprio_indices.max()) >= int(prop.shape[1]):
                raise ValueError(f"proprio_indices out of range for proprio_dim={prop.shape[1]}")
            prop = prop[:, self._proprio_indices]
        obs["proprio"] = torch.from_numpy(prop)

        act = actions[action_indices].astype(np.float32, copy=False)
        if act.shape[1] < self.original_action_dim:
            pad = np.zeros((act.shape[0], self.original_action_dim - act.shape[1]), dtype=np.float32)
            act = np.concatenate([act, pad], axis=1)
        elif act.shape[1] > self.original_action_dim:
            act = act[:, : self.original_action_dim]
        act = torch.from_numpy(act)

        state = obs["proprio"]
        return obs, act, state

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        act_normalizer = get_identity_normalizer_from_stat(self._stats_action)
        normalizer["act"] = act_normalizer
        normalizer["state"] = get_range_normalizer_from_stat(self._stats_proprio)
        for v in self.view_names:
            normalizer[v] = get_image_range_normalizer()
        return normalizer
