"""Dataset adapter for LPB v2 dynamics training from Agilex raw-array cache.

Consumes cache directories produced by:
    benchmark/real_world/scripts/build_cache.sh

Expected layout:
    data/.agilex_train_cache/<task>/<sha1>/
        images_chw.npy   (T, V, 3, H, W) uint8
        proprio.npy      (T, P) float32
        actions.npy      (T, A) float32
        metadata.json    includes camera_names and is_failure
"""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

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


@dataclass(frozen=True)
class _AgilexEpisodeRef:
    task: str
    cache_path: str
    length: int
    is_failure: bool


@dataclass(frozen=True)
class _AnchorRef:
    ep_idx: int
    start_t: int


def _read_metadata(cache_path: Path) -> dict:
    with open(cache_path / "metadata.json", "r") as fp:
        return json.load(fp)


def _source_name(is_failure: bool) -> str:
    return "failure" if bool(is_failure) else "success"


class AgilexCacheDynamicsModelDataset(Dataset):
    """LPB v2 transition dataset backed by real-world Agilex cache folders."""

    def __init__(
        self,
        zarr_path=None,  # ignored; kept for train.py compatibility
        *,
        cache_root: str = "data/.agilex_train_cache",
        tasks: Sequence[str] = ("candy_in_plate",),
        train_sources: Sequence[str] = ("success",),
        num_hist: int = 1,
        num_pred: int = 1,
        frameskip: int = 1,
        view_names: Sequence[str] = ("cam_high",),
        abs_action: bool = False,
        use_crop: bool = False,
        train: bool = True,
        shape_obs: Optional[dict] = None,
        original_img_size: int = 224,
        cropped_img_size: int = 224,
        action_dim: int = 7,
        proprio_indices: Optional[Sequence[int]] = None,
        max_trajectories: Optional[int] = None,
        max_cached_episodes: int = 2,
        load_all_into_ram: bool = False,
        camera_to_view: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__()
        del zarr_path, abs_action, use_crop, train, shape_obs, cropped_img_size

        self.cache_root = str(Path(cache_root).resolve())
        self.tasks = [str(t) for t in tasks]
        self.train_sources = {str(x) for x in train_sources}
        self.original_img_size = int(original_img_size)
        self.original_action_dim = int(action_dim)
        self.view_names = [str(v) for v in view_names]
        self.num_hist = int(num_hist)
        self.num_pred = int(num_pred)
        self.frameskip = int(frameskip)
        self.num_frames = self.num_hist + self.num_pred
        self.action_dim = self.original_action_dim * self.frameskip
        self.camera_to_view = {str(k): str(v) for k, v in (camera_to_view or {}).items()}

        self._proprio_indices = (
            None if proprio_indices is None else np.asarray(proprio_indices, dtype=np.int64)
        )

        episodes = self._discover_episodes(max_trajectories=max_trajectories)
        if not episodes:
            raise RuntimeError(
                f"No usable Agilex cache episodes found under {self.cache_root} "
                f"for tasks={self.tasks}, train_sources={sorted(self.train_sources)}"
            )
        self._episodes = episodes

        anchors: List[_AnchorRef] = []
        for ep_idx, ep in enumerate(self._episodes):
            max_start = int(ep.length) - self.num_pred * self.frameskip - 1
            if max_start >= 0:
                anchors.extend(_AnchorRef(ep_idx=ep_idx, start_t=t) for t in range(max_start))
        if not anchors:
            raise RuntimeError("No valid anchors found in Agilex cache episodes.")
        self._anchors = anchors

        self.load_all_into_ram = bool(load_all_into_ram)
        self._max_cached_episodes = 0 if self.load_all_into_ram else int(max_cached_episodes)
        self._memo: "OrderedDict[str, Dict[str, np.ndarray | dict]]" = OrderedDict()

        props: List[np.ndarray] = []
        acts: List[np.ndarray] = []
        for ep in self._sample_for_stats(self._episodes):
            demo = self._load_episode(ep.cache_path)
            props.append(self._adapt_proprio(np.asarray(demo["proprio"], dtype=np.float32)))
            acts.append(self._adapt_actions(np.asarray(demo["actions"], dtype=np.float32)))

        self._stats_proprio = array_to_stats(np.concatenate(props, axis=0))
        self._stats_action = array_to_stats(np.concatenate(acts, axis=0))
        self.states_dim = int(props[0].shape[1])
        self.proprio_dim = int(props[0].shape[1])

        if self.load_all_into_ram:
            for ep in self._episodes:
                if ep.cache_path not in self._memo:
                    self._memo[ep.cache_path] = self._read_episode(ep.cache_path)

    def _discover_episodes(self, max_trajectories: Optional[int]) -> List[_AgilexEpisodeRef]:
        root = Path(self.cache_root)
        episodes: List[_AgilexEpisodeRef] = []
        task_counts: Dict[str, int] = {}
        for task in self.tasks:
            task_dir = root / str(task)
            if not task_dir.is_dir():
                continue
            for cache_path in sorted(p for p in task_dir.iterdir() if p.is_dir()):
                meta_path = cache_path / "metadata.json"
                if not meta_path.is_file():
                    continue
                meta = _read_metadata(cache_path)
                source = _source_name(bool(meta.get("is_failure", False)))
                if source not in self.train_sources:
                    continue
                if max_trajectories is not None and int(max_trajectories) > 0:
                    if task_counts.get(task, 0) >= int(max_trajectories):
                        continue
                length = self._episode_length(cache_path)
                if length <= self.num_pred * self.frameskip:
                    continue
                episodes.append(
                    _AgilexEpisodeRef(
                        task=str(task),
                        cache_path=str(cache_path),
                        length=int(length),
                        is_failure=bool(meta.get("is_failure", False)),
                    )
                )
                task_counts[task] = task_counts.get(task, 0) + 1
        return episodes

    @staticmethod
    def _episode_length(cache_path: Path) -> int:
        lengths = []
        for name in ("images_chw.npy", "proprio.npy", "actions.npy"):
            arr = np.load(str(cache_path / name), mmap_mode="r")
            lengths.append(int(arr.shape[0]))
        return int(min(lengths))

    @staticmethod
    def _sample_for_stats(episodes: Sequence[_AgilexEpisodeRef]) -> List[_AgilexEpisodeRef]:
        by_task: Dict[str, List[_AgilexEpisodeRef]] = {}
        for ep in episodes:
            by_task.setdefault(ep.task, []).append(ep)
        per_task_budget = max(1, 64 // max(1, len(by_task)))
        sampled: List[_AgilexEpisodeRef] = []
        for eps in by_task.values():
            sampled.extend(eps[:per_task_budget])
        return sampled

    @staticmethod
    def _read_episode(path: str) -> Dict[str, np.ndarray | dict]:
        cache_path = Path(path)
        return {
            "images_chw": np.asarray(np.load(str(cache_path / "images_chw.npy"), mmap_mode="r")),
            "proprio": np.asarray(np.load(str(cache_path / "proprio.npy"), mmap_mode="r"), dtype=np.float32),
            "actions": np.asarray(np.load(str(cache_path / "actions.npy"), mmap_mode="r"), dtype=np.float32),
            "metadata": _read_metadata(cache_path),
        }

    def _load_episode(self, path: str) -> Dict[str, np.ndarray | dict]:
        if self.load_all_into_ram:
            cached = self._memo.get(path)
            if cached is not None:
                return cached
            payload = self._read_episode(path)
            self._memo[path] = payload
            return payload
        if self._max_cached_episodes > 0:
            cached = self._memo.get(path)
            if cached is not None:
                self._memo.move_to_end(path)
                return cached
        payload = self._read_episode(path)
        if self._max_cached_episodes > 0:
            self._memo[path] = payload
            self._memo.move_to_end(path)
            while len(self._memo) > self._max_cached_episodes:
                self._memo.popitem(last=False)
        return payload

    def _adapt_proprio(self, proprio: np.ndarray) -> np.ndarray:
        prop = np.asarray(proprio, dtype=np.float32)
        if self._proprio_indices is not None:
            if int(self._proprio_indices.max()) >= int(prop.shape[1]):
                raise ValueError(f"proprio_indices out of range for proprio_dim={prop.shape[1]}")
            prop = prop[:, self._proprio_indices]
        return prop.astype(np.float32, copy=False)

    def _adapt_actions(self, actions: np.ndarray) -> np.ndarray:
        act = np.asarray(actions, dtype=np.float32)
        if act.shape[1] < self.original_action_dim:
            pad = np.zeros((act.shape[0], self.original_action_dim - act.shape[1]), dtype=np.float32)
            act = np.concatenate([act, pad], axis=1)
        elif act.shape[1] > self.original_action_dim:
            act = act[:, : self.original_action_dim]
        return act.astype(np.float32, copy=False)

    def _camera_index(self, metadata: dict, view_name: str) -> int:
        camera_name = self.camera_to_view.get(str(view_name), str(view_name))
        if camera_name == str(view_name):
            for cam, view in self.camera_to_view.items():
                if str(view) == str(view_name):
                    camera_name = str(cam)
                    break
        camera_names = [str(x) for x in metadata.get("camera_names", [])]
        if camera_name not in camera_names:
            raise KeyError(
                f"Camera {camera_name!r} for view {view_name!r} not in cached cameras {camera_names}"
            )
        return int(camera_names.index(camera_name))

    def __len__(self) -> int:
        return int(len(self._anchors))

    def __getitem__(self, idx: int):
        ref = self._anchors[int(idx)]
        ep = self._episodes[ref.ep_idx]
        demo = self._load_episode(ep.cache_path)

        start = int(ref.start_t)
        end = start + self.num_frames * self.frameskip
        obs_indices = list(range(start, end, self.frameskip))
        action_indices = list(range(start, end))
        action_indices[-self.frameskip :] = [obs_indices[-1] - 1] * self.frameskip

        images = np.asarray(demo["images_chw"])
        proprio = self._adapt_proprio(np.asarray(demo["proprio"], dtype=np.float32))
        actions = self._adapt_actions(np.asarray(demo["actions"], dtype=np.float32))
        metadata = dict(demo["metadata"])

        length = min(int(images.shape[0]), int(proprio.shape[0]), int(actions.shape[0]))
        obs_indices = [i for i in obs_indices if 0 <= i < length]
        action_indices = [min(max(i, 0), length - 1) for i in action_indices]

        obs: Dict[str, Dict] = {"visual": {}}
        for vname in self.view_names:
            cam_idx = self._camera_index(metadata, str(vname))
            arr = images[obs_indices, cam_idx]
            if arr.shape[-2] != self.original_img_size or arr.shape[-1] != self.original_img_size:
                raise ValueError(
                    f"Cached images for view={vname} have shape {arr.shape[-2:]} "
                    f"!= original_img_size={self.original_img_size}. Rebuild Agilex cache "
                    "or override env.original_img_size."
                )
            obs["visual"][str(vname)] = torch.from_numpy(arr.astype(np.float32) / 255.0)

        obs["proprio"] = torch.from_numpy(proprio[obs_indices].astype(np.float32, copy=False))
        act = torch.from_numpy(actions[action_indices].astype(np.float32, copy=False))
        state = obs["proprio"]
        return obs, act, state

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        del kwargs
        normalizer = LinearNormalizer()
        normalizer["act"] = get_identity_normalizer_from_stat(self._stats_action)
        normalizer["state"] = get_range_normalizer_from_stat(self._stats_proprio)
        for v in self.view_names:
            normalizer[v] = get_image_range_normalizer()
        return normalizer
