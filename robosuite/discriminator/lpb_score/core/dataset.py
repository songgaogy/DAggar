"""Dataset and batch-building utilities for windowed DSM training."""

from __future__ import annotations

import glob
import math
import os
import queue
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Sequence

import h5py
import numpy as np
import torch
from hydra.utils import to_absolute_path
from torch.utils.data import Dataset

from robosuite.discriminator.dyn_bce.modules.flow_encoder import FlowMultitaskEncoder
from robosuite.discriminator.dyn_bce.task_registry import ordered_task_names


DATA_TYPE_ORDER = ["expert", "success_rollout", "fail_rollout"]


@dataclass(frozen=True)
class SplitCounts:
    num_pos_traj: int
    num_neg_traj: int


@dataclass(frozen=True)
class TaskDataSpec:
    task_name: str
    expert_dir: str
    success_rollout_dir: str
    fail_rollout_dir: str
    train: SplitCounts
    val: SplitCounts
    test: SplitCounts

    def dir_for(self, data_type: str) -> str:
        if data_type == "expert":
            return self.expert_dir
        if data_type == "success_rollout":
            return self.success_rollout_dir
        if data_type == "fail_rollout":
            return self.fail_rollout_dir
        raise KeyError(f"Unsupported data_type: {data_type}")


@dataclass(frozen=True)
class DemoRef:
    task_name: str
    data_type: str
    split: str
    file_path: str
    demo_key: str


@dataclass(frozen=True)
class PreparedTrajectory:
    """Trajectory tensors materialized from raw demos or cached preprocessing."""

    images: np.ndarray
    proprio: np.ndarray
    actions: np.ndarray
    task_name: str
    task_index: int
    data_type: str
    data_type_index: int
    split: str
    file_path: str
    demo_key: str


@dataclass(frozen=True)
class TensorTrajectory:
    images: torch.Tensor
    proprio: torch.Tensor
    actions: torch.Tensor
    task_index_tensor: torch.Tensor
    data_type_index_tensor: torch.Tensor
    task_name: str
    task_index: int
    data_type: str
    data_type_index: int
    split: str
    file_path: str
    demo_key: str
    is_memory_mapped: bool = False


@dataclass(frozen=True)
class CachedTrajectoryRef:
    """Metadata for a trajectory stored in the preprocessed on-disk cache."""

    cache_path: str
    cache_format: str
    images_path: str | None
    proprio_path: str | None
    actions_path: str | None
    num_steps: int
    num_cameras: int
    proprio_dim: int
    task_name: str
    task_index: int
    data_type: str
    data_type_index: int
    split: str
    file_path: str
    demo_key: str


@dataclass(frozen=True)
class TransitionRef:
    trajectory_index: int
    t: int
    traj_type: int


@dataclass(frozen=True)
class BatchableTrajectory:
    images: torch.Tensor
    proprio: torch.Tensor
    task_index: int
    task_index_tensor: torch.Tensor
    data_type_index: int
    data_type_index_tensor: torch.Tensor
    traj_type: int
    is_memory_mapped: bool = False


@dataclass(frozen=True)
class SampleRef:
    trajectory_index: int
    t: int
    task_index: int
    traj_type: int
    data_type_index: int


def _batchable_trajectory_nbytes(traj: BatchableTrajectory) -> int:
    total = 0
    for value in (traj.images, traj.proprio):
        total += int(value.numel() * value.element_size())
    return int(total)


def _cfg_get(cfg: Any, key: str, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def resolve_window_size(cfg: Any, default: int = 8) -> int:
    dataset_cfg = _cfg_get(cfg, "dataset", None)
    value = _cfg_get(dataset_cfg, "window_size", None)
    if value is None:
        data_cfg = _cfg_get(cfg, "data", None)
        value = _cfg_get(data_cfg, "window_size", None)
    if value is None:
        value = default
    window_size = int(value)
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    return window_size


def resolve_preprocessed_cache_root(cfg_data: Any, cache_root: str | None) -> str | None:
    if cache_root is None or str(cache_root) == "":
        return None
    return to_absolute_path(str(cache_root))


def _split_counts_from_cfg(cfg: Any) -> SplitCounts:
    return SplitCounts(
        num_pos_traj=int(_cfg_get(cfg, "num_pos_traj", 0)),
        num_neg_traj=int(_cfg_get(cfg, "num_neg_traj", 0)),
    )


def _get_eval_split_cfg(cfg: Any, default: Any = None) -> Any:
    eval_cfg = _cfg_get(cfg, "eval", None)
    if eval_cfg is not None:
        return eval_cfg
    return _cfg_get(cfg, "val", default)


def parse_task_specs(cfg_data: Any) -> dict[str, TaskDataSpec]:
    """Parse per-task data roots and split sizes from the config tree."""
    specs: dict[str, TaskDataSpec] = {}
    tasks_cfg = _cfg_get(cfg_data, "tasks")
    global_splits = _cfg_get(cfg_data, "splits")
    if tasks_cfg is None:
        raise ValueError("cfg.data.tasks must be provided")
    if global_splits is None:
        raise ValueError("cfg.data.splits must be provided")

    global_train = _split_counts_from_cfg(_cfg_get(global_splits, "train"))
    global_eval = _split_counts_from_cfg(_get_eval_split_cfg(global_splits))
    global_test = _split_counts_from_cfg(_cfg_get(global_splits, "test"))

    for task_name in ordered_task_names(list(tasks_cfg.keys())):
        task_cfg = tasks_cfg[task_name]
        specs[task_name] = TaskDataSpec(
            task_name=task_name,
            expert_dir=to_absolute_path(str(_cfg_get(task_cfg, "expert_dir"))),
            success_rollout_dir=to_absolute_path(str(_cfg_get(task_cfg, "success_rollout_dir"))),
            fail_rollout_dir=to_absolute_path(str(_cfg_get(task_cfg, "fail_rollout_dir"))),
            train=_split_counts_from_cfg(_cfg_get(task_cfg, "train", global_train)),
            val=_split_counts_from_cfg(_get_eval_split_cfg(task_cfg, global_eval)),
            test=_split_counts_from_cfg(_cfg_get(task_cfg, "test", global_test)),
        )
    return specs


def _list_hdf5_demo_refs(task_name: str, data_type: str, data_dir: str) -> list[tuple[str, str]]:
    if not os.path.isdir(data_dir):
        print(f"[lpb_score] Missing directory for {task_name}/{data_type}: {data_dir}. Using 0 trajectories.")
        return []

    refs: list[tuple[str, str]] = []
    hdf5_files = sorted(glob.glob(os.path.join(data_dir, "*.hdf5")))
    if not hdf5_files:
        print(f"[lpb_score] No .hdf5 files for {task_name}/{data_type}: {data_dir}. Using 0 trajectories.")
        return []

    for file_path in hdf5_files:
        with h5py.File(file_path, "r") as file_handle:
            if "demos" not in file_handle:
                continue
            for demo_key in sorted(file_handle["demos"].keys()):
                refs.append((file_path, demo_key))

    if not refs:
        print(f"[lpb_score] No demos found for {task_name}/{data_type}: {data_dir}. Using 0 trajectories.")
    return refs


def _split_refs_with_counts(
    refs: Sequence[DemoRef],
    train_count: int,
    val_count: int,
    test_count: int,
    seed: int,
    task_name: str,
    bucket_name: str,
) -> dict[str, list[DemoRef]]:
    requested_total = int(train_count) + int(val_count) + int(test_count)
    if requested_total <= 0 or len(refs) <= 0:
        return {"train": [], "val": [], "test": []}

    rng = np.random.default_rng(int(seed))
    indices = np.arange(len(refs))
    rng.shuffle(indices)
    selected = [refs[idx] for idx in indices[: min(len(refs), requested_total)]]

    if len(selected) < requested_total:
        print(
            f"[lpb_score] task={task_name} bucket={bucket_name} requested "
            f"(train={train_count}, val={val_count}, test={test_count}, total={requested_total}) "
            f"but only found {len(refs)} trajectories. Using all available trajectories for this bucket."
        )

    train_end = min(int(train_count), len(selected))
    val_end = min(train_end + int(val_count), len(selected))
    test_end = min(val_end + int(test_count), len(selected))
    return {
        "train": selected[:train_end],
        "val": selected[train_end:val_end],
        "test": selected[val_end:test_end],
    }


def build_split_refs(
    cfg_data: Any,
    seed: int,
    include_val: bool = True,
) -> tuple[dict[str, list[DemoRef]], dict[str, dict[str, dict[str, int]]], dict[str, int]]:
    """Sample trajectory references for each split and build the task index map."""
    task_specs = parse_task_specs(cfg_data)
    split_refs = {"train": [], "val": [], "test": []}
    split_summary: dict[str, dict[str, dict[str, int]]] = {}
    task_to_index = {
        task_name: idx
        for idx, task_name in enumerate(ordered_task_names(list(task_specs.keys())))
    }

    for task_offset, task_name in enumerate(ordered_task_names(list(task_specs.keys()))):
        spec = task_specs[task_name]
        split_summary[task_name] = {"train": {}, "val": {}, "test": {}}
        split_cfgs = {"train": spec.train, "val": spec.val, "test": spec.test}
        if not bool(include_val):
            split_cfgs["val"] = SplitCounts(num_pos_traj=0, num_neg_traj=0)
        positive_refs: list[DemoRef] = []
        for data_type in ["expert", "success_rollout"]:
            refs = _list_hdf5_demo_refs(
                task_name=task_name,
                data_type=data_type,
                data_dir=spec.dir_for(data_type),
            )
            positive_refs.extend(
                DemoRef(
                    task_name=task_name,
                    data_type=data_type,
                    split="",
                    file_path=file_path,
                    demo_key=demo_key,
                )
                for file_path, demo_key in refs
            )

        negative_refs = [
            DemoRef(
                task_name=task_name,
                data_type="fail_rollout",
                split="",
                file_path=file_path,
                demo_key=demo_key,
            )
            for file_path, demo_key in _list_hdf5_demo_refs(
                task_name=task_name,
                data_type="fail_rollout",
                data_dir=spec.dir_for("fail_rollout"),
            )
        ]

        chosen_pos = _split_refs_with_counts(
            refs=positive_refs,
            train_count=split_cfgs["train"].num_pos_traj,
            val_count=split_cfgs["val"].num_pos_traj,
            test_count=split_cfgs["test"].num_pos_traj,
            seed=int(seed + task_offset * 97),
            task_name=task_name,
            bucket_name="positive",
        )
        chosen_neg = _split_refs_with_counts(
            refs=negative_refs,
            train_count=split_cfgs["train"].num_neg_traj,
            val_count=split_cfgs["val"].num_neg_traj,
            test_count=split_cfgs["test"].num_neg_traj,
            seed=int(seed + task_offset * 97 + 17),
            task_name=task_name,
            bucket_name="negative",
        )

        for split_name in ["train", "val", "test"]:
            split_pos = list(chosen_pos[split_name])
            split_neg = list(chosen_neg[split_name])
            split_summary[task_name][split_name] = {
                "num_pos_traj": int(len(split_pos)),
                "num_neg_traj": int(len(split_neg)),
                "expert": int(sum(ref.data_type == "expert" for ref in split_pos)),
                "success_rollout": int(sum(ref.data_type == "success_rollout" for ref in split_pos)),
                "fail_rollout": int(len(split_neg)),
            }
            for ref in split_pos + split_neg:
                split_refs[split_name].append(
                    DemoRef(
                        task_name=ref.task_name,
                        data_type=ref.data_type,
                        split=split_name,
                        file_path=ref.file_path,
                        demo_key=ref.demo_key,
                    )
                )

    return split_refs, split_summary, task_to_index


def filter_refs_by_data_types(
    refs: Sequence[DemoRef],
    data_types: Sequence[str] | None,
) -> list[DemoRef]:
    if data_types is None:
        return list(refs)
    allowed = {str(name) for name in data_types}
    return [ref for ref in refs if ref.data_type in allowed]


def prepare_trajectories(
    refs: Sequence[DemoRef],
    *,
    encoder: FlowMultitaskEncoder,
    task_to_index: dict[str, int],
    cache_root: str | None = None,
    use_preprocessed_cache: bool = False,
    refresh_preprocessed_cache: bool = False,
) -> list[PreparedTrajectory]:
    """Materialize demos into tensors used by the DSM training pipeline.

    Images are stored as contiguous `uint8` arrays to keep the resident RAM
    footprint low. Conversion to normalized floating point happens later inside
    the encoder path.
    """
    trajectories: list[PreparedTrajectory] = []
    for ref in refs:
        prepared = encoder.materialize_demo_inputs(
            task_name=ref.task_name,
            file_path=ref.file_path,
            demo_key=ref.demo_key,
            cache_root=cache_root,
            use_cache=bool(use_preprocessed_cache),
            refresh_cache=bool(refresh_preprocessed_cache),
        )
        length = min(
            int(prepared.images_chw.shape[0]),
            int(prepared.proprio.shape[0]),
            int(prepared.actions.shape[0]),
        )
        if length <= 0:
            continue
        images = np.asarray(prepared.images_chw[:length])
        if images.dtype != np.uint8:
            images = np.clip(np.rint(np.asarray(images, dtype=np.float32) * 255.0), 0.0, 255.0).astype(np.uint8)
        images = np.ascontiguousarray(images)
        data_type_index = DATA_TYPE_ORDER.index(ref.data_type) if ref.data_type in DATA_TYPE_ORDER else -1
        trajectories.append(
            PreparedTrajectory(
                images=images,
                proprio=np.asarray(prepared.proprio[:length], dtype=np.float32),
                actions=np.asarray(prepared.actions[:length], dtype=np.float32),
                task_name=ref.task_name,
                task_index=int(task_to_index[ref.task_name]),
                data_type=ref.data_type,
                data_type_index=int(data_type_index),
                split=ref.split,
                file_path=ref.file_path,
                demo_key=ref.demo_key,
            )
        )
    return trajectories


def build_cached_trajectory_refs(
    refs: Sequence[DemoRef],
    *,
    encoder: FlowMultitaskEncoder,
    task_to_index: dict[str, int],
    cache_root: str | None = None,
    use_preprocessed_cache: bool = False,
    refresh_preprocessed_cache: bool = False,
) -> list[CachedTrajectoryRef]:
    """Index the preprocessed cache without keeping full trajectories in memory."""
    if not bool(use_preprocessed_cache):
        raise ValueError("build_cached_trajectory_refs requires use_preprocessed_cache=True.")
    if cache_root is None or str(cache_root) == "":
        raise ValueError("build_cached_trajectory_refs requires a non-empty cache_root.")

    cached_refs: list[CachedTrajectoryRef] = []
    for ref in refs:
        cache_record = encoder.describe_preprocessed_demo(
            cache_root=cache_root,
            task_name=ref.task_name,
            file_path=ref.file_path,
            demo_key=ref.demo_key,
            refresh_cache=bool(refresh_preprocessed_cache),
            allow_upgrade=True,
        )
        num_steps = int(cache_record.num_steps)
        num_cameras = int(cache_record.num_cameras)
        proprio_dim = int(cache_record.proprio_dim)

        if num_steps <= 0:
            continue
        data_type_index = DATA_TYPE_ORDER.index(ref.data_type) if ref.data_type in DATA_TYPE_ORDER else -1
        cached_refs.append(
            CachedTrajectoryRef(
                cache_path=cache_record.cache_path,
                cache_format=str(cache_record.cache_format),
                images_path=cache_record.images_path,
                proprio_path=cache_record.proprio_path,
                actions_path=cache_record.actions_path,
                num_steps=num_steps,
                num_cameras=num_cameras,
                proprio_dim=proprio_dim,
                task_name=ref.task_name,
                task_index=int(task_to_index[ref.task_name]),
                data_type=ref.data_type,
                data_type_index=int(data_type_index),
                split=ref.split,
                file_path=ref.file_path,
                demo_key=ref.demo_key,
            )
        )
    return cached_refs


def estimate_trajectories_nbytes(trajectories: Sequence[PreparedTrajectory]) -> int:
    """Estimate the resident RAM used by a list of prepared trajectories."""
    total = 0
    for traj in trajectories:
        total += int(traj.images.nbytes)
        total += int(traj.proprio.nbytes)
        total += int(traj.actions.nbytes)
    return int(total)


class LatentTransitionDataset(Dataset):
    """Windowed trajectory dataset backed by RAM, disk cache, or a hybrid mix."""

    def __init__(
        self,
        trajectories: Sequence[PreparedTrajectory] | None = None,
        cached_refs: Sequence[CachedTrajectoryRef] | None = None,
        resident_trajectories: Sequence[tuple[int, PreparedTrajectory]] | None = None,
        window_size: int = 8,
        cache_trajectory_limit: int = 4,
    ) -> None:
        super().__init__()
        self.trajectories = list(trajectories or [])
        self.cached_refs = list(cached_refs or [])
        self.resident_trajectories = list(resident_trajectories or [])
        self.window_size = int(window_size)
        self.cache_trajectory_limit = max(0, int(cache_trajectory_limit))
        if self.window_size <= 0:
            raise ValueError(f"window_size must be positive, got {self.window_size}")
        if bool(self.trajectories) and bool(self.cached_refs):
            raise ValueError("Provide either trajectories or cached_refs, not both.")
        if not self.trajectories and not self.cached_refs:
            raise ValueError("Provide trajectories or cached_refs.")
        if self.trajectories and self.resident_trajectories:
            raise ValueError("resident_trajectories can only be used with cached_refs.")

        self._transition_refs: list[TransitionRef] = []
        self._sample_is_positive: list[bool] = []
        self._trajectory_cache: OrderedDict[int, TensorTrajectory] = OrderedDict()
        self._trajectory_cache_lock = threading.Lock()
        self._trajectory_tensors: list[TensorTrajectory] = []
        self._resident_trajectory_tensors: dict[int, TensorTrajectory] = {}
        self._window_index_cache: dict[int, torch.Tensor] = {}
        self._sample_refs_array_cache: np.ndarray | None = None

        if self.resident_trajectories and not self.cached_refs:
            raise ValueError("resident_trajectories requires cached_refs.")
        resident_indices_seen: set[int] = set()
        for dataset_index, traj in self.resident_trajectories:
            dataset_index = int(dataset_index)
            if dataset_index < 0 or dataset_index >= len(self.cached_refs):
                raise IndexError(
                    f"resident trajectory index out of range: {dataset_index} not in [0, {len(self.cached_refs) - 1}]"
                )
            if dataset_index in resident_indices_seen:
                raise ValueError(f"Duplicate resident trajectory index: {dataset_index}")
            resident_indices_seen.add(dataset_index)
            self._resident_trajectory_tensors[dataset_index] = self._prepare_tensor_trajectory(traj)

        sources: Sequence[PreparedTrajectory | CachedTrajectoryRef]
        if self.trajectories:
            self._num_cameras = int(self.trajectories[0].images.shape[1])
            self._proprio_dim = int(self.trajectories[0].proprio.shape[1])
            sources = self.trajectories
        else:
            if self._resident_trajectory_tensors:
                first_resident = next(iter(self._resident_trajectory_tensors.values()))
                self._num_cameras = int(first_resident.images.shape[1])
                self._proprio_dim = int(first_resident.proprio.shape[1])
            else:
                self._num_cameras = int(self.cached_refs[0].num_cameras)
                self._proprio_dim = int(self.cached_refs[0].proprio_dim)
            sources = self.cached_refs

        for traj_idx, traj in enumerate(sources):
            if isinstance(traj, PreparedTrajectory):
                if int(traj.images.shape[0]) != int(traj.proprio.shape[0]):
                    raise ValueError(
                        f"Trajectory length mismatch for {traj.file_path}:{traj.demo_key}: "
                        f"images={traj.images.shape[0]} proprio={traj.proprio.shape[0]}"
                    )
                if int(traj.images.shape[1]) != self._num_cameras:
                    raise ValueError("All trajectories must use the same camera count.")
                if int(traj.proprio.shape[1]) != self._proprio_dim:
                    raise ValueError("All trajectories must use the same proprio dim.")
                tensor_traj = self._prepare_tensor_trajectory(traj)
                self._trajectory_tensors.append(tensor_traj)
                length = int(tensor_traj.images.shape[0])
                data_type = tensor_traj.data_type
            else:
                resident_traj = self._resident_trajectory_tensors.get(traj_idx)
                if resident_traj is not None:
                    if int(resident_traj.images.shape[1]) != self._num_cameras:
                        raise ValueError("All resident trajectories must use the same camera count.")
                    if int(resident_traj.proprio.shape[1]) != self._proprio_dim:
                        raise ValueError("All resident trajectories must use the same proprio dim.")
                    if int(resident_traj.images.shape[0]) != int(traj.num_steps):
                        raise ValueError(
                            f"Resident trajectory length mismatch for index={traj_idx}: "
                            f"resident={resident_traj.images.shape[0]} cached={traj.num_steps}"
                        )
                    length = int(resident_traj.images.shape[0])
                    data_type = resident_traj.data_type
                else:
                    if int(traj.num_cameras) != self._num_cameras:
                        raise ValueError("All cached trajectories must use the same camera count.")
                    if int(traj.proprio_dim) != self._proprio_dim:
                        raise ValueError("All cached trajectories must use the same proprio dim.")
                    length = int(traj.num_steps)
                    data_type = traj.data_type
            self._window_index_cache[traj_idx] = self._build_window_indices(length)
            traj_type = 1 if data_type == "fail_rollout" else 0
            for t in range(max(0, length)):
                self._transition_refs.append(
                    TransitionRef(
                        trajectory_index=traj_idx,
                        t=t,
                        traj_type=traj_type,
                    )
                )
                self._sample_is_positive.append(traj_type == 0)

        if not self._transition_refs:
            raise RuntimeError("No valid temporal windows found for the provided trajectories.")

    @property
    def sample_is_positive(self) -> list[bool]:
        return self._sample_is_positive

    def iter_positive_trajectories(self) -> Iterable[PreparedTrajectory]:
        if self.trajectories:
            for traj in self.trajectories:
                if traj.data_type != "fail_rollout":
                    yield traj
            return
        for idx, ref in enumerate(self.cached_refs):
            if ref.data_type != "fail_rollout":
                yield self._get_trajectory(idx)

    @property
    def num_positive_samples(self) -> int:
        return int(sum(self._sample_is_positive))

    @property
    def num_negative_samples(self) -> int:
        return len(self._sample_is_positive) - self.num_positive_samples

    @property
    def num_cameras(self) -> int:
        return self._num_cameras

    @property
    def proprio_dim(self) -> int:
        return self._proprio_dim

    @property
    def num_trajectories(self) -> int:
        return len(self.trajectories) if self.trajectories else len(self.cached_refs)

    @property
    def uses_in_memory_trajectories(self) -> bool:
        return bool(self.trajectories)

    @property
    def num_resident_trajectories(self) -> int:
        if self.trajectories:
            return len(self.trajectories)
        return len(self._resident_trajectory_tensors)

    @property
    def image_shape(self) -> tuple[int, ...]:
        if self.trajectories:
            return tuple(int(x) for x in self.trajectories[0].images.shape[1:])
        if self._resident_trajectory_tensors:
            first_resident = next(iter(self._resident_trajectory_tensors.values()))
            return tuple(int(x) for x in first_resident.images.shape[1:])
        traj = self._get_trajectory(0)
        return tuple(int(x) for x in traj.images.shape[1:])

    def __len__(self) -> int:
        return len(self._transition_refs)

    @property
    def sample_refs_array(self) -> np.ndarray:
        if self._sample_refs_array_cache is not None:
            return self._sample_refs_array_cache
        refs = np.empty((len(self._transition_refs), 5), dtype=np.int32)
        if self.trajectories:
            task_index_lookup = [int(traj.task_index) for traj in self.trajectories]
            data_type_lookup = [int(traj.data_type_index) for traj in self.trajectories]
        else:
            task_index_lookup = [int(ref.task_index) for ref in self.cached_refs]
            data_type_lookup = [int(ref.data_type_index) for ref in self.cached_refs]
        for idx, ref in enumerate(self._transition_refs):
            refs[idx, 0] = int(ref.trajectory_index)
            refs[idx, 1] = int(ref.t)
            refs[idx, 2] = task_index_lookup[int(ref.trajectory_index)]
            refs[idx, 3] = int(ref.traj_type)
            refs[idx, 4] = data_type_lookup[int(ref.trajectory_index)]
        self._sample_refs_array_cache = refs
        return refs

    @staticmethod
    def _as_image_tensor(value: np.ndarray) -> torch.Tensor:
        array = np.asarray(value)
        if array.dtype == np.uint8:
            image_array = array if bool(array.flags.c_contiguous) else np.ascontiguousarray(array)
        else:
            image_array = np.clip(np.rint(np.asarray(array, dtype=np.float32) * 255.0), 0.0, 255.0).astype(np.uint8)
            if not bool(image_array.flags.c_contiguous):
                image_array = np.ascontiguousarray(image_array)
        return torch.from_numpy(image_array)

    @staticmethod
    def _as_long_scalar(value: int) -> torch.Tensor:
        return torch.tensor(int(value), dtype=torch.int64)

    def _prepare_tensor_trajectory(self, traj: PreparedTrajectory) -> TensorTrajectory:
        proprio_array = np.asarray(traj.proprio)
        if proprio_array.dtype != np.float32:
            proprio_array = np.asarray(proprio_array, dtype=np.float32)
        if not bool(proprio_array.flags.c_contiguous):
            proprio_array = np.ascontiguousarray(proprio_array)
        actions_array = np.asarray(traj.actions)
        if actions_array.dtype != np.float32:
            actions_array = np.asarray(actions_array, dtype=np.float32)
        if not bool(actions_array.flags.c_contiguous):
            actions_array = np.ascontiguousarray(actions_array)
        return TensorTrajectory(
            images=self._as_image_tensor(traj.images),
            proprio=torch.from_numpy(proprio_array),
            actions=torch.from_numpy(actions_array),
            task_index_tensor=self._as_long_scalar(traj.task_index),
            data_type_index_tensor=self._as_long_scalar(traj.data_type_index),
            task_name=traj.task_name,
            task_index=int(traj.task_index),
            data_type=traj.data_type,
            data_type_index=int(traj.data_type_index),
            split=traj.split,
            file_path=traj.file_path,
            demo_key=traj.demo_key,
            is_memory_mapped=bool(isinstance(traj.images, np.memmap))
            or bool(isinstance(proprio_array, np.memmap))
            or bool(isinstance(actions_array, np.memmap)),
        )

    def _load_cached_trajectory(self, ref: CachedTrajectoryRef) -> TensorTrajectory:
        if str(ref.cache_format) == "bundle":
            if ref.images_path is None or ref.proprio_path is None or ref.actions_path is None:
                raise ValueError(f"Bundle cache ref is missing array paths: {ref.cache_path}")
            images = np.load(ref.images_path, mmap_mode="r", allow_pickle=False)
            proprio = np.load(ref.proprio_path, mmap_mode="r", allow_pickle=False)
            actions = np.load(ref.actions_path, mmap_mode="r", allow_pickle=False)
        else:
            with np.load(ref.cache_path) as cached:
                images = np.asarray(cached["images_chw"])
                proprio = np.asarray(cached["proprio"], dtype=np.float32)
                actions = np.asarray(cached["actions"], dtype=np.float32)
        return self._prepare_tensor_trajectory(
            PreparedTrajectory(
                images=images,
                proprio=proprio,
                actions=actions,
                task_name=ref.task_name,
                task_index=int(ref.task_index),
                data_type=ref.data_type,
                data_type_index=int(ref.data_type_index),
                split=ref.split,
                file_path=ref.file_path,
                demo_key=ref.demo_key,
            )
        )

    def _get_cached_trajectory(self, trajectory_index: int) -> TensorTrajectory:
        trajectory_index = int(trajectory_index)
        with self._trajectory_cache_lock:
            cached = self._trajectory_cache.get(trajectory_index)
            if cached is not None:
                self._trajectory_cache.move_to_end(trajectory_index)
                return cached
        traj = self._load_cached_trajectory(self.cached_refs[trajectory_index])
        with self._trajectory_cache_lock:
            cached = self._trajectory_cache.get(trajectory_index)
            if cached is not None:
                self._trajectory_cache.move_to_end(trajectory_index)
                return cached
            if self.cache_trajectory_limit > 0:
                self._trajectory_cache[trajectory_index] = traj
                while len(self._trajectory_cache) > self.cache_trajectory_limit:
                    self._trajectory_cache.popitem(last=False)
        return traj

    def _get_trajectory(self, trajectory_index: int) -> TensorTrajectory:
        if self.trajectories:
            return self._trajectory_tensors[int(trajectory_index)]
        resident = self._resident_trajectory_tensors.get(int(trajectory_index))
        if resident is not None:
            return resident
        return self._get_cached_trajectory(int(trajectory_index))

    def _make_batchable_trajectory(self, traj: TensorTrajectory) -> BatchableTrajectory:
        images = traj.images.contiguous()
        proprio = traj.proprio.contiguous()
        return BatchableTrajectory(
            images=images,
            proprio=proprio,
            task_index=int(traj.task_index),
            task_index_tensor=traj.task_index_tensor,
            data_type_index=int(traj.data_type_index),
            data_type_index_tensor=traj.data_type_index_tensor,
            traj_type=1 if traj.data_type == "fail_rollout" else 0,
            is_memory_mapped=bool(traj.is_memory_mapped),
        )

    def build_batchable_trajectory(self, trajectory_index: int) -> BatchableTrajectory:
        return self._make_batchable_trajectory(self._get_trajectory(int(trajectory_index)))

    def _build_window_indices(self, length: int) -> torch.Tensor:
        steps = torch.arange(int(length), dtype=torch.long).unsqueeze(1)
        offsets = torch.arange(self.window_size - 1, -1, -1, dtype=torch.long).unsqueeze(0)
        return torch.clamp(steps - offsets, min=0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        ref = self._transition_refs[index]
        traj = self._get_trajectory(ref.trajectory_index)
        window_index = self._window_index_cache[int(ref.trajectory_index)][int(ref.t)]
        image_window = traj.images.index_select(0, window_index)
        proprio_window = traj.proprio.index_select(0, window_index)
        return {
            "image_window": image_window,
            "proprio_window": proprio_window,
            "task_index": traj.task_index_tensor,
            "data_type_index": traj.data_type_index_tensor,
            "traj_type": self._as_long_scalar(ref.traj_type),
        }


class _BatchedTrainEpochIterator:
    def __init__(
        self,
        *,
        loader: "BatchedTrainLoader",
        batch_refs: list[np.ndarray],
    ) -> None:
        self._loader = loader
        self._batch_refs = batch_refs
        self._index = 0
        self._queue: queue.Queue[object] | None = None
        self._thread: threading.Thread | None = None
        self._sentinel = object()
        if bool(self._loader.use_prefetch_thread):
            self._queue = queue.Queue(maxsize=max(1, int(self._loader.prefetch_depth)))
            self._thread = threading.Thread(
                target=self._producer,
                name="lpb-score-batch-prefetch",
                daemon=True,
            )
            self._thread.start()

    def _producer(self) -> None:
        assert self._queue is not None
        try:
            if int(self._loader.batch_build_workers) <= 1:
                for batch_ref in self._batch_refs:
                    self._queue.put(self._loader.build_batch(batch_ref))
            else:
                with ThreadPoolExecutor(
                    max_workers=int(self._loader.batch_build_workers),
                    thread_name_prefix="lpb-score-batch-build",
                ) as executor:
                    for batch in executor.map(self._loader.build_batch, self._batch_refs, chunksize=1):
                        self._queue.put(batch)
        except BaseException as exc:  # pragma: no cover - surfaced in consumer
            self._queue.put(exc)
        finally:
            self._queue.put(self._sentinel)

    def __iter__(self) -> "_BatchedTrainEpochIterator":
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        if self._queue is None:
            if self._index >= len(self._batch_refs):
                raise StopIteration
            batch = self._loader.build_batch(self._batch_refs[self._index])
            self._index += 1
            return batch

        item = self._queue.get()
        if item is self._sentinel:
            raise StopIteration
        if isinstance(item, BaseException):
            raise item
        assert isinstance(item, dict)
        return item


class BatchedTrainLoader:
    """Build balanced window batches directly from trajectory references.

    The loader samples positive and negative windows at the batch level, then
    gathers all windows for the same trajectory together to amortize cache and
    indexing overhead.
    """

    def __init__(
        self,
        dataset: LatentTransitionDataset,
        *,
        batch_size: int,
        positive_sampling_ratio: float = 0.5,
        seed: int = 0,
        pin_memory: bool = False,
        prefetch_depth: int = 2,
        use_prefetch_thread: bool = True,
        batch_build_workers: int = 1,
        trajectory_cache_limit_gb: float = 8.0,
        num_replicas: int = 1,
        rank: int = 0,
    ) -> None:
        if int(batch_size) <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if len(dataset) <= 0:
            raise ValueError("BatchedTrainLoader requires a non-empty dataset.")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.positive_sampling_ratio = float(min(max(positive_sampling_ratio, 0.0), 1.0))
        self.seed = int(seed)
        self.pin_memory = bool(pin_memory)
        self.prefetch_depth = max(1, int(prefetch_depth))
        self.use_prefetch_thread = bool(use_prefetch_thread)
        self.batch_build_workers = max(1, int(batch_build_workers))
        self.trajectory_cache_limit_bytes = max(0, int(float(trajectory_cache_limit_gb) * (1024.0 ** 3)))
        self.num_replicas = max(1, int(num_replicas))
        self.rank = int(rank)
        if self.rank < 0 or self.rank >= self.num_replicas:
            raise ValueError(f"rank must be in [0, {self.num_replicas}), got {self.rank}")
        if self.batch_size < self.num_replicas:
            raise ValueError(
                f"global batch_size={self.batch_size} must be >= num_replicas={self.num_replicas}"
            )
        if self.batch_size % self.num_replicas != 0:
            raise ValueError(
                f"global batch_size={self.batch_size} must be divisible by num_replicas={self.num_replicas}"
            )
        self.local_batch_size = self.batch_size // self.num_replicas
        self.window_size = int(dataset.window_size)
        self.num_batches = int(math.ceil(len(dataset) / float(self.batch_size)))
        self._epoch = 0
        self._window_offsets = torch.arange(self.window_size, dtype=torch.long)
        self._sample_refs = dataset.sample_refs_array
        self._positive_pool = np.flatnonzero(self._sample_refs[:, 3] == 0).astype(np.int64, copy=False)
        self._negative_pool = np.flatnonzero(self._sample_refs[:, 3] == 1).astype(np.int64, copy=False)
        self._image_shape = tuple(int(x) for x in dataset.image_shape)
        self._proprio_shape = (int(dataset.proprio_dim),)
        self._batchable_cache: OrderedDict[int, BatchableTrajectory] = OrderedDict()
        self._batchable_cache_lock = threading.Lock()
        self._batchable_cache_nbytes = 0

    def __len__(self) -> int:
        return self.num_batches

    def _sample_indices(
        self,
        *,
        rng: np.random.Generator,
        pool: np.ndarray,
        size: int,
    ) -> np.ndarray:
        if int(size) <= 0:
            return np.empty((0,), dtype=np.int64)
        if pool.size <= 0:
            return np.empty((0,), dtype=np.int64)
        choice = rng.integers(0, int(pool.size), size=int(size), endpoint=False)
        return pool[choice]

    def _build_epoch_plan(self) -> list[np.ndarray]:
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1
        batches: list[np.ndarray] = []
        for _ in range(self.num_batches):
            current_batch_size = self.batch_size
            num_pos = int(round(float(current_batch_size) * self.positive_sampling_ratio))
            num_pos = min(max(num_pos, 0), current_batch_size)
            num_neg = current_batch_size - num_pos

            selected_pos = self._sample_indices(rng=rng, pool=self._positive_pool, size=num_pos)
            selected_neg = self._sample_indices(rng=rng, pool=self._negative_pool, size=num_neg)
            if selected_pos.size < num_pos:
                refill = self._sample_indices(rng=rng, pool=self._negative_pool, size=num_pos - int(selected_pos.size))
                selected_pos = np.concatenate([selected_pos, refill], axis=0)
            if selected_neg.size < num_neg:
                refill = self._sample_indices(rng=rng, pool=self._positive_pool, size=num_neg - int(selected_neg.size))
                selected_neg = np.concatenate([selected_neg, refill], axis=0)
            batch_indices = np.concatenate([selected_pos, selected_neg], axis=0)
            if batch_indices.size != current_batch_size:
                fallback_pool = self._positive_pool if self._positive_pool.size > 0 else self._negative_pool
                refill = self._sample_indices(
                    rng=rng,
                    pool=fallback_pool,
                    size=current_batch_size - int(batch_indices.size),
                )
                batch_indices = np.concatenate([batch_indices, refill], axis=0)
            order = rng.permutation(current_batch_size)
            ordered_batch = self._sample_refs[batch_indices[order]]
            start = self.rank * self.local_batch_size
            end = start + self.local_batch_size
            batches.append(ordered_batch[start:end])
        return batches

    def _get_batchable_trajectory(self, trajectory_index: int) -> BatchableTrajectory:
        trajectory_index = int(trajectory_index)
        with self._batchable_cache_lock:
            cached = self._batchable_cache.get(trajectory_index)
            if cached is not None:
                self._batchable_cache.move_to_end(trajectory_index)
                return cached
        built = self.dataset.build_batchable_trajectory(trajectory_index)
        if bool(built.is_memory_mapped):
            # Keep memmap-backed trajectories ephemeral. The OS page cache already
            # provides reuse, and retaining tensor wrappers for every trajectory can
            # balloon shared-memory RSS under DDP.
            return built
        if self.trajectory_cache_limit_bytes > 0:
            built_nbytes = _batchable_trajectory_nbytes(built)
            with self._batchable_cache_lock:
                cached = self._batchable_cache.get(trajectory_index)
                if cached is not None:
                    self._batchable_cache.move_to_end(trajectory_index)
                    return cached
                self._batchable_cache[trajectory_index] = built
                self._batchable_cache_nbytes += built_nbytes
                while len(self._batchable_cache) > 1 and self._batchable_cache_nbytes > self.trajectory_cache_limit_bytes:
                    _, evicted = self._batchable_cache.popitem(last=False)
                    self._batchable_cache_nbytes -= _batchable_trajectory_nbytes(evicted)
        return built

    def _empty_batch(self, batch_size: int) -> dict[str, torch.Tensor]:
        pin = bool(self.pin_memory)
        return {
            "image_window": torch.empty(
                (batch_size, self.window_size, *self._image_shape),
                dtype=torch.uint8,
                pin_memory=pin,
            ),
            "proprio_window": torch.empty(
                (batch_size, self.window_size, *self._proprio_shape),
                dtype=torch.float32,
                pin_memory=pin,
            ),
            "task_index": torch.empty((batch_size,), dtype=torch.int64, pin_memory=pin),
            "data_type_index": torch.empty((batch_size,), dtype=torch.int64, pin_memory=pin),
            "traj_type": torch.empty((batch_size,), dtype=torch.int64, pin_memory=pin),
        }

    def build_batch(self, batch_refs: np.ndarray) -> dict[str, torch.Tensor]:
        batch_size = int(batch_refs.shape[0])
        batch = self._empty_batch(batch_size)
        batch["task_index"].copy_(torch.from_numpy(batch_refs[:, 2].astype(np.int64, copy=False)))
        batch["data_type_index"].copy_(torch.from_numpy(batch_refs[:, 4].astype(np.int64, copy=False)))
        batch["traj_type"].copy_(torch.from_numpy(batch_refs[:, 3].astype(np.int64, copy=False)))

        sorted_positions = np.argsort(batch_refs[:, 0], kind="stable")
        sorted_refs = batch_refs[sorted_positions]
        start = 0
        while start < batch_size:
            end = start + 1
            trajectory_index = int(sorted_refs[start, 0])
            while end < batch_size and int(sorted_refs[end, 0]) == trajectory_index:
                end += 1
            out_positions = sorted_positions[start:end]
            timesteps = sorted_refs[start:end, 1].astype(np.int64, copy=False)
            traj = self._get_batchable_trajectory(trajectory_index)
            window_index = torch.from_numpy(timesteps).long().unsqueeze(1) - (self.window_size - 1) + self._window_offsets.unsqueeze(0)
            window_index = torch.clamp(window_index, min=0)
            flat_window_index = window_index.reshape(-1)
            image_window = traj.images.index_select(0, flat_window_index).reshape(
                end - start,
                self.window_size,
                *self._image_shape,
            )
            proprio_window = traj.proprio.index_select(0, flat_window_index).reshape(
                end - start,
                self.window_size,
                *self._proprio_shape,
            )
            position_index = torch.from_numpy(out_positions.astype(np.int64, copy=False))
            batch["image_window"].index_copy_(0, position_index, image_window)
            batch["proprio_window"].index_copy_(0, position_index, proprio_window)
            start = end
        return batch

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        return _BatchedTrainEpochIterator(
            loader=self,
            batch_refs=self._build_epoch_plan(),
        )
