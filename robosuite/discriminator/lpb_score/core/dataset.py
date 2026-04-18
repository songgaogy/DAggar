"""Raw-input data pipeline for joint encoder + chunk DSM training."""

from __future__ import annotations

import glob
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

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


@dataclass(frozen=True)
class CachedTrajectoryRef:
    cache_path: str
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
) -> tuple[dict[str, list[DemoRef]], dict[str, dict[str, dict[str, int]]], dict[str, int]]:
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


def print_split_summary(split_summary: dict[str, dict[str, dict[str, int]]]) -> None:
    for task_name, task_summary in split_summary.items():
        for split_name, split_counts in task_summary.items():
            msg = ", ".join(
                [
                    f"num_pos_traj={int(split_counts.get('num_pos_traj', 0))}",
                    f"num_neg_traj={int(split_counts.get('num_neg_traj', 0))}",
                    f"expert={int(split_counts.get('expert', 0))}",
                    f"success_rollout={int(split_counts.get('success_rollout', 0))}",
                    f"fail_rollout={int(split_counts.get('fail_rollout', 0))}",
                ]
            )
            print(f"[lpb_score] task={task_name} split={split_name} {msg}")


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
    progress_label: str,
    cache_root: str | None = None,
    use_preprocessed_cache: bool = False,
    refresh_preprocessed_cache: bool = False,
) -> list[PreparedTrajectory]:
    trajectories: list[PreparedTrajectory] = []
    cache_hits = 0
    cache_misses = 0
    for idx, ref in enumerate(refs):
        cache_path = None
        if bool(use_preprocessed_cache) and cache_root is not None and str(cache_root) != "":
            cache_path = encoder.preprocessed_cache_path(
                cache_root=cache_root,
                task_name=ref.task_name,
                file_path=ref.file_path,
                demo_key=ref.demo_key,
            )
            if os.path.isfile(cache_path) and not bool(refresh_preprocessed_cache):
                cache_hits += 1
            else:
                cache_misses += 1
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
        data_type_index = DATA_TYPE_ORDER.index(ref.data_type) if ref.data_type in DATA_TYPE_ORDER else -1
        trajectories.append(
            PreparedTrajectory(
                images=np.asarray(prepared.images_chw[:length], dtype=np.float32),
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
        if (idx + 1) % 200 == 0 or (idx + 1) == len(refs):
            print(f"[lpb_score] {progress_label} {idx + 1}/{len(refs)}")
    if bool(use_preprocessed_cache) and cache_root is not None and str(cache_root) != "":
        print(
            f"[lpb_score] {progress_label} cache_root={to_absolute_path(str(cache_root))} "
            f"cache_hits={cache_hits} cache_misses={cache_misses}"
        )
    return trajectories


def build_cached_trajectory_refs(
    refs: Sequence[DemoRef],
    *,
    encoder: FlowMultitaskEncoder,
    task_to_index: dict[str, int],
    progress_label: str,
    cache_root: str | None = None,
    use_preprocessed_cache: bool = False,
    refresh_preprocessed_cache: bool = False,
) -> list[CachedTrajectoryRef]:
    if not bool(use_preprocessed_cache):
        raise ValueError("build_cached_trajectory_refs requires use_preprocessed_cache=True.")
    if cache_root is None or str(cache_root) == "":
        raise ValueError("build_cached_trajectory_refs requires a non-empty cache_root.")

    cached_refs: list[CachedTrajectoryRef] = []
    cache_hits = 0
    cache_misses = 0
    for idx, ref in enumerate(refs):
        cache_path = encoder.preprocessed_cache_path(
            cache_root=cache_root,
            task_name=ref.task_name,
            file_path=ref.file_path,
            demo_key=ref.demo_key,
        )
        prepared = None
        if os.path.isfile(cache_path) and not bool(refresh_preprocessed_cache):
            cache_hits += 1
        else:
            cache_misses += 1
            prepared = encoder.materialize_demo_inputs(
                task_name=ref.task_name,
                file_path=ref.file_path,
                demo_key=ref.demo_key,
                cache_root=cache_root,
                use_cache=True,
                refresh_cache=bool(refresh_preprocessed_cache),
            )

        if prepared is None:
            with np.load(cache_path) as cached:
                num_steps = int(cached["images_chw"].shape[0])
                num_cameras = int(cached["images_chw"].shape[1])
                proprio_dim = int(cached["proprio"].shape[1])
        else:
            num_steps = int(prepared.images_chw.shape[0])
            num_cameras = int(prepared.images_chw.shape[1])
            proprio_dim = int(prepared.proprio.shape[1])

        if num_steps <= 0:
            continue
        data_type_index = DATA_TYPE_ORDER.index(ref.data_type) if ref.data_type in DATA_TYPE_ORDER else -1
        cached_refs.append(
            CachedTrajectoryRef(
                cache_path=cache_path,
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
        if (idx + 1) % 200 == 0 or (idx + 1) == len(refs):
            print(f"[lpb_score] {progress_label} {idx + 1}/{len(refs)} hits={cache_hits} misses={cache_misses}")

    print(
        f"[lpb_score] {progress_label} cache_root={to_absolute_path(str(cache_root))} "
        f"cache_hits={cache_hits} cache_misses={cache_misses}"
    )
    return cached_refs


def estimate_trajectories_nbytes(trajectories: Sequence[PreparedTrajectory]) -> int:
    total = 0
    for traj in trajectories:
        total += int(traj.images.nbytes)
        total += int(traj.proprio.nbytes)
        total += int(traj.actions.nbytes)
    return int(total)


class LatentTransitionDataset(Dataset):
    """Temporal raw-input windows with task and class labels."""

    def __init__(
        self,
        trajectories: Sequence[PreparedTrajectory] | None = None,
        cached_refs: Sequence[CachedTrajectoryRef] | None = None,
        window_size: int = 8,
        cache_trajectory_limit: int = 4,
    ) -> None:
        super().__init__()
        self.trajectories = list(trajectories or [])
        self.cached_refs = list(cached_refs or [])
        self.window_size = int(window_size)
        self.cache_trajectory_limit = max(0, int(cache_trajectory_limit))
        if self.window_size <= 0:
            raise ValueError(f"window_size must be positive, got {self.window_size}")
        if bool(self.trajectories) == bool(self.cached_refs):
            raise ValueError("Provide exactly one of trajectories or cached_refs.")

        self._transition_refs: list[TransitionRef] = []
        self._sample_is_positive: list[bool] = []
        self._trajectory_cache: OrderedDict[int, TensorTrajectory] = OrderedDict()
        self._trajectory_tensors: list[TensorTrajectory] = []
        self._window_index_cache: dict[int, torch.Tensor] = {}

        sources: Sequence[PreparedTrajectory | CachedTrajectoryRef]
        if self.trajectories:
            self._num_cameras = int(self.trajectories[0].images.shape[1])
            self._proprio_dim = int(self.trajectories[0].proprio.shape[1])
            sources = self.trajectories
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
                yield self._get_cached_trajectory(idx)

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

    def __len__(self) -> int:
        return len(self._transition_refs)

    @staticmethod
    def _as_float_tensor(value: np.ndarray) -> torch.Tensor:
        array = np.asarray(value)
        if array.dtype == np.uint8:
            array = array.astype(np.float32) / 255.0
        else:
            array = np.asarray(array, dtype=np.float32)
        return torch.from_numpy(np.ascontiguousarray(array))

    @staticmethod
    def _as_long_scalar(value: int) -> torch.Tensor:
        return torch.tensor(int(value), dtype=torch.int64)

    def _prepare_tensor_trajectory(self, traj: PreparedTrajectory) -> TensorTrajectory:
        return TensorTrajectory(
            images=self._as_float_tensor(traj.images),
            proprio=torch.from_numpy(np.ascontiguousarray(np.asarray(traj.proprio, dtype=np.float32))),
            actions=torch.from_numpy(np.ascontiguousarray(np.asarray(traj.actions, dtype=np.float32))),
            task_index_tensor=self._as_long_scalar(traj.task_index),
            data_type_index_tensor=self._as_long_scalar(traj.data_type_index),
            task_name=traj.task_name,
            task_index=int(traj.task_index),
            data_type=traj.data_type,
            data_type_index=int(traj.data_type_index),
            split=traj.split,
            file_path=traj.file_path,
            demo_key=traj.demo_key,
        )

    def _load_cached_trajectory(self, ref: CachedTrajectoryRef) -> TensorTrajectory:
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
        cached = self._trajectory_cache.get(int(trajectory_index))
        if cached is not None:
            self._trajectory_cache.move_to_end(int(trajectory_index))
            return cached
        traj = self._load_cached_trajectory(self.cached_refs[int(trajectory_index)])
        if self.cache_trajectory_limit > 0:
            self._trajectory_cache[int(trajectory_index)] = traj
            while len(self._trajectory_cache) > self.cache_trajectory_limit:
                self._trajectory_cache.popitem(last=False)
        return traj

    def _get_trajectory(self, trajectory_index: int) -> TensorTrajectory:
        if self.trajectories:
            return self._trajectory_tensors[int(trajectory_index)]
        return self._get_cached_trajectory(int(trajectory_index))

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
