"""Latent data pipeline: HDF5 splits, ``.npz`` caches, and temporal window indexing."""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import h5py
import numpy as np
import torch
from hydra.utils import to_absolute_path
from torch.utils.data import Dataset

from robosuite.discriminator.dyn_bce.modules.flow_encoder import FrozenFlowMultitaskEncoder
from robosuite.discriminator.dyn_bce.task_registry import ordered_task_names


# Stable indices for ``EncodedTrajectoryRef.data_type_index`` (metadata and batch tensors).
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
class EncodedTrajectoryRef:
    task_name: str
    task_index: int
    data_type: str
    data_type_index: int
    split: str
    file_path: str
    demo_key: str
    cache_path: str


@dataclass(frozen=True)
class TransitionRef:
    trajectory_index: int
    t: int
    traj_type: int


@dataclass(frozen=True)
class LatentTrajectory:
    latents: np.ndarray
    actions: np.ndarray
    task_name: str
    task_index: int
    data_type: str
    data_type_index: int
    split: str
    file_path: str
    demo_key: str


def _cfg_get(cfg: Any, key: str, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def resolve_window_size(cfg: Any, default: int = 8) -> int:
    """Resolve the canonical window size with a legacy fallback."""
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
    """Normalize per-task config into absolute paths and split counts."""
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
    """Sample demo references for train/val/test across all tasks and buckets."""
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


def _print_split_summary(split_summary: dict[str, dict[str, dict[str, int]]]) -> None:
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


def prepare_cached_trajectories(
    refs: Sequence[DemoRef],
    encoder: FrozenFlowMultitaskEncoder,
    cache_root: str,
    task_to_index: dict[str, int],
    encode_demo_batch_size: int = 8,
    fallback_cache_roots: Sequence[str] | None = None,
    build_missing_cache: bool = True,
) -> list[EncodedTrajectoryRef]:
    """Encode raw demos once and persist latent caches for later reuse."""
    cache_root = to_absolute_path(str(cache_root))
    os.makedirs(cache_root, exist_ok=True)
    batch_size = max(1, int(encode_demo_batch_size))
    fallback_roots = [to_absolute_path(str(root)) for root in (fallback_cache_roots or [])]

    prepared_batch = []
    prepared_meta: list[DemoRef] = []
    encoded_refs: list[EncodedTrajectoryRef] = []

    def flush_batch() -> None:
        nonlocal prepared_batch, prepared_meta
        if not prepared_batch:
            return
        # Batch encoding amortizes the frozen policy forward pass.
        encoded_batch = encoder.encode_prepared_demos(prepared_batch)
        encoded_lookup = {
            (encoded.file_path, encoded.demo_key): encoded
            for encoded in encoded_batch
        }
        for ref in prepared_meta:
            key = (ref.file_path, ref.demo_key)
            encoded = encoded_lookup.get(key)
            if encoded is None:
                raise KeyError(f"Missing encoded batch result for {key}")
            cache_path = encoder.save_encoded_demo(cache_root=cache_root, encoded=encoded)
            encoded_refs.append(
                EncodedTrajectoryRef(
                    task_name=ref.task_name,
                    task_index=int(task_to_index[ref.task_name]),
                    data_type=ref.data_type,
                    data_type_index=int(DATA_TYPE_ORDER.index(ref.data_type)),
                    split=ref.split,
                    file_path=ref.file_path,
                    demo_key=ref.demo_key,
                    cache_path=cache_path,
                )
            )
        prepared_batch = []
        prepared_meta = []

    for idx, ref in enumerate(refs):
        cache_path = encoder.cache_path(
            cache_root=cache_root,
            task_name=ref.task_name,
            file_path=ref.file_path,
            demo_key=ref.demo_key,
        )
        resolved_cache_path = cache_path
        if not os.path.isfile(resolved_cache_path):
            for legacy_root in fallback_roots:
                legacy_cache_path = encoder.cache_path(
                    cache_root=legacy_root,
                    task_name=ref.task_name,
                    file_path=ref.file_path,
                    demo_key=ref.demo_key,
                )
                if os.path.isfile(legacy_cache_path):
                    resolved_cache_path = legacy_cache_path
                    break

        if os.path.isfile(resolved_cache_path):
            encoded_refs.append(
                EncodedTrajectoryRef(
                    task_name=ref.task_name,
                    task_index=int(task_to_index[ref.task_name]),
                    data_type=ref.data_type,
                    data_type_index=int(DATA_TYPE_ORDER.index(ref.data_type)),
                    split=ref.split,
                    file_path=ref.file_path,
                    demo_key=ref.demo_key,
                    cache_path=resolved_cache_path,
                )
            )
        else:
            if not bool(build_missing_cache):
                raise FileNotFoundError(
                    "Missing latent cache while build_missing_cache=False: "
                    f"task={ref.task_name} data_type={ref.data_type} split={ref.split} "
                    f"file={ref.file_path} demo={ref.demo_key}"
                )
            prepared_batch.append(
                encoder.load_demo_raw(
                    task_name=ref.task_name,
                    file_path=ref.file_path,
                    demo_key=ref.demo_key,
                )
            )
            prepared_meta.append(ref)
            if len(prepared_batch) >= batch_size:
                flush_batch()

        if (idx + 1) % 1000 == 0 or (idx + 1) == len(refs):
            print(f"[lpb_score] prepared_cached_trajectories {idx + 1}/{len(refs)}")

    flush_batch()
    return encoded_refs


def load_cached_latent_trajectory(ref: EncodedTrajectoryRef) -> LatentTrajectory:
    with np.load(ref.cache_path) as cached:
        latents = np.asarray(cached["latents"], dtype=np.float32)
        actions = np.asarray(cached["actions"], dtype=np.float32)
    return LatentTrajectory(
        latents=latents,
        actions=actions,
        task_name=ref.task_name,
        task_index=ref.task_index,
        data_type=ref.data_type,
        data_type_index=ref.data_type_index,
        split=ref.split,
        file_path=ref.file_path,
        demo_key=ref.demo_key,
    )


def load_latent_trajectories(refs: Sequence[EncodedTrajectoryRef]) -> list[LatentTrajectory]:
    trajectories = [load_cached_latent_trajectory(ref) for ref in refs]
    return [traj for traj in trajectories if int(traj.latents.shape[0]) > 0]


def filter_refs_by_data_types(
    refs: Sequence[EncodedTrajectoryRef],
    data_types: Sequence[str] | None,
) -> list[EncodedTrajectoryRef]:
    if data_types is None:
        return list(refs)
    allowed = {str(name) for name in data_types}
    return [ref for ref in refs if ref.data_type in allowed]


class LatentTransitionDataset(Dataset):
    """Temporal latent-window samples with ``traj_type`` and ``task_index``."""

    def __init__(
        self,
        trajectory_refs: Sequence[EncodedTrajectoryRef],
        window_size: int = 8,
        preload_to_memory: bool = False,
    ) -> None:
        super().__init__()
        self.trajectory_refs = list(trajectory_refs)
        self.window_size = int(window_size)
        if self.window_size <= 0:
            raise ValueError(f"window_size must be positive, got {self.window_size}")
        if not self.trajectory_refs:
            raise ValueError("trajectory_refs cannot be empty")

        self.preload_to_memory = bool(preload_to_memory)
        self._trajectory_cache: dict[int, LatentTrajectory] = {}
        self._latent_dim: Optional[int] = None
        self._transition_refs: list[TransitionRef] = []
        self._sample_is_positive: list[bool] = []

        for traj_idx, ref in enumerate(self.trajectory_refs):
            traj = load_cached_latent_trajectory(ref)
            length = int(traj.latents.shape[0])
            if length <= 0:
                continue
            if self._latent_dim is None:
                self._latent_dim = int(traj.latents.shape[-1])
            elif int(traj.latents.shape[-1]) != self._latent_dim:
                raise ValueError(
                    f"Latent dim mismatch in {traj.file_path}:{traj.demo_key}. "
                    f"expected={self._latent_dim}, got={traj.latents.shape[-1]}"
                )
            for t in range(length):
                traj_type = 1 if ref.data_type == "fail_rollout" else 0
                self._transition_refs.append(
                    TransitionRef(
                        trajectory_index=traj_idx,
                        t=t,
                        traj_type=traj_type,
                    )
                )
                self._sample_is_positive.append(traj_type == 0)
            if self.preload_to_memory:
                self._trajectory_cache[traj_idx] = traj

        if not self._transition_refs:
            raise RuntimeError("No valid latent windows found for the provided trajectory refs.")

    @property
    def latent_dim(self) -> int:
        assert self._latent_dim is not None
        return self._latent_dim

    @property
    def sample_is_positive(self) -> list[bool]:
        return self._sample_is_positive

    @property
    def num_positive_samples(self) -> int:
        return int(sum(self._sample_is_positive))

    @property
    def num_negative_samples(self) -> int:
        return len(self._sample_is_positive) - self.num_positive_samples

    def __len__(self) -> int:
        return len(self._transition_refs)

    def _get_trajectory(self, trajectory_index: int) -> LatentTrajectory:
        traj = self._trajectory_cache.get(trajectory_index)
        if traj is not None:
            return traj
        traj = load_cached_latent_trajectory(self.trajectory_refs[trajectory_index])
        self._trajectory_cache[trajectory_index] = traj
        return traj

    def _build_latent_window(self, traj: LatentTrajectory, t: int) -> np.ndarray:
        latents = np.asarray(traj.latents, dtype=np.float32)
        start = int(t) - self.window_size + 1
        if start >= 0:
            return latents[start : int(t) + 1]
        pad = np.repeat(latents[:1], -start, axis=0)
        return np.concatenate([pad, latents[: int(t) + 1]], axis=0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return tensors for one latent window."""
        ref = self._transition_refs[index]
        traj = self._get_trajectory(ref.trajectory_index)
        latent_window = self._build_latent_window(traj, t=int(ref.t))
        return {
            "latent_window": torch.from_numpy(np.asarray(latent_window, dtype=np.float32)),
            "task_index": torch.tensor(int(traj.task_index), dtype=torch.int64),
            "data_type_index": torch.tensor(int(traj.data_type_index), dtype=torch.int64),
            "traj_type": torch.tensor(int(ref.traj_type), dtype=torch.int64),
        }


def build_cached_splits(
    cfg_data: Any,
    encoder: FrozenFlowMultitaskEncoder,
    seed: int,
    build_missing_cache: bool = True,
) -> tuple[dict[str, list[EncodedTrajectoryRef]], dict[str, dict[str, dict[str, int]]], dict[str, int]]:
    """Split demos by config, then resolve or create per-demo ``.npz`` caches under ``cache_dir``."""
    split_refs, split_summary, task_to_index = build_split_refs(cfg_data=cfg_data, seed=seed)
    _print_split_summary(split_summary)

    cached_splits: dict[str, list[EncodedTrajectoryRef]] = {}
    cache_root = to_absolute_path(str(_cfg_get(cfg_data, "cache_dir")))
    batch_size = int(_cfg_get(cfg_data, "encode_demo_batch_size", 8))
    legacy_cache_root = to_absolute_path("./data/.lpb_new_cache")
    fallback_cache_roots: list[str] = []
    if os.path.abspath(cache_root) != os.path.abspath(legacy_cache_root) and os.path.isdir(legacy_cache_root):
        fallback_cache_roots.append(legacy_cache_root)
        print(f"[lpb_score] Reusing legacy cache when available: {legacy_cache_root}")

    for split_name, refs in split_refs.items():
        action_label = "building latent cache" if bool(build_missing_cache) else "loading latent cache"
        print(f"[lpb_score] {action_label} for split={split_name} num_trajectories={len(refs)}")
        cached_splits[split_name] = prepare_cached_trajectories(
            refs=refs,
            encoder=encoder,
            cache_root=cache_root,
            task_to_index=task_to_index,
            encode_demo_batch_size=batch_size,
            fallback_cache_roots=fallback_cache_roots,
            build_missing_cache=bool(build_missing_cache),
        )
    return cached_splits, split_summary, task_to_index
