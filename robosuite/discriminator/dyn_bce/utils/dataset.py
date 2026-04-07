from __future__ import annotations

import concurrent.futures
import glob
import hashlib
import json
import os
from bisect import bisect_right
from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np
import torch
from hydra.utils import to_absolute_path
from torch.utils.data import Dataset
from tqdm import tqdm

from robosuite.discriminator.dyn_bce.modules.flow_encoder import EncodedDemo, FrozenFlowMultitaskEncoder
from robosuite.discriminator.dyn_bce.task_registry import ordered_task_names


DATA_TYPE_ORDER = ["expert", "success_rollout", "fail_rollout"]
ARRAY_DTYPES = {
    "current_latent": np.float32,
    "next_latent": np.float32,
    "action_sequence": np.float32,
    "risk_target": np.float32,
    "hard_label": np.int64,
    "occ_weight": np.float32,
    "fuse_weight": np.float32,
    "dyn_weight": np.float32,
    "task_index": np.int64,
    "data_type_index": np.int64,
}


@dataclass(frozen=True)
class SplitCounts:
    num_expert_traj: int
    num_success_traj: int
    num_fail_traj: int

    def by_data_type(self) -> dict[str, int]:
        return {
            "expert": int(self.num_expert_traj),
            "success_rollout": int(self.num_success_traj),
            "fail_rollout": int(self.num_fail_traj),
        }


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
class SplitMemmapInfo:
    split_name: str
    manifest_path: str
    num_trajectories: int
    num_transitions: int


class DynBCETransitionDataset(Dataset):
    def __init__(self, manifest_path: str) -> None:
        super().__init__()
        self.manifest_path = to_absolute_path(str(manifest_path))
        with open(self.manifest_path, "r", encoding="utf-8") as file_handle:
            manifest = json.load(file_handle)

        self.split_name = str(manifest["split_name"])
        self.transition_horizon = int(manifest["transition_horizon"])
        self.num_transitions = int(manifest["num_transitions"])
        self.task_names = [str(name) for name in manifest["task_names"]]
        self.shards = list(manifest["shards"])
        self._cumulative_sizes = np.asarray(
            manifest["cumulative_sizes"],
            dtype=np.int64,
        )
        self._cumulative_sizes_list = self._cumulative_sizes.tolist()
        self._opened_arrays: dict[int, dict[str, np.ndarray]] = {}

    def __len__(self) -> int:
        return self.num_transitions

    def _open_shard(self, shard_idx: int) -> dict[str, np.ndarray]:
        opened = self._opened_arrays.get(shard_idx)
        if opened is not None:
            return opened

        shard = self.shards[shard_idx]
        arrays = {
            name: np.load(to_absolute_path(path), mmap_mode="r")
            for name, path in shard["files"].items()
        }
        self._opened_arrays[shard_idx] = arrays
        return arrays

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= self.num_transitions:
            raise IndexError(f"Index out of range: {index}")

        shard_idx = int(bisect_right(self._cumulative_sizes_list, index))
        prev_end = int(self._cumulative_sizes[shard_idx - 1]) if shard_idx > 0 else 0
        local_idx = int(index - prev_end)
        arrays = self._open_shard(shard_idx)

        return {
            "current_latent": torch.tensor(arrays["current_latent"][local_idx], dtype=torch.float32),
            "next_latent": torch.tensor(arrays["next_latent"][local_idx], dtype=torch.float32),
            "action_sequence": torch.tensor(arrays["action_sequence"][local_idx], dtype=torch.float32),
            "risk_target": torch.tensor(float(arrays["risk_target"][local_idx]), dtype=torch.float32),
            "hard_label": torch.tensor(int(arrays["hard_label"][local_idx]), dtype=torch.int64),
            "occ_weight": torch.tensor(float(arrays["occ_weight"][local_idx]), dtype=torch.float32),
            "fuse_weight": torch.tensor(float(arrays["fuse_weight"][local_idx]), dtype=torch.float32),
            "dyn_weight": torch.tensor(float(arrays["dyn_weight"][local_idx]), dtype=torch.float32),
            "task_index": torch.tensor(int(arrays["task_index"][local_idx]), dtype=torch.int64),
            "data_type_index": torch.tensor(int(arrays["data_type_index"][local_idx]), dtype=torch.int64),
        }


def _list_hdf5_demo_refs(task_name: str, data_type: str, data_dir: str) -> list[tuple[str, str]]:
    if not os.path.isdir(data_dir):
        print(f"[dyn_bce] Missing directory for {task_name}/{data_type}: {data_dir}. Using 0 trajectories.")
        return []

    refs: list[tuple[str, str]] = []
    hdf5_files = sorted(glob.glob(os.path.join(data_dir, "*.hdf5")))
    if not hdf5_files:
        print(f"[dyn_bce] No .hdf5 files for {task_name}/{data_type}: {data_dir}. Using 0 trajectories.")
        return []

    for file_path in hdf5_files:
        with h5py.File(file_path, "r") as file_handle:
            if "demos" not in file_handle:
                continue
            for demo_key in sorted(file_handle["demos"].keys()):
                refs.append((file_path, demo_key))

    if not refs:
        print(f"[dyn_bce] No demos found for {task_name}/{data_type}: {data_dir}. Using 0 trajectories.")
    return refs


def _split_refs_with_counts(
    refs: list[tuple[str, str]],
    train_count: int,
    val_count: int,
    test_count: int,
    seed: int,
    task_name: str,
    data_type: str,
) -> dict[str, list[tuple[str, str]]]:
    requested_total = int(train_count) + int(val_count) + int(test_count)
    if requested_total <= 0 or len(refs) <= 0:
        return {"train": [], "val": [], "test": []}

    rng = np.random.default_rng(int(seed))
    indices = np.arange(len(refs))
    rng.shuffle(indices)
    selected = [refs[idx] for idx in indices[: min(len(refs), requested_total)]]

    if len(selected) < requested_total:
        print(
            f"[dyn_bce] task={task_name} data_type={data_type} requested "
            f"(train={train_count}, val={val_count}, test={test_count}, total={requested_total}) "
            f"but only found {len(refs)} trajectories. "
            "Only this task/data_type bucket will fall back to loading all available trajectories; "
            "other task buckets still follow the configured CLI counts."
        )

    train_end = min(int(train_count), len(selected))
    val_end = min(train_end + int(val_count), len(selected))
    test_end = min(val_end + int(test_count), len(selected))
    return {
        "train": selected[:train_end],
        "val": selected[train_end:val_end],
        "test": selected[val_end:test_end],
    }


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def estimate_weighted_occupancy_positive_prior(
    fail_onset_ratio: float,
    risk_temperature: float,
    occ_fail_prefix_min_weight: float,
    num_steps: int = 10001,
) -> float:
    if int(num_steps) <= 1:
        progress = np.asarray([0.5], dtype=np.float64)
    else:
        progress = np.linspace(0.0, 1.0, int(num_steps), dtype=np.float64)

    center = float(fail_onset_ratio)
    temperature = max(float(risk_temperature), 1e-4)
    risk = _sigmoid((progress - center) / temperature).astype(np.float64)
    occ_weights = (
        float(occ_fail_prefix_min_weight)
        + (1.0 - float(occ_fail_prefix_min_weight)) * risk
    ).astype(np.float64)
    positive_mass = (1.0 - risk) * occ_weights
    prior = positive_mass.sum() / occ_weights.sum().clip(min=1e-12)
    return float(np.clip(prior, 1e-4, 1.0 - 1e-4))


def _build_fail_schedule(
    num_steps: int,
    fail_onset_ratio: float,
    risk_temperature: float,
    fuse_prefix_min_weight: float,
    dyn_prefix_min_weight: float,
    occ_fail_prefix_min_weight: float,
) -> dict[str, np.ndarray]:
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")

    if num_steps == 1:
        progress = np.asarray([1.0], dtype=np.float32)
    else:
        progress = np.linspace(0.0, 1.0, num_steps, dtype=np.float32)

    center = float(fail_onset_ratio)
    temperature = max(float(risk_temperature), 1e-4)
    sigmoid_curve = _sigmoid((progress - center) / temperature).astype(np.float32)
    risk = sigmoid_curve
    occ_weights = (
        float(occ_fail_prefix_min_weight)
        + (1.0 - float(occ_fail_prefix_min_weight)) * sigmoid_curve
    ).astype(np.float32)
    fuse_weights = (
        float(fuse_prefix_min_weight)
        + (1.0 - float(fuse_prefix_min_weight)) * sigmoid_curve
    ).astype(np.float32)
    dyn_weights = (
        float(dyn_prefix_min_weight)
        + (1.0 - float(dyn_prefix_min_weight)) * (1.0 - sigmoid_curve)
    ).astype(np.float32)
    hard_labels = (sigmoid_curve >= 0.5).astype(np.int64)

    return {
        "risk_targets": risk,
        "hard_labels": hard_labels,
        "occ_weights": occ_weights,
        "fuse_weights": fuse_weights,
        "dyn_weights": dyn_weights,
    }


def _build_positive_schedule(num_steps: int) -> dict[str, np.ndarray]:
    zeros = np.zeros(num_steps, dtype=np.float32)
    ones = np.ones(num_steps, dtype=np.float32)
    return {
        "risk_targets": zeros,
        "hard_labels": np.zeros(num_steps, dtype=np.int64),
        "occ_weights": ones,
        "fuse_weights": ones,
        "dyn_weights": ones,
    }


def _cfg_get(cfg: Any, key: str, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _split_counts_from_cfg(cfg: Any) -> SplitCounts:
    return SplitCounts(
        num_expert_traj=int(_cfg_get(cfg, "num_expert_traj")),
        num_success_traj=int(_cfg_get(cfg, "num_success_traj")),
        num_fail_traj=int(_cfg_get(cfg, "num_fail_traj")),
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


def _build_ref_splits(
    task_specs: dict[str, TaskDataSpec],
    seed: int,
) -> tuple[dict[str, list[DemoRef]], dict[str, dict[str, dict[str, int]]]]:
    split_refs = {"train": [], "val": [], "test": []}
    split_summary: dict[str, dict[str, dict[str, int]]] = {}

    for task_offset, task_name in enumerate(ordered_task_names(list(task_specs.keys()))):
        spec = task_specs[task_name]
        split_summary[task_name] = {"train": {}, "val": {}, "test": {}}
        split_cfgs = {"train": spec.train, "val": spec.val, "test": spec.test}
        for data_type_idx, data_type in enumerate(DATA_TYPE_ORDER):
            refs = _list_hdf5_demo_refs(
                task_name=task_name,
                data_type=data_type,
                data_dir=spec.dir_for(data_type),
            )
            chosen = _split_refs_with_counts(
                refs=refs,
                train_count=split_cfgs["train"].by_data_type()[data_type],
                val_count=split_cfgs["val"].by_data_type()[data_type],
                test_count=split_cfgs["test"].by_data_type()[data_type],
                seed=int(seed + task_offset * 97 + data_type_idx * 17),
                task_name=task_name,
                data_type=data_type,
            )
            for split_name in ["train", "val", "test"]:
                split_summary[task_name][split_name][data_type] = int(len(chosen[split_name]))
                for file_path, demo_key in chosen[split_name]:
                    split_refs[split_name].append(
                        DemoRef(
                            task_name=task_name,
                            data_type=data_type,
                            split=split_name,
                            file_path=file_path,
                            demo_key=demo_key,
                        )
                    )
    return split_refs, split_summary


def _make_schedule(
    data_type: str,
    num_steps: int,
    fail_onset_ratio: float,
    risk_temperature: float,
    fuse_prefix_min_weight: float,
    dyn_prefix_min_weight: float,
    occ_fail_prefix_min_weight: float,
) -> dict[str, np.ndarray]:
    if data_type == "fail_rollout":
        return _build_fail_schedule(
            num_steps=num_steps,
            fail_onset_ratio=float(fail_onset_ratio),
            risk_temperature=float(risk_temperature),
            fuse_prefix_min_weight=float(fuse_prefix_min_weight),
            dyn_prefix_min_weight=float(dyn_prefix_min_weight),
            occ_fail_prefix_min_weight=float(occ_fail_prefix_min_weight),
        )
    return _build_positive_schedule(num_steps=num_steps)


def _build_transition_arrays(
    encoded,
    ref: DemoRef,
    task_index: int,
    transition_horizon: int,
    fail_onset_ratio: float,
    risk_temperature: float,
    fuse_prefix_min_weight: float,
    dyn_prefix_min_weight: float,
    occ_fail_prefix_min_weight: float,
) -> dict[str, np.ndarray] | None:
    usable = min(int(encoded.latents.shape[0]), int(encoded.actions.shape[0])) - int(transition_horizon)
    if usable <= 0:
        return None

    schedule = _make_schedule(
        data_type=ref.data_type,
        num_steps=usable,
        fail_onset_ratio=fail_onset_ratio,
        risk_temperature=risk_temperature,
        fuse_prefix_min_weight=fuse_prefix_min_weight,
        dyn_prefix_min_weight=dyn_prefix_min_weight,
        occ_fail_prefix_min_weight=occ_fail_prefix_min_weight,
    )
    action_sequences = np.stack(
        [encoded.actions[t : t + transition_horizon] for t in range(usable)],
        axis=0,
    ).astype(np.float32)

    return {
        "current_latent": encoded.latents[:usable].astype(np.float32),
        "next_latent": encoded.latents[transition_horizon : transition_horizon + usable].astype(np.float32),
        "action_sequence": action_sequences,
        "risk_target": schedule["risk_targets"].astype(np.float32),
        "hard_label": schedule["hard_labels"].astype(np.int64),
        "occ_weight": schedule["occ_weights"].astype(np.float32),
        "fuse_weight": schedule["fuse_weights"].astype(np.float32),
        "dyn_weight": schedule["dyn_weights"].astype(np.float32),
        "task_index": np.full(usable, int(task_index), dtype=np.int64),
        "data_type_index": np.full(usable, DATA_TYPE_ORDER.index(ref.data_type), dtype=np.int64),
    }


def _memmap_signature(
    split_refs: dict[str, list[DemoRef]],
    task_names: list[str],
    transition_horizon: int,
    fail_onset_ratio: float,
    risk_temperature: float,
    fuse_prefix_min_weight: float,
    dyn_prefix_min_weight: float,
    occ_fail_prefix_min_weight: float,
    checkpoint_path: str,
    image_size: int,
) -> str:
    payload = {
        "task_names": list(task_names),
        "transition_horizon": int(transition_horizon),
        "fail_onset_ratio": float(fail_onset_ratio),
        "risk_temperature": float(risk_temperature),
        "fuse_prefix_min_weight": float(fuse_prefix_min_weight),
        "dyn_prefix_min_weight": float(dyn_prefix_min_weight),
        "occ_fail_prefix_min_weight": float(occ_fail_prefix_min_weight),
        "checkpoint_path": to_absolute_path(str(checkpoint_path)),
        "image_size": int(image_size),
        "splits": {
            split_name: [
                {
                    "task_name": ref.task_name,
                    "data_type": ref.data_type,
                    "file_path": os.path.abspath(ref.file_path),
                    "demo_key": ref.demo_key,
                }
                for ref in refs
            ]
            for split_name, refs in split_refs.items()
        },
    }
    return hashlib.sha1(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _append_chunk(
    buffer: dict[str, list[np.ndarray]],
    arrays: dict[str, np.ndarray],
    start_idx: int,
    end_idx: int,
) -> None:
    for key in ARRAY_DTYPES:
        buffer[key].append(np.asarray(arrays[key][start_idx:end_idx], dtype=ARRAY_DTYPES[key]))


def _flush_shard(
    split_dir: str,
    split_name: str,
    shard_idx: int,
    buffer: dict[str, list[np.ndarray]],
) -> dict[str, Any]:
    num_samples = int(sum(chunk.shape[0] for chunk in buffer["risk_target"]))
    if num_samples <= 0:
        raise ValueError(f"Cannot flush empty shard for split={split_name}")

    shard_prefix = f"{split_name}_shard_{shard_idx:05d}"
    files: dict[str, str] = {}
    for key, dtype in ARRAY_DTYPES.items():
        shard_array = np.concatenate(buffer[key], axis=0).astype(dtype, copy=False)
        file_path = os.path.join(split_dir, f"{shard_prefix}_{key}.npy")
        np.save(file_path, shard_array)
        files[key] = file_path
    return {
        "shard_idx": int(shard_idx),
        "num_samples": num_samples,
        "files": files,
    }


def _load_encoded_demo_from_cache(cache_path: str) -> EncodedDemo:
    with np.load(cache_path) as cached:
        return EncodedDemo(
            latents=np.asarray(cached["latents"], dtype=np.float32),
            actions=np.asarray(cached["actions"], dtype=np.float32),
            task_name=str(cached["task_name"].item()),
            file_path=str(cached["file_path"].item()),
            demo_key=str(cached["demo_key"].item()),
        )


def _iter_encoded_demos(
    refs: list[DemoRef],
    encoder: FrozenFlowMultitaskEncoder,
    latent_cache_dir: str,
    split_name: str,
    num_build_workers: int,
    encode_demo_batch_size: int,
    max_pending_raw_demos: int,
):
    cache_hits: list[tuple[DemoRef, str]] = []
    cache_misses: list[DemoRef] = []
    for ref in refs:
        cache_path = encoder.cache_path(
            cache_root=latent_cache_dir,
            task_name=ref.task_name,
            file_path=ref.file_path,
            demo_key=ref.demo_key,
        )
        if os.path.isfile(cache_path):
            cache_hits.append((ref, cache_path))
        else:
            cache_misses.append(ref)

    print(
        f"[dyn_bce] split={split_name} cache_hits={len(cache_hits)} cache_misses={len(cache_misses)} "
        f"build_workers={num_build_workers} encode_demo_batch_size={encode_demo_batch_size} "
        f"max_pending_raw_demos={max_pending_raw_demos}"
    )

    if cache_hits:
        max_workers = max(1, min(int(num_build_workers), len(cache_hits)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ref = {
                executor.submit(_load_encoded_demo_from_cache, cache_path): ref
                for ref, cache_path in cache_hits
            }
            for future in tqdm(
                concurrent.futures.as_completed(future_to_ref),
                total=len(future_to_ref),
                desc=f"loading dyn_bce cached split={split_name}",
            ):
                ref = future_to_ref[future]
                yield ref, future.result()

    if cache_misses:
        miss_ref_map = {
            (ref.file_path, ref.demo_key): ref
            for ref in cache_misses
        }
        max_workers = max(1, min(int(num_build_workers), len(cache_misses)))
        pending_limit = max(
            int(encode_demo_batch_size),
            min(int(max_pending_raw_demos), len(cache_misses)),
        )
        submitted = 0
        completed = 0
        prepared_batch = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            pending_futures: dict[concurrent.futures.Future, DemoRef] = {}

            progress = tqdm(total=len(cache_misses), desc=f"preloading dyn_bce raw split={split_name}")

            def submit_until_limit() -> None:
                nonlocal submitted
                while submitted < len(cache_misses) and len(pending_futures) < pending_limit:
                    ref = cache_misses[submitted]
                    future = executor.submit(
                        encoder.load_demo_raw,
                        task_name=ref.task_name,
                        file_path=ref.file_path,
                        demo_key=ref.demo_key,
                    )
                    pending_futures[future] = ref
                    submitted += 1

            submit_until_limit()
            while pending_futures:
                done, _ = concurrent.futures.wait(
                    pending_futures.keys(),
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    ref = pending_futures.pop(future)
                    prepared_batch.append(future.result())
                    completed += 1
                    progress.update(1)
                    if len(prepared_batch) >= int(encode_demo_batch_size):
                        encoded_batch = encoder.encode_prepared_demos(prepared_batch)
                        for encoded in encoded_batch:
                            encoder.save_encoded_demo(cache_root=latent_cache_dir, encoded=encoded)
                            matched_ref = miss_ref_map[(encoded.file_path, encoded.demo_key)]
                            yield matched_ref, encoded
                        prepared_batch = []
                submit_until_limit()

            progress.close()

        if prepared_batch:
            encoded_batch = encoder.encode_prepared_demos(prepared_batch)
            for encoded in encoded_batch:
                encoder.save_encoded_demo(cache_root=latent_cache_dir, encoded=encoded)
                matched_ref = miss_ref_map[(encoded.file_path, encoded.demo_key)]
                yield matched_ref, encoded


def _write_split_memmap(
    split_name: str,
    refs: list[DemoRef],
    encoder: FrozenFlowMultitaskEncoder,
    memmap_root: str,
    task_to_index: dict[str, int],
    transition_horizon: int,
    fail_onset_ratio: float,
    risk_temperature: float,
    fuse_prefix_min_weight: float,
    dyn_prefix_min_weight: float,
    occ_fail_prefix_min_weight: float,
    max_transitions_per_shard: int,
    latent_cache_dir: str,
    num_build_workers: int,
    encode_demo_batch_size: int,
    max_pending_raw_demos: int,
) -> SplitMemmapInfo:
    split_dir = os.path.join(memmap_root, split_name)
    os.makedirs(split_dir, exist_ok=True)

    buffer = {key: [] for key in ARRAY_DTYPES}
    buffered_samples = 0
    shard_idx = 0
    shard_specs: list[dict[str, Any]] = []
    num_transitions = 0

    for ref, encoded in _iter_encoded_demos(
        refs=refs,
        encoder=encoder,
        latent_cache_dir=latent_cache_dir,
        split_name=split_name,
        num_build_workers=num_build_workers,
        encode_demo_batch_size=encode_demo_batch_size,
        max_pending_raw_demos=max_pending_raw_demos,
    ):
        arrays = _build_transition_arrays(
            encoded=encoded,
            ref=ref,
            task_index=task_to_index[ref.task_name],
            transition_horizon=transition_horizon,
            fail_onset_ratio=fail_onset_ratio,
            risk_temperature=risk_temperature,
            fuse_prefix_min_weight=fuse_prefix_min_weight,
            dyn_prefix_min_weight=dyn_prefix_min_weight,
            occ_fail_prefix_min_weight=occ_fail_prefix_min_weight,
        )
        if arrays is None:
            continue

        total = int(arrays["risk_target"].shape[0])
        start = 0
        while start < total:
            remaining_capacity = max(int(max_transitions_per_shard) - buffered_samples, 1)
            stop = min(start + remaining_capacity, total)
            _append_chunk(buffer=buffer, arrays=arrays, start_idx=start, end_idx=stop)
            chunk_size = stop - start
            buffered_samples += chunk_size
            num_transitions += chunk_size
            start = stop

            if buffered_samples >= int(max_transitions_per_shard):
                shard_specs.append(
                    _flush_shard(
                        split_dir=split_dir,
                        split_name=split_name,
                        shard_idx=shard_idx,
                        buffer=buffer,
                    )
                )
                shard_idx += 1
                buffer = {key: [] for key in ARRAY_DTYPES}
                buffered_samples = 0

    if buffered_samples > 0:
        shard_specs.append(
            _flush_shard(
                split_dir=split_dir,
                split_name=split_name,
                shard_idx=shard_idx,
                buffer=buffer,
            )
        )

    cumulative_sizes: list[int] = []
    running = 0
    for spec in shard_specs:
        running += int(spec["num_samples"])
        cumulative_sizes.append(running)

    manifest = {
        "split_name": split_name,
        "transition_horizon": int(transition_horizon),
        "num_trajectories": int(len(refs)),
        "num_transitions": int(num_transitions),
        "task_names": list(task_to_index.keys()),
        "shards": shard_specs,
        "cumulative_sizes": cumulative_sizes,
    }
    manifest_path = os.path.join(split_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as file_handle:
        json.dump(manifest, file_handle, indent=2, sort_keys=True)

    return SplitMemmapInfo(
        split_name=split_name,
        manifest_path=manifest_path,
        num_trajectories=int(len(refs)),
        num_transitions=int(num_transitions),
    )


def _load_split_memmap_info(memmap_root: str, split_name: str) -> SplitMemmapInfo:
    manifest_path = os.path.join(memmap_root, split_name, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as file_handle:
        manifest = json.load(file_handle)
    return SplitMemmapInfo(
        split_name=split_name,
        manifest_path=manifest_path,
        num_trajectories=int(manifest["num_trajectories"]),
        num_transitions=int(manifest["num_transitions"]),
    )


def _build_or_load_memmap_splits(
    split_refs: dict[str, list[DemoRef]],
    cfg_data: Any,
    cfg_labels: Any,
    encoder: FrozenFlowMultitaskEncoder,
    task_names: list[str],
    task_to_index: dict[str, int],
) -> tuple[dict[str, SplitMemmapInfo], str]:
    transition_horizon = int(_cfg_get(cfg_data, "transition_horizon", 1))
    fail_onset_ratio = float(_cfg_get(cfg_labels, "fail_onset_ratio"))
    risk_temperature = float(_cfg_get(cfg_labels, "risk_temperature"))
    fuse_prefix_min_weight = float(_cfg_get(cfg_labels, "fuse_prefix_min_weight"))
    dyn_prefix_min_weight = float(_cfg_get(cfg_labels, "dyn_prefix_min_weight", 0.0))
    occ_fail_prefix_min_weight = float(_cfg_get(cfg_labels, "occ_fail_prefix_min_weight", 0.1))

    latent_cache_dir = to_absolute_path(str(_cfg_get(cfg_data, "cache_dir")))
    memmap_cfg = _cfg_get(cfg_data, "memmap", {})
    memmap_base_dir = to_absolute_path(
        str(_cfg_get(memmap_cfg, "dir", os.path.join(latent_cache_dir, "transition_memmap")))
    )
    max_transitions_per_shard = int(_cfg_get(memmap_cfg, "max_transitions_per_shard", 32768))
    num_build_workers = int(_cfg_get(memmap_cfg, "num_build_workers", 8))
    encode_demo_batch_size = int(_cfg_get(memmap_cfg, "encode_demo_batch_size", 8))
    max_pending_raw_demos = int(_cfg_get(memmap_cfg, "max_pending_raw_demos", 16))
    rebuild = bool(_cfg_get(memmap_cfg, "rebuild", False))

    signature = _memmap_signature(
        split_refs=split_refs,
        task_names=task_names,
        transition_horizon=transition_horizon,
        fail_onset_ratio=fail_onset_ratio,
        risk_temperature=risk_temperature,
        fuse_prefix_min_weight=fuse_prefix_min_weight,
        dyn_prefix_min_weight=dyn_prefix_min_weight,
        occ_fail_prefix_min_weight=occ_fail_prefix_min_weight,
        checkpoint_path=encoder.checkpoint_path,
        image_size=encoder.image_size,
    )
    memmap_root = os.path.join(memmap_base_dir, signature)

    split_infos: dict[str, SplitMemmapInfo] = {}
    manifests_exist = all(
        os.path.isfile(os.path.join(memmap_root, split_name, "manifest.json"))
        for split_name in ["train", "val", "test"]
    )
    if not rebuild and manifests_exist:
        for split_name in ["train", "val", "test"]:
            split_infos[split_name] = _load_split_memmap_info(memmap_root=memmap_root, split_name=split_name)
        return split_infos, memmap_root

    os.makedirs(memmap_root, exist_ok=True)
    for split_name in ["train", "val", "test"]:
        split_infos[split_name] = _write_split_memmap(
            split_name=split_name,
            refs=split_refs[split_name],
            encoder=encoder,
            memmap_root=memmap_root,
            task_to_index=task_to_index,
            transition_horizon=transition_horizon,
            fail_onset_ratio=fail_onset_ratio,
            risk_temperature=risk_temperature,
            fuse_prefix_min_weight=fuse_prefix_min_weight,
            dyn_prefix_min_weight=dyn_prefix_min_weight,
            occ_fail_prefix_min_weight=occ_fail_prefix_min_weight,
            max_transitions_per_shard=max_transitions_per_shard,
            latent_cache_dir=latent_cache_dir,
            num_build_workers=num_build_workers,
            encode_demo_batch_size=encode_demo_batch_size,
            max_pending_raw_demos=max_pending_raw_demos,
        )
    return split_infos, memmap_root


def build_datasets(
    cfg_data: Any,
    cfg_labels: Any,
    encoder: FrozenFlowMultitaskEncoder,
    seed: int,
) -> tuple[dict[str, DynBCETransitionDataset], dict[str, int], dict[str, Any]]:
    task_specs = parse_task_specs(cfg_data)
    split_refs, split_summary = _build_ref_splits(task_specs=task_specs, seed=int(seed))

    task_names = ordered_task_names(list(task_specs.keys()))
    task_to_index = {task_name: idx for idx, task_name in enumerate(task_names)}
    transition_horizon = int(_cfg_get(cfg_data, "transition_horizon", 1))
    split_infos, memmap_root = _build_or_load_memmap_splits(
        split_refs=split_refs,
        cfg_data=cfg_data,
        cfg_labels=cfg_labels,
        encoder=encoder,
        task_names=task_names,
        task_to_index=task_to_index,
    )

    datasets = {
        split_name: DynBCETransitionDataset(split_infos[split_name].manifest_path)
        for split_name in ["train", "val", "test"]
    }
    datasets["eval"] = datasets["val"]
    metadata = {
        "task_names": task_names,
        "split_summary": split_summary,
        "num_trajectories": {
            split_name: int(split_infos[split_name].num_trajectories)
            for split_name in ["train", "val", "test"]
        },
        "num_transitions": {
            split_name: int(split_infos[split_name].num_transitions)
            for split_name in ["train", "val", "test"]
        },
        "latent_cache_dir": to_absolute_path(str(_cfg_get(cfg_data, "cache_dir"))),
        "memmap_root": memmap_root,
        "transition_horizon": transition_horizon,
    }
    metadata["num_trajectories"]["eval"] = metadata["num_trajectories"]["val"]
    metadata["num_transitions"]["eval"] = metadata["num_transitions"]["val"]
    return datasets, task_to_index, metadata
