from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np

from robosuite.discriminator.dyn_bce.task_registry import normalize_task_name, resolve_checkpoint_task_name

from ..core.dataset import LatentTrajectory


@dataclass(frozen=True)
class SuboptimalDemoRef:
    task_name: str
    split: str
    file_path: str
    demo_key: str
    sub_start: int
    sub_stop: int


@dataclass(frozen=True)
class LabeledLatentTrajectory:
    trajectory: LatentTrajectory
    labels: np.ndarray
    split: str


def list_suboptimal_demo_refs(cfg: Any) -> tuple[list[SuboptimalDemoRef], dict[str, dict[str, int]]]:
    root_dir = os.path.abspath(str(cfg.suboptimal.root_dir))
    train_count = int(cfg.suboptimal.train_count_per_task)
    eval_count = int(cfg.suboptimal.eval_count_per_task)
    total_required = train_count + eval_count
    seed = int(cfg.suboptimal.seed)

    refs: list[SuboptimalDemoRef] = []
    split_summary: dict[str, dict[str, int]] = {}

    for task_offset, task_name_raw in enumerate(list(cfg.suboptimal.tasks)):
        task_name = normalize_task_name(str(task_name_raw))
        sub_dir = os.path.join(root_dir, f"{resolve_checkpoint_task_name(task_name)}_allview")
        if not os.path.isdir(sub_dir):
            raise FileNotFoundError(f"Missing suboptimal directory for {task_name}: {sub_dir}")

        task_refs: list[SuboptimalDemoRef] = []
        for file_path in sorted(glob.glob(os.path.join(sub_dir, "*.hdf5"))):
            with h5py.File(file_path, "r") as file_handle:
                if "demos" not in file_handle:
                    continue
                for demo_key in sorted(file_handle["demos"].keys()):
                    demo = file_handle["demos"][demo_key]
                    task_refs.append(
                        SuboptimalDemoRef(
                            task_name=task_name,
                            split="",
                            file_path=file_path,
                            demo_key=demo_key,
                            sub_start=int(np.asarray(demo["sub_start"])[()]),
                            sub_stop=int(np.asarray(demo["sub_stop"])[()]),
                        )
                    )

        if len(task_refs) < total_required:
            raise ValueError(
                f"Task {task_name} requires at least {total_required} suboptimal demos, found {len(task_refs)}."
            )

        rng = np.random.default_rng(seed + task_offset * 97)
        indices = np.arange(len(task_refs))
        rng.shuffle(indices)
        selected_refs = [task_refs[int(idx)] for idx in indices[:total_required]]

        split_summary[task_name] = {"train": 0, "eval": 0}
        for local_idx, ref in enumerate(selected_refs):
            split_name = "train" if local_idx < train_count else "eval"
            refs.append(
                SuboptimalDemoRef(
                    task_name=ref.task_name,
                    split=split_name,
                    file_path=ref.file_path,
                    demo_key=ref.demo_key,
                    sub_start=int(ref.sub_start),
                    sub_stop=int(ref.sub_stop),
                )
            )
            split_summary[task_name][split_name] += 1

    return refs, split_summary


def encode_suboptimal_refs(
    refs: list[SuboptimalDemoRef],
    *,
    encoder,
    task_to_index: dict[str, int],
    horizon: int,
    batch_size: int,
) -> tuple[list[LabeledLatentTrajectory], dict[str, Any]]:
    encoded: list[LabeledLatentTrajectory] = []
    dropped: list[dict[str, Any]] = []
    prepared_batch = []
    ref_batch: list[SuboptimalDemoRef] = []

    def flush_batch() -> None:
        nonlocal prepared_batch, ref_batch
        if not prepared_batch:
            return
        encoded_batch = encoder.encode_prepared_demos(prepared_batch)
        encoded_lookup = {
            (item.file_path, item.demo_key): item
            for item in encoded_batch
        }
        for ref in ref_batch:
            item = encoded_lookup.get((ref.file_path, ref.demo_key))
            if item is None:
                raise KeyError(f"Missing encoded suboptimal demo for {(ref.file_path, ref.demo_key)}")
            length = min(int(item.latents.shape[0]), int(item.actions.shape[0]))
            valid_len = length - int(horizon)
            if valid_len <= 0:
                dropped.append(
                    {
                        "task_name": ref.task_name,
                        "split": ref.split,
                        "file_path": ref.file_path,
                        "demo_key": ref.demo_key,
                        "reason": "too_short_for_horizon",
                    }
                )
                continue

            start = int(np.clip(ref.sub_start, 0, valid_len))
            stop = int(np.clip(ref.sub_stop, 0, valid_len))
            if stop <= start:
                dropped.append(
                    {
                        "task_name": ref.task_name,
                        "split": ref.split,
                        "file_path": ref.file_path,
                        "demo_key": ref.demo_key,
                        "reason": "empty_positive_interval_after_clip",
                    }
                )
                continue

            labels = np.zeros((valid_len,), dtype=np.int64)
            labels[start:stop] = 1
            encoded.append(
                LabeledLatentTrajectory(
                    trajectory=LatentTrajectory(
                        latents=np.asarray(item.latents, dtype=np.float32),
                        actions=np.asarray(item.actions, dtype=np.float32),
                        task_name=ref.task_name,
                        task_index=int(task_to_index[ref.task_name]),
                        data_type="suboptimal",
                        data_type_index=-1,
                        split=ref.split,
                        file_path=ref.file_path,
                        demo_key=ref.demo_key,
                    ),
                    labels=labels,
                    split=ref.split,
                )
            )
        prepared_batch = []
        ref_batch = []

    for idx, ref in enumerate(refs):
        prepared_batch.append(
            encoder.load_demo_raw(
                task_name=ref.task_name,
                file_path=ref.file_path,
                demo_key=ref.demo_key,
            )
        )
        ref_batch.append(ref)
        if len(prepared_batch) >= int(batch_size):
            flush_batch()
        if (idx + 1) % 20 == 0 or (idx + 1) == len(refs):
            print(f"[lpb_dice] encoded_suboptimal {idx + 1}/{len(refs)}")

    flush_batch()
    return encoded, {
        "num_requested": int(len(refs)),
        "num_encoded": int(len(encoded)),
        "num_dropped": int(len(dropped)),
        "dropped": dropped,
    }

