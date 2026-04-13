from __future__ import annotations

import glob
import json
import os
from typing import Any

import h5py
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from robosuite.discriminator.dyn_bce.task_registry import (
    normalize_task_name,
    ordered_task_names,
    resolve_checkpoint_task_name,
)
from robosuite.discriminator.lpb_score.core.dataset import (
    DemoRef,
    load_latent_trajectories,
    prepare_cached_trajectories,
)
from robosuite.discriminator.tsne.schemas import (
    AnnotatedFailureDemoRef,
    EXPERT_COLOR,
    FAILURE_END_COLOR,
    FAILURE_NORMAL_COLOR,
    FAILURE_START_COLOR,
    FailureSegment,
    OOD_COLOR,
    SUBOPTIMAL_COLOR,
    SUCCESS_COLOR,
    SuboptimalDemoRef,
    TrajectorySequence,
    _blend_hex,
)


def _list_hdf5_demos(data_dir: str) -> list[tuple[str, str]]:
    refs: list[tuple[str, str]] = []
    if not os.path.isdir(data_dir):
        return refs
    for file_path in sorted(glob.glob(os.path.join(data_dir, "*.hdf5"))):
        with h5py.File(file_path, "r") as file_handle:
            if "demos" not in file_handle:
                continue
            for demo_key in sorted(file_handle["demos"].keys()):
                refs.append((file_path, demo_key))
    return refs


def _sample_items(items: list[Any], max_items: int, seed: int) -> list[Any]:
    if int(max_items) <= 0 or len(items) <= int(max_items):
        return list(items)
    rng = np.random.default_rng(int(seed))
    indices = np.arange(len(items))
    rng.shuffle(indices)
    chosen = np.sort(indices[: int(max_items)])
    return [items[int(idx)] for idx in chosen.tolist()]


def _build_rollout_id(task_name: str, source_name: str, file_path: str, demo_key: str) -> str:
    stem = os.path.basename(file_path).replace(".hdf5", "")
    return f"{task_name}|{source_name}|{stem}|{demo_key}"


def _normalize_failure_mode(value: Any) -> str:
    if value is None:
        return "unlabeled"
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = str(value).strip()
    return text if text else "unlabeled"


def _sample_suboptimal_point_styles(
    sample_steps: np.ndarray,
    *,
    ood_start: int,
    ood_stop: int,
) -> tuple[list[str], list[str], np.ndarray]:
    groups = ["suboptimal_normal"] * int(sample_steps.shape[0])
    colors = [SUBOPTIMAL_COLOR] * int(sample_steps.shape[0])
    progress = np.full((int(sample_steps.shape[0]),), np.nan, dtype=np.float32)

    ood_mask = (sample_steps >= int(ood_start)) & (sample_steps < int(ood_stop))
    ood_indices = np.flatnonzero(ood_mask)
    if ood_indices.size == 0:
        return groups, colors, progress

    denom = max(1, int(ood_indices.size - 1))
    for local_rank, frame_index in enumerate(ood_indices.tolist()):
        blend_weight = float(local_rank) / float(denom)
        groups[int(frame_index)] = "suboptimal_ood"
        colors[int(frame_index)] = _blend_hex(OOD_COLOR, SUBOPTIMAL_COLOR, blend_weight)
        progress[int(frame_index)] = float(blend_weight)
    return groups, colors, progress


def _sample_failure_segment_point_styles(
    sample_steps: np.ndarray,
    *,
    failure_segments: tuple[FailureSegment, ...],
) -> tuple[list[str], list[str], np.ndarray, list[str]]:
    groups = ["fail_rollout_normal"] * int(sample_steps.shape[0])
    colors = [FAILURE_NORMAL_COLOR] * int(sample_steps.shape[0])
    progress = np.full((int(sample_steps.shape[0]),), np.nan, dtype=np.float32)
    modes = [""] * int(sample_steps.shape[0])

    for segment in failure_segments:
        segment_mask = (sample_steps >= int(segment.start)) & (sample_steps <= int(segment.end))
        segment_indices = np.flatnonzero(segment_mask)
        if segment_indices.size == 0:
            continue
        denom = max(1, int(segment_indices.size - 1))
        for local_rank, frame_index in enumerate(segment_indices.tolist()):
            blend_weight = float(local_rank) / float(denom)
            groups[int(frame_index)] = "fail_rollout_segment"
            colors[int(frame_index)] = _blend_hex(FAILURE_START_COLOR, FAILURE_END_COLOR, blend_weight)
            progress[int(frame_index)] = float(blend_weight)
            modes[int(frame_index)] = str(segment.mode)
    return groups, colors, progress, modes


def _fallback_cache_roots(cache_root: str) -> list[str]:
    fallback_roots: list[str] = []
    legacy_cache_root = to_absolute_path("./data/.lpb_new_cache")
    if os.path.isdir(legacy_cache_root) and os.path.abspath(cache_root) != os.path.abspath(legacy_cache_root):
        fallback_roots.append(legacy_cache_root)
    return fallback_roots


def _list_standard_demo_refs(
    cfg: DictConfig,
    *,
    task_names_override: list[str] | None = None,
    source_names_override: list[str] | None = None,
    max_rollouts_override: int | None = None,
) -> tuple[list[DemoRef], dict[str, dict[str, dict[str, int]]], list[str]]:
    task_names = ordered_task_names(
        list(cfg.data.tasks) if task_names_override is None else list(task_names_override)
    )
    source_names = (
        [str(name) for name in list(cfg.data.source_data_types)]
        if source_names_override is None
        else [str(name) for name in list(source_names_override)]
    )
    root_dir = to_absolute_path(str(cfg.data.root_dir))
    max_rollouts = (
        int(cfg.data.max_rollouts_per_task_source)
        if max_rollouts_override is None
        else int(max_rollouts_override)
    )
    sample_seed = int(cfg.data.sample_seed)

    refs: list[DemoRef] = []
    summary: dict[str, dict[str, dict[str, int]]] = {}
    for task_offset, task_name in enumerate(task_names):
        summary[task_name] = {}
        for source_offset, source_name in enumerate(source_names):
            data_dir = os.path.join(root_dir, task_name, source_name)
            raw_refs = [
                DemoRef(
                    task_name=task_name,
                    data_type=source_name,
                    split="analysis",
                    file_path=file_path,
                    demo_key=demo_key,
                )
                for file_path, demo_key in _list_hdf5_demos(data_dir)
            ]
            sampled_refs = _sample_items(
                raw_refs,
                max_items=max_rollouts,
                seed=sample_seed + task_offset * 131 + source_offset * 17,
            )
            refs.extend(sampled_refs)
            summary[task_name][source_name] = {
                "available": int(len(raw_refs)),
                "selected": int(len(sampled_refs)),
            }
    return refs, summary, task_names


def _sorted_task_names(task_names: list[str]) -> list[str]:
    ordered = ordered_task_names()
    task_set = {normalize_task_name(name) for name in task_names}
    known = [task_name for task_name in ordered if task_name in task_set]
    extras = [task_name for task_name in ordered_task_names(task_names) if task_name not in set(known)]
    return known + extras


def _infer_demo_length(demo) -> int:
    lengths: list[int] = []
    if "length" in demo.attrs:
        lengths.append(int(demo.attrs["length"]))
    if "states" in demo:
        lengths.append(int(demo["states"].shape[0]))
    if "actions" in demo:
        lengths.append(int(demo["actions"].shape[0]))
    if "annotations" in demo and "failure_frame_mask" in demo["annotations"]:
        lengths.append(int(demo["annotations"]["failure_frame_mask"].shape[0]))
    lengths = [length for length in lengths if int(length) > 0]
    if not lengths:
        raise ValueError("Unable to infer demo length from failure HDF5 demo.")
    return int(min(lengths))


def _normalize_inclusive_segment(start: Any, end: Any, *, length: int) -> tuple[int, int] | None:
    if int(length) <= 0:
        return None
    if start is None or end is None:
        return None
    seg_start = int(start)
    seg_end = int(end)
    if seg_start > seg_end:
        seg_start, seg_end = seg_end, seg_start
    seg_start = max(0, min(seg_start, int(length - 1)))
    seg_end = max(0, min(seg_end, int(length - 1)))
    if seg_end < seg_start:
        return None
    return int(seg_start), int(seg_end)


def _segments_from_failure_mask(mask: np.ndarray) -> tuple[FailureSegment, ...]:
    mask_bool = np.asarray(mask, dtype=np.uint8).reshape(-1) > 0
    if mask_bool.size == 0:
        return ()

    segments: list[FailureSegment] = []
    start_idx: int | None = None
    for idx, is_failure in enumerate(mask_bool.tolist()):
        if is_failure and start_idx is None:
            start_idx = int(idx)
        elif not is_failure and start_idx is not None:
            segments.append(FailureSegment(start=int(start_idx), end=int(idx - 1), mode="unlabeled"))
            start_idx = None
    if start_idx is not None:
        segments.append(FailureSegment(start=int(start_idx), end=int(mask_bool.size - 1), mode="unlabeled"))
    return tuple(segments)


def _coerce_1d_values(dataset) -> list[Any]:
    values = np.asarray(dataset)
    if values.ndim == 0:
        values = values.reshape(1)
    return values.tolist()


def _extract_failure_metadata_from_demo(
    demo,
    *,
    length: int,
) -> tuple[np.ndarray, tuple[FailureSegment, ...]]:
    failure_mask: np.ndarray | None = None
    if "annotations" in demo and "failure_frame_mask" in demo["annotations"]:
        failure_mask = np.asarray(demo["annotations"]["failure_frame_mask"][:], dtype=np.uint8).reshape(-1)
        if failure_mask.shape[0] < int(length):
            pad = int(length - failure_mask.shape[0])
            failure_mask = np.pad(failure_mask, (0, pad), constant_values=0)
        failure_mask = failure_mask[: int(length)]

    failure_segments: list[FailureSegment] = []
    if "failure_segments_json" in demo.attrs:
        raw_segments = json.loads(str(demo.attrs["failure_segments_json"]))
        if isinstance(raw_segments, dict):
            raw_segments = [raw_segments]
        for segment in raw_segments:
            if not isinstance(segment, dict):
                continue
            normalized = _normalize_inclusive_segment(
                segment.get("start", segment.get("failure_start_frame")),
                segment.get("end", segment.get("failure_end_frame")),
                length=int(length),
            )
            if normalized is not None:
                failure_segments.append(
                    FailureSegment(
                        start=int(normalized[0]),
                        end=int(normalized[1]),
                        mode=_normalize_failure_mode(segment.get("mode")),
                    )
                )
    elif "failure_start_frame" in demo and "failure_end_frame" in demo:
        start_values = _coerce_1d_values(demo["failure_start_frame"])
        end_values = _coerce_1d_values(demo["failure_end_frame"])
        mode_values = (
            _coerce_1d_values(demo["mode"])
            if "mode" in demo
            else ["unlabeled"] * int(min(len(start_values), len(end_values)))
        )
        num_values = min(len(start_values), len(end_values))
        for idx in range(num_values):
            normalized = _normalize_inclusive_segment(
                start_values[idx],
                end_values[idx],
                length=int(length),
            )
            if normalized is not None:
                mode_value = mode_values[idx] if idx < len(mode_values) else "unlabeled"
                failure_segments.append(
                    FailureSegment(
                        start=int(normalized[0]),
                        end=int(normalized[1]),
                        mode=_normalize_failure_mode(mode_value),
                    )
                )

    if failure_mask is None:
        failure_mask = np.zeros((int(length),), dtype=np.uint8)
        for segment in failure_segments:
            failure_mask[int(segment.start) : int(segment.end) + 1] = 1

    if not failure_segments:
        failure_segments = list(_segments_from_failure_mask(failure_mask))
    return failure_mask.astype(np.uint8, copy=False), tuple(failure_segments)


def _discover_annotated_failure_tasks(root_dir: str, hdf5_name: str) -> list[str]:
    discovered: list[str] = []
    if not os.path.isdir(root_dir):
        return discovered
    for entry in sorted(os.listdir(root_dir)):
        task_dir = os.path.join(root_dir, entry)
        if not os.path.isdir(task_dir):
            continue
        if not os.path.isfile(os.path.join(task_dir, hdf5_name)):
            continue
        discovered.append(normalize_task_name(entry))
    return _sorted_task_names(discovered)


def _resolve_annotated_failure_hdf5(root_dir: str, task_name: str, hdf5_name: str) -> tuple[str, str]:
    candidate_names: list[str] = []
    for candidate in [
        str(task_name),
        normalize_task_name(task_name),
        resolve_checkpoint_task_name(task_name),
    ]:
        if candidate not in candidate_names:
            candidate_names.append(candidate)
    for candidate in candidate_names:
        file_path = os.path.join(root_dir, candidate, hdf5_name)
        if os.path.isfile(file_path):
            return file_path, candidate
    raise FileNotFoundError(
        f"Missing annotated failure HDF5 for {task_name} under {root_dir}. Tried: {candidate_names}"
    )


def _list_annotated_failure_demo_refs(
    cfg: DictConfig,
) -> tuple[list[AnnotatedFailureDemoRef], dict[str, dict[str, Any]], list[str]]:
    if not bool(cfg.annotated_failures.enabled):
        return [], {}, []

    root_dir = to_absolute_path(str(cfg.annotated_failures.root_dir))
    hdf5_name = str(cfg.annotated_failures.hdf5_name)
    task_names = (
        ordered_task_names(list(cfg.annotated_failures.tasks))
        if len(list(cfg.annotated_failures.tasks)) > 0
        else _discover_annotated_failure_tasks(root_dir=root_dir, hdf5_name=hdf5_name)
    )
    max_rollouts = int(cfg.annotated_failures.max_rollouts_per_task)
    sample_seed = int(cfg.annotated_failures.sample_seed)

    refs: list[AnnotatedFailureDemoRef] = []
    summary: dict[str, dict[str, Any]] = {}
    for task_offset, task_name in enumerate(task_names):
        file_path, source_task_name = _resolve_annotated_failure_hdf5(
            root_dir=root_dir,
            task_name=task_name,
            hdf5_name=hdf5_name,
        )
        task_refs: list[AnnotatedFailureDemoRef] = []
        dropped_without_failure = 0
        with h5py.File(file_path, "r") as file_handle:
            if "demos" not in file_handle:
                raise KeyError(f"Expected demos group in {file_path}")
            for demo_key in sorted(file_handle["demos"].keys()):
                demo = file_handle["demos"][demo_key]
                demo_length = _infer_demo_length(demo)
                failure_mask, failure_segments = _extract_failure_metadata_from_demo(
                    demo,
                    length=demo_length,
                )
                if not np.any(failure_mask):
                    dropped_without_failure += 1
                    continue
                task_refs.append(
                    AnnotatedFailureDemoRef(
                        task_name=task_name,
                        file_path=file_path,
                        demo_key=demo_key,
                        failure_mask=failure_mask,
                        failure_segments=failure_segments,
                        source_task_name=source_task_name,
                    )
                )
        sampled_refs = _sample_items(
            task_refs,
            max_items=max_rollouts,
            seed=sample_seed + task_offset * 97,
        )
        refs.extend(sampled_refs)
        summary[task_name] = {
            "source_task_name": str(source_task_name),
            "hdf5_path": str(file_path),
            "available": int(len(task_refs)),
            "selected": int(len(sampled_refs)),
            "dropped_without_failure_frames": int(dropped_without_failure),
        }
    return refs, summary, task_names


def _labels_from_source_name(source_name: str, length: int) -> np.ndarray:
    labels = np.zeros((int(length),), dtype=np.int64)
    if str(source_name) == "fail_rollout":
        labels.fill(1)
    return labels


def _encode_standard_sequences(
    *,
    refs: list[DemoRef],
    encoder,
    task_to_index: dict[str, int],
    cfg: DictConfig,
) -> tuple[list[TrajectorySequence], dict[str, Any]]:
    if not refs:
        return [], {"num_requested": 0, "num_encoded": 0, "num_sequences": 0}

    cache_root = to_absolute_path(str(cfg.data.cache_dir))
    cached_refs = prepare_cached_trajectories(
        refs=refs,
        encoder=encoder,
        cache_root=cache_root,
        task_to_index=task_to_index,
        encode_demo_batch_size=int(cfg.data.encode_demo_batch_size),
        fallback_cache_roots=_fallback_cache_roots(cache_root),
        build_missing_cache=bool(cfg.data.build_missing_cache),
    )
    trajectories = load_latent_trajectories(cached_refs)

    sequences: list[TrajectorySequence] = []
    for trajectory in trajectories:
        length = min(int(trajectory.latents.shape[0]), int(trajectory.actions.shape[0]))
        if length <= 0:
            continue
        source_name = str(trajectory.data_type)
        sequences.append(
            TrajectorySequence(
                latents=np.asarray(trajectory.latents[:length], dtype=np.float32),
                failure_labels=_labels_from_source_name(source_name=source_name, length=length),
                task_name=str(trajectory.task_name),
                task_index=int(trajectory.task_index),
                source_name=source_name,
                split=str(trajectory.split),
                file_path=str(trajectory.file_path),
                demo_key=str(trajectory.demo_key),
                rollout_id=_build_rollout_id(
                    task_name=str(trajectory.task_name),
                    source_name=source_name,
                    file_path=str(trajectory.file_path),
                    demo_key=str(trajectory.demo_key),
                ),
                ood_start=None,
                ood_stop=None,
            )
        )

    return (
        sequences,
        {
            "num_requested": int(len(refs)),
            "num_cached_refs": int(len(cached_refs)),
            "num_sequences": int(len(sequences)),
        },
    )


def _encode_annotated_failure_sequences(
    *,
    refs: list[AnnotatedFailureDemoRef],
    encoder,
    task_to_index: dict[str, int],
    cfg: DictConfig,
) -> tuple[list[TrajectorySequence], dict[str, Any]]:
    if not refs:
        return [], {"num_requested": 0, "num_cached_refs": 0, "num_sequences": 0, "num_dropped": 0}

    demo_refs = [
        DemoRef(
            task_name=ref.task_name,
            data_type="fail_rollout",
            split="analysis",
            file_path=ref.file_path,
            demo_key=ref.demo_key,
        )
        for ref in refs
    ]
    ref_lookup = {
        (ref.file_path, ref.demo_key): ref
        for ref in refs
    }

    cache_root = to_absolute_path(str(cfg.data.cache_dir))
    cached_refs = prepare_cached_trajectories(
        refs=demo_refs,
        encoder=encoder,
        cache_root=cache_root,
        task_to_index=task_to_index,
        encode_demo_batch_size=int(cfg.data.encode_demo_batch_size),
        fallback_cache_roots=_fallback_cache_roots(cache_root),
        build_missing_cache=bool(cfg.data.build_missing_cache),
    )
    trajectories = load_latent_trajectories(cached_refs)

    sequences: list[TrajectorySequence] = []
    dropped: list[dict[str, Any]] = []
    for trajectory in trajectories:
        ref = ref_lookup.get((str(trajectory.file_path), str(trajectory.demo_key)))
        if ref is None:
            dropped.append(
                {
                    "task_name": str(trajectory.task_name),
                    "file_path": str(trajectory.file_path),
                    "demo_key": str(trajectory.demo_key),
                    "reason": "missing_failure_metadata",
                }
            )
            continue

        length = min(
            int(trajectory.latents.shape[0]),
            int(trajectory.actions.shape[0]),
            int(ref.failure_mask.shape[0]),
        )
        if length <= 0:
            dropped.append(
                {
                    "task_name": str(trajectory.task_name),
                    "file_path": str(trajectory.file_path),
                    "demo_key": str(trajectory.demo_key),
                    "reason": "empty_sequence",
                }
            )
            continue

        failure_mask = np.asarray(ref.failure_mask[:length], dtype=np.int64)
        failure_segments = tuple(
            FailureSegment(
                start=int(max(0, segment.start)),
                end=int(min(length - 1, segment.end)),
                mode=str(segment.mode),
            )
            for segment in ref.failure_segments
            if int(segment.start) < int(length) and int(segment.end) >= 0
        )
        failure_segments = tuple(
            segment
            for segment in failure_segments
            if int(segment.end) >= int(segment.start)
        )
        if not np.any(failure_mask):
            dropped.append(
                {
                    "task_name": str(trajectory.task_name),
                    "file_path": str(trajectory.file_path),
                    "demo_key": str(trajectory.demo_key),
                    "reason": "no_failure_frames_after_clip",
                    "length": int(length),
                }
            )
            continue
        if not failure_segments:
            failure_segments = _segments_from_failure_mask(failure_mask)

        sequences.append(
            TrajectorySequence(
                latents=np.asarray(trajectory.latents[:length], dtype=np.float32),
                failure_labels=failure_mask,
                task_name=str(trajectory.task_name),
                task_index=int(trajectory.task_index),
                source_name="fail_rollout",
                split=str(trajectory.split),
                file_path=str(trajectory.file_path),
                demo_key=str(trajectory.demo_key),
                rollout_id=_build_rollout_id(
                    task_name=str(trajectory.task_name),
                    source_name="fail_rollout",
                    file_path=str(trajectory.file_path),
                    demo_key=str(trajectory.demo_key),
                ),
                failure_segments=failure_segments,
            )
        )

    return (
        sequences,
        {
            "num_requested": int(len(refs)),
            "num_cached_refs": int(len(cached_refs)),
            "num_sequences": int(len(sequences)),
            "num_dropped": int(len(dropped)),
            "dropped": dropped,
        },
    )


def _list_suboptimal_demo_refs(cfg: DictConfig) -> tuple[list[SuboptimalDemoRef], dict[str, dict[str, int]]]:
    if not bool(cfg.suboptimal.enabled):
        return [], {}

    root_dir = to_absolute_path(str(cfg.suboptimal.root_dir))
    task_names = ordered_task_names(list(cfg.suboptimal.tasks))
    max_rollouts = int(cfg.suboptimal.max_rollouts_per_task)
    sample_seed = int(cfg.suboptimal.sample_seed)

    refs: list[SuboptimalDemoRef] = []
    summary: dict[str, dict[str, int]] = {}
    for task_offset, task_name in enumerate(task_names):
        checkpoint_name = resolve_checkpoint_task_name(task_name)
        sub_dir = os.path.join(root_dir, f"{checkpoint_name}_allview")
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
                            file_path=file_path,
                            demo_key=demo_key,
                            sub_start=int(np.asarray(demo["sub_start"])[()]),
                            sub_stop=int(np.asarray(demo["sub_stop"])[()]),
                        )
                    )
        sampled_refs = _sample_items(
            task_refs,
            max_items=max_rollouts,
            seed=sample_seed + task_offset * 97,
        )
        refs.extend(sampled_refs)
        summary[task_name] = {
            "available": int(len(task_refs)),
            "selected": int(len(sampled_refs)),
        }
    return refs, summary


def _encode_suboptimal_sequences(
    *,
    refs: list[SuboptimalDemoRef],
    encoder,
    task_to_index: dict[str, int],
    cfg: DictConfig,
) -> tuple[list[TrajectorySequence], dict[str, Any]]:
    if not refs:
        return [], {"num_requested": 0, "num_sequences": 0, "num_dropped": 0}

    cache_root = to_absolute_path(str(cfg.data.cache_dir))
    sequences: list[TrajectorySequence] = []
    dropped: list[dict[str, Any]] = []
    for idx, ref in enumerate(refs):
        encoded = encoder.load_or_encode_demo(
            cache_root=cache_root,
            task_name=ref.task_name,
            file_path=ref.file_path,
            demo_key=ref.demo_key,
        )
        length = min(int(encoded.latents.shape[0]), int(encoded.actions.shape[0]))
        if length <= 0:
            dropped.append(
                {
                    "task_name": ref.task_name,
                    "file_path": ref.file_path,
                    "demo_key": ref.demo_key,
                    "reason": "empty_sequence",
                }
            )
            continue

        start = int(np.clip(ref.sub_start, 0, length))
        stop = int(np.clip(ref.sub_stop, 0, length))
        if stop <= start:
            dropped.append(
                {
                    "task_name": ref.task_name,
                    "file_path": ref.file_path,
                    "demo_key": ref.demo_key,
                    "reason": "empty_positive_interval_after_clip",
                    "sub_start": int(ref.sub_start),
                    "sub_stop": int(ref.sub_stop),
                    "length": int(length),
                }
            )
            continue

        failure_labels = np.zeros((length,), dtype=np.int64)
        failure_labels[start:stop] = 1
        sequences.append(
            TrajectorySequence(
                latents=np.asarray(encoded.latents[:length], dtype=np.float32),
                failure_labels=failure_labels,
                task_name=str(ref.task_name),
                task_index=int(task_to_index[ref.task_name]),
                source_name="suboptimal",
                split="analysis",
                file_path=str(ref.file_path),
                demo_key=str(ref.demo_key),
                rollout_id=_build_rollout_id(
                    task_name=str(ref.task_name),
                    source_name="suboptimal",
                    file_path=str(ref.file_path),
                    demo_key=str(ref.demo_key),
                ),
                ood_start=int(start),
                ood_stop=int(stop),
            )
        )
        if (idx + 1) % 20 == 0 or (idx + 1) == len(refs):
            print(f"[tsne] encoded_suboptimal {idx + 1}/{len(refs)}")

    return (
        sequences,
        {
            "num_requested": int(len(refs)),
            "num_sequences": int(len(sequences)),
            "num_dropped": int(len(dropped)),
            "dropped": dropped,
        },
    )


def _sampled_length(sequence: TrajectorySequence, timestep_stride: int) -> int:
    return int(np.arange(0, int(sequence.latents.shape[0]), max(1, int(timestep_stride))).shape[0])


def _limit_sequences_by_total_points(
    sequences: list[TrajectorySequence],
    *,
    timestep_stride: int,
    max_total_points: int,
    seed: int,
) -> tuple[list[TrajectorySequence], dict[str, int]]:
    point_counts = [_sampled_length(sequence, timestep_stride) for sequence in sequences]
    input_points = int(sum(point_counts))
    if int(max_total_points) <= 0 or input_points <= int(max_total_points):
        return (
            list(sequences),
            {
                "num_input_sequences": int(len(sequences)),
                "num_output_sequences": int(len(sequences)),
                "input_points": input_points,
                "output_points": input_points,
                "max_total_points": int(max_total_points),
            },
        )

    protected_indices: list[int] = []
    protected_pairs: list[tuple[str, str]] = []
    for task_name in ordered_task_names([sequence.task_name for sequence in sequences]):
        protected_pairs.append((str(task_name), "expert"))
        protected_pairs.append((str(task_name), "suboptimal"))

    fallback_sources = ("success_rollout", "fail_rollout")
    for task_name, source_name in protected_pairs:
        protected_idx = next(
            (
                idx
                for idx, sequence in enumerate(sequences)
                if sequence.task_name == task_name and sequence.source_name == source_name
            ),
            None,
        )
        if protected_idx is not None and protected_idx not in protected_indices:
            protected_indices.append(int(protected_idx))
            continue

        fallback_idx = next(
            (
                idx
                for idx, sequence in enumerate(sequences)
                if sequence.task_name == task_name and sequence.source_name in fallback_sources
            ),
            None,
        )
        if fallback_idx is not None and fallback_idx not in protected_indices:
            protected_indices.append(int(fallback_idx))

    selected_indices = list(protected_indices)
    selected_points = int(sum(point_counts[idx] for idx in selected_indices))

    rng = np.random.default_rng(int(seed))
    remaining_indices = [idx for idx in range(len(sequences)) if idx not in set(protected_indices)]
    rng.shuffle(remaining_indices)
    for idx in remaining_indices:
        next_count = int(point_counts[idx])
        if selected_indices and selected_points + next_count > int(max_total_points):
            continue
        selected_indices.append(int(idx))
        selected_points += next_count

    if not selected_indices and sequences:
        selected_indices.append(0)
        selected_points = int(point_counts[0])

    selected_indices = sorted(set(selected_indices))
    limited_sequences = [sequences[idx] for idx in selected_indices]
    return (
        limited_sequences,
        {
            "num_input_sequences": int(len(sequences)),
            "num_output_sequences": int(len(limited_sequences)),
            "input_points": input_points,
            "output_points": int(selected_points),
            "max_total_points": int(max_total_points),
        },
    )


def _limit_annotated_failure_sequences(
    sequences: list[TrajectorySequence],
    *,
    timestep_stride: int,
    max_total_points: int,
    seed: int,
) -> tuple[list[TrajectorySequence], dict[str, Any]]:
    point_counts = [_sampled_length(sequence, timestep_stride) for sequence in sequences]
    input_points = int(sum(point_counts))
    if int(max_total_points) <= 0 or input_points <= int(max_total_points):
        return (
            list(sequences),
            {
                "num_input_sequences": int(len(sequences)),
                "num_output_sequences": int(len(sequences)),
                "input_points": input_points,
                "output_points": input_points,
                "max_total_points": int(max_total_points),
                "num_preserved_fail_rollouts": int(
                    sum(sequence.source_name == "fail_rollout" for sequence in sequences)
                ),
                "num_preserved_success_rollouts": int(
                    sum(sequence.source_name == "success_rollout" for sequence in sequences)
                ),
            },
        )

    fail_indices = [
        idx
        for idx, sequence in enumerate(sequences)
        if sequence.source_name == "fail_rollout"
    ]
    success_indices = [
        idx
        for idx, sequence in enumerate(sequences)
        if sequence.source_name == "success_rollout"
    ]

    selected_indices = list(fail_indices)
    selected_points = int(sum(point_counts[idx] for idx in selected_indices))

    protected_success_indices: list[int] = []
    for task_name in _sorted_task_names([sequence.task_name for sequence in sequences]):
        candidate_idx = next(
            (
                idx
                for idx in success_indices
                if sequences[idx].task_name == task_name
            ),
            None,
        )
        if candidate_idx is None or candidate_idx in selected_indices:
            continue
        if selected_points + int(point_counts[candidate_idx]) > int(max_total_points):
            continue
        protected_success_indices.append(int(candidate_idx))
        selected_indices.append(int(candidate_idx))
        selected_points += int(point_counts[candidate_idx])

    rng = np.random.default_rng(int(seed))
    remaining_success_indices = [idx for idx in success_indices if idx not in set(selected_indices)]
    rng.shuffle(remaining_success_indices)
    for idx in remaining_success_indices:
        next_count = int(point_counts[idx])
        if selected_points + next_count > int(max_total_points):
            continue
        selected_indices.append(int(idx))
        selected_points += next_count

    selected_indices = sorted(set(selected_indices))
    limited_sequences = [sequences[idx] for idx in selected_indices]
    return (
        limited_sequences,
        {
            "num_input_sequences": int(len(sequences)),
            "num_output_sequences": int(len(limited_sequences)),
            "input_points": input_points,
            "output_points": int(selected_points),
            "max_total_points": int(max_total_points),
            "num_preserved_fail_rollouts": int(
                sum(sequences[idx].source_name == "fail_rollout" for idx in selected_indices)
            ),
            "num_preserved_success_rollouts": int(
                sum(sequences[idx].source_name == "success_rollout" for idx in selected_indices)
            ),
            "protected_success_rollouts": int(len(protected_success_indices)),
            "exceeded_max_total_points": bool(selected_points > int(max_total_points)),
        },
    )


def _build_point_dataframe(
    sequences: list[TrajectorySequence],
    *,
    timestep_stride: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    features: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    stride = max(1, int(timestep_stride))

    for rollout_index, sequence in enumerate(sequences):
        sample_steps = np.arange(0, int(sequence.latents.shape[0]), stride, dtype=np.int64)
        if sample_steps.size == 0:
            continue
        features.append(np.asarray(sequence.latents[sample_steps], dtype=np.float32))
        sampled_labels = np.asarray(sequence.failure_labels[sample_steps], dtype=np.int64)
        if sequence.source_name == "expert":
            display_groups = ["expert"] * int(sample_steps.shape[0])
            display_colors = [EXPERT_COLOR] * int(sample_steps.shape[0])
            ood_progress = np.full((int(sample_steps.shape[0]),), np.nan, dtype=np.float32)
            sampled_failure_modes = [""] * int(sample_steps.shape[0])
        elif sequence.source_name == "success_rollout":
            display_groups = ["success_rollout"] * int(sample_steps.shape[0])
            display_colors = [SUCCESS_COLOR] * int(sample_steps.shape[0])
            ood_progress = np.full((int(sample_steps.shape[0]),), np.nan, dtype=np.float32)
            sampled_failure_modes = [""] * int(sample_steps.shape[0])
        elif sequence.source_name == "fail_rollout" and sequence.failure_segments:
            display_groups, display_colors, ood_progress, sampled_failure_modes = (
                _sample_failure_segment_point_styles(
                    sample_steps,
                    failure_segments=sequence.failure_segments,
                )
            )
        elif sequence.source_name == "suboptimal" and sequence.ood_start is not None and sequence.ood_stop is not None:
            display_groups, display_colors, ood_progress = _sample_suboptimal_point_styles(
                sample_steps,
                ood_start=int(sequence.ood_start),
                ood_stop=int(sequence.ood_stop),
            )
            sampled_failure_modes = [""] * int(sample_steps.shape[0])
        else:
            display_groups = [str(sequence.source_name)] * int(sample_steps.shape[0])
            display_colors = ["#7A8798"] * int(sample_steps.shape[0])
            ood_progress = np.full((int(sample_steps.shape[0]),), np.nan, dtype=np.float32)
            sampled_failure_modes = [""] * int(sample_steps.shape[0])

        for local_idx, timestep in enumerate(sample_steps.tolist()):
            failure_label = int(sampled_labels[local_idx])
            records.append(
                {
                    "rollout_index": int(rollout_index),
                    "rollout_id": str(sequence.rollout_id),
                    "task_name": str(sequence.task_name),
                    "task_index": int(sequence.task_index),
                    "timestep": int(timestep),
                    "source_name": str(sequence.source_name),
                    "split": str(sequence.split),
                    "failure_label": failure_label,
                    "success_label": int(1 - failure_label),
                    "outcome_name": "failure" if failure_label == 1 else "success",
                    "file_path": str(sequence.file_path),
                    "demo_key": str(sequence.demo_key),
                    "rollout_length": int(sequence.latents.shape[0]),
                    "display_group": str(display_groups[local_idx]),
                    "display_color": str(display_colors[local_idx]),
                    "failure_mode": str(sampled_failure_modes[local_idx]),
                    "ood_progress": float(ood_progress[local_idx])
                    if not np.isnan(ood_progress[local_idx])
                    else np.nan,
                    "ood_start": int(sequence.ood_start) if sequence.ood_start is not None else None,
                    "ood_stop": int(sequence.ood_stop) if sequence.ood_stop is not None else None,
                }
            )

    if not features:
        raise RuntimeError("No latent points remain after sampling.")

    feature_matrix = np.concatenate(features, axis=0).astype(np.float32, copy=False)
    frame = pd.DataFrame.from_records(records)
    return feature_matrix, frame


def _summary_counts(frame: pd.DataFrame) -> dict[str, Any]:
    counts = (
        frame.groupby(["task_name", "source_name", "outcome_name"], dropna=False)
        .size()
        .reset_index(name="num_points")
    )
    display_counts = (
        frame.groupby(["task_name", "display_group"], dropna=False)
        .size()
        .reset_index(name="num_points")
    )
    rollouts = (
        frame.groupby(["task_name", "source_name"], dropna=False)["rollout_id"]
        .nunique()
        .reset_index(name="num_rollouts")
    )
    mode_frame = frame[
        (frame["source_name"] == "fail_rollout")
        & (frame["failure_label"] == 1)
        & (frame["failure_mode"].astype(str) != "")
    ]
    mode_counts = (
        mode_frame.groupby(["task_name", "failure_mode"], dropna=False)
        .size()
        .reset_index(name="num_points")
    )
    mode_rollouts = (
        mode_frame.groupby(["failure_mode"], dropna=False)["rollout_id"]
        .nunique()
        .reset_index(name="num_rollouts")
    )
    return {
        "points_by_task_source_outcome": counts.to_dict(orient="records"),
        "points_by_task_display_group": display_counts.to_dict(orient="records"),
        "rollouts_by_task_source": rollouts.to_dict(orient="records"),
        "failure_points_by_task_mode": mode_counts.to_dict(orient="records"),
        "failure_rollouts_by_mode": mode_rollouts.to_dict(orient="records"),
    }
