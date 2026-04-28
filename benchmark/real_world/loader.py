"""Agilex trajectory discovery.

Failure side: walks ``<fail_labeled_root>/<task>/out.hdf5`` and iterates
``episodes/<split>/<episode_N>`` groups (split is typically ``fail_rollout``).

Success side: walks ``<success_root>/<task>/success_rollout/episode_*.hdf5``
directly (raw episode files; no manifest).
"""

from __future__ import annotations

import glob
import json
import os
from typing import Optional

import h5py
import numpy as np

from .trajectory import AgilexBenchmarkTrajectory


SUCCESS_SPLIT_DIR = "success_rollout"
FAIL_OUT_HDF5_NAME = "out.hdf5"


def _filter_tasks(all_tasks: list[str], allow: Optional[list[str]]) -> list[str]:
    if allow is None:
        return all_tasks
    allow_set = {str(t).strip() for t in allow}
    return [t for t in all_tasks if t in allow_set]


def _list_image_cameras(g: h5py.Group) -> tuple[str, ...]:
    if "observations" in g and "images" in g["observations"]:
        return tuple(sorted(g["observations"]["images"].keys()))
    return tuple()


def _scalar_str(data: np.lib.npyio.NpzFile, key: str, default: str = "") -> str:
    if key not in data:
        return default
    value = data[key]
    if value.shape == ():
        return str(value.item())
    return str(value.tolist())


def _load_cache_metadata(cache_path: str) -> dict:
    if os.path.isdir(cache_path):
        with open(os.path.join(cache_path, "metadata.json"), "r") as fp:
            return json.load(fp)
    with np.load(cache_path, allow_pickle=False) as data:
        return {
            "task_name": _scalar_str(data, "task_name", os.path.basename(os.path.dirname(cache_path))),
            "video_id": _scalar_str(data, "video_id", os.path.basename(cache_path)),
            "is_failure": bool(int(data["is_failure"].item())),
            "file_path": _scalar_str(data, "file_path", ""),
            "episode_path": _scalar_str(data, "episode_path", ""),
            "source_hdf5_path": _scalar_str(data, "source_hdf5_path", ""),
            "source_demo_key": _scalar_str(data, "source_demo_key", ""),
            "failure_segments": json.loads(_scalar_str(data, "failure_segments_json", "[]")),
            "camera_names": [str(x) for x in data["camera_names"].tolist()],
        }


def _cache_array_shape(cache_path: str, name: str) -> tuple[int, ...]:
    if os.path.isdir(cache_path):
        arr = np.load(os.path.join(cache_path, f"{name}.npy"), mmap_mode="r")
        return tuple(arr.shape)
    with np.load(cache_path, allow_pickle=False) as data:
        return tuple(data[name].shape)


def _probe_group(
    g: h5py.Group,
    *,
    is_failure: bool,
    proprio_field: str,
) -> tuple[int, tuple[str, ...]]:
    if "action" not in g:
        raise KeyError("missing action")
    if "observations" not in g:
        raise KeyError("missing observations")
    if proprio_field not in g["observations"]:
        raise KeyError(f"missing observations/{proprio_field}")
    if is_failure:
        if "annotations" not in g:
            raise KeyError("missing annotations")
        if "failure_frame_mask" not in g["annotations"]:
            raise KeyError("missing annotations/failure_frame_mask")
        if "failure_segment_index" not in g["annotations"]:
            raise KeyError("missing annotations/failure_segment_index")

    lengths = [
        int(g["action"].shape[0]),
        int(g["observations"][proprio_field].shape[0]),
    ]
    if is_failure:
        lengths.extend(
            [
                int(g["annotations"]["failure_frame_mask"].shape[0]),
                int(g["annotations"]["failure_segment_index"].shape[0]),
            ]
        )
    cams = _list_image_cameras(g)
    for cam in cams:
        lengths.append(int(g["observations"]["images"][cam].shape[0]))
    num_frames = min(lengths) if lengths else 0
    if num_frames <= 0:
        raise ValueError("empty trajectory")
    return int(num_frames), cams


def _discover_failures(
    fail_labeled_root: str,
    tasks_allow: Optional[list[str]],
    max_fail_per_task: Optional[int],
    *,
    proprio_field: str,
    proprio_slice: slice,
    action_slice: slice,
) -> list[AgilexBenchmarkTrajectory]:
    out: list[AgilexBenchmarkTrajectory] = []
    if not os.path.isdir(fail_labeled_root):
        return out
    task_dirs = sorted(
        d for d in os.listdir(fail_labeled_root)
        if os.path.isdir(os.path.join(fail_labeled_root, d))
    )
    task_dirs = _filter_tasks(task_dirs, tasks_allow)

    for task in task_dirs:
        out_path = os.path.join(fail_labeled_root, task, FAIL_OUT_HDF5_NAME)
        if not os.path.isfile(out_path):
            continue
        with h5py.File(out_path, "r") as f:
            if "episodes" not in f:
                continue
            remaining = None if max_fail_per_task is None else int(max_fail_per_task)
            for split in sorted(f["episodes"].keys()):
                ep_keys = sorted(f[f"episodes/{split}"].keys())
                for ep_key in ep_keys:
                    if remaining is not None and remaining <= 0:
                        break
                    g = f[f"episodes/{split}/{ep_key}"]
                    try:
                        num_frames, cams = _probe_group(
                            g,
                            is_failure=True,
                            proprio_field=proprio_field,
                        )
                    except Exception as exc:
                        print(f"[real_world] skip failure {out_path}:{split}/{ep_key}: {exc}")
                        continue
                    segments_raw = g.attrs.get("failure_segments_json", "[]")
                    try:
                        segments = json.loads(segments_raw)
                    except Exception:
                        segments = []
                    out.append(
                        AgilexBenchmarkTrajectory(
                            task_name=task,
                            num_frames=num_frames,
                            is_failure=True,
                            video_id=str(g.attrs.get("video_id", f"{task}/{split}/{ep_key}")),
                            available_cameras=cams,
                            failure_segments=list(segments),
                            source_hdf5_path=str(g.attrs.get("source_hdf5_path", "")),
                            source_demo_key=ep_key,
                            file_path=out_path,
                            episode_path=f"episodes/{split}/{ep_key}",
                            proprio_field=proprio_field,
                            proprio_slice=proprio_slice,
                            action_slice=action_slice,
                        )
                    )
                    if remaining is not None:
                        remaining -= 1
                if remaining is not None and remaining <= 0:
                    break
    return out


def _discover_successes(
    success_root: str,
    tasks_allow: Optional[list[str]],
    max_success_per_task: Optional[int],
    *,
    proprio_field: str,
    proprio_slice: slice,
    action_slice: slice,
) -> list[AgilexBenchmarkTrajectory]:
    out: list[AgilexBenchmarkTrajectory] = []
    if not os.path.isdir(success_root):
        return out
    task_dirs = sorted(
        d for d in os.listdir(success_root)
        if os.path.isdir(os.path.join(success_root, d, SUCCESS_SPLIT_DIR))
    )
    task_dirs = _filter_tasks(task_dirs, tasks_allow)

    for task in task_dirs:
        success_dir = os.path.join(success_root, task, SUCCESS_SPLIT_DIR)
        episode_files = sorted(glob.glob(os.path.join(success_dir, "episode_*.hdf5")))
        if max_success_per_task is not None:
            episode_files = episode_files[: int(max_success_per_task)]
        for ep_path in episode_files:
            try:
                with h5py.File(ep_path, "r") as f:
                    num_frames, cams = _probe_group(
                        f,
                        is_failure=False,
                        proprio_field=proprio_field,
                    )
            except Exception as exc:
                print(f"[real_world] skip success {ep_path}: {exc}")
                continue
            stem = os.path.splitext(os.path.basename(ep_path))[0]
            out.append(
                AgilexBenchmarkTrajectory(
                    task_name=task,
                    num_frames=num_frames,
                    is_failure=False,
                    video_id=f"{task}/success_rollout/{stem}",
                    available_cameras=cams,
                    failure_segments=[],
                    source_hdf5_path=ep_path,
                    source_demo_key=stem,
                    file_path=ep_path,
                    episode_path="",
                    proprio_field=proprio_field,
                    proprio_slice=proprio_slice,
                    action_slice=action_slice,
                )
            )
    return out


def discover_cached_agilex_trajectories(
    cache_root: str,
    tasks: Optional[list[str]] = None,
    max_fail_per_task: Optional[int] = None,
    max_success_per_task: Optional[int] = None,
) -> list[AgilexBenchmarkTrajectory]:
    """Enumerate trajectories from raw real-world cache `.npz` files."""
    out: list[AgilexBenchmarkTrajectory] = []
    if not os.path.isdir(cache_root):
        return out
    task_dirs = sorted(
        d for d in os.listdir(cache_root)
        if os.path.isdir(os.path.join(cache_root, d))
    )
    task_dirs = _filter_tasks(task_dirs, tasks)
    counts: dict[tuple[str, bool], int] = {}

    for task in task_dirs:
        cache_paths = sorted(glob.glob(os.path.join(cache_root, task, "*.npz")))
        cache_paths.extend(
            os.path.join(cache_root, task, d)
            for d in sorted(os.listdir(os.path.join(cache_root, task)))
            if os.path.isdir(os.path.join(cache_root, task, d))
            and os.path.isfile(os.path.join(cache_root, task, d, "metadata.json"))
        )
        for cache_path in cache_paths:
            try:
                meta = _load_cache_metadata(cache_path)
                is_failure = bool(meta.get("is_failure", False))
                limit = max_fail_per_task if is_failure else max_success_per_task
                count_key = (task, is_failure)
                if limit is not None and counts.get(count_key, 0) >= int(limit):
                    continue

                lengths = [
                    int(_cache_array_shape(cache_path, "actions")[0]),
                    int(_cache_array_shape(cache_path, "proprio")[0]),
                    int(_cache_array_shape(cache_path, "failure_mask")[0]),
                ]
                images_shape = _cache_array_shape(cache_path, "images_chw")
                if images_shape:
                    lengths.append(int(images_shape[0]))
                num_frames = int(min(lengths))
                if num_frames <= 0:
                    raise ValueError("empty cached trajectory")

                camera_names = tuple(str(x) for x in meta.get("camera_names", []))
                segments = list(meta.get("failure_segments", []))
                out.append(
                    AgilexBenchmarkTrajectory(
                        task_name=str(meta.get("task_name", task)),
                        num_frames=num_frames,
                        is_failure=is_failure,
                        video_id=str(meta.get("video_id", os.path.basename(cache_path))),
                        available_cameras=camera_names,
                        failure_segments=segments if is_failure else [],
                        source_hdf5_path=str(meta.get("source_hdf5_path", "")),
                        source_demo_key=str(meta.get("source_demo_key", "")),
                        file_path=str(meta.get("file_path", "")),
                        episode_path=str(meta.get("episode_path", "")),
                        cache_npz_path=cache_path,
                    )
                )
                counts[count_key] = counts.get(count_key, 0) + 1
            except Exception as exc:
                print(f"[real_world] skip cached trajectory {cache_path}: {exc}")
                continue
    return out


def discover_agilex_trajectories(
    fail_labeled_root: str,
    success_root: str,
    tasks: Optional[list[str]] = None,
    max_fail_per_task: Optional[int] = None,
    max_success_per_task: Optional[int] = None,
    *,
    proprio_field: str = "qpos",
    proprio_slice: slice = slice(7, 14),
    action_slice: slice = slice(7, 14),
) -> list[AgilexBenchmarkTrajectory]:
    """Enumerate agilex benchmark trajectories without loading frames."""
    failed = _discover_failures(
        fail_labeled_root,
        tasks_allow=tasks,
        max_fail_per_task=max_fail_per_task,
        proprio_field=proprio_field,
        proprio_slice=proprio_slice,
        action_slice=action_slice,
    )
    success = _discover_successes(
        success_root,
        tasks_allow=tasks,
        max_success_per_task=max_success_per_task,
        proprio_field=proprio_field,
        proprio_slice=proprio_slice,
        action_slice=action_slice,
    )
    return failed + success
