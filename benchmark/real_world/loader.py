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
                    if "action" not in g or "annotations" not in g:
                        continue
                    num_frames = int(g["action"].shape[0])
                    segments_raw = g.attrs.get("failure_segments_json", "[]")
                    try:
                        segments = json.loads(segments_raw)
                    except Exception:
                        segments = []
                    cams = _list_image_cameras(g)
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
                    if "action" not in f:
                        continue
                    num_frames = int(f["action"].shape[0])
                    cams = _list_image_cameras(f)
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


def discover_agilex_trajectories(
    fail_labeled_root: str,
    success_root: str,
    tasks: Optional[list[str]] = None,
    max_fail_per_task: Optional[int] = None,
    max_success_per_task: Optional[int] = None,
    *,
    proprio_field: str = "qpos",
    proprio_slice: slice = slice(7, 14),
    action_slice: slice = slice(7, 13),
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
