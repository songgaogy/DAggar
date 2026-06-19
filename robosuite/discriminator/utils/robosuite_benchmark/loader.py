"""Discovery for the new robosuite discriminator benchmark layout.

Expected data layout:
    data/<task>/fail_rollout-labeled/*.hdf5
    data/<task>/fail_rollout-val-labeled/*.hdf5
    data/<task>/success_rollout-val/*.hdf5
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from .trajectory import RobosuiteBenchmarkTrajectory


DEFAULT_FAIL_SPLIT = "fail_rollout-val-labeled"
DEFAULT_SUCCESS_SPLIT = "success_rollout-val"
DEFAULT_BANK_SPLIT = "fail_rollout-labeled"

_TASK_ALIASES = {
    "PandaLift": "Lift",
    "Lift": "Lift",
    "PandaPickPlaceCan": "PickPlaceCan",
    "PickPlaceCan": "PickPlaceCan",
    "PandaStack": "Stack",
    "Stack": "Stack",
}


def canonical_task_name(name: str) -> str:
    """Return the short task name used in benchmark outputs."""
    s = str(name).strip()
    return _TASK_ALIASES.get(s, s)


def _task_filter(tasks: Optional[list[str]]) -> Optional[set[str]]:
    if tasks is None:
        return None
    return {canonical_task_name(t) for t in tasks}


def _iter_task_dirs(data_root: str) -> list[tuple[str, str]]:
    root = Path(data_root)
    if not root.is_dir():
        return []

    by_realpath: dict[str, tuple[str, str]] = {}
    for child in sorted(root.iterdir(), key=lambda p: p.name):
        if child.name.startswith(".") or child.name in {"utils", "agilex", "pretrained"}:
            continue
        if not child.is_dir():
            continue
        real = os.path.realpath(str(child))
        task_name = canonical_task_name(child.name)
        current = by_realpath.get(real)
        if current is None:
            by_realpath[real] = (task_name, str(child))
            continue
        current_path = Path(current[1])
        current_is_alias = current_path.name != canonical_task_name(current_path.name)
        child_is_short = child.name == task_name
        if current_is_alias and child_is_short:
            by_realpath[real] = (task_name, str(child))
    return sorted(by_realpath.values(), key=lambda item: item[0])


def _hdf5_root(handle: h5py.File) -> str:
    if "demos" in handle:
        return "demos"
    if "data" in handle:
        return "data"
    raise KeyError("HDF5 file has neither 'demos' nor 'data' root")


def _demo_length(g) -> int:
    if "length" in g.attrs:
        return int(g.attrs["length"])
    lengths = []
    if "states" in g:
        lengths.append(int(g["states"].shape[0]))
    if "actions" in g:
        lengths.append(int(g["actions"].shape[0]))
    if "observations" in g:
        for cam in g["observations"].keys():
            if "images" in g["observations"][cam]:
                lengths.append(int(g["observations"][cam]["images"].shape[0]))
    return int(min(lengths)) if lengths else 0


def _available_cameras(g) -> tuple[str, ...]:
    if "observations" not in g:
        return tuple()
    return tuple(sorted(k for k in g["observations"].keys() if "images" in g["observations"][k]))


def _segments_from_mask(mask: np.ndarray) -> list[dict]:
    m = np.asarray(mask, dtype=np.uint8).reshape(-1)
    if m.size == 0:
        return []
    pad = np.concatenate([[0], m, [0]])
    diff = np.diff(pad)
    starts = np.where(diff == 1)[0].tolist()
    ends = (np.where(diff == -1)[0] - 1).tolist()
    return [
        {"start": int(s), "end": int(e), "mode": ""}
        for s, e in zip(starts, ends)
    ]


def _failure_segments(g, mask: np.ndarray) -> list[dict]:
    raw = g.attrs.get("failure_segments_json", "[]")
    try:
        segments = json.loads(raw)
    except Exception:
        segments = []
    if segments:
        return list(segments)
    return _segments_from_mask(mask)


def _iter_hdf5_paths(task_dir: str, split: str) -> list[str]:
    return sorted(glob.glob(os.path.join(task_dir, split, "*.hdf5")))


def _discover_split(
    data_root: str,
    split: str,
    *,
    is_failure: bool,
    tasks: Optional[list[str]],
    max_per_task: Optional[int],
    require_success_counterpart: bool = False,
    success_split: str = DEFAULT_SUCCESS_SPLIT,
) -> list[RobosuiteBenchmarkTrajectory]:
    allowed = _task_filter(tasks)
    out: list[RobosuiteBenchmarkTrajectory] = []
    for task_name, task_dir in _iter_task_dirs(data_root):
        if allowed is not None and task_name not in allowed:
            continue
        if require_success_counterpart and not _iter_hdf5_paths(task_dir, success_split):
            continue
        hdf5_paths = _iter_hdf5_paths(task_dir, split)
        if not hdf5_paths:
            continue

        remaining = None if max_per_task is None else int(max_per_task)
        for hdf5_path in hdf5_paths:
            if remaining is not None and remaining <= 0:
                break
            file_stem = Path(hdf5_path).stem
            with h5py.File(hdf5_path, "r") as f:
                root = _hdf5_root(f)
                demo_keys = sorted(f[root].keys())
                if remaining is not None:
                    demo_keys = demo_keys[:remaining]
                for demo_key in demo_keys:
                    g = f[root][demo_key]
                    if "states" not in g or "actions" not in g or "observations" not in g:
                        continue
                    num_frames = _demo_length(g)
                    if num_frames <= 0:
                        continue
                    failure_segments: list[dict] = []
                    if is_failure:
                        if "annotations" not in g or "failure_frame_mask" not in g["annotations"]:
                            continue
                        mask = np.asarray(g["annotations"]["failure_frame_mask"][:], dtype=np.uint8)
                        if int(mask.shape[0]) != int(num_frames):
                            num_frames = min(int(num_frames), int(mask.shape[0]))
                        failure_segments = _failure_segments(g, mask)
                    out.append(
                        RobosuiteBenchmarkTrajectory(
                            task_name=task_name,
                            num_frames=int(num_frames),
                            is_failure=bool(is_failure),
                            video_id=f"{task_name}/{split}/{file_stem}/{demo_key}",
                            file_path=str(hdf5_path),
                            demo_path=f"{root}/{demo_key}",
                            split=str(split),
                            available_cameras=_available_cameras(g),
                            failure_segments=failure_segments,
                            source_hdf5_path=str(hdf5_path),
                            source_demo_key=str(demo_key),
                        )
                    )
                if remaining is not None:
                    remaining -= len(demo_keys)
    return out


def discover_trajectories(
    data_root: str = "data",
    tasks: Optional[list[str]] = None,
    fail_split: str = DEFAULT_FAIL_SPLIT,
    success_split: str = DEFAULT_SUCCESS_SPLIT,
    max_fail_per_task: Optional[int] = None,
    max_success_per_task: Optional[int] = None,
) -> list[RobosuiteBenchmarkTrajectory]:
    """Discover eval trajectories from val-labeled failure and val success splits."""
    failed = _discover_split(
        data_root=data_root,
        split=fail_split,
        is_failure=True,
        tasks=tasks,
        max_per_task=max_fail_per_task,
        require_success_counterpart=True,
        success_split=success_split,
    )
    eval_tasks = sorted({t.task_name for t in failed})
    success = _discover_split(
        data_root=data_root,
        split=success_split,
        is_failure=False,
        tasks=eval_tasks,
        max_per_task=max_success_per_task,
    )
    return failed + success


def discover_failure_bank(
    data_root: str = "data",
    tasks: Optional[list[str]] = None,
    split: str = DEFAULT_BANK_SPLIT,
    max_fail_per_task: Optional[int] = None,
) -> list[RobosuiteBenchmarkTrajectory]:
    """Discover GT-labeled failure trajectories for BCE bank training."""
    return _discover_split(
        data_root=data_root,
        split=split,
        is_failure=True,
        tasks=tasks,
        max_per_task=max_fail_per_task,
    )


def discover_success_training_dirs(
    data_root: str = "data",
    tasks: Optional[list[str]] = None,
    split: str = "success_rollout",
) -> tuple[list[str], list[str]]:
    """Return de-duplicated success rollout directories plus skipped task names."""
    allowed = _task_filter(tasks)
    dirs: list[str] = []
    skipped: list[str] = []
    for task_name, task_dir in _iter_task_dirs(data_root):
        if allowed is not None and task_name not in allowed:
            continue
        split_dir = os.path.join(task_dir, split)
        if os.path.isdir(split_dir) and glob.glob(os.path.join(split_dir, "*.hdf5")):
            dirs.append(split_dir)
        else:
            skipped.append(task_name)
    return dirs, skipped
