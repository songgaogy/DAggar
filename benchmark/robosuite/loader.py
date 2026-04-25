"""Robosuite trajectory discovery: walk fail-labeled HDF5 + success manifests."""

from __future__ import annotations

import json
import os
from typing import Optional

import h5py

from .trajectory import RobosuiteBenchmarkTrajectory


FAIL_OUT_HDF5_NAMES = ("out.hdf5", "out_2.hdf5")
SUCCESS_MANIFEST_NAME = "video_manifest.jsonl"


def _task_filter_to_benchmark_dir_names(tasks: Optional[list[str]]) -> Optional[set[str]]:
    if tasks is None:
        return None
    from robosuite.discriminator.dyn_bce.task_registry import resolve_checkpoint_task_name

    return {resolve_checkpoint_task_name(str(name).strip()) for name in tasks}


def _rollout_subdir_to_benchmark_task_key(subdir: str) -> str:
    from robosuite.discriminator.dyn_bce.task_registry import TASK_ALIASES, resolve_checkpoint_task_name

    name = str(subdir).strip()
    if name in TASK_ALIASES:
        return resolve_checkpoint_task_name(name)
    return name


def _probe_source_demo(
    hdf5_path: str,
    demo_key: str,
) -> tuple[int, tuple[str, ...]]:
    with h5py.File(hdf5_path, "r") as f:
        root = "data" if "data" in f else "demos"
        if demo_key not in f[root]:
            raise KeyError(f"demo_key {demo_key!r} not in {hdf5_path}:{root}")
        g = f[root][demo_key]
        num_frames = int(g["actions"].shape[0])
        obs = g["observations"]
        cams = tuple(sorted(obs.keys()))
    return num_frames, cams


def _discover_fail_hdf5_paths(task_dir: str) -> list[str]:
    out_paths: list[str] = []
    for file_name in FAIL_OUT_HDF5_NAMES:
        path = os.path.join(task_dir, file_name)
        if os.path.isfile(path):
            out_paths.append(path)
    return out_paths


def _discover_fail_labeled(
    fail_labeled_root: str,
    task_dir_allow: Optional[set[str]],
    max_fail_per_task: Optional[int],
) -> list[RobosuiteBenchmarkTrajectory]:
    out: list[RobosuiteBenchmarkTrajectory] = []
    if not os.path.isdir(fail_labeled_root):
        return out
    task_dirs = sorted(os.listdir(fail_labeled_root))
    for dir_name in task_dirs:
        benchmark_task = _rollout_subdir_to_benchmark_task_key(dir_name)
        if task_dir_allow is not None and benchmark_task not in task_dir_allow:
            continue
        task_dir = os.path.join(fail_labeled_root, dir_name)
        out_paths = _discover_fail_hdf5_paths(task_dir)
        if not out_paths:
            continue
        remaining = None if max_fail_per_task is None else int(max_fail_per_task)
        for out_path in out_paths:
            if remaining is not None and remaining <= 0:
                break
            with h5py.File(out_path, "r") as f:
                if "demos" not in f:
                    continue
                demo_keys = list(f["demos"].keys())
                if remaining is not None:
                    demo_keys = demo_keys[:remaining]
                for dkey in demo_keys:
                    g = f["demos"][dkey]
                    num_frames = int(g["actions"].shape[0])
                    segments_raw = g.attrs.get("failure_segments_json", "[]")
                    try:
                        segments = json.loads(segments_raw)
                    except Exception:
                        segments = []
                    cams = tuple(sorted(g["observations"].keys()))
                    out.append(
                        RobosuiteBenchmarkTrajectory(
                            task_name=benchmark_task,
                            num_frames=num_frames,
                            is_failure=True,
                            video_id=str(g.attrs.get("video_id", dkey)),
                            file_path=out_path,
                            demo_path=f"demos/{dkey}",
                            available_cameras=cams,
                            failure_segments=list(segments),
                            source_hdf5_path=str(g.attrs.get("source_hdf5_path", "")),
                            source_demo_key=str(g.attrs.get("source_demo_key", "")),
                        )
                    )
                if remaining is not None:
                    remaining -= len(demo_keys)
    return out


def _discover_success(
    success_root: str,
    task_dir_allow: Optional[set[str]],
    max_success_per_task: Optional[int],
) -> list[RobosuiteBenchmarkTrajectory]:
    out: list[RobosuiteBenchmarkTrajectory] = []
    if not os.path.isdir(success_root):
        return out
    for dir_name in sorted(os.listdir(success_root)):
        benchmark_task = _rollout_subdir_to_benchmark_task_key(dir_name)
        if task_dir_allow is not None and benchmark_task not in task_dir_allow:
            continue
        manifest_path = os.path.join(success_root, dir_name, SUCCESS_MANIFEST_NAME)
        if not os.path.isfile(manifest_path):
            continue
        entries: list[dict] = []
        with open(manifest_path, "r") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                entries.append(json.loads(line))
        if max_success_per_task is not None:
            entries = entries[: int(max_success_per_task)]
        for entry in entries:
            src = entry["hdf5_path"]
            dkey = entry["demo_key"]
            if not os.path.isfile(src):
                continue
            try:
                num_frames, cams = _probe_source_demo(src, dkey)
            except Exception as exc:
                print(f"[benchmark] skip success {src}:{dkey}: {exc}")
                continue
            root_prefix = "demos" if _hdf5_has_root(src, "demos") else "data"
            out.append(
                RobosuiteBenchmarkTrajectory(
                    task_name=benchmark_task,
                    num_frames=num_frames,
                    is_failure=False,
                    video_id=str(entry.get("video_id", dkey)),
                    file_path=src,
                    demo_path=f"{root_prefix}/{dkey}",
                    fps=int(entry.get("fps", 20)),
                    available_cameras=cams,
                    failure_segments=[],
                    source_hdf5_path=src,
                    source_demo_key=dkey,
                )
            )
    return out


def _hdf5_has_root(path: str, root_name: str) -> bool:
    with h5py.File(path, "r") as f:
        return root_name in f


def discover_trajectories(
    fail_labeled_root: str,
    success_root: str,
    tasks: Optional[list[str]] = None,
    max_fail_per_task: Optional[int] = None,
    max_success_per_task: Optional[int] = None,
) -> list[RobosuiteBenchmarkTrajectory]:
    """Enumerate available robosuite benchmark trajectories without loading frames."""
    task_dir_allow = _task_filter_to_benchmark_dir_names(tasks)
    failed = _discover_fail_labeled(fail_labeled_root, task_dir_allow, max_fail_per_task)
    success = _discover_success(success_root, task_dir_allow, max_success_per_task)
    return failed + success
