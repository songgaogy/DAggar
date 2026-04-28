"""Robosuite trajectory discovery: walk fail-labeled HDF5 + success manifests."""

from __future__ import annotations

import glob
import hashlib
import json
import os
from typing import Optional

import h5py
import numpy as np

from .trajectory import RobosuiteBenchmarkTrajectory


FAIL_OUT_HDF5_NAMES = ("out.hdf5", "out_2.hdf5")
SUCCESS_MANIFEST_NAME = "video_manifest.jsonl"
DEFAULT_CACHE_CAMERA_NAMES = ("agentview", "birdview", "frontview")


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


def _action_fingerprint(actions: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(actions, dtype=np.float32))
    h = hashlib.sha1()
    h.update(str(tuple(arr.shape)).encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()


def _infer_data_type(file_path: str) -> Optional[str]:
    path = str(file_path)
    if "/expert/" in path:
        return "expert"
    if "/success_rollout/" in path:
        return "success"
    if "/fail_rollout/" in path:
        return "fail"
    return None


def _metadata_scalar(data: np.lib.npyio.NpzFile, key: str, default: str = "") -> str:
    if key not in data:
        return default
    value = data[key]
    if value.shape == ():
        return str(value.item())
    return str(value.tolist())


def _build_metadata_type_lookup(metadata_cache_root: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    if not metadata_cache_root or not os.path.isdir(metadata_cache_root):
        return out

    for dir_name in sorted(os.listdir(metadata_cache_root)):
        task_dir = os.path.join(metadata_cache_root, dir_name)
        if not os.path.isdir(task_dir):
            continue
        benchmark_task = _rollout_subdir_to_benchmark_task_key(dir_name)
        for npz_path in sorted(glob.glob(os.path.join(task_dir, "*.npz"))):
            try:
                with np.load(npz_path, allow_pickle=True) as data:
                    if "actions" not in data or "file_path" not in data:
                        continue
                    data_type = _infer_data_type(_metadata_scalar(data, "file_path"))
                    if data_type is None:
                        continue
                    fp = _action_fingerprint(data["actions"])
                    out.setdefault(str(dir_name), {})[fp] = data_type
                    out.setdefault(str(benchmark_task), {})[fp] = data_type
            except Exception:
                continue
    return out


def _cache_camera_names(
    num_views: int,
    cache_camera_names: Optional[tuple[str, ...]],
) -> tuple[str, ...]:
    names = tuple(cache_camera_names or DEFAULT_CACHE_CAMERA_NAMES)
    if len(names) >= int(num_views):
        return names[: int(num_views)]
    extra = tuple(f"view_{i}" for i in range(len(names), int(num_views)))
    return names + extra


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


def _discover_cached_success(
    success_cache_root: str,
    metadata_cache_root: str,
    task_dir_allow: Optional[set[str]],
    max_success_per_task: Optional[int],
    cache_camera_names: Optional[tuple[str, ...]],
) -> list[RobosuiteBenchmarkTrajectory]:
    out: list[RobosuiteBenchmarkTrajectory] = []
    if not os.path.isdir(success_cache_root):
        return out
    type_lookup = _build_metadata_type_lookup(metadata_cache_root)
    if not type_lookup:
        print(
            f"[benchmark] no metadata type lookup built from {metadata_cache_root}; "
            "cannot discover cached success trajectories"
        )
        return out

    for dir_name in sorted(os.listdir(success_cache_root)):
        task_dir = os.path.join(success_cache_root, dir_name)
        if not os.path.isdir(task_dir):
            continue
        benchmark_task = _rollout_subdir_to_benchmark_task_key(dir_name)
        if task_dir_allow is not None and benchmark_task not in task_dir_allow:
            continue

        remaining = None if max_success_per_task is None else int(max_success_per_task)
        skipped_unknown = 0
        skipped_non_success = 0
        for cache_path in sorted(glob.glob(os.path.join(task_dir, "*.npz"))):
            if remaining is not None and remaining <= 0:
                break
            try:
                with np.load(cache_path, allow_pickle=False) as data:
                    images = data["images_chw"]
                    proprio = data["proprio"]
                    actions = data["actions"]
                    fp = _action_fingerprint(actions)
                    data_type = (
                        type_lookup.get(str(dir_name), {}).get(fp)
                        or type_lookup.get(str(benchmark_task), {}).get(fp)
                    )
                    if data_type is None:
                        skipped_unknown += 1
                        continue
                    if data_type != "success":
                        skipped_non_success += 1
                        continue
                    num_frames = int(
                        min(
                            int(images.shape[0]),
                            int(proprio.shape[0]),
                            int(actions.shape[0]),
                        )
                    )
                    cams = _cache_camera_names(int(images.shape[1]), cache_camera_names)
            except Exception as exc:
                print(f"[benchmark] skip cached success {cache_path}: {exc}")
                continue
            if num_frames <= 0:
                continue
            out.append(
                RobosuiteBenchmarkTrajectory(
                    task_name=benchmark_task,
                    num_frames=num_frames,
                    is_failure=False,
                    video_id=os.path.splitext(os.path.basename(cache_path))[0],
                    file_path=cache_path,
                    demo_path="",
                    available_cameras=cams,
                    failure_segments=[],
                    source_hdf5_path="",
                    source_demo_key="",
                    cache_npz_path=cache_path,
                )
            )
            if remaining is not None:
                remaining -= 1

        if skipped_unknown > 0:
            print(
                f"[benchmark] cached success discovery task={dir_name}: "
                f"skipped_unknown_type={skipped_unknown}"
            )
        if skipped_non_success > 0:
            print(
                f"[benchmark] cached success discovery task={dir_name}: "
                f"skipped_non_success={skipped_non_success}"
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
    success_cache_root: Optional[str] = None,
    metadata_cache_root: Optional[str] = None,
    cache_camera_names: Optional[tuple[str, ...]] = None,
) -> list[RobosuiteBenchmarkTrajectory]:
    """Enumerate available robosuite benchmark trajectories without loading frames."""
    task_dir_allow = _task_filter_to_benchmark_dir_names(tasks)
    failed = _discover_fail_labeled(fail_labeled_root, task_dir_allow, max_fail_per_task)
    if success_cache_root:
        success = _discover_cached_success(
            success_cache_root=str(success_cache_root),
            metadata_cache_root=str(metadata_cache_root or ""),
            task_dir_allow=task_dir_allow,
            max_success_per_task=max_success_per_task,
            cache_camera_names=cache_camera_names,
        )
    else:
        success = _discover_success(success_root, task_dir_allow, max_success_per_task)
    return failed + success
