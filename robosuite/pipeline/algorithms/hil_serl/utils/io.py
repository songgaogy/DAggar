from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import torch

from ..common.types import Transition
from ..common.utils import clone_array_tree


def serialize_transition(transition: Transition) -> dict[str, Any]:
    return {
        "obs": clone_array_tree(transition.obs),
        "action": clone_array_tree(transition.action),
        "reward": None if transition.reward is None else float(transition.reward),
        "next_obs": clone_array_tree(transition.next_obs),
        "done": bool(transition.done),
        "grasp_penalty": None if transition.grasp_penalty is None else float(transition.grasp_penalty),
        "is_intervention": bool(transition.is_intervention),
        "info": dict(transition.info) if transition.info is not None else None,
        "reward_source": transition.reward_source,
        "demo_source": transition.demo_source,
    }


def deserialize_transition(payload: dict[str, Any]) -> Transition:
    return Transition(
        obs=clone_array_tree(payload["obs"]),
        action=clone_array_tree(payload["action"]),
        reward=payload["reward"],
        next_obs=clone_array_tree(payload["next_obs"]),
        done=bool(payload["done"]),
        grasp_penalty=payload.get("grasp_penalty"),
        is_intervention=bool(payload.get("is_intervention", False)),
        info=dict(payload["info"]) if payload.get("info") is not None else None,
        reward_source=payload.get("reward_source"),
        demo_source=payload.get("demo_source"),
    )


def save_transition_shard(path: str | Path, transitions: Sequence[Transition]) -> None:
    payload = [serialize_transition(transition) for transition in transitions]
    torch.save(payload, Path(path))


def load_transition_shard(path: str | Path) -> list[Transition]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    return [deserialize_transition(item) for item in payload]


def list_hdf5_demo_names(path: str | Path) -> list[str]:
    with h5py.File(Path(path), "r") as file_handle:
        demos_group = _get_hdf5_demo_group(file_handle)
        return sorted(str(name) for name in demos_group.keys())


def resolve_task_demo_paths(
    task_name: str,
    *,
    data_root: str | Path = "./data",
    split: str = "expert",
) -> list[Path]:
    demo_dir = Path(data_root) / str(task_name) / str(split)
    if not demo_dir.exists():
        return []

    demo_paths = [
        path
        for path in sorted(demo_dir.iterdir())
        if path.is_file() and path.suffix.lower() in {".pt", ".hdf5", ".h5"}
    ]
    return demo_paths


def load_demo_paths(
    demo_paths: Iterable[str | Path],
    *,
    cache_dir: str | Path | None = None,
    hdf5_loader=None,
    max_num_trajectories: int | None = None,
    cache_key: str | None = None,
) -> list[Transition]:
    transitions: list[Transition] = []
    remaining_trajectories = None if max_num_trajectories is None else int(max_num_trajectories)
    if remaining_trajectories is not None and remaining_trajectories <= 0:
        return transitions

    for raw_path in demo_paths:
        if remaining_trajectories is not None and remaining_trajectories <= 0:
            break

        path = Path(raw_path)
        suffix = path.suffix.lower()
        if suffix == ".pt":
            if remaining_trajectories is not None:
                raise ValueError(
                    "Trajectory-limited demo loading only supports HDF5 expert files. "
                    f"Remove max_num_trajectories or convert {path} to HDF5 input."
                )
            transitions.extend(load_transition_shard(path))
            continue

        if suffix in {".hdf5", ".h5"}:
            if hdf5_loader is None:
                raise ValueError("An hdf5_loader must be provided to read HDF5 demos.")
            selected_demo_names = list_hdf5_demo_names(path)
            if remaining_trajectories is not None:
                selected_demo_names = selected_demo_names[:remaining_trajectories]
            if not selected_demo_names:
                continue
            cache_path = None
            if cache_dir is not None:
                cache_root = Path(cache_dir)
                cache_root.mkdir(parents=True, exist_ok=True)
                cache_stem = path.stem
                if cache_key:
                    cache_stem = f"{cache_stem}_{cache_key}"
                if remaining_trajectories is not None:
                    cache_stem = f"{cache_stem}_first_{len(selected_demo_names):05d}"
                cache_path = cache_root / f"{cache_stem}.pt"
                if cache_path.exists() and cache_path.stat().st_mtime >= path.stat().st_mtime:
                    transitions.extend(load_transition_shard(cache_path))
                    if remaining_trajectories is not None:
                        remaining_trajectories -= len(selected_demo_names)
                    continue
            converted = hdf5_loader(path, demo_names=selected_demo_names)
            transitions.extend(converted)
            if cache_path is not None:
                save_transition_shard(cache_path, converted)
            if remaining_trajectories is not None:
                remaining_trajectories -= len(selected_demo_names)
            continue

        raise ValueError(f"Unsupported demo file type: {path}. Expected .pt, .hdf5, or .h5.")
    
    return transitions


def read_hdf5_env_info(path: str | Path) -> dict[str, Any]:
    with h5py.File(path, "r") as file_handle:
        env_info = file_handle.attrs.get("env_info", None)
        if env_info is None:
            raise KeyError(f"{path} does not contain the root attr 'env_info'.")
        if isinstance(env_info, bytes):
            env_info = env_info.decode("utf-8")
        return json.loads(str(env_info))


def read_hdf5_camera_names(path: str | Path) -> list[str]:
    with h5py.File(path, "r") as file_handle:
        camera_names = file_handle.attrs.get("camera_names", "[]")
        if isinstance(camera_names, bytes):
            camera_names = camera_names.decode("utf-8")
        if isinstance(camera_names, str):
            return [str(name) for name in json.loads(camera_names)]
        return [str(name) for name in camera_names]


def ensure_directory(path: str | Path) -> None:
    os.makedirs(Path(path), exist_ok=True)


def _get_hdf5_demo_group(file_handle: h5py.File | h5py.Group) -> h5py.Group:
    if "demos" in file_handle:
        return file_handle["demos"]
    if "data" in file_handle:
        return file_handle["data"]
    raise KeyError("HDF5 demo file must contain either a 'demos' group or a 'data' group.")
