from __future__ import annotations

import hashlib
import json
import pickle
import random
import shutil
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

import h5py
import numpy as np

from ..environment.robosuite import (
    RobosuiteObservationAdapter,
    build_robosuite_env,
    build_runtime_config_from_env_info,
    compute_grasp_penalty,
    reset_env_from_demo_xml,
    resolve_demo_images,
    sparse_success_reward,
    unpack_robosuite_step,
)
from .transitions import Transition, load_transition_shard, save_transition_shard


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
    return [
        path
        for path in sorted(demo_dir.iterdir())
        if path.is_file() and path.suffix.lower() in {".pt", ".hdf5", ".h5"}
    ]


def load_demo_paths(
    demo_paths: Iterable[str | Path],
    *,
    cache_dir: str | Path | None = None,
    mirror_cache_dir: str | Path | None = None,
    hdf5_loader: Callable[..., list[Transition]] | None = None,
    max_num_trajectories: int | None = None,
    cache_key: str | None = None,
    random_sample: bool = False,
    random_seed: int | None = None,
    selected_demo_callback: Callable[[Path, list[str]], None] | None = None,
) -> list[Transition]:
    transitions: list[Transition] = []
    paths = [Path(raw_path) for raw_path in demo_paths]
    selected_by_path: dict[Path, list[str]] | None = None
    if max_num_trajectories is not None:
        requested_trajectories = int(max_num_trajectories)
        if requested_trajectories < 1:
            raise ValueError(
                f"max_num_trajectories must be at least 1, got {requested_trajectories}."
            )
        unsupported_paths = [
            path for path in paths if path.suffix.lower() not in {".hdf5", ".h5"}
        ]
        if unsupported_paths:
            raise ValueError(
                "Trajectory-limited demo loading only supports HDF5 expert files. "
                f"Remove max_num_trajectories or convert {unsupported_paths[0]} to HDF5 input."
            )
        demo_catalog = [
            (path, demo_name)
            for path in paths
            for demo_name in list_hdf5_demo_names(path)
        ]
        if requested_trajectories > len(demo_catalog):
            raise ValueError(
                "Requested more expert trajectories than are available: "
                f"requested {requested_trajectories}, available {len(demo_catalog)}."
            )
        if random_sample:
            selected_catalog = random.Random(random_seed).sample(
                demo_catalog,
                requested_trajectories,
            )
        else:
            selected_catalog = demo_catalog[:requested_trajectories]
        selected_by_path = {path: [] for path in paths}
        for path, demo_name in selected_catalog:
            selected_by_path[path].append(demo_name)

    for path in paths:
        suffix = path.suffix.lower()
        if suffix == ".pt":
            transitions.extend(load_transition_shard(path))
            continue

        if suffix not in {".hdf5", ".h5"}:
            raise ValueError(f"Unsupported demo file type: {path}. Expected .pt, .hdf5, or .h5.")
        if hdf5_loader is None:
            raise ValueError("An hdf5_loader must be provided to read HDF5 demos.")

        selected_demo_names = (
            list_hdf5_demo_names(path)
            if selected_by_path is None
            else selected_by_path[path]
        )
        if not selected_demo_names:
            continue
        if selected_demo_callback is not None:
            selected_demo_callback(path, list(selected_demo_names))

        cache_path = _resolve_cache_path(
            path,
            cache_dir=cache_dir,
            cache_key=cache_key,
            selected_demo_names=(selected_demo_names if selected_by_path is not None else None),
        )
        if cache_path is not None and _cache_is_current(cache_path, path):
            try:
                cached_transitions = load_transition_shard(cache_path)
            except (
                EOFError,
                ModuleNotFoundError,
                OSError,
                RuntimeError,
                ValueError,
                pickle.UnpicklingError,
            ) as exc:
                print(f"[WARN] Ignoring invalid transition cache {cache_path}: {exc}")
                try:
                    cache_path.unlink()
                except FileNotFoundError:
                    pass
            else:
                _mirror_cache_file(cache_path, mirror_cache_dir)
                transitions.extend(cached_transitions)
                continue

        converted = hdf5_loader(path, demo_names=selected_demo_names)
        transitions.extend(converted)
        if cache_path is not None:
            save_transition_shard(cache_path, converted)
            _mirror_cache_file(cache_path, mirror_cache_dir)

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


def load_hdf5_demos_into_transitions(
    path: str | Path,
    *,
    camera_names: Sequence[str],
    img_height: int,
    img_width: int,
    proprio_keys: Sequence[str],
    renderer: str = "mjviewer",
    control_freq: int = 20,
    grasp_penalty: float = -0.02,
    grasp_command_threshold: float = 0.5,
    gripper_open_threshold: float = 0.9,
    gripper_closed_threshold: float = 0.1,
    demo_names: Optional[Sequence[str]] = None,
) -> list[Transition]:
    path = Path(path)
    with h5py.File(path, "r") as file_handle:
        env_info_raw = file_handle.attrs["env_info"]
        if isinstance(env_info_raw, bytes):
            env_info_raw = env_info_raw.decode("utf-8")
        env_info = json.loads(str(env_info_raw))
        demos_group = _get_hdf5_demo_group(file_handle)
        available_demo_names = {str(name) for name in demos_group.keys()}
        selected_demo_names = (
            sorted(available_demo_names)
            if demo_names is None
            else [str(name) for name in demo_names if str(name) in available_demo_names]
        )

    runtime_cfg = build_runtime_config_from_env_info(
        env_info=env_info,
        camera_names=camera_names,
        img_height=img_height,
        img_width=img_width,
        proprio_keys=proprio_keys,
        has_renderer=False,
        renderer=renderer,
        reward_shaping=False,
        control_freq=control_freq,
    )
    env = build_robosuite_env(runtime_cfg)
    adapter = RobosuiteObservationAdapter(
        env,
        camera_names=camera_names,
        img_height=img_height,
        img_width=img_width,
        proprio_keys=proprio_keys,
    )

    transitions: list[Transition] = []
    try:
        env.reset()
        with h5py.File(path, "r") as file_handle:
            demos_group = _get_hdf5_demo_group(file_handle)
            for demo_name in selected_demo_names:
                demo_group = demos_group[demo_name]
                states = np.asarray(demo_group["states"])
                actions = np.asarray(demo_group["actions"])
                model_xml = demo_group.attrs.get("model_file", None)
                if isinstance(model_xml, bytes):
                    model_xml = model_xml.decode("utf-8")
                intervention_labels = (
                    np.asarray(demo_group["intervention_labels"])
                    if "intervention_labels" in demo_group
                    else np.zeros(len(actions), dtype=np.bool_)
                )
                if len(states) == 0 or len(actions) == 0:
                    continue

                if model_xml:
                    reset_env_from_demo_xml(env, str(model_xml))
                else:
                    env.reset()

                demo_images = _load_demo_images(demo_group, camera_names)
                for step_idx in range(len(actions)):
                    env.done = False
                    env.timestep = int(step_idx)
                    env.cur_time = float(step_idx) * float(env.control_timestep)
                    env.sim.set_state_from_flattened(states[step_idx])
                    env.sim.forward()
                    raw_obs = env._get_observations(force_update=True)
                    grasp_penalty_value = compute_grasp_penalty(
                        env,
                        actions[step_idx],
                        penalty=float(grasp_penalty),
                        command_threshold=float(grasp_command_threshold),
                        open_threshold=float(gripper_open_threshold),
                        closed_threshold=float(gripper_closed_threshold),
                    )
                    obs = adapter.transform(
                        raw_obs,
                        images=resolve_demo_images(demo_images, adapter, step_idx),
                    )
                    next_raw_obs, _, terminated, truncated, info = unpack_robosuite_step(
                        env.step(actions[step_idx])
                    )
                    next_obs = adapter.transform(next_raw_obs)
                    reward, success = sparse_success_reward(env, info)
                    info_payload = dict(info) if isinstance(info, dict) else {"raw_info": info}
                    if grasp_penalty_value is not None:
                        info_payload.setdefault("grasp_penalty", float(grasp_penalty_value))
                    terminal = bool(terminated or success)
                    transitions.append(
                        Transition(
                            obs=obs,
                            action=np.asarray(actions[step_idx], dtype=np.float32),
                            reward=reward,
                            next_obs=next_obs,
                            done=terminal,
                            grasp_penalty=grasp_penalty_value,
                            is_intervention=bool(intervention_labels[step_idx]),
                            info=info_payload,
                            reward_source="env_success",
                            demo_source="offline_demo",
                            terminated=terminal,
                            truncated=bool(truncated and not terminal),
                        )
                    )
                    if terminal or truncated:
                        break
    finally:
        env.close()
    return transitions


def _resolve_cache_path(
    source_path: Path,
    *,
    cache_dir: str | Path | None,
    cache_key: str | None,
    selected_demo_names: Sequence[str] | None,
) -> Path | None:
    if cache_dir is None:
        return None
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_stem = source_path.stem
    if cache_key:
        cache_stem = f"{cache_stem}_{cache_key}"
    if selected_demo_names is not None:
        selection_payload = json.dumps(
            {
                "source": str(source_path.resolve()),
                "demo_names": sorted(selected_demo_names),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        selection_digest = hashlib.sha256(selection_payload).hexdigest()[:16]
        cache_stem = f"{cache_stem}_selected_{selection_digest}"
    return cache_root / f"{cache_stem}.pt"


def _cache_is_current(cache_path: Path, source_path: Path) -> bool:
    return cache_path.exists() and cache_path.stat().st_mtime >= source_path.stat().st_mtime


def _mirror_cache_file(source: str | Path, mirror_cache_dir: str | Path | None) -> None:
    if mirror_cache_dir is None:
        return
    source_path = Path(source)
    mirror_root = Path(mirror_cache_dir)
    mirror_root.mkdir(parents=True, exist_ok=True)
    destination = mirror_root / source_path.name
    if destination.exists() and destination.stat().st_mtime >= source_path.stat().st_mtime:
        return
    shutil.copy2(source_path, destination)


def _load_demo_images(
    demo_group: h5py.Group,
    camera_names: Sequence[str],
) -> dict[str, np.ndarray]:
    demo_images = {}
    if "observations" not in demo_group:
        return demo_images
    for camera_name in camera_names:
        if camera_name in demo_group["observations"]:
            images_dataset = demo_group["observations"][camera_name]["images"]
            demo_images[camera_name] = np.asarray(images_dataset)
    return demo_images


def _get_hdf5_demo_group(file_handle: h5py.File | h5py.Group) -> h5py.Group:
    if "demos" in file_handle:
        return file_handle["demos"]
    if "data" in file_handle:
        return file_handle["data"]
    raise KeyError("HDF5 demo file must contain either a 'demos' group or a 'data' group.")
