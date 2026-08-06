from __future__ import annotations

import hashlib
import random
import shutil
from pathlib import Path
from typing import Mapping, Sequence

import h5py
import numpy as np
import torch

from robosuite.pipeline.src.environment.intervention import sparse_success_reward
from robosuite.pipeline.src.environment.observations import (
    center_crop_resize_batch,
    normalize_policy_observation,
)
from robosuite.pipeline.src.environment.robosuite import (
    build_robosuite_env,
    build_runtime_config,
)

from .transitions import Transition


DEMO_SPLITS = ("expert", "success_rollout", "fail_rollout")


def resolve_demo_paths(
    data_root: str | Path,
    task_name: str,
    split: str,
    *,
    directory: str | Path | None = None,
) -> list[Path]:
    if split not in DEMO_SPLITS:
        raise ValueError(f"Unsupported AWR demo split '{split}'. Expected one of {DEMO_SPLITS}.")
    if directory is None:
        root = Path(data_root).expanduser().resolve() / str(task_name) / split
    else:
        root = Path(directory).expanduser().resolve()
    if not root.exists():
        return []
    return [
        path
        for path in sorted(root.iterdir())
        if path.is_file() and path.suffix.lower() in {".hdf5", ".h5", ".pt"}
    ]


def load_demos(
    *,
    data_root: str | Path,
    task_name: str,
    camera_names: Sequence[str],
    camera_aliases: Mapping[str, str],
    image_height: int,
    image_width: int,
    state_extractor,
    control_freq: int = 20,
    horizon: int = 500,
    trajectory_limits: Mapping[str, int | None] | None = None,
    split_directories: Mapping[str, str | Path] | None = None,
    cache_dir: str | Path | None = None,
    mirror_cache_dir: str | Path | None = None,
    seed: int = 0,
) -> dict[str, list[Transition]]:
    limits = dict(trajectory_limits or {})
    directories = dict(split_directories or {})
    return {
        split: _load_split(
            paths=resolve_demo_paths(
                data_root,
                task_name,
                split,
                directory=directories.get(split),
            ),
            split=split,
            camera_names=camera_names,
            camera_aliases=camera_aliases,
            image_height=image_height,
            image_width=image_width,
            state_extractor=state_extractor,
            control_freq=control_freq,
            horizon=horizon,
            trajectory_limit=limits.get(split),
            cache_dir=cache_dir,
            mirror_cache_dir=mirror_cache_dir,
            seed=seed,
        )
        for split in DEMO_SPLITS
    }


def load_hdf5_demos(
    path: str | Path,
    *,
    split: str,
    camera_names: Sequence[str],
    camera_aliases: Mapping[str, str],
    image_height: int,
    image_width: int,
    state_extractor,
    control_freq: int = 20,
    horizon: int = 500,
    demo_names: Sequence[str] | None = None,
) -> list[Transition]:
    transitions: list[Transition] = []
    with h5py.File(Path(path), "r") as file_handle:
        env_metadata = _read_env_metadata(file_handle)
        demos = _demo_group(file_handle)
        names = sorted(str(name) for name in demos.keys()) if demo_names is None else list(demo_names)
        source_cameras = tuple(sorted({camera_aliases.get(name, name) for name in camera_names}))
        runtime_config = build_runtime_config(
            env_metadata,
            camera_names=source_cameras,
            image_height=image_height,
            image_width=image_width,
            control_freq=control_freq,
            horizon=horizon,
            interactive=False,
        )
        env = build_robosuite_env(runtime_config)
        try:
            env.reset()
            for demo_name in names:
                if demo_name in demos:
                    transitions.extend(
                        _replay_demo(
                            env,
                            demos[demo_name],
                            path=Path(path),
                            demo_name=demo_name,
                            split=split,
                            camera_names=camera_names,
                            camera_aliases=camera_aliases,
                            image_height=image_height,
                            image_width=image_width,
                            state_extractor=state_extractor,
                        )
                    )
        finally:
            env.close()
    return transitions


def list_demo_names(path: str | Path) -> list[str]:
    with h5py.File(Path(path), "r") as file_handle:
        return sorted(str(name) for name in _demo_group(file_handle).keys())


def _load_split(
    *,
    paths: Sequence[Path],
    split: str,
    camera_names: Sequence[str],
    camera_aliases: Mapping[str, str],
    image_height: int,
    image_width: int,
    state_extractor,
    control_freq: int,
    horizon: int,
    trajectory_limit: int | None,
    cache_dir: str | Path | None,
    mirror_cache_dir: str | Path | None,
    seed: int,
) -> list[Transition]:
    if trajectory_limit is not None and int(trajectory_limit) <= 0:
        raise ValueError(f"Trajectory limit for {split} must be positive.")
    remaining = None if trajectory_limit is None else int(trajectory_limit)
    rng = random.Random(int(seed))
    transitions: list[Transition] = []
    for path in paths:
        if remaining == 0:
            break
        if path.suffix.lower() == ".pt":
            if remaining is not None:
                raise ValueError("Trajectory limits are only supported for HDF5 demos.")
            shard = torch.load(path, map_location="cpu", weights_only=False)
            transitions.extend(shard["transitions"] if isinstance(shard, dict) else shard)
            continue

        names = list_demo_names(path)
        if remaining is not None and len(names) > remaining:
            names = rng.sample(names, remaining)
        if not names:
            continue
        cache_path = _cache_path(cache_dir, path, split, names)
        if cache_path is not None and cache_path.exists() and cache_path.stat().st_mtime >= path.stat().st_mtime:
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            converted = cached["transitions"] if isinstance(cached, dict) else cached
        else:
            converted = load_hdf5_demos(
                path,
                split=split,
                camera_names=camera_names,
                camera_aliases=camera_aliases,
                image_height=image_height,
                image_width=image_width,
                state_extractor=state_extractor,
                control_freq=control_freq,
                horizon=horizon,
                demo_names=names,
            )
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache_path.with_suffix(".tmp")
                torch.save({"format": "awr_demo_v1", "transitions": converted}, temporary)
                temporary.replace(cache_path)
        transitions.extend(converted)
        if cache_path is not None:
            _mirror(cache_path, mirror_cache_dir)
        if remaining is not None:
            remaining = max(0, remaining - len(names))
    return transitions


def _cache_path(
    cache_dir: str | Path | None,
    source: Path,
    split: str,
    demo_names: Sequence[str],
) -> Path | None:
    if cache_dir is None:
        return None
    digest = hashlib.sha1("\0".join(demo_names).encode("utf-8")).hexdigest()[:10]
    return Path(cache_dir) / f"{split}_{source.stem}_{digest}.pt"


def _mirror(source: Path, mirror_cache_dir: str | Path | None) -> None:
    if mirror_cache_dir is None:
        return
    destination = Path(mirror_cache_dir) / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() or destination.stat().st_mtime < source.stat().st_mtime:
        shutil.copy2(source, destination)


def _demo_group(file_handle: h5py.File | h5py.Group) -> h5py.Group:
    if "demos" in file_handle:
        return file_handle["demos"]
    if "data" in file_handle:
        return file_handle["data"]
    raise KeyError("HDF5 demo file must contain a 'demos' or 'data' group.")


def _replay_demo(
    env,
    demo: h5py.Group,
    *,
    path: Path,
    demo_name: str,
    split: str,
    camera_names: Sequence[str],
    camera_aliases: Mapping[str, str],
    image_height: int,
    image_width: int,
    state_extractor,
) -> list[Transition]:
    actions = np.asarray(demo["actions"], dtype=np.float32)
    states = np.asarray(demo["states"])
    if len(actions) == 0 or len(states) == 0:
        return []
    if len(states) < len(actions):
        raise ValueError(f"{path}:{demo_name} has fewer states than actions.")

    labels = (
        np.asarray(demo["intervention_labels"], dtype=np.bool_)
        if "intervention_labels" in demo
        else np.zeros(len(actions), dtype=np.bool_)
    )
    observation_group = demo["observations"]
    source_cameras = {camera_aliases.get(name, name) for name in camera_names}
    missing = source_cameras.difference(observation_group.keys())
    if missing:
        raise KeyError(f"{path}:{demo_name} is missing cameras {sorted(missing)}.")
    images = {
        name: center_crop_resize_batch(
            np.asarray(observation_group[name]["images"], dtype=np.uint8),
            image_height,
            image_width,
        )
        for name in source_cameras
    }
    frames = [
        normalize_policy_observation(
            {
                **{name: value[index] for name, value in images.items()},
                "state": state_extractor.extract(states[index]).astype(np.float32),
            },
            camera_names=camera_names,
            camera_aliases=camera_aliases,
        )
        for index in range(len(actions))
    ]

    model_xml = demo.attrs.get("model_file")
    if isinstance(model_xml, bytes):
        model_xml = model_xml.decode("utf-8")
    if model_xml:
        _reset_from_xml(env, str(model_xml))
    else:
        env.reset()

    successful = bool(demo.attrs.get("successful", split != "fail_rollout"))
    transitions: list[Transition] = []
    for index, action in enumerate(actions):
        env.done = False
        env.timestep = index
        env.cur_time = float(index) * float(env.control_timestep)
        env.sim.set_state_from_flattened(states[index])
        env.sim.forward()
        result = env.step(action)
        if len(result) == 5:
            _, _, env_done, truncated, info = result
            env_done = bool(env_done or truncated)
        else:
            _, _, env_done, info = result
        reward, success = sparse_success_reward(env, info)
        last = index == len(actions) - 1
        done = bool(env_done or success or (successful and last))
        info_payload = dict(info) if isinstance(info, dict) else {"raw_info": info}
        info_payload.update(
            {
                "success": success,
                "demo_name": demo_name,
                "split": split,
                "reward_convention": "sparse_success_-1_0",
            }
        )
        transitions.append(
            Transition(
                obs=frames[index],
                action=np.asarray(action, dtype=np.float32),
                reward=float(reward),
                next_obs=frames[min(index + 1, len(frames) - 1)],
                done=done,
                grasp_penalty=None,
                is_intervention=bool(labels[index]),
                info=info_payload,
                reward_source="env_success",
                demo_source=split,
            )
        )
        if done:
            break
    return transitions


def _read_env_metadata(file_handle: h5py.File) -> dict:
    import json

    value = file_handle.attrs.get("env_info")
    if value is None:
        raise KeyError("HDF5 demo file is missing root attribute 'env_info'.")
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(value) if isinstance(value, str) else dict(value)


def _reset_from_xml(env, model_xml: str) -> None:
    env.reset_from_xml_string(env.edit_model_xml(model_xml))
    env.sim.reset()
    env.sim.forward()
    env.done = False
    env.timestep = 0
    env.cur_time = 0.0
