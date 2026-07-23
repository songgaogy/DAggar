from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from robosuite.pipeline.utils.runtime import resolve_path


def load_flow_checkpoint(reference: str | Path | None) -> tuple[Path | None, dict[str, Any] | None]:
    path = resolve_path(reference)
    if path is None:
        return None, None
    if not path.exists():
        raise FileNotFoundError(f"Flow initialization checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a checkpoint mapping, got {type(payload).__name__}.")
    return path, payload


def load_task_metadata(payload: dict[str, Any] | None, task_name: str) -> dict[str, Any] | None:
    if not payload:
        return None
    task_metadata = payload.get("task_metadata_map")
    if isinstance(task_metadata, dict):
        if task_name in task_metadata:
            return dict(task_metadata[task_name])
        if len(task_metadata) == 1:
            return dict(next(iter(task_metadata.values())))
    env_metadata = payload.get("env_metadata")
    return None if env_metadata is None else dict(env_metadata)


def flow_checkpoint_settings(payload: dict[str, Any], task_name: str) -> dict[str, Any]:
    action_mean = payload.get("act_mean")
    settings = {
        "model_cfg": payload.get("model_cfg"),
        "task_prompt_map": payload.get("task_prompt_map"),
        "task_metadata": load_task_metadata(payload, task_name),
        "action_mean": action_mean,
        "action_std": payload.get("act_std"),
        "proprio_mean": payload.get("prop_mean"),
        "proprio_std": payload.get("prop_std"),
    }
    if action_mean is not None:
        settings["action_horizon"] = int(action_mean.shape[0])
    metadata = settings["task_metadata"]
    camera_names = payload.get("camera_names")
    if camera_names is None and metadata is not None:
        camera_names = metadata.get("camera_names", [])
    settings["camera_names"] = [] if camera_names is None else list(camera_names)
    return settings
