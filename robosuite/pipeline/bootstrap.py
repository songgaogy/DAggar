"""Shared construction helpers for AWR entry points."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from robosuite.pipeline.src.awr import (
    AWRAgent,
    AWRTrainer,
    load_qv_cache,
    save_qv_cache,
)
from robosuite.pipeline.utils import resolve_cuda_device


def build_agent_config(
    cfg: DictConfig,
    *,
    camera_names: list[str],
    flow_settings: dict[str, Any],
) -> dict[str, Any]:
    awr = OmegaConf.to_container(cfg.awr, resolve=True)
    if not isinstance(awr, dict):
        raise TypeError("awr must resolve to a mapping.")
    awr["device"] = str(resolve_cuda_device(str(cfg.runtime.learner_device)))
    awr["inference_device"] = str(resolve_cuda_device(str(cfg.runtime.inference_device)))
    awr["task_name"] = str(cfg.task.name)
    if bool(cfg.checkpoint.use_init_model):
        model_cfg = flow_settings.get("model_cfg")
        if model_cfg is None:
            raise KeyError("Flow initialization checkpoint is missing model_cfg.")
        model_cfg = copy.deepcopy(model_cfg)
        image_encoder = model_cfg.get("image_encoder", {})
        pretrained_path = image_encoder.get("pretrained_path")
        if pretrained_path is not None and not Path(str(pretrained_path)).exists():
            image_encoder["pretrained_path"] = None
        awr["model"] = model_cfg
        if flow_settings.get("task_prompt_map") is not None:
            awr["task_prompt_map"] = flow_settings["task_prompt_map"]
    if flow_settings.get("action_horizon") is not None:
        configured_horizon = int(awr["action_horizon"])
        inferred_horizon = int(flow_settings["action_horizon"])
        if int(awr["execute_horizon"]) == configured_horizon:
            awr["execute_horizon"] = inferred_horizon
        awr["action_horizon"] = inferred_horizon
    return {
        "camera_names": list(camera_names),
        "task_name": str(cfg.task.name),
        "encoder": {
            "encoder_type": "flow-multi",
            "image_keys": list(camera_names),
            "proprio_keys": ["state"],
            "image_size": int(cfg.env.img_height),
        },
        "awr": awr,
        "online_buffer": {
            "capacity": int(cfg.buffers.online_capacity),
            "batch_size": int(cfg.trainer.batch_size),
        },
        "demo_buffer": {
            "capacity": int(cfg.buffers.demo_capacity),
            "batch_size": int(cfg.trainer.batch_size),
        },
        "trainer": {
            "batch_size": int(cfg.trainer.batch_size),
            "warmup_steps": 0,
            "updates_per_episode": int(cfg.trainer.updates_per_episode),
            "inference_sync_interval": int(cfg.trainer.inference_sync_interval),
            "value_warmup_steps": int(cfg.trainer.value_warmup_steps),
        },
    }


def resolve_qv_cache_path(cfg: DictConfig) -> Path:
    if cfg.qv_cache.path is not None:
        path = Path(to_absolute_path(str(cfg.qv_cache.path)))
    else:
        path = (
            Path(to_absolute_path(str(cfg.logging.output_root)))
            / str(cfg.qv_cache.directory)
            / f"{cfg.task.name}.pt"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def build_qv_cache_metadata(
    cfg: DictConfig,
    *,
    camera_names: list[str],
    init_checkpoint: Path,
    agent: AWRAgent,
) -> dict[str, Any]:
    return {
        "format": "awr_qv_v1",
        "task": str(cfg.task.name),
        "demo_task": str(cfg.data.task_name),
        "reward": "sparse_success_-1_0",
        "init_checkpoint": str(init_checkpoint.resolve()),
        "value_warmup_steps": int(cfg.trainer.value_warmup_steps),
        "success_num_trajectories": int(cfg.data.success_num_trajectories),
        "fail_num_trajectories": int(cfg.data.fail_num_trajectories),
        "camera_names": list(camera_names),
        "image_size": int(cfg.env.img_height),
        "control_freq": int(cfg.env.control_freq),
        "action_horizon": int(agent.awr_config.action_horizon),
        "discount": float(agent.awr_config.discount),
        "expectile": float(agent.awr_config.expectile),
        "critic_hidden_dims": [int(value) for value in agent.awr_config.critic_hidden_dims],
    }


def try_load_qv_cache(
    cfg: DictConfig,
    agent: AWRAgent,
    trainer: AWRTrainer,
    *,
    path: Path,
    expected_metadata: dict[str, Any],
) -> bool:
    if not path.exists():
        return False
    try:
        load_qv_cache(
            path,
            agent=agent,
            trainer=trainer,
            expected_metadata=(
                expected_metadata if bool(cfg.qv_cache.strict_metadata) else None
            ),
            load_optimizers=True,
        )
    except (KeyError, RuntimeError, ValueError) as exc:
        print(f"[qv-cache] ignored {path}: {exc}")
        return False
    print(f"[qv-cache] loaded {path}")
    return True


def write_qv_cache(
    agent: AWRAgent,
    trainer: AWRTrainer,
    *,
    path: Path,
    metadata: dict[str, Any],
) -> None:
    save_qv_cache(path, agent=agent, trainer=trainer, metadata=metadata)
    print(f"[qv-cache] saved {path}")


__all__ = [
    "build_agent_config",
    "build_qv_cache_metadata",
    "resolve_qv_cache_path",
    "try_load_qv_cache",
    "write_qv_cache",
]
