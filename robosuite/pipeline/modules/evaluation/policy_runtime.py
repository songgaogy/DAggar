"""Shared, entrypoint-free helpers for DIPOLE policy evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

from robosuite.pipeline.algorithms.dipole.common import DipoleConfig, FlowAugmentationConfig
from robosuite.pipeline.algorithms.dipole.models import DipoleFlowPolicy
from robosuite.pipeline.utils import assert_disjoint_seed_ranges, resolve_requested_device


DEFAULT_OUTPUT_ROOT = "./outputs/DIPOLE/eval"
DEFAULT_VIDEO_CAMERA = "agentview"
DEFAULT_VIDEO_FPS = 20
DEFAULT_VIDEO_SIZE = 512
DEFAULT_EVAL_SEED = 900000


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off", ""}:
        return False
    raise ValueError(f"Unable to parse boolean value from {value!r}.")


def _resolve_run_dir(checkpoint_path: Path) -> Path | None:
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent
    return None


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_resolved_config(run_dir: Path | None):
    if run_dir is None:
        return None
    config_path = run_dir / "config_resolved.yaml"
    if not config_path.exists():
        return None
    return OmegaConf.load(config_path)


def _assert_eval_seeds_disjoint(
    run_info: dict[str, Any] | None,
    *,
    eval_base: int,
    eval_count: int,
) -> None:
    if run_info is None:
        print("[determinism][warn] no run_info.json found; skipping train/eval seed-disjoint check.")
        return
    train_base = run_info.get("env_reset_seed")
    if train_base is None:
        print(
            "[determinism][warn] run_info has no env_reset_seed; "
            "skipping train/eval seed-disjoint check."
        )
        return
    train_count = int(run_info.get("episode_index", 0) or 0)
    if train_count <= 0:
        print("[determinism][warn] run_info episode_index is 0/missing; skipping seed-disjoint check.")
        return
    assert_disjoint_seed_ranges(
        int(train_base), train_count, int(eval_base), int(eval_count), context="train vs eval layouts"
    )
    print(
        f"[determinism] seed-disjoint OK: train=[{train_base}, {int(train_base) + train_count}) "
        f"eval=[{eval_base}, {int(eval_base) + int(eval_count)})"
    )


def _resolve_init_checkpoint(
    *,
    requested: str | None,
    run_info: dict[str, Any] | None,
    resolved_cfg,
) -> Path | None:
    if requested:
        path = Path(to_absolute_path(str(requested)))
        return path if path.exists() else None

    if run_info is not None:
        initialized = run_info.get("initialized_checkpoint")
        if initialized:
            path = Path(str(initialized))
            if path.exists():
                return path

    if resolved_cfg is not None:
        init_checkpoint_cfg = getattr(resolved_cfg.runtime, "init_checkpoint", None)
        if init_checkpoint_cfg:
            path = Path(to_absolute_path(str(init_checkpoint_cfg)))
            if path.exists():
                return path
    return None


def _resolve_eval_device(payload: dict[str, Any], override: str | None) -> str:
    if override:
        return resolve_requested_device(override, fallback="cuda:0")
    flow_cfg = dict(payload.get("flow_config", {}))
    requested = flow_cfg.get("inference_device") or flow_cfg.get("device") or "cuda:0"
    return resolve_requested_device(requested, fallback="cuda:0")


def _build_dipole_policy(
    payload: dict[str, Any],
    *,
    task_name: str,
    device: str,
    omega: float,
) -> DipoleFlowPolicy:
    flow_cfg = dict(payload["flow_config"])
    aug_cfg = dict(flow_cfg.get("augmentation", {}) or {})
    config = DipoleConfig(
        action_dim=int(flow_cfg["action_dim"]),
        proprio_dim=int(flow_cfg["proprio_dim"]),
        action_horizon=int(flow_cfg.get("action_horizon", 8)),
        execute_horizon=int(flow_cfg.get("execute_horizon", 1)),
        image_size=int(flow_cfg.get("image_size", 128)),
        learning_rate=float(flow_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(flow_cfg.get("weight_decay", 1e-6)),
        grad_clip_norm=float(flow_cfg.get("grad_clip_norm", 1.0)),
        lambda_endpoint=float(flow_cfg.get("lambda_endpoint", 0.5)),
        lambda_smooth=float(flow_cfg.get("lambda_smooth", 0.05)),
        n_ode_steps=int(flow_cfg.get("n_ode_steps", 8)),
        device=str(device),
        inference_device=str(device),
        task_name=str(payload.get("task_name", task_name)),
        language_instruction=str(payload.get("language_instruction", task_name)),
        augmentation=FlowAugmentationConfig(
            minimal_shift_pad=int(aug_cfg.get("minimal_shift_pad", 2)),
            eye_in_hand_crop_scale=float(aug_cfg.get("eye_in_hand_crop_scale", 0.88)),
        ),
        beta=float(flow_cfg.get("beta", 2.0)),
        k=float(flow_cfg.get("k", 0.0)),
        guidance_omega=float(omega),
        g_clip=float(flow_cfg.get("g_clip", 10.0)),
    )
    policy = DipoleFlowPolicy(
        model_cfg=dict(payload["model_cfg"]),
        config=config,
        camera_names=[str(name) for name in payload["camera_names"]],
    )
    policy.load_dual_model_state(payload["core"])
    policy.set_language_instruction(str(payload.get("language_instruction", task_name)))
    policy.reset_action_chunk()
    return policy


def _reset_env(env) -> tuple[dict[str, Any], dict[str, Any]]:
    reset_output = env.reset()
    if isinstance(reset_output, tuple):
        return reset_output
    return reset_output, {}


def _capture_frame(
    env,
    *,
    video_camera: str,
    video_height: int,
    video_width: int,
) -> np.ndarray:
    frame = env.sim.render(
        height=int(video_height),
        width=int(video_width),
        camera_name=str(video_camera),
    )
    return np.ascontiguousarray(np.flipud(np.asarray(frame, dtype=np.uint8)))


def _write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    if len(frames) == 0:
        return
    with imageio.get_writer(
        path,
        format="FFMPEG",
        mode="I",
        fps=int(fps),
        codec="libx264",
        ffmpeg_params=["-movflags", "+faststart"],
        macro_block_size=1,
    ) as writer:
        for frame in frames:
            writer.append_data(frame)


__all__ = [
    "DEFAULT_EVAL_SEED",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_VIDEO_CAMERA",
    "DEFAULT_VIDEO_FPS",
    "DEFAULT_VIDEO_SIZE",
    "_assert_eval_seeds_disjoint",
    "_build_dipole_policy",
    "_capture_frame",
    "_load_json",
    "_load_resolved_config",
    "_parse_bool",
    "_reset_env",
    "_resolve_eval_device",
    "_resolve_init_checkpoint",
    "_resolve_run_dir",
    "_write_video",
]
