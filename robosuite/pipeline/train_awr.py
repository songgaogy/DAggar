from __future__ import annotations

import builtins
import datetime
import os
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from robosuite.pipeline.algorithms.awr import AWRTrainer
from robosuite.pipeline.algorithms.awr.replay_buffer import apply_awr_reward_to_transition
from robosuite.pipeline.algorithms.awr.workers import AWRDiscriminatorWorker, AWRPolicyWorker
from robosuite.pipeline.factory import build_algorithm
from robosuite.pipeline.envs import (
    RobosuiteInterventionRuntime,
    RobosuiteViewerRuntime,
    build_device,
    build_robosuite_env,
    choose_viewer_backend,
    compute_grasp_penalty,
    load_hdf5_demos_into_transitions,
    make_checkpoint_directory,
    snapshot_env_state,
    sparse_success_reward,
)
from robosuite.pipeline.train_flow_dagger import (
    bind_flow_proprio_extractor,
    build_flow_runtime_cfg,
    convert_env_camera_observation,
    load_hdf5_demos_into_flow_transitions,
    load_init_checkpoint_payload,
    maybe_set_seed,
    reset_flow_policy_observation,
    resolve_flow_task_metadata,
    serialize_seed,
)
from robosuite.pipeline.utils import (
    AsyncCheckpointWriter,
    AsyncTransitionChunkWriter,
    ConsoleLogCapture,
    EMAFpsTracker,
    FixedRateLimiter,
    IntervalGate,
    JsonlEventLogger,
    checkpoint_path,
    checkpoint_step_path,
    format_episode_line,
    list_hdf5_demo_names,
    load_demo_paths,
    load_transition_chunks,
    maybe_build_wandb,
    maybe_log,
    maybe_wrap_visualization,
    now_readable,
    resolve_buffer_chunk_dirs,
    resolve_buffer_snapshot_paths,
    resolve_camera_names,
    resolve_checkpoint_reference,
    resolve_render_camera,
    resolve_requested_device,
    resolve_runtime_fps,
    write_resolved_config,
    write_run_info,
)
from robosuite.pipeline.utils.io import resolve_task_demo_paths
from robosuite.pipeline.utils.train_utils import resolve_demo_task_name

print = partial(builtins.print, flush=True)


AWR_DEMO_LOADER_SCHEMA_VERSION = 2


def _initial_discriminator_images(obs: dict[str, Any], camera_names: list[str]) -> dict[str, np.ndarray]:
    return {
        camera_name: np.asarray(obs[camera_name], dtype=np.uint8)
        for camera_name in camera_names
        if camera_name in obs
    }


def resolve_run_directory(cfg: DictConfig) -> tuple[str, Path]:
    output_root = Path(to_absolute_path(str(cfg.logging.output_root)))
    explicit_run_name = cfg.logging.run_name
    if explicit_run_name is not None:
        run_name = str(explicit_run_name)
        return run_name, make_checkpoint_directory(output_root, run_name)
    run_name = f"awr_{cfg.env.environment}_{now_readable()}"
    return run_name, make_checkpoint_directory(output_root, run_name)


def find_latest_resumable_run(output_root: Path, env_name: str) -> Path | None:
    env_prefix = f"awr_{env_name}_"
    candidates = []
    for latest_checkpoint in output_root.glob("*/checkpoints/latest.pt"):
        run_dir = latest_checkpoint.parent.parent
        if run_dir.name.startswith(env_prefix):
            candidates.append((latest_checkpoint.stat().st_mtime, run_dir))
    if not candidates:
        for latest_checkpoint in output_root.glob("*/checkpoints/latest.pt"):
            candidates.append((latest_checkpoint.stat().st_mtime, latest_checkpoint.parent.parent))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def resume_checkpoint_candidates(cfg: DictConfig, output_root: Path) -> list[Path]:
    candidates: list[Path] = []
    explicit_checkpoint = resolve_checkpoint_reference(cfg.runtime.checkpoint)
    if explicit_checkpoint is not None:
        candidates.append(explicit_checkpoint)
    elif bool(cfg.runtime.resume):
        resumable_run = find_latest_resumable_run(output_root, str(cfg.env.environment))
        if resumable_run is not None:
            latest = checkpoint_path(resumable_run, "latest")
            if latest.exists():
                candidates.append(latest)
            candidates.extend(sorted((resumable_run / "checkpoints").glob("step_*.pt"), reverse=True))

    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen or not candidate.exists():
            continue
        seen.add(resolved)
        deduped.append(candidate)
    return deduped


def resolve_algorithm_devices(algorithm_cfg: dict[str, Any]) -> tuple[str, str]:
    awr_cfg = algorithm_cfg.setdefault("awr", {})
    learner_requested = awr_cfg.get("device", "cpu")
    inference_requested = awr_cfg.get("inference_device", None)
    normalized_learner_request = "cpu" if learner_requested is None else str(learner_requested).strip().lower()

    default_learner = "cuda:0" if torch.cuda.is_available() else "cpu"
    learner_device = resolve_requested_device(learner_requested, fallback=default_learner)

    if inference_requested is None or str(inference_requested).lower() == "auto":
        if normalized_learner_request in {"cuda", "cuda:0"} and torch.cuda.device_count() >= 2:
            learner_device = "cuda:1"
            inference_device = "cuda:0"
        elif learner_device.startswith("cuda"):
            inference_device = "cpu"
        else:
            inference_device = learner_device
    else:
        explicit_inference_request = str(inference_requested).strip().lower()
        inference_fallback = "cpu" if learner_device.startswith("cuda") else learner_device
        inference_device = resolve_requested_device(inference_requested, fallback=inference_fallback)
        if explicit_inference_request.startswith("cuda") and inference_device == learner_device and learner_device.startswith("cuda"):
            inference_device = "cpu"

    awr_cfg["device"] = learner_device
    awr_cfg["inference_device"] = inference_device
    return learner_device, inference_device


def _supports_ansi_color() -> bool:
    return sys.stdout.isatty() and str(os.environ.get("TERM", "")).lower() not in {"", "dumb"}


def _colorize(text: str, color_code: str) -> str:
    if not _supports_ansi_color():
        return text
    return f"{color_code}{text}\033[0m"


def format_runtime_line(
    *,
    step: int,
    episode_index: int,
    overall_fps: float,
    learner_progress: dict[str, int],
    pending_updates: int,
    online_valid_sequences: int,
    online_ready_steps: int,
) -> str:
    return (
        f"[runtime] "
        f"step={step:7d} | ep={episode_index:4d} | fps={overall_fps:5.1f} | "
        f"v={learner_progress['value_updates']:6d} | a={learner_progress['actor_updates']:6d} | "
        f"publish_in={learner_progress['updates_until_publish']:4d} | "
        f"online_valid={online_valid_sequences:5d} | ready={online_ready_steps:5d} | "
        f"pending={pending_updates:4d}"
    )


def format_train_line(step: int, metrics: dict[str, float], pending_updates: int) -> str:
    return (
        f"[train]   "
        f"step={step:7d} | "
        f"actor={metrics.get('actor_loss', float('nan')):8.4f} | "
        f"q={metrics.get('q_loss', float('nan')):8.4f} | "
        f"v={metrics.get('value_loss', float('nan')):8.4f} | "
        f"adv={metrics.get('mean_advantage', float('nan')):7.4f} | "
        f"w={metrics.get('mean_weight', float('nan')):7.4f} | "
        f"publish_in={int(metrics.get('learner_updates_until_publish', 0.0)):4d} | "
        f"pending={pending_updates:4d}"
    )


def format_publish_line(metrics: dict[str, float]) -> str:
    return (
        f"[publish] "
        f"sync={int(metrics.get('learner_publish_count', 0.0)):3d} | "
        f"actor_update={int(metrics.get('learner_last_published_actor_update', 0.0)):7d}"
    )


def format_discriminator_line(step: int, episode_index: int, labeled_episode_step: int, decision) -> str:
    is_fail = int(decision.prediction) == 1
    status = "FAIL" if is_fail else "NORMAL"
    colored_status = _colorize(
        f"{status:<6}",
        "\033[31;1m" if is_fail else "\033[32;1m",
    )
    return (
        f"[disc]    "
        f"step={step:7d} | ep={episode_index:4d} | label_step={labeled_episode_step:4d} | "
        f"state={colored_status} | lambda={float(decision.score):7.4f} | "
        f"thr={float(decision.threshold):7.4f}"
    )


def _sanitize_cache_stem(value: str) -> str:
    sanitized = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(value).strip())
    return sanitized or "task"


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"", "none", "null"}:
        return None
    return int(value)


def resolve_qv_cache_path(cfg: DictConfig, *, task_data_name: str) -> Path:
    qv_cache_cfg = getattr(cfg.runtime, "qv_cache", None)
    cache_path_value = None if qv_cache_cfg is None else getattr(qv_cache_cfg, "path", None)
    if cache_path_value is not None and str(cache_path_value).strip().lower() not in {"", "none", "null"}:
        cache_path = Path(to_absolute_path(str(cache_path_value)))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        return cache_path
    cache_dir_value = "./outputs/awr/qv_cache" if qv_cache_cfg is None else getattr(qv_cache_cfg, "dir", "./outputs/awr/qv_cache")
    cache_dir = Path(to_absolute_path(str(cache_dir_value)))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{_sanitize_cache_stem(task_data_name)}.pt"


def build_qv_cache_metadata(
    cfg: DictConfig,
    *,
    task_name: str,
    task_data_name: str,
    policy_camera_names: list[str],
    init_checkpoint: Path | None,
    awr_config: Any | None = None,
) -> dict[str, Any]:
    model_cfg = OmegaConf.to_container(cfg.algorithm.awr.model, resolve=True)
    action_horizon = int(getattr(awr_config, "action_horizon", cfg.algorithm.awr.action_horizon))
    execute_horizon = int(getattr(awr_config, "execute_horizon", cfg.algorithm.awr.execute_horizon))
    return {
        "schema_version": 2,
        "task_name": str(task_name),
        "task_data_name": str(task_data_name),
        "reward_convention": "sparse_success_-1_0",
        "init_checkpoint": None if init_checkpoint is None else str(init_checkpoint.resolve()),
        "value_warmup_steps": int(cfg.algorithm.trainer.value_warmup_steps),
        "success_num_trajectories": _optional_int(cfg.data.success_num_trajectories),
        "fail_num_trajectories": _optional_int(cfg.data.fail_num_trajectories),
        "policy_camera_names": [str(name) for name in policy_camera_names],
        "img_height": int(cfg.env.img_height),
        "img_width": int(cfg.env.img_width),
        "control_freq": int(cfg.env.control_freq),
        "encoder_type": str(cfg.algorithm.encoder.encoder_type),
        "awr_signature": {
            "q_chunk_critic": True,
            "action_horizon": action_horizon,
            "execute_horizon": execute_horizon,
            "image_size": int(cfg.algorithm.awr.image_size),
            "critic_learning_rate": float(cfg.algorithm.awr.critic_learning_rate),
            "weight_decay": float(cfg.algorithm.awr.weight_decay),
            "grad_clip_norm": float(cfg.algorithm.awr.grad_clip_norm),
            "discount": float(cfg.algorithm.awr.discount),
            "expectile": float(cfg.algorithm.awr.expectile),
            "critic_hidden_dims": [int(dim) for dim in list(cfg.algorithm.awr.critic_hidden_dims)],
            "model_cfg": model_cfg,
        },
    }


def compare_qv_cache_metadata(
    cached_metadata: dict[str, Any] | None,
    expected_metadata: dict[str, Any],
) -> tuple[bool, str]:
    cached = {} if cached_metadata is None else dict(cached_metadata)
    if cached == expected_metadata:
        return True, "match"

    mismatch_keys: list[str] = []
    all_keys = sorted(set(cached.keys()) | set(expected_metadata.keys()))
    for key in all_keys:
        if cached.get(key) != expected_metadata.get(key):
            mismatch_keys.append(str(key))
    if not mismatch_keys:
        return False, "metadata differs"
    return False, f"mismatched_keys={','.join(mismatch_keys)}"


def maybe_load_qv_cache(
    *,
    cfg: DictConfig,
    agent,
    trainer: AWRTrainer,
    cache_path: Path,
    expected_metadata: dict[str, Any],
) -> bool:
    if not cache_path.exists():
        return False
    try:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"[qv_cache] failed_to_load path={cache_path} error={exc}")
        return False

    cached_metadata = payload.get("metadata")
    metadata_matches, reason = compare_qv_cache_metadata(cached_metadata, expected_metadata)
    strict_metadata = bool(getattr(cfg.runtime.qv_cache, "strict_metadata", True))
    if not metadata_matches and strict_metadata:
        print(f"[qv_cache] skip path={cache_path} reason={reason}")
        return False
    if not metadata_matches:
        print(f"[qv_cache] non_strict_load path={cache_path} reason={reason}")

    try:
        extra = agent.load_qv_cache_payload(payload, load_optimizers=True)
    except Exception as exc:
        print(f"[qv_cache] failed_to_apply path={cache_path} error={exc}")
        return False

    trainer_state = extra.get("trainer_state", {})
    if trainer_state:
        trainer.load_state_dict(trainer_state)
    print(
        f"[qv_cache] loaded path={cache_path} "
        f"value_updates={int(trainer.total_value_updates)} warmup_updates={int(trainer.total_value_warmup_updates)}"
    )
    return True


def save_qv_cache(
    *,
    agent,
    trainer: AWRTrainer,
    cache_path: Path,
    metadata: dict[str, Any],
) -> None:
    trainer_state = {
        "total_value_updates": int(trainer.total_value_updates),
        "total_value_warmup_updates": int(trainer.total_value_warmup_updates),
    }
    payload = agent.build_qv_cache_payload(metadata=metadata, trainer_state=trainer_state)
    agent.write_checkpoint_payload(cache_path, payload)
    print(
        f"[qv_cache] saved path={cache_path} "
        f"value_updates={trainer_state['total_value_updates']} warmup_updates={trainer_state['total_value_warmup_updates']}"
    )


def _tag_transitions(transitions: list, *, demo_source: str) -> list:
    tagged = []
    for transition in transitions:
        transition.demo_source = str(demo_source)
        tagged.append(transition)
    return tagged


def _build_transition_loader(
    *,
    policy_camera_names: list[str],
    camera_aliases: dict[str, str],
    img_height: int,
    img_width: int,
    proprio_keys: tuple[str, ...],
    renderer: str,
    control_freq: int,
    state_extractor,
):
    def _loader(path, demo_names=None):
        # Keep flow-style observations (camera + proprio formatting) so they stay compatible
        # with the AWR model + normalizers, but compute reward/done via env replay so that
        # success can occur before the final padded step in fixed-length rollouts.
        available_demo_names = set(list_hdf5_demo_names(path))
        if demo_names is None:
            selected_demo_names = sorted(available_demo_names)
        else:
            selected_demo_names = [str(name) for name in demo_names if str(name) in available_demo_names]

        aligned_transitions = []
        total_flow_transitions = 0
        total_env_transitions = 0
        truncated_demo_count = 0
        for demo_name in selected_demo_names:
            flow_transitions = load_hdf5_demos_into_flow_transitions(
                path,
                policy_camera_names=policy_camera_names,
                camera_aliases=camera_aliases,
                img_height=img_height,
                img_width=img_width,
                proprio_keys=proprio_keys,
                renderer=renderer,
                control_freq=control_freq,
                demo_names=[demo_name],
                state_extractor=state_extractor,
            )
            if not flow_transitions:
                continue

            env_transitions = load_hdf5_demos_into_transitions(
                path,
                camera_names=tuple(policy_camera_names),
                img_height=int(img_height),
                img_width=int(img_width),
                proprio_keys=tuple(proprio_keys),
                renderer=str(renderer),
                control_freq=int(control_freq),
                demo_names=(demo_name,),
            )
            if not env_transitions:
                raise RuntimeError(
                    "Offline demo env replay produced zero transitions for non-empty flow demo: "
                    f"{Path(path)}:{demo_name}"
                )
            if len(env_transitions) > len(flow_transitions):
                raise RuntimeError(
                    "Offline demo env replay produced more transitions than flow loading: "
                    f"{len(env_transitions)} vs {len(flow_transitions)} for {Path(path)}:{demo_name}"
                )

            total_flow_transitions += len(flow_transitions)
            total_env_transitions += len(env_transitions)
            if len(env_transitions) < len(flow_transitions):
                truncated_demo_count += 1

            for flow_t, env_t in zip(flow_transitions, env_transitions):
                flow_t.reward = float(env_t.reward)
                flow_t.done = bool(env_t.done)
                flow_t.reward_source = "env_success"
                aligned_transitions.append(flow_t)

        if truncated_demo_count > 0:
            print(
                "[INFO] Truncated AWR offline demo transitions at env replay terminal steps: "
                f"demos={truncated_demo_count}, kept={total_env_transitions}, original={total_flow_transitions}, "
                f"path={Path(path)}"
            )

        return aligned_transitions

    return _loader


def _load_split_transitions(
    *,
    split_name: str,
    demo_paths: list[Path],
    max_num_trajectories: int | None,
    output_root: Path,
    checkpoint_dir: Path,
    loader,
    cache_key_parts: list[str],
) -> list:
    if not demo_paths:
        return []
    cache_root = output_root / "_demo_cache"
    transitions = load_demo_paths(
        demo_paths,
        cache_dir=cache_root,
        mirror_cache_dir=checkpoint_dir / "demo_cache",
        hdf5_loader=loader,
        max_num_trajectories=max_num_trajectories,
        cache_key="__".join(cache_key_parts + [split_name]),
    )
    return _tag_transitions(transitions, demo_source=split_name)


def _shape_discriminator_reward(
    *,
    env_reward: float,
    score: float,
    threshold: float,
    margin_scale_floor: float,
    success_reward_scale: float,
    discriminator_reward_scale: float,
    discriminator_reward_clip: float,
) -> tuple[float, float, float]:
    safe_score = 0.0 if not np.isfinite(score) else float(score)
    safe_threshold = 0.0 if not np.isfinite(threshold) else float(threshold)
    scale = max(abs(safe_threshold), float(margin_scale_floor))
    normalized_margin = (safe_score - safe_threshold) / scale
    discriminator_reward = float(np.clip(-normalized_margin, -float(discriminator_reward_clip), float(discriminator_reward_clip)))
    total_reward = float(success_reward_scale) * float(env_reward) + float(discriminator_reward_scale) * discriminator_reward
    return total_reward, discriminator_reward, normalized_margin


@hydra.main(version_base="1.2", config_path="./config", config_name="train_awr")
def main(cfg: DictConfig) -> None:
    maybe_set_seed(getattr(cfg, "seed", None))
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    init_checkpoint, init_payload = load_init_checkpoint_payload(cfg)
    task_name = str(cfg.env.environment)
    requested_camera_names = resolve_camera_names(cfg)
    flow_env_metadata = resolve_flow_task_metadata(init_payload, task_name)
    if bool(getattr(cfg.runtime, "use_init_checkpoint_camera_names", True)) and init_payload is not None:
        policy_camera_names = [str(name) for name in init_payload.get("camera_names", [])]
        if len(policy_camera_names) == 0:
            policy_camera_names = list(requested_camera_names)
    else:
        policy_camera_names = list(requested_camera_names)
    render_camera_names = list(policy_camera_names)
    cfg.algorithm.camera_names = list(policy_camera_names)
    camera_aliases = {
        str(key): str(value)
        for key, value in dict(getattr(cfg.algorithm.awr, "camera_aliases", {}) or {}).items()
    }

    run_name, checkpoint_dir = resolve_run_directory(cfg)
    output_root = Path(to_absolute_path(str(cfg.logging.output_root)))
    run_started_at = datetime.datetime.now().isoformat(timespec="seconds")
    run_started_monotonic = time.monotonic()
    console_log_path = checkpoint_dir / "console.log"
    runtime_log_path = checkpoint_dir / "metrics_runtime.jsonl"
    console_capture = ConsoleLogCapture(console_log_path)
    console_capture.start()
    runtime_logger = JsonlEventLogger(runtime_log_path)
    runtime_logger.start()
    runtime_logger.log(
        {
            "event": "run_start",
            "run_name": run_name,
            "run_dir": str(checkpoint_dir),
            "started_at": run_started_at,
            "wall_time": datetime.datetime.now().isoformat(timespec="milliseconds"),
            "run_elapsed_sec": 0.0,
            "training_elapsed_sec": 0.0,
        }
    )
    buffer_save_interval = max(1, int(getattr(cfg.runtime, "buffer_save_interval", 200)))
    buffer_writer = AsyncTransitionChunkWriter(
        checkpoint_dir / "buffers",
        chunk_size=buffer_save_interval,
        event_logger=runtime_logger.log,
    )
    buffer_writer.start()
    print(f"[run] {run_name}")
    print(f"[path] {checkpoint_dir}")
    print(
        f"[env] task={cfg.env.environment} robots={list(cfg.env.robots)} "
        f"render_cameras={render_camera_names} policy_cameras={policy_camera_names}"
    )
    print(f"[view] render_camera={resolve_render_camera(cfg, render_camera_names)}")

    visualize_gripper_markers = bool(getattr(cfg.runtime, "visualize_gripper_markers", True))
    frozen_eval_mode = (
        (not bool(getattr(cfg.runtime, "online_updates_enabled", True)))
        and (not bool(cfg.intervention.enabled))
    )
    if len(render_camera_names) == 0:
        raise ValueError("awr requires at least one policy camera.")

    online_updates_enabled = bool(getattr(cfg.runtime, "online_updates_enabled", True))
    viewer_enabled = bool(cfg.runtime.interactive) and bool(cfg.runtime.viewer_enabled)
    decoupled_viewer_enabled = viewer_enabled and online_updates_enabled
    rollout_has_renderer = viewer_enabled and (not decoupled_viewer_enabled)
    main_renderer = str(cfg.env.renderer)
    if viewer_enabled and main_renderer != "mjviewer":
        print(
            f"[WARN] Overriding env.renderer='{main_renderer}' to 'mjviewer' so the main training env uses "
            "robosuite's native window renderer."
        )
        main_renderer = "mjviewer"
    main_runtime_cfg = build_flow_runtime_cfg(
        cfg,
        env_metadata=flow_env_metadata,
        camera_names=render_camera_names,
        has_renderer=rollout_has_renderer,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        renderer=main_renderer,
    )
    if visualize_gripper_markers:
        print(
            "[INFO] Disabling gripper visualization markers on the main env because its camera observations "
            "are feeding the policy."
        )
    main_env = build_robosuite_env(main_runtime_cfg)
    main_env = maybe_wrap_visualization(
        main_env,
        enabled=False,
        label="training env",
    )
    print("[INFO] Using camera observations directly from the main env.")

    flow_proprio_extractor = bind_flow_proprio_extractor(main_env, flow_env_metadata)
    initial_obs, _ = reset_flow_policy_observation(
        main_env,
        preserve_mjviewer=rollout_has_renderer,
        extractor=flow_proprio_extractor,
        policy_camera_names=policy_camera_names,
        camera_aliases=camera_aliases,
        img_height=int(cfg.env.img_height),
        img_width=int(cfg.env.img_width),
    )
    if rollout_has_renderer and getattr(main_env, "viewer", None) is not None and hasattr(main_env.viewer, "update"):
        main_env.viewer.update()

    action_low, action_high = main_env.action_spec
    action_low = np.asarray(action_low, dtype=np.float32)
    action_high = np.asarray(action_high, dtype=np.float32)
    discriminator_reward_enabled = bool(getattr(cfg.runtime, "discriminator_reward_enabled", True)) and bool(
        getattr(cfg.discriminator, "enabled", False)
    )
    if discriminator_reward_enabled:
        raise NotImplementedError(
            "Discriminator shaping is currently not supported in this training script. "
            "Please disable it by setting `discriminator.enabled=false` and/or "
            "`runtime.discriminator_reward_enabled=false` (and keep `algorithm.awr.reward_from_discriminator=false`)."
        )
    algorithm_cfg = OmegaConf.to_container(cfg.algorithm, resolve=True)
    if isinstance(algorithm_cfg, dict):
        algorithm_cfg["camera_names"] = list(policy_camera_names)
        algorithm_cfg["task_name"] = task_name
        awr_cfg = algorithm_cfg.setdefault("awr", {})
        awr_cfg.setdefault("image_size", int(cfg.env.img_height))
        awr_cfg["reward_from_discriminator"] = bool(discriminator_reward_enabled)
        if bool(getattr(cfg.runtime, "use_init_checkpoint_model", True)) and init_payload is not None:
            if "model_cfg" in init_payload:
                awr_cfg["model"] = init_payload["model_cfg"]
            if "task_prompt_map" in init_payload:
                awr_cfg["task_prompt_map"] = init_payload["task_prompt_map"]
            if init_payload.get("act_mean") is not None:
                configured_horizon = int(getattr(cfg.algorithm.awr, "action_horizon", 8))
                inferred_horizon = int(np.asarray(init_payload["act_mean"]).shape[0])
                configured_execute = int(awr_cfg.get("execute_horizon", configured_horizon))
                awr_cfg["action_horizon"] = inferred_horizon
                if configured_execute == configured_horizon:
                    awr_cfg["execute_horizon"] = inferred_horizon
        model_cfg = awr_cfg.setdefault("model", {})
        image_encoder_cfg = model_cfg.get("image_encoder", None)
        if isinstance(image_encoder_cfg, dict) and image_encoder_cfg.get("pretrained_path"):
            image_encoder_cfg["pretrained_path"] = to_absolute_path(str(image_encoder_cfg["pretrained_path"]))
        language_encoder_cfg = model_cfg.get("language_encoder", None)
        if isinstance(language_encoder_cfg, dict) and language_encoder_cfg.get("pretrained_name"):
            language_encoder_cfg["pretrained_name"] = to_absolute_path(str(language_encoder_cfg["pretrained_name"]))
        learner_device, inference_device = resolve_algorithm_devices(algorithm_cfg)
        print(f"[INFO] Learner device: {learner_device}")
        print(f"[INFO] Inference device: {inference_device}")
        if learner_device.startswith("cuda") and inference_device == "cpu":
            print("[INFO] Inference policy is running on CPU to reduce render-time stutter on single-GPU setups.")

    agent = build_algorithm(
        algorithm_cfg,
        observation_example=initial_obs,
        sample_action=np.zeros_like(action_low, dtype=np.float32),
        action_low=action_low,
        action_high=action_high,
    )
    checkpoint_writer = AsyncCheckpointWriter(agent, checkpoint_dir)
    checkpoint_writer.start()
    trainer = AWRTrainer(agent)

    env = main_env
    obs = initial_obs
    control_fps = resolve_runtime_fps(cfg, "control_fps", float(cfg.env.control_freq))
    render_fps = resolve_runtime_fps(cfg, "render_fps", control_fps)
    policy_fps = resolve_runtime_fps(cfg, "policy_fps", control_fps)
    spacemouse_fps = resolve_runtime_fps(cfg, "spacemouse_fps", control_fps)
    fps_log_interval = max(0.1, float(getattr(cfg.runtime, "fps_log_interval", 1.0)))
    unthrottled_runtime = bool(getattr(cfg.runtime, "unthrottled", False))
    async_updates = bool(getattr(cfg.runtime, "async_updates", False))
    num_train_step = int(getattr(cfg.runtime, "num_train_step", 0))
    stream_training_during_rollout = num_train_step < 0
    eval_episode_max_steps = int(getattr(cfg.runtime, "eval_episode_max_steps", 300))
    train_episode_max_steps = int(getattr(cfg.runtime, "train_episode_max_steps", 300))
    viewer_runtime: RobosuiteViewerRuntime | None = None
    if decoupled_viewer_enabled:
        viewer_requested_backend = str(getattr(cfg.runtime, "viewer_backend", "auto"))
        if viewer_requested_backend.lower() == "auto" and main_renderer == "mjviewer":
            viewer_backend = "mjviewer"
        else:
            viewer_backend = choose_viewer_backend(
                main_renderer,
                render_camera_names,
                requested_backend=viewer_requested_backend,
            )
        viewer_runtime_cfg = build_flow_runtime_cfg(
            cfg,
            env_metadata=flow_env_metadata,
            camera_names=render_camera_names,
            has_renderer=True,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            renderer=main_renderer,
        )
        viewer_runtime_cfg.render_camera = resolve_render_camera(cfg, render_camera_names)
        viewer_env = build_robosuite_env(viewer_runtime_cfg)
        viewer_env = maybe_wrap_visualization(
            viewer_env,
            enabled=visualize_gripper_markers,
            label="viewer env",
        )
        viewer_runtime = RobosuiteViewerRuntime(
            viewer_env,
            render_fps=render_fps,
            async_mode=bool(getattr(cfg.runtime, "viewer_async", False)),
            preview_camera=resolve_render_camera(cfg, render_camera_names),
            backend=viewer_backend,
        )
        viewer_runtime.start()
        viewer_startup_delay = max(0.0, float(getattr(cfg.runtime, "viewer_startup_delay", 0.0)))
        if viewer_startup_delay > 0.0:
            time.sleep(viewer_startup_delay)
    if unthrottled_runtime and bool(cfg.intervention.enabled):
        print("[WARN] runtime.unthrottled=true is incompatible with human intervention. Falling back to throttled mode.")
        unthrottled_runtime = False
    if abs(control_fps - float(cfg.env.control_freq)) > 1e-6:
        print(
            "[WARN] runtime.control_fps does not match env.control_freq. "
            f"Actor loop will target {control_fps:.2f}Hz while the env is configured for {int(cfg.env.control_freq)}Hz."
        )
    if not online_updates_enabled:
        print("[INFO] Online learner updates are disabled. Policy parameters will stay frozen during rollout.")
    elif stream_training_during_rollout:
        print("[INFO] Online learner updates run concurrently with rollout.")
    else:
        print(
            f"[INFO] Episodic learner mode enabled. The rollout loop only runs inference/discriminator; "
            f"each episode then runs num_train_step={num_train_step} updates."
        )
    print("[INFO] Main env camera observations are being used for policy inference.")
    if decoupled_viewer_enabled:
        print(
            "[INFO] Training viewer is decoupled from rollout. "
            f"Window refresh runs at {render_fps:.2f}Hz while policy observations still come from the rollout env."
        )
    if frozen_eval_mode:
        print(
            f"[INFO] Frozen-policy eval mode will reset episodes after {eval_episode_max_steps} steps "
            "if not terminated earlier."
        )
    elif not stream_training_during_rollout:
        print(
            f"[INFO] Training mode will reset episodes after {train_episode_max_steps} steps "
            "if not terminated earlier."
        )
    if unthrottled_runtime:
        print("[INFO] Running without real-time control throttling.")

    loaded_checkpoint = None
    initialized_checkpoint = None
    extra: dict[str, Any] = {}
    for candidate in resume_checkpoint_candidates(cfg, output_root):
        try:
            extra = agent.load_checkpoint(candidate, load_buffers=bool(cfg.runtime.load_buffers))
            loaded_checkpoint = candidate
            print(f"[load] checkpoint={candidate}")
            break
        except Exception as exc:
            print(f"[WARN] Skipping invalid checkpoint {candidate}: {exc}")

    if loaded_checkpoint is None and init_checkpoint is not None:
        init_payload = agent.load_flow_policy_checkpoint(init_checkpoint, task_name=task_name)
        initialized_checkpoint = init_checkpoint
        print(f"[init] flow_policy_checkpoint={init_checkpoint}")
    elif loaded_checkpoint is None:
        raise FileNotFoundError("AWR requires runtime.init_checkpoint or a resumable awr checkpoint.")

    trainer.load_state_dict(extra.get("trainer_state"))
    if loaded_checkpoint is not None and bool(cfg.runtime.load_buffers) and len(agent.demo_buffer) == 0 and len(agent.online_buffer) == 0:
        online_chunk_dir, demo_chunk_dir = resolve_buffer_chunk_dirs(loaded_checkpoint)
        loaded_online_transitions = 0
        loaded_demo_transitions = 0
        if online_chunk_dir is not None:
            loaded_online_transitions = load_transition_chunks(agent.online_buffer, online_chunk_dir)
        if demo_chunk_dir is not None:
            loaded_demo_transitions = load_transition_chunks(agent.demo_buffer, demo_chunk_dir)
        if loaded_online_transitions > 0 or loaded_demo_transitions > 0:
            print(
                f"[load] chunk_buffers online={loaded_online_transitions} demo={loaded_demo_transitions} "
                f"from {loaded_checkpoint.parent.parent / 'buffers'}"
            )

    if loaded_checkpoint is not None:
        start_step = int(extra.get("global_step", -1)) + 1
        episode_index = int(extra.get("episode_index", 0))
    else:
        start_step = 0
        episode_index = 0

    write_resolved_config(cfg, checkpoint_dir)
    write_run_info(
        checkpoint_dir,
        {
            "run_name": run_name,
            "run_dir": str(checkpoint_dir),
            "started_at": run_started_at,
            "env_name": str(cfg.env.environment),
            "robots": [str(name) for name in list(cfg.env.robots)],
            "policy_camera_names": policy_camera_names,
            "render_camera_names": render_camera_names,
            "render_camera": resolve_render_camera(cfg, render_camera_names),
            "loaded_checkpoint": None if loaded_checkpoint is None else str(loaded_checkpoint),
            "initialized_checkpoint": None if initialized_checkpoint is None else str(initialized_checkpoint),
            "resume_enabled": bool(cfg.runtime.resume),
            "load_buffers": bool(cfg.runtime.load_buffers),
            "seed": serialize_seed(getattr(cfg, "seed", None)),
            "console_log": str(console_log_path),
            "runtime_log": str(runtime_log_path),
            "buffer_dir": str(checkpoint_dir / "buffers"),
            "online_chunk_dir": str(checkpoint_dir / "buffers" / "online_chunks"),
            "demo_chunk_dir": str(checkpoint_dir / "buffers" / "demo_chunks"),
        },
    )

    task_data_name = resolve_demo_task_name(cfg)
    qv_cache_cfg = getattr(cfg.runtime, "qv_cache", None)
    qv_cache_enabled = bool(getattr(qv_cache_cfg, "enabled", True)) if qv_cache_cfg is not None else True
    qv_cache_force_rebuild = bool(getattr(qv_cache_cfg, "force_rebuild", False)) if qv_cache_cfg is not None else False
    qv_cache_save_enabled = bool(getattr(qv_cache_cfg, "save", True)) if qv_cache_cfg is not None else True
    qv_cache_direct_load = bool(getattr(qv_cache_cfg, "direct_load", False)) if qv_cache_cfg is not None else False
    qv_cache_require_existing = qv_cache_direct_load
    if qv_cache_cfg is not None:
        qv_cache_require_existing = bool(getattr(qv_cache_cfg, "require_existing", qv_cache_direct_load))
    qv_cache_path = resolve_qv_cache_path(cfg, task_data_name=task_data_name)
    qv_cache_metadata = build_qv_cache_metadata(
        cfg,
        task_name=task_name,
        task_data_name=task_data_name,
        policy_camera_names=policy_camera_names,
        init_checkpoint=init_checkpoint,
        awr_config=agent.awr_config,
    )
    qv_cache_loaded = False
    if loaded_checkpoint is None and qv_cache_enabled and qv_cache_direct_load:
        if qv_cache_force_rebuild:
            raise ValueError("runtime.qv_cache.direct_load=true cannot be combined with force_rebuild=true.")
        qv_cache_loaded = maybe_load_qv_cache(
            cfg=cfg,
            agent=agent,
            trainer=trainer,
            cache_path=qv_cache_path,
            expected_metadata=qv_cache_metadata,
        )
        if not qv_cache_loaded and qv_cache_require_existing:
            raise FileNotFoundError(f"Required Q/V cache could not be loaded from {qv_cache_path}")

    expert_paths = resolve_task_demo_paths(task_data_name, data_root=to_absolute_path(str(cfg.data.demo_root)), split="expert")
    success_paths = []
    fail_paths = []
    if not qv_cache_loaded:
        success_paths = resolve_task_demo_paths(
            task_data_name,
            data_root=to_absolute_path(str(cfg.data.demo_root)),
            split="success_rollout",
        )
        fail_paths = resolve_task_demo_paths(
            task_data_name,
            data_root=to_absolute_path(str(cfg.data.demo_root)),
            split="fail_rollout",
        )
    if not expert_paths:
        raise FileNotFoundError(
            f"AWR requires expert demos under ./data/<task>/expert. No files were found for task '{task_data_name}'."
        )
    if loaded_checkpoint is None and not qv_cache_loaded and (not success_paths or not fail_paths):
        raise FileNotFoundError(
            f"AWR requires both success_rollout and fail_rollout for value warmup. task='{task_data_name}', "
            f"success_files={len(success_paths)}, fail_files={len(fail_paths)}"
        )

    proprio_keys = [str(key) for key in list(cfg.env.proprio_keys or [])]
    cache_key_parts = [
        f"loader-v{AWR_DEMO_LOADER_SCHEMA_VERSION}",
        f"h{int(cfg.env.img_height)}",
        f"w{int(cfg.env.img_width)}",
        f"cams-{'_'.join(policy_camera_names)}",
        f"state-{'_'.join(proprio_keys) if proprio_keys else 'auto'}",
    ]
    transition_loader = _build_transition_loader(
        policy_camera_names=policy_camera_names,
        camera_aliases=camera_aliases,
        img_height=int(cfg.env.img_height),
        img_width=int(cfg.env.img_width),
        proprio_keys=tuple(cfg.env.proprio_keys or []),
        renderer=str(cfg.env.renderer),
        control_freq=int(cfg.env.control_freq),
        state_extractor=flow_proprio_extractor,
    )

    print(f"[INFO] Loading expert demos for task={task_data_name}...")
    expert_transitions = _load_split_transitions(
        split_name="offline_demo",
        demo_paths=expert_paths,
        max_num_trajectories=cfg.data.expert_num_trajectories,
        output_root=output_root,
        checkpoint_dir=checkpoint_dir,
        loader=transition_loader,
        cache_key_parts=cache_key_parts,
    )
    if len(expert_transitions) == 0:
        raise RuntimeError("Expert demo loading completed but produced zero transitions.")
    if not agent.demo_buffer.is_compatible(expert_transitions[0].obs, expert_transitions[0].action):
        print("[WARN] Loaded checkpoint demo_buffer is incompatible with current observation shape. Rebuilding demo buffer.")
        agent.demo_buffer.clear()
    if not agent.online_buffer.is_compatible(expert_transitions[0].obs, expert_transitions[0].action):
        print(
            "[WARN] Loaded checkpoint online_buffer is incompatible with current observation shape. "
            "Rebuilding online buffer."
        )
        agent.online_buffer.clear()

    if len(agent.demo_buffer) == 0:
        trainer.bootstrap_demo_buffer(
            expert_transitions,
            demo_source="offline_demo",
            episode_namespace="offline_demo",
        )
        print(f"[INFO] Loaded expert demo transitions: {len(expert_transitions)}")
    else:
        print(f"[INFO] Reusing checkpoint demo buffer with {len(agent.demo_buffer)} transitions")

    if len(agent.online_buffer) == 0 and qv_cache_loaded:
        print("[INFO] Skipping success/fail rollout loading because Q/V cache was loaded directly.")
    elif len(agent.online_buffer) == 0:
        print(f"[INFO] Loading success/fail rollout trajectories for task={task_data_name}...")
        success_transitions = _load_split_transitions(
            split_name="success_rollout",
            demo_paths=success_paths,
            max_num_trajectories=cfg.data.success_num_trajectories,
            output_root=output_root,
            checkpoint_dir=checkpoint_dir,
            loader=transition_loader,
            cache_key_parts=cache_key_parts,
        )
        fail_transitions = _load_split_transitions(
            split_name="fail_rollout",
            demo_paths=fail_paths,
            max_num_trajectories=cfg.data.fail_num_trajectories,
            output_root=output_root,
            checkpoint_dir=checkpoint_dir,
            loader=transition_loader,
            cache_key_parts=cache_key_parts,
        )
        if len(success_transitions) == 0 or len(fail_transitions) == 0:
            raise RuntimeError(
                "AWR value warmup requires both success_rollout and fail_rollout to produce transitions."
            )
        trainer.bootstrap_online_buffer(
            success_transitions,
            demo_source="success_rollout",
            episode_namespace="success_rollout",
        )
        trainer.bootstrap_online_buffer(
            fail_transitions,
            demo_source="fail_rollout",
            episode_namespace="fail_rollout",
        )
        print(
            f"[INFO] Loaded rollout transitions success={len(success_transitions)} fail={len(fail_transitions)}"
        )
    else:
        print(f"[INFO] Reusing checkpoint online buffer with {len(agent.online_buffer)} transitions")

    if not agent.has_normalizers():
        agent.fit_normalizers_from_transitions(expert_transitions)
        print("[INFO] Fitted AWR normalizers from expert demos.")
    else:
        print("[INFO] Reusing normalizers from checkpoint.")

    if loaded_checkpoint is None and qv_cache_enabled and not qv_cache_direct_load:
        if qv_cache_force_rebuild:
            print(f"[qv_cache] force_rebuild path={qv_cache_path}")
        else:
            qv_cache_loaded = maybe_load_qv_cache(
                cfg=cfg,
                agent=agent,
                trainer=trainer,
                cache_path=qv_cache_path,
                expected_metadata=qv_cache_metadata,
            )

    wandb_run = maybe_build_wandb(cfg, run_name=run_name, run_dir=checkpoint_dir)
    policy_worker = AWRPolicyWorker(agent)
    policy_worker.start()
    discriminator_worker = None
    if discriminator_reward_enabled:
        discriminator_worker = AWRDiscriminatorWorker(cfg.discriminator, task_name=task_name)
        discriminator_worker.start()
    else:
        print("[INFO] Discriminator reward is disabled. LPB discriminator worker will not be started.")

    requested_value_warmup_steps = max(0, int(cfg.algorithm.trainer.value_warmup_steps))
    if loaded_checkpoint is None and start_step == 0 and qv_cache_direct_load and qv_cache_loaded:
        print(f"[warmup] skipped reason=qv_cache_direct_load path={qv_cache_path}")
    elif loaded_checkpoint is None and start_step == 0:
        completed_warmup_steps = int(trainer.total_value_warmup_updates)
        remaining_warmup_steps = max(0, requested_value_warmup_steps - completed_warmup_steps)
        print(
            f"[warmup] remaining_value_steps={remaining_warmup_steps} "
            f"(completed={completed_warmup_steps}, target={requested_value_warmup_steps})"
        )
        for local_step in range(remaining_warmup_steps):
            metrics = trainer.pretrain_value(1)[-1]
            absolute_warmup_step = completed_warmup_steps + local_step + 1
            if absolute_warmup_step % int(cfg.logging.log_interval) == 0:
                maybe_log(
                    wandb_run,
                    {f"warmup/{key}": value for key, value in metrics.items()},
                    step=absolute_warmup_step,
                )
                print(
                    "[warmup] "
                    f"step={absolute_warmup_step} q={metrics.get('q_loss', float('nan')):.4f} "
                    f"v={metrics.get('value_loss', float('nan')):.4f}"
                )
        if (not qv_cache_loaded) and qv_cache_enabled and qv_cache_save_enabled and int(trainer.total_value_warmup_updates) > 0:
            save_qv_cache(
                agent=agent,
                trainer=trainer,
                cache_path=qv_cache_path,
                metadata=qv_cache_metadata,
            )

    device = None
    intervention_runtime = None
    pending_online_transitions: dict[tuple[int, int], dict[str, Any]] = {}
    episode_return = 0.0
    episode_length = 0
    episode_step_index = 0
    success_count = 0
    last_step = start_step - 1
    control_limiter = None if unthrottled_runtime else FixedRateLimiter(control_fps)
    policy_gate = IntervalGate(policy_fps)
    spacemouse_gate = IntervalGate(spacemouse_fps)
    overall_fps_tracker = EMAFpsTracker()
    last_fps_log_time = time.monotonic()
    training_started_monotonic = last_fps_log_time
    cached_policy_action = np.zeros_like(action_low, dtype=np.float32)
    cached_override_action: np.ndarray | None = None
    cached_is_intervention = False
    last_reported_publish_count = int(trainer.progress_snapshot()["publish_count"])
    episode_pause_sec = max(0.0, float(getattr(cfg.runtime, "episode_pause_sec", 1.0)))
    total_transition_count = max(0, int(start_step))
    total_intervention_transitions = 0
    episode_transition_count = 0
    episode_intervention_transitions = 0
    margin_scale_floor = float(cfg.algorithm.awr.margin_scale_floor)
    success_reward_scale = float(cfg.algorithm.awr.success_reward_scale)
    discriminator_reward_scale = float(cfg.algorithm.awr.discriminator_reward_scale)
    discriminator_reward_clip = float(cfg.algorithm.awr.discriminator_reward_clip)

    def event_time_fields() -> dict[str, Any]:
        now = datetime.datetime.now()
        return {
            "wall_time": now.isoformat(timespec="milliseconds"),
            "run_elapsed_sec": round(time.monotonic() - run_started_monotonic, 6),
            "training_elapsed_sec": round(time.monotonic() - training_started_monotonic, 6),
        }

    def refresh_main_viewer() -> None:
        if viewer_runtime is not None:
            viewer_runtime.publish_from_env(env)
            viewer_runtime.render_if_due()
            return
        if not rollout_has_renderer:
            return
        if getattr(env, "viewer", None) is None and hasattr(env, "initialize_renderer"):
            env.initialize_renderer()
        if getattr(env, "viewer", None) is not None and hasattr(env.viewer, "update"):
            env.viewer.update()

    def reset_viewer_preview() -> None:
        if viewer_runtime is None:
            refresh_main_viewer()
            return
        viewer_runtime.reset_preview(
            snapshot_env_state(env),
            warmup_frames=int(getattr(cfg.runtime, "viewer_reset_warmup_frames", 2)),
        )

    def maybe_print_publish_events(metrics_list: list[dict[str, float]]) -> None:
        nonlocal last_reported_publish_count
        for metrics in metrics_list:
            publish_count = int(metrics.get("learner_publish_count", 0.0))
            if publish_count <= last_reported_publish_count:
                continue
            runtime_logger.log(
                {
                    "event": "publish",
                    "step": int(last_step),
                    "episode_index": int(episode_index),
                    "publish_count": publish_count,
                    "learner_actor_updates": int(metrics.get("learner_actor_updates", 0.0)),
                    **event_time_fields(),
                }
            )
            print(format_publish_line(metrics))
            last_reported_publish_count = publish_count

    def _finalize_pending_entry(key: tuple[int, int], *, total_reward: float, env_reward: float, discriminator_reward: float, score: float, threshold: float, normalized_margin: float, source: str, metadata: dict[str, Any] | None = None) -> None:
        online_transition = pending_online_transitions[key]["online_transition"]
        demo_transition = pending_online_transitions[key].get("demo_transition")
        apply_awr_reward_to_transition(
            online_transition,
            total_reward=total_reward,
            env_reward=env_reward,
            discriminator_reward=discriminator_reward,
            score=score,
            threshold=threshold,
            normalized_margin=normalized_margin,
            source=source,
            metadata=metadata,
            reward_ready=True,
        )
        if demo_transition is not None:
            apply_awr_reward_to_transition(
                demo_transition,
                total_reward=total_reward,
                env_reward=env_reward,
                discriminator_reward=discriminator_reward,
                score=score,
                threshold=threshold,
                normalized_margin=normalized_margin,
                source=source,
                metadata=metadata,
                reward_ready=True,
            )
        buffer_writer.request_transition(
            online_transition=online_transition,
            demo_transition=demo_transition,
        )
        pending_online_transitions.pop(key, None)

    def drain_discriminator_results() -> None:
        if discriminator_worker is None:
            return
        for label in discriminator_worker.drain_results():
            decision = label.decision
            awr_fields = agent.get_online_transition_awr_fields(
                episode_namespace="online",
                episode_index=label.episode_index,
                episode_step=label.labeled_episode_step,
            )
            if awr_fields is None:
                continue
            env_reward = float(awr_fields["env_reward"])
            total_reward, discriminator_reward, normalized_margin = _shape_discriminator_reward(
                env_reward=env_reward,
                score=float(decision.score),
                threshold=float(decision.threshold),
                margin_scale_floor=margin_scale_floor,
                success_reward_scale=success_reward_scale,
                discriminator_reward_scale=discriminator_reward_scale,
                discriminator_reward_clip=discriminator_reward_clip,
            )
            agent.patch_transition_reward(
                episode_namespace="online",
                episode_index=label.episode_index,
                episode_step=label.labeled_episode_step,
                total_reward=total_reward,
                env_reward=env_reward,
                discriminator_reward=discriminator_reward,
                score=float(decision.score),
                threshold=float(decision.threshold),
                normalized_margin=normalized_margin,
                source=str(decision.metadata.get("detector_name", "discriminator")),
                metadata=dict(decision.metadata),
                patch_demo=True,
            )
            key = (int(label.episode_index), int(label.labeled_episode_step))
            if key in pending_online_transitions:
                _finalize_pending_entry(
                    key,
                    total_reward=total_reward,
                    env_reward=env_reward,
                    discriminator_reward=discriminator_reward,
                    score=float(decision.score),
                    threshold=float(decision.threshold),
                    normalized_margin=normalized_margin,
                    source=str(decision.metadata.get("detector_name", "discriminator")),
                    metadata=dict(decision.metadata),
                )
            runtime_logger.log(
                {
                    "event": "discriminator_eval",
                    "step": int(label.global_step),
                    "episode_index": int(label.episode_index),
                    "labeled_episode_step": int(label.labeled_episode_step),
                    "lambda": float(decision.score),
                    "threshold": float(decision.threshold),
                    "prediction": int(decision.prediction),
                    "raw_step_score": float(decision.raw_step_score),
                    "normalized_margin": float(normalized_margin),
                    "discriminator_reward": float(discriminator_reward),
                    "total_reward": float(total_reward),
                    **event_time_fields(),
                }
            )
            print(
                format_discriminator_line(
                    label.global_step,
                    label.episode_index,
                    label.labeled_episode_step,
                    decision,
                )
            )

    def finalize_pending_episode(episode_to_finalize: int) -> None:
        if discriminator_worker is not None:
            discriminator_worker.flush(timeout=120.0)
            drain_discriminator_results()
        episode_keys = [item for item in pending_online_transitions.keys() if item[0] == int(episode_to_finalize)]
        for key in episode_keys:
            awr_fields = agent.get_online_transition_awr_fields(
                episode_namespace="online",
                episode_index=key[0],
                episode_step=key[1],
            )
            if awr_fields is None:
                pending_online_transitions.pop(key, None)
                continue
            env_reward = float(awr_fields["env_reward"])
            total_reward = float(success_reward_scale) * env_reward
            agent.patch_transition_reward(
                episode_namespace="online",
                episode_index=key[0],
                episode_step=key[1],
                total_reward=total_reward,
                env_reward=env_reward,
                discriminator_reward=0.0,
                score=float("nan"),
                threshold=float("nan"),
                normalized_margin=0.0,
                source="finalize_without_discriminator",
                metadata={},
                patch_demo=True,
            )
            _finalize_pending_entry(
                key,
                total_reward=total_reward,
                env_reward=env_reward,
                discriminator_reward=0.0,
                score=float("nan"),
                threshold=float("nan"),
                normalized_margin=0.0,
                source="finalize_without_discriminator",
                metadata={},
            )

    def maybe_report_runtime(step: int) -> None:
        nonlocal last_fps_log_time
        now = time.monotonic()
        elapsed = now - last_fps_log_time
        if elapsed < fps_log_interval:
            return
        learner_progress = trainer.progress_snapshot()
        overall_fps = overall_fps_tracker.snapshot(elapsed)
        pending_updates = trainer.pending_async_updates() if async_updates else 0
        online_valid_sequences = agent.online_buffer.num_valid_sequences()
        online_ready_steps = agent.online_buffer.num_ready_steps()
        runtime_payload = {
            "overall_fps": overall_fps,
            "learner_value_updates": float(learner_progress["value_updates"]),
            "learner_actor_updates": float(learner_progress["actor_updates"]),
            "learner_updates_until_publish": float(learner_progress["updates_until_publish"]),
            "learner_publish_count": float(learner_progress["publish_count"]),
            "online_valid_sequences": float(online_valid_sequences),
            "online_ready_steps": float(online_ready_steps),
        }
        if async_updates:
            runtime_payload["pending_updates"] = float(pending_updates)
        maybe_log(wandb_run, runtime_payload, step=step)
        runtime_logger.log(
            {
                "event": "runtime",
                "step": int(step),
                "episode_index": int(episode_index),
                "overall_fps": float(overall_fps),
                "pending_updates": int(pending_updates),
                "learner_value_updates": int(learner_progress["value_updates"]),
                "learner_actor_updates": int(learner_progress["actor_updates"]),
                "learner_publish_count": int(learner_progress["publish_count"]),
                "learner_updates_until_publish": int(learner_progress["updates_until_publish"]),
                "online_valid_sequences": int(online_valid_sequences),
                "online_ready_steps": int(online_ready_steps),
                "demo_valid_sequences": int(agent.demo_buffer.num_valid_sequences()),
                "demo_ready_steps": int(agent.demo_buffer.num_ready_steps()),
                "total_transition_count": int(total_transition_count),
                "total_intervention_transitions": int(total_intervention_transitions),
                **event_time_fields(),
            }
        )
        print(
            format_runtime_line(
                step=step,
                episode_index=episode_index,
                overall_fps=overall_fps,
                learner_progress=learner_progress,
                pending_updates=pending_updates,
                online_valid_sequences=online_valid_sequences,
                online_ready_steps=online_ready_steps,
            )
        )
        last_fps_log_time = now

    def request_checkpoint_save(step: int, tag: str | None = None) -> None:
        buffer_writer.flush(timeout=120.0)
        checkpoint_extra = {
            "global_step": int(step),
            "episode_index": int(episode_index),
            "success_count": int(success_count),
            "trainer_state": trainer.state_dict(),
        }
        if tag is None:
            step_checkpoint = checkpoint_step_path(
                checkpoint_dir,
                step=step,
                learner_updates=trainer.total_actor_updates,
                episode_index=episode_index,
            )
            checkpoint_writer.request_save(
                paths=[step_checkpoint],
                include_buffers=False,
                extra=checkpoint_extra,
                metadata={"step": int(step)},
            )
        checkpoint_writer.request_save(
            paths=[checkpoint_path(checkpoint_dir, "latest")],
            include_buffers=False,
            extra=checkpoint_extra,
            metadata={"step": int(step)},
        )
        print(f"[ckpt] step={step} latest_with_buffers=false")

    if bool(cfg.intervention.enabled):
        device = build_device(env, cfg.intervention)
        intervention_runtime = RobosuiteInterventionRuntime(
            env=env,
            device=device,
            goal_update_mode=str(cfg.intervention.goal_update_mode),
        )

    obs, _ = reset_flow_policy_observation(
        env,
        preserve_mjviewer=rollout_has_renderer,
        extractor=flow_proprio_extractor,
        policy_camera_names=policy_camera_names,
        camera_aliases=camera_aliases,
        img_height=int(cfg.env.img_height),
        img_width=int(cfg.env.img_width),
    )
    reset_viewer_preview()
    policy_worker.reset_policy_state()
    policy_gate.force_ready()
    spacemouse_gate.force_ready()
    if intervention_runtime is not None:
        intervention_runtime.start_episode()
    if discriminator_worker is not None:
        discriminator_worker.reset(
            task_name=task_name,
            initial_state=env.sim.get_state().flatten().copy(),
            initial_images=_initial_discriminator_images(obs, discriminator_worker.runtime.camera_names),
            episode_index=episode_index,
        )

    if async_updates and online_updates_enabled and stream_training_during_rollout:
        trainer.start_async_worker()

    def run_episode_training(step: int) -> list[dict[str, float]]:
        if not online_updates_enabled or stream_training_during_rollout or num_train_step == 0:
            return []
        metrics_list = trainer.train_fixed_steps(num_train_step, env_step=step)
        if not metrics_list:
            print(f"[train_burst] step={step} requested={num_train_step} completed=0")
            return []
        maybe_print_publish_events(metrics_list)
        final_metrics = metrics_list[-1]
        maybe_log(wandb_run, final_metrics, step=step)
        print(
            format_train_line(
                step=step,
                metrics=final_metrics,
                pending_updates=0,
            )
        )
        print(f"[train_burst] step={step} requested={num_train_step} completed={len(metrics_list)}")
        return metrics_list

    try:
        for step in range(start_step, int(cfg.runtime.max_steps)):
            last_step = step
            loop_start = time.monotonic() if control_limiter is None else control_limiter.wait()
            overall_fps_tracker.mark()
            drain_discriminator_results()

            if unthrottled_runtime or policy_gate.ready(loop_start):
                if step < int(cfg.algorithm.trainer.random_steps):
                    cached_policy_action = np.random.uniform(action_low, action_high).astype(np.float32)
                else:
                    cached_policy_action = policy_worker.select_action(obs, deterministic=bool(cfg.runtime.eval_deterministic))

            env_action = np.asarray(cached_policy_action, dtype=np.float32)
            is_intervention = False
            reset_requested = False
            if intervention_runtime is not None and (unthrottled_runtime or spacemouse_gate.ready(loop_start)):
                override_action, sampled_is_intervention, reset_requested = intervention_runtime.maybe_override_action(
                    cached_policy_action
                )
                if reset_requested:
                    cached_override_action = None
                    cached_is_intervention = False
                    policy_worker.reset_policy_state()
                elif sampled_is_intervention:
                    cached_override_action = np.asarray(override_action, dtype=np.float32)
                    cached_is_intervention = True
                    policy_worker.notify_intervention()
                    policy_gate.force_ready()
                else:
                    cached_override_action = None
                    cached_is_intervention = False

            if reset_requested:
                if not bool(cfg.intervention.device_reset_as_episode_reset):
                    print("[INFO] Device reset requested. Exiting training loop.")
                    break

                finalize_pending_episode(episode_index)
                run_episode_training(step)
                reset_episode_payload = {
                    "episode_return": float(episode_return),
                    "episode_length": int(episode_length),
                    "episode_success": 0,
                    "is_success": False,
                    "online_buffer_size": len(agent.online_buffer),
                    "demo_buffer_size": len(agent.demo_buffer),
                    "online_valid_sequences": int(agent.online_buffer.num_valid_sequences()),
                    "online_ready_steps": int(agent.online_buffer.num_ready_steps()),
                    "end_reason": "device_reset",
                }
                maybe_log(wandb_run, reset_episode_payload, step=step)
                runtime_logger.log(
                    {
                        "event": "episode_end",
                        "end_reason": "device_reset",
                        "step": int(step),
                        "episode_index": int(episode_index),
                        "episode_return": float(episode_return),
                        "episode_length": int(episode_length),
                        "episode_success": False,
                        "is_success": False,
                        "episode_transition_count": int(episode_transition_count),
                        "episode_intervention_transitions": int(episode_intervention_transitions),
                        **event_time_fields(),
                    }
                )
                print(
                    format_episode_line(
                        step=step,
                        episode_index=episode_index,
                        episode_return=episode_return,
                        episode_length=episode_length,
                        success=False,
                        online_buffer_size=len(agent.online_buffer),
                        demo_buffer_size=len(agent.demo_buffer),
                    )
                )
                obs, _ = reset_flow_policy_observation(
                    env,
                    preserve_mjviewer=rollout_has_renderer,
                    extractor=flow_proprio_extractor,
                    policy_camera_names=policy_camera_names,
                    camera_aliases=camera_aliases,
                    img_height=int(cfg.env.img_height),
                    img_width=int(cfg.env.img_width),
                )
                reset_viewer_preview()
                policy_worker.reset_policy_state()
                episode_return = 0.0
                episode_length = 0
                episode_step_index = 0
                episode_index += 1
                episode_transition_count = 0
                episode_intervention_transitions = 0
                policy_gate.force_ready()
                spacemouse_gate.force_ready()
                if intervention_runtime is not None:
                    intervention_runtime.start_episode()
                if discriminator_worker is not None:
                    discriminator_worker.reset(
                        task_name=task_name,
                        initial_state=env.sim.get_state().flatten().copy(),
                        initial_images=_initial_discriminator_images(obs, discriminator_worker.runtime.camera_names),
                        episode_index=episode_index,
                    )
                maybe_report_runtime(step)
                continue

            if cached_is_intervention and cached_override_action is not None:
                env_action = np.asarray(cached_override_action, dtype=np.float32)
                is_intervention = True

            grasp_penalty = compute_grasp_penalty(env, env_action)
            step_output = env.step(env_action)
            if len(step_output) == 5:
                raw_next_obs, _, done, truncated, info = step_output
                done = bool(done or truncated)
            else:
                raw_next_obs, _, done, info = step_output
            if isinstance(info, dict) and grasp_penalty is not None:
                info.setdefault("grasp_penalty", float(grasp_penalty))

            # NOTE: here we use -1/0 reward
            reward, success = sparse_success_reward(env, info if isinstance(info, dict) else None)
            next_obs = convert_env_camera_observation(
                raw_next_obs,
                env=env,
                extractor=flow_proprio_extractor,
                policy_camera_names=policy_camera_names,
                camera_aliases=camera_aliases,
                img_height=int(cfg.env.img_height),
                img_width=int(cfg.env.img_width),
            )
            next_state_full = env.sim.get_state().flatten().copy()
            next_images_for_disc = _initial_discriminator_images(next_obs, discriminator_worker.runtime.camera_names) if discriminator_worker is not None else {}
            refresh_main_viewer()
            if frozen_eval_mode and (episode_length + 1) >= eval_episode_max_steps:
                done = True
            elif (not frozen_eval_mode) and train_episode_max_steps > 0 and (episode_length + 1) >= train_episode_max_steps:
                done = True
            done = bool(done or success)

            recorded_transition = trainer.record_transition(
                obs=obs,
                action=env_action,
                next_obs=next_obs,
                done=done,
                reward=reward,
                grasp_penalty=info.get("grasp_penalty") if isinstance(info, dict) else None,
                is_intervention=is_intervention,
                info=dict(info) if isinstance(info, dict) else {"raw_info": info},
                reward_source="env_success",
                demo_source="intervention" if is_intervention else None,
                episode_index=episode_index,
                episode_step=episode_step_index,
                episode_namespace="online",
            )
            serialized_online = agent.online_buffer.snapshot_transition(recorded_transition)
            serialized_demo = agent.demo_buffer.snapshot_transition(recorded_transition) if is_intervention else None
            transition_key = (episode_index, episode_step_index)

            if discriminator_worker is not None:
                pending_online_transitions[transition_key] = {
                    "online_transition": serialized_online,
                    "demo_transition": serialized_demo,
                }
                discriminator_worker.record_step(
                    action=env_action,
                    next_state=next_state_full,
                    next_images=next_images_for_disc,
                    episode_index=episode_index,
                    global_step=step,
                )
            else:
                total_reward = float(success_reward_scale) * float(reward)
                agent.patch_transition_reward(
                    episode_namespace="online",
                    episode_index=episode_index,
                    episode_step=episode_step_index,
                    total_reward=total_reward,
                    env_reward=float(reward),
                    discriminator_reward=0.0,
                    score=float("nan"),
                    threshold=float("nan"),
                    normalized_margin=0.0,
                    source="no_discriminator",
                    metadata={},
                    patch_demo=True,
                )
                apply_awr_reward_to_transition(
                    serialized_online,
                    total_reward=total_reward,
                    env_reward=float(reward),
                    discriminator_reward=0.0,
                    score=float("nan"),
                    threshold=float("nan"),
                    normalized_margin=0.0,
                    source="no_discriminator",
                    metadata={},
                    reward_ready=True,
                )
                if serialized_demo is not None:
                    apply_awr_reward_to_transition(
                        serialized_demo,
                        total_reward=total_reward,
                        env_reward=float(reward),
                        discriminator_reward=0.0,
                        score=float("nan"),
                        threshold=float("nan"),
                        normalized_margin=0.0,
                        source="no_discriminator",
                        metadata={},
                        reward_ready=True,
                    )
                buffer_writer.request_transition(
                    online_transition=serialized_online,
                    demo_transition=serialized_demo,
                )

            total_transition_count += 1
            episode_transition_count += 1
            episode_step_index += 1
            if is_intervention:
                total_intervention_transitions += 1
                episode_intervention_transitions += 1
            episode_return += reward
            episode_length += 1
            success_count += int(success)

            drain_discriminator_results()
            if online_updates_enabled and stream_training_during_rollout:
                update_metrics_list = trainer.maybe_update_async() if async_updates else trainer.maybe_update()
            else:
                update_metrics_list = []
            if update_metrics_list:
                maybe_print_publish_events(update_metrics_list)
                update_metrics = update_metrics_list[-1]
                if step % int(cfg.logging.log_interval) == 0:
                    maybe_log(wandb_run, update_metrics, step=step)
                    print(
                        format_train_line(
                            step=step,
                            metrics=update_metrics,
                            pending_updates=trainer.pending_async_updates() if async_updates else 0,
                        )
                    )

            if done:
                finalize_pending_episode(episode_index)
                run_episode_training(step)
                episode_payload = {
                    "episode_return": float(episode_return),
                    "episode_length": int(episode_length),
                    "episode_success": int(success),
                    "is_success": bool(success),
                    "online_buffer_size": len(agent.online_buffer),
                    "demo_buffer_size": len(agent.demo_buffer),
                    "online_valid_sequences": int(agent.online_buffer.num_valid_sequences()),
                    "online_ready_steps": int(agent.online_buffer.num_ready_steps()),
                }
                maybe_log(wandb_run, episode_payload, step=step)
                runtime_logger.log(
                    {
                        "event": "episode_end",
                        "step": int(step),
                        "episode_index": int(episode_index),
                        "episode_return": float(episode_return),
                        "episode_length": int(episode_length),
                        "episode_success": bool(success),
                        "is_success": bool(success),
                        "episode_transition_count": int(episode_transition_count),
                        "episode_intervention_transitions": int(episode_intervention_transitions),
                        **event_time_fields(),
                    }
                )
                print(
                    format_episode_line(
                        step=step,
                        episode_index=episode_index,
                        episode_return=episode_return,
                        episode_length=episode_length,
                        success=bool(success),
                        online_buffer_size=len(agent.online_buffer),
                        demo_buffer_size=len(agent.demo_buffer),
                    )
                )
                if episode_pause_sec > 0.0:
                    time.sleep(episode_pause_sec)

                obs, _ = reset_flow_policy_observation(
                    env,
                    preserve_mjviewer=rollout_has_renderer,
                    extractor=flow_proprio_extractor,
                    policy_camera_names=policy_camera_names,
                    camera_aliases=camera_aliases,
                    img_height=int(cfg.env.img_height),
                    img_width=int(cfg.env.img_width),
                )
                reset_viewer_preview()
                policy_worker.reset_policy_state()
                episode_return = 0.0
                episode_length = 0
                episode_step_index = 0
                episode_index += 1
                episode_transition_count = 0
                episode_intervention_transitions = 0
                cached_override_action = None
                cached_is_intervention = False
                policy_gate.force_ready()
                spacemouse_gate.force_ready()
                if intervention_runtime is not None:
                    intervention_runtime.start_episode()
                if discriminator_worker is not None:
                    discriminator_worker.reset(
                        task_name=task_name,
                        initial_state=env.sim.get_state().flatten().copy(),
                        initial_images=_initial_discriminator_images(obs, discriminator_worker.runtime.camera_names),
                        episode_index=episode_index,
                    )
            else:
                obs = next_obs

            if step > 0 and step % int(cfg.logging.checkpoint_interval) == 0:
                request_checkpoint_save(step)
            maybe_report_runtime(step)
    finally:
        if discriminator_worker is not None:
            try:
                finalize_pending_episode(episode_index)
            except Exception:
                pass
        if async_updates:
            trainer.flush_async_updates()
            flushed_metrics = trainer.drain_async_metrics()
            if flushed_metrics:
                maybe_print_publish_events(flushed_metrics)
        maybe_report_runtime(last_step if last_step >= 0 else 0)
        buffer_writer.flush(timeout=120.0)
        checkpoint_writer.request_save(
            paths=[checkpoint_path(checkpoint_dir, "latest")],
            include_buffers=False,
            extra={
                "global_step": int(last_step),
                "episode_index": int(episode_index),
                "success_count": int(success_count),
                "trainer_state": trainer.state_dict(),
            },
            metadata={"step": int(last_step)},
        )
        checkpoint_writer.flush(timeout=300.0)
        runtime_logger.log(
            {
                "event": "run_end",
                "step": int(last_step),
                "episode_index": int(episode_index),
                "success_count": int(success_count),
                "total_transition_count": int(total_transition_count),
                "total_intervention_transitions": int(total_intervention_transitions),
                **event_time_fields(),
            }
        )
        write_run_info(
            checkpoint_dir,
            {
                "run_name": run_name,
                "run_dir": str(checkpoint_dir),
                "started_at": run_started_at,
                "finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
                "env_name": str(cfg.env.environment),
                "robots": [str(name) for name in list(cfg.env.robots)],
                "policy_camera_names": policy_camera_names,
                "render_camera_names": render_camera_names,
                "render_camera": resolve_render_camera(cfg, render_camera_names),
                "loaded_checkpoint": None if loaded_checkpoint is None else str(loaded_checkpoint),
                "initialized_checkpoint": None if initialized_checkpoint is None else str(initialized_checkpoint),
                "resume_enabled": bool(cfg.runtime.resume),
                "load_buffers": bool(cfg.runtime.load_buffers),
                "seed": serialize_seed(getattr(cfg, "seed", None)),
                "console_log": str(console_log_path),
                "runtime_log": str(runtime_log_path),
                "buffer_dir": str(checkpoint_dir / "buffers"),
                "online_chunk_dir": str(checkpoint_dir / "buffers" / "online_chunks"),
                "demo_chunk_dir": str(checkpoint_dir / "buffers" / "demo_chunks"),
                "qv_cache_path": str(qv_cache_path),
                "last_step": int(last_step),
                "episode_index": int(episode_index),
                "success_count": int(success_count),
                "latest_checkpoint": str(checkpoint_path(checkpoint_dir, "latest")),
                "trainer_state": trainer.state_dict(),
            },
        )
        if async_updates:
            trainer.close_async_worker(wait=False)
        if intervention_runtime is not None:
            intervention_runtime.close()
        if discriminator_worker is not None:
            discriminator_worker.close()
        policy_worker.close()
        if wandb_run is not None:
            wandb_run.finish()
        try:
            env.close()
        finally:
            try:
                if viewer_runtime is not None:
                    viewer_runtime.close()
            finally:
                try:
                    if flow_proprio_extractor is not None:
                        flow_proprio_extractor.close()
                finally:
                    try:
                        checkpoint_writer.close()
                    finally:
                        try:
                            runtime_logger.close()
                        finally:
                            try:
                                buffer_writer.close()
                            finally:
                                console_capture.stop()


if __name__ == "__main__":
    main()
