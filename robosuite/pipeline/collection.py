from __future__ import annotations

import datetime
import json
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from robosuite.pipeline.algorithms.discriminator.encoder import SharedDynamicsEncoder
from robosuite.pipeline.algorithms.discriminator.nnpu import FrozenNNPUDiscriminator
from robosuite.pipeline.algorithms.discriminator.runtime import (
    EnterKeyListener,
    build_nnpu_runtime,
    render_nnpu_hud,
)
from robosuite.pipeline.common.environment import (
    RobosuiteInterventionRuntime,
    RobosuiteViewerRuntime,
    build_device,
    build_robosuite_env,
    choose_viewer_backend,
    compute_grasp_penalty,
    sparse_success_reward,
)
from robosuite.pipeline.factory import build_algorithm
from robosuite.pipeline.common.flow import (
    bind_flow_proprio_extractor,
    build_flow_runtime_cfg,
    convert_env_camera_observation,
    load_init_checkpoint_payload,
    merge_checkpoint_model_config,
    maybe_set_seed,
    reset_flow_policy_observation,
    resolve_camera_names,
    resolve_flow_task_metadata,
    serialize_seed,
)
from robosuite.pipeline.utils import (
    EMAFpsTracker,
    EnvRandomReducer,
    FixedRateLimiter,
    IntervalGate,
    maybe_wrap_visualization,
    resolve_render_camera,
    resolve_runtime_fps,
)


_EPISODE_SHARD_FORMAT = "episode_shards_v1"


def _require_cuda_device(device: Any, *, name: str) -> str:
    value = str(device).strip()
    if not torch.cuda.is_available():
        raise RuntimeError(f"{name} requires CUDA, but torch.cuda.is_available() is False.")
    if not value.startswith("cuda"):
        raise ValueError(f"{name} must be a CUDA device, got {value!r}.")
    parsed = torch.device(value)
    if parsed.type != "cuda":
        raise ValueError(f"{name} must be a CUDA device, got {value!r}.")
    if parsed.index is not None and parsed.index >= torch.cuda.device_count():
        raise ValueError(
            f"{name}={value!r} is unavailable; visible CUDA device count is {torch.cuda.device_count()}."
        )
    return value


def _resolve_existing_file(raw: Any, *, what: str) -> Path:
    if raw is None or str(raw).strip().lower() in {"", "none", "null"}:
        raise RuntimeError(f"{what} must be set.")
    path = Path(to_absolute_path(str(raw)))
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}")
    return path


def _resolve_output_path(cfg: DictConfig, task_name: str) -> Path:
    output_dir_raw = OmegaConf.select(cfg, "offline_collect.output_dir", default=None)
    if output_dir_raw is None or str(output_dir_raw).strip() == "":
        data_root = Path(to_absolute_path(str(getattr(cfg.data, "demo_root", "./data"))))
        output_dir = data_root / str(task_name) / "offline_data"
    else:
        output_dir = Path(to_absolute_path(str(output_dir_raw)))

    output_file_raw = str(
        OmegaConf.select(cfg, "offline_collect.output_file", default="offline_episodes.pt")
    )
    output_file = Path(output_file_raw)
    if output_file.is_absolute() or output_file.parent != Path("."):
        return Path(to_absolute_path(output_file_raw))
    return output_dir / output_file


def _copy_obs(obs: dict[str, Any], camera_names: list[str]) -> dict[str, np.ndarray]:
    copied = {"state": np.asarray(obs["state"], dtype=np.float32).copy()}
    for camera_name in camera_names:
        copied[camera_name] = np.asarray(obs[camera_name], dtype=np.uint8).copy()
    return copied


def _stack_obs(frames: list[dict[str, np.ndarray]], camera_names: list[str]) -> dict[str, np.ndarray]:
    return {
        "state": np.stack([np.asarray(frame["state"], dtype=np.float32) for frame in frames], axis=0),
        **{
            camera_name: np.stack(
                [np.asarray(frame[camera_name], dtype=np.uint8) for frame in frames],
                axis=0,
            )
            for camera_name in camera_names
        },
    }


class _EpisodeBuilder:
    def __init__(self, *, episode_index: int, episode_seed: int | None, camera_names: list[str]) -> None:
        self.episode_index = int(episode_index)
        self.episode_seed = None if episode_seed is None else int(episode_seed)
        self.camera_names = list(camera_names)
        self.obs: list[dict[str, np.ndarray]] = []
        self.next_obs: list[dict[str, np.ndarray]] = []
        self.executed_action: list[np.ndarray] = []
        self.policy_action: list[np.ndarray] = []
        self.human_action: list[np.ndarray] = []
        self.is_intervention: list[bool] = []
        self.reward: list[float] = []
        self.done: list[bool] = []
        self.success: list[bool] = []
        self.nnpu_pred: list[int] = []
        self.nnpu_score: list[float] = []
        self.nnpu_threshold: list[float] = []
        self.grasp_penalty: list[float] = []
        self.policy_chunk_start: list[bool] = []
        self.policy_candidate_omegas: list[np.ndarray] = []
        self.policy_candidate_nnpu_scores: list[np.ndarray] = []
        self.policy_selected_candidate_index: list[int] = []
        self.policy_selected_guidance_omega: list[float] = []
        self.policy_context_latency_ms: list[float] = []
        self.policy_ode_latency_ms: list[float] = []
        self.policy_d2h_latency_ms: list[float] = []
        self.policy_total_latency_ms: list[float] = []
        self.discriminator_latency_ms: list[float] = []
        self.selector_total_latency_ms: list[float] = []

    def __len__(self) -> int:
        return len(self.executed_action)

    def append(
        self,
        *,
        obs: dict[str, Any],
        next_obs: dict[str, Any],
        executed_action: np.ndarray,
        policy_action: np.ndarray,
        human_action: np.ndarray,
        is_intervention: bool,
        reward: float,
        done: bool,
        success: bool,
        nnpu_pred: int,
        nnpu_score: float,
        nnpu_threshold: float,
        grasp_penalty: float | None,
        policy_chunk_start: bool,
        policy_candidate_omegas: np.ndarray,
        policy_candidate_nnpu_scores: np.ndarray,
        policy_selected_candidate_index: int,
        policy_selected_guidance_omega: float,
        policy_latency: dict[str, float] | None = None,
    ) -> None:
        self.obs.append(_copy_obs(obs, self.camera_names))
        self.next_obs.append(_copy_obs(next_obs, self.camera_names))
        self.executed_action.append(np.asarray(executed_action, dtype=np.float32).copy())
        self.policy_action.append(np.asarray(policy_action, dtype=np.float32).copy())
        self.human_action.append(np.asarray(human_action, dtype=np.float32).copy())
        self.is_intervention.append(bool(is_intervention))
        self.reward.append(float(reward))
        self.done.append(bool(done))
        self.success.append(bool(success))
        self.nnpu_pred.append(int(nnpu_pred))
        self.nnpu_score.append(float(nnpu_score))
        self.nnpu_threshold.append(float(nnpu_threshold))
        self.grasp_penalty.append(float("nan") if grasp_penalty is None else float(grasp_penalty))
        self.policy_chunk_start.append(bool(policy_chunk_start))
        self.policy_candidate_omegas.append(
            np.asarray(policy_candidate_omegas, dtype=np.float32).copy()
        )
        self.policy_candidate_nnpu_scores.append(
            np.asarray(policy_candidate_nnpu_scores, dtype=np.float32).copy()
        )
        self.policy_selected_candidate_index.append(int(policy_selected_candidate_index))
        self.policy_selected_guidance_omega.append(float(policy_selected_guidance_omega))
        timing = policy_latency or {}
        self.policy_context_latency_ms.append(float(timing.get("context_ms", np.nan)))
        self.policy_ode_latency_ms.append(float(timing.get("ode_ms", np.nan)))
        self.policy_d2h_latency_ms.append(float(timing.get("d2h_ms", np.nan)))
        self.policy_total_latency_ms.append(float(timing.get("policy_total_ms", np.nan)))
        self.discriminator_latency_ms.append(float(timing.get("discriminator_ms", np.nan)))
        self.selector_total_latency_ms.append(float(timing.get("selector_total_ms", np.nan)))

    def mark_terminal(self) -> None:
        if self.done:
            self.done[-1] = True

    def to_payload(self, terminal_reason: str) -> dict[str, Any]:
        n = len(self)
        return {
            "episode_index": int(self.episode_index),
            "episode_seed": self.episode_seed,
            "episode_step": np.arange(n, dtype=np.int32),
            "terminal_reason": str(terminal_reason),
            "obs": _stack_obs(self.obs, self.camera_names),
            "next_obs": _stack_obs(self.next_obs, self.camera_names),
            "executed_action": np.stack(self.executed_action, axis=0).astype(np.float32),
            "policy_action": np.stack(self.policy_action, axis=0).astype(np.float32),
            "human_action": np.stack(self.human_action, axis=0).astype(np.float32),
            "is_intervention": np.asarray(self.is_intervention, dtype=np.bool_),
            "gt_fail": np.asarray(self.is_intervention, dtype=np.bool_),
            "reward": np.asarray(self.reward, dtype=np.float32),
            "done": np.asarray(self.done, dtype=np.bool_),
            "success": np.asarray(self.success, dtype=np.bool_),
            "nnpu_pred": np.asarray(self.nnpu_pred, dtype=np.int8),
            "nnpu_score": np.asarray(self.nnpu_score, dtype=np.float32),
            "nnpu_threshold": np.asarray(self.nnpu_threshold, dtype=np.float32),
            "grasp_penalty": np.asarray(self.grasp_penalty, dtype=np.float32),
            "policy_chunk_start": np.asarray(self.policy_chunk_start, dtype=np.bool_),
            "policy_candidate_omegas": np.stack(
                self.policy_candidate_omegas, axis=0
            ).astype(np.float32),
            "policy_candidate_nnpu_scores": np.stack(
                self.policy_candidate_nnpu_scores, axis=0
            ).astype(np.float32),
            "policy_selected_candidate_index": np.asarray(
                self.policy_selected_candidate_index, dtype=np.int16
            ),
            "policy_selected_guidance_omega": np.asarray(
                self.policy_selected_guidance_omega, dtype=np.float32
            ),
            "policy_context_latency_ms": np.asarray(self.policy_context_latency_ms, dtype=np.float32),
            "policy_ode_latency_ms": np.asarray(self.policy_ode_latency_ms, dtype=np.float32),
            "policy_d2h_latency_ms": np.asarray(self.policy_d2h_latency_ms, dtype=np.float32),
            "policy_total_latency_ms": np.asarray(self.policy_total_latency_ms, dtype=np.float32),
            "discriminator_latency_ms": np.asarray(self.discriminator_latency_ms, dtype=np.float32),
            "selector_total_latency_ms": np.asarray(self.selector_total_latency_ms, dtype=np.float32),
        }


def _terminal_reason(*, success: bool, env_done: bool, max_steps: bool) -> str | None:
    if success:
        return "success"
    if env_done:
        return "env_done"
    if max_steps:
        return "max_steps"
    return None


def _select_action_candidate(
    candidates: np.ndarray,
    omegas: np.ndarray,
    failure_scores: np.ndarray,
) -> tuple[int, np.ndarray, float]:
    candidate_array = np.asarray(candidates, dtype=np.float32)
    omega_array = np.asarray(omegas, dtype=np.float32).reshape(-1)
    score_array = np.asarray(failure_scores, dtype=np.float32).reshape(-1)
    if candidate_array.ndim != 3:
        raise ValueError(
            f"candidates must be (K, H, A), got {candidate_array.shape}."
        )
    if candidate_array.shape[0] != len(omega_array) or len(omega_array) != len(score_array):
        raise ValueError("Candidate, omega, and failure-score counts must match.")
    if not np.isfinite(score_array).all():
        raise RuntimeError("Discriminator returned non-finite candidate scores.")
    selected = int(np.argmin(score_array))
    return selected, candidate_array[selected].copy(), float(omega_array[selected])


def _checkpoint_has_negative_policy(payload: dict[str, Any]) -> bool:
    core = payload.get("core")
    return isinstance(core, dict) and "core_neg" in core


def _validate_resume_guidance(
    checkpoint: dict[str, Any],
    *,
    has_episodes: bool,
    configured: list[float],
    effective: list[float],
) -> None:
    saved_configured = checkpoint.get("configured_guidance_omegas")
    saved_effective = checkpoint.get("effective_guidance_omegas")
    if has_episodes and (saved_configured is None or saved_effective is None):
        raise ValueError(
            "Cannot resume a legacy partial collection without guidance metadata."
        )
    if saved_configured is not None and list(saved_configured) != configured:
        raise ValueError(
            "Cannot resume collection with different configured guidance omegas."
        )
    if saved_effective is not None and list(saved_effective) != effective:
        raise ValueError(
            "Cannot resume collection with different effective guidance omegas."
        )


@dataclass
class _EpisodeStats:
    num_episodes: int = 0
    num_transitions: int = 0
    num_interventions: int = 0
    terminal_reasons: dict[str, int] = field(default_factory=dict)

    def add(self, episode: dict[str, Any]) -> None:
        self.num_episodes += 1
        self.num_transitions += int(len(episode["executed_action"]))
        self.num_interventions += int(
            np.asarray(episode["is_intervention"], dtype=np.bool_).sum()
        )
        reason = str(episode["terminal_reason"])
        self.terminal_reasons[reason] = self.terminal_reasons.get(reason, 0) + 1

    @classmethod
    def from_episodes(cls, episodes: list[dict[str, Any]]) -> "_EpisodeStats":
        stats = cls()
        for episode in episodes:
            stats.add(episode)
        return stats


def _metadata_from_stats(
    *,
    payload_path: Path,
    task_name: str,
    policy_checkpoint: Path,
    nnpu_checkpoint: Path,
    camera_names: list[str],
    img_height: int,
    img_width: int,
    action_dim: int,
    stats: _EpisodeStats,
    started_at: str,
    finished_at: str,
    seed: int | None,
    configured_guidance_omegas: list[float] | None = None,
    effective_guidance_omegas: list[float] | None = None,
    policy_inference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "path": str(payload_path),
        "task_name": str(task_name),
        "policy_checkpoint": str(policy_checkpoint),
        "nnpu_checkpoint": str(nnpu_checkpoint),
        "camera_names": list(camera_names),
        "img_height": int(img_height),
        "img_width": int(img_width),
        "action_dim": int(action_dim),
        "num_episodes": int(stats.num_episodes),
        "num_transitions": int(stats.num_transitions),
        "num_intervention_transitions": int(stats.num_interventions),
        "intervention_ratio": (
            float(stats.num_interventions / stats.num_transitions)
            if stats.num_transitions > 0
            else 0.0
        ),
        "terminal_reasons": dict(stats.terminal_reasons),
        "started_at": str(started_at),
        "finished_at": str(finished_at),
        "seed": seed,
        "configured_guidance_omegas": list(configured_guidance_omegas or []),
        "effective_guidance_omegas": list(effective_guidance_omegas or []),
        "policy_inference": dict(policy_inference or {}),
    }


def _episode_shard_dir(output_path: Path) -> Path:
    return output_path.with_suffix(".shards")


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def _atomic_json_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _save_episode_shard(
    output_path: Path,
    *,
    shard_index: int,
    episode: dict[str, Any],
) -> tuple[Path, float]:
    shard_path = _episode_shard_dir(output_path) / f"episode_{shard_index:06d}.pt"
    started = time.monotonic()
    _atomic_torch_save(
        {
            "checkpoint_format": _EPISODE_SHARD_FORMAT,
            "shard_index": int(shard_index),
            "episode": episode,
        },
        shard_path,
    )
    return shard_path, time.monotonic() - started


def _load_episode_shards(output_path: Path, *, manifest_count: int) -> list[dict[str, Any]]:
    shard_dir = _episode_shard_dir(output_path)
    shard_paths = sorted(shard_dir.glob("episode_*.pt")) if shard_dir.is_dir() else []
    if len(shard_paths) < manifest_count:
        raise RuntimeError(
            f"Episode shard checkpoint is incomplete: manifest records {manifest_count} "
            f"episodes but only {len(shard_paths)} shards exist in {shard_dir}."
        )

    episodes: list[dict[str, Any]] = []
    for shard_index, shard_path in enumerate(shard_paths):
        expected_path = shard_dir / f"episode_{shard_index:06d}.pt"
        if shard_path != expected_path:
            raise RuntimeError(
                f"Episode shards must be contiguous; expected {expected_path}, got {shard_path}."
            )
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        if shard.get("checkpoint_format") != _EPISODE_SHARD_FORMAT:
            raise RuntimeError(f"Unsupported episode shard format in {shard_path}.")
        if int(shard.get("shard_index", -1)) != shard_index:
            raise RuntimeError(f"Episode shard index mismatch in {shard_path}.")
        episode = shard.get("episode")
        if not isinstance(episode, dict):
            raise RuntimeError(f"Episode shard does not contain an episode mapping: {shard_path}.")
        episodes.append(episode)
    return episodes


def _load_collection_checkpoint(
    output_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], bool]:
    shard_dir = _episode_shard_dir(output_path)
    if not output_path.is_file():
        if shard_dir.is_dir():
            raise RuntimeError(
                f"Episode shard directory exists without its manifest: {shard_dir}."
            )
        return {}, [], False

    checkpoint = torch.load(output_path, map_location="cpu", weights_only=False)
    if checkpoint.get("checkpoint_format") == _EPISODE_SHARD_FORMAT:
        manifest_count = int(checkpoint.get("num_episodes", 0))
        if int(checkpoint.get("num_shards", -1)) != manifest_count:
            raise RuntimeError(
                "Episode shard manifest has inconsistent num_shards and num_episodes."
            )
        episodes = _load_episode_shards(
            output_path,
            manifest_count=manifest_count,
        )
        return checkpoint, episodes, False
    if bool(checkpoint.get("final", False)):
        return checkpoint, list(checkpoint.get("episodes", [])), True
    raise RuntimeError(
        "Legacy monolithic partial collection checkpoints cannot be resumed. "
        f"Remove {output_path} and restart the collection round."
    )


def _cleanup_episode_shards(output_path: Path) -> None:
    shard_dir = _episode_shard_dir(output_path)
    try:
        shutil.rmtree(shard_dir)
    except FileNotFoundError:
        pass
    except OSError as exc:
        print(f"[WARN] Failed to remove completed episode shards at {shard_dir}: {exc}")


def _write_collection_manifest(
    output_path: Path,
    *,
    payload: dict[str, Any],
    metadata: dict[str, Any],
    num_shards: int,
) -> None:
    manifest = {
        **payload,
        "checkpoint_format": _EPISODE_SHARD_FORMAT,
        "num_shards": int(num_shards),
    }
    _atomic_torch_save(manifest, output_path)
    _atomic_json_save(metadata, output_path.with_suffix(".meta.json"))


def _finalize_collection_checkpoint(
    output_path: Path,
    *,
    payload: dict[str, Any],
    metadata: dict[str, Any],
    episodes: list[dict[str, Any]],
) -> tuple[float, int]:
    started = time.monotonic()
    _atomic_torch_save({**payload, "episodes": episodes}, output_path)
    _atomic_json_save(metadata, output_path.with_suffix(".meta.json"))
    elapsed = time.monotonic() - started
    output_bytes = output_path.stat().st_size
    _cleanup_episode_shards(output_path)
    return elapsed, output_bytes


def run_collection(cfg: DictConfig) -> None:
    maybe_set_seed(getattr(cfg, "seed", None))
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available():
        raise RuntimeError("Offline collection requires CUDA.")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    init_checkpoint, init_payload = load_init_checkpoint_payload(cfg)
    if init_checkpoint is None:
        raise RuntimeError("runtime.init_checkpoint must point to the fixed policy checkpoint.")
    policy_checkpoint = _resolve_existing_file(init_checkpoint, what="runtime.init_checkpoint")
    nnpu_checkpoint = _resolve_existing_file(
        cfg.algorithm.discriminator.checkpoint,
        what="algorithm.discriminator.checkpoint",
    )

    algorithm_cfg = OmegaConf.to_container(cfg.algorithm, resolve=True)
    if not isinstance(algorithm_cfg, dict):
        raise TypeError("cfg.algorithm must resolve to a dict.")
    flow_cfg = algorithm_cfg.setdefault("flow", {})
    policy_device = _require_cuda_device(flow_cfg.get("device", "cuda:0"), name="algorithm.flow.device")
    inference_raw = flow_cfg.get("inference_device", policy_device)
    if inference_raw is None or str(inference_raw).strip().lower() == "auto":
        inference_raw = policy_device
    inference_device = _require_cuda_device(inference_raw, name="algorithm.flow.inference_device")
    flow_cfg["device"] = policy_device
    flow_cfg["inference_device"] = inference_device

    disc_cfg = cfg.algorithm.discriminator
    disc_cfg.checkpoint = str(nnpu_checkpoint)
    disc_device = _require_cuda_device(
        getattr(disc_cfg.inference, "device", "cuda:0"),
        name="algorithm.discriminator.inference.device",
    )
    disc_cfg.inference.device = disc_device

    task_name = str(cfg.env.environment)
    requested_camera_names = resolve_camera_names(cfg)
    flow_env_metadata = resolve_flow_task_metadata(init_payload, task_name)
    if bool(getattr(cfg.runtime, "use_init_checkpoint_camera_names", True)) and init_payload is not None:
        policy_camera_names = [str(name) for name in init_payload.get("camera_names", [])]
        if len(policy_camera_names) == 0:
            policy_camera_names = list(requested_camera_names)
    else:
        policy_camera_names = list(requested_camera_names)
    if len(policy_camera_names) == 0:
        raise ValueError("Offline collection requires at least one policy camera.")
    cfg.algorithm.camera_names = list(policy_camera_names)
    camera_aliases = {
        str(key): str(value)
        for key, value in dict(getattr(cfg.algorithm.flow, "camera_aliases", {}) or {}).items()
    }

    viewer_enabled = bool(cfg.runtime.interactive) and bool(cfg.runtime.viewer_enabled)
    main_renderer = str(cfg.env.renderer)
    if viewer_enabled and main_renderer != "mjviewer":
        print(f"[WARN] Overriding env.renderer={main_renderer!r} to 'mjviewer' for interactive collection.")
        main_renderer = "mjviewer"

    runtime_cfg = build_flow_runtime_cfg(
        cfg,
        env_metadata=flow_env_metadata,
        camera_names=list(policy_camera_names),
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        renderer=main_renderer,
    )
    env = build_robosuite_env(runtime_cfg)
    extractor = bind_flow_proprio_extractor(env, flow_env_metadata)
    obs, _ = reset_flow_policy_observation(
        env,
        preserve_mjviewer=False,
        extractor=extractor,
        policy_camera_names=policy_camera_names,
        camera_aliases=camera_aliases,
        img_height=int(cfg.env.img_height),
        img_width=int(cfg.env.img_width),
    )

    action_low, action_high = env.action_spec
    action_low = np.asarray(action_low, dtype=np.float32)
    action_high = np.asarray(action_high, dtype=np.float32)

    algorithm_cfg["camera_names"] = list(policy_camera_names)
    algorithm_cfg["task_name"] = task_name
    flow_cfg.setdefault("image_size", int(cfg.env.img_height))
    if bool(getattr(cfg.runtime, "use_init_checkpoint_model", True)) and init_payload is not None:
        merge_checkpoint_model_config(flow_cfg, init_payload)
        if "task_prompt_map" in init_payload:
            flow_cfg["task_prompt_map"] = init_payload["task_prompt_map"]
        if init_payload.get("act_mean") is not None:
            flow_cfg["action_horizon"] = int(np.asarray(init_payload["act_mean"]).shape[0])
            flow_cfg.setdefault("execute_horizon", 1)
    model_cfg = flow_cfg.setdefault("model", {})
    image_encoder_cfg = model_cfg.get("image_encoder", None)
    if isinstance(image_encoder_cfg, dict) and image_encoder_cfg.get("pretrained_path"):
        image_encoder_cfg["pretrained_path"] = to_absolute_path(str(image_encoder_cfg["pretrained_path"]))
    language_encoder_cfg = model_cfg.get("language_encoder", None)
    if isinstance(language_encoder_cfg, dict) and language_encoder_cfg.get("pretrained_name"):
        language_encoder_cfg["pretrained_name"] = to_absolute_path(str(language_encoder_cfg["pretrained_name"]))

    print(f"[collect] task={task_name} policy_device={policy_device} inference_device={inference_device}")
    print(f"[collect] disc_device={disc_device} cameras={policy_camera_names}")
    agent = build_algorithm(
        algorithm_cfg,
        observation_example=obs,
        sample_action=np.zeros_like(action_low, dtype=np.float32),
        action_low=action_low,
        action_high=action_high,
    )
    policy_payload = agent.load_policy_checkpoint(policy_checkpoint, task_name=task_name)
    has_negative_policy = _checkpoint_has_negative_policy(policy_payload)
    agent.core.model_pos.eval()
    agent.core.model_neg.eval()
    agent.core.inference_model_pos.eval()
    agent.core.inference_model_neg.eval()
    print(f"[collect] loaded fixed policy checkpoint={policy_checkpoint}")

    camera_to_view = {
        str(k): str(v)
        for k, v in dict(getattr(disc_cfg, "camera_to_view", {}) or {}).items()
    }
    encoder_ckpt = None
    if disc_cfg.encoder_ckpt is not None and str(disc_cfg.encoder_ckpt).strip().lower() not in ("", "null"):
        encoder_ckpt = to_absolute_path(str(disc_cfg.encoder_ckpt))
        disc_cfg.encoder_ckpt = encoder_ckpt
    nnpu_encoder = SharedDynamicsEncoder(
        nnpu_ckpt_path=str(nnpu_checkpoint),
        encoder_ckpt=encoder_ckpt,
        device=disc_device,
        camera_to_view=camera_to_view,
    )
    nnpu_encoder.bind_policy_cameras(list(agent.camera_names))
    nnpu_discriminator = FrozenNNPUDiscriminator(
        nnpu_ckpt_path=str(nnpu_checkpoint),
        task_name=task_name,
        device=disc_device,
        encoder=nnpu_encoder,
    )
    nnpu_runtime = build_nnpu_runtime(
        disc_cfg,
        policy_camera_names=list(agent.camera_names),
        shared_encoder=nnpu_encoder,
        discriminator=nnpu_discriminator,
    )
    enter_listener = EnterKeyListener()

    intervention_runtime = None
    if bool(cfg.intervention.enabled):
        device = build_device(env, cfg.intervention)
        intervention_runtime = RobosuiteInterventionRuntime(
            env=env,
            device=device,
            goal_update_mode=str(cfg.intervention.goal_update_mode),
        )

    output_path = _resolve_output_path(cfg, task_name)
    num_episodes = int(
        OmegaConf.select(
            cfg,
            "offline_collect.num_episodes",
            default=getattr(cfg.data, "num_trajectories", 1) or 1,
        )
    )
    max_episode_steps = int(
        OmegaConf.select(
            cfg,
            "offline_collect.episode_max_steps",
            default=getattr(cfg.runtime, "eval_episode_max_steps", 300),
        )
    )
    deterministic = bool(OmegaConf.select(cfg, "offline_collect.deterministic", default=False))
    configured_guidance_omegas = [
        float(value)
        for value in OmegaConf.select(
            cfg, "offline_collect.guidance_omegas", default=[0.0]
        )
    ]
    if not configured_guidance_omegas:
        raise ValueError("offline_collect.guidance_omegas must be non-empty.")
    effective_guidance_omegas = (
        configured_guidance_omegas if has_negative_policy else [0.0]
    )
    print(
        f"[collect][policy][candidates] guidance_omegas={effective_guidance_omegas} "
        f"negative_policy={'enabled' if has_negative_policy else 'unavailable'}"
    )
    save_every_episode = bool(OmegaConf.select(cfg, "offline_collect.save_every_episode", default=False))
    save_interval_episodes = max(
        0,
        int(OmegaConf.select(cfg, "offline_collect.save_interval_episodes", default=0)),
    )

    control_fps = resolve_runtime_fps(cfg, "control_fps", float(cfg.env.control_freq))
    render_fps = resolve_runtime_fps(cfg, "render_fps", control_fps)
    policy_fps = resolve_runtime_fps(cfg, "policy_fps", control_fps)
    spacemouse_fps = resolve_runtime_fps(cfg, "spacemouse_fps", control_fps)
    unthrottled_runtime = bool(getattr(cfg.runtime, "unthrottled", False))
    if unthrottled_runtime and bool(cfg.intervention.enabled):
        print("[WARN] runtime.unthrottled=true is incompatible with human intervention. Using throttled mode.")
        unthrottled_runtime = False
    control_limiter = None if unthrottled_runtime else FixedRateLimiter(control_fps)
    policy_gate = IntervalGate(policy_fps)
    spacemouse_gate = IntervalGate(spacemouse_fps)
    env_random_reducer = EnvRandomReducer(serialize_seed(getattr(cfg, "seed", None)))

    viewer_runtime: RobosuiteViewerRuntime | None = None
    if viewer_enabled:
        viewer_requested_backend = str(getattr(cfg.runtime, "viewer_backend", "auto"))
        if viewer_requested_backend.lower() == "auto" and main_renderer == "mjviewer":
            viewer_backend = "mjviewer"
        else:
            viewer_backend = choose_viewer_backend(
                main_renderer,
                policy_camera_names,
                requested_backend=viewer_requested_backend,
            )
        viewer_runtime_cfg = build_flow_runtime_cfg(
            cfg,
            env_metadata=flow_env_metadata,
            camera_names=list(policy_camera_names),
            has_renderer=True,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            renderer=main_renderer,
        )
        viewer_runtime_cfg.render_camera = resolve_render_camera(cfg, list(policy_camera_names))
        viewer_env = build_robosuite_env(viewer_runtime_cfg)
        viewer_env = maybe_wrap_visualization(
            viewer_env,
            enabled=bool(getattr(cfg.runtime, "visualize_gripper_markers", True)),
            label="viewer env",
        )
        viewer_runtime = RobosuiteViewerRuntime(
            viewer_env,
            render_fps=render_fps,
            async_mode=bool(getattr(cfg.runtime, "viewer_async", False)),
            preview_camera=resolve_render_camera(cfg, list(policy_camera_names)),
            backend=viewer_backend,
        )
        viewer_runtime.start()
        viewer_startup_delay = max(0.0, float(getattr(cfg.runtime, "viewer_startup_delay", 0.0)))
        if viewer_startup_delay > 0.0:
            time.sleep(viewer_startup_delay)

    resume_collection = bool(
        OmegaConf.select(cfg, "offline_collect.resume", default=True)
    )
    episodes: list[dict[str, Any]] = []
    started_at = datetime.datetime.now().isoformat(timespec="seconds")
    resumed_final_checkpoint = False
    if resume_collection and output_path.is_file():
        existing, episodes, resumed_final_checkpoint = _load_collection_checkpoint(output_path)
        if str(existing.get("task_name")) != task_name:
            raise ValueError(
                f"Collection task mismatch: file={existing.get('task_name')!r}, runtime={task_name!r}."
            )
        if Path(str(existing.get("policy_checkpoint"))).resolve() != policy_checkpoint:
            raise ValueError("Cannot resume collection with a different policy checkpoint.")
        if Path(str(existing.get("nnpu_checkpoint"))).resolve() != nnpu_checkpoint:
            raise ValueError("Cannot resume collection with a different discriminator checkpoint.")
        _validate_resume_guidance(
            existing,
            has_episodes=bool(episodes),
            configured=configured_guidance_omegas,
            effective=effective_guidance_omegas,
        )
        started_at = str(existing.get("started_at", started_at))
        if len(episodes) > num_episodes:
            raise ValueError(
                f"Partial collection has {len(episodes)} episodes, above target {num_episodes}."
            )
        if resumed_final_checkpoint and len(episodes) != num_episodes:
            raise ValueError(
                f"Final collection has {len(episodes)} episodes, expected {num_episodes}."
            )
        print(f"[collect] resuming with {len(episodes)}/{num_episodes} completed episodes")
    elif resume_collection and _episode_shard_dir(output_path).is_dir():
        _load_collection_checkpoint(output_path)
    print("[collect] preparing online policy inference (compile + warmup)...")
    policy_inference_info = agent.prepare_online_inference(
        obs=obs,
        omegas=effective_guidance_omegas,
        use_negative=has_negative_policy,
    )
    print(
        "[collect] online policy ready "
        f"precision={policy_inference_info['policy_precision']} "
        f"strategy={policy_inference_info['compile_strategy']} "
        f"compile_warmup_seconds={policy_inference_info['compile_warmup_seconds']:.3f}"
    )
    if nnpu_runtime is not None:
        nnpu_runtime.start()
        enter_listener.start()
        print(
            f"[collect] nnPU scorer started device={nnpu_runtime.cfg.device} "
            f"fps={nnpu_runtime.cfg.fps:g} threshold={nnpu_runtime.discriminator.threshold:+.3f}"
        )
    episode_stats = _EpisodeStats.from_episodes(episodes)
    persisted_episode_count = len(episodes)
    cached_policy_action = np.zeros_like(action_low, dtype=np.float32)
    cached_override_action: np.ndarray | None = None
    cached_is_intervention = False
    cached_candidate_omegas = np.asarray(effective_guidance_omegas, dtype=np.float32)
    cached_candidate_scores = np.full(
        (len(effective_guidance_omegas),), np.nan, dtype=np.float32
    )
    cached_selected_candidate_index = 0
    cached_selected_guidance_omega = float(effective_guidance_omegas[0])
    cached_policy_latency = {
        "context_ms": float("nan"),
        "ode_ms": float("nan"),
        "d2h_ms": float("nan"),
        "policy_total_ms": float("nan"),
        "discriminator_ms": float("nan"),
        "selector_total_ms": float("nan"),
    }
    latency_history: dict[str, list[float]] = {
        key: [] for key in cached_policy_latency
    }
    fps_log_interval = max(
        float(getattr(cfg.runtime, "fps_log_interval", 1.0)),
        1e-6,
    )
    control_fps_tracker = EMAFpsTracker()
    policy_fps_tracker = EMAFpsTracker()
    fps_window_started = time.monotonic()
    measured_control_fps: float | None = None
    measured_policy_fps: float | None = None
    global_step = 0
    episode_index = (
        max(int(episode.get("episode_index", -1)) for episode in episodes) + 1
        if episodes
        else 0
    )
    episode_step = 0
    total_intervention_transitions = int(
        sum(
            np.asarray(episode["is_intervention"], dtype=np.bool_).sum()
            for episode in episodes
        )
    )

    def refresh_viewer() -> None:
        if viewer_runtime is None:
            return
        viewer_runtime.publish_from_env(env)
        viewer_runtime.render_if_due()

    def reset_episode(index: int) -> tuple[dict[str, Any], int | None]:
        episode_seed = env_random_reducer.prepare_episode(env, index, seed_global=False)
        reset_obs, _ = reset_flow_policy_observation(
            env,
            preserve_mjviewer=False,
            extractor=extractor,
            policy_camera_names=policy_camera_names,
            camera_aliases=camera_aliases,
            img_height=int(cfg.env.img_height),
            img_width=int(cfg.env.img_width),
        )
        refresh_viewer()
        agent.reset_policy_state()
        policy_gate.force_ready()
        spacemouse_gate.force_ready()
        if intervention_runtime is not None:
            intervention_runtime.start_episode()
        if nnpu_runtime is not None:
            nnpu_runtime.on_episode_reset()
        return reset_obs, episode_seed

    obs, episode_seed = reset_episode(episode_index)
    current = _EpisodeBuilder(
        episode_index=episode_index,
        episode_seed=episode_seed,
        camera_names=list(agent.camera_names),
    )
    print(
        f"[collect] target_episodes={num_episodes} max_episode_steps={max_episode_steps} "
        f"output={output_path}"
    )

    def flush_episode(reason: str) -> None:
        nonlocal current
        if len(current) == 0:
            return
        current.mark_terminal()
        payload = current.to_payload(reason)
        episodes.append(payload)
        episode_stats.add(payload)
        n_interventions = int(np.asarray(payload["is_intervention"], dtype=np.bool_).sum())
        print(
            f"[collect][episode {payload['episode_index']}] "
            f"steps={len(payload['executed_action'])} interventions={n_interventions} reason={reason}"
        )
        should_checkpoint = (
            save_every_episode
            or (save_interval_episodes > 0 and len(episodes) % save_interval_episodes == 0)
        )
        if should_checkpoint:
            save_payload(final=False)

    def save_payload(*, final: bool) -> None:
        nonlocal persisted_episode_count
        for shard_index in range(persisted_episode_count, len(episodes)):
            shard_path, save_seconds = _save_episode_shard(
                output_path,
                shard_index=shard_index,
                episode=episodes[shard_index],
            )
            persisted_episode_count = shard_index + 1
            print(
                f"[collect][checkpoint {shard_index}] seconds={save_seconds:.3f} "
                f"bytes={shard_path.stat().st_size} path={shard_path}"
            )

        finished_at = datetime.datetime.now().isoformat(timespec="seconds")
        metadata = _metadata_from_stats(
            payload_path=output_path,
            task_name=task_name,
            policy_checkpoint=policy_checkpoint,
            nnpu_checkpoint=nnpu_checkpoint,
            camera_names=list(agent.camera_names),
            img_height=int(cfg.env.img_height),
            img_width=int(cfg.env.img_width),
            action_dim=int(action_low.reshape(-1).shape[0]),
            stats=episode_stats,
            started_at=started_at,
            finished_at=finished_at,
            seed=serialize_seed(getattr(cfg, "seed", None)),
            configured_guidance_omegas=configured_guidance_omegas,
            effective_guidance_omegas=effective_guidance_omegas,
            policy_inference=policy_inference_info,
        )
        payload = {
            **metadata,
            "policy_device": policy_device,
            "inference_device": inference_device,
            "disc_device": disc_device,
            "deterministic": bool(deterministic),
            "final": bool(final),
        }
        if final:
            finalization_seconds, output_bytes = _finalize_collection_checkpoint(
                output_path,
                payload=payload,
                metadata=metadata,
                episodes=episodes,
            )
            print(
                f"[collect][finalize] seconds={finalization_seconds:.3f} "
                f"bytes={output_bytes} episodes={len(episodes)}"
            )
            return

        _write_collection_manifest(
            output_path,
            payload=payload,
            metadata=metadata,
            num_shards=persisted_episode_count,
        )

    collection_complete = False
    try:
        if not resumed_final_checkpoint:
            save_payload(final=False)
        while len(episodes) < num_episodes:
            loop_start = time.monotonic() if control_limiter is None else control_limiter.wait()

            while nnpu_runtime is not None and nnpu_runtime.pause_requested():
                if nnpu_runtime.cfg.hud_enabled:
                    render_nnpu_hud(sys.__stdout__, nnpu_runtime.status(), step=global_step, episode_step=episode_step)
                resume_requested = enter_listener.consume()
                if intervention_runtime is not None and not resume_requested:
                    _, sampled_intervention, sampled_reset = intervention_runtime.maybe_override_action(
                        cached_policy_action
                    )
                    resume_requested = bool(sampled_intervention or sampled_reset)
                if resume_requested:
                    nnpu_runtime.resume()
                    print("\n[collect] nnPU pause resumed")
                    break
                refresh_viewer()
                time.sleep(0.02)

            new_policy_chunk = False
            if unthrottled_runtime or policy_gate.ready(loop_start):
                new_policy_chunk = agent.needs_action_chunk()
                if new_policy_chunk:
                    if nnpu_runtime is None:
                        raise RuntimeError(
                            "Online candidate selection requires the nnPU runtime."
                        )
                    selector_started = time.perf_counter()
                    candidates = agent.plan_action_candidates(
                        obs,
                        omegas=configured_guidance_omegas,
                        deterministic=deterministic,
                        use_negative=has_negative_policy,
                    )
                    cached_candidate_omegas = np.asarray(
                        effective_guidance_omegas, dtype=np.float32
                    )
                    discriminator_started = time.perf_counter()
                    score_tensor = nnpu_runtime.score_action_candidates(
                        images_per_view={name: obs[name] for name in agent.camera_names},
                        proprio=obs["state"],
                        action_candidates=candidates,
                    )
                    cached_candidate_scores = (
                        score_tensor.detach().cpu().numpy().astype(np.float32)
                    )
                    discriminator_ms = (
                        time.perf_counter() - discriminator_started
                    ) * 1000.0
                    (
                        cached_selected_candidate_index,
                        selected_action_chunk,
                        cached_selected_guidance_omega,
                    ) = _select_action_candidate(
                        candidates,
                        cached_candidate_omegas,
                        cached_candidate_scores,
                    )
                    agent.install_action_chunk(selected_action_chunk)
                    policy_stats = agent.last_online_inference_stats()
                    cached_policy_latency = {
                        "context_ms": float(policy_stats["context_ms"]),
                        "ode_ms": float(policy_stats["ode_ms"]),
                        "d2h_ms": float(policy_stats["d2h_ms"]),
                        "policy_total_ms": float(policy_stats["total_ms"]),
                        "discriminator_ms": float(discriminator_ms),
                        "selector_total_ms": float(
                            (time.perf_counter() - selector_started) * 1000.0
                        ),
                    }
                    for latency_name, latency_value in cached_policy_latency.items():
                        history = latency_history[latency_name]
                        history.append(latency_value)
                        if len(history) > 100:
                            del history[:-100]
                    policy_fps_tracker.mark()
                    fps_now = time.monotonic()
                    fps_elapsed = fps_now - fps_window_started
                    if fps_elapsed >= fps_log_interval:
                        measured_control_fps = control_fps_tracker.snapshot(fps_elapsed)
                        measured_policy_fps = policy_fps_tracker.snapshot(fps_elapsed)
                        fps_window_started = fps_now
                    score_text = ", ".join(
                        f"w={omega:g}:{score:+.3f}"
                        for omega, score in zip(
                            cached_candidate_omegas, cached_candidate_scores
                        )
                    )
                    print(
                        f"[collect][policy][candidates] candidates=[{score_text}] "
                        f"selected_w={cached_selected_guidance_omega:g} "
                        + " ".join(
                            f"{name}=p50:{np.percentile(values, 50):.1f}/p95:"
                            f"{np.percentile(values, 95):.1f}ms"
                            for name, values in latency_history.items()
                        )
                    )
                    selected_score = float(
                        cached_candidate_scores[cached_selected_candidate_index]
                    )
                    discriminator_threshold = float(
                        nnpu_runtime.discriminator.threshold
                    )
                    discriminator_failed = selected_score >= discriminator_threshold
                    discriminator_label = "FAIL" if discriminator_failed else "SAFE"
                    discriminator_color = (
                        "\033[1;31m" if discriminator_failed else "\033[1;32m"
                    )
                    fps_text = (
                        "control=warming policy=warming"
                        if measured_control_fps is None or measured_policy_fps is None
                        else (
                            f"control={measured_control_fps:.1f} "
                            f"policy={measured_policy_fps:.1f}"
                        )
                    )
                    print(
                        f"[collect][policy] selected_w={cached_selected_guidance_omega:g} "
                        f"disc={discriminator_color}{discriminator_label}\033[0m "
                        f"disc_score={selected_score:+.3f} "
                        f"disc_threshold={discriminator_threshold:+.3f} "
                        f"selector_ms={cached_policy_latency['selector_total_ms']:.1f} "
                        f"fps({fps_text})"
                    )
                cached_policy_action = agent.select_action(obs, deterministic=deterministic)

            env_action = np.asarray(cached_policy_action, dtype=np.float32)
            policy_action_for_step = np.asarray(cached_policy_action, dtype=np.float32).copy()
            human_action = np.zeros_like(env_action, dtype=np.float32)
            is_intervention = False
            reset_requested = False

            if intervention_runtime is not None and (unthrottled_runtime or spacemouse_gate.ready(loop_start)):
                was_intervening = cached_is_intervention
                override_action, sampled_is_intervention, reset_requested = intervention_runtime.maybe_override_action(
                    cached_policy_action
                )
                if reset_requested:
                    cached_override_action = None
                    cached_is_intervention = False
                    agent.reset_policy_state()
                elif sampled_is_intervention:
                    cached_override_action = np.asarray(override_action, dtype=np.float32)
                    cached_is_intervention = True
                    if not was_intervening:
                        agent.notify_intervention()
                        policy_gate.force_ready()
                else:
                    cached_override_action = None
                    cached_is_intervention = False
                    if was_intervening:
                        agent.reset_policy_state()
                        policy_gate.force_ready()

            if reset_requested:
                if not bool(cfg.intervention.device_reset_as_episode_reset):
                    print("[collect] Device reset requested; exiting collection loop.")
                    break
                flush_episode("manual_reset")
                episode_index += 1
                episode_step = 0
                cached_override_action = None
                cached_is_intervention = False
                obs, episode_seed = reset_episode(episode_index)
                current = _EpisodeBuilder(
                    episode_index=episode_index,
                    episode_seed=episode_seed,
                    camera_names=list(agent.camera_names),
                )
                continue

            if cached_is_intervention and cached_override_action is not None:
                env_action = np.asarray(cached_override_action, dtype=np.float32)
                human_action = env_action.copy()
                is_intervention = True

            if nnpu_runtime is not None:
                nnpu_runtime.publish(
                    images_per_view={name: obs[name] for name in agent.camera_names},
                    proprio=obs["state"],
                    executed_action=env_action,
                    planned_chunk=agent.planned_action_chunk(),
                    is_new_chunk=new_policy_chunk,
                )
                if nnpu_runtime.cfg.hud_enabled:
                    render_nnpu_hud(sys.__stdout__, nnpu_runtime.status(), step=global_step, episode_step=episode_step)
                nnpu_status = nnpu_runtime.status()
            else:
                nnpu_status = None

            grasp_penalty = compute_grasp_penalty(env, env_action)
            step_output = env.step(env_action)
            env_done = False
            if len(step_output) == 5:
                raw_next_obs, _, done, truncated, info = step_output
                env_done = bool(done or truncated)
            else:
                raw_next_obs, _, done, info = step_output
                env_done = bool(done)
            if isinstance(info, dict) and grasp_penalty is not None:
                info.setdefault("grasp_penalty", float(grasp_penalty))
            reward, success = sparse_success_reward(env, info if isinstance(info, dict) else None)
            next_obs = convert_env_camera_observation(
                raw_next_obs,
                env=env,
                extractor=extractor,
                policy_camera_names=policy_camera_names,
                camera_aliases=camera_aliases,
                img_height=int(cfg.env.img_height),
                img_width=int(cfg.env.img_width),
            )
            refresh_viewer()

            max_steps_reached = (episode_step + 1) >= max_episode_steps
            reason = _terminal_reason(success=bool(success), env_done=env_done, max_steps=max_steps_reached)
            done = bool(reason is not None)
            current.append(
                obs=obs,
                next_obs=next_obs,
                executed_action=env_action,
                policy_action=policy_action_for_step,
                human_action=human_action,
                is_intervention=is_intervention,
                reward=float(reward),
                done=done,
                success=bool(success),
                nnpu_pred=-1 if nnpu_status is None else int(nnpu_status.pred),
                nnpu_score=float("nan") if nnpu_status is None else float(nnpu_status.score),
                nnpu_threshold=float("nan") if nnpu_status is None else float(nnpu_status.threshold),
                grasp_penalty=grasp_penalty,
                policy_chunk_start=new_policy_chunk,
                policy_candidate_omegas=cached_candidate_omegas,
                policy_candidate_nnpu_scores=cached_candidate_scores,
                policy_selected_candidate_index=cached_selected_candidate_index,
                policy_selected_guidance_omega=cached_selected_guidance_omega,
                policy_latency=cached_policy_latency,
            )
            total_intervention_transitions += int(is_intervention)
            control_fps_tracker.mark()
            global_step += 1
            episode_step += 1

            if done:
                flush_episode(str(reason))
                if len(episodes) >= num_episodes:
                    break
                episode_index += 1
                episode_step = 0
                cached_override_action = None
                cached_is_intervention = False
                obs, episode_seed = reset_episode(episode_index)
                current = _EpisodeBuilder(
                    episode_index=episode_index,
                    episode_seed=episode_seed,
                    camera_names=list(agent.camera_names),
                )
            else:
                obs = next_obs
        collection_complete = len(episodes) >= num_episodes
    finally:
        save_payload(final=collection_complete)
        if nnpu_runtime is not None:
            nnpu_runtime.stop()
        if intervention_runtime is not None:
            intervention_runtime.close()
        try:
            env.close()
        finally:
            try:
                if viewer_runtime is not None:
                    viewer_runtime.close()
            finally:
                try:
                    extractor.close()
                except Exception:
                    pass

    print(
        f"[collect] saved episodes={len(episodes)} transitions="
        f"{sum(len(ep['executed_action']) for ep in episodes)} "
        f"interventions={total_intervention_transitions} -> {output_path}"
    )
