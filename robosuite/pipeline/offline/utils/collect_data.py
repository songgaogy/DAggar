from __future__ import annotations

import datetime
import json
import sys
import time
from pathlib import Path
from typing import Any

import hydra
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
from robosuite.pipeline.envs import (
    RobosuiteInterventionRuntime,
    RobosuiteViewerRuntime,
    build_device,
    build_robosuite_env,
    choose_viewer_backend,
    compute_grasp_penalty,
    sparse_success_reward,
)
from robosuite.pipeline.factory import build_algorithm
from robosuite.pipeline.train_dipole import (
    bind_flow_proprio_extractor,
    build_flow_runtime_cfg,
    convert_env_camera_observation,
    load_init_checkpoint_payload,
    maybe_set_seed,
    reset_flow_policy_observation,
    resolve_camera_names,
    resolve_flow_task_metadata,
    serialize_seed,
)
from robosuite.pipeline.utils import (
    EnvRandomReducer,
    FixedRateLimiter,
    IntervalGate,
    maybe_wrap_visualization,
    resolve_render_camera,
    resolve_runtime_fps,
)


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
        }


def _terminal_reason(*, success: bool, env_done: bool, max_steps: bool) -> str | None:
    if success:
        return "success"
    if env_done:
        return "env_done"
    if max_steps:
        return "max_steps"
    return None


def _metadata_from_episodes(
    *,
    payload_path: Path,
    task_name: str,
    policy_checkpoint: Path,
    nnpu_checkpoint: Path,
    camera_names: list[str],
    img_height: int,
    img_width: int,
    action_dim: int,
    episodes: list[dict[str, Any]],
    started_at: str,
    finished_at: str,
    seed: int | None,
) -> dict[str, Any]:
    num_transitions = int(sum(len(ep["executed_action"]) for ep in episodes))
    num_interventions = int(sum(np.asarray(ep["is_intervention"], dtype=np.bool_).sum() for ep in episodes))
    terminal_reasons: dict[str, int] = {}
    for episode in episodes:
        reason = str(episode["terminal_reason"])
        terminal_reasons[reason] = terminal_reasons.get(reason, 0) + 1
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
        "num_episodes": int(len(episodes)),
        "num_transitions": int(num_transitions),
        "num_intervention_transitions": int(num_interventions),
        "intervention_ratio": float(num_interventions / num_transitions) if num_transitions > 0 else 0.0,
        "terminal_reasons": terminal_reasons,
        "started_at": str(started_at),
        "finished_at": str(finished_at),
        "seed": seed,
    }


@hydra.main(version_base="1.2", config_path="../../config", config_name="train_dipole")
def main(cfg: DictConfig) -> None:
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
        if "model_cfg" in init_payload:
            flow_cfg["model"] = init_payload["model_cfg"]
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
    agent.load_flow_policy_checkpoint(policy_checkpoint, task_name=task_name)
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
    if nnpu_runtime is not None:
        nnpu_runtime.start()
        enter_listener.start()
        print(
            f"[collect] nnPU scorer started device={nnpu_runtime.cfg.device} "
            f"fps={nnpu_runtime.cfg.fps:g} threshold={nnpu_runtime.discriminator.threshold:+.3f}"
        )

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

    episodes: list[dict[str, Any]] = []
    started_at = datetime.datetime.now().isoformat(timespec="seconds")
    cached_policy_action = np.zeros_like(action_low, dtype=np.float32)
    cached_override_action: np.ndarray | None = None
    cached_is_intervention = False
    global_step = 0
    episode_index = 0
    episode_step = 0
    total_intervention_transitions = 0

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
        finished_at = datetime.datetime.now().isoformat(timespec="seconds")
        metadata = _metadata_from_episodes(
            payload_path=output_path,
            task_name=task_name,
            policy_checkpoint=policy_checkpoint,
            nnpu_checkpoint=nnpu_checkpoint,
            camera_names=list(agent.camera_names),
            img_height=int(cfg.env.img_height),
            img_width=int(cfg.env.img_width),
            action_dim=int(action_low.reshape(-1).shape[0]),
            episodes=episodes,
            started_at=started_at,
            finished_at=finished_at,
            seed=serialize_seed(getattr(cfg, "seed", None)),
        )
        payload = {
            **metadata,
            "policy_device": policy_device,
            "inference_device": inference_device,
            "disc_device": disc_device,
            "deterministic": bool(deterministic),
            "final": bool(final),
            "episodes": episodes,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output_path.with_name(f".{output_path.name}.tmp")
        torch.save(payload, tmp_path)
        tmp_path.replace(output_path)
        meta_path = output_path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(metadata, indent=2))

    try:
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
            )
            total_intervention_transitions += int(is_intervention)
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
    finally:
        if len(current) > 0 and len(episodes) < num_episodes:
            flush_episode("interrupted")
        save_payload(final=True)
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


if __name__ == "__main__":
    main()
