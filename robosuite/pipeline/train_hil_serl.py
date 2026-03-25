from __future__ import annotations

import datetime
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from robosuite.wrappers import VisualizationWrapper

from robosuite.pipeline.factory import build_algorithm
from robosuite.pipeline.algorithms.hil_serl import HILSERLTrainer
from robosuite.pipeline.algorithms.hil_serl.utils import load_demo_paths, resolve_task_demo_paths
from robosuite.pipeline.algorithms.hil_serl.envs import (
    RobosuiteInterventionRuntime,
    RobosuiteObservationAdapter,
    RobosuiteRuntimeConfig,
    RobosuiteViewerRuntime,
    choose_viewer_backend,
    build_device,
    build_robosuite_env,
    load_hdf5_demos_into_transitions,
    make_checkpoint_directory,
    sparse_success_reward,
    snapshot_env_state,
)


def now_readable() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_camera_names(cfg: DictConfig) -> list[str]:
    camera_names = [str(name) for name in list(cfg.env.camera_names or [])]
    if len(camera_names) == 0:
        camera_names = ["agentview"]
    return camera_names


def resolve_render_camera(cfg: DictConfig, camera_names: list[str]) -> str | None:
    render_camera = getattr(cfg.env, "render_camera", None)
    if render_camera is not None:
        return str(render_camera)
    if len(camera_names) > 0:
        return str(camera_names[0])
    return None


def resolve_demo_task_name(cfg: DictConfig) -> str:
    explicit_task_name = cfg.data.task_name
    if explicit_task_name is not None:
        return str(explicit_task_name)

    robots = [str(name) for name in list(cfg.env.robots or [])]
    if len(robots) == 1:
        return f"{robots[0]}{cfg.env.environment}"
    return str(cfg.env.environment)


def resolve_demo_inputs(cfg: DictConfig) -> tuple[str, list[Path], int | None]:
    max_num_trajectories = cfg.data.num_trajectories
    if max_num_trajectories is not None:
        max_num_trajectories = int(max_num_trajectories)
        if max_num_trajectories <= 0:
            raise ValueError("data.num_trajectories must be positive when provided.")

    explicit_demo_paths = [Path(to_absolute_path(str(path))) for path in list(cfg.data.demo_paths or [])]
    if explicit_demo_paths:
        return "explicit_paths", explicit_demo_paths, max_num_trajectories

    task_name = resolve_demo_task_name(cfg)
    demo_paths = resolve_task_demo_paths(
        task_name=task_name,
        data_root=to_absolute_path(str(cfg.data.demo_root)),
        split=str(cfg.data.demo_split),
    )
    return task_name, demo_paths, max_num_trajectories


def maybe_build_wandb(cfg: DictConfig, run_name: str | None = None):
    if not bool(cfg.logging.use_wandb):
        return None
    try:
        import wandb
    except ImportError:
        print("wandb is not installed. Falling back to stdout logging.")
        return None

    os.environ.setdefault("WANDB_MODE", str(cfg.logging.wandb_mode))
    run = wandb.init(
        project=str(cfg.logging.project),
        entity=str(cfg.logging.entity),
        mode=str(cfg.logging.wandb_mode),
        config=OmegaConf.to_container(cfg, resolve=True),
        name=str(cfg.logging.run_name or run_name) if (cfg.logging.run_name is not None or run_name is not None) else None,
        dir=to_absolute_path(str(cfg.logging.output_root)),
    )
    return run


def build_runtime_cfg(cfg: DictConfig, camera_names: list[str], has_renderer: bool) -> RobosuiteRuntimeConfig:
    render_camera = resolve_render_camera(cfg, camera_names)
    return RobosuiteRuntimeConfig(
        env_name=str(cfg.env.environment),
        robots=[str(name) for name in list(cfg.env.robots)],
        env_configuration=str(cfg.env.config) if cfg.env.config is not None else None,
        controller=str(cfg.env.controller) if cfg.env.controller is not None else None,
        controller_configs=None,
        renderer=str(cfg.env.renderer),
        render_camera=render_camera,
        camera_names=tuple(camera_names),
        img_height=int(cfg.env.img_height),
        img_width=int(cfg.env.img_width),
        reward_shaping=False,
        control_freq=int(cfg.env.control_freq),
        has_renderer=bool(has_renderer),
        has_offscreen_renderer=bool(len(camera_names) > 0),
        ignore_done=False,
        use_camera_obs=False,
        proprio_keys=tuple(cfg.env.proprio_keys or []),
        horizon=int(cfg.env.horizon) if cfg.env.horizon is not None else None,
    )


def resolve_viewer_async(cfg: DictConfig) -> bool:
    requested_async = bool(cfg.runtime.viewer_async)
    renderer = str(cfg.env.renderer)
    if requested_async and renderer == "mjviewer":
        print("[INFO] Forcing synchronous GUI rendering because mjviewer must run on the main thread.")
        return False
    return requested_async


def maybe_log(run, payload: dict[str, Any], step: int) -> None:
    if run is None:
        return
    run.log(payload, step=step)


def checkpoint_path(checkpoint_dir: Path, tag: str) -> Path:
    return checkpoint_dir / "checkpoints" / f"{tag}.pt"


def resume_checkpoint_candidates(cfg: DictConfig, checkpoint_dir: Path) -> list[Path]:
    candidates: list[Path] = []

    explicit_checkpoint = cfg.runtime.checkpoint
    if explicit_checkpoint is not None:
        candidates.append(Path(to_absolute_path(str(explicit_checkpoint))))

    latest = checkpoint_path(checkpoint_dir, "latest")
    if bool(cfg.runtime.resume) and latest.exists():
        candidates.append(latest)

    if bool(cfg.runtime.resume):
        step_candidates = sorted(
            (checkpoint_dir / "checkpoints").glob("step_*.pt"),
            reverse=True,
        )
        candidates.extend(step_candidates)

    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen or not candidate.exists():
            continue
        seen.add(resolved)
        deduped.append(candidate)
    return deduped


def find_latest_resumable_run(output_root: Path, env_name: str) -> Path | None:
    env_prefix = f"hil_serl_{env_name}_"
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


def resolve_run_directory(cfg: DictConfig) -> tuple[str, Path]:
    output_root = Path(to_absolute_path(str(cfg.logging.output_root)))
    explicit_run_name = cfg.logging.run_name
    explicit_checkpoint = cfg.runtime.checkpoint

    if explicit_checkpoint is not None:
        checkpoint = Path(to_absolute_path(str(explicit_checkpoint)))
        if checkpoint.exists() and checkpoint.parent.name == "checkpoints":
            run_dir = checkpoint.parent.parent
            make_checkpoint_directory(output_root, run_dir.name)
            return run_dir.name, run_dir

    if explicit_run_name is not None:
        run_name = str(explicit_run_name)
        return run_name, make_checkpoint_directory(output_root, run_name)

    if bool(cfg.runtime.resume):
        resumable_run = find_latest_resumable_run(output_root, str(cfg.env.environment))
        if resumable_run is not None:
            make_checkpoint_directory(output_root, resumable_run.name)
            return resumable_run.name, resumable_run

    run_name = f"hil_serl_{cfg.env.environment}_{now_readable()}"
    return run_name, make_checkpoint_directory(output_root, run_name)


class FixedRateLimiter:
    def __init__(self, fps: float) -> None:
        if fps <= 0.0:
            raise ValueError("FixedRateLimiter requires a positive fps.")
        self.period = 1.0 / float(fps)
        self._next_tick_time: float | None = None

    def wait(self) -> float:
        now = time.monotonic()
        if self._next_tick_time is None:
            self._next_tick_time = now + self.period
            return now
        if now < self._next_tick_time:
            time.sleep(self._next_tick_time - now)
            now = time.monotonic()
        while self._next_tick_time <= now:
            self._next_tick_time += self.period
        return now


class IntervalGate:
    def __init__(self, fps: float) -> None:
        if fps <= 0.0:
            raise ValueError("IntervalGate requires a positive fps.")
        self.period = 1.0 / float(fps)
        self._next_fire_time: float | None = None

    def ready(self, now: float) -> bool:
        if self._next_fire_time is None:
            self._next_fire_time = now + self.period
            return True
        if now < self._next_fire_time:
            return False
        while self._next_fire_time <= now:
            self._next_fire_time += self.period
        return True

    def force_ready(self) -> None:
        self._next_fire_time = None


class EMAFpsTracker:
    def __init__(self, alpha: float = 0.4) -> None:
        self.alpha = float(alpha)
        self._pending_count = 0
        self._ema = 0.0
        self._initialized = False

    def mark(self, count: int = 1) -> None:
        self._pending_count += int(count)

    def snapshot(self, elapsed: float) -> float:
        instantaneous_fps = float(self._pending_count) / max(float(elapsed), 1e-6)
        self._pending_count = 0
        if not self._initialized:
            self._ema = instantaneous_fps
            self._initialized = True
        else:
            self._ema = self.alpha * instantaneous_fps + (1.0 - self.alpha) * self._ema
        return self._ema


def resolve_runtime_fps(cfg: DictConfig, key: str, default_value: float) -> float:
    runtime_value = getattr(cfg.runtime, key, None)
    if runtime_value is None:
        return float(default_value)
    return float(runtime_value)


def resolve_requested_device(requested: Any, *, fallback: str) -> str:
    if requested is None:
        return fallback
    normalized = str(requested).strip()
    if normalized == "":
        return fallback
    if not normalized.startswith("cuda"):
        return normalized
    if not torch.cuda.is_available():
        print(f"[WARN] Requested CUDA device '{normalized}' but CUDA is unavailable. Falling back to cpu.")
        return "cpu"
    if normalized == "cuda":
        return "cuda:0"
    try:
        device_index = int(normalized.split(":", 1)[1])
    except (IndexError, ValueError):
        print(f"[WARN] Invalid CUDA device '{normalized}'. Falling back to {fallback}.")
        return fallback
    if device_index >= torch.cuda.device_count():
        print(
            f"[WARN] Requested CUDA device '{normalized}' but only {torch.cuda.device_count()} visible GPU(s) exist. "
            f"Falling back to {fallback}."
        )
        return fallback
    return normalized


def resolve_algorithm_devices(algorithm_cfg: dict[str, Any]) -> tuple[str, str]:
    sac_cfg = algorithm_cfg.setdefault("sac", {})
    learner_requested = sac_cfg.get("device", "cpu")
    inference_requested = sac_cfg.get("inference_device", None)
    normalized_learner_request = "cpu" if learner_requested is None else str(learner_requested).strip().lower()

    default_learner = "cuda:0" if torch.cuda.is_available() else "cpu"
    learner_device = resolve_requested_device(learner_requested, fallback=default_learner)

    if inference_requested is None or str(inference_requested).lower() == "auto":
        if normalized_learner_request in {"cuda", "cuda:0"} and torch.cuda.device_count() >= 2:
            learner_device = "cuda:1"
            inference_device = "cuda:0"
        else:
            inference_device = learner_device
    else:
        inference_device = resolve_requested_device(inference_requested, fallback=learner_device)

    sac_cfg["device"] = learner_device
    sac_cfg["inference_device"] = inference_device
    return learner_device, inference_device


def maybe_wrap_visualization(env, *, enabled: bool, label: str):
    if not enabled:
        return env
    wrapped_env = VisualizationWrapper(env)
    print(f"[INFO] Enabled robosuite gripper visualization markers on {label}.")
    return wrapped_env


@hydra.main(version_base="1.2", config_path="./config", config_name="train_hil_serl")
def main(cfg: DictConfig) -> None:
    set_seed(int(cfg.seed))
    camera_names = resolve_camera_names(cfg)
    run_name, checkpoint_dir = resolve_run_directory(cfg)
    print(f"Run directory: {checkpoint_dir}")
    print(f"Cameras: {camera_names}")
    print(f"GUI render camera: {resolve_render_camera(cfg, camera_names)}")
    visualize_gripper_markers = bool(getattr(cfg.runtime, "visualize_gripper_markers", True))

    bootstrap_runtime_cfg = build_runtime_cfg(cfg, camera_names=camera_names, has_renderer=False)
    bootstrap_env = build_robosuite_env(bootstrap_runtime_cfg)
    bootstrap_env = maybe_wrap_visualization(
        bootstrap_env,
        enabled=visualize_gripper_markers,
        label="training env",
    )
    bootstrap_adapter = RobosuiteObservationAdapter(
        bootstrap_env,
        camera_names=camera_names,
        img_height=int(cfg.env.img_height),
        img_width=int(cfg.env.img_width),
        proprio_keys=tuple(cfg.env.proprio_keys or []),
        image_obs_fps=float(getattr(cfg.runtime, "image_obs_fps", cfg.runtime.control_fps)),
    )
    initial_obs, _ = bootstrap_adapter.reset()

    action_low, action_high = bootstrap_adapter.action_spec()
    algorithm_cfg = OmegaConf.to_container(cfg.algorithm, resolve=True)
    if isinstance(algorithm_cfg, dict):
        encoder_cfg = algorithm_cfg.get("encoder", None)
        if isinstance(encoder_cfg, dict) and encoder_cfg.get("pretrained_path"):
            encoder_cfg["pretrained_path"] = to_absolute_path(str(encoder_cfg["pretrained_path"]))
        learner_device, inference_device = resolve_algorithm_devices(algorithm_cfg)
        print(f"[INFO] Learner device: {learner_device}")
        print(f"[INFO] Inference device: {inference_device}")
    agent = build_algorithm(
        algorithm_cfg,
        observation_example=initial_obs,
        sample_action=np.zeros_like(action_low, dtype=np.float32),
        action_low=action_low,
        action_high=action_high,
    )
    trainer = HILSERLTrainer(agent)

    demo_source_name, demo_paths, max_num_trajectories = resolve_demo_inputs(cfg)
    if not demo_paths:
        raise FileNotFoundError(
            "HIL-SERL requires offline demos before training starts. "
            f"No demo files were found for '{demo_source_name}'. "
            "Place demos under ./data/<task>/expert or set data.demo_paths explicitly."
        )

    env = bootstrap_env
    adapter = bootstrap_adapter
    obs = initial_obs
    control_fps = resolve_runtime_fps(cfg, "control_fps", float(cfg.env.control_freq))
    render_fps = resolve_runtime_fps(cfg, "render_fps", float(getattr(cfg.runtime, "viewer_fps", control_fps)))
    policy_fps = resolve_runtime_fps(cfg, "policy_fps", control_fps)
    spacemouse_fps = resolve_runtime_fps(cfg, "spacemouse_fps", control_fps)
    image_obs_fps = resolve_runtime_fps(cfg, "image_obs_fps", control_fps)
    fps_log_interval = max(0.1, float(getattr(cfg.runtime, "fps_log_interval", 1.0)))
    viewer_backend_request = str(getattr(cfg.runtime, "viewer_backend", "auto"))
    viewer_startup_delay = max(0.0, float(getattr(cfg.runtime, "viewer_startup_delay", 0.0)))
    viewer_reset_warmup_frames = max(0, int(getattr(cfg.runtime, "viewer_reset_warmup_frames", 2)))
    async_updates = bool(getattr(cfg.runtime, "async_updates", False))
    if abs(control_fps - float(cfg.env.control_freq)) > 1e-6:
        print(
            "[WARN] runtime.control_fps does not match env.control_freq. "
            f"Actor loop will target {control_fps:.2f}Hz while the env is configured for {int(cfg.env.control_freq)}Hz."
        )
    if image_obs_fps < control_fps:
        print(
            "[INFO] Image observations are rate-limited below control frequency "
            f"({image_obs_fps:.2f}Hz images vs {control_fps:.2f}Hz control)."
        )

    viewer_runtime = None
    viewer_backend = "disabled"
    if bool(cfg.runtime.interactive) and bool(cfg.runtime.viewer_enabled):
        viewer_backend = choose_viewer_backend(
            str(cfg.env.renderer),
            camera_names,
            requested_backend=viewer_backend_request,
        )
        viewer_runtime_cfg = build_runtime_cfg(cfg, camera_names=camera_names, has_renderer=True)
        viewer_runtime_cfg.has_offscreen_renderer = True
        viewer_runtime_cfg.renderer = "mujoco" if viewer_backend == "opencv" else viewer_runtime_cfg.renderer
        viewer_env = build_robosuite_env(viewer_runtime_cfg)
        viewer_env = maybe_wrap_visualization(
            viewer_env,
            enabled=visualize_gripper_markers,
            label=f"{viewer_backend} preview env",
        )
        viewer_env.reset()
        viewer_async = resolve_viewer_async(cfg)
        if viewer_backend != "mjviewer":
            viewer_async = False
        viewer_runtime = RobosuiteViewerRuntime(
            viewer_env,
            render_fps=render_fps,
            async_mode=viewer_async,
            preview_camera=resolve_render_camera(cfg, camera_names),
            backend=viewer_backend,
        )
        viewer_runtime.publish(snapshot_env_state(env))
        viewer_runtime.start()
        if viewer_backend == "opencv":
            print(
                "[INFO] Using OpenCV preview window instead of mjviewer because "
                "mjviewer passive mode can segfault when mixed with offscreen observation rendering."
            )
        elif viewer_startup_delay > 0.0:
            print(f"[INFO] Waiting {viewer_startup_delay:.3f}s after mjviewer launch.")
            time.sleep(viewer_startup_delay)
        print(
            "[INFO] GUI viewer decoupled from control env "
            f"(backend={viewer_backend}, async={viewer_async}, fps={render_fps:.1f})"
        )

    # set resume path
    loaded_checkpoint = None
    extra: dict[str, Any] = {}
    for candidate in resume_checkpoint_candidates(cfg, checkpoint_dir):
        try:
            extra = agent.load_checkpoint(candidate, load_buffers=bool(cfg.runtime.load_buffers))
            loaded_checkpoint = candidate
            print(f"Loaded checkpoint: {candidate}")
            break
        except Exception as exc:
            print(f"[WARN] Skipping invalid checkpoint {candidate}: {exc}")

    if loaded_checkpoint is not None:
        start_step = int(extra.get("global_step", -1)) + 1
        episode_index = int(extra.get("episode_index", 0))
    else:
        start_step = 0
        episode_index = 0

    # load offline demos from `./data/<task>/expert`
    print(f"[INFO] Loading offline demos from {demo_source_name}...")
    proprio_keys = [str(key) for key in list(cfg.env.proprio_keys or [])]
    cache_key_parts = [
        f"h{int(cfg.env.img_height)}",
        f"w{int(cfg.env.img_width)}",
        f"cams-{'_'.join(camera_names)}",
        f"state-{'_'.join(proprio_keys) if proprio_keys else 'auto'}",
    ]
    transitions = load_demo_paths(
        demo_paths,
        cache_dir=checkpoint_dir / "demo_cache",
        hdf5_loader=lambda path, demo_names=None: load_hdf5_demos_into_transitions(
            path,
            camera_names=camera_names,
            img_height=int(cfg.env.img_height),
            img_width=int(cfg.env.img_width),
            proprio_keys=tuple(cfg.env.proprio_keys or []),
            renderer=str(cfg.env.renderer),
            control_freq=int(cfg.env.control_freq),
            demo_names=demo_names,
        ),
        max_num_trajectories=max_num_trajectories,
        cache_key="__".join(cache_key_parts),
    )
    if len(transitions) == 0:
        raise RuntimeError(
            "Offline demo loading completed but produced zero transitions. "
            "Check that the expert files contain valid trajectories before starting training."
        )

    if not agent.demo_buffer.is_compatible(transitions[0].obs, transitions[0].action):
        print("[WARN] Loaded checkpoint demo_buffer is incompatible with current observation shape. Rebuilding demo buffer.")
        agent.demo_buffer.clear()

    if len(agent.demo_buffer) == 0:
        trainer.bootstrap_demo_buffer(transitions, demo_source="offline_demo")
        if max_num_trajectories is None:
            print(f"[INFO] Loaded offline demo transitions: {len(transitions)}")
        else:
            print(f"[INFO] Loaded offline demo transitions: {len(transitions)} from up to {max_num_trajectories} trajectories")
    else:
        print(f"[INFO] Reusing checkpoint demo buffer with {len(agent.demo_buffer)} transitions")

    wandb_run = maybe_build_wandb(cfg, run_name=run_name)
    device = None
    intervention_runtime = None
    if bool(cfg.intervention.enabled):
        device = build_device(env, cfg.intervention)
        intervention_runtime = RobosuiteInterventionRuntime(
            env=env,
            device=device,
            goal_update_mode=str(cfg.intervention.goal_update_mode),
        )
        intervention_runtime.start_episode()
    if async_updates:
        trainer.start_async_worker()

    episode_return = 0.0
    episode_length = 0
    success_count = 0
    last_step = start_step - 1
    control_limiter = FixedRateLimiter(control_fps)
    policy_gate = IntervalGate(policy_fps)
    spacemouse_gate = IntervalGate(spacemouse_fps)
    render_gate = IntervalGate(render_fps)
    fps_trackers = {
        "control": EMAFpsTracker(),
        "policy": EMAFpsTracker(),
        "spacemouse": EMAFpsTracker(),
        "render": EMAFpsTracker(),
        "update": EMAFpsTracker(),
    }
    last_fps_log_time = time.monotonic()
    cached_policy_action = np.zeros_like(action_low, dtype=np.float32)
    cached_override_action: np.ndarray | None = None
    cached_is_intervention = False

    def publish_viewer_snapshot() -> None:
        if viewer_runtime is None:
            return
        viewer_runtime.publish(snapshot_env_state(env))

    def reset_viewer_preview() -> None:
        if viewer_runtime is None:
            return
        snapshot = snapshot_env_state(env)
        viewer_runtime.reset_preview(snapshot, warmup_frames=viewer_reset_warmup_frames)

    def render_viewer_if_due(now: float | None = None) -> None:
        if viewer_runtime is None or viewer_runtime.async_mode:
            return
        current_time = time.monotonic() if now is None else float(now)
        if render_gate.ready(current_time):
            viewer_runtime.render_if_due()

    def collect_viewer_render_stats() -> None:
        if viewer_runtime is None:
            return
        render_count = viewer_runtime.consume_render_count()
        if render_count > 0:
            fps_trackers["render"].mark(render_count)

    def maybe_report_runtime(step: int) -> None:
        nonlocal last_fps_log_time
        now = time.monotonic()
        elapsed = now - last_fps_log_time
        if elapsed < fps_log_interval:
            return
        collect_viewer_render_stats()
        runtime_payload = {
            "control_fps": fps_trackers["control"].snapshot(elapsed),
            "policy_fps": fps_trackers["policy"].snapshot(elapsed),
            "spacemouse_fps": fps_trackers["spacemouse"].snapshot(elapsed),
            "render_fps": fps_trackers["render"].snapshot(elapsed),
            "update_fps": fps_trackers["update"].snapshot(elapsed),
        }
        if async_updates:
            runtime_payload["pending_updates"] = float(trainer.pending_async_updates())
        maybe_log(wandb_run, runtime_payload, step=step)
        print(f"[step {step}] runtime: {json.dumps(runtime_payload, sort_keys=True)}")
        last_fps_log_time = now

    try:
        for step in range(start_step, int(cfg.runtime.max_steps)):
            last_step = step
            loop_start = control_limiter.wait()
            fps_trackers["control"].mark()

            if policy_gate.ready(loop_start):
                if step < int(cfg.algorithm.trainer.random_steps):
                    cached_policy_action = np.random.uniform(action_low, action_high).astype(np.float32)
                else:
                    cached_policy_action = agent.select_action(obs, deterministic=bool(cfg.runtime.eval_deterministic))
                fps_trackers["policy"].mark()

            env_action = np.asarray(cached_policy_action, dtype=np.float32)
            is_intervention = False
            reset_requested = False
            if intervention_runtime is not None and spacemouse_gate.ready(loop_start):
                # Keep device polling on its own cadence and reuse the latest override between polls.
                override_action, sampled_is_intervention, reset_requested = intervention_runtime.maybe_override_action(
                    cached_policy_action
                )
                fps_trackers["spacemouse"].mark()
                if reset_requested:
                    cached_override_action = None
                    cached_is_intervention = False
                elif sampled_is_intervention:
                    cached_override_action = np.asarray(override_action, dtype=np.float32)
                    cached_is_intervention = True
                else:
                    cached_override_action = None
                    cached_is_intervention = False

            if reset_requested:
                if not bool(cfg.intervention.device_reset_as_episode_reset):
                    print("[INFO] Device reset requested. Exiting training loop.")
                    break
                obs, _ = adapter.reset()
                reset_viewer_preview()
                render_gate.force_ready()
                render_viewer_if_due()
                collect_viewer_render_stats()
                episode_return = 0.0
                episode_length = 0
                episode_index += 1
                policy_gate.force_ready()
                spacemouse_gate.force_ready()
                if intervention_runtime is not None:
                    intervention_runtime.start_episode()
                maybe_report_runtime(step)
                continue

            if cached_is_intervention and cached_override_action is not None:
                env_action = np.asarray(cached_override_action, dtype=np.float32)
                is_intervention = True

            # collect current transition
            step_output = env.step(env_action)
            if len(step_output) == 5:
                raw_next_obs, _, done, truncated, info = step_output
                done = bool(done or truncated)
            else:
                raw_next_obs, _, done, info = step_output
            reward, success = sparse_success_reward(env, info if isinstance(info, dict) else None)
            next_obs = adapter.transform(raw_next_obs)
            done = bool(done or success)
            trainer.record_transition(
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
            )
            episode_return += reward
            episode_length += 1
            success_count += int(success)
            publish_viewer_snapshot()
            render_viewer_if_due()
            collect_viewer_render_stats()

            # update when necessary
            if async_updates:
                update_metrics_list = trainer.maybe_update_async()
            else:
                update_metrics_list = trainer.maybe_update()
            if update_metrics_list:
                fps_trackers["update"].mark(len(update_metrics_list))
                collect_viewer_render_stats()
                update_metrics = update_metrics_list[-1]
                if step % int(cfg.logging.log_interval) == 0:
                    maybe_log(wandb_run, update_metrics, step=step)
                    print(f"[step {step}] train: {json.dumps(update_metrics, sort_keys=True)}")

            if done:
                episode_payload = {
                    "episode_return": float(episode_return),
                    "episode_length": int(episode_length),
                    "episode_success": int(success),
                    "online_buffer_size": len(agent.online_buffer),
                    "demo_buffer_size": len(agent.demo_buffer),
                }
                maybe_log(wandb_run, episode_payload, step=step)
                print(f"[step {step}] episode: {json.dumps(episode_payload, sort_keys=True)}")
                obs, _ = adapter.reset()
                reset_viewer_preview()
                render_gate.force_ready()
                render_viewer_if_due()
                collect_viewer_render_stats()
                episode_return = 0.0
                episode_length = 0
                episode_index += 1
                cached_override_action = None
                cached_is_intervention = False
                policy_gate.force_ready()
                spacemouse_gate.force_ready()
                if intervention_runtime is not None:
                    intervention_runtime.start_episode()
            else:
                obs = next_obs

            if step > 0 and step % int(cfg.logging.checkpoint_interval) == 0:
                if async_updates:
                    trainer.flush_async_updates()
                    flushed_metrics = trainer.drain_async_metrics()
                    if flushed_metrics:
                        fps_trackers["update"].mark(len(flushed_metrics))
                        latest_metrics = flushed_metrics[-1]
                        maybe_log(wandb_run, latest_metrics, step=step)
                        print(f"[step {step}] train: {json.dumps(latest_metrics, sort_keys=True)}")
                agent.save_checkpoint(
                    checkpoint_path(checkpoint_dir, f"step_{step:08d}"),
                    extra={"global_step": step, "episode_index": episode_index, "success_count": success_count},
                )
                agent.save_checkpoint(
                    checkpoint_path(checkpoint_dir, "latest"),
                    extra={"global_step": step, "episode_index": episode_index, "success_count": success_count},
                )
                print(f"[INFO] Saved checkpoint at step {step}")
            maybe_report_runtime(step)
    finally:
        if async_updates:
            trainer.flush_async_updates()
            flushed_metrics = trainer.drain_async_metrics()
            if flushed_metrics:
                fps_trackers["update"].mark(len(flushed_metrics))
        collect_viewer_render_stats()
        maybe_report_runtime(last_step if last_step >= 0 else 0)
        agent.save_checkpoint(
            checkpoint_path(checkpoint_dir, "latest"),
            extra={"global_step": last_step, "episode_index": episode_index, "success_count": success_count},
        )
        if async_updates:
            trainer.close_async_worker(wait=False)
        if intervention_runtime is not None:
            intervention_runtime.close()
        if wandb_run is not None:
            wandb_run.finish()
        if viewer_runtime is not None:
            viewer_runtime.close()
        env.close()


if __name__ == "__main__":
    main()
