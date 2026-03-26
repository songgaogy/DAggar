from __future__ import annotations

import datetime
import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from robosuite.pipeline.algorithms.hg_dagger import HGDaggerTrainer
from robosuite.pipeline.factory import build_algorithm
from robosuite.pipeline.algorithms.hil_serl import (
    RobosuiteInterventionRuntime,
    RobosuiteObservationAdapter,
    load_demo_paths,
    load_hdf5_demos_into_transitions,
    build_robosuite_env,
)
from robosuite.pipeline.algorithms.hil_serl.envs import (
    build_device,
    compute_grasp_penalty,
    make_checkpoint_directory,
    sparse_success_reward,
)
from robosuite.pipeline.train_hil_serl import (
    AsyncCheckpointWriter,
    AsyncTransitionChunkWriter,
    ConsoleLogCapture,
    EMAFpsTracker,
    FixedRateLimiter,
    IntervalGate,
    JsonlEventLogger,
    build_runtime_cfg,
    checkpoint_path,
    checkpoint_step_path,
    format_episode_line,
    load_transition_chunks,
    maybe_build_wandb,
    maybe_log,
    maybe_wrap_visualization,
    reset_observation_adapter,
    resolve_buffer_chunk_dirs,
    resolve_buffer_snapshot_paths,
    resolve_camera_names,
    resolve_checkpoint_reference,
    resolve_checkpoint_run_dir,
    resolve_demo_inputs,
    resolve_render_camera,
    resolve_requested_device,
    resolve_runtime_fps,
    set_seed,
    write_resolved_config,
    write_run_info,
)


def now_readable() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def resolve_run_directory(cfg: DictConfig) -> tuple[str, Path]:
    output_root = Path(to_absolute_path(str(cfg.logging.output_root)))
    explicit_run_name = cfg.logging.run_name

    if explicit_run_name is not None:
        run_name = str(explicit_run_name)
        return run_name, make_checkpoint_directory(output_root, run_name)

    run_name = f"hg_dagger_{cfg.env.environment}_{now_readable()}"
    return run_name, make_checkpoint_directory(output_root, run_name)


def find_latest_resumable_run(output_root: Path, env_name: str) -> Path | None:
    env_prefix = f"hg_dagger_{env_name}_"
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
            candidates.extend(sorted((resumable_run / "checkpoints").glob("pretrain_*.pt"), reverse=True))

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
    bc_cfg = algorithm_cfg.setdefault("bc", {})
    learner_requested = bc_cfg.get("device", "cpu")
    inference_requested = bc_cfg.get("inference_device", None)
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

    bc_cfg["device"] = learner_device
    bc_cfg["inference_device"] = inference_device
    return learner_device, inference_device


def format_runtime_line(
    *,
    step: int,
    episode_index: int,
    overall_fps: float,
    learner_progress: dict[str, int],
    pending_updates: int,
) -> str:
    return (
        f"[runtime] step={step} ep={episode_index} fps={overall_fps:5.1f} "
        f"bc_updates={learner_progress['actor_updates']} "
        f"next_publish_in={learner_progress['updates_until_publish']} pending={pending_updates}"
    )


def format_train_line(step: int, metrics: dict[str, float], pending_updates: int) -> str:
    return (
        f"[train] step={step} "
        f"actor_loss={metrics.get('actor_loss', float('nan')):.4f} "
        f"mse={metrics.get('mse', float('nan')):.4f} "
        f"log_prob={metrics.get('log_prob', float('nan')):.4f} "
        f"bc_updates={int(metrics.get('learner_actor_updates', 0.0))} "
        f"next_publish_in={int(metrics.get('learner_updates_until_publish', 0.0))} pending={pending_updates}"
    )


def format_publish_line(metrics: dict[str, float]) -> str:
    return (
        f"[publish] policy #{int(metrics.get('learner_publish_count', 0.0))} synced "
        f"at learner_update={int(metrics.get('learner_last_published_update', 0.0))}"
    )


@hydra.main(version_base="1.2", config_path="./config", config_name="train_hg_dagger")
def main(cfg: DictConfig) -> None:
    set_seed(int(cfg.seed))
    camera_names = resolve_camera_names(cfg)
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
    print(f"[env] task={cfg.env.environment} robots={list(cfg.env.robots)} cameras={camera_names}")
    print(f"[view] render_camera={resolve_render_camera(cfg, camera_names)}")
    visualize_gripper_markers = bool(getattr(cfg.runtime, "visualize_gripper_markers", True))

    main_has_renderer = bool(cfg.runtime.interactive) and bool(cfg.runtime.viewer_enabled)
    main_renderer = str(cfg.env.renderer)
    if main_has_renderer and main_renderer != "mjviewer":
        print(
            f"[WARN] Overriding env.renderer='{main_renderer}' to 'mjviewer' so the main training env uses "
            "robosuite's native window renderer."
        )
        main_renderer = "mjviewer"

    main_runtime_cfg = build_runtime_cfg(
        cfg,
        camera_names=camera_names,
        has_renderer=main_has_renderer,
        has_offscreen_renderer=False,
        renderer=main_renderer,
    )
    main_env = build_robosuite_env(main_runtime_cfg)
    main_env = maybe_wrap_visualization(
        main_env,
        enabled=visualize_gripper_markers,
        label="training env",
    )

    obs_runtime_cfg = build_runtime_cfg(
        cfg,
        camera_names=camera_names,
        has_renderer=False,
        has_offscreen_renderer=bool(len(camera_names) > 0),
    )
    obs_render_env = build_robosuite_env(obs_runtime_cfg)
    obs_render_env = maybe_wrap_visualization(
        obs_render_env,
        enabled=visualize_gripper_markers,
        label="observation render env",
    )
    obs_render_env.reset()

    adapter = RobosuiteObservationAdapter(
        main_env,
        render_env=obs_render_env,
        camera_names=camera_names,
        img_height=int(cfg.env.img_height),
        img_width=int(cfg.env.img_width),
        proprio_keys=tuple(cfg.env.proprio_keys or []),
        image_obs_fps=float(getattr(cfg.runtime, "image_obs_fps", cfg.runtime.control_fps)),
    )
    initial_obs, _ = reset_observation_adapter(adapter, preserve_mjviewer=main_has_renderer)
    if main_has_renderer and getattr(main_env, "viewer", None) is not None and hasattr(main_env.viewer, "update"):
        main_env.viewer.update()

    action_low, action_high = adapter.action_spec()
    algorithm_cfg = OmegaConf.to_container(cfg.algorithm, resolve=True)
    if isinstance(algorithm_cfg, dict):
        encoder_cfg = algorithm_cfg.get("encoder", None)
        if isinstance(encoder_cfg, dict) and encoder_cfg.get("pretrained_path"):
            encoder_cfg["pretrained_path"] = to_absolute_path(str(encoder_cfg["pretrained_path"]))
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
    trainer = HGDaggerTrainer(agent)

    demo_source_name, demo_paths, max_num_trajectories = resolve_demo_inputs(cfg)
    if not demo_paths:
        raise FileNotFoundError(
            "HG-DAgger requires offline demos before training starts. "
            f"No demo files were found for '{demo_source_name}'. "
            "Place demos under ./data/<task>/expert or set data.demo_paths explicitly."
        )

    env = main_env
    obs = initial_obs
    control_fps = resolve_runtime_fps(cfg, "control_fps", float(cfg.env.control_freq))
    policy_fps = resolve_runtime_fps(cfg, "policy_fps", control_fps)
    spacemouse_fps = resolve_runtime_fps(cfg, "spacemouse_fps", control_fps)
    image_obs_fps = resolve_runtime_fps(cfg, "image_obs_fps", control_fps)
    fps_log_interval = max(0.1, float(getattr(cfg.runtime, "fps_log_interval", 1.0)))
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

    if main_has_renderer:
        print(
            "[INFO] Main training env owns the robosuite mjviewer window; "
            "image observations are rendered from a separate headless env."
        )

    loaded_checkpoint = None
    extra: dict[str, Any] = {}
    for candidate in resume_checkpoint_candidates(cfg, output_root):
        try:
            extra = agent.load_checkpoint(candidate, load_buffers=bool(cfg.runtime.load_buffers))
            loaded_checkpoint = candidate
            print(f"[load] checkpoint={candidate}")
            break
        except Exception as exc:
            print(f"[WARN] Skipping invalid checkpoint {candidate}: {exc}")

    trainer.load_state_dict(extra.get("trainer_state"))
    if bool(cfg.runtime.load_buffers):
        online_buffer_path, demo_buffer_path, _ = resolve_buffer_snapshot_paths(loaded_checkpoint)
        if online_buffer_path is not None and demo_buffer_path is not None:
            agent.online_buffer.load(online_buffer_path)
            agent.demo_buffer.load(demo_buffer_path)
            print(f"[load] buffers={online_buffer_path.parent}")
        elif loaded_checkpoint is not None:
            print("[load] no legacy buffer snapshot found, checking chunk buffers after demo bootstrap")

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
            "camera_names": camera_names,
            "render_camera": resolve_render_camera(cfg, camera_names),
            "loaded_checkpoint": None if loaded_checkpoint is None else str(loaded_checkpoint),
            "resume_enabled": bool(cfg.runtime.resume),
            "load_buffers": bool(cfg.runtime.load_buffers),
            "seed": int(cfg.seed),
            "console_log": str(console_log_path),
            "runtime_log": str(runtime_log_path),
            "buffer_dir": str(checkpoint_dir / "buffers"),
            "online_chunk_dir": str(checkpoint_dir / "buffers" / "online_chunks"),
            "demo_chunk_dir": str(checkpoint_dir / "buffers" / "demo_chunks"),
        },
    )

    print(f"[INFO] Loading offline demos from {demo_source_name}...")
    proprio_keys = [str(key) for key in list(cfg.env.proprio_keys or [])]
    shared_demo_cache_dir = output_root / "_demo_cache"
    cache_key_parts = [
        f"h{int(cfg.env.img_height)}",
        f"w{int(cfg.env.img_width)}",
        f"cams-{'_'.join(camera_names)}",
        f"state-{'_'.join(proprio_keys) if proprio_keys else 'auto'}",
    ]
    transitions = load_demo_paths(
        demo_paths,
        cache_dir=shared_demo_cache_dir,
        mirror_cache_dir=checkpoint_dir / "demo_cache",
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

    if bool(cfg.runtime.load_buffers):
        online_chunk_dir, demo_chunk_dir = resolve_buffer_chunk_dirs(loaded_checkpoint)
        loaded_online_transitions = 0
        loaded_demo_transitions = 0
        if online_chunk_dir is not None:
            agent.online_buffer.clear()
            loaded_online_transitions = load_transition_chunks(agent.online_buffer, online_chunk_dir)
        if demo_chunk_dir is not None:
            loaded_demo_transitions = load_transition_chunks(agent.demo_buffer, demo_chunk_dir)
        if loaded_online_transitions > 0 or loaded_demo_transitions > 0:
            print(
                f"[load] chunk_buffers online={loaded_online_transitions} demo={loaded_demo_transitions} "
                f"from {resolve_checkpoint_run_dir(loaded_checkpoint) / 'buffers'}"
            )

    wandb_run = maybe_build_wandb(cfg, run_name=run_name, run_dir=checkpoint_dir)
    device = None
    intervention_runtime = None

    episode_return = 0.0
    episode_length = 0
    success_count = 0
    last_step = start_step - 1
    control_limiter = FixedRateLimiter(control_fps)
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

    def event_time_fields() -> dict[str, Any]:
        now = datetime.datetime.now()
        return {
            "wall_time": now.isoformat(timespec="milliseconds"),
            "run_elapsed_sec": round(time.monotonic() - run_started_monotonic, 6),
            "training_elapsed_sec": round(time.monotonic() - training_started_monotonic, 6),
        }

    def refresh_main_viewer() -> None:
        if not main_has_renderer:
            return
        if getattr(env, "viewer", None) is None and hasattr(env, "initialize_renderer"):
            env.initialize_renderer()
        if getattr(env, "viewer", None) is not None and hasattr(env.viewer, "update"):
            env.viewer.update()

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
                    "learner_total_updates": int(metrics.get("learner_total_updates", 0.0)),
                    "learner_actor_updates": int(metrics.get("learner_actor_updates", 0.0)),
                    **event_time_fields(),
                }
            )
            print(format_publish_line(metrics))
            last_reported_publish_count = publish_count

    def maybe_report_runtime(step: int) -> None:
        nonlocal last_fps_log_time
        now = time.monotonic()
        elapsed = now - last_fps_log_time
        if elapsed < fps_log_interval:
            return
        learner_progress = trainer.progress_snapshot()
        overall_fps = overall_fps_tracker.snapshot(elapsed)
        pending_updates = trainer.pending_async_updates() if async_updates else 0
        runtime_payload = {
            "overall_fps": overall_fps,
            "learner_actor_updates": float(learner_progress["actor_updates"]),
            "learner_updates_until_publish": float(learner_progress["updates_until_publish"]),
            "learner_publish_count": float(learner_progress["publish_count"]),
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
                "learner_total_updates": int(learner_progress["total_updates"]),
                "learner_actor_updates": int(learner_progress["actor_updates"]),
                "learner_publish_count": int(learner_progress["publish_count"]),
                "learner_updates_until_publish": int(learner_progress["updates_until_publish"]),
                "total_transition_count": int(total_transition_count),
                "total_intervention_transitions": int(total_intervention_transitions),
                "global_intervention_ratio": (
                    float(total_intervention_transitions) / float(total_transition_count)
                    if total_transition_count > 0
                    else 0.0
                ),
                "episode_transition_count": int(episode_transition_count),
                "episode_intervention_transitions": int(episode_intervention_transitions),
                "episode_intervention_ratio": (
                    float(episode_intervention_transitions) / float(episode_transition_count)
                    if episode_transition_count > 0
                    else 0.0
                ),
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
            )
        )
        last_fps_log_time = now

    def request_checkpoint_save(step: int, tag: str | None = None) -> None:
        step_checkpoint = (
            checkpoint_path(checkpoint_dir, tag)
            if tag is not None
            else checkpoint_step_path(
                checkpoint_dir,
                step=step,
                learner_updates=trainer.total_updates,
                episode_index=episode_index,
            )
        )
        checkpoint_extra = {
            "global_step": int(step),
            "episode_index": int(episode_index),
            "success_count": int(success_count),
            "trainer_state": trainer.state_dict(),
        }
        checkpoint_writer.request_save(
            paths=[step_checkpoint, checkpoint_path(checkpoint_dir, "latest")],
            include_buffers=False,
            extra=checkpoint_extra,
            metadata={"step": int(step)},
        )
        pending_updates = trainer.pending_async_updates() if async_updates else 0
        runtime_logger.log(
            {
                "event": "checkpoint_queued",
                "step": int(step),
                "episode_index": int(episode_index),
                "pending_updates": int(pending_updates),
                "learner_total_updates": int(trainer.total_updates),
                "checkpoint_name": step_checkpoint.name,
                **event_time_fields(),
            }
        )
        print(f"[ckpt] step={step} pending={pending_updates} queued={step_checkpoint.name}")

    requested_pretrain_steps = max(0, int(cfg.algorithm.trainer.pretrain_steps))
    pretrain_steps = requested_pretrain_steps
    if bool(cfg.intervention.enabled) and bool(cfg.runtime.interactive) and pretrain_steps > 0:
        print(
            "[INFO] Skipping blocking offline BC pretraining because interactive intervention is enabled. "
            "This keeps SpaceMouse teleoperation available immediately, matching the existing HIL-SERL "
            "robosuite workflow more closely. Run with intervention.enabled=false if you want offline "
            f"pretraining first (requested pretrain_steps={requested_pretrain_steps})."
        )
        pretrain_steps = 0
    completed_pretrain_steps = int(trainer.total_pretrain_updates)
    if completed_pretrain_steps < pretrain_steps:
        remaining_pretrain_steps = pretrain_steps - completed_pretrain_steps
        print(
            f"[pretrain] starting remaining_pretrain_steps={remaining_pretrain_steps} "
            f"(completed={completed_pretrain_steps}, target={pretrain_steps})"
        )
        for local_step in range(remaining_pretrain_steps):
            metrics = trainer.pretrain(1)[-1]
            absolute_pretrain_step = completed_pretrain_steps + local_step + 1
            if absolute_pretrain_step % int(cfg.logging.log_interval) == 0:
                maybe_log(
                    wandb_run,
                    {f"pretrain/{key}": value for key, value in metrics.items()},
                    step=absolute_pretrain_step,
                )
                print(
                    "[pretrain] "
                    f"step={absolute_pretrain_step} actor_loss={metrics.get('actor_loss', float('nan')):.4f} "
                    f"mse={metrics.get('mse', float('nan')):.4f}"
                )
            maybe_print_publish_events([metrics])
        request_checkpoint_save(start_step - 1, tag=f"pretrain_{pretrain_steps:08d}")

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

    try:
        for step in range(start_step, int(cfg.runtime.max_steps)):
            last_step = step
            loop_start = control_limiter.wait()
            overall_fps_tracker.mark()

            if policy_gate.ready(loop_start):
                if step < int(cfg.algorithm.trainer.random_steps):
                    cached_policy_action = np.random.uniform(action_low, action_high).astype(np.float32)
                else:
                    cached_policy_action = agent.select_action(obs, deterministic=bool(cfg.runtime.eval_deterministic))

            env_action = np.asarray(cached_policy_action, dtype=np.float32)
            is_intervention = False
            reset_requested = False
            if intervention_runtime is not None and spacemouse_gate.ready(loop_start):
                override_action, sampled_is_intervention, reset_requested = intervention_runtime.maybe_override_action(
                    cached_policy_action
                )
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
                obs, _ = reset_observation_adapter(adapter, preserve_mjviewer=main_has_renderer)
                refresh_main_viewer()
                episode_return = 0.0
                episode_length = 0
                episode_index += 1
                episode_transition_count = 0
                episode_intervention_transitions = 0
                policy_gate.force_ready()
                spacemouse_gate.force_ready()
                if intervention_runtime is not None:
                    intervention_runtime.start_episode()
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
            reward, success = sparse_success_reward(env, info if isinstance(info, dict) else None)
            next_obs = adapter.transform(raw_next_obs)
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
            )
            serialized_transition = agent.online_buffer.snapshot_transition(recorded_transition)
            buffer_writer.request_transition(
                online_transition=serialized_transition,
                demo_transition=serialized_transition if is_intervention else None,
            )
            total_transition_count += 1
            episode_transition_count += 1
            if is_intervention:
                total_intervention_transitions += 1
                episode_intervention_transitions += 1
                runtime_logger.log(
                    {
                        "event": "intervention_transition",
                        "step": int(step),
                        "episode_index": int(episode_index),
                        "episode_transition_index": int(episode_transition_count),
                        "total_transition_count": int(total_transition_count),
                        "episode_intervention_transitions": int(episode_intervention_transitions),
                        "total_intervention_transitions": int(total_intervention_transitions),
                        "episode_intervention_ratio": float(episode_intervention_transitions)
                        / float(episode_transition_count),
                        "global_intervention_ratio": float(total_intervention_transitions)
                        / float(total_transition_count),
                        **event_time_fields(),
                    }
                )
            episode_return += reward
            episode_length += 1
            success_count += int(success)

            if async_updates:
                update_metrics_list = trainer.maybe_update_async()
            else:
                update_metrics_list = trainer.maybe_update()
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
                episode_payload = {
                    "episode_return": float(episode_return),
                    "episode_length": int(episode_length),
                    "episode_success": int(success),
                    "online_buffer_size": len(agent.online_buffer),
                    "demo_buffer_size": len(agent.demo_buffer),
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
                        "episode_transition_count": int(episode_transition_count),
                        "episode_intervention_transitions": int(episode_intervention_transitions),
                        "episode_intervention_ratio": (
                            float(episode_intervention_transitions) / float(episode_transition_count)
                            if episode_transition_count > 0
                            else 0.0
                        ),
                        "total_transition_count": int(total_transition_count),
                        "total_intervention_transitions": int(total_intervention_transitions),
                        "global_intervention_ratio": (
                            float(total_intervention_transitions) / float(total_transition_count)
                            if total_transition_count > 0
                            else 0.0
                        ),
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
                obs, _ = reset_observation_adapter(adapter, preserve_mjviewer=main_has_renderer)
                refresh_main_viewer()
                episode_return = 0.0
                episode_length = 0
                episode_index += 1
                episode_transition_count = 0
                episode_intervention_transitions = 0
                cached_override_action = None
                cached_is_intervention = False
                policy_gate.force_ready()
                spacemouse_gate.force_ready()
                if intervention_runtime is not None:
                    intervention_runtime.start_episode()
            else:
                obs = next_obs

            if step > 0 and step % int(cfg.logging.checkpoint_interval) == 0:
                request_checkpoint_save(step)
            maybe_report_runtime(step)
    finally:
        if async_updates:
            trainer.flush_async_updates()
            flushed_metrics = trainer.drain_async_metrics()
            if flushed_metrics:
                maybe_print_publish_events(flushed_metrics)
        maybe_report_runtime(last_step if last_step >= 0 else 0)
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
        checkpoint_writer.flush(timeout=120.0)
        buffer_writer.flush(timeout=120.0)
        runtime_logger.log(
            {
                "event": "run_end",
                "step": int(last_step),
                "episode_index": int(episode_index),
                "success_count": int(success_count),
                "total_transition_count": int(total_transition_count),
                "total_intervention_transitions": int(total_intervention_transitions),
                "global_intervention_ratio": (
                    float(total_intervention_transitions) / float(total_transition_count)
                    if total_transition_count > 0
                    else 0.0
                ),
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
                "camera_names": camera_names,
                "render_camera": resolve_render_camera(cfg, camera_names),
                "loaded_checkpoint": None if loaded_checkpoint is None else str(loaded_checkpoint),
                "resume_enabled": bool(cfg.runtime.resume),
                "load_buffers": bool(cfg.runtime.load_buffers),
                "seed": int(cfg.seed),
                "console_log": str(console_log_path),
                "runtime_log": str(runtime_log_path),
                "buffer_dir": str(checkpoint_dir / "buffers"),
                "online_chunk_dir": str(checkpoint_dir / "buffers" / "online_chunks"),
                "demo_chunk_dir": str(checkpoint_dir / "buffers" / "demo_chunks"),
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
        if wandb_run is not None:
            wandb_run.finish()
        try:
            env.close()
        finally:
            try:
                obs_render_env.close()
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
