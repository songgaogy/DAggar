"""DIPOLE-RL training entry: DIPOLE flow + IQL Q-chunking + online BCE disc.

This module is a sibling of ``train_dipole.py`` (not a replacement). The rollout
loop body is a near-verbatim copy from ``train_dipole.main()``; RL additions are
marked with ``# RL-ADD`` / ``# RL-EDIT`` comments. Helper functions
(``load_hdf5_demos_into_flow_transitions`` etc.) are imported from
``train_dipole`` to keep both entry points in lock-step.

See ``robosuite/pipeline/docs/prompts/05_integrated_trainer.md`` for the spec
and ``docs/DIPOLE_RL.md`` for the architecture diagram.
"""
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

from robosuite.pipeline.algorithms.dipole import DipoleTrainer, LPBV2GProvider
from robosuite.pipeline.algorithms.dipole.advantage_g_provider import AdvantageGProvider
from robosuite.pipeline.algorithms.discriminator.encoder import SharedFrozenEncoder
from robosuite.pipeline.algorithms.discriminator.online_bce import (
    DiscriminatorConfig,
    OnlineBCEDiscriminator,
)
from robosuite.pipeline.algorithms.discriminator.replay import DiscriminatorReplayBuffer
from robosuite.pipeline.algorithms.q_learning.common import IQLConfig
from robosuite.pipeline.algorithms.q_learning.iql import IQLLearner
from robosuite.pipeline.algorithms.q_learning.replay import IQLReplayBuffer
from robosuite.pipeline.common.types import Transition
from robosuite.pipeline.envs import (
    RobosuiteInterventionRuntime,
    RobosuiteViewerRuntime,
    build_device,
    build_robosuite_env,
    choose_viewer_backend,
    compute_grasp_penalty,
    snapshot_env_state,
    sparse_success_reward,
)
from robosuite.pipeline.factory import build_algorithm
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
    load_demo_paths,
    load_transition_chunks,
    maybe_build_wandb,
    maybe_log,
    maybe_wrap_visualization,
    resolve_buffer_chunk_dirs,
    resolve_buffer_snapshot_paths,
    resolve_checkpoint_run_dir,
    resolve_demo_inputs,
    resolve_render_camera,
    resolve_runtime_fps,
    write_resolved_config,
    write_run_info,
)

# Reuse all the helpers from train_dipole (module-level functions only).
from robosuite.pipeline.train_dipole import (
    DISCRIMINATOR_DISPLAY_HZ,
    bind_flow_proprio_extractor,
    build_flow_runtime_cfg,
    convert_env_camera_observation,
    format_base_policy_trajectory_tag,
    format_discriminator_line,
    format_publish_line,
    load_hdf5_demos_into_flow_transitions,
    load_init_checkpoint_payload,
    maybe_set_seed,
    reset_flow_policy_observation,
    resolve_algorithm_devices,
    resolve_base_policy_checkpoint_path,
    resolve_camera_names,
    resolve_flow_task_metadata,
    resolve_run_directory,
    resume_checkpoint_candidates,
    serialize_seed,
)


def _annotate_offline_demos(
    transitions: list[Transition],
    *,
    namespace: str = "offline_demo",
    starting_episode_index: int = 0,
) -> list[Transition]:
    """Stamp episode_index / episode_step / buffer_role onto ``info``.

    ``DiscriminatorReplayBuffer.bootstrap_from_demos`` calls ``base.add(t)``
    directly without per-episode bookkeeping, but the IQL replay's chunk-window
    valid-start cache reads ``info['episode_index']`` and ``info['buffer_role']``.
    We pre-annotate so the two replays see consistent metadata.
    """
    out: list[Transition] = []
    episode_index = int(starting_episode_index)
    episode_step = 0
    for t in transitions:
        info = {} if t.info is None else dict(t.info)
        info.setdefault("episode_index", int(episode_index))
        info.setdefault("episode_step", int(episode_step))
        info.setdefault("episode_namespace", str(namespace))
        info["buffer_role"] = "offline"
        out.append(
            Transition(
                obs=t.obs,
                action=t.action,
                reward=t.reward,
                next_obs=t.next_obs,
                done=t.done,
                grasp_penalty=t.grasp_penalty,
                is_intervention=bool(t.is_intervention),
                info=info,
                reward_source=t.reward_source or "precomputed",
                demo_source=t.demo_source or namespace,
            )
        )
        episode_step += 1
        if bool(t.done):
            episode_index += 1
            episode_step = 0
    return out


def _load_iql_warmup_state(
    learner: IQLLearner,
    warmup_ckpt: str,
    *,
    expected_context_dim: int,
    expected_action_dim: int,
) -> None:
    """Load an IQL state-dict ckpt produced by ``algorithms/q_learning/warmup.py``."""
    path = Path(to_absolute_path(str(warmup_ckpt)))
    if not path.exists():
        raise FileNotFoundError(f"IQL warmup ckpt not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    learner.load_state_dict(payload["iql_state"], strict=True)
    meta = payload.get("encoder_meta", {})
    if int(meta.get("context_dim", -1)) != int(expected_context_dim):
        raise ValueError(
            f"IQL warmup ckpt context_dim={meta.get('context_dim')} != "
            f"shared_encoder.context_dim={expected_context_dim}."
        )
    if int(meta.get("policy_action_dim", -1)) != int(expected_action_dim):
        raise ValueError(
            f"IQL warmup ckpt policy_action_dim={meta.get('policy_action_dim')} != "
            f"agent action_dim={expected_action_dim}."
        )
    print(f"[iql_warmup] loaded {path}")


@hydra.main(version_base="1.2", config_path="./config", config_name="train_dipole_rl")
def main(cfg: DictConfig) -> None:  # noqa: C901 — near-verbatim copy of train_dipole.main
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
        for key, value in dict(getattr(cfg.algorithm.flow, "camera_aliases", {}) or {}).items()
    }
    run_name, checkpoint_dir = resolve_run_directory(cfg)
    output_root = Path(to_absolute_path(str(cfg.logging.output_root)))
    run_started_at = datetime.datetime.now().isoformat(timespec="seconds")
    run_started_monotonic = time.monotonic()
    console_log_path = checkpoint_dir / "console.log"
    runtime_log_path = checkpoint_dir / "metrics_runtime.jsonl"
    train_log_path = checkpoint_dir / "metrics_train.jsonl"
    console_capture = ConsoleLogCapture(console_log_path)
    console_capture.start()
    runtime_logger = JsonlEventLogger(runtime_log_path)
    runtime_logger.start()
    train_logger = JsonlEventLogger(train_log_path)
    train_logger.start()
    runtime_logger.log(
        {
            "event": "run_start",
            "run_name": run_name,
            "run_dir": str(checkpoint_dir),
            "started_at": run_started_at,
            "wall_time": datetime.datetime.now().isoformat(timespec="milliseconds"),
            "run_elapsed_sec": 0.0,
            "training_elapsed_sec": 0.0,
            "algorithm_type": "dipole_rl",
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
        raise ValueError("flow-dagger requires at least one policy camera.")

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
    main_env = maybe_wrap_visualization(main_env, enabled=False, label="training env")
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
    algorithm_cfg = OmegaConf.to_container(cfg.algorithm, resolve=True)
    if isinstance(algorithm_cfg, dict):
        algorithm_cfg["camera_names"] = list(policy_camera_names)
        algorithm_cfg["task_name"] = task_name
        flow_cfg = algorithm_cfg.setdefault("flow", {})
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

    # ------------------------------------------------------------------ #
    # RL-ADD: shared encoder, IQL, discriminator, two replay wrappers.   #
    # ------------------------------------------------------------------ #
    bce_warm_ckpt_raw = cfg.algorithm.discriminator.warm_start_ckpt
    if bce_warm_ckpt_raw is None or str(bce_warm_ckpt_raw).strip().lower() in ("", "null"):
        raise RuntimeError(
            "DIPOLE-RL requires algorithm.discriminator.warm_start_ckpt to point at a "
            "pretrained LPB BCE checkpoint (used to seed the SharedFrozenEncoder)."
        )
    bce_warm_ckpt = to_absolute_path(str(bce_warm_ckpt_raw))
    if not Path(bce_warm_ckpt).exists():
        raise FileNotFoundError(f"LPB BCE checkpoint not found: {bce_warm_ckpt}")
    rl_learner_device = str(cfg.algorithm.q_learning.config.device)
    shared_encoder = SharedFrozenEncoder(bce_warm_ckpt, device=rl_learner_device)
    shared_encoder.bind_policy_cameras(list(agent.camera_names))
    print(
        f"[rl] shared_encoder ckpt={bce_warm_ckpt} device={rl_learner_device} "
        f"context_dim={shared_encoder.context_dim} views={shared_encoder.view_names}"
    )

    # Legacy LPB-only G provider — still constructed unconditionally so the
    # discriminator-display block at the rollout-loop site has a working
    # compute_g_for_observation() implementation regardless of g_mode.
    lpb_cfg = cfg.algorithm.dipole.lpb_detector
    lpb_ckpt_raw = lpb_cfg.ckpt_path if lpb_cfg is not None else None
    if lpb_ckpt_raw is None or str(lpb_ckpt_raw).strip().lower() in ("", "null"):
        raise RuntimeError(
            "DIPOLE-RL requires algorithm.dipole.lpb_detector.ckpt_path even in "
            "g_mode=advantage (used by the discriminator-display path)."
        )
    lpb_ckpt_path = Path(to_absolute_path(str(lpb_ckpt_raw)))
    if not lpb_ckpt_path.exists():
        raise FileNotFoundError(f"LPB BCE checkpoint not found: {lpb_ckpt_path}")
    try:
        legacy_camera_to_view = {str(k): str(v) for k, v in dict(lpb_cfg.camera_to_view or {}).items()}
    except Exception:
        legacy_camera_to_view = {}
    legacy_g_provider = LPBV2GProvider(
        ckpt_path=str(lpb_ckpt_path),
        task_name=str(task_name),
        device=str(cfg.algorithm.flow.device),
        camera_to_view=legacy_camera_to_view,
        shared_encoder=shared_encoder,
    )
    legacy_g_provider.bind_policy_cameras(list(agent.camera_names))

    # IQL learner + replay. Gated by `algorithm.q_learning.enabled` so the
    # legacy DIPOLE regression path (g_mode=bce_frozen + q_learning.enabled=
    # false) degenerates into the existing DipoleTrainer with no IQL work
    # per tick.
    policy_action_dim = int(agent.flow_config.action_dim)
    policy_action_horizon = int(agent.flow_config.action_horizon)
    iql_cfg_dict = OmegaConf.to_container(cfg.algorithm.q_learning.config, resolve=True)
    iql_cfg = IQLConfig(**iql_cfg_dict)
    q_learning_enabled = bool(getattr(cfg.algorithm.q_learning, "enabled", True))
    if q_learning_enabled:
        iql_learner = IQLLearner(
            iql_cfg, context_dim=int(shared_encoder.context_dim), action_dim=policy_action_dim
        )
        iql_replay = IQLReplayBuffer(agent.online_buffer, iql_cfg)
    else:
        iql_learner = None
        iql_replay = None
        print("[rl] algorithm.q_learning.enabled=false — IQL learner / replay skipped")

    # Discriminator + replay. Gated by `algorithm.discriminator.online_train`
    # so the legacy regression run leaves the BCE head frozen on disk.
    disc_cfg_dict = OmegaConf.to_container(cfg.algorithm.discriminator.config, resolve=True)
    disc_cfg = DiscriminatorConfig(**disc_cfg_dict)
    disc_online_train = bool(getattr(cfg.algorithm.discriminator, "online_train", True))
    if disc_online_train:
        discriminator = OnlineBCEDiscriminator(
            cfg=disc_cfg,
            encoder=shared_encoder,
            context_dim=int(shared_encoder.context_dim),
            action_dim=policy_action_dim,
            action_horizon=policy_action_horizon,
        )
        disc_replay = DiscriminatorReplayBuffer(
            disc_cfg,
            agent.online_buffer,
            encoder=shared_encoder,
            action_horizon=policy_action_horizon,
        )
    else:
        discriminator = None
        disc_replay = None
        print("[rl] algorithm.discriminator.online_train=false — online disc / replay skipped")

    # Hard-assert device consistency on the learner side.
    if not (iql_cfg.device == disc_cfg.device == shared_encoder.device == rl_learner_device):
        raise RuntimeError(
            "DIPOLE-RL device mismatch: "
            f"iql_cfg.device={iql_cfg.device} disc_cfg.device={disc_cfg.device} "
            f"encoder.device={shared_encoder.device} expected={rl_learner_device}"
        )

    # Trainer constructed AFTER all RL modules so it can hold their references.
    trainer = DipoleTrainer(
        agent,
        iql_learner=iql_learner,
        discriminator=discriminator,
        iql_replay=iql_replay,
        disc_replay=disc_replay,
        shared_encoder=shared_encoder,
        learner_device=rl_learner_device,
        iql_batch_size=int(cfg.algorithm.trainer.batch_size),
        disc_batch_size=int(disc_cfg.batch_size),
        disc_update_every_n_steps=int(disc_cfg.update_every_n_steps),
    )
    if iql_learner is not None:
        agent.attach_iql_learner(iql_learner)
    if discriminator is not None:
        agent.attach_discriminator(discriminator)

    # Until warmup completes (or always, when bootstrap_g_with_frozen_bce=true)
    # the flow loss G is sourced from the frozen LPB BCE detector.
    bootstrap_with_frozen = bool(getattr(cfg.runtime, "bootstrap_g_with_frozen_bce", True))
    agent.attach_g_provider(legacy_g_provider)
    print(
        f"[dipole] attached legacy BCE G provider (bootstrap_g_with_frozen_bce={bootstrap_with_frozen})"
    )
    # ------------------------------------------------------------------ #
    # END RL-ADD initial wiring.                                         #
    # ------------------------------------------------------------------ #

    demo_source_name, demo_paths, max_num_trajectories = resolve_demo_inputs(cfg)
    if not demo_paths:
        raise FileNotFoundError(
            "flow-dagger requires offline demos before training starts. "
            f"No demo files were found for '{demo_source_name}'. "
            "Place demos under ./data/<task>/expert or set data.demo_paths explicitly."
        )

    env = main_env
    obs = initial_obs
    control_fps = resolve_runtime_fps(cfg, "control_fps", float(cfg.env.control_freq))
    render_fps = resolve_runtime_fps(cfg, "render_fps", control_fps)
    policy_fps = resolve_runtime_fps(cfg, "policy_fps", control_fps)
    spacemouse_fps = resolve_runtime_fps(cfg, "spacemouse_fps", control_fps)
    image_obs_fps = resolve_runtime_fps(cfg, "image_obs_fps", control_fps)
    fps_log_interval = max(0.1, float(getattr(cfg.runtime, "fps_log_interval", 1.0)))
    unthrottled_runtime = bool(getattr(cfg.runtime, "unthrottled", False))
    async_updates = bool(getattr(cfg.runtime, "async_updates", False))
    eval_episode_max_steps = int(getattr(cfg.runtime, "eval_episode_max_steps", 300))
    viewer_runtime: RobosuiteViewerRuntime | None = None
    if decoupled_viewer_enabled:
        viewer_requested_backend = str(getattr(cfg.runtime, "viewer_backend", "auto"))
        if viewer_requested_backend.lower() == "auto" and main_renderer == "mjviewer":
            viewer_backend = "mjviewer"
        else:
            viewer_backend = choose_viewer_backend(
                main_renderer, render_camera_names, requested_backend=viewer_requested_backend,
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
            viewer_env, enabled=visualize_gripper_markers, label="viewer env",
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
    if image_obs_fps < control_fps:
        print(
            "[INFO] Image observations are rate-limited below control frequency "
            f"({image_obs_fps:.2f}Hz images vs {control_fps:.2f}Hz control)."
        )
    if not online_updates_enabled:
        print("[INFO] Online learner updates are disabled. Policy parameters will stay frozen during rollout.")
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

    if loaded_checkpoint is None and initialized_checkpoint is None:
        raise RuntimeError(
            "DIPOLE-RL requires either a resumable checkpoint (runtime.checkpoint / runtime.resume) "
            "or runtime.init_checkpoint pointing to a flow-policy checkpoint to initialize the "
            "shared backbone."
        )

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
            "camera_names": policy_camera_names,
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
            "train_log": str(train_log_path),
            "buffer_dir": str(checkpoint_dir / "buffers"),
            "online_chunk_dir": str(checkpoint_dir / "buffers" / "online_chunks"),
            "demo_chunk_dir": str(checkpoint_dir / "buffers" / "demo_chunks"),
            "algorithm_type": "dipole_rl",
            "g_mode": str(cfg.algorithm.dipole.g_mode),
        },
    )

    print(f"[INFO] Loading offline demos from {demo_source_name}...")
    proprio_keys = [str(key) for key in list(cfg.env.proprio_keys or [])]
    shared_demo_cache_dir = output_root / "_demo_cache"
    cache_key_parts = [
        f"h{int(cfg.env.img_height)}",
        f"w{int(cfg.env.img_width)}",
        f"cams-{'_'.join(policy_camera_names)}",
        f"state-{'_'.join(proprio_keys) if proprio_keys else 'auto'}",
    ]
    transitions = load_demo_paths(
        demo_paths,
        cache_dir=shared_demo_cache_dir,
        mirror_cache_dir=checkpoint_dir / "demo_cache",
        hdf5_loader=lambda path, demo_names=None: load_hdf5_demos_into_flow_transitions(
            path,
            policy_camera_names=policy_camera_names,
            camera_aliases=camera_aliases,
            img_height=int(cfg.env.img_height),
            img_width=int(cfg.env.img_width),
            proprio_keys=tuple(cfg.env.proprio_keys or []),
            renderer=str(cfg.env.renderer),
            control_freq=int(cfg.env.control_freq),
            demo_names=demo_names,
            state_extractor=flow_proprio_extractor,
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

    if not agent.has_normalizers():
        agent.fit_normalizers_from_transitions(transitions)
        print("[INFO] Fitted flow normalizers from offline demos.")
    else:
        print("[INFO] Reusing normalizers from checkpoint.")

    # ------------------------------------------------------------------ #
    # RL-ADD: seed disc/IQL non-failure pool from offline demos.         #
    # ------------------------------------------------------------------ #
    # Demos carry is_intervention=False → label=0 (non-failure / good
    # behavior) per the warm-started lpb_v2 BCE convention. The IQL replay
    # and the disc replay both wrap agent.online_buffer by reference, so a
    # single seed call serves both subsystems. In the legacy regression
    # path (disc_online_train=false AND q_learning.enabled=false) we skip
    # seeding entirely — agent.online_buffer is only used for record-keeping
    # of the rollout transitions in that path.
    if (iql_replay is not None or disc_replay is not None) and len(agent.online_buffer) == 0:
        annotated_demos = _annotate_offline_demos(transitions, namespace="offline_demo")
        if disc_replay is not None:
            disc_replay.bootstrap_from_demos(annotated_demos)
            print(
                f"[rl] disc/iql online buffer seeded with {len(annotated_demos)} offline demo transitions "
                f"(failure pool={disc_replay.num_failure}, non_failure pool={disc_replay.num_non_failure})"
            )
        else:
            for t in annotated_demos:
                agent.online_buffer.add(t)
            print(
                f"[rl] iql online buffer seeded with {len(annotated_demos)} offline demo transitions "
                f"(disc disabled — no bucket classification)"
            )
    elif len(agent.online_buffer) > 0:
        print(
            f"[rl] reusing checkpoint online buffer with {len(agent.online_buffer)} transitions; "
            "failure/non_failure pools will be classified lazily on next disc/iql sample"
        )
    # ------------------------------------------------------------------ #
    # END RL-ADD: demo bootstrap.                                        #
    # ------------------------------------------------------------------ #

    requested_pretrain_steps = max(0, int(cfg.algorithm.trainer.pretrain_steps))
    base_policy_checkpoint = resolve_base_policy_checkpoint_path(
        output_root,
        env_name=str(cfg.env.environment),
        demo_source_name=demo_source_name,
        max_num_trajectories=max_num_trajectories,
        pretrain_steps=requested_pretrain_steps,
    )
    base_policy_metadata = {
        "env_name": str(cfg.env.environment),
        "demo_source_name": demo_source_name,
        "num_trajectories": None if max_num_trajectories is None else int(max_num_trajectories),
        "offline_transition_count": int(len(transitions)),
        "pretrain_steps": int(requested_pretrain_steps),
        "checkpoint_path": str(base_policy_checkpoint),
    }
    base_policy_reused = False
    if loaded_checkpoint is None and initialized_checkpoint is None:
        if base_policy_checkpoint.exists():
            try:
                extra = agent.load_checkpoint(base_policy_checkpoint, load_buffers=False)
                trainer.load_state_dict(extra.get("trainer_state"))
                loaded_checkpoint = base_policy_checkpoint
                base_policy_reused = True
                print(
                    "[base_policy] reusing "
                    f"{base_policy_checkpoint.name} "
                    f"(trajectories={format_base_policy_trajectory_tag(max_num_trajectories)}, "
                    f"pretrain_steps={requested_pretrain_steps})"
                )
            except Exception as exc:
                print(f"[WARN] Failed to load reusable base policy {base_policy_checkpoint}: {exc}")
        elif requested_pretrain_steps <= 0:
            raise ValueError(
                "Base policy checkpoint is missing and algorithm.trainer.pretrain_steps <= 0. "
                "Set a positive pretrain_steps value to build the base policy from offline demos, "
                "or provide an existing checkpoint."
            )
    elif initialized_checkpoint is not None:
        base_policy_reused = True
        print(f"[base_policy] initialized from multitask flow checkpoint={initialized_checkpoint}")

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
    episode_step_index = 0
    success_count = 0
    last_step = start_step - 1
    control_limiter = None if unthrottled_runtime else FixedRateLimiter(control_fps)
    policy_gate = IntervalGate(policy_fps)
    spacemouse_gate = IntervalGate(spacemouse_fps)
    overall_fps_tracker = EMAFpsTracker()
    discriminator_display_hz = float(
        getattr(cfg.runtime, "discriminator_display_hz", DISCRIMINATOR_DISPLAY_HZ)
    )
    disc_log_gate = IntervalGate(discriminator_display_hz) if discriminator_display_hz > 0.0 else None
    if disc_log_gate is None:
        print("[INFO] Discriminator display disabled (runtime.discriminator_display_hz <= 0).")
    else:
        print(
            f"[INFO] Discriminator display rate: {discriminator_display_hz:.2f} Hz "
            f"(BCE threshold tau={legacy_g_provider.threshold:+.3f})"
        )
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

    # ------------------------------------------------------------------ #
    # Flow base-policy pretrain (same as train_dipole.py).               #
    # ------------------------------------------------------------------ #
    if loaded_checkpoint is None and initialized_checkpoint is None and start_step == 0:
        completed_pretrain_steps = int(trainer.total_pretrain_updates)
        remaining_pretrain_steps = requested_pretrain_steps - completed_pretrain_steps
        print(
            f"[base_policy] training remaining_pretrain_steps={remaining_pretrain_steps} "
            f"(completed={completed_pretrain_steps}, target={requested_pretrain_steps})"
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
                    "[base_policy] "
                    f"step={absolute_pretrain_step} loss={metrics.get('actor_loss', float('nan')):.4f} "
                    f"flow={metrics.get('flow_loss', float('nan')):.4f} "
                    f"endpoint={metrics.get('endpoint_loss', float('nan')):.4f}"
                )
            maybe_print_publish_events([metrics])
        base_policy_extra = {
            "global_step": -1,
            "episode_index": 0,
            "success_count": 0,
            "trainer_state": trainer.state_dict(),
            "base_policy_metadata": dict(base_policy_metadata),
        }
        agent.save_checkpoint(base_policy_checkpoint, include_buffers=False, extra=base_policy_extra)
        print(f"[base_policy] saved {base_policy_checkpoint}")
    elif base_policy_reused:
        print(f"[base_policy] ready from checkpoint={base_policy_checkpoint}")

    # ------------------------------------------------------------------ #
    # RL-ADD: IQL warmup (load ckpt or run pretrain_iql_value/full).     #
    # ------------------------------------------------------------------ #
    warmup_ckpt_raw = cfg.algorithm.q_learning.warmup_ckpt
    if iql_learner is None:
        print("[iql_warmup] skipped (algorithm.q_learning.enabled=false)")
    elif warmup_ckpt_raw is not None and str(warmup_ckpt_raw).strip().lower() not in ("", "null"):
        _load_iql_warmup_state(
            iql_learner,
            str(warmup_ckpt_raw),
            expected_context_dim=int(shared_encoder.context_dim),
            expected_action_dim=policy_action_dim,
        )
    else:
        warmup_v_steps = int(cfg.algorithm.q_learning.warmup_value_steps)
        warmup_full_steps = int(cfg.algorithm.q_learning.warmup_full_steps)
        print(f"[iql_warmup] in-process value={warmup_v_steps} full={warmup_full_steps}")
        v_log = trainer.pretrain_iql_value(warmup_v_steps)
        if v_log:
            print(f"[iql_warmup] value tail metrics: {v_log[-1]}")
        full_log = trainer.pretrain_iql_full(warmup_full_steps)
        if full_log:
            print(f"[iql_warmup] full tail metrics: {full_log[-1]}")
    # ------------------------------------------------------------------ #
    # RL-ADD: flip g_mode → advantage if requested.                      #
    # ------------------------------------------------------------------ #
    g_mode = str(cfg.algorithm.dipole.g_mode)
    if g_mode == "advantage":
        if iql_learner is None or discriminator is None:
            raise RuntimeError(
                "g_mode='advantage' requires both algorithm.q_learning.enabled "
                "and algorithm.discriminator.online_train to be true; got "
                f"q_learning.enabled={iql_learner is not None}, "
                f"discriminator.online_train={discriminator is not None}."
            )
        advantage_g = AdvantageGProvider(
            iql_learner=iql_learner,
            discriminator=discriminator,
            encoder=shared_encoder,
            alpha=float(cfg.algorithm.advantage_g_provider.alpha),
            beta=float(cfg.algorithm.advantage_g_provider.beta),
            advantage_normalization=str(cfg.algorithm.advantage_g_provider.advantage_normalization),
            disc_normalization=str(cfg.algorithm.advantage_g_provider.disc_normalization),
        )
        advantage_g.bind_policy_cameras(list(agent.camera_names))
        agent.attach_g_provider(advantage_g)
        print(
            f"[dipole] switched to AdvantageGProvider "
            f"(alpha={cfg.algorithm.advantage_g_provider.alpha}, "
            f"beta={cfg.algorithm.advantage_g_provider.beta})"
        )
    elif g_mode == "bce_frozen":
        # legacy_g_provider already attached above. No-op.
        pass
    else:
        raise ValueError(f"Unknown algorithm.dipole.g_mode: {g_mode}")
    # ------------------------------------------------------------------ #
    # END RL-ADD: warmup + g_mode switch.                                #
    # ------------------------------------------------------------------ #

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
    agent.reset_policy_state()
    policy_gate.force_ready()
    spacemouse_gate.force_ready()
    if intervention_runtime is not None:
        intervention_runtime.start_episode()

    if async_updates and online_updates_enabled:
        trainer.start_async_worker()

    try:
        for step in range(start_step, int(cfg.runtime.max_steps)):
            last_step = step
            loop_start = time.monotonic() if control_limiter is None else control_limiter.wait()
            overall_fps_tracker.mark()

            if unthrottled_runtime or policy_gate.ready(loop_start):
                if step < int(cfg.algorithm.trainer.random_steps):
                    cached_policy_action = np.random.uniform(action_low, action_high).astype(np.float32)
                else:
                    cached_policy_action = agent.select_action(obs, deterministic=bool(cfg.runtime.eval_deterministic))

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
                    agent.reset_policy_state()
                elif sampled_is_intervention:
                    cached_override_action = np.asarray(override_action, dtype=np.float32)
                    cached_is_intervention = True
                    agent.notify_intervention()
                    policy_gate.force_ready()
                else:
                    cached_override_action = None
                    cached_is_intervention = False

            if reset_requested:
                if not bool(cfg.intervention.device_reset_as_episode_reset):
                    print("[INFO] Device reset requested. Exiting training loop.")
                    break
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
                agent.reset_policy_state()
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
            next_obs = convert_env_camera_observation(
                raw_next_obs,
                env=env,
                extractor=flow_proprio_extractor,
                policy_camera_names=policy_camera_names,
                camera_aliases=camera_aliases,
                img_height=int(cfg.env.img_height),
                img_width=int(cfg.env.img_width),
            )
            refresh_main_viewer()
            if frozen_eval_mode and (episode_length + 1) >= eval_episode_max_steps:
                done = True
            done = bool(done or success)
            # RL-EDIT: tag `buffer_role="online"` so IQL replay can distinguish
            # online transitions from offline-bootstrap ones.
            info_payload = dict(info) if isinstance(info, dict) else {"raw_info": info}
            info_payload.setdefault("buffer_role", "online")
            recorded_transition = trainer.record_transition(
                obs=obs,
                action=env_action,
                next_obs=next_obs,
                done=done,
                reward=reward,
                grasp_penalty=info.get("grasp_penalty") if isinstance(info, dict) else None,
                is_intervention=is_intervention,
                info=info_payload,
                reward_source="env_success",
                demo_source="intervention" if is_intervention else None,
                episode_index=episode_index,
                episode_step=episode_step_index,
            )
            serialized_transition = agent.online_buffer.snapshot_transition(recorded_transition)
            buffer_writer.request_transition(
                online_transition=serialized_transition,
                demo_transition=serialized_transition if is_intervention else None,
            )
            total_transition_count += 1
            episode_transition_count += 1
            episode_step_index += 1
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

            if disc_log_gate is not None and disc_log_gate.ready(time.monotonic()):
                try:
                    chunk_for_disc = agent.plan_action_chunk(
                        next_obs, deterministic=bool(cfg.runtime.eval_deterministic)
                    )
                    # RL-EDIT: use legacy_g_provider for the disc-display path because
                    # AdvantageGProvider.compute_g_for_observation raises NotImplementedError.
                    disc_info = legacy_g_provider.compute_g_for_observation(next_obs, chunk_for_disc)
                except Exception as exc:
                    runtime_logger.log(
                        {
                            "event": "discriminator_display_error",
                            "step": int(step),
                            "episode_index": int(episode_index),
                            "error": repr(exc),
                            **event_time_fields(),
                        }
                    )
                else:
                    is_failure = bool(disc_info["is_failure"])
                    raw_score = float(disc_info["raw"])
                    tau_value = float(disc_info["tau"])
                    print(
                        format_discriminator_line(
                            step=step,
                            episode_index=episode_index,
                            raw=raw_score,
                            tau=tau_value,
                            is_failure=is_failure,
                        )
                    )
                    runtime_logger.log(
                        {
                            "event": "discriminator_display",
                            "step": int(step),
                            "episode_index": int(episode_index),
                            "raw": raw_score,
                            "G": -raw_score,
                            "tau": tau_value,
                            "is_failure": int(is_failure),
                            **event_time_fields(),
                        }
                    )

            if online_updates_enabled:
                if async_updates:
                    update_metrics_list = trainer.maybe_update_async()
                else:
                    update_metrics_list = trainer.maybe_update()
            else:
                update_metrics_list = []
            if update_metrics_list:
                maybe_print_publish_events(update_metrics_list)
                update_metrics = update_metrics_list[-1]
                if step % int(cfg.logging.log_interval) == 0:
                    maybe_log(wandb_run, update_metrics, step=step)
                    train_logger.log(
                        {
                            "event": "train_step",
                            "step": int(step),
                            "episode_index": int(episode_index),
                            "pending_updates": int(
                                trainer.pending_async_updates() if async_updates else 0
                            ),
                            **{k: float(v) for k, v in update_metrics.items()},
                            **event_time_fields(),
                        }
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
                agent.reset_policy_state()
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
                "train_log": str(train_log_path),
                "buffer_dir": str(checkpoint_dir / "buffers"),
                "online_chunk_dir": str(checkpoint_dir / "buffers" / "online_chunks"),
                "demo_chunk_dir": str(checkpoint_dir / "buffers" / "demo_chunks"),
                "last_step": int(last_step),
                "episode_index": int(episode_index),
                "success_count": int(success_count),
                "latest_checkpoint": str(checkpoint_path(checkpoint_dir, "latest")),
                "trainer_state": trainer.state_dict(),
                "base_policy_checkpoint": str(base_policy_checkpoint),
                "base_policy_reused": bool(base_policy_reused),
                "base_policy_metadata": dict(base_policy_metadata),
                "algorithm_type": "dipole_rl",
                "g_mode": str(cfg.algorithm.dipole.g_mode),
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
                                train_logger.close()
                            finally:
                                try:
                                    buffer_writer.close()
                                finally:
                                    console_capture.stop()


if __name__ == "__main__":
    main()
