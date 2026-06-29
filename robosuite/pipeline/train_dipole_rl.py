"""DIPOLE-RL training entry: DIPOLE flow + IQL Q-chunking + frozen nnPU.

This module is a sibling of ``train_dipole.py`` (not a replacement). The rollout
loop body is a near-verbatim copy from ``train_dipole.main()``; RL additions are
marked with ``# RL-ADD`` / ``# RL-EDIT`` comments. Helper functions
(``load_hdf5_demos_into_flow_transitions`` etc.) are imported from
``train_dipole`` to keep both entry points in lock-step.

The complete architecture and runtime contract are documented in
``robosuite/pipeline/README.md``.
"""
from __future__ import annotations

import datetime
import sys
import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from robosuite.pipeline.algorithms.dipole import DipoleTrainer, NNPUGProvider
from robosuite.pipeline.algorithms.dipole.advantage_g_provider import AdvantageGProvider
from robosuite.pipeline.algorithms.discriminator.encoder import SharedDynamicsEncoder
from robosuite.pipeline.algorithms.discriminator.nnpu import FrozenNNPUDiscriminator
from robosuite.pipeline.algorithms.discriminator.runtime import (
    EnterKeyListener,
    build_nnpu_runtime,
    render_nnpu_hud,
)
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
    EnvRandomReducer,
    FixedRateLimiter,
    IntervalGate,
    JsonlEventLogger,
    checkpoint_path,
    checkpoint_step_path,
    load_demo_paths,
    load_transition_chunks,
    maybe_build_tensorboard,
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
    bind_flow_proprio_extractor,
    build_flow_runtime_cfg,
    convert_env_camera_observation,
    format_publish_line,
    load_hdf5_demos_into_flow_transitions,
    load_init_checkpoint_payload,
    maybe_set_seed,
    reset_flow_policy_observation,
    resolve_algorithm_devices,
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

    The IQL replay's chunk-window valid-start cache reads
    ``info['episode_index']`` and ``info['buffer_role']``. We pre-annotate the
    demonstrations so offline and online windows remain separate.
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
    expected_state_feature_dim: int,
    expected_chunk_feature_dim: int,
    expected_action_dim: int,
) -> None:
    """Load an IQL state-dict ckpt produced by ``algorithms/q_learning/warmup.py``."""
    path = Path(to_absolute_path(str(warmup_ckpt)))
    if not path.exists():
        raise FileNotFoundError(f"IQL warmup ckpt not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    learner.load_state_dict(payload["iql_state"], strict=True)
    meta = payload.get("encoder_meta", {})
    if int(meta.get("state_feature_dim", -1)) != int(expected_state_feature_dim):
        raise ValueError(
            f"IQL warmup ckpt state_feature_dim={meta.get('state_feature_dim')} != "
            f"shared_encoder.state_feature_dim={expected_state_feature_dim}."
        )
    if int(meta.get("chunk_feature_dim", -1)) != int(expected_chunk_feature_dim):
        raise ValueError(
            f"IQL warmup ckpt chunk_feature_dim={meta.get('chunk_feature_dim')} != "
            f"shared_encoder.chunk_feature_dim={expected_chunk_feature_dim}."
        )
    if int(meta.get("policy_action_dim", -1)) != int(expected_action_dim):
        raise ValueError(
            f"IQL warmup ckpt policy_action_dim={meta.get('policy_action_dim')} != "
            f"agent action_dim={expected_action_dim}."
        )
    print(f"[iql_warmup] loaded {path}")


@hydra.main(version_base="1.2", config_path="./config", config_name="train_dipole_rl")
def main(cfg: DictConfig) -> None:  # noqa: C901 — near-verbatim copy of train_dipole.main
    
    # ------------------------------------------------------------------ #
    # Initialize the training environment.                               #
    # ------------------------------------------------------------------ #
    
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

    # Shared frozen dynamics encoder + calibrated nnPU head.
    disc_node = cfg.algorithm.discriminator
    nnpu_ckpt_raw = disc_node.checkpoint
    if nnpu_ckpt_raw is None or str(nnpu_ckpt_raw).strip().lower() in ("", "null"):
        raise RuntimeError(
            "DIPOLE-RL requires algorithm.discriminator.checkpoint to point at "
            "a task-calibrated pu_bce_head.pth."
        )
    nnpu_ckpt = to_absolute_path(str(nnpu_ckpt_raw))
    if not Path(nnpu_ckpt).exists():
        raise FileNotFoundError(f"nnPU checkpoint not found: {nnpu_ckpt}")
    disc_node.checkpoint = nnpu_ckpt
    rl_learner_device = str(cfg.algorithm.q_learning.config.device)
    camera_to_view = {
        str(k): str(v) for k, v in dict(disc_node.camera_to_view or {}).items()
    }
    encoder_override = getattr(disc_node, "encoder_ckpt", None)
    encoder_ckpt = (
        None
        if encoder_override is None or str(encoder_override).strip().lower() in ("", "null")
        else to_absolute_path(str(encoder_override))
    )
    if encoder_ckpt is not None:
        disc_node.encoder_ckpt = encoder_ckpt
    shared_encoder = SharedDynamicsEncoder(
        nnpu_ckpt_path=nnpu_ckpt,
        encoder_ckpt=encoder_ckpt,
        device=rl_learner_device,
        camera_to_view=camera_to_view,
    )
    shared_encoder.bind_policy_cameras(list(agent.camera_names))
    print(
        f"[rl] dynamics_encoder ckpt={nnpu_ckpt} device={rl_learner_device} "
        f"state_dim={shared_encoder.state_feature_dim} "
        f"chunk_dim={shared_encoder.chunk_feature_dim} views={shared_encoder.view_names}"
    )
    discriminator = FrozenNNPUDiscriminator(
        nnpu_ckpt_path=nnpu_ckpt,
        task_name=str(task_name),
        device=rl_learner_device,
        encoder=shared_encoder,
    )
    nnpu_g_provider = NNPUGProvider(
        encoder=shared_encoder,
        discriminator=discriminator,
    )

    # IQL learner + replay. The frozen nnPU path can run without IQL updates.
    policy_action_dim = int(agent.flow_config.action_dim)
    iql_cfg_dict = OmegaConf.to_container(cfg.algorithm.q_learning.config, resolve=True)
    iql_cfg = IQLConfig(**iql_cfg_dict)
    q_learning_enabled = bool(getattr(cfg.algorithm.q_learning, "enabled", True))

    if q_learning_enabled:
        iql_learner = IQLLearner(
            iql_cfg,
            state_feature_dim=int(shared_encoder.state_feature_dim),
            chunk_feature_dim=int(shared_encoder.chunk_feature_dim),
            action_dim=policy_action_dim,
            n_tokens=int(shared_encoder.inner_encoder.num_patches),
            proprio_dim=int(shared_encoder.inner_encoder.proprio_emb_dim),
        )
        iql_replay = IQLReplayBuffer(agent.online_buffer, iql_cfg)
    else:
        iql_learner = None
        iql_replay = None
        print("[rl] algorithm.q_learning.enabled=false — IQL learner / replay skipped")

    if not (str(iql_cfg.device) == str(shared_encoder.device) == rl_learner_device):
        raise RuntimeError(
            "DIPOLE-RL device mismatch: "
            f"iql_cfg.device={iql_cfg.device} "
            f"encoder.device={shared_encoder.device} expected={rl_learner_device}"
        )

    # Trainer constructed AFTER all RL modules so it can hold their references.
    trainer = DipoleTrainer(
        agent,
        iql_learner=iql_learner,
        discriminator=discriminator,
        iql_replay=iql_replay,
        shared_encoder=shared_encoder,
        learner_device=rl_learner_device,
        iql_batch_size=int(cfg.algorithm.trainer.batch_size),
        iql_update_freq=int(iql_cfg.update_freq),
    )
    if iql_learner is not None:
        agent.attach_iql_learner(iql_learner)
    agent.attach_discriminator(discriminator)

    # Until warmup completes the flow loss can use discriminator-only G.
    bootstrap_with_frozen = bool(
        getattr(cfg.runtime, "bootstrap_g_with_frozen_nnpu", True)
    )
    agent.attach_g_provider(nnpu_g_provider)
    print(
        "[dipole] attached frozen nnPU G provider "
        f"(bootstrap_g_with_frozen_nnpu={bootstrap_with_frozen})"
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
    env_random_reducer = EnvRandomReducer(serialize_seed(getattr(cfg, "seed", None)))
    if env_random_reducer.enabled:
        print(f"[determinism] env_reset_seed={env_random_reducer.base_seed} rule=base_seed+episode_index")
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
            "env_reset_seed": env_random_reducer.base_seed,
            "env_reset_seed_rule": "base_seed+episode_index" if env_random_reducer.enabled else None,
            "episode_index": int(episode_index),
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
    # RL-ADD: seed the IQL replay from offline demonstrations.           #
    # ------------------------------------------------------------------ #

    # load offline demos into online buffer
    if iql_replay is not None and len(agent.online_buffer) == 0:
        annotated_demos = _annotate_offline_demos(transitions, namespace="offline_demo")
        for transition in annotated_demos:
            agent.online_buffer.add(transition)
        print(f"[rl] iql online buffer seeded with {len(annotated_demos)} offline demo transitions")
    elif len(agent.online_buffer) > 0:
        print(
            f"[rl] reusing checkpoint online buffer with {len(agent.online_buffer)} transitions; "
            "offline/online windows will be indexed lazily on the next IQL sample"
        )

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

    
    # ------------------------------------------------------------------ #
    # Initialize the training loop.                                      #
    # ------------------------------------------------------------------ #

    metric_logger = maybe_build_tensorboard(cfg, run_name=run_name, run_dir=checkpoint_dir)
    device = None
    intervention_runtime = None
    nnpu_runtime = None
    enter_listener = EnterKeyListener()

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

    def reset_rollout_observation() -> tuple[dict[str, Any], int | None]:
        episode_seed = env_random_reducer.prepare_episode(env, episode_index, seed_global=False)
        reset_obs, _ = reset_flow_policy_observation(
            env,
            preserve_mjviewer=rollout_has_renderer,
            extractor=flow_proprio_extractor,
            policy_camera_names=policy_camera_names,
            camera_aliases=camera_aliases,
            img_height=int(cfg.env.img_height),
            img_width=int(cfg.env.img_width),
        )
        runtime_logger.log(
            {
                "event": "episode_reset",
                "step": int(last_step),
                "episode_index": int(episode_index),
                "episode_seed": None if episode_seed is None else int(episode_seed),
                **event_time_fields(),
            }
        )
        return reset_obs, episode_seed

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
            "runtime/overall_fps": overall_fps,
            "runtime/pending_updates": float(pending_updates),
            "runtime/elapsed_sec": float(time.monotonic() - training_started_monotonic),
            "learner/total_updates": float(learner_progress["total_updates"]),
            "learner/actor_updates": float(learner_progress["actor_updates"]),
            "learner/updates_until_publish": float(learner_progress["updates_until_publish"]),
            "learner/publish_count": float(learner_progress["publish_count"]),
            "buffer/total_transition_count": float(total_transition_count),
            "intervention/total_transitions": float(total_intervention_transitions),
            "intervention/global_ratio": (
                float(total_intervention_transitions) / float(total_transition_count)
                if total_transition_count > 0
                else 0.0
            ),
            "episode/current_transition_count": float(episode_transition_count),
            "episode/current_intervention_transitions": float(episode_intervention_transitions),
            "episode/current_intervention_ratio": (
                float(episode_intervention_transitions) / float(episode_transition_count)
                if episode_transition_count > 0
                else 0.0
            ),
        }
        maybe_log(metric_logger, runtime_payload, step=step)
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

    # RL-ADD: IQL warmup (load ckpt or run pretrain_iql_value/full).
    warmup_ckpt_raw = cfg.algorithm.q_learning.warmup_ckpt
    if iql_learner is None:
        print("[iql_warmup] skipped (algorithm.q_learning.enabled=false)")
    elif warmup_ckpt_raw is not None and str(warmup_ckpt_raw).strip().lower() not in ("", "null"):
        _load_iql_warmup_state(
            iql_learner,
            str(warmup_ckpt_raw),
            expected_state_feature_dim=int(shared_encoder.state_feature_dim),
            expected_chunk_feature_dim=int(shared_encoder.chunk_feature_dim),
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

    # RL-ADD: flip g_mode → advantage if requested.
    g_mode = str(cfg.algorithm.dipole.g_mode)
    if g_mode == "advantage":
        if iql_learner is None:
            raise RuntimeError(
                "g_mode='advantage' requires algorithm.q_learning.enabled=true."
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
    elif g_mode == "nnpu_frozen":
        # The frozen nnPU provider was attached before warmup.
        pass
    else:
        raise ValueError(f"Unknown algorithm.dipole.g_mode: {g_mode}")

    # env runtime settings
    if bool(cfg.intervention.enabled):
        device = build_device(env, cfg.intervention)
        intervention_runtime = RobosuiteInterventionRuntime(
            env=env,
            device=device,
            goal_update_mode=str(cfg.intervention.goal_update_mode),
        )

    try:
        nnpu_runtime = build_nnpu_runtime(
            cfg.algorithm.discriminator,
            policy_camera_names=list(agent.camera_names),
            shared_encoder=dynamics_encoder,
            discriminator=discriminator,
        )
        if nnpu_runtime is not None:
            nnpu_runtime.start()
            enter_listener.start()
            print(
                f"[nnPU HUD] scorer started device={nnpu_runtime.cfg.device} "
                f"fps={nnpu_runtime.cfg.fps:g} threshold={nnpu_runtime.discriminator.threshold:+.3f}"
            )
    except Exception as exc:
        nnpu_runtime = None
        print(f"[WARN] nnPU HUD/scorer disabled: {type(exc).__name__}: {exc}")
    obs, _ = reset_rollout_observation()
    reset_viewer_preview()
    agent.reset_policy_state()
    policy_gate.force_ready()
    spacemouse_gate.force_ready()
    if intervention_runtime is not None:
        intervention_runtime.start_episode()
    if nnpu_runtime is not None:
        nnpu_runtime.on_episode_reset()

    if async_updates and online_updates_enabled:
        trainer.start_async_worker()

    try:
        for step in range(start_step, int(cfg.runtime.max_steps)):
            last_step = step
            loop_start = time.monotonic() if control_limiter is None else control_limiter.wait()
            overall_fps_tracker.mark()

            while nnpu_runtime is not None and nnpu_runtime.pause_requested():
                render_nnpu_hud(
                    sys.__stdout__, nnpu_runtime.status(), step=step, episode_step=episode_step_index
                )
                resume_requested = enter_listener.consume()
                if intervention_runtime is not None and not resume_requested:
                    _, sampled_intervention, sampled_reset = intervention_runtime.maybe_override_action(
                        cached_policy_action
                    )
                    resume_requested = bool(sampled_intervention or sampled_reset)
                if resume_requested:
                    nnpu_runtime.resume()
                    print("\n[nnPU HUD] rollout resumed")
                    break
                refresh_main_viewer()
                time.sleep(0.02)

            new_policy_chunk = False
            if unthrottled_runtime or policy_gate.ready(loop_start):
                new_policy_chunk = agent.needs_action_chunk()
                cached_policy_action = agent.select_action(
                    obs, deterministic=bool(cfg.runtime.eval_deterministic)
                )

            # action: validation intervention
            env_action = np.asarray(cached_policy_action, dtype=np.float32)
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

            # reset the environment
            if reset_requested:
                if not bool(cfg.intervention.device_reset_as_episode_reset):
                    print("[INFO] Device reset requested. Exiting training loop.")
                    break
                episode_index += 1
                obs, _ = reset_rollout_observation()
                reset_viewer_preview()
                agent.reset_policy_state()
                episode_return = 0.0
                episode_length = 0
                episode_step_index = 0
                episode_transition_count = 0
                episode_intervention_transitions = 0
                policy_gate.force_ready()
                spacemouse_gate.force_ready()
                if intervention_runtime is not None:
                    intervention_runtime.start_episode()
                if nnpu_runtime is not None:
                    nnpu_runtime.on_episode_reset()
                maybe_report_runtime(step)
                continue

            if cached_is_intervention and cached_override_action is not None:
                env_action = np.asarray(cached_override_action, dtype=np.float32)
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
                    render_nnpu_hud(
                        sys.__stdout__,
                        nnpu_runtime.status(),
                        step=step,
                        episode_step=episode_step_index,
                    )

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

            # RL statistics
            if frozen_eval_mode and (episode_length + 1) >= eval_episode_max_steps:
                done = True
            done = bool(done or success)
            # RL-EDIT: tag `buffer_role="online"` so IQL replay can distinguish
            # online transitions from offline-bootstrap ones.
            info_payload = dict(info) if isinstance(info, dict) else {"raw_info": info}
            info_payload.setdefault("buffer_role", "online")
            info_payload.setdefault("episode_seed", env_random_reducer.seed_for_episode(episode_index))
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
                maybe_log(
                    metric_logger,
                    {
                        "intervention/total_transitions": int(total_intervention_transitions),
                        "intervention/global_ratio": float(total_intervention_transitions)
                        / float(total_transition_count),
                        "episode/current_intervention_transitions": int(
                            episode_intervention_transitions
                        ),
                        "episode/current_intervention_ratio": float(
                            episode_intervention_transitions
                        )
                        / float(episode_transition_count),
                    },
                    step=step,
                )
            episode_return += reward
            episode_length += 1
            success_count += int(success)

            # -----------------------------------------------------------------
            # update all
            # -----------------------------------------------------------------

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
                    train_payload = {
                        "train/pending_updates": float(
                            trainer.pending_async_updates() if async_updates else 0
                        ),
                        "train/online_buffer_size": float(len(agent.online_buffer)),
                        "train/demo_buffer_size": float(len(agent.demo_buffer)),
                        **{f"train/{key}": value for key, value in update_metrics.items()},
                    }
                    maybe_log(metric_logger, train_payload, step=step)
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
                maybe_log(
                    metric_logger,
                    {
                        "episode/return": float(episode_return),
                        "episode/length": int(episode_length),
                        "episode/success": int(success),
                        "episode/transition_count": int(episode_transition_count),
                        "episode/intervention_transitions": int(episode_intervention_transitions),
                        "episode/intervention_ratio": (
                            float(episode_intervention_transitions) / float(episode_transition_count)
                            if episode_transition_count > 0
                            else 0.0
                        ),
                        "buffer/online_size": len(agent.online_buffer),
                        "buffer/demo_size": len(agent.demo_buffer),
                        "intervention/global_ratio": (
                            float(total_intervention_transitions) / float(total_transition_count)
                            if total_transition_count > 0
                            else 0.0
                        ),
                    },
                    step=step,
                )
                runtime_logger.log(
                    {
                        "event": "episode_end",
                        "step": int(step),
                        "episode_index": int(episode_index),
                        "episode_seed": env_random_reducer.seed_for_episode(episode_index),
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
                episode_index += 1
                obs, _ = reset_rollout_observation()
                reset_viewer_preview()
                agent.reset_policy_state()
                episode_return = 0.0
                episode_length = 0
                episode_step_index = 0
                episode_transition_count = 0
                episode_intervention_transitions = 0
                cached_override_action = None
                cached_is_intervention = False
                policy_gate.force_ready()
                spacemouse_gate.force_ready()
                if intervention_runtime is not None:
                    intervention_runtime.start_episode()
                if nnpu_runtime is not None:
                    nnpu_runtime.on_episode_reset()
            else:
                obs = next_obs

            if step > 0 and step % int(cfg.logging.checkpoint_interval) == 0:
                request_checkpoint_save(step)
            maybe_report_runtime(step)
    finally:
        if nnpu_runtime is not None:
            nnpu_runtime.stop()
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
                "env_reset_seed": env_random_reducer.base_seed,
                "env_reset_seed_rule": "base_seed+episode_index" if env_random_reducer.enabled else None,
                "console_log": str(console_log_path),
                "runtime_log": str(runtime_log_path),
                "train_log": str(train_log_path),
                "tensorboard_log": None if metric_logger is None else str(metric_logger.log_dir),
                "buffer_dir": str(checkpoint_dir / "buffers"),
                "online_chunk_dir": str(checkpoint_dir / "buffers" / "online_chunks"),
                "demo_chunk_dir": str(checkpoint_dir / "buffers" / "demo_chunks"),
                "last_step": int(last_step),
                "episode_index": int(episode_index),
                "success_count": int(success_count),
                "latest_checkpoint": str(checkpoint_path(checkpoint_dir, "latest")),
                "trainer_state": trainer.state_dict(),
                "algorithm_type": "dipole_rl",
                "g_mode": str(cfg.algorithm.dipole.g_mode),
            },
        )
        if async_updates:
            trainer.close_async_worker(wait=False)
        if intervention_runtime is not None:
            intervention_runtime.close()
        maybe_log(
            metric_logger,
            {
                "run/success_count": int(success_count),
                "run/total_transition_count": int(total_transition_count),
                "run/total_intervention_transitions": int(total_intervention_transitions),
                "run/global_intervention_ratio": (
                    float(total_intervention_transitions) / float(total_transition_count)
                    if total_transition_count > 0
                    else 0.0
                ),
            },
            step=last_step if last_step >= 0 else 0,
        )
        if metric_logger is not None:
            metric_logger.close()
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
