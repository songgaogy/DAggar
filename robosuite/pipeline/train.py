"""Train the single supported AWR baseline in robosuite."""

from __future__ import annotations

import datetime
import time
from contextlib import ExitStack
from pathlib import Path
import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from robosuite.pipeline.bootstrap import (
    build_agent_config,
    build_qv_cache_metadata,
    resolve_qv_cache_path,
    try_load_qv_cache,
    write_qv_cache,
)
from robosuite.pipeline.src.awr import AWRAgent, AWRTrainer
from robosuite.pipeline.src.data import TransitionChunkWriter, load_demos
from robosuite.pipeline.src.environment import (
    InterventionRuntime,
    bind_proprio_extractor,
    build_policy_observation,
    build_robosuite_env,
    build_runtime_config,
    build_spacemouse,
    compute_grasp_penalty,
    flow_checkpoint_settings,
    load_flow_checkpoint,
    load_task_metadata,
    render_mjviewer,
    reset_policy_observation,
    sparse_success_reward,
)
from robosuite.pipeline.utils import (
    ConsoleLogCapture,
    JsonlEventLogger,
    TensorBoardLogger,
    create_run_directory,
    require_cuda,
    resolve_path,
    set_seed,
    write_json,
)

def _checkpoint_reference(cfg: DictConfig) -> Path | None:
    explicit = resolve_path(cfg.checkpoint.path)
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {explicit}")
        return explicit
    if not bool(cfg.checkpoint.resume):
        return None
    root = Path(to_absolute_path(str(cfg.logging.output_root))) / str(cfg.task.name)
    candidates = sorted(
        root.glob(f"*/{cfg.checkpoint.directory}/latest.pt"),
        key=lambda path: path.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def _save_checkpoint(
    agent: AWRAgent,
    trainer: AWRTrainer,
    run_dir: Path,
    cfg: DictConfig,
    *,
    episode_index: int,
    success_count: int,
) -> Path:
    checkpoint_dir = run_dir / str(cfg.checkpoint.directory)
    completed_episodes = int(episode_index)
    trainer_state = {
        **trainer.state_dict(),
        "episode_index": int(episode_index),
        "success_count": int(success_count),
    }
    episode_path = checkpoint_dir / f"episode_{completed_episodes:08d}.pt"
    latest_path = checkpoint_dir / "latest.pt"
    agent.save_checkpoint(
        episode_path,
        include_buffers=False,
        trainer_state=trainer_state,
    )
    agent.save_checkpoint(
        latest_path,
        include_buffers=True,
        trainer_state=trainer_state,
    )
    return episode_path


def _checkpoint_due(total_episodes: int, interval_episodes: int) -> bool:
    return total_episodes > 0 and total_episodes % interval_episodes == 0


def _format_episode_summary(
    episode_index: int,
    reason: str,
    metrics: dict[str, float],
) -> str:
    return (
        f"[episode] index={episode_index} reason={reason} "
        f"success={int(metrics['success'])} return={metrics['return']:.2f} "
        f"length={int(metrics['length'])} fps={metrics['env_fps']:.1f} "
        f"intervention_steps={int(metrics['intervention_steps'])} "
        f"intervention_rate={metrics['intervention_rate']:.3f} "
        f"updates={int(metrics['learner_updates'])} "
        f"learner_sec={metrics['learner_seconds']:.2f} "
        f"reset_sec={metrics['reset_seconds']:.2f} "
        f"checkpoint_sec={metrics['checkpoint_seconds']:.2f} "
        f"pause_sec={metrics['pause_seconds']:.2f} "
        f"boundary_sec={metrics['boundary_seconds']:.2f}"
    )


class _RateLimiter:
    def __init__(self, frequency: float) -> None:
        self.period = 1.0 / float(frequency)
        self.deadline: float | None = None

    def wait(self) -> None:
        now = time.monotonic()
        if self.deadline is None:
            self.deadline = now + self.period
            return
        if self.deadline > now:
            time.sleep(self.deadline - now)
        self.deadline = max(self.deadline + self.period, time.monotonic())


def _run(cfg: DictConfig, resources: ExitStack) -> None:
    require_cuda()
    if not bool(cfg.intervention.enabled):
        raise ValueError("The AWR-only training pipeline requires human intervention.")
    if str(cfg.intervention.device) != "spacemouse":
        raise ValueError("The AWR-only training pipeline only supports SpaceMouse.")
    if not bool(cfg.runtime.interactive) or not bool(cfg.runtime.viewer_enabled):
        raise ValueError("AWR training requires the interactive mjviewer runtime.")
    set_seed(int(cfg.seed))
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    init_checkpoint, init_payload = load_flow_checkpoint(cfg.checkpoint.init_path)
    if init_checkpoint is None or init_payload is None:
        raise FileNotFoundError("checkpoint.init_path must reference a flow-multi checkpoint.")
    settings = flow_checkpoint_settings(init_payload, str(cfg.task.name))
    env_metadata = load_task_metadata(init_payload, str(cfg.task.name))
    if env_metadata is None:
        raise KeyError(f"Flow checkpoint has no environment metadata for {cfg.task.name}.")
    requested_cameras = [str(name) for name in cfg.env.camera_names]
    checkpoint_cameras = [str(name) for name in settings.get("camera_names", [])]
    camera_names = (
        checkpoint_cameras
        if bool(cfg.checkpoint.use_init_camera_names) and checkpoint_cameras
        else requested_cameras
    )
    camera_aliases = {
        str(key): str(value)
        for key, value in dict(getattr(cfg.awr, "camera_aliases", {}) or {}).items()
    }

    run_name, run_dir = create_run_directory(
        to_absolute_path(str(cfg.logging.output_root)),
        str(cfg.task.name),
        cfg.logging.run_name,
    )
    console = ConsoleLogCapture(run_dir / str(cfg.logging.console_filename))
    events = JsonlEventLogger(run_dir / str(cfg.logging.metrics_filename))
    tensorboard = TensorBoardLogger(
        run_dir / str(cfg.tensorboard.directory),
        enabled=bool(cfg.tensorboard.enabled),
        flush_secs=int(cfg.tensorboard.flush_secs),
    )
    console.start()
    events.start()
    resources.callback(console.stop)
    resources.callback(events.close)
    resources.callback(tensorboard.close)
    OmegaConf.save(cfg, run_dir / str(cfg.logging.resolved_config_filename))

    runtime_config = build_runtime_config(
        env_metadata,
        camera_names=camera_names,
        image_height=int(cfg.env.img_height),
        image_width=int(cfg.env.img_width),
        control_freq=int(cfg.env.control_freq),
        horizon=int(cfg.env.horizon),
        interactive=True,
    )
    env = build_robosuite_env(runtime_config)
    resources.callback(env.close)
    extractor = bind_proprio_extractor(env, env_metadata)
    resources.callback(extractor.close)
    obs, _ = reset_policy_observation(
        env,
        preserve_mjviewer=True,
        extractor=extractor,
        camera_names=camera_names,
        camera_aliases=camera_aliases,
        image_height=int(cfg.env.img_height),
        image_width=int(cfg.env.img_width),
    )
    action_low, action_high = (
        np.asarray(value, dtype=np.float32) for value in env.action_spec
    )
    agent = AWRAgent.from_config(
        build_agent_config(cfg, camera_names=camera_names, flow_settings=settings),
        observation_example=obs,
        action_low=action_low,
        action_high=action_high,
    )
    trainer = AWRTrainer(agent)

    checkpoint = _checkpoint_reference(cfg)
    episode_index = 0
    success_count = 0
    if checkpoint is None:
        agent.load_flow_policy_checkpoint(init_checkpoint, task_name=str(cfg.task.name))
    else:
        trainer_state = agent.load_checkpoint(
            checkpoint,
            load_buffers=bool(cfg.checkpoint.load_buffers),
        )
        trainer.load_state_dict(trainer_state)
        episode_index = int(trainer_state.get("episode_index", 0))
        success_count = int(trainer_state.get("success_count", 0))
        print(f"[resume] {checkpoint}")

    demos = load_demos(
        data_root=to_absolute_path(str(cfg.data.demo_root)),
        task_name=str(cfg.data.task_name),
        camera_names=camera_names,
        camera_aliases=camera_aliases,
        image_height=int(cfg.env.img_height),
        image_width=int(cfg.env.img_width),
        state_extractor=extractor,
        control_freq=int(cfg.env.control_freq),
        horizon=int(cfg.env.horizon),
        trajectory_limits={
            "expert": int(cfg.data.expert_num_trajectories),
            "success_rollout": int(cfg.data.success_num_trajectories),
            "fail_rollout": int(cfg.data.fail_num_trajectories),
        },
        split_directories={
            "expert": to_absolute_path(str(cfg.data.expert_dir)),
            "success_rollout": to_absolute_path(str(cfg.data.success_dir)),
            "fail_rollout": to_absolute_path(str(cfg.data.fail_dir)),
        },
        cache_dir=(
            Path(to_absolute_path(str(cfg.logging.output_root)))
            / str(cfg.task.name)
            / "_demo_cache"
        ),
        mirror_cache_dir=run_dir / "demo_cache",
        seed=int(cfg.seed),
    )
    if not demos["expert"]:
        raise RuntimeError("AWR requires non-empty expert demonstrations.")
    if len(agent.demo_buffer) == 0:
        trainer.bootstrap_demo_buffer(demos["expert"])
    if not agent.has_normalizers():
        agent.fit_normalizers_from_transitions(demos["expert"])

    cache_path = resolve_qv_cache_path(cfg)
    cache_metadata = build_qv_cache_metadata(
        cfg,
        camera_names=camera_names,
        init_checkpoint=init_checkpoint,
        agent=agent,
    )
    cache_loaded = False
    direct_cache_load = bool(cfg.qv_cache.direct_load)
    if checkpoint is None and bool(cfg.qv_cache.enabled) and not bool(cfg.qv_cache.force_rebuild):
        cache_loaded = try_load_qv_cache(
            cfg,
            agent,
            trainer,
            path=cache_path,
            expected_metadata=cache_metadata,
        )
    if (
        direct_cache_load
        and bool(cfg.qv_cache.require_existing)
        and not cache_loaded
    ):
        raise FileNotFoundError(f"Required Q/V cache was not loaded: {cache_path}")
    if len(agent.online_buffer) == 0 and not (cache_loaded and direct_cache_load):
        if not demos["success_rollout"] or not demos["fail_rollout"]:
            raise RuntimeError("AWR value warmup requires success_rollout and fail_rollout.")
        trainer.bootstrap_online_buffer(
            demos["success_rollout"],
            demo_source="success_rollout",
            episode_namespace="success_rollout",
        )
        trainer.bootstrap_online_buffer(
            demos["fail_rollout"],
            demo_source="fail_rollout",
            episode_namespace="fail_rollout",
        )
    if checkpoint is None and not cache_loaded:
        remaining = max(
            0,
            int(cfg.trainer.value_warmup_steps)
            - int(trainer.total_value_warmup_updates),
        )
        for _ in range(remaining):
            metrics = trainer.pretrain_value(1)[0]
            warmup_step = int(trainer.total_value_warmup_updates)
            if warmup_step % int(cfg.logging.log_interval_updates) == 0:
                tensorboard.log(metrics, step=warmup_step, prefix="warmup")
                events.log({"event": "warmup", "step": warmup_step, **metrics})
        if bool(cfg.qv_cache.enabled) and bool(cfg.qv_cache.save):
            write_qv_cache(
                agent,
                trainer,
                path=cache_path,
                metadata=cache_metadata,
            )
    metadata = {
        "format": "awr_run_v1",
        "run_name": run_name,
        "task": str(cfg.task.name),
        "seed": int(cfg.seed),
        "init_checkpoint": str(init_checkpoint),
        "resumed_checkpoint": None if checkpoint is None else str(checkpoint),
        "camera_names": camera_names,
        "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    write_json(run_dir / str(cfg.logging.metadata_filename), metadata)
    print(f"[run] {run_name}")
    print(f"[path] {run_dir}")

    device = build_spacemouse(
        env,
        pos_sensitivity=float(cfg.intervention.pos_sensitivity),
        rot_sensitivity=float(cfg.intervention.rot_sensitivity),
    )
    intervention = InterventionRuntime(
        env,
        device,
        goal_update_mode=str(cfg.intervention.goal_update_mode),
    )
    resources.callback(intervention.close)
    intervention.start_episode()
    writer = TransitionChunkWriter(
        run_dir / str(cfg.checkpoint.buffer_directory),
        chunk_size=int(cfg.checkpoint.buffer_interval_env_steps),
    )
    resources.callback(writer.close)
    limiter = _RateLimiter(float(cfg.runtime.control_fps))
    episode_return = 0.0
    episode_length = 0
    episode_interventions = 0
    episodes_per_train = int(cfg.trainer.episodes_per_train)
    if episodes_per_train <= 0:
        raise ValueError("trainer.episodes_per_train must be positive.")
    episodes_since_train = int(episode_index) % episodes_per_train
    last_runtime_log = time.monotonic()
    last_runtime_step = int(trainer.total_env_steps)
    episode_started_at = time.perf_counter()

    def finish_episode(
        *,
        success: bool,
        reason: str,
        pause_after: bool,
    ) -> None:
        nonlocal obs
        nonlocal episode_index
        nonlocal episode_return
        nonlocal episode_length
        nonlocal episode_interventions
        nonlocal episode_started_at
        nonlocal episodes_since_train

        collection_seconds = max(
            time.perf_counter() - episode_started_at,
            1e-6,
        )
        boundary_started_at = time.perf_counter()
        episodes_since_train += 1
        metrics_list: list[dict[str, float]] = []
        learner_seconds = 0.0
        if episodes_since_train >= episodes_per_train:
            learner_started_at = time.perf_counter()
            metrics_list = trainer.train_episode(
                updates=int(cfg.trainer.updates_per_train),
                episodes=episodes_since_train,
                show_progress=True,
            )
            learner_seconds = time.perf_counter() - learner_started_at
            episodes_since_train = 0
            for metrics in metrics_list:
                update = int(metrics["learner/actor_updates"])
                if update % int(cfg.logging.log_interval_updates) == 0:
                    tensorboard.log(metrics, step=update, prefix="train")
                    events.log({"event": "train", "step": update, **metrics})

        checkpoint_seconds = 0.0
        completed_episodes = episode_index + 1
        if _checkpoint_due(
            completed_episodes,
            int(cfg.checkpoint.interval_episodes),
        ):
            checkpoint_started_at = time.perf_counter()
            saved = _save_checkpoint(
                agent,
                trainer,
                run_dir,
                cfg,
                episode_index=completed_episodes,
                success_count=success_count,
            )
            tensorboard.log(
                {"saved": 1.0},
                step=completed_episodes,
                prefix="checkpoint",
            )
            checkpoint_seconds = time.perf_counter() - checkpoint_started_at
            print(f"[checkpoint] {saved}")

        pause_seconds = 0.0
        if pause_after and float(cfg.runtime.episode_pause_sec) > 0:
            pause_started_at = time.perf_counter()
            time.sleep(float(cfg.runtime.episode_pause_sec))
            pause_seconds = time.perf_counter() - pause_started_at

        reset_started_at = time.perf_counter()
        obs, _ = reset_policy_observation(
            env,
            preserve_mjviewer=True,
            extractor=extractor,
            camera_names=camera_names,
            camera_aliases=camera_aliases,
            image_height=int(cfg.env.img_height),
            image_width=int(cfg.env.img_width),
        )
        reset_seconds = time.perf_counter() - reset_started_at
        agent.reset_policy_state()
        intervention.start_episode()
        boundary_seconds = time.perf_counter() - boundary_started_at
        episode_metrics = {
            "return": float(episode_return),
            "length": float(episode_length),
            "success": float(success),
            "env_fps": float(episode_length) / collection_seconds,
            "intervention_steps": float(episode_interventions),
            "intervention_rate": float(episode_interventions)
            / max(1, episode_length),
            "learner_updates": float(len(metrics_list)),
            "learner_seconds": learner_seconds,
            "reset_seconds": reset_seconds,
            "checkpoint_seconds": checkpoint_seconds,
            "pause_seconds": pause_seconds,
            "boundary_seconds": boundary_seconds,
            "online_size": float(len(agent.online_buffer)),
            "demo_size": float(len(agent.demo_buffer)),
        }
        tensorboard.log(episode_metrics, step=episode_index, prefix="episode")
        events.log(
            {
                "event": "episode",
                "episode": episode_index,
                "reason": reason,
                **episode_metrics,
            }
        )
        print(_format_episode_summary(episode_index, reason, episode_metrics))

        episode_index += 1
        episode_return = 0.0
        episode_length = 0
        episode_interventions = 0
        episode_started_at = time.perf_counter()

    while trainer.total_env_steps < int(cfg.runtime.max_env_steps):
        limiter.wait()
        action = np.asarray(agent.select_action(obs, deterministic=False), dtype=np.float32)
        override, active, reset_requested = intervention.override(action)
        if reset_requested:
            if not bool(cfg.intervention.device_reset_as_episode_reset):
                break
            completed_episode = episode_length > 0
            if completed_episode:
                finish_episode(
                    success=False,
                    reason="device_reset",
                    pause_after=False,
                )
            else:
                obs, _ = reset_policy_observation(
                    env,
                    preserve_mjviewer=True,
                    extractor=extractor,
                    camera_names=camera_names,
                    camera_aliases=camera_aliases,
                    image_height=int(cfg.env.img_height),
                    image_width=int(cfg.env.img_width),
                )
                agent.reset_policy_state()
                intervention.start_episode()
                episode_started_at = time.perf_counter()
            continue
        if active and override is not None:
            action = np.asarray(override, dtype=np.float32)
            agent.notify_intervention()

        grasp_penalty = compute_grasp_penalty(
            env,
            action,
            penalty=float(cfg.grasp_penalty.penalty),
            command_threshold=float(cfg.grasp_penalty.command_threshold),
            open_threshold=float(cfg.grasp_penalty.open_threshold),
            closed_threshold=float(cfg.grasp_penalty.closed_threshold),
        )
        step_result = env.step(action)
        if len(step_result) == 5:
            raw_next_obs, _, terminated, truncated, info = step_result
        else:
            raw_next_obs, _, terminated, info = step_result
            truncated = False
        info = dict(info) if isinstance(info, dict) else {"raw_info": info}
        reward, success = sparse_success_reward(env, info)
        reward *= float(cfg.awr.success_reward_scale)
        terminated = bool(terminated or success)
        truncated = bool(
            truncated or episode_length + 1 >= int(cfg.env.horizon)
        )
        done = bool(terminated or truncated)
        info["is_success"] = bool(success)
        info["truncated"] = bool(truncated)
        if grasp_penalty is not None:
            info["grasp_penalty"] = float(grasp_penalty)
        next_obs = build_policy_observation(
            env,
            extractor=extractor,
            camera_names=camera_names,
            camera_aliases=camera_aliases,
            image_height=int(cfg.env.img_height),
            image_width=int(cfg.env.img_width),
            raw_observation=raw_next_obs,
        )
        transition = trainer.record_transition(
            obs=obs,
            action=action,
            next_obs=next_obs,
            done=done,
            reward=reward,
            grasp_penalty=grasp_penalty,
            is_intervention=bool(active),
            info=info,
            reward_source="env_success",
            demo_source="intervention" if active else None,
            episode_index=episode_index,
            episode_step=episode_length,
        )
        writer.append(transition)
        episode_return += reward
        episode_length += 1
        episode_interventions += int(active)
        success_count += int(success)
        render_mjviewer(
            env,
            visualize_gripper_markers=bool(
                cfg.runtime.visualize_gripper_markers
            ),
        )

        if done:
            finish_episode(
                success=success,
                reason="environment",
                pause_after=True,
            )
        else:
            obs = next_obs

        now = time.monotonic()
        if now - last_runtime_log >= float(cfg.logging.runtime_log_interval_seconds):
            elapsed = max(now - last_runtime_log, 1e-6)
            current_env_step = int(trainer.total_env_steps)
            runtime_metrics = {
                "env_steps": float(current_env_step),
                "env_fps": float(current_env_step - last_runtime_step) / elapsed,
                "value_updates": float(trainer.total_value_updates),
                "actor_updates": float(trainer.total_actor_updates),
                "online_size": float(len(agent.online_buffer)),
                "demo_size": float(len(agent.demo_buffer)),
            }
            tensorboard.log(
                runtime_metrics,
                step=int(trainer.total_env_steps),
                prefix="runtime",
            )
            events.log({"event": "runtime", **runtime_metrics})
            last_runtime_log = now
            last_runtime_step = current_env_step

    writer.flush()
    final = _save_checkpoint(
        agent,
        trainer,
        run_dir,
        cfg,
        episode_index=episode_index,
        success_count=success_count,
    )
    print(f"[done] {final}")


@hydra.main(version_base="1.3", config_path="./config", config_name="overall")
def main(cfg: DictConfig) -> None:
    with ExitStack() as resources:
        _run(cfg, resources)


if __name__ == "__main__":
    main()
