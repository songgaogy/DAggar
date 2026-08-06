"""Train the HIL-SERL baseline in robosuite."""

from __future__ import annotations

import datetime
import json
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from robosuite.pipeline.src.data.demos import (
    load_demo_paths,
    load_hdf5_demos_into_transitions,
    resolve_task_demo_paths,
)
from robosuite.pipeline.src.data.transitions import AsyncTransitionChunkWriter
from robosuite.pipeline.src.environment import (
    RobosuiteInterventionRuntime,
    RobosuiteObservationAdapter,
    build_device,
    build_robosuite_env,
    compute_grasp_penalty,
    sparse_success_reward,
    unpack_robosuite_step,
)
from robosuite.pipeline.src.hil_serl import HILSERLAgent, HILSERLTrainer
from robosuite.pipeline.utils.logging import (
    ConsoleLogCapture,
    JsonlEventLogger,
    TensorBoardLogger,
)
from robosuite.pipeline.utils.runtime import (
    EMAFpsTracker,
    FixedRateLimiter,
    IntervalGate,
    build_runtime_cfg,
    maybe_wrap_visualization,
    now_readable,
    reset_observation_adapter,
    resolve_camera_names,
    set_seed,
    write_resolved_config,
)


def _algorithm_config(cfg: DictConfig) -> dict[str, Any]:
    algorithm = OmegaConf.to_container(cfg.algorithm, resolve=True)
    if not isinstance(algorithm, dict):
        raise TypeError("algorithm must resolve to a mapping.")
    encoder = algorithm["encoder"]
    encoder["pretrained_path"] = to_absolute_path(str(encoder["pretrained_path"]))
    algorithm["sac"]["device"] = str(cfg.runtime.learner_device)
    algorithm["sac"]["inference_device"] = str(cfg.runtime.inference_device)
    trainer = algorithm["trainer"]
    trainer["warmup_steps"] = int(trainer.pop("training_starts"))
    trainer["steps_per_update"] = int(trainer.pop("policy_publish_interval"))
    trainer["max_learner_steps"] = int(cfg.runtime.max_learner_steps)
    return algorithm


def _run_directory(cfg: DictConfig) -> tuple[str, Path]:
    root = Path(to_absolute_path(str(cfg.logging.output_root))) / str(cfg.task.name)
    run_name = f"{cfg.logging.run_name or cfg.task.name}_{now_readable()}"
    run_dir = root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_name, run_dir


def _checkpoint_reference(cfg: DictConfig) -> Path | None:
    if cfg.checkpoint.path is not None:
        path = Path(to_absolute_path(str(cfg.checkpoint.path)))
        if path.is_dir():
            path = path / str(cfg.checkpoint.directory) / "latest.pt"
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        return path
    if not bool(cfg.checkpoint.resume):
        return None
    root = Path(to_absolute_path(str(cfg.logging.output_root))) / str(cfg.task.name)
    prefix = str(cfg.logging.run_name or cfg.task.name)
    candidates = sorted(
        root.glob(f"{prefix}_*/{cfg.checkpoint.directory}/latest.pt"),
        key=lambda path: path.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def _write_metadata(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _episode_checkpoint_due(episode_index: int, interval: int) -> bool:
    return interval > 0 and episode_index > 0 and episode_index % interval == 0


def _drain_learner_metrics(
    trainer: HILSERLTrainer,
    tensorboard: TensorBoardLogger,
    events: JsonlEventLogger,
    *,
    log_interval: int,
) -> None:
    for metrics in trainer.drain_async_metrics():
        learner_step = int(metrics["learner_total_updates"])
        if learner_step % max(1, int(log_interval)) == 0 or bool(metrics.get("learner_published", 0.0)):
            tensorboard.log(metrics, step=learner_step, prefix="train")
            events.log({"event": "learner_update", "learner_step": learner_step, **metrics})


def _run(cfg: DictConfig, resources: ExitStack) -> None:
    set_seed(int(cfg.seed))
    run_name, run_dir = _run_directory(cfg)
    console = ConsoleLogCapture(run_dir / str(cfg.logging.console_filename))
    events = JsonlEventLogger(run_dir / str(cfg.logging.metrics_filename))
    console.start()
    resources.callback(console.stop)
    events.start()
    resources.callback(events.close)
    tensorboard = TensorBoardLogger(
        run_dir / str(cfg.tensorboard.directory),
        enabled=bool(cfg.tensorboard.enabled),
        flush_secs=int(cfg.tensorboard.flush_secs),
    )
    resources.callback(tensorboard.close)
    buffer_writer = AsyncTransitionChunkWriter(
        run_dir / str(cfg.checkpoint.buffer_directory),
        chunk_size=int(cfg.checkpoint.buffer_interval_env_steps),
        event_logger=events.log,
    )
    buffer_writer.start()
    resources.callback(buffer_writer.close)
    write_resolved_config(cfg, run_dir)

    camera_names = resolve_camera_names(cfg)
    interactive = bool(cfg.runtime.interactive and cfg.runtime.viewer_enabled)
    runtime_cfg = build_runtime_cfg(
        cfg,
        camera_names,
        has_renderer=interactive,
        has_offscreen_renderer=False,
        renderer=str(cfg.env.renderer),
    )
    env = maybe_wrap_visualization(
        build_robosuite_env(runtime_cfg),
        enabled=bool(cfg.runtime.visualize_gripper_markers),
        label="training env",
    )
    resources.callback(env.close)
    if interactive:
        render_env = env
        if str(runtime_cfg.renderer).lower() == "mujoco":
            env.viewer.width = int(cfg.env.viewer_width)
            env.viewer.height = int(cfg.env.viewer_height)
    else:
        render_cfg = build_runtime_cfg(
            cfg,
            camera_names,
            has_renderer=False,
            has_offscreen_renderer=True,
        )
        render_env = maybe_wrap_visualization(
            build_robosuite_env(render_cfg),
            enabled=bool(cfg.runtime.visualize_gripper_markers),
            label="observation render env",
        )
        resources.callback(render_env.close)
        render_env.reset()
    adapter = RobosuiteObservationAdapter(
        env,
        render_env=render_env,
        camera_names=camera_names,
        img_height=int(cfg.env.img_height),
        img_width=int(cfg.env.img_width),
        proprio_keys=tuple(cfg.env.proprio_keys or []),
        image_obs_fps=float(cfg.runtime.image_obs_fps),
    )
    obs, _ = reset_observation_adapter(adapter, preserve_mjviewer=interactive)
    action_low, action_high = adapter.action_spec()
    agent = HILSERLAgent.from_config(
        _algorithm_config(cfg),
        observation_example=obs,
        action_low=action_low,
        action_high=action_high,
    )
    trainer = HILSERLTrainer(agent)

    checkpoint = _checkpoint_reference(cfg)
    episode_index = 0
    success_count = 0
    if checkpoint is not None:
        extra = agent.load_checkpoint(checkpoint, load_buffers=bool(cfg.checkpoint.load_buffers))
        trainer.load_state_dict(extra.get("trainer_state"))
        episode_index = int(extra.get("episode_index", 0))
        success_count = int(extra.get("success_count", 0))
        print(f"[load] {checkpoint}")

    demo_paths = resolve_task_demo_paths(
        str(cfg.data.demo_task),
        data_root=to_absolute_path(str(cfg.data.demo_root)),
        split="expert",
    )
    if not demo_paths:
        raise FileNotFoundError(f"No expert demonstrations found under {cfg.data.demo_path}.")
    selected_demo_manifest: list[dict[str, str]] = []
    demo_root = Path(to_absolute_path(str(cfg.data.demo_root))).resolve()

    def record_selected_demos(path: Path, demo_names: list[str]) -> None:
        source_path = path.resolve()
        try:
            source = str(source_path.relative_to(demo_root))
        except ValueError:
            source = str(source_path)
        selected_demo_manifest.extend(
            {"source": source, "demo_name": str(demo_name)} for demo_name in demo_names
        )

    demo_transitions = load_demo_paths(
        demo_paths,
        cache_dir=Path(to_absolute_path(str(cfg.logging.output_root))) / "_demo_cache",
        mirror_cache_dir=run_dir / "demo_cache",
        hdf5_loader=lambda path, demo_names=None: load_hdf5_demos_into_transitions(
            path,
            camera_names=camera_names,
            img_height=int(cfg.env.img_height),
            img_width=int(cfg.env.img_width),
            proprio_keys=tuple(cfg.env.proprio_keys or []),
            renderer=str(cfg.env.renderer),
            control_freq=int(cfg.env.control_freq),
            grasp_penalty=float(cfg.algorithm.grasp_penalty.penalty),
            grasp_command_threshold=float(cfg.algorithm.grasp_penalty.command_threshold),
            gripper_open_threshold=float(cfg.algorithm.grasp_penalty.open_threshold),
            gripper_closed_threshold=float(cfg.algorithm.grasp_penalty.closed_threshold),
            demo_names=demo_names,
        ),
        max_num_trajectories=cfg.data.num_trajectories,
        random_sample=bool(cfg.data.random_sample),
        random_seed=None if cfg.data.random_seed is None else int(cfg.data.random_seed),
        selected_demo_callback=record_selected_demos,
        cache_key=(
            f"{cfg.task.name}_{cfg.env.img_height}x{cfg.env.img_width}_{'_'.join(camera_names)}"
            f"_proprio-{'-'.join(cfg.env.proprio_keys) if cfg.env.proprio_keys else 'none'}"
            "_reward01_gripper-openness-abs-range-v1"
            f"_gp{cfg.algorithm.grasp_penalty.penalty}"
            f"_gc{cfg.algorithm.grasp_penalty.command_threshold}"
            f"_go{cfg.algorithm.grasp_penalty.open_threshold}"
            f"_gx{cfg.algorithm.grasp_penalty.closed_threshold}"
        ),
    )
    selected_demo_manifest.sort(key=lambda item: (item["source"], item["demo_name"]))
    if (
        cfg.data.num_trajectories is not None
        and len(selected_demo_manifest) != int(cfg.data.num_trajectories)
    ):
        raise RuntimeError(
            "Demo selection manifest does not match the requested trajectory count: "
            f"requested {cfg.data.num_trajectories}, selected {len(selected_demo_manifest)}."
        )
    _write_metadata(
        run_dir / str(cfg.data.selection_manifest_filename),
        {
            "num_requested": (
                None if cfg.data.num_trajectories is None else int(cfg.data.num_trajectories)
            ),
            "random_sample": bool(cfg.data.random_sample),
            "random_seed": None if cfg.data.random_seed is None else int(cfg.data.random_seed),
            "num_selected": len(selected_demo_manifest),
            "trajectories": selected_demo_manifest,
        },
    )
    if not demo_transitions:
        raise RuntimeError("Expert demonstration conversion produced no transitions.")
    if len(agent.demo_buffer) == 0:
        trainer.bootstrap_demo_buffer(demo_transitions)
    print(f"[run] {run_name}")
    print(f"[path] {run_dir}")
    print(f"[demo] {len(agent.demo_buffer)} transitions")

    intervention = None
    if bool(cfg.intervention.enabled):
        device = build_device(env, cfg.intervention)
        intervention = RobosuiteInterventionRuntime(
            env,
            device,
            goal_update_mode=str(cfg.intervention.goal_update_mode),
        )
        resources.callback(intervention.close)
        intervention.start_episode()

    trainer.start_async_worker()
    control_limiter = FixedRateLimiter(float(cfg.runtime.control_fps))
    policy_gate = IntervalGate(float(cfg.runtime.policy_fps))
    intervention_gate = IntervalGate(float(cfg.runtime.spacemouse_fps))
    fps = EMAFpsTracker()
    last_runtime_log = time.monotonic()
    cached_policy_action = np.zeros_like(action_low, dtype=np.float32)
    cached_override_action: np.ndarray | None = None
    cached_is_intervention = False
    episode_return = 0.0
    episode_length = 0
    episode_intervention_steps = 0
    episode_intervention_segments = 0
    total_intervention_steps = 0
    total_intervention_segments = 0
    last_runtime_update = trainer.total_updates
    stage_seconds = {"policy": 0.0, "intervention": 0.0, "environment": 0.0, "replay": 0.0}
    started_at = datetime.datetime.now().isoformat(timespec="seconds")
    metadata_path = run_dir / str(cfg.logging.metadata_filename)

    def save_checkpoint(tag: str, *, resume_learner: bool) -> None:
        trainer.flush_async_updates(timeout=120.0)
        path = run_dir / str(cfg.checkpoint.directory) / f"{tag}.pt"
        agent.save_checkpoint(
            path,
            include_buffers=True,
            extra={
                "trainer_state": trainer.state_dict(),
                "episode_index": int(episode_index),
                "success_count": int(success_count),
                "run_name": run_name,
            },
        )
        if tag != "latest":
            latest = run_dir / str(cfg.checkpoint.directory) / "latest.pt"
            agent.save_checkpoint(
                latest,
                include_buffers=True,
                extra={
                    "trainer_state": trainer.state_dict(),
                    "episode_index": int(episode_index),
                    "success_count": int(success_count),
                    "run_name": run_name,
                },
            )
        if resume_learner and not trainer.learner_finished:
            trainer.start_async_worker()

    def save_episode_checkpoint_if_due() -> None:
        checkpoint_interval = int(cfg.checkpoint.interval_online_episodes)
        if _episode_checkpoint_due(episode_index, checkpoint_interval):
            save_checkpoint(f"episode_{episode_index:08d}", resume_learner=True)

    try:
        while trainer.total_env_steps < int(cfg.runtime.max_env_steps):
            trainer.raise_if_failed()
            now = control_limiter.wait()
            fps.mark()
            env_step = trainer.total_env_steps
            if policy_gate.ready(now):
                stage_started = time.perf_counter()
                if env_step < int(agent.trainer_config.random_steps):
                    cached_policy_action = np.random.uniform(action_low, action_high).astype(np.float32)
                else:
                    cached_policy_action = agent.select_action(obs, deterministic=False)
                stage_seconds["policy"] += time.perf_counter() - stage_started

            action = np.asarray(cached_policy_action, dtype=np.float32)
            reset_requested = False
            if intervention is not None and intervention_gate.ready(now):
                stage_started = time.perf_counter()
                was_intervening = cached_is_intervention
                override, active, reset_requested = intervention.maybe_override_action(cached_policy_action)
                cached_override_action = np.asarray(override, dtype=np.float32) if active else None
                cached_is_intervention = bool(active)
                if cached_is_intervention and not was_intervening:
                    episode_intervention_segments += 1
                    total_intervention_segments += 1
                stage_seconds["intervention"] += time.perf_counter() - stage_started
            if reset_requested:
                print(
                    f"[episode] index={episode_index} reason=manual_reset "
                    f"return={episode_return:.2f} length={episode_length} "
                    f"intervention_rate={episode_intervention_steps / max(1, episode_length):.3f} "
                    f"updates={trainer.total_updates}"
                )
                obs, _ = reset_observation_adapter(adapter, preserve_mjviewer=interactive)
                episode_return = 0.0
                episode_length = 0
                episode_intervention_steps = 0
                episode_intervention_segments = 0
                episode_index += 1
                save_episode_checkpoint_if_due()
                policy_gate.force_ready()
                intervention_gate.force_ready()
                intervention.start_episode()
                continue
            if cached_is_intervention and cached_override_action is not None:
                action = cached_override_action

            penalty_cfg = cfg.algorithm.grasp_penalty
            grasp_penalty = compute_grasp_penalty(
                env,
                action,
                penalty=float(penalty_cfg.penalty),
                command_threshold=float(penalty_cfg.command_threshold),
                open_threshold=float(penalty_cfg.open_threshold),
                closed_threshold=float(penalty_cfg.closed_threshold),
            )
            stage_started = time.perf_counter()
            step_output = env.step(action)
            if interactive:
                env.render()
            stage_seconds["environment"] += time.perf_counter() - stage_started
            raw_next_obs, _, terminated, truncated, info = unpack_robosuite_step(step_output)
            info = dict(info) if isinstance(info, dict) else {"raw_info": info}
            reward, is_success = sparse_success_reward(env, info)
            terminated = bool(terminated or is_success)
            if terminated:
                truncated = False
            info["is_success"] = bool(is_success)
            if grasp_penalty is not None:
                info["grasp_penalty"] = float(grasp_penalty)
            next_obs = adapter.transform(raw_next_obs)
            stage_started = time.perf_counter()
            transition = trainer.record_transition(
                obs=obs,
                action=action,
                next_obs=next_obs,
                terminated=terminated,
                truncated=bool(truncated),
                is_success=bool(is_success),
                reward=reward,
                grasp_penalty=grasp_penalty,
                is_intervention=cached_is_intervention,
                info=info,
            )
            snapshot = agent.online_buffer.snapshot_transition(transition)
            buffer_writer.request_transition(
                online_transition=snapshot,
                demo_transition=snapshot if cached_is_intervention else None,
            )
            stage_seconds["replay"] += time.perf_counter() - stage_started
            episode_return += reward
            episode_length += 1
            if cached_is_intervention:
                episode_intervention_steps += 1
                total_intervention_steps += 1
            _drain_learner_metrics(
                trainer,
                tensorboard,
                events,
                log_interval=int(cfg.logging.log_interval_learner_steps),
            )

            if bool(terminated or truncated or is_success):
                success_count += int(is_success)
                episode_metrics = {
                    "return": episode_return,
                    "length": episode_length,
                    "success": int(is_success),
                    "intervention_steps": episode_intervention_steps,
                    "intervention_segments": episode_intervention_segments,
                    "intervention_step_ratio": episode_intervention_steps / max(1, episode_length),
                    "online_buffer_size": len(agent.online_buffer),
                    "demo_buffer_size": len(agent.demo_buffer),
                }
                tensorboard.log(episode_metrics, step=episode_index, prefix="episode")
                events.log({"event": "episode_end", "episode": episode_index, **episode_metrics})
                reason = "success" if is_success else ("terminated" if terminated else "truncated")
                print(
                    f"[episode] index={episode_index} reason={reason} "
                    f"return={episode_return:.2f} length={episode_length} "
                    f"intervention_rate={episode_metrics['intervention_step_ratio']:.3f} "
                    f"updates={trainer.total_updates}"
                )
                obs, _ = reset_observation_adapter(adapter, preserve_mjviewer=interactive)
                episode_index += 1
                save_episode_checkpoint_if_due()
                episode_return = 0.0
                episode_length = 0
                episode_intervention_steps = 0
                episode_intervention_segments = 0
                cached_override_action = None
                cached_is_intervention = False
                policy_gate.force_ready()
                intervention_gate.force_ready()
                if intervention is not None:
                    intervention.start_episode()
                if float(cfg.runtime.episode_pause_sec) > 0:
                    time.sleep(float(cfg.runtime.episode_pause_sec))
            else:
                obs = next_obs

            progress = trainer.progress_snapshot()
            elapsed = time.monotonic() - last_runtime_log
            if elapsed >= float(cfg.logging.runtime_log_interval_seconds):
                learner_updates = progress["total_updates"] - last_runtime_update
                runtime_metrics = {
                    **progress,
                    "env_fps": fps.snapshot(elapsed),
                    "learner_updates_per_second": learner_updates / max(elapsed, 1e-6),
                    "online_buffer_size": len(agent.online_buffer),
                    "demo_buffer_size": len(agent.demo_buffer),
                    "total_intervention_steps": total_intervention_steps,
                    "total_intervention_segments": total_intervention_segments,
                    "intervention_step_ratio": total_intervention_steps / max(1, trainer.total_env_steps),
                    **{f"timing/{name}_seconds": value for name, value in stage_seconds.items()},
                }
                tensorboard.log(runtime_metrics, step=trainer.total_env_steps, prefix="runtime")
                events.log({"event": "runtime", **runtime_metrics})
                print(
                    f"[status] episode={episode_index} env_steps={progress['env_steps']} "
                    f"updates={progress['total_updates']} "
                    f"env_fps={runtime_metrics['env_fps']:.1f} "
                    f"updates_per_second={runtime_metrics['learner_updates_per_second']:.1f} "
                    f"intervention_rate={runtime_metrics['intervention_step_ratio']:.3f} "
                    f"online_buffer={runtime_metrics['online_buffer_size']} "
                    f"demo_buffer={runtime_metrics['demo_buffer_size']}"
                )
                last_runtime_update = progress["total_updates"]
                stage_seconds = {name: 0.0 for name in stage_seconds}
                last_runtime_log = time.monotonic()

        while not trainer.learner_finished:
            trainer.raise_if_failed()
            _drain_learner_metrics(
                trainer,
                tensorboard,
                events,
                log_interval=int(cfg.logging.log_interval_learner_steps),
            )
            time.sleep(0.05)
    finally:
        active_exception = sys.exc_info()[0] is not None
        cleanup_errors: list[tuple[str, BaseException]] = []

        def attempt(label: str, callback) -> None:
            try:
                callback()
            except BaseException as exc:
                cleanup_errors.append((label, exc))
                print(f"[cleanup-error] {label}: {exc}")

        attempt("learner stop", trainer.close_async_worker)
        attempt(
            "learner metric drain",
            lambda: _drain_learner_metrics(
                trainer,
                tensorboard,
                events,
                log_interval=int(cfg.logging.log_interval_learner_steps),
            ),
        )
        attempt("buffer flush", lambda: buffer_writer.flush(timeout=120.0))
        final_checkpoint = run_dir / str(cfg.checkpoint.directory) / "latest.pt"
        attempt(
            "final checkpoint",
            lambda: agent.save_checkpoint(
                final_checkpoint,
                include_buffers=True,
                extra={
                    "trainer_state": trainer.state_dict(),
                    "episode_index": int(episode_index),
                    "success_count": int(success_count),
                    "run_name": run_name,
                },
            ),
        )
        finished_at = datetime.datetime.now().isoformat(timespec="seconds")
        attempt(
            "metadata",
            lambda: _write_metadata(
                metadata_path,
                {
                    "run_name": run_name,
                    "task": str(cfg.task.name),
                    "run_dir": str(run_dir),
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "trainer_state": trainer.state_dict(),
                    "episode_index": episode_index,
                    "success_count": success_count,
                    "checkpoint": str(final_checkpoint),
                    "cleanup_errors": [f"{label}: {error}" for label, error in cleanup_errors],
                },
            ),
        )
        attempt("run end event", lambda: events.log({"event": "run_end", **trainer.state_dict()}))
        attempt("TensorBoard flush", tensorboard.flush)
        if cleanup_errors and not active_exception:
            details = "; ".join(f"{label}: {error}" for label, error in cleanup_errors)
            raise RuntimeError(f"HIL-SERL cleanup failed: {details}")


@hydra.main(version_base="1.2", config_path="config", config_name="overall")
def main(cfg: DictConfig) -> None:
    with ExitStack() as resources:
        _run(cfg, resources)


if __name__ == "__main__":
    main()
