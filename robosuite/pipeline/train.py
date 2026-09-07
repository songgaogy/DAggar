"""WidowX-style DSRL-SAC training entry point for robosuite."""

from __future__ import annotations

import os
import socket
import time
import uuid
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from robosuite.pipeline.build_cache import ensure_warmup_cache
from robosuite.pipeline.orchestration import (
    CollectionSummary,
    PolicyDecision,
    distribute_episode_quota,
    run_episode_collection,
)
from robosuite.pipeline.src.data import (
    CompactTransition,
    CudaBatchPrefetcher,
    OnlineReplay,
    UniformReplay,
    WarmupReplay,
)
from robosuite.pipeline.src.dsrl import (
    DSRLAgent,
    DSRLBatch,
    DSRLConfig,
    DSRLInferencePolicy,
    DSRLTrainer,
    NetworkConfig,
)
from robosuite.pipeline.src.environment import (
    FlowContext,
    FlowObservation,
    FlowPolicyAdapter,
    RobosuiteVectorRuntime,
    observation_batch_to_cuda,
)
from robosuite.pipeline.utils import (
    ConsoleLogCapture,
    JsonlEventLogger,
    TensorBoardLogger,
    capture_rng_state,
    create_run_directory,
    file_identity,
    require_cuda,
    restore_rng_state,
    set_seed,
    write_json,
)


@dataclass(frozen=True)
class EncodedState:
    visual_features: torch.Tensor
    proprio: torch.Tensor
    flow_context: FlowContext


def _canonical_observation(
    observation: Mapping[str, np.ndarray], camera_names: Sequence[str]
) -> dict[str, np.ndarray]:
    canonical = dict(observation)
    for camera_name in camera_names:
        output_key = f"{camera_name}_image"
        if output_key not in canonical and camera_name in canonical:
            canonical[output_key] = canonical[camera_name]
    return canonical


def _select_context(context: FlowContext, index: int) -> FlowContext:
    selection = slice(index, index + 1)
    return FlowContext(
        context.task_scene_cond[selection],
        context.context_tokens[selection],
        context.context_padding_mask[selection],
    )


def _stack_states(states: Sequence[EncodedState]) -> EncodedState:
    if not states:
        raise ValueError("Cannot stack an empty state sequence.")
    return EncodedState(
        visual_features=torch.cat([state.visual_features for state in states]),
        proprio=torch.cat([state.proprio for state in states]),
        flow_context=FlowContext.stack([state.flow_context for state in states]),
    )


def _state_to_numpy(state: EncodedState, feature_dtype: np.dtype) -> dict[str, np.ndarray]:
    return {
        "visual_features": state.visual_features[0].detach().cpu().numpy().astype(
            feature_dtype, copy=False
        ),
        "proprio": state.proprio[0].detach().float().cpu().numpy().astype(np.float32, copy=False),
    }


def _batch_to_dsrl(values: Mapping[str, torch.Tensor]) -> DSRLBatch:
    def floating(name: str) -> torch.Tensor:
        return values[name].to(dtype=torch.float32)

    return DSRLBatch(
        visual_features=floating("visual_features"),
        proprio=floating("proprio"),
        latents=floating("actions"),
        rewards=floating("rewards"),
        dones=floating("dones"),
        next_visual_features=floating("next_visual_features"),
        next_proprio=floating("next_proprio"),
    )


def _network_config(cfg: DictConfig, flow: FlowPolicyAdapter) -> NetworkConfig:
    visual_dim = (
        len(flow.camera_names) * flow.image_tokens_per_camera * flow.image_token_dim
    )
    return NetworkConfig(
        visual_dim=visual_dim,
        proprio_dim=flow.proprio_dim,
        action_horizon=int(cfg.flow.action_horizon),
        action_dim=int(cfg.flow.action_dim),
        hidden_dims=tuple(int(value) for value in cfg.algorithm.hidden_dims),
        latent_limit=float(cfg.algorithm.latent_action_bound),
        log_std_min=float(cfg.algorithm.log_std_min),
        log_std_max=float(cfg.algorithm.log_std_max),
    )


def _agent_config(cfg: DictConfig, network: NetworkConfig) -> DSRLConfig:
    return DSRLConfig(
        network=network,
        learner_device=str(cfg.runtime.learner_device),
        learning_rate=float(cfg.algorithm.learning_rate),
        gamma=float(cfg.algorithm.gamma),
        tau=float(cfg.algorithm.tau),
        batch_size=int(cfg.algorithm.batch_size),
        utd_steps=int(cfg.algorithm.utd),
        target_entropy=float(cfg.algorithm.target_entropy),
        grad_clip_norm=(
            None
            if cfg.algorithm.grad_clip_norm is None
            else float(cfg.algorithm.grad_clip_norm)
        ),
    )


def _atomic_torch_save(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    torch.save(dict(payload), temporary)
    temporary.replace(destination)


def _gpu_metrics(device: torch.device, prefix: str) -> dict[str, float]:
    metrics = {
        f"{prefix}_memory_allocated_bytes": float(torch.cuda.memory_allocated(device)),
        f"{prefix}_memory_reserved_bytes": float(torch.cuda.memory_reserved(device)),
    }
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(int(device.index))
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        metrics[f"{prefix}_utilization_percent"] = float(utilization.gpu)
        metrics[f"{prefix}_memory_used_bytes"] = float(memory.used)
    except (ImportError, RuntimeError):
        pass
    return metrics


def _resume_signature(config: Mapping[str, Any]) -> dict[str, Any]:
    signature = {
        key: config.get(key)
        for key in ("seed", "task", "environment", "inputs", "vision", "flow", "algorithm", "warmup")
    }
    runtime = dict(config.get("runtime", {}))
    runtime.pop("resume", None)
    runtime.pop("checkpoint", None)
    signature["runtime"] = runtime
    storage = dict(config.get("storage", {}))
    signature["checkpoint_interval_episodes"] = storage.get(
        "checkpoint_interval_episodes"
    )
    return signature


def _save_training_state(
    *,
    trainer: DSRLTrainer,
    run_dir: Path,
    cfg: DictConfig,
    progress: CollectionSummary,
    online: OnlineReplay,
    replay: UniformReplay,
    warmup: WarmupReplay,
    flow_checkpoint: Path,
    environment_rng: Mapping[str, Any] | None = None,
) -> Path:
    replay_path = (
        run_dir
        / str(cfg.storage.online_replay_directory)
        / f"episode_{progress.completed_episodes:08d}"
    )
    replay_manifest = online.snapshot(replay_path)
    payload = {
        "format": "dsrl_sac_widowx_v1",
        "trainer": trainer.state_dict(),
        "counters": {
            "completed_episodes": progress.completed_episodes,
            "completed_by_worker": list(progress.completed_by_worker),
            "primitive_steps": progress.primitive_steps,
            "macro_steps": progress.macro_steps,
            "online_vector_steps": progress.online_vector_steps,
        },
        "rng": capture_rng_state(),
        "environment_rng": environment_rng,
        "replay_rng": replay.state_dict(),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "fingerprints": {
            "warmup": warmup.manifest["fingerprint"],
            "flow": file_identity(flow_checkpoint),
        },
        "warmup_cache": {"path": os.fspath(warmup.path)},
        "online_replay": {
            "path": os.fspath(replay_path),
            "manifest": replay_manifest,
        },
    }
    checkpoint_dir = run_dir / str(cfg.storage.checkpoint_directory)
    episode_path = checkpoint_dir / f"episode_{progress.completed_episodes:08d}.pt"
    _atomic_torch_save(payload, episode_path)
    _atomic_torch_save(payload, checkpoint_dir / "latest.pt")
    return episode_path


def _run(cfg: DictConfig) -> None:
    require_cuda()
    learner_device = torch.device(str(cfg.runtime.learner_device))
    inference_device = torch.device(str(cfg.runtime.inference_device))
    if learner_device.type != "cuda" or inference_device.type != "cuda":
        raise ValueError("Learner and inference devices must both be CUDA devices.")
    total_episodes = int(cfg.runtime.total_episodes)
    warmup_episodes = int(cfg.warmup.episodes)
    num_envs = int(cfg.runtime.num_envs)
    checkpoint_interval = int(cfg.storage.checkpoint_interval_episodes)
    if not 0 < warmup_episodes < total_episodes:
        raise ValueError("warmup.episodes must be between zero and runtime.total_episodes.")
    if warmup_episodes % checkpoint_interval != 0 or total_episodes % checkpoint_interval != 0:
        raise ValueError("Warmup and total episode counts must align with the checkpoint interval.")

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    set_seed(int(cfg.seed))

    warmup, warmup_fingerprint, flow_metadata, flow = ensure_warmup_cache(cfg)
    set_seed(int(cfg.seed))
    expected_warmup_counts = distribute_episode_quota(warmup_episodes, num_envs)
    if tuple(warmup.manifest["completed_by_worker"]) != expected_warmup_counts:
        raise ValueError("Warmup cache worker episode distribution does not match this run.")

    task_name = str(cfg.task.name)
    flow_checkpoint = Path(to_absolute_path(str(cfg.inputs.base_policy_checkpoint)))
    _, run_dir = create_run_directory(
        to_absolute_path(str(cfg.storage.output_root)),
        task_name,
        None if cfg.storage.run_name is None else str(cfg.storage.run_name),
    )
    OmegaConf.save(
        OmegaConf.create(OmegaConf.to_container(cfg, resolve=True)),
        run_dir / str(cfg.storage.resolved_config_filename),
    )

    with ExitStack() as resources:
        console = ConsoleLogCapture(run_dir / str(cfg.storage.console_filename))
        events = JsonlEventLogger(run_dir / str(cfg.storage.metrics_filename))
        tensorboard = TensorBoardLogger(
            run_dir / str(cfg.tensorboard.directory),
            enabled=bool(cfg.tensorboard.enabled),
            flush_secs=int(cfg.tensorboard.flush_seconds),
        )
        console.start()
        events.start()
        resources.callback(console.stop)
        resources.callback(events.close)
        resources.callback(tensorboard.close)

        metadata = {
            "algorithm": "DSRL-SAC",
            "setting": "WidowX Pick-and-Place",
            "task": task_name,
            "seed": int(cfg.seed),
            "warmup_seed": int(cfg.warmup.seed),
            "hostname": socket.gethostname(),
            "learner_device": str(learner_device),
            "inference_device": str(inference_device),
            "flow_checkpoint": file_identity(flow_checkpoint),
            "warmup_fingerprint": warmup_fingerprint,
            "started_at": time.time(),
        }
        write_json(run_dir / str(cfg.storage.metadata_filename), metadata)

        cameras = tuple(flow_metadata["camera_names"])
        prompt = flow_metadata["prompts"][int(cfg.flow.prompt_index)]
        network = _network_config(cfg, flow)
        agent = DSRLAgent(_agent_config(cfg, network))
        inference_policy = DSRLInferencePolicy(network, inference_device)

        resume_path = (
            None
            if cfg.runtime.checkpoint is None
            else Path(to_absolute_path(str(cfg.runtime.checkpoint)))
        )
        resume_payload = None
        if bool(cfg.runtime.resume):
            if resume_path is None or not resume_path.is_file():
                raise FileNotFoundError("runtime.checkpoint is required when runtime.resume=true.")
            resume_payload = torch.load(resume_path, map_location="cpu", weights_only=False)
            if resume_payload.get("format") != "dsrl_sac_widowx_v1":
                raise ValueError("Checkpoint is not compatible with WidowX DSRL-SAC.")
            if resume_payload["fingerprints"]["warmup"] != warmup_fingerprint:
                raise ValueError("Resume checkpoint warmup fingerprint does not match.")
            current_config = OmegaConf.to_container(cfg, resolve=True)
            if _resume_signature(resume_payload["config"]) != _resume_signature(current_config):
                raise ValueError("Resume checkpoint experiment configuration does not match.")
            if resume_payload["fingerprints"]["flow"] != file_identity(flow_checkpoint):
                raise ValueError("Resume checkpoint flow fingerprint does not match.")

        maximum_online_macros = (total_episodes - warmup_episodes) * (
            (int(cfg.environment.horizon) + int(cfg.flow.action_horizon) - 1)
            // int(cfg.flow.action_horizon)
        )
        online = (
            OnlineReplay.load_snapshot(resume_payload["online_replay"]["path"])
            if resume_payload is not None
            else OnlineReplay(maximum_online_macros)
        )
        replay = UniformReplay(warmup, online, seed=int(cfg.seed))

        def provide_batch(_batch_size: int) -> DSRLBatch:
            return _batch_to_dsrl(prefetcher.next())

        trainer = DSRLTrainer(agent, provide_batch)
        if resume_payload is not None:
            trainer.load_state_dict(resume_payload["trainer"], strict=True)
            replay.load_state_dict(resume_payload["replay_rng"])
            counters: Mapping[str, Any] = resume_payload["counters"]
            restore_rng_state(resume_payload["rng"])
        else:
            counters = {
                "completed_episodes": warmup_episodes,
                "completed_by_worker": list(expected_warmup_counts),
                "primitive_steps": int(warmup.manifest["primitive_steps"]),
                "macro_steps": int(warmup.manifest["macro_steps"]),
                "online_vector_steps": 0,
            }
        inference_policy.load_inference_state(agent.inference_state_dict())

        initial_progress = CollectionSummary(
            completed_episodes=int(counters["completed_episodes"]),
            completed_by_worker=tuple(int(v) for v in counters["completed_by_worker"]),
            primitive_steps=int(counters["primitive_steps"]),
            macro_steps=int(counters["macro_steps"]),
            online_vector_steps=int(counters.get("online_vector_steps", 0)),
        )
        checkpoint: Path | None = None
        if resume_payload is None:
            checkpoint = _save_training_state(
                trainer=trainer,
                run_dir=run_dir,
                cfg=cfg,
                progress=initial_progress,
                online=online,
                replay=replay,
                warmup=warmup,
                flow_checkpoint=flow_checkpoint,
            )
            events.log({"type": "checkpoint", "episode": warmup_episodes, "path": os.fspath(checkpoint)})
        resume_environment_rng = (
            None if resume_payload is None else resume_payload.get("environment_rng")
        )
        if (
            resume_payload is not None
            and initial_progress.completed_episodes > warmup_episodes
            and resume_environment_rng is None
        ):
            raise ValueError("Resume checkpoint is missing vector environment RNG state.")
        del resume_payload

        prefetcher = CudaBatchPrefetcher(
            replay,
            batch_size=int(cfg.algorithm.batch_size),
            device=learner_device,
            depth=int(cfg.runtime.prefetch_batches),
        )
        resources.callback(lambda: prefetcher.close())
        runtime = RobosuiteVectorRuntime(
            env_metadata=dict(flow_metadata["task_metadata"]),
            camera_names=cameras,
            image_height=int(cfg.environment.image_height),
            image_width=int(cfg.environment.image_width),
            control_frequency=int(cfg.environment.control_frequency),
            horizon=int(cfg.environment.horizon),
            num_envs=num_envs,
            seed=int(cfg.seed),
        )
        resources.callback(runtime.close)
        if resume_environment_rng is not None:
            runtime.load_state_dict(dict(resume_environment_rng))
        feature_dtype = np.float16 if str(cfg.vision.cache_dtype) == "float16" else np.float32

        def encode_observations(
            observations: Mapping[int, Mapping[str, np.ndarray]],
        ) -> dict[int, EncodedState]:
            worker_ids = list(observations)
            canonical = [
                _canonical_observation(observations[worker], cameras) for worker in worker_ids
            ]
            images, proprio = observation_batch_to_cuda(canonical, cameras, "state", inference_device)
            normalized_proprio = flow.normalize_proprio(proprio)
            context, image_tokens = flow.encode_context_with_image_tokens(
                FlowObservation(images, proprio, prompt)
            )
            return {
                worker: EncodedState(
                    image_tokens[row : row + 1],
                    normalized_proprio[row : row + 1],
                    _select_context(context, row),
                )
                for row, worker in enumerate(worker_ids)
            }

        def act(worker_ids: Sequence[int], states: Mapping[int, EncodedState]) -> PolicyDecision:
            batched = _stack_states([states[worker] for worker in worker_ids])
            latents = inference_policy.latent(
                batched.visual_features, batched.proprio, deterministic=False
            )
            actions = flow.decode_noise(batched.flow_context, latents)
            return PolicyDecision(actions.cpu().numpy(), latents.cpu().numpy())

        def add_transition(
            _worker: int,
            state: EncodedState,
            next_state: EncodedState,
            latent: np.ndarray,
            result: Any,
        ) -> None:
            current = _state_to_numpy(state, feature_dtype)
            following = _state_to_numpy(next_state, feature_dtype)
            online.add(
                CompactTransition(
                    **current,
                    next_visual_features=following["visual_features"],
                    next_proprio=following["proprio"],
                    actions=np.asarray(latent, dtype=np.float32),
                    reward=float(result.reward),
                    done=bool(result.done),
                    executed_length=int(result.executed_length),
                )
            )

        def on_episode(values: dict[str, Any]) -> None:
            events.log({"type": "episode", **values})
            tensorboard.log(
                {key: value for key, value in values.items() if key != "termination_reason"},
                step=int(values["episode"]),
                prefix="episode",
            )
            print(
                f"[episode] index={values['episode']} worker={values['worker']} "
                f"success={int(values['success'])} return={values['return']:.1f} "
                f"primitive={values['primitive_length']} macro={values['macro_length']} "
                f"reason={values['termination_reason']} fps={values['fps']:.1f}"
            )

        def on_update(values: Mapping[str, float], vector_step: int) -> None:
            if not all(np.isfinite(float(value)) for value in values.values()):
                raise FloatingPointError("Learner produced non-finite metrics.")
            events.log({"type": "learner", "online_vector_step": vector_step, **values})
            tensorboard.log(values, step=vector_step, prefix="learner")
            system = {
                "warmup_replay": len(warmup),
                "online_replay": len(online),
                **_gpu_metrics(learner_device, "learner"),
                **_gpu_metrics(inference_device, "inference"),
            }
            events.log({"type": "system", "online_vector_step": vector_step, **system})
            tensorboard.log(system, step=vector_step, prefix="system")

        def on_checkpoint(progress: CollectionSummary) -> None:
            nonlocal checkpoint, prefetcher
            prefetcher.close()
            tensorboard.flush()
            checkpoint = _save_training_state(
                trainer=trainer,
                run_dir=run_dir,
                cfg=cfg,
                progress=progress,
                online=online,
                replay=replay,
                warmup=warmup,
                flow_checkpoint=flow_checkpoint,
                environment_rng=runtime.state_dict(),
            )
            events.log({"type": "checkpoint", "episode": progress.completed_episodes, "path": os.fspath(checkpoint)})
            if progress.completed_episodes < total_episodes:
                prefetcher = CudaBatchPrefetcher(
                    replay,
                    batch_size=int(cfg.algorithm.batch_size),
                    device=learner_device,
                    depth=int(cfg.runtime.prefetch_batches),
                )

        summary = run_episode_collection(
            runtime=runtime,
            target_by_worker=distribute_episode_quota(total_episodes, num_envs),
            encode=encode_observations,
            act=act,
            add_transition=add_transition,
            update=trainer.update_cycle,
            sync_inference=lambda: inference_policy.load_inference_state(
                agent.inference_state_dict()
            ),
            on_episode=on_episode,
            on_update=on_update,
            update_interval_vector_steps=int(cfg.algorithm.update_interval_vector_steps),
            on_checkpoint=on_checkpoint,
            checkpoint_interval_episodes=checkpoint_interval,
            initial_completed_by_worker=initial_progress.completed_by_worker,
            initial_primitive_steps=initial_progress.primitive_steps,
            initial_macro_steps=initial_progress.macro_steps,
            initial_online_vector_steps=initial_progress.online_vector_steps,
        )
        if summary.completed_episodes != total_episodes:
            raise RuntimeError(f"Episode quota mismatch: {summary}.")
        if checkpoint is None or checkpoint.name != f"episode_{total_episodes:08d}.pt":
            on_checkpoint(summary)
        assert checkpoint is not None
        events.log({"type": "completed", "checkpoint": os.fspath(checkpoint), **summary.__dict__})
        tensorboard.flush()
        metadata.update({"finished_at": time.time(), "summary": summary.__dict__, "checkpoint": os.fspath(checkpoint)})
        write_json(run_dir / str(cfg.storage.metadata_filename), metadata)
        print(f"Training completed: episodes={summary.completed_episodes} checkpoint={checkpoint}")


run_online_collection = run_episode_collection


@hydra.main(version_base=None, config_path="config", config_name="overall")
def main(cfg: DictConfig) -> None:
    _run(cfg)


if __name__ == "__main__":
    main()
