"""Online DSRL training entry point for robosuite."""

from __future__ import annotations

import copy
import os
import socket
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from robosuite.pipeline.src.data import (
    CacheDataset,
    CompactTransition,
    CudaBatchPrefetcher,
    OnlineReplay,
    UniformReplay,
    build_cache_fingerprint,
    build_feature_cache,
    load_legacy_buffer,
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
from robosuite.pipeline.src.vision import DinoV2Encoder
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
    dino_features: torch.Tensor
    proprio: torch.Tensor
    flow_context: FlowContext


@dataclass(frozen=True)
class PolicyDecision:
    environment_actions: np.ndarray
    normalized_actions: np.ndarray


@dataclass(frozen=True)
class CollectionSummary:
    completed_episodes: int
    completed_by_worker: tuple[int, ...]
    primitive_steps: int
    macro_steps: int


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
        dino_features=torch.cat([state.dino_features for state in states], dim=0),
        proprio=torch.cat([state.proprio for state in states], dim=0),
        flow_context=FlowContext.stack([state.flow_context for state in states]),
    )


def _state_to_numpy(state: EncodedState) -> dict[str, np.ndarray]:
    return {
        "dino_features": state.dino_features.detach().to(torch.float16).cpu().numpy()[0],
        "proprio": state.proprio.detach().to(torch.float16).cpu().numpy()[0],
        "task_scene_cond": state.flow_context.task_scene_cond.detach().to(torch.float16).cpu().numpy()[0],
        "context_tokens": state.flow_context.context_tokens.detach().to(torch.float16).cpu().numpy()[0],
        "context_padding_mask": state.flow_context.context_padding_mask.detach().to(torch.bool).cpu().numpy()[0],
    }


def _canonical_observation(
    observation: Mapping[str, np.ndarray], camera_names: Sequence[str]
) -> dict[str, np.ndarray]:
    """Accept both legacy bare camera keys and runtime ``*_image`` keys."""

    canonical = dict(observation)
    for camera_name in camera_names:
        output_key = f"{camera_name}_image"
        if output_key not in canonical and camera_name in canonical:
            canonical[output_key] = canonical[camera_name]
    return canonical


def _batch_to_dsrl(values: Mapping[str, torch.Tensor]) -> DSRLBatch:
    def floating(name: str) -> torch.Tensor:
        return values[name].to(dtype=torch.float32)

    return DSRLBatch(
        dino_features=floating("dino_features"),
        proprio=floating("proprio"),
        flow_context=FlowContext.from_mapping(
            {
                "task_scene_cond": floating("task_scene_cond"),
                "context_tokens": floating("context_tokens"),
                "context_padding_mask": values["context_padding_mask"],
            }
        ),
        actions=floating("actions"),
        rewards=floating("rewards"),
        dones=floating("dones"),
        next_dino_features=floating("next_dino_features"),
        next_proprio=floating("next_proprio"),
        next_flow_context=FlowContext.from_mapping(
            {
                "task_scene_cond": floating("next_task_scene_cond"),
                "context_tokens": floating("next_context_tokens"),
                "context_padding_mask": values["next_context_padding_mask"],
            }
        ),
    )


def run_online_collection(
    *,
    runtime: Any,
    num_envs: int,
    episodes_per_env: int,
    encode: Callable[[Mapping[int, Mapping[str, np.ndarray]]], dict[int, Any]],
    act: Callable[[Sequence[int], Mapping[int, Any]], PolicyDecision],
    add_transition: Callable[[int, Any, Any, np.ndarray, Any], None],
    update: Callable[[], Mapping[str, float]],
    sync_inference: Callable[[], None],
    on_episode: Callable[[dict[str, Any]], None],
    on_update: Callable[[Mapping[str, float], int], None],
    on_checkpoint: Callable[[CollectionSummary], None] | None = None,
    checkpoint_interval_episodes: int | None = None,
    initial_completed_by_worker: Sequence[int] | None = None,
    initial_primitive_steps: int = 0,
    initial_macro_steps: int = 0,
) -> CollectionSummary:
    """Collect exact per-worker quotas with one learner burst per parallel macro-step."""

    if num_envs <= 0 or episodes_per_env <= 0:
        raise ValueError("Environment and episode counts must be positive.")
    completed = (
        [0] * num_envs
        if initial_completed_by_worker is None
        else [int(value) for value in initial_completed_by_worker]
    )
    if len(completed) != num_envs or any(not 0 <= value <= episodes_per_env for value in completed):
        raise ValueError("Initial per-worker episode counters are invalid.")
    returns = [0.0] * num_envs
    macro_lengths = [0] * num_envs
    episode_started = [time.monotonic()] * num_envs
    active_at_start = [index for index, count in enumerate(completed) if count < episodes_per_env]
    observations = runtime.reset(active_at_start) if active_at_start else {}
    states = encode(observations) if observations else {}
    primitive_total = int(initial_primitive_steps)
    macro_total = int(initial_macro_steps)
    completed_total = sum(completed)
    interval = None if checkpoint_interval_episodes is None else int(checkpoint_interval_episodes)
    next_checkpoint_episode = (
        None
        if interval is None or interval <= 0
        else (completed_total // interval + 1) * interval
    )

    while completed_total < num_envs * episodes_per_env:
        worker_ids = [index for index, count in enumerate(completed) if count < episodes_per_env]
        decision = act(worker_ids, states)
        results = runtime.step(worker_ids, decision.environment_actions)
        next_observations = {worker_id: results[worker_id].observation for worker_id in worker_ids}
        next_states = encode(next_observations)
        for row, worker_id in enumerate(worker_ids):
            result = results[worker_id]
            add_transition(
                worker_id,
                states[worker_id],
                next_states[worker_id],
                decision.normalized_actions[row],
                result,
            )
            returns[worker_id] += float(result.reward)
            macro_lengths[worker_id] += 1
            primitive_total += int(result.executed_length)
            macro_total += 1

        learner_metrics = update()
        sync_inference()
        on_update(learner_metrics, macro_total)

        reset_ids: list[int] = []
        for worker_id in worker_ids:
            result = results[worker_id]
            if not result.done:
                states[worker_id] = next_states[worker_id]
                continue
            completed[worker_id] += 1
            completed_total += 1
            elapsed = max(time.monotonic() - episode_started[worker_id], 1e-9)
            on_episode(
                {
                    "episode": completed_total,
                    "worker": worker_id,
                    "worker_episode": completed[worker_id],
                    "success": bool(result.success),
                    "return": returns[worker_id],
                    "primitive_length": int(result.primitive_steps),
                    "macro_length": macro_lengths[worker_id],
                    "termination_reason": str(result.reason),
                    "fps": float(result.primitive_steps) / elapsed,
                }
            )
            returns[worker_id] = 0.0
            macro_lengths[worker_id] = 0
            if completed[worker_id] < episodes_per_env:
                reset_ids.append(worker_id)
                episode_started[worker_id] = time.monotonic()
            else:
                states.pop(worker_id, None)
        if reset_ids:
            states.update(encode(runtime.reset(reset_ids)))
        if (
            on_checkpoint is not None
            and next_checkpoint_episode is not None
            and completed_total >= next_checkpoint_episode
        ):
            on_checkpoint(
                CollectionSummary(
                    completed_episodes=completed_total,
                    completed_by_worker=tuple(completed),
                    primitive_steps=primitive_total,
                    macro_steps=macro_total,
                )
            )
            while next_checkpoint_episode <= completed_total:
                next_checkpoint_episode += interval

    return CollectionSummary(
        completed_episodes=completed_total,
        completed_by_worker=tuple(completed),
        primitive_steps=primitive_total,
        macro_steps=macro_total,
    )


def _load_policy_metadata(checkpoint_path: Path, task_name: str, prompt_index: int) -> tuple[dict[str, Any], str]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata_map = payload.get("task_metadata_map")
    if not isinstance(metadata_map, Mapping) or task_name not in metadata_map:
        raise KeyError(f"Flow checkpoint has no environment metadata for '{task_name}'.")
    prompt_map = payload.get("task_prompt_map", {})
    prompts = prompt_map.get(task_name, task_name) if isinstance(prompt_map, Mapping) else task_name
    prompts = list(prompts) if isinstance(prompts, (list, tuple)) else [prompts]
    if not 0 <= prompt_index < len(prompts):
        raise IndexError(f"Prompt index {prompt_index} is invalid for task '{task_name}'.")
    return dict(metadata_map[task_name]), str(prompts[prompt_index])


def _network_config(cfg: DictConfig, proprio_dim: int) -> NetworkConfig:
    return NetworkConfig(
        visual_dim=int(cfg.vision.embedding_dim) * len(cfg.environment.camera_names),
        proprio_dim=int(proprio_dim),
        state_dim=int(cfg.vision.bottleneck_dim),
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
        qw_steps=int(cfg.algorithm.noise_critic_steps),
        target_entropy=float(cfg.algorithm.target_entropy),
        grad_clip_norm=None if cfg.algorithm.grad_clip_norm is None else float(cfg.algorithm.grad_clip_norm),
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


def _save_training_state(
    *,
    trainer: DSRLTrainer,
    run_dir: Path,
    cfg: DictConfig,
    completed_episodes: int,
    completed_by_worker: Sequence[int],
    primitive_steps: int,
    macro_steps: int,
    online: OnlineReplay,
    replay: UniformReplay,
    cache: CacheDataset,
    flow_checkpoint: Path,
    dino_checkpoint: Path,
    normalizers: Mapping[str, Any],
) -> Path:
    replay_path = run_dir / str(cfg.storage.online_replay_directory) / f"episode_{completed_episodes:08d}"
    replay_manifest = online.snapshot(replay_path)
    payload = {
        "trainer": trainer.state_dict(),
        "counters": {
            "completed_episodes": int(completed_episodes),
            "completed_by_worker": list(map(int, completed_by_worker)),
            "primitive_steps": int(primitive_steps),
            "macro_steps": int(macro_steps),
        },
        "rng": capture_rng_state(),
        "replay_rng": replay.state_dict(),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "normalizers": dict(normalizers),
        "fingerprints": {
            "cache": cache.manifest["fingerprint"],
            "flow": file_identity(flow_checkpoint),
            "dinov2": file_identity(dino_checkpoint),
        },
        "online_replay": {
            "path": os.fspath(replay_path),
            "manifest": replay_manifest,
        },
    }
    checkpoint_dir = run_dir / str(cfg.storage.checkpoint_directory)
    episode_path = checkpoint_dir / f"episode_{completed_episodes:08d}.pt"
    _atomic_torch_save(payload, episode_path)
    _atomic_torch_save(payload, checkpoint_dir / "latest.pt")
    return episode_path


def _run(cfg: DictConfig) -> None:
    require_cuda()
    learner_device = torch.device(str(cfg.runtime.learner_device))
    inference_device = torch.device(str(cfg.runtime.inference_device))
    if learner_device.type != "cuda" or inference_device.type != "cuda":
        raise ValueError("Learner and inference devices must both be CUDA devices.")
    if learner_device == inference_device:
        raise ValueError("The approved run requires separate learner and inference CUDA devices.")
    set_seed(int(cfg.seed))
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    task_name = str(cfg.task.name)
    flow_checkpoint = Path(to_absolute_path(str(cfg.inputs.base_policy_checkpoint)))
    dino_checkpoint = Path(to_absolute_path(str(cfg.inputs.dinov2_checkpoint)))
    offline_path = Path(to_absolute_path(str(cfg.inputs.offline_transitions)))
    metadata_path = Path(to_absolute_path(str(cfg.inputs.offline_metadata)))
    env_metadata, prompt = _load_policy_metadata(
        flow_checkpoint, task_name, int(cfg.flow.prompt_index)
    )
    _, run_dir = create_run_directory(
        to_absolute_path(str(cfg.storage.output_root)),
        task_name,
        None if cfg.storage.run_name is None else str(cfg.storage.run_name),
    )
    resolved_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.save(resolved_cfg, run_dir / str(cfg.storage.resolved_config_filename))

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
            "task": task_name,
            "seed": int(cfg.seed),
            "hostname": socket.gethostname(),
            "learner_device": str(learner_device),
            "inference_device": str(inference_device),
            "flow_checkpoint": file_identity(flow_checkpoint),
            "dinov2_checkpoint": file_identity(dino_checkpoint),
            "started_at": time.time(),
        }
        write_json(run_dir / str(cfg.storage.metadata_filename), metadata)

        inference_flow = FlowPolicyAdapter(
            flow_checkpoint,
            inference_device,
            expected_camera_names=list(cfg.environment.camera_names),
            expected_action_horizon=int(cfg.flow.action_horizon),
            expected_action_dim=int(cfg.flow.action_dim),
            ode_steps=int(cfg.flow.ode_steps),
            image_size=int(cfg.environment.image_height),
        )
        dino = DinoV2Encoder(dino_checkpoint, inference_device)
        learner_flow = FlowPolicyAdapter(
            flow_checkpoint,
            learner_device,
            expected_camera_names=list(cfg.environment.camera_names),
            expected_action_horizon=int(cfg.flow.action_horizon),
            expected_action_dim=int(cfg.flow.action_dim),
            ode_steps=int(cfg.flow.ode_steps),
            image_size=int(cfg.environment.image_height),
        )

        def encode_observations(
            observations: Mapping[int, Mapping[str, np.ndarray]],
        ) -> dict[int, EncodedState]:
            worker_ids = list(observations)
            canonical = [
                _canonical_observation(observations[index], list(cfg.environment.camera_names))
                for index in worker_ids
            ]
            images, proprio = observation_batch_to_cuda(
                canonical,
                list(cfg.environment.camera_names),
                "state",
                inference_device,
            )
            dino_features = dino(images)
            normalized_proprio = inference_flow.normalize_proprio(proprio)
            context = inference_flow.encode_context(
                FlowObservation(images=images, proprio=proprio, language=prompt)
            )
            return {
                worker_id: EncodedState(
                    dino_features[index : index + 1],
                    normalized_proprio[index : index + 1],
                    _select_context(context, index),
                )
                for index, worker_id in enumerate(worker_ids)
            }

        normalizer = {
            "act_mean": inference_flow.act_mean.detach().to("cpu").tolist(),
            "act_std": inference_flow.act_std.detach().to("cpu").tolist(),
            "prop_mean": inference_flow.prop_mean.detach().to("cpu").tolist(),
            "prop_std": inference_flow.prop_std.detach().to("cpu").tolist(),
        }
        fingerprint, ingredients = build_cache_fingerprint(
            source_path=offline_path,
            metadata_path=metadata_path,
            dino_weights=dino_checkpoint,
            flow_checkpoint=flow_checkpoint,
            camera_names=list(cfg.environment.camera_names),
            image_size=int(cfg.vision.input_size),
            normalizer=normalizer,
            prompt=prompt,
            action_horizon=int(cfg.flow.action_horizon),
            ode_steps=int(cfg.flow.ode_steps),
        )
        cache_root = Path(to_absolute_path(str(cfg.storage.cache_root))) / task_name
        cache_path = cache_root / fingerprint
        cache_hit = cache_path.is_dir()
        if cache_hit:
            cache = CacheDataset(cache_path, expected_fingerprint=fingerprint)
        else:
            legacy = load_legacy_buffer(
                offline_path,
                metadata_path=metadata_path,
                expected_task=task_name,
                expected_horizon=int(cfg.flow.action_horizon),
                expected_cameras=list(cfg.environment.camera_names),
            )

            def cache_features(observations: Sequence[Mapping[str, np.ndarray]]) -> Mapping[str, np.ndarray]:
                indexed = {index: observation for index, observation in enumerate(observations)}
                encoded = encode_observations(indexed)
                rows = [_state_to_numpy(encoded[index]) for index in range(len(indexed))]
                return {
                    "dino_cls": np.stack([row["dino_features"] for row in rows]),
                    "proprio": np.stack([row["proprio"] for row in rows]),
                    "task_scene_cond": np.stack([row["task_scene_cond"] for row in rows]),
                    "context_tokens": np.stack([row["context_tokens"] for row in rows]),
                    "context_padding_mask": np.stack([row["context_padding_mask"] for row in rows]),
                }

            def normalize_actions(actions: np.ndarray) -> np.ndarray:
                tensor = torch.from_numpy(np.ascontiguousarray(actions)).to(inference_device)
                return inference_flow.normalize_actions(tensor).to("cpu").numpy()

            cache = build_feature_cache(
                cache_root,
                legacy,
                fingerprint=fingerprint,
                fingerprint_ingredients=ingredients,
                action_horizon=int(cfg.flow.action_horizon),
                feature_extractor=cache_features,
                action_normalizer=normalize_actions,
                feature_batch_size=int(cfg.runtime.cache_batch_size),
            )
        events.log({"type": "cache", "hit": cache_hit, "size": len(cache), "fingerprint": fingerprint})

        resume_path = None if cfg.runtime.checkpoint is None else Path(to_absolute_path(str(cfg.runtime.checkpoint)))
        resume_payload = None
        if bool(cfg.runtime.resume):
            if resume_path is None or not resume_path.is_file():
                raise FileNotFoundError("runtime.checkpoint is required when runtime.resume=true.")
            resume_payload = torch.load(resume_path, map_location="cpu", weights_only=False)
            if resume_payload["fingerprints"]["cache"] != cache.manifest["fingerprint"]:
                raise ValueError("Resume checkpoint cache fingerprint does not match the active cache.")

        maximum_online_macros = (
            int(cfg.runtime.num_envs)
            * int(cfg.runtime.episodes_per_env)
            * ((int(cfg.environment.horizon) + int(cfg.flow.action_horizon) - 1) // int(cfg.flow.action_horizon))
        )
        online = (
            OnlineReplay.load_snapshot(resume_payload["online_replay"]["path"])
            if resume_payload is not None
            else OnlineReplay(min(int(cfg.algorithm.replay_capacity), maximum_online_macros))
        )
        replay = UniformReplay(cache, online, seed=int(cfg.seed))

        def learner_decode(context: Any, noise: torch.Tensor) -> torch.Tensor:
            if isinstance(context, Mapping):
                context = FlowContext.from_mapping(context)
            normalized = learner_flow.decode_noise(context, noise, denormalize=False)
            return learner_flow.project_normalized_actions(normalized)

        network = _network_config(cfg, inference_flow.proprio_dim)
        agent = DSRLAgent(_agent_config(cfg, network), learner_decode)

        def provide_batch(_batch_size: int) -> DSRLBatch:
            return _batch_to_dsrl(prefetcher.next())

        trainer = DSRLTrainer(agent, provide_batch)
        inference_policy = DSRLInferencePolicy(network, inference_device)

        resume_counters: Mapping[str, Any] = {}
        if resume_payload is not None:
            trainer.load_state_dict(resume_payload["trainer"], strict=True)
            replay.load_state_dict(resume_payload["replay_rng"])
            resume_counters = resume_payload.get("counters", {})
            restore_rng_state(resume_payload["rng"])
            del resume_payload
        inference_policy.load_inference_state(agent.inference_state_dict())
        prefetcher = CudaBatchPrefetcher(
            replay,
            batch_size=int(cfg.algorithm.batch_size),
            device=learner_device,
            depth=int(cfg.runtime.prefetch_batches),
        )
        resources.callback(lambda: prefetcher.close())

        runtime = RobosuiteVectorRuntime(
            env_metadata=env_metadata,
            camera_names=list(cfg.environment.camera_names),
            image_height=int(cfg.environment.image_height),
            image_width=int(cfg.environment.image_width),
            control_frequency=int(cfg.environment.control_frequency),
            horizon=int(cfg.environment.horizon),
            num_envs=int(cfg.runtime.num_envs),
            seed=int(cfg.seed),
        )
        resources.callback(runtime.close)

        def act(worker_ids: Sequence[int], states: Mapping[int, EncodedState]) -> PolicyDecision:
            batched = _stack_states([states[index] for index in worker_ids])
            latent = inference_policy.latent(
                batched.dino_features,
                batched.proprio,
                deterministic=False,
            )
            environment_actions = inference_flow.decode_noise(batched.flow_context, latent)
            normalized_actions = inference_flow.normalize_actions(environment_actions)
            return PolicyDecision(
                environment_actions.to("cpu").numpy(),
                normalized_actions.to("cpu").numpy(),
            )

        def add_transition(
            _worker_id: int,
            state: EncodedState,
            next_state: EncodedState,
            normalized_actions: np.ndarray,
            result: Any,
        ) -> None:
            current = _state_to_numpy(state)
            following = _state_to_numpy(next_state)
            online.add(
                CompactTransition(
                    **current,
                    next_dino_features=following["dino_features"],
                    next_proprio=following["proprio"],
                    next_task_scene_cond=following["task_scene_cond"],
                    next_context_tokens=following["context_tokens"],
                    next_context_padding_mask=following["context_padding_mask"],
                    actions=normalized_actions,
                    reward=float(result.reward),
                    done=bool(result.done),
                    executed_length=int(result.executed_length),
                )
            )

        def on_update(metrics: Mapping[str, float], macro_step: int) -> None:
            if not all(np.isfinite(float(value)) for value in metrics.values()):
                raise FloatingPointError("Learner produced non-finite metrics.")
            record = {"type": "learner", "macro_step": macro_step, **metrics}
            events.log(record)
            tensorboard.log(metrics, step=macro_step, prefix="learner")
            system_metrics = {
                "offline_replay": len(cache),
                "online_replay": len(online),
                **_gpu_metrics(learner_device, "learner"),
                **_gpu_metrics(inference_device, "inference"),
            }
            events.log({"type": "system", "macro_step": macro_step, **system_metrics})
            tensorboard.log(system_metrics, step=macro_step, prefix="system")

        def on_episode(metrics: dict[str, Any]) -> None:
            events.log({"type": "episode", **metrics})
            tensorboard.log(
                {key: value for key, value in metrics.items() if key not in {"termination_reason"}},
                step=int(metrics["episode"]),
                prefix="episode",
            )
            print(
                f"[episode] index={metrics['episode']} worker={metrics['worker']} "
                f"success={int(metrics['success'])} return={metrics['return']:.1f} "
                f"primitive={metrics['primitive_length']} macro={metrics['macro_length']} "
                f"reason={metrics['termination_reason']} fps={metrics['fps']:.1f}"
            )

        checkpoint: Path | None = None
        expected = int(cfg.runtime.num_envs) * int(cfg.runtime.episodes_per_env)

        def on_checkpoint(progress: CollectionSummary) -> None:
            nonlocal checkpoint, prefetcher
            prefetcher.close()
            events.log({"type": "checkpoint_pending", "episode": progress.completed_episodes})
            tensorboard.flush()
            checkpoint = _save_training_state(
                trainer=trainer,
                run_dir=run_dir,
                cfg=cfg,
                completed_episodes=progress.completed_episodes,
                completed_by_worker=progress.completed_by_worker,
                primitive_steps=progress.primitive_steps,
                macro_steps=progress.macro_steps,
                online=online,
                replay=replay,
                cache=cache,
                flow_checkpoint=flow_checkpoint,
                dino_checkpoint=dino_checkpoint,
                normalizers=normalizer,
            )
            events.log(
                {
                    "type": "checkpoint",
                    "episode": progress.completed_episodes,
                    "path": os.fspath(checkpoint),
                }
            )
            if progress.completed_episodes < expected:
                prefetcher = CudaBatchPrefetcher(
                    replay,
                    batch_size=int(cfg.algorithm.batch_size),
                    device=learner_device,
                    depth=int(cfg.runtime.prefetch_batches),
                )

        summary = run_online_collection(
            runtime=runtime,
            num_envs=int(cfg.runtime.num_envs),
            episodes_per_env=int(cfg.runtime.episodes_per_env),
            encode=encode_observations,
            act=act,
            add_transition=add_transition,
            update=trainer.update_cycle,
            sync_inference=lambda: inference_policy.load_inference_state(
                agent.inference_state_dict()
            ),
            on_episode=on_episode,
            on_update=on_update,
            on_checkpoint=on_checkpoint,
            checkpoint_interval_episodes=int(cfg.storage.checkpoint_interval_episodes),
            initial_completed_by_worker=resume_counters.get("completed_by_worker"),
            initial_primitive_steps=int(resume_counters.get("primitive_steps", 0)),
            initial_macro_steps=int(resume_counters.get("macro_steps", 0)),
        )
        if summary.completed_episodes != expected or any(
            count != int(cfg.runtime.episodes_per_env) for count in summary.completed_by_worker
        ):
            raise RuntimeError(f"Episode quota mismatch: {summary}.")
        if checkpoint is None or checkpoint.name != f"episode_{summary.completed_episodes:08d}.pt":
            on_checkpoint(summary)
        assert checkpoint is not None
        events.log({"type": "completed", "checkpoint": os.fspath(checkpoint), **summary.__dict__})
        tensorboard.flush()
        metadata.update({"finished_at": time.time(), "summary": summary.__dict__, "checkpoint": os.fspath(checkpoint)})
        write_json(run_dir / str(cfg.storage.metadata_filename), metadata)
        print(f"Training completed: episodes={summary.completed_episodes} checkpoint={checkpoint}")


@hydra.main(version_base=None, config_path="config", config_name="overall")
def main(cfg: DictConfig) -> None:
    _run(cfg)


if __name__ == "__main__":
    main()
