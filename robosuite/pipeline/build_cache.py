from __future__ import annotations

"""Build or validate the shared DSRL-SAC warmup cache."""

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

from robosuite.pipeline.orchestration import (
    PolicyDecision,
    distribute_episode_quota,
    run_episode_collection,
)
from robosuite.pipeline.src.data import (
    WarmupReplay,
    WarmupValidationError,
    build_warmup_fingerprint,
    load_warmup_cache,
    save_warmup_cache,
)
from robosuite.pipeline.src.environment import (
    FlowContext,
    FlowObservation,
    FlowPolicyAdapter,
    RobosuiteVectorRuntime,
    observation_batch_to_cuda,
)
from robosuite.pipeline.utils import require_cuda


@dataclass(frozen=True)
class WarmupState:
    visual_features: torch.Tensor
    proprio: torch.Tensor
    flow_context: FlowContext


def _absolute(path: str) -> Path:
    return Path(to_absolute_path(path)).resolve()


def load_flow_metadata(path: str | Path, task_name: str) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    task_metadata = checkpoint.get("task_metadata_map", {}).get(task_name)
    prompts = checkpoint.get("task_prompt_map", {}).get(task_name)
    if not isinstance(task_metadata, Mapping):
        raise KeyError(f"Flow checkpoint has no environment metadata for {task_name}.")
    if isinstance(prompts, str):
        prompts = [prompts]
    if not isinstance(prompts, Sequence) or not prompts:
        raise KeyError(f"Flow checkpoint has no prompt for {task_name}.")
    result = {
        "camera_names": tuple(str(name) for name in checkpoint["camera_names"]),
        "task_metadata": dict(task_metadata),
        "prompts": tuple(str(prompt) for prompt in prompts),
    }
    del checkpoint
    return result


def _stack_context(contexts: Sequence[FlowContext]) -> FlowContext:
    return FlowContext.stack(contexts)


def _stack_states(states: Sequence[WarmupState]) -> WarmupState:
    return WarmupState(
        visual_features=torch.cat([state.visual_features for state in states]),
        proprio=torch.cat([state.proprio for state in states]),
        flow_context=_stack_context([state.flow_context for state in states]),
    )


def _state_numpy(state: WarmupState, feature_dtype: np.dtype) -> tuple[np.ndarray, np.ndarray]:
    visual = state.visual_features[0].detach().cpu().numpy().astype(feature_dtype, copy=False)
    proprio = state.proprio[0].detach().float().cpu().numpy().astype(np.float32, copy=False)
    return visual, proprio


def warmup_identity(
    cfg: DictConfig,
    metadata: Mapping[str, Any],
    *,
    tokens_per_camera: int,
    token_dim: int,
    proprio_dim: int,
) -> tuple[str, dict[str, Any]]:
    prompt_index = int(cfg.flow.prompt_index)
    prompts = metadata["prompts"]
    if not 0 <= prompt_index < len(prompts):
        raise IndexError(f"flow.prompt_index {prompt_index} is out of range.")
    cameras = tuple(metadata["camera_names"])
    return build_warmup_fingerprint(
        task_name=str(cfg.task.name),
        warmup_seed=int(cfg.warmup.seed),
        base_checkpoint=_absolute(str(cfg.inputs.base_policy_checkpoint)),
        env_metadata={
            **dict(metadata["task_metadata"]),
            "dsrl_collection": {
                "episodes": int(cfg.warmup.episodes),
                "num_envs": int(cfg.runtime.num_envs),
                "horizon": int(cfg.environment.horizon),
                "control_frequency": int(cfg.environment.control_frequency),
                "image_height": int(cfg.environment.image_height),
                "image_width": int(cfg.environment.image_width),
            },
        },
        camera_names=cameras,
        action_horizon=int(cfg.flow.action_horizon),
        ode_config={
            "steps": int(cfg.flow.ode_steps),
            "prompt_index": prompt_index,
            "latent_distribution": "clipped_standard_normal",
            "latent_bound": float(cfg.algorithm.latent_action_bound),
        },
        feature_schema={
            "visual_features": [len(cameras), int(tokens_per_camera), int(token_dim)],
            "proprio": [int(proprio_dim)],
            "dtype": str(cfg.vision.cache_dtype),
        },
        reward_schema={"failure": 0.0, "success": 1.0},
    )


def ensure_warmup_cache(
    cfg: DictConfig,
    *,
    allow_generate: bool = True,
) -> tuple[WarmupReplay, str, Mapping[str, Any], FlowPolicyAdapter]:
    """Return the task cache, collecting frozen-policy episodes only on a miss."""

    require_cuda()
    task_name = str(cfg.task.name)
    checkpoint_path = _absolute(str(cfg.inputs.base_policy_checkpoint))
    metadata = load_flow_metadata(checkpoint_path, task_name)
    device = torch.device(str(cfg.runtime.inference_device))
    cameras = tuple(metadata["camera_names"])
    flow = FlowPolicyAdapter(
        checkpoint_path,
        device,
        expected_camera_names=cameras,
        expected_action_horizon=int(cfg.flow.action_horizon),
        expected_action_dim=int(cfg.flow.action_dim),
        ode_steps=int(cfg.flow.ode_steps),
        image_size=int(cfg.environment.image_height),
    )
    fingerprint, ingredients = warmup_identity(
        cfg,
        metadata,
        tokens_per_camera=flow.image_tokens_per_camera,
        token_dim=flow.image_token_dim,
        proprio_dim=flow.proprio_dim,
    )
    cache_root = _absolute(str(cfg.storage.warmup_cache_root)) / task_name
    cache_path = cache_root / fingerprint
    try:
        replay = load_warmup_cache(cache_root, fingerprint)
        print(f"[warmup-cache] hit path={replay.path} transitions={len(replay)}")
        return replay, fingerprint, metadata, flow
    except WarmupValidationError:
        if cache_path.exists():
            raise
        if not allow_generate:
            raise

    prompt = metadata["prompts"][int(cfg.flow.prompt_index)]
    feature_dtype = np.float16 if str(cfg.vision.cache_dtype) == "float16" else np.float32
    generator = torch.Generator(device=device)
    generator.manual_seed(int(cfg.warmup.seed))
    episode_rows: list[list[dict[str, np.ndarray]]] = []
    active_rows: dict[int, list[dict[str, np.ndarray]]] = {
        worker: [] for worker in range(int(cfg.runtime.num_envs))
    }

    def encode_observations(
        observations: Mapping[int, Mapping[str, np.ndarray]],
    ) -> dict[int, WarmupState]:
        worker_ids = list(observations)
        batch = [observations[worker] for worker in worker_ids]
        images, proprio = observation_batch_to_cuda(batch, cameras, "state", device)
        normalized_proprio = flow.normalize_proprio(proprio)
        context, image_tokens = flow.encode_context_with_image_tokens(
            FlowObservation(images, proprio, prompt)
        )
        return {
            worker: WarmupState(
                image_tokens[row : row + 1],
                normalized_proprio[row : row + 1],
                FlowContext(
                    context.task_scene_cond[row : row + 1],
                    context.context_tokens[row : row + 1],
                    context.context_padding_mask[row : row + 1],
                ),
            )
            for row, worker in enumerate(worker_ids)
        }

    def act(worker_ids: Sequence[int], states: Mapping[int, WarmupState]) -> PolicyDecision:
        batched = _stack_states([states[worker] for worker in worker_ids])
        latents = torch.randn(
            (len(worker_ids), int(cfg.flow.action_horizon), int(cfg.flow.action_dim)),
            device=device,
            dtype=torch.float32,
            generator=generator,
        ).clamp_(-float(cfg.algorithm.latent_action_bound), float(cfg.algorithm.latent_action_bound))
        actions = flow.decode_noise(batched.flow_context, latents)
        return PolicyDecision(actions.cpu().numpy(), latents.cpu().numpy())

    def add_transition(
        worker: int,
        state: WarmupState,
        next_state: WarmupState,
        latent: np.ndarray,
        result: Any,
    ) -> None:
        visual, proprio = _state_numpy(state, feature_dtype)
        next_visual, next_proprio = _state_numpy(next_state, feature_dtype)
        active_rows[worker].append(
            {
                "visual_features": visual,
                "proprio": proprio,
                "next_visual_features": next_visual,
                "next_proprio": next_proprio,
                "actions": np.asarray(latent, dtype=np.float32),
                "rewards": np.asarray([result.reward], dtype=np.float32),
                "dones": np.asarray([result.done], dtype=np.bool_),
                "executed_length": np.asarray([result.executed_length], dtype=np.uint8),
            }
        )

    def finish_episode(metrics: Mapping[str, Any]) -> None:
        worker = int(metrics["worker"])
        episode_rows.append(active_rows[worker])
        active_rows[worker] = []
        print(
            f"[warmup] episode={metrics['episode']}/{cfg.warmup.episodes} "
            f"worker={worker} success={int(metrics['success'])}"
        )

    print(f"[warmup-cache] miss; collecting {cfg.warmup.episodes} episodes on {device}")
    with ExitStack() as resources:
        runtime = RobosuiteVectorRuntime(
            env_metadata=dict(metadata["task_metadata"]),
            camera_names=cameras,
            image_height=int(cfg.environment.image_height),
            image_width=int(cfg.environment.image_width),
            control_frequency=int(cfg.environment.control_frequency),
            horizon=int(cfg.environment.horizon),
            num_envs=int(cfg.runtime.num_envs),
            seed=int(cfg.warmup.seed),
        )
        resources.callback(runtime.close)
        summary = run_episode_collection(
            runtime=runtime,
            target_by_worker=distribute_episode_quota(
                int(cfg.warmup.episodes), int(cfg.runtime.num_envs)
            ),
            encode=encode_observations,
            act=act,
            add_transition=add_transition,
            on_episode=finish_episode,
        )

    rows = [row for episode in episode_rows for row in episode]
    if any(active_rows.values()) or len(episode_rows) != int(cfg.warmup.episodes):
        raise RuntimeError("Warmup collection ended with incomplete episode data.")
    arrays = {
        key: np.stack([row[key] for row in rows])
        for key in rows[0]
    }
    boundaries = [0]
    for episode in episode_rows:
        boundaries.append(boundaries[-1] + len(episode))
    replay = save_warmup_cache(
        cache_root,
        fingerprint=fingerprint,
        fingerprint_ingredients=ingredients,
        arrays=arrays,
        episode_boundaries=boundaries,
        metadata={
            "completed_by_worker": list(summary.completed_by_worker),
            "primitive_steps": summary.primitive_steps,
            "macro_steps": summary.macro_steps,
            "vector_steps": summary.online_vector_steps,
            "task": task_name,
        },
    )
    print(f"[warmup-cache] complete path={replay.path} transitions={len(replay)}")
    return replay, fingerprint, metadata, flow


@hydra.main(version_base="1.2", config_path="config", config_name="overall")
def main(cfg: DictConfig) -> None:
    if OmegaConf.is_missing(cfg, "task"):
        raise ValueError("Specify a task, for example: task=PickPlaceCereal")
    ensure_warmup_cache(cfg)


if __name__ == "__main__":
    main()


__all__ = ["ensure_warmup_cache", "load_flow_metadata", "warmup_identity"]
