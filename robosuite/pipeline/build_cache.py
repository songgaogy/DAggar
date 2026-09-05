from __future__ import annotations

"""Build the PickPlaceCereal DSRL feature cache on CUDA."""

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from robosuite.pipeline.src.data import (
    CacheDataset,
    CacheValidationError,
    build_cache_fingerprint,
    build_feature_cache,
    load_legacy_buffer,
)
from robosuite.pipeline.src.environment import (
    FlowObservation,
    FlowPolicyAdapter,
    observation_batch_to_cuda,
)
from robosuite.pipeline.src.vision import DinoV2Encoder
from robosuite.pipeline.utils.tensor import resolve_cuda_device


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
        "act_mean": np.asarray(checkpoint["act_mean"], dtype=np.float32),
        "act_std": np.asarray(checkpoint["act_std"], dtype=np.float32),
        "prop_mean": np.asarray(checkpoint["prop_mean"], dtype=np.float32),
        "prop_std": np.asarray(checkpoint["prop_std"], dtype=np.float32),
        "task_metadata": dict(task_metadata),
        "prompts": tuple(str(prompt) for prompt in prompts),
    }
    del checkpoint
    return result


def cache_identity(cfg: DictConfig, metadata: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    prompt_index = int(cfg.flow.prompt_index)
    prompts = metadata["prompts"]
    if not 0 <= prompt_index < len(prompts):
        raise IndexError(f"flow.prompt_index {prompt_index} is out of range.")
    normalizer = {
        "act_mean": metadata["act_mean"].tolist(),
        "act_std": metadata["act_std"].tolist(),
        "prop_mean": metadata["prop_mean"].tolist(),
        "prop_std": metadata["prop_std"].tolist(),
    }
    return build_cache_fingerprint(
        source_path=_absolute(cfg.inputs.offline_transitions),
        metadata_path=_absolute(cfg.inputs.offline_metadata),
        dino_weights=_absolute(cfg.inputs.dinov2_checkpoint),
        flow_checkpoint=_absolute(cfg.inputs.base_policy_checkpoint),
        camera_names=metadata["camera_names"],
        image_size=int(cfg.vision.input_size),
        normalizer=normalizer,
        prompt=prompts[prompt_index],
        action_horizon=int(cfg.flow.action_horizon),
        ode_steps=int(cfg.flow.ode_steps),
    )


def ensure_feature_cache(
    cfg: DictConfig,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[CacheDataset, str, Mapping[str, Any]]:
    """Return a valid cache, atomically building it when absent or stale."""

    resolve_cuda_device(cfg.runtime.inference_device)
    task_name = str(cfg.task.name)
    checkpoint_path = _absolute(cfg.inputs.base_policy_checkpoint)
    metadata = load_flow_metadata(checkpoint_path, task_name) if metadata is None else metadata
    fingerprint, ingredients = cache_identity(cfg, metadata)
    cache_root = _absolute(cfg.storage.cache_root) / task_name
    destination = cache_root / fingerprint
    if destination.exists():
        try:
            cache = CacheDataset(destination, expected_fingerprint=fingerprint)
            print(f"[cache] hit path={cache.path} transitions={len(cache)}")
            return cache, fingerprint, metadata
        except CacheValidationError as error:
            print(f"[cache] stale path={destination} reason={error}")

    device = torch.device(str(cfg.runtime.inference_device))
    cameras = tuple(metadata["camera_names"])
    prompt = metadata["prompts"][int(cfg.flow.prompt_index)]
    print(f"[cache] loading frozen encoders on {device}")
    dino = DinoV2Encoder(_absolute(cfg.inputs.dinov2_checkpoint), device)
    flow = FlowPolicyAdapter(
        checkpoint_path,
        device,
        expected_camera_names=cameras,
        expected_action_horizon=int(cfg.flow.action_horizon),
        expected_action_dim=int(cfg.flow.action_dim),
        ode_steps=int(cfg.flow.ode_steps),
        image_size=int(cfg.environment.image_height),
    )
    feature_dtype = np.float16 if str(cfg.vision.cache_dtype) == "float16" else np.float32
    processed = 0
    started = time.perf_counter()

    def extract_features(observations: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
        nonlocal processed
        images, proprio = observation_batch_to_cuda(observations, cameras, "state", device)
        normalized_proprio = flow.normalize_proprio(proprio)
        with torch.inference_mode():
            dino_cls = dino(images)
            context = flow.encode_context(FlowObservation(images, proprio, prompt))
        processed += len(observations)
        if processed % (int(cfg.runtime.cache_batch_size) * 20) == 0:
            elapsed = max(time.perf_counter() - started, 1e-6)
            print(f"[cache] states={processed} throughput={processed / elapsed:.1f}/s")
        return {
            "dino_cls": dino_cls.to(torch.float16).cpu().numpy().astype(feature_dtype, copy=False),
            "proprio": normalized_proprio.to(torch.float16).cpu().numpy().astype(feature_dtype, copy=False),
            "task_scene_cond": context.task_scene_cond.to(torch.float16).cpu().numpy().astype(feature_dtype, copy=False),
            "context_tokens": context.context_tokens.to(torch.float16).cpu().numpy().astype(feature_dtype, copy=False),
            "context_padding_mask": context.context_padding_mask.cpu().numpy().astype(np.bool_, copy=False),
        }

    def normalize_actions(actions: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(np.ascontiguousarray(actions)).to(device, non_blocking=True)
        return flow.normalize_actions(tensor).cpu().numpy()

    print(f"[cache] materializing legacy replay once: {_absolute(cfg.inputs.offline_transitions)}")
    legacy = load_legacy_buffer(
        _absolute(cfg.inputs.offline_transitions),
        metadata_path=_absolute(cfg.inputs.offline_metadata),
        expected_task=task_name,
        expected_horizon=int(cfg.flow.action_horizon),
        expected_cameras=cameras,
    )
    cache = build_feature_cache(
        cache_root,
        legacy,
        fingerprint=fingerprint,
        fingerprint_ingredients=ingredients,
        action_horizon=int(cfg.flow.action_horizon),
        feature_extractor=extract_features,
        action_normalizer=normalize_actions,
        feature_batch_size=int(cfg.runtime.cache_batch_size),
        action_batch_size=4096,
    )
    print(f"[cache] complete path={cache.path} transitions={len(cache)}")
    return cache, fingerprint, metadata


@hydra.main(version_base="1.2", config_path="config", config_name="overall")
def main(cfg: DictConfig) -> None:
    if OmegaConf.is_missing(cfg, "task"):
        raise ValueError("Specify a task, for example: task=PickPlaceCereal")
    ensure_feature_cache(cfg)


if __name__ == "__main__":
    main()


__all__ = [
    "CacheDataset",
    "build_cache_fingerprint",
    "build_feature_cache",
    "cache_identity",
    "ensure_feature_cache",
    "load_flow_metadata",
    "load_legacy_buffer",
]
