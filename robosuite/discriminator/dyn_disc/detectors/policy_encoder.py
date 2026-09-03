"""Frozen flow-policy encoder used by the discriminator ablation."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch


PREPROCESS_VERSION = "flow_multi_eval_center_crop_128_imagenet_v1"
LATENT_NAME = "task_scene_cond"


def _load_module(module_name: str, path: Path):
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _policy_source_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "policy" / "flow_multi-update"


def _load_policy_builders():
    source_dir = _policy_source_dir()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"flow_multi-update source not found: {source_dir}")
    source_str = str(source_dir)
    if source_str not in sys.path:
        sys.path.insert(0, source_str)
    model_module = _load_module("_flow_multi_policy_model", source_dir / "model.py")
    env_module = _load_module("_flow_multi_policy_env_util", source_dir / "utils" / "env_util.py")
    return model_module.build_flow_policy, env_module.RobosuiteProprioExtractor


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class PolicyFeatureEncoder:
    """CUDA-only FP32 wrapper exposing the policy ``task_scene_cond`` latent."""

    feature_dim = 256
    latent_name = LATENT_NAME
    preprocess_version = PREPROCESS_VERSION

    def __init__(self, policy_ckpt: str, device: str = "cuda") -> None:
        requested_device = torch.device(device)
        if requested_device.type != "cuda":
            raise ValueError(f"PolicyFeatureEncoder requires a CUDA device, got {device!r}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for policy feature extraction")

        self.device = requested_device
        self.policy_ckpt = str(Path(policy_ckpt).resolve())
        ckpt_path = Path(self.policy_ckpt)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Policy checkpoint not found: {ckpt_path}")
        self.checkpoint_sha256 = sha256_file(ckpt_path)
        self.policy_ckpt_hash = self.checkpoint_sha256

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        required = {
            "ema_model",
            "model_cfg",
            "camera_names",
            "prop_mean",
            "prop_std",
            "act_mean",
            "task_prompt_map",
            "task_metadata_map",
        }
        missing = sorted(required - set(checkpoint))
        if missing:
            raise KeyError(f"Policy checkpoint is missing required keys: {missing}")

        self.camera_names = [str(name) for name in checkpoint["camera_names"]]
        if len(self.camera_names) != 3:
            raise ValueError(
                f"Expected the policy checkpoint to contain three cameras, got {self.camera_names}"
            )
        self.task_prompt_map = dict(checkpoint["task_prompt_map"])
        self.task_metadata_map = dict(checkpoint["task_metadata_map"])
        self.prop_mean = np.asarray(checkpoint["prop_mean"], dtype=np.float32).reshape(-1)
        self.prop_std = np.asarray(checkpoint["prop_std"], dtype=np.float32).reshape(-1)
        if self.prop_mean.shape != self.prop_std.shape or np.any(self.prop_std <= 0):
            raise ValueError("Invalid policy proprio normalization statistics")

        action_stats = np.asarray(checkpoint["act_mean"])
        if action_stats.ndim < 2:
            raise ValueError(f"Invalid action statistics shape: {action_stats.shape}")
        action_dim = int(action_stats.shape[-1])
        build_flow_policy, extractor_cls = _load_policy_builders()
        self._extractor_cls = extractor_cls
        self.model = build_flow_policy(
            checkpoint["model_cfg"],
            proprio_dim=int(self.prop_mean.shape[0]),
            action_dim=action_dim,
            camera_names=self.camera_names,
        ).to(device=self.device, dtype=torch.float32)
        self.model.load_state_dict(checkpoint["ema_model"], strict=True)
        self.model.requires_grad_(False)
        self.model.eval()
        if int(self.model.condition_aggregator.output_dim) != self.feature_dim:
            raise ValueError(
                "Policy task_scene_cond dimension mismatch: "
                f"expected {self.feature_dim}, got {self.model.condition_aggregator.output_dim}"
            )

        self._language_cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._extractors: dict[str, Any] = {}
        self._extractor_lock = threading.Lock()
        self._resize_index_cache: dict[int, torch.Tensor] = {}
        self._mean = torch.tensor(
            [0.485, 0.456, 0.406], device=self.device, dtype=torch.float32
        ).view(1, 1, 3, 1, 1)
        self._std = torch.tensor(
            [0.229, 0.224, 0.225], device=self.device, dtype=torch.float32
        ).view(1, 1, 3, 1, 1)
        del checkpoint

    @property
    def prompt_map(self) -> dict[str, Any]:
        return self.task_prompt_map

    def prompt_for_task(self, task_name: str) -> str:
        if task_name not in self.task_prompt_map:
            raise KeyError(f"Task {task_name!r} is not present in policy task_prompt_map")
        value = self.task_prompt_map[task_name]
        if isinstance(value, (list, tuple)):
            if not value:
                raise ValueError(f"Task {task_name!r} has an empty prompt list")
            return str(value[0])
        return str(value)

    def _extractor_for_task(self, task_name: str):
        if task_name not in self.task_metadata_map:
            raise KeyError(f"Task {task_name!r} is not present in policy task_metadata_map")
        with self._extractor_lock:
            extractor = self._extractors.get(task_name)
            if extractor is None:
                extractor = self._extractor_cls(
                    env_kwargs=self.task_metadata_map[task_name],
                    has_renderer=False,
                    has_offscreen_renderer=False,
                    use_camera_obs=False,
                    camera_names=None,
                    reward_shaping=False,
                )
                self._extractors[task_name] = extractor
        return extractor

    def extract_proprio(self, states: np.ndarray, task_name: str) -> np.ndarray:
        """Vectorized equivalent of the policy dataset's qpos/qvel extractor."""
        states = np.asarray(states)
        if states.ndim != 2:
            raise ValueError(f"states must be 2D, got {states.shape}")
        extractor = self._extractor_for_task(str(task_name))
        nq = int(extractor.sim.model.nq)
        nv = int(extractor.sim.model.nv)
        na = int(getattr(extractor.sim.model, "na", 0))
        state_dim = int(states.shape[1])
        core_dim = nq + nv
        if state_dim in (core_dim, core_dim + na):
            offset = 0
        elif state_dim in (core_dim + 1, core_dim + na + 1):
            offset = 1
        else:
            raise ValueError(
                f"Unexpected flattened state length {state_dim} for task {task_name!r}"
            )
        qpos = states[:, offset : offset + nq]
        qvel = states[:, offset + nq : offset + nq + nv]
        proprio = np.concatenate(
            [qpos[:, extractor.qpos_indices], qvel[:, extractor.qvel_indices]], axis=1
        ).astype(np.float32, copy=False)
        if proprio.shape[1] != self.prop_mean.shape[0]:
            raise ValueError(
                f"Proprio dimension mismatch for {task_name!r}: "
                f"checkpoint={self.prop_mean.shape[0]} extracted={proprio.shape[1]}"
            )
        return ((proprio - self.prop_mean) / self.prop_std).astype(np.float32, copy=False)

    def preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """Apply the exact deterministic flow-policy evaluation preprocessing."""
        images = images.to(device=self.device, non_blocking=True)
        if images.ndim != 5:
            raise ValueError(f"images must be 5D, got {tuple(images.shape)}")
        if images.shape[-1] == 3:
            images = images.permute(0, 1, 4, 2, 3)
        elif images.shape[2] != 3:
            raise ValueError(f"images must be BHWC or BCHW per camera, got {tuple(images.shape)}")
        scale_uint8 = images.dtype == torch.uint8
        images = images.to(dtype=torch.float32)
        if scale_uint8:
            images = images.div_(255.0)

        height, width = int(images.shape[-2]), int(images.shape[-1])
        crop_size = min(height, width)
        y0 = (height - crop_size) // 2
        x0 = (width - crop_size) // 2
        images = images[..., y0 : y0 + crop_size, x0 : x0 + crop_size]
        if crop_size != 128:
            indices = self._resize_index_cache.get(crop_size)
            if indices is None:
                indices = torch.from_numpy(
                    np.linspace(0, crop_size - 1, 128).astype(np.int64)
                ).to(self.device)
                self._resize_index_cache[crop_size] = indices
            images = images.index_select(-2, indices).index_select(-1, indices)
        return (images - self._mean) / self._std

    def _language_features(self, task_name: str):
        cached = self._language_cache.get(task_name)
        if cached is None:
            with torch.inference_mode():
                cached = tuple(
                    value.detach()
                    for value in self.model.language_encoder([self.prompt_for_task(task_name)])
                )
            self._language_cache[task_name] = cached
        return cached

    def _encode_with_cached_language(
        self,
        images: torch.Tensor,
        proprio: torch.Tensor,
        task_name: str,
    ) -> torch.Tensor:
        batch_size, num_cameras, channels, height, width = images.shape
        if num_cameras != len(self.camera_names):
            raise ValueError(f"Expected {len(self.camera_names)} cameras, got {num_cameras}")
        flat_images = images.reshape(batch_size * num_cameras, channels, height, width)
        flat_images = flat_images.contiguous(memory_format=torch.channels_last)
        image_tokens = self.model.image_encoder(flat_images)
        image_tokens = image_tokens.reshape(
            batch_size, num_cameras * image_tokens.shape[1], image_tokens.shape[2]
        )
        proprio_tokens = self.model.proprio_tokenizer(proprio)
        language_tokens, language_global, language_mask = self._language_features(task_name)
        language_tokens = language_tokens.expand(batch_size, -1, -1)
        language_global = language_global.expand(batch_size, -1)
        language_mask = language_mask.expand(batch_size, -1)
        image_tokens, proprio_tokens = self.model.language_guided_modulation(
            visual_tokens=image_tokens,
            proprio_tokens=proprio_tokens,
            language_global=language_global,
        )
        fused_tokens, token_padding_mask = self.model.fusion(
            language_tokens=language_tokens,
            language_mask=language_mask,
            proprio_tokens=proprio_tokens,
            image_tokens=image_tokens,
            language_global=language_global,
        )
        return self.model.condition_aggregator(
            fused_tokens=fused_tokens,
            token_padding_mask=token_padding_mask,
            language_global=language_global,
        )

    @torch.inference_mode()
    def encode_batch(
        self,
        images: torch.Tensor,
        proprio: torch.Tensor,
        task_name: str,
    ) -> torch.Tensor:
        self.model.eval()
        images = self.preprocess_images(images)
        proprio = proprio.to(
            device=self.device, dtype=torch.float32, non_blocking=True
        )
        features = self._encode_with_cached_language(images, proprio, str(task_name))
        if features.ndim != 2 or features.shape != (images.shape[0], self.feature_dim):
            raise RuntimeError(f"Unexpected policy latent shape: {tuple(features.shape)}")
        return features.to(dtype=torch.float32)

    def metadata(self) -> dict[str, Any]:
        return {
            "policy_ckpt": self.policy_ckpt,
            "policy_ckpt_hash": self.policy_ckpt_hash,
            "policy_weight_source": "ema_model",
            "feature_source": "policy_task_scene_cond",
            "latent": self.latent_name,
            "feature_dim": self.feature_dim,
            "dtype": "float32",
            "camera_names": list(self.camera_names),
            "task_prompts": {
                task: self.prompt_for_task(task) for task in sorted(self.task_prompt_map)
            },
            "preprocess_version": self.preprocess_version,
        }

    def close(self) -> None:
        for extractor in self._extractors.values():
            extractor.close()
        self._extractors.clear()
        self._language_cache.clear()
