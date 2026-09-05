from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from robosuite.pipeline.utils.tensor import resolve_cuda_device
from robosuite.policy.flow_multi import build_flow_policy


@dataclass(frozen=True)
class FlowObservation:
    images: torch.Tensor
    proprio: torch.Tensor
    language: str | Sequence[str]
    images_preprocessed: bool = False
    proprio_normalized: bool = False


@dataclass(frozen=True)
class FlowContext:
    task_scene_cond: torch.Tensor
    context_tokens: torch.Tensor
    context_padding_mask: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.task_scene_cond.shape[0])

    def to_mapping(self) -> dict[str, torch.Tensor]:
        return {
            "task_scene_cond": self.task_scene_cond,
            "context_tokens": self.context_tokens,
            "context_padding_mask": self.context_padding_mask,
        }

    @classmethod
    def from_mapping(cls, values: Mapping[str, torch.Tensor]) -> "FlowContext":
        required = ("task_scene_cond", "context_tokens", "context_padding_mask")
        missing = [key for key in required if key not in values]
        if missing:
            raise KeyError(f"Cached flow context is missing keys: {missing}")
        return cls(*(values[key] for key in required))

    @classmethod
    def stack(cls, contexts: Sequence["FlowContext"]) -> "FlowContext":
        if not contexts:
            raise ValueError("Cannot stack an empty flow context sequence")
        return cls(
            task_scene_cond=torch.cat([item.task_scene_cond for item in contexts], dim=0),
            context_tokens=torch.cat([item.context_tokens for item in contexts], dim=0),
            context_padding_mask=torch.cat(
                [item.context_padding_mask for item in contexts], dim=0
            ),
        )


def _nested_get(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


class FlowPolicyAdapter(nn.Module):
    """Strict CUDA-only adapter around the frozen importable flow_multi policy."""

    required_checkpoint_keys = (
        "model_cfg",
        "camera_names",
        "act_mean",
        "act_std",
        "prop_mean",
        "prop_std",
    )

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device,
        *,
        expected_camera_names: Sequence[str] | None = None,
        expected_action_horizon: int | None = None,
        expected_action_dim: int | None = None,
        ode_steps: int = 10,
        image_size: int | None = None,
        action_low: float | Sequence[float] | np.ndarray = -1.0,
        action_high: float | Sequence[float] | np.ndarray = 1.0,
        model_builder: Callable[..., nn.Module] = build_flow_policy,
    ) -> None:
        super().__init__()
        self.device = resolve_cuda_device(device)
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Flow checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, Mapping):
            raise TypeError("Flow checkpoint must be a mapping")
        missing = [key for key in self.required_checkpoint_keys if key not in checkpoint]
        if missing:
            raise KeyError(f"Flow checkpoint is missing required keys: {missing}")
        if "ema_model" not in checkpoint and "model" not in checkpoint:
            raise KeyError("Flow checkpoint must contain 'ema_model' or 'model'")

        self.camera_names = tuple(str(name) for name in checkpoint["camera_names"])
        if not self.camera_names:
            raise ValueError("Flow checkpoint camera_names must not be empty")
        if expected_camera_names is not None and tuple(expected_camera_names) != self.camera_names:
            raise ValueError(
                f"Flow camera mismatch: expected {tuple(expected_camera_names)}, "
                f"checkpoint has {self.camera_names}"
            )
        configured_cameras = _nested_get(checkpoint, "cfg", "data", "camera_names")
        if configured_cameras is not None and tuple(configured_cameras) != self.camera_names:
            raise ValueError(
                f"Flow checkpoint camera metadata is inconsistent: top-level={self.camera_names}, "
                f"cfg.data={tuple(configured_cameras)}"
            )

        act_mean, act_std = self._validate_normalizer(
            "action", checkpoint["act_mean"], checkpoint["act_std"], rank=2
        )
        prop_mean, prop_std = self._validate_normalizer(
            "proprio", checkpoint["prop_mean"], checkpoint["prop_std"], rank=1
        )
        self.action_horizon, self.action_dim = map(int, act_mean.shape)
        self.proprio_dim = int(prop_mean.shape[0])
        configured_horizon = _nested_get(checkpoint, "cfg", "data", "action_horizon")
        if configured_horizon is not None and int(configured_horizon) != self.action_horizon:
            raise ValueError(
                f"Flow checkpoint horizon metadata is inconsistent: normalizer={self.action_horizon}, "
                f"cfg.data={configured_horizon}"
            )
        if expected_action_horizon is not None and expected_action_horizon != self.action_horizon:
            raise ValueError(
                f"Flow horizon mismatch: expected {expected_action_horizon}, "
                f"checkpoint has {self.action_horizon}"
            )
        if expected_action_dim is not None and expected_action_dim != self.action_dim:
            raise ValueError(
                f"Flow action dimension mismatch: expected {expected_action_dim}, "
                f"checkpoint has {self.action_dim}"
            )
        if int(ode_steps) <= 0:
            raise ValueError("ode_steps must be positive")
        self.ode_steps = int(ode_steps)

        checkpoint_image_size = _nested_get(checkpoint, "cfg", "data", "image_size")
        if image_size is None:
            image_size = 128 if checkpoint_image_size is None else int(checkpoint_image_size)
        if checkpoint_image_size is not None and int(image_size) != int(checkpoint_image_size):
            raise ValueError(
                f"Flow image size mismatch: requested {image_size}, "
                f"checkpoint has {checkpoint_image_size}"
            )
        self.image_size = int(image_size)

        model_cfg = copy.deepcopy(checkpoint["model_cfg"])
        image_encoder_cfg = (
            model_cfg.get("image_encoder")
            if isinstance(model_cfg, Mapping)
            else getattr(model_cfg, "image_encoder", None)
        )
        pretrained_path = (
            image_encoder_cfg.get("pretrained_path")
            if isinstance(image_encoder_cfg, Mapping)
            else getattr(image_encoder_cfg, "pretrained_path", None)
        )
        if pretrained_path is not None and not Path(str(pretrained_path)).expanduser().is_file():
            # The full flow checkpoint strictly restores every encoder parameter, so
            # external ImageNet initialization is unnecessary at inference time.
            if isinstance(image_encoder_cfg, Mapping):
                image_encoder_cfg["pretrained_path"] = None
            else:
                image_encoder_cfg.pretrained_path = None
        self.model = model_builder(
            model_cfg,
            proprio_dim=self.proprio_dim,
            action_dim=self.action_dim,
            camera_names=list(self.camera_names),
        )
        state = checkpoint.get("ema_model", checkpoint.get("model"))
        if not isinstance(state, Mapping):
            raise TypeError("Flow model state must be a mapping")
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval().requires_grad_(False)

        self.register_buffer("act_mean", act_mean.to(self.device), persistent=True)
        self.register_buffer("act_std", act_std.to(self.device), persistent=True)
        self.register_buffer("prop_mean", prop_mean.to(self.device), persistent=True)
        self.register_buffer("prop_std", prop_std.to(self.device), persistent=True)
        low = self._action_bound(action_low, "low")
        high = self._action_bound(action_high, "high")
        if torch.any(low >= high):
            raise ValueError("Every action low bound must be smaller than its high bound")
        self.register_buffer("action_low", low.to(self.device), persistent=True)
        self.register_buffer("action_high", high.to(self.device), persistent=True)
        self.register_buffer(
            "pixel_mean",
            torch.tensor((0.485, 0.456, 0.406), device=self.device).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor((0.229, 0.224, 0.225), device=self.device).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.eval()

    def _validate_normalizer(
        self, name: str, mean: Any, std: Any, rank: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean_tensor = torch.as_tensor(np.asarray(mean), dtype=torch.float32, device=self.device)
        std_tensor = torch.as_tensor(np.asarray(std), dtype=torch.float32, device=self.device)
        if mean_tensor.ndim != rank or std_tensor.shape != mean_tensor.shape:
            raise ValueError(
                f"Invalid {name} normalizer shapes: mean={tuple(mean_tensor.shape)}, "
                f"std={tuple(std_tensor.shape)}"
            )
        if not torch.isfinite(mean_tensor).all() or not torch.isfinite(std_tensor).all():
            raise ValueError(f"{name.capitalize()} normalizer contains non-finite values")
        if torch.any(std_tensor <= 0):
            raise ValueError(f"{name.capitalize()} standard deviation must be positive")
        return mean_tensor, std_tensor

    def _action_bound(self, value: Any, name: str) -> torch.Tensor:
        bound = torch.as_tensor(np.asarray(value), dtype=torch.float32, device=self.device)
        if bound.ndim == 0:
            bound = bound.repeat(self.action_dim)
        if tuple(bound.shape) != (self.action_dim,):
            raise ValueError(
                f"Action {name} bound must be scalar or [{self.action_dim}], got {tuple(bound.shape)}"
            )
        if not torch.isfinite(bound).all():
            raise ValueError(f"Action {name} bound contains non-finite values")
        return bound

    def train(self, mode: bool = True) -> "FlowPolicyAdapter":
        super().train(False)
        self.model.eval()
        return self

    def _require_cuda_tensor(self, tensor: torch.Tensor, name: str) -> None:
        if tensor.device.type != "cuda":
            raise ValueError(f"{name} must be on CUDA")
        if tensor.device != self.device:
            raise ValueError(f"{name} must be on {self.device}, got {tensor.device}")

    def preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        self._require_cuda_tensor(images, "images")
        if images.ndim != 5 or images.shape[1] != len(self.camera_names) or images.shape[2] != 3:
            raise ValueError(
                f"Expected images [B,{len(self.camera_names)},3,H,W], got {tuple(images.shape)}"
            )
        normalized = images
        if normalized.dtype == torch.uint8:
            normalized = normalized.to(torch.float32).div_(255.0)
        elif normalized.is_floating_point():
            normalized = normalized.to(torch.float32)
        else:
            raise TypeError(f"Expected uint8 or floating-point images, got {normalized.dtype}")
        batch_size, camera_count = normalized.shape[:2]
        flat = normalized.flatten(0, 1)
        height, width = flat.shape[-2:]
        crop_size = min(height, width)
        y0 = (height - crop_size) // 2
        x0 = (width - crop_size) // 2
        flat = flat[:, :, y0 : y0 + crop_size, x0 : x0 + crop_size]
        if crop_size != self.image_size:
            flat = F.interpolate(flat, (self.image_size, self.image_size), mode="nearest")
        normalized = flat.reshape(batch_size, camera_count, 3, self.image_size, self.image_size)
        return (normalized - self.pixel_mean) / self.pixel_std

    def normalize_proprio(self, proprio: torch.Tensor) -> torch.Tensor:
        self._require_cuda_tensor(proprio, "proprio")
        if proprio.ndim != 2 or proprio.shape[1] != self.proprio_dim:
            raise ValueError(f"Expected proprio [B,{self.proprio_dim}], got {tuple(proprio.shape)}")
        return (proprio.float() - self.prop_mean) / self.prop_std

    def normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        self._require_cuda_tensor(actions, "actions")
        if actions.shape[-2:] != (self.action_horizon, self.action_dim):
            raise ValueError(
                f"Expected actions [...,{self.action_horizon},{self.action_dim}], "
                f"got {tuple(actions.shape)}"
            )
        return (actions.float() - self.act_mean) / self.act_std

    def denormalize_actions(self, normalized_actions: torch.Tensor, *, clamp: bool = True) -> torch.Tensor:
        self._require_cuda_tensor(normalized_actions, "normalized_actions")
        if normalized_actions.shape[-2:] != (self.action_horizon, self.action_dim):
            raise ValueError(
                f"Expected actions [...,{self.action_horizon},{self.action_dim}], "
                f"got {tuple(normalized_actions.shape)}"
            )
        actions = normalized_actions.float() * self.act_std + self.act_mean
        if clamp:
            actions = torch.maximum(torch.minimum(actions, self.action_high), self.action_low)
        return actions

    def project_normalized_actions(self, normalized_actions: torch.Tensor) -> torch.Tensor:
        """Project normalized actions through the environment's executable bounds."""
        executable = self.denormalize_actions(normalized_actions, clamp=True)
        return self.normalize_actions(executable)

    @torch.no_grad()
    def encode_context(self, observation: FlowObservation) -> FlowContext:
        images = observation.images
        proprio = observation.proprio
        self._require_cuda_tensor(images, "images")
        self._require_cuda_tensor(proprio, "proprio")
        if not observation.images_preprocessed:
            images = self.preprocess_images(images)
        if not observation.proprio_normalized:
            proprio = self.normalize_proprio(proprio)
        language: Sequence[str] | str = observation.language
        if isinstance(language, str):
            language = [language] * int(proprio.shape[0])
        elif len(language) != int(proprio.shape[0]):
            raise ValueError("Language batch size must match proprio batch size")
        raw = self.model.encode_multimodal_context(images, proprio, language)
        required = ("task_scene_cond", "context_tokens", "context_padding_mask")
        missing = [key for key in required if key not in raw]
        if missing:
            raise KeyError(f"Flow context output is missing keys: {missing}")
        return FlowContext(*(raw[key] for key in required))

    @torch.no_grad()
    def decode_noise(
        self,
        context: FlowContext,
        noise: torch.Tensor,
        *,
        denormalize: bool = True,
        clamp: bool = True,
    ) -> torch.Tensor:
        self._require_cuda_tensor(noise, "noise")
        expected = (context.batch_size, self.action_horizon, self.action_dim)
        if tuple(noise.shape) != expected:
            raise ValueError(f"Expected noise {expected}, got {tuple(noise.shape)}")
        for name, tensor in (
            ("task_scene_cond", context.task_scene_cond),
            ("context_tokens", context.context_tokens),
            ("context_padding_mask", context.context_padding_mask),
        ):
            self._require_cuda_tensor(tensor, name)
            if tensor.shape[0] != context.batch_size:
                raise ValueError(f"{name} batch size does not match context")

        model_dtype = next(self.model.parameters()).dtype
        task_scene_cond = context.task_scene_cond.to(dtype=model_dtype)
        context_tokens = context.context_tokens.to(dtype=model_dtype)
        context_padding_mask = context.context_padding_mask.to(dtype=torch.bool)
        state = noise.to(dtype=model_dtype).transpose(1, 2)
        dt = 1.0 / float(self.ode_steps)
        for step in range(self.ode_steps):
            timesteps = torch.full(
                (context.batch_size,),
                float(step) / float(self.ode_steps),
                device=self.device,
                dtype=torch.float32,
            )
            velocity = self.model.flow_head(
                x_t=state,
                timesteps=timesteps,
                task_scene_cond=task_scene_cond,
                context_tokens=context_tokens,
                context_padding_mask=context_padding_mask,
            )
            if velocity.shape != state.shape:
                raise ValueError(
                    f"Flow head returned {tuple(velocity.shape)}, expected {tuple(state.shape)}"
                )
            state = state + dt * velocity
        normalized = state.transpose(1, 2)
        return self.denormalize_actions(normalized, clamp=clamp) if denormalize else normalized
