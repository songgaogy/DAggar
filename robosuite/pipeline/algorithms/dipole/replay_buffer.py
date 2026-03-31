from __future__ import annotations

import bisect
import math
import threading
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from robosuite.pipeline.common.types import ReplayBufferConfig, Transition
from robosuite.pipeline.common.utils import (
    assert_same_structure,
    clone_array_tree,
    to_numpy,
)

from .common import DipoleBatch, FlowAugmentationConfig


DIPOLE_INFO_KEY = "dipole"


def _center_crop_resize(image: np.ndarray, image_size: int) -> np.ndarray:
    height, width = image.shape[:2]
    crop_size = min(height, width)
    y0 = (height - crop_size) // 2
    x0 = (width - crop_size) // 2
    crop = image[y0 : y0 + crop_size, x0 : x0 + crop_size]
    if crop_size == image_size:
        return crop
    ys = np.linspace(0, crop_size - 1, image_size).astype(np.int32)
    xs = np.linspace(0, crop_size - 1, image_size).astype(np.int32)
    return crop[ys][:, xs]


def _random_shift(images: torch.Tensor, pad: int) -> torch.Tensor:
    if pad <= 0:
        return images
    batch_size, channels, height, width = images.shape
    padded = F.pad(images, (pad, pad, pad, pad), mode="replicate")
    padded_height = height + 2 * pad
    padded_width = width + 2 * pad

    eps_y = 1.0 / padded_height
    eps_x = 1.0 / padded_width
    base_y = torch.linspace(
        -1.0 + eps_y,
        1.0 - eps_y,
        padded_height,
        device=images.device,
        dtype=images.dtype,
    )[:height]
    base_x = torch.linspace(
        -1.0 + eps_x,
        1.0 - eps_x,
        padded_width,
        device=images.device,
        dtype=images.dtype,
    )[:width]
    grid_y, grid_x = torch.meshgrid(base_y, base_x, indexing="ij")
    base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(batch_size, -1, -1, -1)

    shift_x = torch.randint(0, 2 * pad + 1, (batch_size, 1, 1), device=images.device)
    shift_y = torch.randint(0, 2 * pad + 1, (batch_size, 1, 1), device=images.device)
    shift = torch.stack(
        [
            shift_x.to(dtype=images.dtype) * (2.0 / padded_width),
            shift_y.to(dtype=images.dtype) * (2.0 / padded_height),
        ],
        dim=-1,
    )
    grid = base_grid + shift
    return F.grid_sample(padded, grid, padding_mode="zeros", align_corners=False)


def _random_crop_resize(images: torch.Tensor, crop_scale: float) -> torch.Tensor:
    if crop_scale >= 0.999:
        return images
    batch_size, channels, height, width = images.shape
    crop_h = max(1, int(round(height * crop_scale)))
    crop_w = max(1, int(round(width * crop_scale)))
    y0 = torch.randint(0, height - crop_h + 1, (batch_size,), device=images.device)
    x0 = torch.randint(0, width - crop_w + 1, (batch_size,), device=images.device)

    center_x = ((x0.to(images.dtype) + 0.5 * crop_w) * 2.0 / width) - 1.0
    center_y = ((y0.to(images.dtype) + 0.5 * crop_h) * 2.0 / height) - 1.0
    scale_x = torch.full((batch_size,), float(crop_w) / float(width), device=images.device, dtype=images.dtype)
    scale_y = torch.full((batch_size,), float(crop_h) / float(height), device=images.device, dtype=images.dtype)

    theta = torch.zeros((batch_size, 2, 3), device=images.device, dtype=images.dtype)
    theta[:, 0, 0] = scale_x
    theta[:, 1, 1] = scale_y
    theta[:, 0, 2] = center_x
    theta[:, 1, 2] = center_y

    grid = F.affine_grid(theta, size=images.shape, align_corners=False)
    return F.grid_sample(images, grid, mode="bilinear", padding_mode="border", align_corners=False)


def _dipole_info(info: dict[str, Any] | None) -> dict[str, Any]:
    if info is None:
        return {}
    nested = info.get(DIPOLE_INFO_KEY, None)
    if isinstance(nested, dict):
        return nested
    return {}


def _uses_positive_demo_defaults(transition: Transition) -> bool:
    demo_source = str(transition.demo_source or "").strip().lower()
    if demo_source in {"offline_demo", "intervention"}:
        return True
    return bool(transition.is_intervention)


def apply_dipole_label_to_transition(
    transition: Transition,
    *,
    lambda_value: float,
    threshold: float,
    prediction: int,
    raw_step_score: float,
    source: str,
    metadata: dict[str, Any] | None = None,
    label_ready: bool = True,
) -> Transition:
    info = {} if transition.info is None else dict(transition.info)
    nested = dict(_dipole_info(info))
    nested.update(
        {
            "lambda": float(lambda_value),
            "threshold": float(threshold),
            "prediction": int(prediction),
            "raw_step_score": float(raw_step_score),
            "source": str(source),
            "label_ready": bool(label_ready),
            "force_positive": bool(nested.get("force_positive", False)),
        }
    )
    if metadata is not None:
        nested["metadata"] = dict(metadata)
    info[DIPOLE_INFO_KEY] = nested
    transition.info = info
    return transition


def get_transition_dipole_fields(transition: Transition) -> dict[str, Any]:
    info = {} if transition.info is None else dict(transition.info)
    nested = dict(_dipole_info(info))
    return {
        "lambda": float(nested.get("lambda", 0.0)) if math.isfinite(float(nested.get("lambda", 0.0))) else 0.0,
        "threshold": float(nested.get("threshold", float("nan"))),
        "prediction": int(nested.get("prediction", 0)),
        "raw_step_score": float(nested.get("raw_step_score", float("nan"))),
        "source": str(nested.get("source", "unknown")),
        "label_ready": bool(nested.get("label_ready", False)),
        "force_positive": bool(nested.get("force_positive", False)),
        "metadata": dict(nested.get("metadata", {})) if isinstance(nested.get("metadata"), dict) else {},
    }


def transition_is_ready_for_dipole(transition: Transition) -> bool:
    fields = get_transition_dipole_fields(transition)
    return bool(fields["force_positive"] or fields["label_ready"])


class DipoleReplayBuffer:
    def __init__(
        self,
        config: ReplayBufferConfig,
        *,
        name: str,
        camera_names: list[str],
        action_horizon: int,
        image_size: int,
        augmentation_config: FlowAugmentationConfig | None = None,
        force_positive_enabled: bool = True,
    ) -> None:
        self.config = config
        self.name = str(name)
        self.capacity = int(config.capacity)
        self.camera_names = [str(name) for name in camera_names]
        self.action_horizon = int(action_horizon)
        self.image_size = int(image_size)
        self.augmentation_config = augmentation_config or FlowAugmentationConfig()
        self.force_positive_enabled = bool(force_positive_enabled)
        self._storage: list[Transition] = []
        self._position = 0
        self._reference_obs: Any = None
        self._reference_action: np.ndarray | None = None
        self._lock = threading.RLock()
        self._valid_start_cache: list[int] | None = None
        self._episode_step_to_index: dict[tuple[int, int], int] = {}

    def __len__(self) -> int:
        with self._lock:
            return len(self._storage)

    def clear(self) -> None:
        with self._lock:
            self._storage = []
            self._position = 0
            self._reference_obs = None
            self._reference_action = None
            self._valid_start_cache = None
            self._episode_step_to_index = {}

    def add(self, transition: Transition) -> None:
        normalized = self._normalize_transition(transition)
        with self._lock:
            self._validate_transition(normalized)
            overwrite_index = self._position if len(self._storage) >= self.capacity else None
            if overwrite_index is not None:
                self._drop_episode_mapping_locked(self._storage[overwrite_index])
                self._storage[overwrite_index] = normalized
                self._rebuild_valid_start_cache_locked()
            else:
                self._storage.append(normalized)
                self._append_valid_start_locked()
            stored_index = self._position
            self._position = (self._position + 1) % self.capacity
            self._register_episode_mapping_locked(normalized, stored_index)

    def extend(self, transitions: list[Transition]) -> None:
        for transition in transitions:
            self.add(transition)

    def patch_dipole_label(
        self,
        *,
        episode_index: int,
        episode_step: int,
        lambda_value: float,
        threshold: float,
        prediction: int,
        raw_step_score: float,
        source: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        with self._lock:
            key = (int(episode_index), int(episode_step))
            storage_index = self._episode_step_to_index.get(key, None)
            if storage_index is None:
                return False
            apply_dipole_label_to_transition(
                self._storage[storage_index],
                lambda_value=lambda_value,
                threshold=threshold,
                prediction=prediction,
                raw_step_score=raw_step_score,
                source=source,
                metadata=metadata,
                label_ready=True,
            )
            self._update_valid_start_for_index_locked(storage_index)
            return True

    def num_valid_sequences(self) -> int:
        with self._lock:
            return len(self._get_valid_start_indices_locked())

    def sample(
        self,
        batch_size: int,
        *,
        action_mean: np.ndarray | None = None,
        action_std: np.ndarray | None = None,
        proprio_mean: np.ndarray | None = None,
        proprio_std: np.ndarray | None = None,
        device: torch.device | str | None = None,
        augment: bool = True,
        buffer_role: str = "online",
    ) -> DipoleBatch:
        with self._lock:
            valid_starts = self._get_valid_start_indices_locked()
            if len(valid_starts) == 0:
                raise ValueError(f"{self.name} does not contain any valid sequences.")
            indices = np.random.randint(0, len(valid_starts), size=int(batch_size))
            start_indices = [valid_starts[int(index)] for index in indices]
            transitions = [self._storage[start : start + self.action_horizon] for start in start_indices]

        image_batch = []
        proprio_batch = []
        action_batch = []
        lambda_batch = []
        threshold_batch = []
        force_positive_batch = []
        episode_ids = []
        episode_steps = []
        dipole_sources = []
        for sequence in transitions:
            first = sequence[0]
            obs = first.obs
            images = []
            for camera_name in self.camera_names:
                if camera_name not in obs:
                    raise KeyError(f"{self.name} observation is missing camera '{camera_name}'.")
                image = np.asarray(obs[camera_name], dtype=np.uint8)
                image = _center_crop_resize(image, self.image_size)
                images.append(np.transpose(image, (2, 0, 1)))
            image_batch.append(np.stack(images, axis=0))
            proprio_batch.append(np.asarray(obs["state"], dtype=np.float32))
            action_batch.append(np.stack([np.asarray(item.action, dtype=np.float32) for item in sequence], axis=0))
            info = first.info or {}
            episode_ids.append(int(info.get("episode_index", -1)))
            episode_steps.append(int(info.get("episode_step", -1)))
            dipole_fields = get_transition_dipole_fields(first)
            lambda_batch.append(dipole_fields["lambda"])
            threshold_batch.append(dipole_fields["threshold"])
            force_positive_batch.append(1.0 if dipole_fields["force_positive"] else 0.0)
            dipole_sources.append(dipole_fields["source"])

        image_tensor = torch.from_numpy(np.ascontiguousarray(np.stack(image_batch, axis=0)))
        proprio_tensor = torch.from_numpy(np.ascontiguousarray(np.stack(proprio_batch, axis=0)))
        action_tensor = torch.from_numpy(np.ascontiguousarray(np.stack(action_batch, axis=0)))
        lambda_tensor = torch.as_tensor(np.asarray(lambda_batch, dtype=np.float32).reshape(-1, 1))
        threshold_tensor = torch.as_tensor(np.asarray(threshold_batch, dtype=np.float32).reshape(-1, 1))
        force_positive_tensor = torch.as_tensor(np.asarray(force_positive_batch, dtype=np.float32).reshape(-1, 1))
        is_online_tensor = torch.full(
            (int(batch_size), 1),
            1.0 if str(buffer_role).lower() == "online" else 0.0,
            dtype=torch.float32,
        )

        image_tensor = self._preprocess_images(image_tensor, augment=augment)

        if proprio_mean is not None and proprio_std is not None:
            proprio_tensor = (
                proprio_tensor.float()
                - torch.as_tensor(proprio_mean, dtype=torch.float32).view(1, -1)
            ) / torch.as_tensor(proprio_std, dtype=torch.float32).view(1, -1)
        else:
            proprio_tensor = proprio_tensor.float()

        if action_mean is not None and action_std is not None:
            action_tensor = (
                action_tensor.float()
                - torch.as_tensor(action_mean, dtype=torch.float32).view(1, self.action_horizon, -1)
            ) / torch.as_tensor(action_std, dtype=torch.float32).view(1, self.action_horizon, -1)
        else:
            action_tensor = action_tensor.float()

        batch = DipoleBatch(
            image_obs=image_tensor,
            proprio=proprio_tensor,
            action_sequences=action_tensor,
            lambda_values=lambda_tensor,
            threshold_values=threshold_tensor,
            force_positive=force_positive_tensor,
            is_online=is_online_tensor,
            metadata={
                "start_indices": start_indices,
                "episode_ids": episode_ids,
                "episode_steps": episode_steps,
                "dipole_sources": dipole_sources,
                "buffer_role": [str(buffer_role)] * int(batch_size),
            },
        )
        if device is not None:
            batch = batch.to(device)
        return batch

    def state_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "name": self.name,
                "config": {
                    "capacity": self.capacity,
                    "batch_size": int(self.config.batch_size),
                },
                "camera_names": list(self.camera_names),
                "action_horizon": int(self.action_horizon),
                "image_size": int(self.image_size),
                "force_positive_enabled": bool(self.force_positive_enabled),
                "position": int(self._position),
                "storage": list(self._storage),
            }

    def snapshot_state_dict(self) -> dict[str, Any]:
        return self.state_dict()

    def snapshot_transition(self, transition: Transition) -> Transition:
        return self._normalize_transition(transition)

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        with self._lock:
            self.name = str(state_dict["name"])
            config = state_dict.get("config", {})
            self.config = ReplayBufferConfig(
                capacity=int(config.get("capacity", self.capacity)),
                batch_size=int(config.get("batch_size", self.config.batch_size)),
            )
            self.capacity = int(self.config.capacity)
            self.camera_names = [str(name) for name in state_dict.get("camera_names", self.camera_names)]
            self.action_horizon = int(state_dict.get("action_horizon", self.action_horizon))
            self.image_size = int(state_dict.get("image_size", self.image_size))
            self.force_positive_enabled = bool(state_dict.get("force_positive_enabled", self.force_positive_enabled))
            self._position = int(state_dict.get("position", 0))
            self._storage = list(state_dict.get("storage", []))
            if self._storage:
                self._reference_obs = clone_array_tree(self._storage[0].obs)
                self._reference_action = to_numpy(self._storage[0].action, dtype=np.float32).reshape(-1)
            else:
                self._reference_obs = None
                self._reference_action = None
            self._rebuild_episode_mapping_locked()
            self._rebuild_valid_start_cache_locked()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.tmp")
        torch.save(self.state_dict(), tmp_path)
        tmp_path.replace(path)

    def load(self, path: str | Path) -> None:
        state_dict = torch.load(Path(path), map_location="cpu", weights_only=False)
        self.load_state_dict(state_dict)

    def is_compatible(self, obs: Any, action: Any) -> bool:
        with self._lock:
            if self._reference_obs is None or self._reference_action is None:
                return True
            try:
                normalized_obs = self._normalize_observation(obs)
                assert_same_structure(self._reference_obs, normalized_obs, path=f"{self.name}.obs")
                action_array = to_numpy(action, dtype=np.float32).reshape(-1)
                return tuple(self._reference_action.shape) == tuple(action_array.shape)
            except Exception:
                return False

    def _normalize_transition(self, transition: Transition) -> Transition:
        if transition.reward is None:
            raise ValueError(f"{self.name} requires every transition to contain a reward.")
        if transition.next_obs is None:
            raise ValueError(f"{self.name} requires next_obs for every transition.")
        action = to_numpy(transition.action, dtype=np.float32).reshape(-1)
        normalized = Transition(
            obs=self._normalize_observation(transition.obs),
            action=action,
            reward=float(transition.reward),
            next_obs=self._normalize_observation(transition.next_obs),
            done=bool(transition.done),
            grasp_penalty=None if transition.grasp_penalty is None else float(transition.grasp_penalty),
            is_intervention=bool(transition.is_intervention),
            info=dict(transition.info) if transition.info is not None else None,
            reward_source=transition.reward_source,
            demo_source=transition.demo_source,
        )
        return self._initialize_dipole_info(normalized)

    def _initialize_dipole_info(self, transition: Transition) -> Transition:
        info = {} if transition.info is None else dict(transition.info)
        nested = dict(_dipole_info(info))
        positive_demo_defaults = _uses_positive_demo_defaults(transition)
        force_positive = bool(nested.get("force_positive", self.force_positive_enabled and positive_demo_defaults))
        label_ready = bool(nested.get("label_ready", force_positive or positive_demo_defaults))
        lambda_value = float(nested.get("lambda", 0.0 if label_ready else float("nan")))
        if not np.isfinite(lambda_value):
            lambda_value = 0.0
        threshold_value = float(nested.get("threshold", 0.0 if label_ready else float("nan")))
        if not np.isfinite(threshold_value) and label_ready:
            threshold_value = 0.0
        nested.update(
            {
                "lambda": float(lambda_value),
                "threshold": float(threshold_value),
                "prediction": int(nested.get("prediction", 0)),
                "raw_step_score": float(nested.get("raw_step_score", float("nan"))),
                "source": str(
                    nested.get(
                        "source",
                        (
                            "force_positive"
                            if force_positive
                            else ("ready_without_force_positive" if label_ready else "pending_discriminator")
                        ),
                    )
                ),
                "label_ready": bool(label_ready),
                "force_positive": bool(force_positive),
            }
        )
        metadata = nested.get("metadata", {})
        nested["metadata"] = dict(metadata) if isinstance(metadata, dict) else {}
        info[DIPOLE_INFO_KEY] = nested
        transition.info = info
        return transition

    def _normalize_observation(self, obs: Any) -> Any:
        normalized = clone_array_tree(obs)
        if not isinstance(normalized, Mapping):
            return normalized
        normalized = dict(normalized)
        if "state" in normalized:
            normalized["state"] = np.asarray(normalized["state"], dtype=np.float32).reshape(-1)
        for camera_name in self.camera_names:
            if camera_name not in normalized:
                continue
            image = np.asarray(normalized[camera_name], dtype=np.uint8)
            if image.ndim != 3:
                raise ValueError(
                    f"{self.name}.{camera_name} expected an HWC image, got shape {tuple(image.shape)}."
                )
            normalized[camera_name] = _center_crop_resize(image, self.image_size)
        return normalized

    def _validate_transition(self, transition: Transition) -> None:
        if self._reference_obs is None:
            self._reference_obs = clone_array_tree(transition.obs)
            self._reference_action = transition.action.copy()
            return
        assert_same_structure(self._reference_obs, transition.obs, path=f"{self.name}.obs")
        if tuple(self._reference_action.shape) != tuple(transition.action.shape):
            raise ValueError(
                f"{self.name}.action shape mismatch. Expected {tuple(self._reference_action.shape)}, "
                f"got {tuple(transition.action.shape)}."
            )

    def _append_valid_start_locked(self) -> None:
        if self._valid_start_cache is None:
            self._rebuild_valid_start_cache_locked()
            return
        start_index = len(self._storage) - self.action_horizon
        if start_index < 0:
            return
        if self._is_valid_start_locked(start_index):
            self._valid_start_cache.append(start_index)

    def _update_valid_start_for_index_locked(self, index: int) -> None:
        if self._valid_start_cache is None:
            return
        if index < 0 or index + self.action_horizon > len(self._storage):
            return
        pos = bisect.bisect_left(self._valid_start_cache, index)
        already_present = pos < len(self._valid_start_cache) and self._valid_start_cache[pos] == index
        is_valid = self._is_valid_start_locked(index)
        if is_valid and not already_present:
            self._valid_start_cache.insert(pos, index)
        elif (not is_valid) and already_present:
            self._valid_start_cache.pop(pos)

    def _rebuild_valid_start_cache_locked(self) -> None:
        valid_starts = []
        max_start = len(self._storage) - self.action_horizon
        for start in range(max_start + 1):
            if self._is_valid_start_locked(start):
                valid_starts.append(start)
        self._valid_start_cache = valid_starts

    def _get_valid_start_indices_locked(self) -> list[int]:
        if self._valid_start_cache is None:
            self._rebuild_valid_start_cache_locked()
        return list(self._valid_start_cache or [])

    def _is_valid_start_locked(self, start: int) -> bool:
        if start < 0 or (start + self.action_horizon) > len(self._storage):
            return False
        first = self._storage[start]
        if not transition_is_ready_for_dipole(first):
            return False
        first_info = first.info or {}
        first_episode = first_info.get("episode_index", None)
        first_step = first_info.get("episode_step", None)
        for offset in range(self.action_horizon):
            transition = self._storage[start + offset]
            info = transition.info or {}
            if offset < self.action_horizon - 1 and bool(transition.done):
                return False
            if first_episode is None or first_step is None:
                continue
            if info.get("episode_index", None) != first_episode:
                return False
            if int(info.get("episode_step", -1)) != int(first_step) + offset:
                return False
        return True

    def _drop_episode_mapping_locked(self, transition: Transition) -> None:
        info = transition.info or {}
        episode_index = info.get("episode_index", None)
        episode_step = info.get("episode_step", None)
        if episode_index is None or episode_step is None:
            return
        self._episode_step_to_index.pop((int(episode_index), int(episode_step)), None)

    def _register_episode_mapping_locked(self, transition: Transition, storage_index: int) -> None:
        info = transition.info or {}
        episode_index = info.get("episode_index", None)
        episode_step = info.get("episode_step", None)
        if episode_index is None or episode_step is None:
            return
        self._episode_step_to_index[(int(episode_index), int(episode_step))] = int(storage_index)

    def _rebuild_episode_mapping_locked(self) -> None:
        self._episode_step_to_index = {}
        for index, transition in enumerate(self._storage):
            self._register_episode_mapping_locked(transition, index)

    def _preprocess_images(self, image_tensor: torch.Tensor, *, augment: bool) -> torch.Tensor:
        image_tensor = image_tensor.to(dtype=torch.float32).div_(255.0)
        if augment:
            for view_idx, camera_name in enumerate(self.camera_names):
                if camera_name == "robot0_eye_in_hand":
                    image_tensor[:, view_idx] = _random_crop_resize(
                        image_tensor[:, view_idx],
                        crop_scale=float(self.augmentation_config.eye_in_hand_crop_scale),
                    )
                else:
                    image_tensor[:, view_idx] = _random_shift(
                        image_tensor[:, view_idx],
                        pad=int(self.augmentation_config.minimal_shift_pad),
                    )
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=image_tensor.dtype).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=image_tensor.dtype).view(1, 1, 3, 1, 1)
        return (image_tensor - mean) / std


__all__ = [
    "DIPOLE_INFO_KEY",
    "DipoleReplayBuffer",
    "apply_dipole_label_to_transition",
    "get_transition_dipole_fields",
    "transition_is_ready_for_dipole",
]
