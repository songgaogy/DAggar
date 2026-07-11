from __future__ import annotations

from typing import Any

import numpy as np
import torch

from robosuite.pipeline.algorithms.flow_dagger.replay_buffer import (
    FlowDaggerReplayBuffer,
    _center_crop_resize,
    _random_crop_resize,
    _random_shift,
)

from .common import DipoleBatch, FlowAugmentationConfig


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class DipoleOfflineStaticCache:
    """Static tensor cache for offline DIPOLE policy training.

    The source replay buffer must not be mutated after this cache is built.
    """

    def __init__(self, buffer: "DipoleReplayBuffer", *, pin_memory: bool = True) -> None:
        self.name = f"{buffer.name}_static_cache"
        self.camera_names = list(buffer.camera_names)
        self.action_horizon = int(buffer.action_horizon)
        self.image_size = int(buffer.image_size)
        self.augmentation_config = buffer.augmentation_config

        with buffer._lock:  # noqa: SLF001 - static offline cache reads buffer internals.
            valid_starts = list(buffer._get_valid_start_indices_locked())  # noqa: SLF001
            if len(valid_starts) == 0:
                raise ValueError(f"{buffer.name} does not contain any valid sequences.")

            image_batch = []
            proprio_batch = []
            action_batch = []
            is_intervention = []
            episode_ids = []
            episode_steps = []
            routes = []
            for start in valid_starts:
                sequence = buffer._storage[int(start) : int(start) + self.action_horizon]  # noqa: SLF001
                first = sequence[0]
                obs = first.obs
                images = []
                for camera_name in self.camera_names:
                    if camera_name not in obs:
                        raise KeyError(f"{buffer.name} observation is missing camera '{camera_name}'.")
                    image = np.asarray(obs[camera_name], dtype=np.uint8)
                    if image.shape[:2] != (self.image_size, self.image_size):
                        image = _center_crop_resize(image, self.image_size)
                    images.append(np.transpose(image, (2, 0, 1)))
                image_batch.append(np.stack(images, axis=0))
                proprio_batch.append(np.asarray(obs["state"], dtype=np.float32))
                action_batch.append(
                    np.stack([np.asarray(item.action, dtype=np.float32) for item in sequence], axis=0)
                )
                info = first.info or {}
                is_intervention.append(bool(first.is_intervention))
                episode_ids.append(int(info.get("episode_index", -1)))
                episode_steps.append(int(info.get("episode_step", -1)))
                routes.append(str(info.get("route", "advantage")))

        self.routes = routes
        self.images = torch.from_numpy(np.ascontiguousarray(np.stack(image_batch, axis=0)))
        self.proprio_raw = torch.from_numpy(np.ascontiguousarray(np.stack(proprio_batch, axis=0))).float()
        self.actions_raw = torch.from_numpy(np.ascontiguousarray(np.stack(action_batch, axis=0))).float()
        self.is_intervention = torch.tensor(is_intervention, dtype=torch.bool)
        self.start_indices = torch.tensor(valid_starts, dtype=torch.long)
        self.episode_ids = torch.tensor(episode_ids, dtype=torch.long)
        self.episode_steps = torch.tensor(episode_steps, dtype=torch.long)

        self.pin_memory_requested = bool(pin_memory)
        self.pin_memory = False
        if self.pin_memory_requested:
            try:
                self.images = self.images.pin_memory()
                self.proprio_raw = self.proprio_raw.pin_memory()
                self.actions_raw = self.actions_raw.pin_memory()
                self.is_intervention = self.is_intervention.pin_memory()
                self.start_indices = self.start_indices.pin_memory()
                self.episode_ids = self.episode_ids.pin_memory()
                self.episode_steps = self.episode_steps.pin_memory()
                self.pin_memory = True
            except RuntimeError:
                self.pin_memory = False

    def __len__(self) -> int:
        return int(self.start_indices.numel())

    @property
    def estimated_bytes(self) -> int:
        tensors = (
            self.images,
            self.proprio_raw,
            self.actions_raw,
            self.is_intervention,
            self.start_indices,
            self.episode_ids,
            self.episode_steps,
        )
        return int(sum(t.numel() * t.element_size() for t in tensors))

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
    ) -> DipoleBatch:
        if len(self) == 0:
            raise ValueError(f"{self.name} does not contain any valid sequences.")
        row_idx = torch.as_tensor(
            np.random.randint(0, len(self), size=int(batch_size)),
            dtype=torch.long,
        )
        target_device = torch.device("cpu" if device is None else device)
        non_blocking = bool(self.pin_memory and target_device.type == "cuda")

        image_tensor = self.images.index_select(0, row_idx).to(target_device, non_blocking=non_blocking)
        proprio_tensor_raw = self.proprio_raw.index_select(0, row_idx).to(
            target_device, non_blocking=non_blocking,
        )
        action_tensor_raw = self.actions_raw.index_select(0, row_idx).to(
            target_device, non_blocking=non_blocking,
        )
        is_intervention_tensor = self.is_intervention.index_select(0, row_idx).to(
            target_device, non_blocking=non_blocking,
        )

        image_tensor_unit, image_tensor_norm = self._preprocess_images_dual(image_tensor, augment=augment)

        if proprio_mean is not None and proprio_std is not None:
            proprio_tensor_norm = (
                proprio_tensor_raw
                - torch.as_tensor(proprio_mean, dtype=torch.float32, device=target_device).view(1, -1)
            ) / torch.as_tensor(proprio_std, dtype=torch.float32, device=target_device).view(1, -1)
        else:
            proprio_tensor_norm = proprio_tensor_raw

        if action_mean is not None and action_std is not None:
            action_tensor_norm = (
                action_tensor_raw
                - torch.as_tensor(action_mean, dtype=torch.float32, device=target_device).view(
                    1, self.action_horizon, -1,
                )
            ) / torch.as_tensor(action_std, dtype=torch.float32, device=target_device).view(
                1, self.action_horizon, -1,
            )
        else:
            action_tensor_norm = action_tensor_raw

        start_indices = self.start_indices.index_select(0, row_idx).tolist()
        episode_ids = self.episode_ids.index_select(0, row_idx).tolist()
        episode_steps = self.episode_steps.index_select(0, row_idx).tolist()
        row_list = row_idx.tolist()
        return DipoleBatch(
            image_obs=image_tensor_norm,
            image_obs_raw=image_tensor_unit,
            proprio=proprio_tensor_norm,
            proprio_raw=proprio_tensor_raw,
            action_sequences=action_tensor_norm,
            action_sequences_raw=action_tensor_raw,
            is_intervention=is_intervention_tensor,
            metadata={
                "start_indices": [int(x) for x in start_indices],
                "episode_ids": [int(x) for x in episode_ids],
                "episode_steps": [int(x) for x in episode_steps],
                "route": [self.routes[int(i)] for i in row_list],
            },
        )

    def _preprocess_images_dual(
        self,
        images: torch.Tensor,
        *,
        augment: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return DipoleReplayBuffer._preprocess_images_dual(self, images, augment=augment)


class DipoleReplayBuffer(FlowDaggerReplayBuffer):
    """FlowDagger buffer that additionally exposes per-sample is_intervention and
    raw (un-ImageNet-normalized) images / proprio / actions for the dynamics encoder."""

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
    ) -> DipoleBatch:
        with self._lock:
            valid_starts = self._get_valid_start_indices_locked()
            if len(valid_starts) == 0:
                raise ValueError(f"{self.name} does not contain any valid sequences.")
            indices = np.random.randint(0, len(valid_starts), size=int(batch_size))
            start_indices = [valid_starts[int(i)] for i in indices]
            transitions = [self._storage[start : start + self.action_horizon] for start in start_indices]

        image_batch = []
        proprio_batch = []
        action_batch = []
        episode_ids = []
        episode_steps = []
        is_intervention_batch = []
        routes = []
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
            action_batch.append(
                np.stack([np.asarray(item.action, dtype=np.float32) for item in sequence], axis=0)
            )
            info = first.info or {}
            episode_ids.append(int(info.get("episode_index", -1)))
            episode_steps.append(int(info.get("episode_step", -1)))
            is_intervention_batch.append(bool(first.is_intervention))
            routes.append(str(info.get("route", "advantage")))

        image_tensor = torch.from_numpy(np.ascontiguousarray(np.stack(image_batch, axis=0)))
        proprio_tensor_raw = torch.from_numpy(np.ascontiguousarray(np.stack(proprio_batch, axis=0))).float()
        action_tensor_raw = torch.from_numpy(np.ascontiguousarray(np.stack(action_batch, axis=0))).float()
        is_intervention_tensor = torch.tensor(is_intervention_batch, dtype=torch.bool)

        image_tensor_unit, image_tensor_norm = self._preprocess_images_dual(image_tensor, augment=augment)

        if proprio_mean is not None and proprio_std is not None:
            proprio_tensor_norm = (
                proprio_tensor_raw
                - torch.as_tensor(proprio_mean, dtype=torch.float32).view(1, -1)
            ) / torch.as_tensor(proprio_std, dtype=torch.float32).view(1, -1)
        else:
            proprio_tensor_norm = proprio_tensor_raw

        if action_mean is not None and action_std is not None:
            action_tensor_norm = (
                action_tensor_raw
                - torch.as_tensor(action_mean, dtype=torch.float32).view(1, self.action_horizon, -1)
            ) / torch.as_tensor(action_std, dtype=torch.float32).view(1, self.action_horizon, -1)
        else:
            action_tensor_norm = action_tensor_raw

        batch = DipoleBatch(
            image_obs=image_tensor_norm,
            image_obs_raw=image_tensor_unit,
            proprio=proprio_tensor_norm,
            proprio_raw=proprio_tensor_raw,
            action_sequences=action_tensor_norm,
            action_sequences_raw=action_tensor_raw,
            is_intervention=is_intervention_tensor,
            metadata={
                "start_indices": start_indices,
                "episode_ids": episode_ids,
                "episode_steps": episode_steps,
                "route": routes,
            },
        )
        if device is not None:
            batch = batch.to(device)
        return batch

    def _preprocess_images_dual(
        self,
        images: torch.Tensor,
        *,
        augment: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (raw_unit, imagenet_normalized) image tensors.

        - raw_unit: [0, 1] floats after the same augmentation pipeline (consumed by DynEncoder)
        - imagenet_normalized: raw_unit with ImageNet mean/std subtracted (consumed by policy)
        """
        images = images.to(dtype=torch.float32).div_(255.0)
        if augment:
            for view_idx, camera_name in enumerate(self.camera_names):
                if camera_name == "robot0_eye_in_hand":
                    images[:, view_idx] = _random_crop_resize(
                        images[:, view_idx],
                        crop_scale=float(self.augmentation_config.eye_in_hand_crop_scale),
                    )
                else:
                    images[:, view_idx] = _random_shift(
                        images[:, view_idx],
                        pad=int(self.augmentation_config.minimal_shift_pad),
                    )
        raw_unit = images
        mean = torch.tensor(_IMAGENET_MEAN, dtype=images.dtype, device=images.device).view(1, 1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, dtype=images.dtype, device=images.device).view(1, 1, 3, 1, 1)
        imagenet_norm = (raw_unit - mean) / std
        return raw_unit, imagenet_norm

    def build_static_cache(self, *, pin_memory: bool = True) -> DipoleOfflineStaticCache:
        return DipoleOfflineStaticCache(self, pin_memory=pin_memory)


__all__ = ["DipoleReplayBuffer", "DipoleOfflineStaticCache"]
