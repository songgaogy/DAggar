from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from robosuite.pipeline.src.data.transitions import (
    ReplayBufferConfig,
    Transition,
    assert_same_structure,
    clone_array_tree,
    to_numpy,
)

from .batches import AWRActorBatch, AWRStepBatch
from .config import FlowAugmentationConfig, require_cuda_device


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
    batch_size, _, height, width = images.shape
    padded = F.pad(images, (pad, pad, pad, pad), mode="replicate")
    padded_height, padded_width = height + 2 * pad, width + 2 * pad
    base_y = torch.linspace(
        -1.0 + 1.0 / padded_height,
        1.0 - 1.0 / padded_height,
        padded_height,
        device=images.device,
        dtype=images.dtype,
    )[:height]
    base_x = torch.linspace(
        -1.0 + 1.0 / padded_width,
        1.0 - 1.0 / padded_width,
        padded_width,
        device=images.device,
        dtype=images.dtype,
    )[:width]
    grid_y, grid_x = torch.meshgrid(base_y, base_x, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(
        batch_size, -1, -1, -1
    )
    shift_x = torch.randint(
        0, 2 * pad + 1, (batch_size, 1, 1), device=images.device
    )
    shift_y = torch.randint(
        0, 2 * pad + 1, (batch_size, 1, 1), device=images.device
    )
    shift = torch.stack(
        [
            shift_x.to(images.dtype) * (2.0 / padded_width),
            shift_y.to(images.dtype) * (2.0 / padded_height),
        ],
        dim=-1,
    )
    return F.grid_sample(
        padded, grid + shift, padding_mode="zeros", align_corners=False
    )


def _random_crop_resize(images: torch.Tensor, scale: float) -> torch.Tensor:
    if scale >= 0.999:
        return images
    batch_size, _, height, width = images.shape
    crop_h = max(1, round(height * scale))
    crop_w = max(1, round(width * scale))
    y0 = torch.randint(0, height - crop_h + 1, (batch_size,), device=images.device)
    x0 = torch.randint(0, width - crop_w + 1, (batch_size,), device=images.device)
    theta = torch.zeros(
        (batch_size, 2, 3), device=images.device, dtype=images.dtype
    )
    theta[:, 0, 0] = float(crop_w) / width
    theta[:, 1, 1] = float(crop_h) / height
    theta[:, 0, 2] = ((x0.to(images.dtype) + 0.5 * crop_w) * 2.0 / width) - 1.0
    theta[:, 1, 2] = ((y0.to(images.dtype) + 0.5 * crop_h) * 2.0 / height) - 1.0
    grid = F.affine_grid(theta, size=images.shape, align_corners=False)
    return F.grid_sample(
        images, grid, mode="bilinear", padding_mode="border", align_corners=False
    )


class AWRReplayBuffer:
    def __init__(
        self,
        config: ReplayBufferConfig,
        *,
        name: str,
        camera_names: list[str],
        action_horizon: int,
        image_size: int,
        augmentation_config: FlowAugmentationConfig | None = None,
    ) -> None:
        self.config = config
        self.name = str(name)
        self.capacity = int(config.capacity)
        self.camera_names = [str(name) for name in camera_names]
        self.action_horizon = int(action_horizon)
        self.image_size = int(image_size)
        self.augmentation_config = augmentation_config or FlowAugmentationConfig()
        self._storage: list[Transition] = []
        self._position = 0
        self._reference_obs: Any = None
        self._reference_action: np.ndarray | None = None
        self._lock = threading.RLock()
        self._valid_start_cache: list[int] | None = None

    def __len__(self) -> int:
        with self._lock:
            return len(self._storage)

    def clear(self) -> None:
        with self._lock:
            self._storage.clear()
            self._position = 0
            self._reference_obs = None
            self._reference_action = None
            self._valid_start_cache = None

    def add(self, transition: Transition) -> None:
        normalized = self._normalize_transition(transition)
        with self._lock:
            self._validate_transition(normalized)
            if len(self._storage) < self.capacity:
                self._storage.append(normalized)
            else:
                self._storage[self._position] = normalized
            self._position = (self._position + 1) % self.capacity
            self._valid_start_cache = None

    def extend(self, transitions: list[Transition]) -> None:
        for transition in transitions:
            self.add(transition)

    def num_valid_sequences(self) -> int:
        with self._lock:
            return len(self._valid_starts())

    def num_ready_steps(self) -> int:
        return self.num_valid_sequences()

    def sample_actor_batch(
        self,
        batch_size: int,
        *,
        action_mean: np.ndarray | None = None,
        action_std: np.ndarray | None = None,
        proprio_mean: np.ndarray | None = None,
        proprio_std: np.ndarray | None = None,
        device: torch.device | str,
        augment: bool = True,
        buffer_role: str = "online",
    ) -> AWRActorBatch:
        target = require_cuda_device(str(device), name=f"{self.name} sample device")
        starts, sequences = self._sample_sequences(batch_size)
        images, proprio, actions, first_actions = [], [], [], []
        episode_ids, episode_steps = [], []
        for sequence in sequences:
            first = sequence[0]
            images.append(self._observation_images(first.obs))
            proprio.append(np.asarray(first.obs["state"], dtype=np.float32))
            actions.append(
                np.stack(
                    [np.asarray(item.action, dtype=np.float32) for item in sequence]
                )
            )
            first_actions.append(np.asarray(first.action, dtype=np.float32))
            info = first.info or {}
            episode_ids.append(int(info.get("episode_index", -1)))
            episode_steps.append(int(info.get("episode_step", -1)))

        image_tensor = torch.from_numpy(np.ascontiguousarray(np.stack(images))).to(target)
        proprio_tensor = torch.from_numpy(np.ascontiguousarray(np.stack(proprio))).to(target)
        action_tensor = torch.from_numpy(np.ascontiguousarray(np.stack(actions))).to(target)
        raw_action_tensor = action_tensor.clone()
        first_action_tensor = torch.from_numpy(
            np.ascontiguousarray(np.stack(first_actions))
        ).to(target)
        image_tensor = self._preprocess_images(image_tensor, augment=augment)
        proprio_tensor = self._normalize(
            proprio_tensor, proprio_mean, proprio_std, target
        )
        if action_mean is not None and action_std is not None:
            mean = torch.as_tensor(action_mean, dtype=torch.float32, device=target)
            std = torch.as_tensor(action_std, dtype=torch.float32, device=target)
            action_tensor = (action_tensor.float() - mean) / std
        else:
            action_tensor = action_tensor.float()
        return AWRActorBatch(
            image_obs=image_tensor,
            proprio=proprio_tensor,
            action_sequences=action_tensor,
            raw_action_sequences=raw_action_tensor.float(),
            first_actions=first_action_tensor.float(),
            is_online=torch.full(
                (int(batch_size), 1),
                1.0 if buffer_role == "online" else 0.0,
                device=target,
            ),
            metadata={
                "start_indices": starts,
                "episode_ids": episode_ids,
                "episode_steps": episode_steps,
                "buffer_role": [buffer_role] * int(batch_size),
            },
        )

    def sample_step_batch(
        self,
        batch_size: int,
        *,
        discount: float,
        proprio_mean: np.ndarray | None = None,
        proprio_std: np.ndarray | None = None,
        device: torch.device | str,
        augment: bool = True,
        buffer_role: str = "online",
    ) -> AWRStepBatch:
        target = require_cuda_device(str(device), name=f"{self.name} sample device")
        starts, sequences = self._sample_sequences(batch_size)
        images, proprio, actions, rewards = [], [], [], []
        next_images, next_proprio, dones = [], [], []
        powers = np.power(
            float(discount), np.arange(self.action_horizon, dtype=np.float32)
        )
        for sequence in sequences:
            first, last = sequence[0], sequence[-1]
            images.append(self._observation_images(first.obs))
            next_images.append(self._observation_images(last.next_obs))
            proprio.append(np.asarray(first.obs["state"], dtype=np.float32))
            next_proprio.append(
                np.asarray(last.next_obs["state"], dtype=np.float32)
            )
            actions.append(
                np.stack(
                    [np.asarray(item.action, dtype=np.float32) for item in sequence]
                )
            )
            rewards.append(
                float(
                    np.sum(
                        np.asarray([item.reward for item in sequence], dtype=np.float32)
                        * powers
                    )
                )
            )
            dones.append(float(any(item.done for item in sequence)))

        image_tensor = torch.from_numpy(np.ascontiguousarray(np.stack(images))).to(target)
        next_image_tensor = torch.from_numpy(
            np.ascontiguousarray(np.stack(next_images))
        ).to(target)
        proprio_tensor = torch.from_numpy(
            np.ascontiguousarray(np.stack(proprio))
        ).to(target)
        next_proprio_tensor = torch.from_numpy(
            np.ascontiguousarray(np.stack(next_proprio))
        ).to(target)
        return AWRStepBatch(
            image_obs=self._preprocess_images(image_tensor, augment=augment),
            proprio=self._normalize(
                proprio_tensor, proprio_mean, proprio_std, target
            ),
            actions=torch.from_numpy(np.ascontiguousarray(np.stack(actions)))
            .to(target)
            .float(),
            rewards=torch.as_tensor(
                rewards, dtype=torch.float32, device=target
            ).view(-1, 1),
            next_image_obs=self._preprocess_images(
                next_image_tensor, augment=augment
            ),
            next_proprio=self._normalize(
                next_proprio_tensor, proprio_mean, proprio_std, target
            ),
            dones=torch.as_tensor(
                dones, dtype=torch.float32, device=target
            ).view(-1, 1),
            is_online=torch.full(
                (int(batch_size), 1),
                1.0 if buffer_role == "online" else 0.0,
                device=target,
            ),
            metadata={
                "start_indices": starts,
                "buffer_role": [buffer_role] * int(batch_size),
                "discount": float(discount),
                "action_horizon": self.action_horizon,
            },
        )

    def state_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "version": 1,
                "name": self.name,
                "capacity": self.capacity,
                "batch_size": int(self.config.batch_size),
                "camera_names": self.camera_names,
                "action_horizon": self.action_horizon,
                "image_size": self.image_size,
                "position": self._position,
                "storage": list(self._storage),
            }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state.get("version", -1)) != 1:
            raise ValueError("Unsupported AWR replay buffer format.")
        with self._lock:
            self.name = str(state["name"])
            self.capacity = int(state["capacity"])
            self.config = ReplayBufferConfig(
                capacity=self.capacity, batch_size=int(state["batch_size"])
            )
            self.camera_names = [str(name) for name in state["camera_names"]]
            self.action_horizon = int(state["action_horizon"])
            self.image_size = int(state["image_size"])
            self._position = int(state["position"])
            self._storage = list(state["storage"])
            self._valid_start_cache = None
            if self._storage:
                self._reference_obs = clone_array_tree(self._storage[0].obs)
                self._reference_action = to_numpy(
                    self._storage[0].action, np.float32
                ).reshape(-1)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        torch.save(self.state_dict(), temporary)
        temporary.replace(target)

    def load(self, path: str | Path) -> None:
        self.load_state_dict(torch.load(Path(path), weights_only=False))

    def is_compatible(self, obs: Any, action: Any) -> bool:
        with self._lock:
            if self._reference_obs is None:
                return True
            try:
                normalized = self._normalize_observation(obs)
                assert_same_structure(
                    self._reference_obs, normalized, f"{self.name}.obs"
                )
                return self._reference_action.shape == to_numpy(
                    action, np.float32
                ).reshape(-1).shape
            except Exception:
                return False

    def _sample_sequences(
        self, batch_size: int
    ) -> tuple[list[int], list[list[Transition]]]:
        with self._lock:
            starts = self._valid_starts()
            if not starts:
                raise ValueError(
                    f"{self.name} does not contain a complete action chunk."
                )
            sampled = np.random.randint(0, len(starts), size=int(batch_size))
            selected = [starts[int(index)] for index in sampled]
            return selected, [
                self._storage[start : start + self.action_horizon]
                for start in selected
            ]

    def _valid_starts(self) -> list[int]:
        if self._valid_start_cache is None:
            self._valid_start_cache = [
                start
                for start in range(len(self._storage) - self.action_horizon + 1)
                if self._is_valid_start(start)
            ]
        return list(self._valid_start_cache)

    def _is_valid_start(self, start: int) -> bool:
        sequence = self._storage[start : start + self.action_horizon]
        if len(sequence) != self.action_horizon:
            return False
        first_info = sequence[0].info or {}
        episode = first_info.get("episode_index")
        step = first_info.get("episode_step")
        namespace = first_info.get("episode_namespace")
        for offset, transition in enumerate(sequence):
            if offset < self.action_horizon - 1 and transition.done:
                return False
            info = transition.info or {}
            if episode is not None and info.get("episode_index") != episode:
                return False
            if namespace is not None and info.get("episode_namespace") != namespace:
                return False
            if step is not None and int(info.get("episode_step", -1)) != int(step) + offset:
                return False
        return True

    def _normalize_transition(self, transition: Transition) -> Transition:
        if transition.reward is None:
            raise ValueError(f"{self.name} requires a reward.")
        if transition.next_obs is None:
            raise ValueError(f"{self.name} requires next_obs.")
        return Transition(
            obs=self._normalize_observation(transition.obs),
            action=to_numpy(transition.action, np.float32).reshape(-1),
            reward=float(transition.reward),
            next_obs=self._normalize_observation(transition.next_obs),
            done=bool(transition.done),
            grasp_penalty=None
            if transition.grasp_penalty is None
            else float(transition.grasp_penalty),
            is_intervention=bool(transition.is_intervention),
            info=None if transition.info is None else dict(transition.info),
            reward_source=transition.reward_source,
            demo_source=transition.demo_source,
        )

    def _normalize_observation(self, obs: Any) -> Any:
        normalized = clone_array_tree(obs)
        if not isinstance(normalized, Mapping):
            return normalized
        normalized = dict(normalized)
        if "state" in normalized:
            normalized["state"] = np.asarray(
                normalized["state"], dtype=np.float32
            ).reshape(-1)
        for camera_name in self.camera_names:
            if camera_name in normalized:
                image = np.asarray(normalized[camera_name], dtype=np.uint8)
                if image.ndim != 3:
                    raise ValueError(
                        f"{self.name}.{camera_name} expected HWC image, got {image.shape}."
                    )
                normalized[camera_name] = _center_crop_resize(image, self.image_size)
        return normalized

    def _validate_transition(self, transition: Transition) -> None:
        if self._reference_obs is None:
            self._reference_obs = clone_array_tree(transition.obs)
            self._reference_action = np.asarray(transition.action).copy()
            return
        assert_same_structure(
            self._reference_obs, transition.obs, f"{self.name}.obs"
        )
        if self._reference_action.shape != np.asarray(transition.action).shape:
            raise ValueError(f"{self.name}.action shape mismatch.")

    def _observation_images(self, obs: Mapping[str, Any]) -> np.ndarray:
        images = []
        for camera_name in self.camera_names:
            if camera_name not in obs:
                raise KeyError(
                    f"{self.name} observation is missing camera {camera_name!r}."
                )
            image = _center_crop_resize(
                np.asarray(obs[camera_name], dtype=np.uint8), self.image_size
            )
            images.append(np.transpose(image, (2, 0, 1)))
        return np.stack(images)

    @staticmethod
    def _normalize(
        tensor: torch.Tensor,
        mean: np.ndarray | None,
        std: np.ndarray | None,
        device: str,
    ) -> torch.Tensor:
        tensor = tensor.float()
        if mean is None or std is None:
            return tensor
        return (
            tensor - torch.as_tensor(mean, dtype=torch.float32, device=device)
        ) / torch.as_tensor(std, dtype=torch.float32, device=device)

    def _preprocess_images(
        self, image_tensor: torch.Tensor, *, augment: bool
    ) -> torch.Tensor:
        image_tensor = image_tensor.float().div_(255.0)
        if augment:
            for index, camera_name in enumerate(self.camera_names):
                if camera_name == "robot0_eye_in_hand":
                    image_tensor[:, index] = _random_crop_resize(
                        image_tensor[:, index],
                        self.augmentation_config.eye_in_hand_crop_scale,
                    )
                else:
                    image_tensor[:, index] = _random_shift(
                        image_tensor[:, index],
                        self.augmentation_config.minimal_shift_pad,
                    )
        mean = torch.tensor(
            [0.485, 0.456, 0.406],
            dtype=image_tensor.dtype,
            device=image_tensor.device,
        ).view(1, 1, 3, 1, 1)
        std = torch.tensor(
            [0.229, 0.224, 0.225],
            dtype=image_tensor.dtype,
            device=image_tensor.device,
        ).view(1, 1, 3, 1, 1)
        return (image_tensor - mean) / std


__all__ = ["AWRReplayBuffer"]
