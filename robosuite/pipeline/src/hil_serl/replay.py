from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from robosuite.pipeline.src.data.transitions import Transition
from robosuite.pipeline.utils.tensor import (
    assert_same_structure,
    clone_array_tree,
    nested_to_torch,
    require_cuda_device,
    stack_tree,
    to_numpy,
)


TensorObservation = torch.Tensor | dict[str, Any]


@dataclass
class ReplayBatch:
    obs: TensorObservation
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: TensorObservation
    dones: torch.Tensor
    grasp_penalty: torch.Tensor
    is_intervention: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    def to(self, device: torch.device | str) -> "ReplayBatch":
        device = require_cuda_device(device, name="replay_batch_device")
        return ReplayBatch(
            obs=_tree_to_device(self.obs, device),
            actions=self.actions.to(device),
            rewards=self.rewards.to(device),
            next_obs=_tree_to_device(self.next_obs, device),
            dones=self.dones.to(device),
            grasp_penalty=self.grasp_penalty.to(device),
            is_intervention=self.is_intervention.to(device),
            metadata=self.metadata,
        )

    @property
    def batch_size(self) -> int:
        return int(self.actions.shape[0])


def _tree_to_device(value: Any, device: torch.device | str) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _tree_to_device(item, device) for key, item in value.items()}
    return value


@dataclass
class ReplayBufferConfig:
    capacity: int = 200_000
    batch_size: int = 256


def concat_replay_batches(first: ReplayBatch, second: ReplayBatch) -> ReplayBatch:
    return ReplayBatch(
        obs=_concat_tree(first.obs, second.obs),
        actions=torch.cat([first.actions, second.actions], dim=0),
        rewards=torch.cat([first.rewards, second.rewards], dim=0),
        next_obs=_concat_tree(first.next_obs, second.next_obs),
        dones=torch.cat([first.dones, second.dones], dim=0),
        grasp_penalty=torch.cat([first.grasp_penalty, second.grasp_penalty], dim=0),
        is_intervention=torch.cat([first.is_intervention, second.is_intervention], dim=0),
        metadata={
            "infos": list(first.metadata.get("infos", [])) + list(second.metadata.get("infos", [])),
            "reward_source": list(first.metadata.get("reward_source", []))
            + list(second.metadata.get("reward_source", [])),
            "demo_source": list(first.metadata.get("demo_source", []))
            + list(second.metadata.get("demo_source", [])),
            "indices": list(first.metadata.get("indices", [])) + list(second.metadata.get("indices", [])),
        },
    )


def _concat_tree(first: Any, second: Any) -> Any:
    if isinstance(first, Mapping):
        return {key: _concat_tree(first[key], second[key]) for key in first.keys()}
    return torch.cat([first, second], dim=0)


class HILSERLReplayBuffer:
    def __init__(self, config: ReplayBufferConfig, name: str = "replay_buffer") -> None:
        self.config = config
        self.name = str(name)
        self.capacity = int(config.capacity)
        self._storage: list[Transition] = []
        self._position = 0
        self._reference_obs: Any = None
        self._reference_action: np.ndarray | None = None
        self._lock = threading.RLock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._storage)

    def clear(self) -> None:
        with self._lock:
            self._storage = []
            self._position = 0
            self._reference_obs = None
            self._reference_action = None

    def add(self, transition: Transition) -> None:
        normalized = self._normalize_transition(transition)
        with self._lock:
            self._validate_transition(normalized)
            if len(self._storage) < self.capacity:
                self._storage.append(normalized)
            else:
                self._storage[self._position] = normalized
            self._position = (self._position + 1) % self.capacity

    def extend(self, transitions: list[Transition]) -> None:
        for transition in transitions:
            self.add(transition)

    def sample(self, batch_size: int, device: torch.device | str | None = None) -> ReplayBatch:
        if device is None:
            raise ValueError("HIL-SERL replay sampling requires an explicit CUDA device.")
        device = require_cuda_device(device, name=f"{self.name}.sample_device")
        with self._lock:
            if len(self._storage) == 0:
                raise ValueError(f"{self.name} is empty.")
            indices = np.random.randint(0, len(self._storage), size=int(batch_size))
            batch = [self._storage[int(index)] for index in indices]
        obs = nested_to_torch(stack_tree([item.obs for item in batch]), device=device)
        next_obs = nested_to_torch(stack_tree([item.next_obs for item in batch]), device=device)
        actions = torch.as_tensor(np.stack([item.action for item in batch], axis=0), device=device, dtype=torch.float32)
        rewards = torch.as_tensor(
            np.asarray([item.reward for item in batch], dtype=np.float32).reshape(-1, 1),
            device=device,
        )
        dones = torch.as_tensor(
            np.asarray([item.done for item in batch], dtype=np.float32).reshape(-1, 1),
            device=device,
        )
        grasp_penalty = torch.as_tensor(
            np.asarray(
                [0.0 if item.grasp_penalty is None else item.grasp_penalty for item in batch],
                dtype=np.float32,
            ).reshape(-1, 1),
            device=device,
        )
        is_intervention = torch.as_tensor(
            np.asarray([item.is_intervention for item in batch], dtype=np.float32).reshape(-1, 1),
            device=device,
        )
        metadata = {
            "infos": [item.info for item in batch],
            "reward_source": [item.reward_source for item in batch],
            "demo_source": [item.demo_source for item in batch],
            "indices": indices.tolist(),
        }
        return ReplayBatch(
            obs=obs,
            actions=actions,
            rewards=rewards,
            next_obs=next_obs,
            dones=dones,
            grasp_penalty=grasp_penalty,
            is_intervention=is_intervention,
            metadata=metadata,
        )

    def state_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "name": self.name,
                "config": {
                    "capacity": self.capacity,
                    "batch_size": int(self.config.batch_size),
                },
                "position": self._position,
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
            self._position = int(state_dict.get("position", 0))
            self._storage = list(state_dict.get("storage", []))
            if self._storage:
                self._reference_obs = clone_array_tree(self._storage[0].obs)
                self._reference_action = to_numpy(self._storage[0].action, dtype=np.float32).reshape(-1)
            else:
                self._reference_obs = None
                self._reference_action = None

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
                assert_same_structure(self._reference_obs, obs, path=f"{self.name}.obs")
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
        return Transition(
            obs=clone_array_tree(transition.obs),
            action=action,
            reward=float(transition.reward),
            next_obs=clone_array_tree(transition.next_obs),
            done=bool(transition.done),
            grasp_penalty=None if transition.grasp_penalty is None else float(transition.grasp_penalty),
            is_intervention=bool(transition.is_intervention),
            info=dict(transition.info) if transition.info is not None else None,
            reward_source=transition.reward_source,
            demo_source=transition.demo_source,
            terminated=transition.terminated,
            truncated=bool(transition.truncated),
        )

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
