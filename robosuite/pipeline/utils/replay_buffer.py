from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..common.types import ReplayBatch, ReplayBufferConfig, Transition
from ..common.utils import assert_same_structure, clone_array_tree, nested_to_torch, stack_tree, to_numpy


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
