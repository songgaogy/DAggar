from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .config import require_cuda_device


@dataclass
class AWRActorBatch:
    image_obs: torch.Tensor
    proprio: torch.Tensor
    action_sequences: torch.Tensor
    raw_action_sequences: torch.Tensor
    first_actions: torch.Tensor
    is_online: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    def to(self, device: torch.device | str) -> "AWRActorBatch":
        target = require_cuda_device(str(device), name="actor batch device")
        return AWRActorBatch(
            image_obs=self.image_obs.to(target),
            proprio=self.proprio.to(target),
            action_sequences=self.action_sequences.to(target),
            raw_action_sequences=self.raw_action_sequences.to(target),
            first_actions=self.first_actions.to(target),
            is_online=self.is_online.to(target),
            metadata=self.metadata,
        )

    @property
    def batch_size(self) -> int:
        return int(self.action_sequences.shape[0])

    @classmethod
    def concat(cls, batches: list["AWRActorBatch"]) -> "AWRActorBatch":
        batches = [batch for batch in batches if batch.batch_size]
        if not batches:
            raise ValueError("At least one non-empty actor batch is required.")
        if len(batches) == 1:
            return batches[0]
        return cls(
            image_obs=torch.cat([batch.image_obs for batch in batches]),
            proprio=torch.cat([batch.proprio for batch in batches]),
            action_sequences=torch.cat([batch.action_sequences for batch in batches]),
            raw_action_sequences=torch.cat([batch.raw_action_sequences for batch in batches]),
            first_actions=torch.cat([batch.first_actions for batch in batches]),
            is_online=torch.cat([batch.is_online for batch in batches]),
            metadata=_concat_metadata(batches),
        )


@dataclass
class AWRStepBatch:
    image_obs: torch.Tensor
    proprio: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_image_obs: torch.Tensor
    next_proprio: torch.Tensor
    dones: torch.Tensor
    is_online: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    def to(self, device: torch.device | str) -> "AWRStepBatch":
        target = require_cuda_device(str(device), name="critic batch device")
        return AWRStepBatch(
            image_obs=self.image_obs.to(target),
            proprio=self.proprio.to(target),
            actions=self.actions.to(target),
            rewards=self.rewards.to(target),
            next_image_obs=self.next_image_obs.to(target),
            next_proprio=self.next_proprio.to(target),
            dones=self.dones.to(target),
            is_online=self.is_online.to(target),
            metadata=self.metadata,
        )

    @property
    def batch_size(self) -> int:
        return int(self.actions.shape[0])

    @classmethod
    def concat(cls, batches: list["AWRStepBatch"]) -> "AWRStepBatch":
        batches = [batch for batch in batches if batch.batch_size]
        if not batches:
            raise ValueError("At least one non-empty step batch is required.")
        if len(batches) == 1:
            return batches[0]
        return cls(
            image_obs=torch.cat([batch.image_obs for batch in batches]),
            proprio=torch.cat([batch.proprio for batch in batches]),
            actions=torch.cat([batch.actions for batch in batches]),
            rewards=torch.cat([batch.rewards for batch in batches]),
            next_image_obs=torch.cat([batch.next_image_obs for batch in batches]),
            next_proprio=torch.cat([batch.next_proprio for batch in batches]),
            dones=torch.cat([batch.dones for batch in batches]),
            is_online=torch.cat([batch.is_online for batch in batches]),
            metadata=_concat_metadata(batches),
        )


def _concat_metadata(batches: list[Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for batch in batches:
        for key, value in batch.metadata.items():
            merged.setdefault(key, [])
            merged[key].extend(value if isinstance(value, list) else [value])
    return merged


__all__ = ["AWRActorBatch", "AWRStepBatch"]
