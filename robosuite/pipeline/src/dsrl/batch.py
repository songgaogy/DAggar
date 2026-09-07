from __future__ import annotations

from dataclasses import dataclass, fields
import torch


@dataclass(frozen=True)
class DSRLBatch:
    visual_features: torch.Tensor
    proprio: torch.Tensor
    latents: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    next_visual_features: torch.Tensor
    next_proprio: torch.Tensor

    def validate(self, *, action_horizon: int, action_dim: int, device: torch.device) -> None:
        batch_size = self.visual_features.shape[0]
        for item in fields(self):
            value = getattr(self, item.name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{item.name} must be a torch.Tensor.")
            if value.device != device:
                raise ValueError(f"{item.name} must be on {device}, got {value.device}.")
            if value.shape[0] != batch_size:
                raise ValueError(f"{item.name} has an inconsistent batch dimension.")
            if not value.is_floating_point():
                raise TypeError(f"{item.name} must be floating point.")
        expected_action_shape = (batch_size, action_horizon, action_dim)
        if self.latents.shape != expected_action_shape:
            raise ValueError(f"latents must have shape {expected_action_shape}, got {tuple(self.latents.shape)}.")
        if self.rewards.numel() != batch_size or self.dones.numel() != batch_size:
            raise ValueError("rewards and dones must contain one scalar per sample.")
        for name, value in (("rewards", self.rewards), ("dones", self.dones)):
            if not bool(torch.all((value == 0) | (value == 1)).item()):
                raise ValueError(f"{name} must contain only zero or one.")

    @property
    def reward_column(self) -> torch.Tensor:
        return self.rewards.reshape(-1, 1)

    @property
    def done_column(self) -> torch.Tensor:
        return self.dones.reshape(-1, 1)
