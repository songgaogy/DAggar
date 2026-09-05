from __future__ import annotations

import math

import torch
from torch import nn


class SharedBottleneck(nn.Module):
    def __init__(self, visual_dim: int, proprio_dim: int, state_dim: int) -> None:
        super().__init__()
        self.input_dim = visual_dim + proprio_dim
        self.projection = nn.Linear(self.input_dim, state_dim)
        self.normalization = nn.LayerNorm(state_dim)

    def forward(self, visual_features: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        if visual_features.ndim not in (2, 3) or proprio.ndim != 2:
            raise ValueError("DINO features must be rank two or three and proprio must be rank two.")
        visual_features = visual_features.flatten(start_dim=1)
        inputs = torch.cat((visual_features, proprio), dim=-1).float()
        if inputs.shape[-1] != self.input_dim:
            raise ValueError(f"Expected bottleneck input width {self.input_dim}, got {inputs.shape[-1]}.")
        return torch.tanh(self.normalization(self.projection(inputs)))


def build_mlp(input_dim: int, output_dim: int, hidden_dims: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = input_dim
    for hidden_dim in hidden_dims:
        layers.extend((nn.Linear(width, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh()))
        width = hidden_dim
    layers.append(nn.Linear(width, output_dim))
    return nn.Sequential(*layers)


class TanhGaussianActor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        latent_dim: int,
        hidden_dims: tuple[int, ...],
        latent_limit: float,
        log_std_min: float,
        log_std_max: float,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.latent_limit = float(latent_limit)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.network = build_mlp(state_dim, 2 * latent_dim, hidden_dims)

    def distribution_parameters(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.network(state).chunk(2, dim=-1)
        return mean, log_std.clamp(self.log_std_min, self.log_std_max)

    def sample(self, state: torch.Tensor, *, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.distribution_parameters(state)
        std = log_std.exp()
        pre_tanh = mean if deterministic else mean + std * torch.randn_like(mean)
        unit_action = torch.tanh(pre_tanh)
        latent = unit_action * self.latent_limit
        if deterministic:
            return latent, torch.zeros((state.shape[0], 1), device=state.device, dtype=state.dtype)
        gaussian_log_prob = -0.5 * (
            ((pre_tanh - mean) / std).square() + 2.0 * log_std + math.log(2.0 * math.pi)
        )
        correction = torch.log(self.latent_limit * (1.0 - unit_action.square()) + 1e-6)
        log_prob = (gaussian_log_prob - correction).sum(dim=-1, keepdim=True)
        return latent, log_prob

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.sample(state, deterministic=True)[0]


class TwinQ(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        self.q1 = build_mlp(state_dim + action_dim, 1, hidden_dims)
        self.q2 = build_mlp(state_dim + action_dim, 1, hidden_dims)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat_action = action.flatten(start_dim=1)
        inputs = torch.cat((state, flat_action), dim=-1)
        return self.q1(inputs), self.q2(inputs)

    def minimum(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self(state, action)
        return torch.minimum(q1, q2)
