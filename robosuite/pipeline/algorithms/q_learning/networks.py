"""Q-chunk and V networks operating on frozen-encoder latents.

These modules never own the encoder. The encoder is held by
`robosuite.pipeline.algorithms.discriminator.encoder.SharedFrozenEncoder` and
shared between DIPOLE flow training, the IQL learner, and the online
discriminator (see docs/DIPOLE_RL.md §C). Callers must invoke encoder
forwards under `torch.no_grad()` and pass the resulting `context` tensor
into `QChunkNetwork.forward` / `VNetwork.forward`.

Architecture:
    Each hidden block is Linear -> LayerNorm -> GELU. Hidden Linears use
    Kaiming-normal init (mode='fan_in', nonlinearity='relu' as a stand-in
    for GELU). The final Linear is zero-init for weight + bias to help
    Q/V calibration at startup.

    Q: MLP on concat(context, flatten(action_chunk)) -> (B, 1)
    V: MLP on context                                -> (B, 1)
"""

from __future__ import annotations

import torch
from torch import nn


def _build_mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = int(input_dim)
    for hidden_dim in hidden_dims:
        linear = nn.Linear(last_dim, int(hidden_dim))
        nn.init.kaiming_normal_(linear.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(linear.bias)
        layers.append(linear)
        layers.append(nn.LayerNorm(int(hidden_dim)))
        layers.append(nn.GELU())
        last_dim = int(hidden_dim)
    final = nn.Linear(last_dim, int(output_dim))
    nn.init.zeros_(final.weight)
    nn.init.zeros_(final.bias)
    layers.append(final)
    return nn.Sequential(*layers)


class QChunkNetwork(nn.Module):
    """Q(context, a_chunk) -> (B, 1).

    Args:
        context_dim: dimensionality D_ctx of the frozen encoder output.
        action_dim: per-step action dimensionality D_a (policy action dim,
            NOT the encoder's internal action_dim_per_step).
        action_horizon: H — chunk length; input dim is D_ctx + H * D_a.
        hidden_dims: MLP hidden layer widths.
    """

    def __init__(
        self,
        context_dim: int,
        action_dim: int,
        action_horizon: int,
        hidden_dims: tuple[int, ...] = (512, 512),
    ) -> None:
        super().__init__()
        self.context_dim = int(context_dim)
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self._input_dim = self.context_dim + self.action_dim * self.action_horizon
        self.net = _build_mlp(self._input_dim, self.hidden_dims, 1)

    def forward(self, context: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        """Args:
            context:      (B, D_ctx)
            action_chunk: (B, H, D_a)
        Returns:
            (B, 1) Q value for the entire chunk.
        """
        if context.dim() != 2 or context.shape[1] != self.context_dim:
            raise ValueError(
                f"QChunkNetwork expected context (B, {self.context_dim}); got {tuple(context.shape)}"
            )
        if action_chunk.dim() != 3:
            raise ValueError(
                f"QChunkNetwork expected action_chunk (B, H, D_a); got {tuple(action_chunk.shape)}"
            )
        B = context.shape[0]
        x = torch.cat([context, action_chunk.reshape(B, -1)], dim=-1)
        return self.net(x)


class VNetwork(nn.Module):
    """V(context) -> (B, 1)."""

    def __init__(
        self,
        context_dim: int,
        hidden_dims: tuple[int, ...] = (512, 512),
    ) -> None:
        super().__init__()
        self.context_dim = int(context_dim)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.net = _build_mlp(self.context_dim, self.hidden_dims, 1)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Args:
            context: (B, D_ctx)
        Returns:
            (B, 1) baseline value.
        """
        if context.dim() != 2 or context.shape[1] != self.context_dim:
            raise ValueError(
                f"VNetwork expected context (B, {self.context_dim}); got {tuple(context.shape)}"
            )
        return self.net(context)
