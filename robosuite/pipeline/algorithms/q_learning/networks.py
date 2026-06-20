"""Q-chunk and V networks operating on frozen nnPU encoder features.

These modules never own the encoder. The encoder is held by
shared between DIPOLE flow training, IQL, and the frozen nnPU discriminator.
Callers pass the action-conditioned chunk feature to Q and the action-free
state feature to V.

Architecture:
    Each hidden block is Linear -> LayerNorm -> GELU. Hidden Linears use
    Kaiming-normal init (mode='fan_in', nonlinearity='relu' as a stand-in
    for GELU). The final Linear is zero-init for weight + bias to help
    Q/V calibration at startup.

    Q: MLP on the nnPU transformer's action-conditioned chunk feature.
    V: MLP on the nnPU encoder's action-free state feature.
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
    """Q(chunk_feature) -> (B, 1).

    Args:
        chunk_feature_dim: dimensionality of the action-conditioned feature.
        hidden_dims: MLP hidden layer widths.

    Action encoding and normalization are owned by the frozen nnPU encoder.
    """

    def __init__(
        self,
        chunk_feature_dim: int,
        hidden_dims: tuple[int, ...] = (512, 512),
    ) -> None:
        super().__init__()
        self.chunk_feature_dim = int(chunk_feature_dim)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)

        self._input_dim = self.chunk_feature_dim
        self.net = _build_mlp(self._input_dim, self.hidden_dims, 1)

    def forward(self, chunk_feature: torch.Tensor) -> torch.Tensor:
        """Args:
            chunk_feature: (B, D_chunk)
        Returns:
            (B, 1) Q value for the entire chunk.
        """
        if chunk_feature.dim() != 2 or chunk_feature.shape[1] != self.chunk_feature_dim:
            raise ValueError(
                "QChunkNetwork expected chunk_feature "
                f"(B, {self.chunk_feature_dim}); got {tuple(chunk_feature.shape)}"
            )
        return self.net(chunk_feature)


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
