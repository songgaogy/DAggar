"""Q-chunk and V networks operating on frozen-encoder latents.

These modules never own the encoder. The encoder is held by
`robosuite.pipeline.algorithms.discriminator.encoder.SharedFrozenEncoder` and
shared between DIPOLE flow training, the IQL learner, and the online
discriminator (see docs/DIPOLE_RL.md §C). Callers must invoke encoder
forwards under `torch.no_grad()` and pass the resulting `context` tensor
into `QChunkNetwork.forward` / `VNetwork.forward`.

Architecture (mirrors baseline awr/models/flow.py:73-111):
    Q: MLP on concat(context, flatten(action_chunk)) -> (B, 1)
    V: MLP on context                                -> (B, 1)
"""

from __future__ import annotations

import torch
from torch import nn


class QChunkNetwork(nn.Module):
    """Q(context, a_chunk) -> (B, 1).

    Args:
        context_dim: dimensionality D_ctx of the frozen encoder output.
        action_dim: per-step action dimensionality D_a.
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
        self.context_dim = context_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.hidden_dims = hidden_dims
        # Implementation: build _build_mlp(in=context_dim + action_dim*action_horizon,
        # hidden=hidden_dims, out=1). Mirror baseline awr/models/flow.py:109-111.

    def forward(self, context: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        """Args:
            context:      (B, D_ctx)
            action_chunk: (B, H, D_a)
        Returns:
            (B, 1) Q value for the entire chunk.
        """
        raise NotImplementedError


class VNetwork(nn.Module):
    """V(context) -> (B, 1)."""

    def __init__(
        self,
        context_dim: int,
        hidden_dims: tuple[int, ...] = (512, 512),
    ) -> None:
        super().__init__()
        self.context_dim = context_dim
        self.hidden_dims = hidden_dims

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Args:
            context: (B, D_ctx)
        Returns:
            (B, 1) baseline value.
        """
        raise NotImplementedError
