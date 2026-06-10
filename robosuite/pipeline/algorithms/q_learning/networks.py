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

    Q (action-sensitive design, borrowed from EXPO-FT critic tricks): the
    frozen-encoder context is unbounded and large in magnitude, so the raw
    action would be drowned out by a plain concat. Instead:
      - context  -> LayerNorm -> tanh           (bounded to [-1, 1])
      - action   -> z-score (per-dim buffers)
                 -> Linear -> LayerNorm -> tanh  (256-d embedding, Kaiming init)
      - concat(ctx, a_emb) -> MLP -> (B, 1)
    The action z-score buffers (action_mean / action_std) are registered so
    they round-trip with state_dict; they default to 0/1 (identity) and are
    populated from the offline dataset during warmup.

    V: MLP on context -> (B, 1)
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


_ACTION_EMBED_DIM = 256


class QChunkNetwork(nn.Module):
    """Q(context, a_chunk) -> (B, 1).

    Args:
        context_dim: dimensionality D_ctx of the frozen encoder output.
        action_dim: per-step action dimensionality D_a (policy action dim,
            NOT the encoder's internal action_dim_per_step).
        action_horizon: H — chunk length.
        hidden_dims: MLP hidden layer widths.

    Input layout (see module docstring): the context is squashed via
    LayerNorm -> tanh, the flattened action chunk is z-scored then projected
    to a 256-d embedding (Linear -> LayerNorm -> tanh), and the two are
    concatenated before the MLP. Q input dim is therefore D_ctx + 256.
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

        # Per-action-dim z-score buffers (broadcast over (B, H, D_a)). Default
        # identity (0/1); populated from the offline dataset during warmup via
        # IQLLearner.set_action_norm_stats. Registered buffers => round-trip
        # with state_dict.
        self.register_buffer("action_mean", torch.zeros(self.action_dim))
        self.register_buffer("action_std", torch.ones(self.action_dim))

        # Context squash: bound the unbounded encoder latent to [-1, 1].
        self.ctx_norm = nn.LayerNorm(self.context_dim)

        # Independent action projection branch. NOTE: the Linear is Kaiming-init
        # (NOT zero-init) — zero-init would make the action embedding identically
        # zero at startup, defeating the whole point of action sensitivity.
        action_input_dim = self.action_dim * self.action_horizon
        action_linear = nn.Linear(action_input_dim, _ACTION_EMBED_DIM)
        nn.init.kaiming_normal_(action_linear.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(action_linear.bias)
        self.action_proj = nn.Sequential(
            action_linear,
            nn.LayerNorm(_ACTION_EMBED_DIM),
        )

        self._input_dim = self.context_dim + _ACTION_EMBED_DIM
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
        ctx = torch.tanh(self.ctx_norm(context))                  # (B, D_ctx), bounded
        a = (action_chunk - self.action_mean) / self.action_std   # z-score, broadcast (D_a,)
        a = a.reshape(B, -1)                                       # (B, H * D_a)
        a = torch.tanh(self.action_proj(a))                       # (B, 256), bounded embedding
        x = torch.cat([ctx, a], dim=-1)                           # (B, D_ctx + 256)
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
