"""Q-chunk and V networks operating on frozen-encoder latents.

These modules never own the encoder. The encoder is held by
`robosuite.pipeline.algorithms.discriminator.encoder.SharedFrozenEncoder` and
shared between DIPOLE flow training, the IQL learner, and the online
discriminator (see docs/DIPOLE_RL.md §C). Callers must invoke encoder
forwards under `torch.no_grad()` and pass the resulting `context` tensor
into `QChunkNetwork.forward` / `VNetwork.forward`.

Architecture (EXPO-FT aligned):
    Each hidden block is Linear -> LayerNorm -> ReLU. Hidden Linears use
    Kaiming-normal init (mode='fan_in', nonlinearity='relu'). The final
    Linear is zero-init for weight + bias to help Q/V calibration at startup.

    Q: MLP on concat(tanh(context), normalize(flatten(action_chunk))) -> (B, 1).
       Following EXPO-FT, the context branch is tanh-bounded to [-1, 1] and the
       action chunk is quantile-normalized (q01/q99 -> [-1, 1]) so the action
       is scale-matched with the context and the Q does not collapse onto V.
       There is NO context compression bottleneck — the full D_ctx latent is
       concatenated directly ("bare concat").
    V: MLP on context                                -> (B, 1).
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
        layers.append(nn.ReLU())
        last_dim = int(hidden_dim)
    final = nn.Linear(last_dim, int(output_dim))
    nn.init.zeros_(final.weight)
    nn.init.zeros_(final.bias)
    layers.append(final)
    return nn.Sequential(*layers)


class QChunkNetwork(nn.Module):
    """Q(context, a_chunk) -> (B, 1).

    Args:
        context_dim: dimensionality D_ctx of the frozen encoder output. The full
            latent is concatenated directly (bare concat; no bottleneck).
        action_dim: per-step action dimensionality D_a (policy action dim,
            NOT the encoder's internal action_dim_per_step).
        action_horizon: H — chunk length; input dim is D_ctx + H * D_a.
        hidden_dims: MLP hidden layer widths.

    Action normalization:
        ``act_q01`` / ``act_q99`` are per-dim quantile buffers (default identity,
        i.e. q01=-1, q99=+1). ``set_action_norm_stats`` is used by the offline
        warmup to fill them from the dataset; they are saved/loaded via
        ``state_dict`` so the vis path reuses the exact same normalization.
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
        self.q_net = _build_mlp(self._input_dim, self.hidden_dims, 1)
        # Quantile-norm buffers (q01/q99 -> [-1, 1]); default = identity for
        # actions already in [-1, 1]. Saved/loaded with the module state.
        self.register_buffer("act_q01", -torch.ones(self.action_dim))
        self.register_buffer("act_q99", torch.ones(self.action_dim))

    def set_action_norm_stats(self, q01: torch.Tensor, q99: torch.Tensor) -> None:
        """Overwrite the action quantile buffers (shape (D_a,))."""
        q01 = torch.as_tensor(q01, dtype=self.act_q01.dtype, device=self.act_q01.device).reshape(-1)
        q99 = torch.as_tensor(q99, dtype=self.act_q99.dtype, device=self.act_q99.device).reshape(-1)
        if q01.numel() != self.action_dim or q99.numel() != self.action_dim:
            raise ValueError(
                f"set_action_norm_stats expected (D_a={self.action_dim},); "
                f"got q01={tuple(q01.shape)} q99={tuple(q99.shape)}"
            )
        self.act_q01.copy_(q01)
        self.act_q99.copy_(q99)

    def _normalize_action(self, action_chunk: torch.Tensor) -> torch.Tensor:
        """Quantile-normalize the chunk to ~[-1, 1] per action dim."""
        span = (self.act_q99 - self.act_q01).clamp_min(1e-6)
        return 2.0 * (action_chunk - self.act_q01) / span - 1.0

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
        ctx = torch.tanh(context)
        action_norm = self._normalize_action(action_chunk)
        x = torch.cat([ctx, action_norm.reshape(B, -1)], dim=-1)
        return self.q_net(x)


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
