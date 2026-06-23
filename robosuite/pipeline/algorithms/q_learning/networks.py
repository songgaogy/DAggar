"""Q-chunk and V networks operating on frozen nnPU encoder features.

These modules never own the encoder. The encoder is held by / shared between
DIPOLE flow training, IQL, and the frozen nnPU discriminator. Callers pass the
action-conditioned chunk feature to Q and the action-free state feature to V.

Architecture (dim-reduced, anti-overfit):
    The frozen encoder emits very high-dim features (chunk 14272-d, state
    12288-d) that have a token structure (16 patch-tokens x per-token dim,
    plus a 64-d proprio block on the state side). Feeding them straight into a
    512x512 MLP at ~zero weight-decay overfits the few effective offline demos.

    We therefore insert a light *Token/Group* projector BEFORE the Q/V head:
    a single Linear is shared across the 16 tokens and compresses each token to
    a small width, the tokens are flattened, LayerNorm + Mish are applied, and
    only then does a small MLP head produce the scalar value.

    Q additionally re-injects the action: token-projection of the 14272 chunk
    feature dilutes the ~7% action signal it carries, so a separate learnable
    ActionProjector embeds the raw action chunk and is concatenated with the
    compressed chunk feature before the Q head.

    Each hidden block in the head is Linear -> LayerNorm -> activation. Hidden
    Linears use Kaiming-normal init; the final Linear is zero-init (weight +
    bias) to keep Q/V calibrated at startup.
"""

from __future__ import annotations

import torch
from torch import nn


def _make_activation(name: str) -> nn.Module:
    key = str(name).lower()
    if key == "mish":
        return nn.Mish()
    if key == "gelu":
        return nn.GELU()
    if key == "relu":
        return nn.ReLU()
    if key in ("silu", "swish"):
        return nn.SiLU()
    raise ValueError(f"Unsupported activation {name!r} (expected mish|gelu|relu|silu).")


def _build_mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
    activation: str = "mish",
) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = int(input_dim)
    for hidden_dim in hidden_dims:
        linear = nn.Linear(last_dim, int(hidden_dim))
        nn.init.kaiming_normal_(linear.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(linear.bias)
        layers.append(linear)
        layers.append(nn.LayerNorm(int(hidden_dim)))
        layers.append(_make_activation(activation))
        last_dim = int(hidden_dim)
    final = nn.Linear(last_dim, int(output_dim))
    nn.init.zeros_(final.weight)
    nn.init.zeros_(final.bias)
    layers.append(final)
    return nn.Sequential(*layers)


class TokenProjector(nn.Module):
    """Token/Group projection of a flattened token-structured feature.

    The input ``(B, n_tokens * token_dim [+ extra_in])`` is split into its
    ``n_tokens`` tokens (each ``token_dim``) and an optional trailing
    ``extra_in`` block (e.g. the 64-d proprio block appended to the state
    feature). One ``Linear(token_dim, out_per_token)`` is *shared* across all
    tokens, the projected tokens are flattened, the optional extra block is
    projected by its own ``Linear`` and concatenated, and finally
    ``LayerNorm -> activation`` is applied.

    Output dim = ``n_tokens * out_per_token + extra_out``.
    """

    def __init__(
        self,
        *,
        n_tokens: int,
        token_dim: int,
        out_per_token: int,
        extra_in: int = 0,
        extra_out: int = 0,
        activation: str = "mish",
    ) -> None:
        super().__init__()
        self.n_tokens = int(n_tokens)
        self.token_dim = int(token_dim)
        self.out_per_token = int(out_per_token)
        self.extra_in = int(extra_in)
        self.extra_out = int(extra_out) if self.extra_in > 0 else 0
        self.token_input_dim = self.n_tokens * self.token_dim
        self.input_dim = self.token_input_dim + self.extra_in
        self.output_dim = self.n_tokens * self.out_per_token + self.extra_out

        self.token_proj = nn.Linear(self.token_dim, self.out_per_token)
        self.extra_proj: nn.Module | None = (
            nn.Linear(self.extra_in, self.extra_out) if self.extra_in > 0 else None
        )
        self.norm = nn.LayerNorm(self.output_dim)
        self.act = _make_activation(activation)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.dim() != 2 or feature.shape[1] != self.input_dim:
            raise ValueError(
                f"TokenProjector expected (B, {self.input_dim}); got {tuple(feature.shape)}"
            )
        batch = feature.shape[0]
        tokens = feature[:, : self.token_input_dim].reshape(
            batch, self.n_tokens, self.token_dim
        )
        projected = self.token_proj(tokens).reshape(batch, self.n_tokens * self.out_per_token)
        if self.extra_proj is not None:
            extra = self.extra_proj(feature[:, self.token_input_dim :])
            projected = torch.cat([projected, extra], dim=-1)
        return self.act(self.norm(projected))


class ActionProjector(nn.Module):
    """Embed the raw action chunk ``(B, H, D_a)`` -> ``(B, action_proj_dim)``.

    Input LayerNorm makes the projector robust to the raw (unnormalized) policy
    action scale; output ``LayerNorm -> activation`` matches the chunk path.
    """

    def __init__(
        self,
        *,
        action_flat_dim: int,
        action_proj_dim: int,
        activation: str = "mish",
    ) -> None:
        super().__init__()
        self.action_flat_dim = int(action_flat_dim)
        self.output_dim = int(action_proj_dim)
        self.in_norm = nn.LayerNorm(self.action_flat_dim)
        self.proj = nn.Linear(self.action_flat_dim, self.output_dim)
        self.out_norm = nn.LayerNorm(self.output_dim)
        self.act = _make_activation(activation)

    def forward(self, action_chunk: torch.Tensor) -> torch.Tensor:
        if action_chunk.dim() == 3:
            flat = action_chunk.reshape(action_chunk.shape[0], -1)
        elif action_chunk.dim() == 2:
            flat = action_chunk
        else:
            raise ValueError(
                f"ActionProjector expected (B, H, D_a) or (B, F); got {tuple(action_chunk.shape)}"
            )
        if flat.shape[1] != self.action_flat_dim:
            raise ValueError(
                f"ActionProjector expected flat action dim {self.action_flat_dim}; "
                f"got {flat.shape[1]} from {tuple(action_chunk.shape)}"
            )
        return self.act(self.out_norm(self.proj(self.in_norm(flat))))


class QHead(nn.Module):
    """Scalar Q head on ``concat(chunk_proj, action_proj)`` -> (B, 1).

    The shared chunk/action projectors live on the IQL learner; this module is
    only the per-ensemble-member MLP head.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (256, 256),
        activation: str = "mish",
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.net = _build_mlp(self.input_dim, self.hidden_dims, 1, activation=activation)

    def forward(self, projected: torch.Tensor) -> torch.Tensor:
        if projected.dim() != 2 or projected.shape[1] != self.input_dim:
            raise ValueError(
                f"QHead expected (B, {self.input_dim}); got {tuple(projected.shape)}"
            )
        return self.net(projected)


class VStateNetwork(nn.Module):
    """V(state_feature) -> (B, 1) with a built-in Token/Group projector.

    Owns its state projector internally (V is a single network, not an
    ensemble) so that ``target_v`` — a full Polyak copy of this module — tracks
    the projector together with the head.
    """

    def __init__(
        self,
        *,
        state_feature_dim: int,
        n_tokens: int,
        proprio_dim: int,
        state_proj_dim: int,
        proprio_proj_dim: int,
        hidden_dims: tuple[int, ...] = (256, 256),
        activation: str = "mish",
    ) -> None:
        super().__init__()
        self.state_feature_dim = int(state_feature_dim)
        self.n_tokens = int(n_tokens)
        self.proprio_dim = int(proprio_dim)
        visual_dim = self.state_feature_dim - self.proprio_dim
        if visual_dim <= 0 or visual_dim % self.n_tokens != 0:
            raise ValueError(
                f"VNetwork: state visual dim {visual_dim} not divisible by "
                f"n_tokens {self.n_tokens} (state_feature_dim={self.state_feature_dim}, "
                f"proprio_dim={self.proprio_dim})."
            )
        if int(state_proj_dim) % self.n_tokens != 0:
            raise ValueError(
                f"VNetwork: state_proj_dim {state_proj_dim} not divisible by n_tokens {self.n_tokens}."
            )
        self.projector = TokenProjector(
            n_tokens=self.n_tokens,
            token_dim=visual_dim // self.n_tokens,
            out_per_token=int(state_proj_dim) // self.n_tokens,
            extra_in=self.proprio_dim,
            extra_out=int(proprio_proj_dim),
            activation=activation,
        )
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.head = _build_mlp(
            self.projector.output_dim, self.hidden_dims, 1, activation=activation
        )

    def forward(self, state_feature: torch.Tensor) -> torch.Tensor:
        if state_feature.dim() != 2 or state_feature.shape[1] != self.state_feature_dim:
            raise ValueError(
                f"VNetwork expected (B, {self.state_feature_dim}); got {tuple(state_feature.shape)}"
            )
        return self.head(self.projector(state_feature))
