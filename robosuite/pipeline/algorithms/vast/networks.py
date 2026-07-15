"""VAST G/V networks operating on frozen nnPU encoder features.

These modules never own the encoder. The encoder is held by / shared between
DIPOLE flow training, VAST, and the frozen nnPU discriminator. Callers pass the
action-free state feature to the value and goal-conditioned return networks.

Architecture (dim-reduced, anti-overfit):
    The frozen encoder emits a very high-dim state feature (12288-d) with a
    token structure (16 patch-tokens x per-token dim, plus a 64-d proprio block
    on the state side). Feeding it straight into a 512x512 MLP at ~zero
    weight-decay overfits the few effective offline demos.

    We therefore insert a light *Token/Group* projector BEFORE the V head:
    a single Linear is shared across the 16 tokens and compresses each token to
    a small width, the tokens are flattened, LayerNorm + Mish are applied, and
    only then does a small MLP head produce the scalar value.

    Each hidden block in the head is Linear -> LayerNorm -> activation. Hidden
    Linears use Kaiming-normal init; the final Linear is zero-init (weight +
    bias) to keep V calibrated at startup.
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


class GoalConditionedValueNetwork(nn.Module):
    """Single-head ``G(s, s_k, k)`` used by VAST value stitching.

    The frozen dynamics encoder remains external. Current and future state
    features pass through one shared token projector, matching their common
    encoder feature space, then the two projected states and the raw scalar
    macro horizon ``k`` are concatenated for a LayerNorm MLP scalar head.
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
                f"GoalConditionedValueNetwork: state visual dim {visual_dim} "
                f"not divisible by n_tokens {self.n_tokens}."
            )
        if int(state_proj_dim) % self.n_tokens != 0:
            raise ValueError(
                "GoalConditionedValueNetwork: state_proj_dim "
                f"{state_proj_dim} not divisible by n_tokens {self.n_tokens}."
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
            2 * self.projector.output_dim + 1,
            self.hidden_dims,
            1,
            activation=activation,
        )

    def forward(
        self,
        state_feature: torch.Tensor,
        future_state_feature: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        if state_feature.shape != future_state_feature.shape:
            raise ValueError(
                "GoalConditionedValueNetwork requires matching current/future "
                f"feature shapes, got {tuple(state_feature.shape)} and "
                f"{tuple(future_state_feature.shape)}."
            )
        if state_feature.dim() != 2 or state_feature.shape[1] != self.state_feature_dim:
            raise ValueError(
                "GoalConditionedValueNetwork expected state features "
                f"(B, {self.state_feature_dim}); got {tuple(state_feature.shape)}."
            )
        k = k.to(device=state_feature.device, dtype=state_feature.dtype).reshape(-1, 1)
        if k.shape[0] != state_feature.shape[0]:
            raise ValueError(
                "GoalConditionedValueNetwork k batch size mismatch: "
                f"features={state_feature.shape[0]} k={k.shape[0]}."
            )
        current = self.projector(state_feature)
        future = self.projector(future_state_feature)
        return self.head(torch.cat([current, future, k], dim=-1))


class VEnsemble(nn.Module):
    """N independent :class:`VStateNetwork` heads -> (B, N).

    Each head is a fully independent network with its own Token/Group projector,
    value head, paired target, optimizer, and clipping step. All heads use the
    full batch; diversity comes from independent initialization and optimization.

    Note each head's final Linear is zero-init (calibrated V≡0 at startup), so
    the ensemble std starts at 0 and grows as the heads diverge under training —
    that is expected.

    ``forward`` returns the stacked per-head scalar values ``(B, N)``; the
    learner reduces them to their mean. ``N == 1`` recovers a single head.
    """

    def __init__(
        self,
        *,
        ensemble_size: int,
        state_feature_dim: int,
        n_tokens: int,
        proprio_dim: int,
        state_proj_dim: int,
        proprio_proj_dim: int,
        hidden_dims: tuple[int, ...] = (256, 256),
        activation: str = "mish",
    ) -> None:
        super().__init__()
        self.ensemble_size = int(ensemble_size)
        if self.ensemble_size < 1:
            raise ValueError(f"VEnsemble: ensemble_size must be >= 1, got {ensemble_size}.")
        self.heads = nn.ModuleList(
            VStateNetwork(
                state_feature_dim=state_feature_dim,
                n_tokens=n_tokens,
                proprio_dim=proprio_dim,
                state_proj_dim=state_proj_dim,
                proprio_proj_dim=proprio_proj_dim,
                hidden_dims=hidden_dims,
                activation=activation,
            )
            for _ in range(self.ensemble_size)
        )

    def forward(self, state_feature: torch.Tensor) -> torch.Tensor:
        # each head -> (B, 1); stack along a new last axis -> (B, N)
        return torch.cat([head(state_feature) for head in self.heads], dim=-1)
