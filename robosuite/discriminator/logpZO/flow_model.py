"""Normalizing flow (RealNVP) used as the density estimator in FAIL-Detect logpZO.

A bijective map `f: z -> s` between a standard Gaussian latent `z ~ N(0, I)` and
the state occupancy `d(s)` of successful demonstrations. We evaluate both
    log p(s)      = log p_Z(f^{-1}(s)) + log|det df^{-1}/ds|          (full NLL)
    log p_Z(f^{-1}(s))                                                (logpZO)
The latter is the "logpZO" variant of the FAIL-Detect framework.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn


def _build_alternating_masks(dim: int, num_layers: int) -> torch.Tensor:
    """Return (num_layers, dim) {0,1} mask. Odd layers flip the pattern."""
    base = torch.zeros(dim, dtype=torch.float32)
    base[::2] = 1.0
    masks = torch.empty(num_layers, dim, dtype=torch.float32)
    for i in range(num_layers):
        masks[i] = base if i % 2 == 0 else 1.0 - base
    return masks


class CouplingLayer(nn.Module):
    """Affine coupling layer (RealNVP).

    Forward direction computes z = f^{-1}(x) on the "active" half
    conditioned on the "frozen" half, plus the log |det| term.
    """

    def __init__(self, dim: int, hidden_dim: int, mask: torch.Tensor, scale_clamp: float = 3.0) -> None:
        super().__init__()
        self.register_buffer("mask", mask.clone())
        self.scale_clamp = float(scale_clamp)

        self.s_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )
        self.t_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )
        # Initialize last layer of s_net small so early-training is near-identity.
        nn.init.zeros_(self.s_net[-1].weight)
        nn.init.zeros_(self.s_net[-1].bias)
        nn.init.zeros_(self.t_net[-1].weight)
        nn.init.zeros_(self.t_net[-1].bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x -> z, log|det df^{-1}/dx| summed over dims."""
        x_frozen = x * self.mask
        s = torch.tanh(self.s_net(x_frozen)) * self.scale_clamp
        t = self.t_net(x_frozen)
        # Only active (1 - mask) positions are transformed.
        active = 1.0 - self.mask
        z = x_frozen + active * ((x - t) * torch.exp(-s))
        log_det = -(active * s).sum(dim=-1)
        return z, log_det

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        z_frozen = z * self.mask
        s = torch.tanh(self.s_net(z_frozen)) * self.scale_clamp
        t = self.t_net(z_frozen)
        active = 1.0 - self.mask
        x = z_frozen + active * (z * torch.exp(s) + t)
        return x


class RealNVPFlow(nn.Module):
    """Stack of RealNVP coupling layers with input standardization."""

    def __init__(
        self,
        dim: int,
        num_layers: int = 8,
        hidden_dim: int = 512,
        scale_clamp: float = 3.0,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = int(dim)
        self.num_layers = int(num_layers)

        masks = _build_alternating_masks(dim, num_layers)
        self.layers = nn.ModuleList(
            [CouplingLayer(dim, hidden_dim, masks[i], scale_clamp) for i in range(num_layers)]
        )

        # Running standardization statistics, populated by `set_standardization`.
        self.register_buffer("feat_mean", torch.zeros(dim))
        self.register_buffer("feat_std", torch.ones(dim))

        self._log2pi = math.log(2.0 * math.pi)

    # ------------------------------------------------------------------ #
    # Standardization                                                    #
    # ------------------------------------------------------------------ #

    def set_standardization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Fix input normalization; call once on training features."""
        if mean.shape[-1] != self.dim or std.shape[-1] != self.dim:
            raise ValueError(
                f"standardization dim mismatch: expected {self.dim}, got mean={tuple(mean.shape)}, std={tuple(std.shape)}"
            )
        std = torch.clamp(std, min=1e-6)
        self.feat_mean.data.copy_(mean.to(self.feat_mean.device))
        self.feat_std.data.copy_(std.to(self.feat_std.device))

    def _standardize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.feat_mean) / self.feat_std

    # ------------------------------------------------------------------ #
    # Forward / log-likelihood                                           #
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return latent z and total log|det df^{-1}/dx| (standardization included)."""
        x_std = self._standardize(x)
        log_det = -torch.log(self.feat_std).sum().expand(x.shape[0])
        z = x_std
        for layer in self.layers:
            z, ld = layer(z)
            log_det = log_det + ld
        return z, log_det

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Full log p(x) under the flow (change-of-variables)."""
        z, log_det = self.forward(x)
        log_pz = -0.5 * (z * z).sum(dim=-1) - 0.5 * self.dim * self._log2pi
        return log_pz + log_det

    def log_pz_only(self, x: torch.Tensor) -> torch.Tensor:
        """logpZO: log p_Z(f^{-1}(x)) only (drop Jacobian)."""
        z, _ = self.forward(x)
        return -0.5 * (z * z).sum(dim=-1) - 0.5 * self.dim * self._log2pi

    def nll(self, x: torch.Tensor) -> torch.Tensor:
        """Mean negative log-likelihood over a batch (training loss)."""
        return -self.log_prob(x).mean()
