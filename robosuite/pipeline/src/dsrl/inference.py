from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from .config import NetworkConfig
from .networks import SharedBottleneck, TanhGaussianActor


class DSRLInferencePolicy(nn.Module):
    """Inference-only copy of the DSRL bottleneck and latent actor."""

    def __init__(self, config: NetworkConfig, device: str | torch.device) -> None:
        super().__init__()
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("DSRL inference requires a CUDA device.")
        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        self.bottleneck = SharedBottleneck(
            config.visual_dim, config.proprio_dim, config.state_dim
        )
        self.actor = TanhGaussianActor(
            config.state_dim,
            config.chunk_dim,
            config.hidden_dims,
            config.latent_limit,
            config.log_std_min,
            config.log_std_max,
        )
        self.to(self.device).eval().requires_grad_(False)

    @torch.inference_mode()
    def load_inference_state(self, state: Mapping[str, Any]) -> None:
        self.bottleneck.load_state_dict(state["bottleneck"], strict=True)
        self.actor.load_state_dict(state["actor"], strict=True)

    @torch.inference_mode()
    def latent(
        self,
        dino_features: torch.Tensor,
        proprio: torch.Tensor,
        *,
        deterministic: bool,
    ) -> torch.Tensor:
        representation = self.bottleneck(dino_features, proprio)
        latent, _ = self.actor.sample(representation, deterministic=deterministic)
        return latent.reshape(-1, self.action_horizon, self.action_dim)


__all__ = ["DSRLInferencePolicy"]
