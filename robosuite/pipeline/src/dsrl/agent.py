from __future__ import annotations

import copy
import math
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .batch import DSRLBatch
from .config import DSRLConfig
from .networks import SharedBottleneck, TanhGaussianActor, TwinQ

FlowDecoder = Callable[[Any, torch.Tensor], torch.Tensor]


def require_cuda(device_name: str) -> torch.device:
    if not device_name.startswith("cuda:"):
        raise RuntimeError("DSRL tensor computation requires an explicit CUDA device.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; DSRL does not support CPU fallback.")
    device = torch.device(device_name)
    if device.index is None or device.index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device {device_name} is unavailable.")
    return device


@contextmanager
def _freeze_parameters(module: nn.Module):
    original = [parameter.requires_grad for parameter in module.parameters()]
    try:
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, requires_grad in zip(module.parameters(), original, strict=True):
            parameter.requires_grad_(requires_grad)


def _polyak_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for source_parameter, target_parameter in zip(source.parameters(), target.parameters(), strict=True):
            target_parameter.lerp_(source_parameter, tau)


class DSRLAgent:
    def __init__(self, config: DSRLConfig, flow_decoder: FlowDecoder) -> None:
        config.validate()
        self.config = config
        self.device = require_cuda(config.learner_device)
        self.flow_decoder = flow_decoder
        network = config.network

        self.bottleneck = SharedBottleneck(network.visual_dim, network.proprio_dim, network.state_dim).to(self.device)
        self.target_bottleneck = copy.deepcopy(self.bottleneck).to(self.device).eval()
        self.actor = TanhGaussianActor(
            network.state_dim,
            network.chunk_dim,
            network.hidden_dims,
            network.latent_limit,
            network.log_std_min,
            network.log_std_max,
        ).to(self.device)
        self.target_actor = copy.deepcopy(self.actor).to(self.device).eval()
        self.qa = TwinQ(network.state_dim, network.chunk_dim, network.hidden_dims).to(self.device)
        self.target_qa = copy.deepcopy(self.qa).to(self.device).eval()
        self.qw = TwinQ(network.state_dim, network.chunk_dim, network.hidden_dims).to(self.device)

        for target in (self.target_bottleneck, self.target_actor, self.target_qa):
            target.requires_grad_(False)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.qa_optimizer = torch.optim.Adam(self.qa.parameters(), lr=config.learning_rate)
        self.qw_optimizer = torch.optim.Adam(self.qw.parameters(), lr=config.learning_rate)
        self.bottleneck_optimizer = torch.optim.Adam(self.bottleneck.parameters(), lr=config.learning_rate)
        self.log_alpha = torch.tensor(
            math.log(config.initial_alpha), device=self.device, dtype=torch.float32, requires_grad=True
        )
        self.alpha_optimizer = torch.optim.Adam((self.log_alpha,), lr=config.learning_rate)
        self.qa_updates = 0
        self.actor_updates = 0
        self.alpha_updates = 0
        self.qw_updates = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def _validate_batch(self, batch: DSRLBatch) -> None:
        batch.validate(
            action_horizon=self.config.network.action_horizon,
            action_dim=self.config.network.action_dim,
            device=self.device,
        )

    def _decode(self, context: Any, latent_flat: torch.Tensor) -> torch.Tensor:
        latent = latent_flat.reshape(
            latent_flat.shape[0], self.config.network.action_horizon, self.config.network.action_dim
        )
        with torch.no_grad():
            actions = self.flow_decoder(context, latent)
        expected = latent.shape
        if actions.device != self.device or actions.shape != expected:
            raise ValueError(f"Flow decoder output must have shape {tuple(expected)} on {self.device}.")
        return actions.detach()

    def sample_latent(
        self, visual_features: torch.Tensor, proprio: torch.Tensor, *, deterministic: bool = False
    ) -> torch.Tensor:
        if visual_features.device != self.device or proprio.device != self.device:
            raise ValueError(f"Actor inputs must be on {self.device}.")
        with torch.no_grad():
            state = self.bottleneck(visual_features, proprio)
            latent, _ = self.actor.sample(state, deterministic=deterministic)
        return latent.reshape(-1, self.config.network.action_horizon, self.config.network.action_dim)

    def act(
        self,
        visual_features: torch.Tensor,
        proprio: torch.Tensor,
        flow_context: Any,
        *,
        deterministic: bool = False,
    ) -> torch.Tensor:
        latent = self.sample_latent(visual_features, proprio, deterministic=deterministic)
        return self._decode(flow_context, latent.flatten(start_dim=1))

    def update_qa_actor(self, batch: DSRLBatch) -> dict[str, float]:
        self._validate_batch(batch)
        config = self.config
        entropy_coefficient = self.alpha.detach()

        with torch.no_grad():
            next_state = self.target_bottleneck(batch.next_dino_features, batch.next_proprio)
            next_latent, next_log_prob = self.target_actor.sample(next_state)
            next_actions = self._decode(batch.next_flow_context, next_latent)
            next_q = self.target_qa.minimum(next_state, next_actions)
            target_q = batch.reward_column + config.gamma * (1.0 - batch.done_column) * (
                next_q - entropy_coefficient * next_log_prob
            )

        state = self.bottleneck(batch.dino_features, batch.proprio)
        q1, q2 = self.qa(state, batch.actions)
        qa_loss = 0.5 * (F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q))
        self._require_finite("qa_loss", qa_loss)
        self.qa_optimizer.zero_grad(set_to_none=True)
        self.bottleneck_optimizer.zero_grad(set_to_none=True)
        qa_loss.backward()
        qa_grad_norm = self._clip_gradients((*self.qa.parameters(), *self.bottleneck.parameters()))
        self.qa_optimizer.step()
        self.bottleneck_optimizer.step()
        self.qa_updates += 1

        actor_state = self.bottleneck(batch.dino_features, batch.proprio).detach()
        latent, log_prob = self.actor.sample(actor_state)
        alpha_loss = -(self.log_alpha * (log_prob + config.target_entropy).detach()).mean()
        self._require_finite("alpha_loss", alpha_loss)
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha_updates += 1

        with _freeze_parameters(self.qw):
            actor_q = self.qw.minimum(actor_state, latent)
            actor_loss = (entropy_coefficient * log_prob - actor_q).mean()
            self._require_finite("actor_loss", actor_loss)
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
        actor_grad_norm = self._clip_gradients(tuple(self.actor.parameters()))
        self.actor_optimizer.step()
        self.actor_updates += 1

        _polyak_update(self.bottleneck, self.target_bottleneck, config.tau)
        _polyak_update(self.actor, self.target_actor, config.tau)
        _polyak_update(self.qa, self.target_qa, config.tau)
        return {
            "qa_loss": float(qa_loss.detach().item()),
            "actor_loss": float(actor_loss.detach().item()),
            "alpha_loss": float(alpha_loss.detach().item()),
            "alpha": float(self.alpha.detach().item()),
            "q_mean": float(torch.minimum(q1, q2).detach().mean().item()),
            "target_q_mean": float(target_q.mean().item()),
            "qa_grad_norm": qa_grad_norm,
            "actor_grad_norm": actor_grad_norm,
        }

    def update_qw(self, batch: DSRLBatch) -> dict[str, float]:
        self._validate_batch(batch)
        batch_size = batch.dino_features.shape[0]
        latent = torch.randn(
            (batch_size, self.config.network.chunk_dim), device=self.device, dtype=torch.float32
        )
        actions = self._decode(batch.flow_context, latent)
        state = self.bottleneck(batch.dino_features, batch.proprio)
        with torch.no_grad():
            target_q1, target_q2 = self.qa(state.detach(), actions)
        q1, q2 = self.qw(state, latent)
        qw_loss = 0.5 * (F.mse_loss(q1, target_q1) + F.mse_loss(q2, target_q2))
        self._require_finite("qw_loss", qw_loss)
        self.qw_optimizer.zero_grad(set_to_none=True)
        self.bottleneck_optimizer.zero_grad(set_to_none=True)
        qw_loss.backward()
        qw_grad_norm = self._clip_gradients((*self.qw.parameters(), *self.bottleneck.parameters()))
        self.qw_optimizer.step()
        self.bottleneck_optimizer.step()
        self.qw_updates += 1
        return {
            "qw_loss": float(qw_loss.detach().item()),
            "qw_mean": float(torch.minimum(q1, q2).detach().mean().item()),
            "qw_target_mean": float(torch.minimum(target_q1, target_q2).mean().item()),
            "qw_grad_norm": qw_grad_norm,
        }

    def _clip_gradients(self, parameters: tuple[nn.Parameter, ...]) -> float:
        if self.config.grad_clip_norm is None:
            norms = [parameter.grad.detach().norm() for parameter in parameters if parameter.grad is not None]
            if not norms:
                return 0.0
            return float(torch.linalg.vector_norm(torch.stack(norms)).item())
        return float(torch.nn.utils.clip_grad_norm_(parameters, self.config.grad_clip_norm).item())

    @staticmethod
    def _require_finite(name: str, value: torch.Tensor) -> None:
        if not bool(torch.isfinite(value).item()):
            raise FloatingPointError(f"Non-finite {name} detected.")

    def inference_state_dict(self) -> dict[str, Any]:
        return {
            "bottleneck": self.bottleneck.state_dict(),
            "actor": self.actor.state_dict(),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "bottleneck": self.bottleneck.state_dict(),
            "target_bottleneck": self.target_bottleneck.state_dict(),
            "actor": self.actor.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "qa": self.qa.state_dict(),
            "target_qa": self.target_qa.state_dict(),
            "qw": self.qw.state_dict(),
            "log_alpha": self.log_alpha.detach().clone(),
            "optimizers": {
                "bottleneck": self.bottleneck_optimizer.state_dict(),
                "actor": self.actor_optimizer.state_dict(),
                "qa": self.qa_optimizer.state_dict(),
                "qw": self.qw_optimizer.state_dict(),
                "alpha": self.alpha_optimizer.state_dict(),
            },
            "updates": {
                "qa": self.qa_updates,
                "actor": self.actor_updates,
                "alpha": self.alpha_updates,
                "qw": self.qw_updates,
            },
        }

    def load_state_dict(self, state: dict[str, Any], *, strict: bool = True) -> None:
        if strict and state.get("config") != self.config.to_dict():
            raise ValueError("Checkpoint DSRL configuration does not match the current configuration.")
        for name in ("bottleneck", "target_bottleneck", "actor", "target_actor", "qa", "target_qa", "qw"):
            getattr(self, name).load_state_dict(state[name], strict=strict)
        with torch.no_grad():
            self.log_alpha.copy_(state["log_alpha"].to(self.device))
        optimizers = state["optimizers"]
        self.bottleneck_optimizer.load_state_dict(optimizers["bottleneck"])
        self.actor_optimizer.load_state_dict(optimizers["actor"])
        self.qa_optimizer.load_state_dict(optimizers["qa"])
        self.qw_optimizer.load_state_dict(optimizers["qw"])
        self.alpha_optimizer.load_state_dict(optimizers["alpha"])
        updates = state.get("updates", {})
        self.qa_updates = int(updates.get("qa", 0))
        self.actor_updates = int(updates.get("actor", 0))
        self.alpha_updates = int(updates.get("alpha", 0))
        self.qw_updates = int(updates.get("qw", 0))
