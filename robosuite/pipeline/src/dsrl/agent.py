from __future__ import annotations

import copy
import math
from contextlib import contextmanager
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .batch import DSRLBatch
from .config import DSRLConfig
from .networks import TanhGaussianActor, TwinQ, flatten_state


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
    def __init__(self, config: DSRLConfig) -> None:
        config.validate()
        self.config = config
        self.device = require_cuda(config.learner_device)
        network = config.network

        self.actor = TanhGaussianActor(
            network.state_dim,
            network.chunk_dim,
            network.hidden_dims,
            network.latent_limit,
            network.log_std_min,
            network.log_std_max,
        ).to(self.device)
        self.qa = TwinQ(network.state_dim, network.chunk_dim, network.hidden_dims).to(self.device)
        self.target_qa = copy.deepcopy(self.qa).to(self.device).eval()

        self.target_qa.requires_grad_(False)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.qa_optimizer = torch.optim.Adam(self.qa.parameters(), lr=config.learning_rate)
        self.log_alpha = torch.tensor(
            math.log(config.initial_alpha), device=self.device, dtype=torch.float32, requires_grad=True
        )
        self.alpha_optimizer = torch.optim.Adam((self.log_alpha,), lr=config.learning_rate)
        self.qa_updates = 0
        self.actor_updates = 0
        self.alpha_updates = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def _validate_batch(self, batch: DSRLBatch) -> None:
        batch.validate(
            action_horizon=self.config.network.action_horizon,
            action_dim=self.config.network.action_dim,
            device=self.device,
        )

    def _state(self, visual_features: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        if visual_features.device != self.device or proprio.device != self.device:
            raise ValueError(f"Actor inputs must be on {self.device}.")
        return flatten_state(visual_features, proprio, self.config.network.state_dim)

    def sample_latent(
        self, visual_features: torch.Tensor, proprio: torch.Tensor, *, deterministic: bool = False
    ) -> torch.Tensor:
        if visual_features.device != self.device or proprio.device != self.device:
            raise ValueError(f"Actor inputs must be on {self.device}.")
        with torch.no_grad():
            state = self._state(visual_features, proprio)
            latent, _ = self.actor.sample(state, deterministic=deterministic)
        return latent.reshape(-1, self.config.network.action_horizon, self.config.network.action_dim)

    def act(self, visual_features: torch.Tensor, proprio: torch.Tensor, *, deterministic: bool = False) -> torch.Tensor:
        return self.sample_latent(visual_features, proprio, deterministic=deterministic)

    def update(self, batch: DSRLBatch) -> dict[str, float]:
        self._validate_batch(batch)
        config = self.config
        entropy_coefficient = self.alpha.detach()

        with torch.no_grad():
            next_state = self._state(batch.next_visual_features, batch.next_proprio)
            next_latent, next_log_prob = self.actor.sample(next_state)
            next_q = self.target_qa.minimum(next_state, next_latent)
            target_q = batch.reward_column + config.gamma * (1.0 - batch.done_column) * (
                next_q - entropy_coefficient * next_log_prob
            )

        state = self._state(batch.visual_features, batch.proprio)
        q1, q2 = self.qa(state, batch.latents)
        qa_loss = 0.5 * (F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q))
        self._require_finite("qa_loss", qa_loss)
        self.qa_optimizer.zero_grad(set_to_none=True)
        qa_loss.backward()
        qa_grad_norm = self._clip_gradients(tuple(self.qa.parameters()))
        self.qa_optimizer.step()
        self.qa_updates += 1

        actor_state = state.detach()
        latent, log_prob = self.actor.sample(actor_state)
        alpha_loss = -(self.log_alpha * (log_prob + config.target_entropy).detach()).mean()
        self._require_finite("alpha_loss", alpha_loss)
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha_updates += 1

        with _freeze_parameters(self.qa):
            actor_q = self.qa.minimum(actor_state, latent)
            actor_loss = (entropy_coefficient * log_prob - actor_q).mean()
            self._require_finite("actor_loss", actor_loss)
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
        actor_grad_norm = self._clip_gradients(tuple(self.actor.parameters()))
        self.actor_optimizer.step()
        self.actor_updates += 1

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

    update_qa_actor = update

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
            "actor": self.actor.state_dict(),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "actor": self.actor.state_dict(),
            "qa": self.qa.state_dict(),
            "target_qa": self.target_qa.state_dict(),
            "log_alpha": self.log_alpha.detach().clone(),
            "optimizers": {
                "actor": self.actor_optimizer.state_dict(),
                "qa": self.qa_optimizer.state_dict(),
                "alpha": self.alpha_optimizer.state_dict(),
            },
            "updates": {
                "qa": self.qa_updates,
                "actor": self.actor_updates,
                "alpha": self.alpha_updates,
            },
        }

    def load_state_dict(self, state: dict[str, Any], *, strict: bool = True) -> None:
        if strict and state.get("config") != self.config.to_dict():
            raise ValueError("Checkpoint DSRL configuration does not match the current configuration.")
        for name in ("actor", "qa", "target_qa"):
            getattr(self, name).load_state_dict(state[name], strict=strict)
        with torch.no_grad():
            self.log_alpha.copy_(state["log_alpha"].to(self.device))
        optimizers = state["optimizers"]
        self.actor_optimizer.load_state_dict(optimizers["actor"])
        self.qa_optimizer.load_state_dict(optimizers["qa"])
        self.alpha_optimizer.load_state_dict(optimizers["alpha"])
        updates = state.get("updates", {})
        self.qa_updates = int(updates.get("qa", 0))
        self.actor_updates = int(updates.get("actor", 0))
        self.alpha_updates = int(updates.get("alpha", 0))
