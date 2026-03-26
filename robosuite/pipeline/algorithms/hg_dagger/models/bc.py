from __future__ import annotations

import copy
import threading
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from robosuite.pipeline.algorithms.hil_serl.common.types import EncoderConfig, ReplayBatch
from robosuite.pipeline.algorithms.hil_serl.common.utils import nested_to_torch
from robosuite.pipeline.algorithms.hil_serl.models.encoders import MLP, build_encoder

from ..common import BCConfig


LOG_PROB_EPS = 1e-6


class GaussianBCPolicy(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims,
        action_dim: int,
        log_std_min: float,
        log_std_max: float,
        action_low: np.ndarray,
        action_high: np.ndarray,
        tanh_squash_distribution: bool = True,
    ) -> None:
        super().__init__()
        hidden_dims = list(hidden_dims)
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.Tanh(),
            MLP([hidden_dims[0], *hidden_dims[1:]], use_layer_norm=True, activate_final=True)
            if len(hidden_dims) > 1
            else nn.Identity(),
        )
        final_dim = int(hidden_dims[-1])
        self.mean_head = nn.Linear(final_dim, action_dim)
        self.log_std_head = nn.Linear(final_dim, action_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.tanh_squash_distribution = bool(tanh_squash_distribution)
        action_scale = 0.5 * (action_high - action_low)
        action_bias = 0.5 * (action_high + action_low)
        self.register_buffer("action_scale", torch.as_tensor(action_scale, dtype=torch.float32))
        self.register_buffer("action_bias", torch.as_tensor(action_bias, dtype=torch.float32))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(features)
        mean = self.mean_head(hidden)
        log_std = torch.clamp(self.log_std_head(hidden), min=self.log_std_min, max=self.log_std_max)
        return mean, log_std

    def sample(self, features: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        mean, log_std = self(features)
        if deterministic:
            action = self._squash(mean) if self.tanh_squash_distribution else mean
            return action, None

        std = log_std.exp()
        distribution = Normal(mean, std)
        pre_tanh = distribution.rsample()
        action = self._squash(pre_tanh) if self.tanh_squash_distribution else pre_tanh
        log_prob = self.log_prob_from_params(mean, log_std, action)
        return action, log_prob

    def mode(self, features: torch.Tensor) -> torch.Tensor:
        mean, _ = self(features)
        return self._squash(mean) if self.tanh_squash_distribution else mean

    def log_prob(self, features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        mean, log_std = self(features)
        return self.log_prob_from_params(mean, log_std, actions)

    def log_prob_from_params(self, mean: torch.Tensor, log_std: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        std = log_std.exp()
        distribution = Normal(mean, std)
        if not self.tanh_squash_distribution:
            return distribution.log_prob(actions).sum(dim=-1, keepdim=True)

        normalized_action = (actions - self.action_bias) / torch.clamp(self.action_scale, min=LOG_PROB_EPS)
        clipped_action = torch.clamp(normalized_action, min=-1.0 + 1e-6, max=1.0 - 1e-6)
        pre_tanh = torch.atanh(clipped_action)
        log_prob = distribution.log_prob(pre_tanh)
        correction = torch.log(torch.clamp(self.action_scale * (1.0 - clipped_action.pow(2)), min=LOG_PROB_EPS))
        return (log_prob - correction).sum(dim=-1, keepdim=True)

    def _squash(self, values: torch.Tensor) -> torch.Tensor:
        squashed = torch.tanh(values)
        return squashed * self.action_scale + self.action_bias


class HGDaggerBC:
    def __init__(
        self,
        observation_example,
        encoder_config: EncoderConfig,
        config: BCConfig,
        action_low: np.ndarray | None = None,
        action_high: np.ndarray | None = None,
    ) -> None:
        self.encoder_config = encoder_config
        self.config = config
        self.device = torch.device(config.device)
        self.inference_device = torch.device(config.inference_device or config.device)
        self.action_dim = int(config.action_dim)

        action_low = np.asarray(action_low if action_low is not None else -np.ones(self.action_dim), dtype=np.float32)
        action_high = np.asarray(action_high if action_high is not None else np.ones(self.action_dim), dtype=np.float32)

        self.encoder = build_encoder(observation_example=observation_example, config=encoder_config).to(self.device)
        self.policy = GaussianBCPolicy(
            input_dim=int(self.encoder.output_dim),
            hidden_dims=list(config.hidden_dims),
            action_dim=self.action_dim,
            log_std_min=float(config.log_std_min),
            log_std_max=float(config.log_std_max),
            action_low=action_low,
            action_high=action_high,
            tanh_squash_distribution=bool(config.tanh_squash_distribution),
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.policy.parameters()),
            lr=float(config.learning_rate),
        )

        self.inference_encoder = copy.deepcopy(self.encoder).to(self.inference_device)
        self.inference_encoder.eval()
        self.inference_policy = copy.deepcopy(self.policy).to(self.inference_device)
        self.inference_policy.eval()
        self._inference_shadow_encoder = copy.deepcopy(self.inference_encoder).to(self.inference_device)
        self._inference_shadow_encoder.eval()
        self._inference_shadow_policy = copy.deepcopy(self.inference_policy).to(self.inference_device)
        self._inference_shadow_policy.eval()
        self._state_lock = threading.RLock()
        self._inference_lock = threading.Lock()
        self.sync_inference_policy()

    def select_action(self, obs, deterministic: bool = False) -> np.ndarray:
        obs_t = _obs_to_device(obs, self.inference_device)
        with self._inference_lock:
            inference_encoder = self.inference_encoder
            inference_policy = self.inference_policy
        with torch.no_grad():
            features = inference_encoder(obs_t)
            actions, _ = inference_policy.sample(features, deterministic=deterministic)
        return actions.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def update(self, batch: ReplayBatch) -> dict[str, float]:
        batch = batch.to(self.device)
        self.encoder.train(True)
        self.policy.train(True)

        features = self.encoder(batch.obs)
        log_prob = self.policy.log_prob(features, batch.actions)
        predicted_actions = self.policy.mode(features)
        mse = torch.mean((predicted_actions - batch.actions) ** 2)
        actor_loss = -log_prob.mean()

        self.optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.optimizer.step()

        mean_log_prob = float(log_prob.mean().detach().cpu().item())
        return {
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "mse": float(mse.detach().cpu().item()),
            "log_prob": mean_log_prob,
        }

    def sync_inference_policy(self) -> None:
        with self._state_lock:
            self._inference_shadow_encoder.load_state_dict(self.encoder.state_dict())
            self._inference_shadow_policy.load_state_dict(self.policy.state_dict())
            self._inference_shadow_encoder.eval()
            self._inference_shadow_policy.eval()
        with self._inference_lock:
            self.inference_encoder, self._inference_shadow_encoder = self._inference_shadow_encoder, self.inference_encoder
            self.inference_policy, self._inference_shadow_policy = self._inference_shadow_policy, self.inference_policy

    def state_dict(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                "encoder": self.encoder.state_dict(),
                "policy": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        with self._state_lock:
            self.encoder.load_state_dict(state_dict["encoder"])
            self.policy.load_state_dict(state_dict["policy"])
            optimizer_state = state_dict.get("optimizer")
            if optimizer_state is not None:
                self.optimizer.load_state_dict(optimizer_state)
        self.sync_inference_policy()


def _obs_to_device(obs, device: torch.device) -> Any:
    obs_t = nested_to_torch(obs, device=device)
    if isinstance(obs_t, dict):
        return {
            key: value if value.ndim > 1 else value.unsqueeze(0)
            for key, value in obs_t.items()
        }
    if obs_t.ndim == 1:
        return obs_t.unsqueeze(0)
    return obs_t
