from __future__ import annotations

import copy
import random
import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from robosuite.pipeline.utils.tensor import (
    random_crop_observations,
    require_cuda_device,
    soft_update,
)
from .encoders import EncoderConfig, MLP, build_encoder
from .replay import ReplayBatch


@dataclass
class SACConfig:
    action_dim: int
    actor_hidden_dims: Sequence[int] = field(default_factory=lambda: (256, 256))
    critic_hidden_dims: Sequence[int] = field(default_factory=lambda: (256, 256))
    discount: float = 0.97
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    init_temperature: float = 1e-2
    target_entropy: Optional[float] = None
    auto_entropy_tuning: bool = True
    backup_entropy: bool = False
    reward_bias: float = 0.0
    critic_ensemble_size: int = 2
    critic_subsample_size: Optional[int] = None
    std_min: float = 1e-5
    std_max: float = 5.0
    augmentation_padding: int = 4
    device: str = "cuda:0"
    inference_device: Optional[str] = "cuda:1"

LOG_PROB_EPS = 1e-6


def _clone_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_to_cpu(item) for item in value)
    return copy.deepcopy(value)


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims,
        action_dim: int,
        std_min: float,
        std_max: float,
        action_low: np.ndarray,
        action_high: np.ndarray,
    ) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.Tanh(),
            MLP([hidden_dims[0], *hidden_dims[1:]], use_layer_norm=True, activate_final=True)
            if len(hidden_dims) > 1
            else nn.Identity(),
        )
        final_dim = hidden_dims[-1]
        self.mean_head = nn.Linear(final_dim, action_dim)
        self.log_std_head = nn.Linear(final_dim, action_dim)
        self.std_min = float(std_min)
        self.std_max = float(std_max)
        action_scale = 0.5 * (action_high - action_low)
        action_bias = 0.5 * (action_high + action_low)
        self.register_buffer("action_scale", torch.as_tensor(action_scale, dtype=torch.float32))
        self.register_buffer("action_bias", torch.as_tensor(action_bias, dtype=torch.float32))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(features)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        return mean, log_std

    def sample(
        self,
        features: torch.Tensor,
        *,
        temperature: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        mean, log_std = self(features)
        if deterministic:
            squashed = torch.tanh(mean)
            action = squashed * self.action_scale + self.action_bias
            return action, None

        std = log_std.exp().clamp(min=self.std_min, max=self.std_max) * temperature.sqrt()
        distribution = Normal(mean, std)
        pre_tanh = distribution.rsample()
        squashed = torch.tanh(pre_tanh)
        action = squashed * self.action_scale + self.action_bias

        log_prob = distribution.log_prob(pre_tanh)
        correction = torch.log(self.action_scale * (1.0 - squashed.pow(2)) + LOG_PROB_EPS)
        log_prob = (log_prob - correction).sum(dim=-1, keepdim=True)
        return action, log_prob


class CriticHead(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dims) -> None:
        super().__init__()
        self.q_network = nn.Sequential(
            nn.Linear(input_dim + action_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.Tanh(),
            MLP([hidden_dims[0], *hidden_dims[1:]], use_layer_norm=True, activate_final=True)
            if len(hidden_dims) > 1
            else nn.Identity(),
        )
        self.head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, features: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([features, action], dim=-1)
        return self.head(self.q_network(inputs))


class CriticEnsemble(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dims, ensemble_size: int) -> None:
        super().__init__()
        self.q_heads = nn.ModuleList(
            [
                CriticHead(input_dim=input_dim, action_dim=action_dim, hidden_dims=hidden_dims)
                for _ in range(int(ensemble_size))
            ]
        )

    def forward(self, features: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        outputs = [head(features, action) for head in self.q_heads]
        return torch.stack(outputs, dim=0)


class GraspCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dims, num_actions: int = 3) -> None:
        super().__init__()
        self.q_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.Tanh(),
            MLP([hidden_dims[0], *hidden_dims[1:]], use_layer_norm=True, activate_final=True)
            if len(hidden_dims) > 1
            else nn.Identity(),
        )
        self.head = nn.Linear(hidden_dims[-1], int(num_actions))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.q_network(features))


class HILSERLSAC:
    def __init__(
        self,
        observation_example,
        encoder_config: EncoderConfig,
        config: SACConfig,
        action_low: np.ndarray | None = None,
        action_high: np.ndarray | None = None,
    ) -> None:
        self.encoder_config = encoder_config
        self.config = config
        self.device = require_cuda_device(config.device, name="SAC learner device")
        self.inference_device = require_cuda_device(
            config.inference_device or config.device,
            name="SAC inference device",
        )
        self.action_dim = int(config.action_dim)
        if self.action_dim < 2:
            raise ValueError("HIL-SERL hybrid mode requires at least one continuous action plus one gripper action.")
        self.continuous_action_dim = self.action_dim - 1

        action_low = np.asarray(action_low if action_low is not None else -np.ones(self.action_dim), dtype=np.float32)
        action_high = np.asarray(action_high if action_high is not None else np.ones(self.action_dim), dtype=np.float32)
        self.full_action_low = action_low
        self.full_action_high = action_high
        self.continuous_action_low = action_low[:-1]
        self.continuous_action_high = action_high[:-1]
        self.grasp_action_values_np = np.linspace(action_low[-1], action_high[-1], num=3, dtype=np.float32)
        self.grasp_action_values = torch.as_tensor(self.grasp_action_values_np, dtype=torch.float32, device=self.device)
        self.inference_grasp_action_values = torch.as_tensor(
            self.grasp_action_values_np,
            dtype=torch.float32,
            device=self.inference_device,
        )

        with torch.device(self.device):
            self.encoder = build_encoder(observation_example=observation_example, config=encoder_config)
        self.image_keys = tuple(getattr(self.encoder, "image_keys", encoder_config.image_keys))
        with torch.no_grad():
            self.encoder(_obs_to_device(observation_example, self.device))
        self.target_encoder = copy.deepcopy(self.encoder).to(self.device)
        self.target_encoder.eval()

        encoder_dim = int(self.encoder.output_dim)
        with torch.device(self.device):
            self.actor = GaussianPolicy(
                input_dim=encoder_dim,
                hidden_dims=list(config.actor_hidden_dims),
                action_dim=self.continuous_action_dim,
                std_min=config.std_min,
                std_max=config.std_max,
                action_low=self.continuous_action_low,
                action_high=self.continuous_action_high,
            )
            self.critic = CriticEnsemble(
                input_dim=encoder_dim,
                action_dim=self.continuous_action_dim,
                hidden_dims=list(config.critic_hidden_dims),
                ensemble_size=config.critic_ensemble_size,
            )
            self.grasp_critic = GraspCritic(
                input_dim=encoder_dim,
                hidden_dims=list(config.critic_hidden_dims),
                num_actions=3,
            )
        self.target_critic = copy.deepcopy(self.critic).to(self.device)
        self.target_critic.eval()
        self.target_grasp_critic = copy.deepcopy(self.grasp_critic).to(self.device)
        self.target_grasp_critic.eval()

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(config.actor_lr))
        self.critic_optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.critic.parameters()),
            lr=float(config.critic_lr),
        )
        self.grasp_critic_optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.grasp_critic.parameters()),
            lr=float(config.critic_lr),
        )

        init_temperature = np.log(np.expm1(float(config.init_temperature)))
        self.temperature_parameter = torch.nn.Parameter(
            torch.tensor([init_temperature], dtype=torch.float32, device=self.device)
        )
        self.alpha_optimizer = torch.optim.Adam([self.temperature_parameter], lr=float(config.alpha_lr))
        self.inference_temperature = torch.tensor(
            [float(config.init_temperature)],
            dtype=torch.float32,
            device=self.inference_device,
        )

        self.target_entropy = (
            float(config.target_entropy) if config.target_entropy is not None else -float(self.action_dim) / 2.0
        )

        self.inference_encoder = copy.deepcopy(self.encoder).to(self.inference_device)
        self.inference_encoder.eval()
        self.inference_actor = copy.deepcopy(self.actor).to(self.inference_device)
        self.inference_actor.eval()
        self.inference_grasp_critic = copy.deepcopy(self.grasp_critic).to(self.inference_device)
        self.inference_grasp_critic.eval()
        self._inference_shadow_encoder = copy.deepcopy(self.inference_encoder).to(self.inference_device)
        self._inference_shadow_encoder.eval()
        self._inference_shadow_actor = copy.deepcopy(self.inference_actor).to(self.inference_device)
        self._inference_shadow_actor.eval()
        self._inference_shadow_grasp_critic = copy.deepcopy(self.inference_grasp_critic).to(self.inference_device)
        self._inference_shadow_grasp_critic.eval()
        self._state_lock = threading.RLock()
        self._inference_lock = threading.Lock()
        self.sync_inference_policy()

    @property
    def alpha(self) -> torch.Tensor:
        return F.softplus(self.temperature_parameter)

    def select_action(self, obs, deterministic: bool = False) -> np.ndarray:
        obs_t = _obs_to_device(obs, self.inference_device)
        with self._inference_lock:
            inference_temperature = self.inference_temperature.clone()
            with torch.no_grad():
                features = self.inference_encoder(obs_t)
                continuous_action, _ = self.inference_actor.sample(
                    features,
                    temperature=inference_temperature,
                    deterministic=deterministic,
                )
                grasp_logits = self.inference_grasp_critic(features)
                grasp_indices = grasp_logits.argmax(dim=-1)
                grasp_action = self.inference_grasp_action_values.index_select(0, grasp_indices).unsqueeze(-1)
                action = torch.cat([continuous_action, grasp_action], dim=-1)
        return action.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def sync_inference_policy(self) -> None:
        with self._state_lock, self._inference_lock:
            self._inference_shadow_encoder.load_state_dict(self.encoder.state_dict())
            self._inference_shadow_actor.load_state_dict(self.actor.state_dict())
            self._inference_shadow_grasp_critic.load_state_dict(self.grasp_critic.state_dict())
            inference_temperature = self.alpha.detach().to(self.inference_device)
            self._inference_shadow_encoder.eval()
            self._inference_shadow_actor.eval()
            self._inference_shadow_grasp_critic.eval()
            self.inference_encoder, self._inference_shadow_encoder = (
                self._inference_shadow_encoder,
                self.inference_encoder,
            )
            self.inference_actor, self._inference_shadow_actor = self._inference_shadow_actor, self.inference_actor
            self.inference_grasp_critic, self._inference_shadow_grasp_critic = (
                self._inference_shadow_grasp_critic,
                self.inference_grasp_critic,
            )
            self.inference_temperature.copy_(inference_temperature)

    def update(
        self,
        batch: ReplayBatch,
        *,
        update_actor: bool = True,
        update_temperature: bool = True,
    ) -> dict[str, float]:
        with self._state_lock:
            batch = batch.to(self.device)
            batch.obs = random_crop_observations(
                batch.obs,
                self.image_keys,
                padding=int(self.config.augmentation_padding),
            )
            batch.next_obs = random_crop_observations(
                batch.next_obs,
                self.image_keys,
                padding=int(self.config.augmentation_padding),
            )
            rewards = batch.rewards + float(self.config.reward_bias)

            critic_info = self._update_critics(batch=batch, rewards=rewards)
            metrics = dict(critic_info)
            soft_update(self.target_encoder, self.encoder, tau=float(self.config.tau))
            soft_update(self.target_critic, self.critic, tau=float(self.config.tau))
            soft_update(self.target_grasp_critic, self.grasp_critic, tau=float(self.config.tau))

            if update_actor:
                actor_info = self._update_actor(batch=batch)
                metrics.update(actor_info)

            if self.config.auto_entropy_tuning and update_temperature:
                alpha_info = self._update_temperature(batch=batch)
                metrics.update(alpha_info)

            metrics["alpha"] = float(self.alpha.detach().cpu().item())
            return metrics

    def state_dict(self) -> dict[str, Any]:
        with self._state_lock, self._inference_lock:
            return {
                "encoder_config": asdict(self.encoder_config),
                "config": asdict(self.config),
                "encoder": _clone_to_cpu(self.encoder.state_dict()),
                "target_encoder": _clone_to_cpu(self.target_encoder.state_dict()),
                "actor": _clone_to_cpu(self.actor.state_dict()),
                "critic": _clone_to_cpu(self.critic.state_dict()),
                "grasp_critic": _clone_to_cpu(self.grasp_critic.state_dict()),
                "target_critic": _clone_to_cpu(self.target_critic.state_dict()),
                "target_grasp_critic": _clone_to_cpu(self.target_grasp_critic.state_dict()),
                "temperature_parameter": self.temperature_parameter.detach().cpu().clone(),
                "actor_optimizer": _clone_to_cpu(self.actor_optimizer.state_dict()),
                "critic_optimizer": _clone_to_cpu(self.critic_optimizer.state_dict()),
                "grasp_critic_optimizer": _clone_to_cpu(self.grasp_critic_optimizer.state_dict()),
                "alpha_optimizer": _clone_to_cpu(self.alpha_optimizer.state_dict()),
                "target_entropy": float(self.target_entropy),
                "inference_encoder": _clone_to_cpu(self.inference_encoder.state_dict()),
                "inference_actor": _clone_to_cpu(self.inference_actor.state_dict()),
                "inference_grasp_critic": _clone_to_cpu(self.inference_grasp_critic.state_dict()),
                "inference_temperature": self.inference_temperature.detach().cpu().clone(),
                "numpy_rng_state": np.random.get_state(),
                "python_rng_state": random.getstate(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all(),
            }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        has_inference_snapshot = "inference_encoder" in state_dict
        with self._state_lock, self._inference_lock:
            self.encoder.load_state_dict(state_dict["encoder"])
            self.target_encoder.load_state_dict(state_dict["target_encoder"])
            self.actor.load_state_dict(state_dict["actor"])
            self.critic.load_state_dict(state_dict["critic"])
            self.grasp_critic.load_state_dict(state_dict["grasp_critic"])
            self.target_critic.load_state_dict(state_dict["target_critic"])
            self.target_grasp_critic.load_state_dict(state_dict["target_grasp_critic"])
            temperature_parameter = state_dict["temperature_parameter"]
            self.temperature_parameter.data.copy_(
                torch.as_tensor(temperature_parameter, device=self.device)
            )
            self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
            self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])
            if "grasp_critic_optimizer" in state_dict:
                self.grasp_critic_optimizer.load_state_dict(state_dict["grasp_critic_optimizer"])
            self.alpha_optimizer.load_state_dict(state_dict["alpha_optimizer"])
            self.target_entropy = float(state_dict.get("target_entropy", self.target_entropy))
            if "numpy_rng_state" in state_dict:
                np.random.set_state(state_dict["numpy_rng_state"])
            if "python_rng_state" in state_dict:
                random.setstate(state_dict["python_rng_state"])
            if "torch_rng_state" in state_dict:
                torch.set_rng_state(state_dict["torch_rng_state"])
            if "cuda_rng_state" in state_dict:
                torch.cuda.set_rng_state_all(state_dict["cuda_rng_state"])
            if has_inference_snapshot:
                self.inference_encoder.load_state_dict(state_dict["inference_encoder"])
                self.inference_actor.load_state_dict(state_dict["inference_actor"])
                self.inference_grasp_critic.load_state_dict(state_dict["inference_grasp_critic"])
                self.inference_temperature.copy_(
                    torch.as_tensor(state_dict["inference_temperature"], device=self.inference_device)
                )
                self._inference_shadow_encoder.load_state_dict(state_dict["inference_encoder"])
                self._inference_shadow_actor.load_state_dict(state_dict["inference_actor"])
                self._inference_shadow_grasp_critic.load_state_dict(state_dict["inference_grasp_critic"])
        if not has_inference_snapshot:
            self.sync_inference_policy()

    def _update_critics(self, batch: ReplayBatch, rewards: torch.Tensor) -> dict[str, float]:
        with torch.no_grad():
            target_next_features = self.target_encoder(batch.next_obs)
            next_features_for_policy = self.encoder(batch.next_obs, stop_gradient=True)
            next_continuous_actions, next_log_probs = self.actor.sample(
                next_features_for_policy,
                temperature=self.alpha.detach(),
                deterministic=False,
            )
            target_qs = self.target_critic(target_next_features, next_continuous_actions)
            if self.config.critic_subsample_size is not None:
                subset = min(int(self.config.critic_subsample_size), int(target_qs.shape[0]))
                indices = torch.randperm(int(target_qs.shape[0]), device=self.device)[:subset]
                target_qs = target_qs.index_select(0, indices)
            target_min_q = target_qs.min(dim=0).values
            if self.config.backup_entropy:
                target_min_q = target_min_q - self.alpha.detach() * next_log_probs
            target_q = rewards + float(self.config.discount) * (1.0 - batch.dones) * target_min_q

            target_next_grasp_qs = self.target_grasp_critic(target_next_features)
            next_grasp_qs = self.grasp_critic(self.encoder(batch.next_obs, stop_gradient=True))
            best_next_grasp_action = next_grasp_qs.argmax(dim=-1, keepdim=True)
            target_next_grasp_q = target_next_grasp_qs.gather(-1, best_next_grasp_action)
            grasp_rewards = rewards + batch.grasp_penalty
            target_grasp_q = grasp_rewards + float(self.config.discount) * (1.0 - batch.dones) * target_next_grasp_q

        critic_features = self.encoder(batch.obs)
        predicted_qs = self.critic(critic_features, batch.actions[..., :-1])
        critic_loss = F.mse_loss(predicted_qs, target_q.unsqueeze(0).expand_as(predicted_qs))

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        grasp_features = self.encoder(batch.obs)
        grasp_action_indices = self._grasp_action_indices(batch.actions[..., -1:])
        predicted_grasp_qs = self.grasp_critic(grasp_features)
        predicted_grasp_q = predicted_grasp_qs.gather(-1, grasp_action_indices)
        grasp_critic_loss = F.mse_loss(predicted_grasp_q, target_grasp_q)

        self.grasp_critic_optimizer.zero_grad(set_to_none=True)
        grasp_critic_loss.backward()
        self.grasp_critic_optimizer.step()

        return {
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "grasp_critic_loss": float(grasp_critic_loss.detach().cpu().item()),
            "predicted_q": float(predicted_qs.mean().detach().cpu().item()),
            "target_q": float(target_q.mean().detach().cpu().item()),
            "predicted_grasp_q": float(predicted_grasp_q.mean().detach().cpu().item()),
            "target_grasp_q": float(target_grasp_q.mean().detach().cpu().item()),
            "reward_mean": float(rewards.mean().detach().cpu().item()),
            "grasp_reward_mean": float(grasp_rewards.mean().detach().cpu().item()),
        }

    def _update_actor(self, batch: ReplayBatch) -> dict[str, float]:
        features = self.encoder(batch.obs, stop_gradient=True)
        actions, log_probs = self.actor.sample(
            features,
            temperature=self.alpha.detach(),
            deterministic=False,
        )
        critic_requires_grad = [param.requires_grad for param in self.critic.parameters()]
        for param in self.critic.parameters():
            param.requires_grad_(False)
        try:
            q_values = self.critic(features, actions).mean(dim=0)
            actor_loss = (self.alpha.detach() * log_probs - q_values).mean()
        finally:
            for param, requires_grad in zip(self.critic.parameters(), critic_requires_grad):
                param.requires_grad_(requires_grad)

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        return {
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "entropy": float((-log_probs).mean().detach().cpu().item()),
        }

    def _update_temperature(self, batch: ReplayBatch) -> dict[str, float]:
        with torch.no_grad():
            next_features = self.encoder(batch.next_obs, stop_gradient=True)
            _, next_log_probs = self.actor.sample(
                next_features,
                temperature=self.alpha.detach(),
                deterministic=False,
            )
        entropy_gap = (-next_log_probs.mean() - self.target_entropy).detach()
        alpha_loss = self.alpha * entropy_gap
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()
        return {"alpha_loss": float(alpha_loss.detach().cpu().item())}

    def _grasp_action_indices(self, grasp_actions: torch.Tensor) -> torch.Tensor:
        distances = (grasp_actions - self.grasp_action_values.view(1, -1)).abs()
        return distances.argmin(dim=-1, keepdim=True)


def _obs_to_device(obs, device: torch.device):
    if isinstance(obs, dict):
        return {key: _obs_to_device(value, device) for key, value in obs.items()}
    tensor = torch.as_tensor(obs, device=device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor
