from __future__ import annotations

import copy
import threading
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from robosuite.pipeline.src.data.transitions import clone_array_tree
from robosuite.policy.flow_multi.model import build_flow_policy

from .batches import AWRActorBatch, AWRStepBatch
from .config import AWRConfig


def _center_crop_resize(image: np.ndarray, size: int) -> np.ndarray:
    height, width = image.shape[:2]
    crop_size = min(height, width)
    y0, x0 = (height - crop_size) // 2, (width - crop_size) // 2
    crop = image[y0 : y0 + crop_size, x0 : x0 + crop_size]
    if crop_size == size:
        return crop
    ys = np.linspace(0, crop_size - 1, size).astype(np.int32)
    xs = np.linspace(0, crop_size - 1, size).astype(np.int32)
    return crop[ys][:, xs]


def expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
    weight = torch.where(
        diff > 0,
        torch.full_like(diff, float(expectile)),
        torch.full_like(diff, 1.0 - float(expectile)),
    )
    return weight * diff.square()


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.reshape(-1)
    return torch.sum(values.reshape(-1) * weights) / weights.sum().clamp_min(1e-6)


def _build_mlp(
    input_dim: int, hidden_dims: tuple[int, ...], output_dim: int
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = int(input_dim)
    for hidden in hidden_dims:
        layers.extend((nn.Linear(current, int(hidden)), nn.ReLU(inplace=True)))
        current = int(hidden)
    layers.append(nn.Linear(current, int(output_dim)))
    return nn.Sequential(*layers)


class AWRFlowModel(nn.Module):
    def __init__(
        self,
        *,
        model_cfg: dict[str, Any],
        proprio_dim: int,
        action_dim: int,
        action_horizon: int,
        camera_names: list[str],
        critic_hidden_dims: tuple[int, ...],
    ) -> None:
        super().__init__()
        base = build_flow_policy(
            copy.deepcopy(model_cfg),
            proprio_dim=proprio_dim,
            action_dim=action_dim,
            camera_names=camera_names,
        )
        self.camera_names = list(base.camera_names)
        self.action_dim = int(base.action_dim)
        self.context_dim = int(base.condition_aggregator.output_dim)
        self.image_encoder = base.image_encoder
        self.proprio_tokenizer = base.proprio_tokenizer
        self.language_encoder = base.language_encoder
        self.language_guided_modulation = base.language_guided_modulation
        self.fusion = base.fusion
        self.condition_aggregator = base.condition_aggregator
        self.flow_head = base.flow_head
        full_action_dim = self.action_dim * int(action_horizon)
        critic_input_dim = self.context_dim + full_action_dim
        self.q1 = _build_mlp(critic_input_dim, critic_hidden_dims, 1)
        self.q2 = _build_mlp(critic_input_dim, critic_hidden_dims, 1)
        self.value = _build_mlp(self.context_dim, critic_hidden_dims, 1)

    def encode_context(
        self,
        images: torch.Tensor,
        proprio: torch.Tensor,
        language: list[str],
    ) -> dict[str, torch.Tensor]:
        batch_size, num_cameras, channels, height, width = images.shape
        if num_cameras != len(self.camera_names):
            raise ValueError(
                f"Expected {len(self.camera_names)} cameras, got {num_cameras}."
            )
        flat_images = images.reshape(
            batch_size * num_cameras, channels, height, width
        ).contiguous(memory_format=torch.channels_last)
        image_tokens = self.image_encoder(flat_images)
        image_tokens = image_tokens.reshape(
            batch_size,
            num_cameras * image_tokens.shape[1],
            image_tokens.shape[2],
        )
        proprio_tokens = self.proprio_tokenizer(proprio)
        language_tokens, language_global, language_mask = self.language_encoder(
            language
        )
        image_tokens, proprio_tokens = self.language_guided_modulation(
            visual_tokens=image_tokens,
            proprio_tokens=proprio_tokens,
            language_global=language_global,
        )
        fused_tokens, padding_mask = self.fusion(
            language_tokens=language_tokens,
            language_mask=language_mask,
            proprio_tokens=proprio_tokens,
            image_tokens=image_tokens,
            language_global=language_global,
        )
        return {
            "task_scene_cond": self.condition_aggregator(
                fused_tokens=fused_tokens,
                token_padding_mask=padding_mask,
                language_global=language_global,
            ),
            "context_tokens": torch.cat([language_tokens, fused_tokens], dim=1),
            "context_padding_mask": torch.cat(
                [~language_mask, padding_mask], dim=1
            ),
        }

    def actor(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        context: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.flow_head(
            x_t=x_t,
            timesteps=timesteps,
            task_scene_cond=context["task_scene_cond"],
            context_tokens=context["context_tokens"],
            context_padding_mask=context["context_padding_mask"],
        )

    def qs(
        self, context: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actions = actions.reshape(actions.shape[0], -1)
        inputs = torch.cat([context, actions], dim=-1)
        return self.q1(inputs), self.q2(inputs)


@torch.no_grad()
def sample_action_sequence(
    model: AWRFlowModel,
    *,
    images: torch.Tensor,
    proprio: torch.Tensor,
    language: list[str],
    action_horizon: int,
    n_steps: int,
    deterministic: bool,
) -> torch.Tensor:
    batch_size = proprio.shape[0]
    context = model.encode_context(images, proprio, language)
    shape = (batch_size, model.action_dim, action_horizon)
    x = (
        torch.zeros(shape, device=proprio.device, dtype=proprio.dtype)
        if deterministic
        else torch.randn(shape, device=proprio.device, dtype=proprio.dtype)
    )
    for step in range(n_steps):
        time = torch.full(
            (batch_size,),
            float(step) / n_steps,
            device=proprio.device,
            dtype=proprio.dtype,
        )
        x = x + model.actor(x, time, context) / float(n_steps)
    return x.transpose(1, 2)


class AWRFlowPolicy:
    def __init__(
        self,
        *,
        model_cfg: dict[str, Any],
        config: AWRConfig,
        camera_names: list[str],
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("AWR requires CUDA; CPU fallback is not supported.")
        self.config = config
        self.device = torch.device(config.device)
        self.inference_device = torch.device(config.inference_device)
        self.camera_names = list(camera_names)
        self.language_instruction = str(
            config.language_instruction or config.task_name
        )
        self.model = AWRFlowModel(
            model_cfg=model_cfg,
            proprio_dim=config.proprio_dim,
            action_dim=config.action_dim,
            action_horizon=config.action_horizon,
            camera_names=self.camera_names,
            critic_hidden_dims=tuple(config.critic_hidden_dims),
        ).to(self.device)
        actor_modules = (
            self.model.image_encoder,
            self.model.proprio_tokenizer,
            self.model.language_encoder,
            self.model.language_guided_modulation,
            self.model.fusion,
            self.model.condition_aggregator,
            self.model.flow_head,
        )
        actor_parameters = [
            parameter
            for module in actor_modules
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        self.actor_optimizer = torch.optim.AdamW(
            actor_parameters,
            lr=config.actor_learning_rate,
            weight_decay=config.weight_decay,
        )
        self.q_optimizer = torch.optim.AdamW(
            list(self.model.q1.parameters()) + list(self.model.q2.parameters()),
            lr=config.critic_learning_rate,
            weight_decay=config.weight_decay,
        )
        self.v_optimizer = torch.optim.AdamW(
            self.model.value.parameters(),
            lr=config.critic_learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scaler = torch.amp.GradScaler(device="cuda", enabled=True)
        self.inference_model = copy.deepcopy(self.model).to(self.inference_device)
        self.inference_model.eval()
        self._inference_lock = threading.Lock()
        self.act_mean: np.ndarray | None = None
        self.act_std: np.ndarray | None = None
        self.prop_mean: np.ndarray | None = None
        self.prop_std: np.ndarray | None = None
        self.current_chunk: np.ndarray | None = None
        self.step_in_chunk = 0
        self._image_mean = torch.tensor(
            [0.485, 0.456, 0.406], device=self.inference_device
        ).view(1, 1, 3, 1, 1)
        self._image_std = torch.tensor(
            [0.229, 0.224, 0.225], device=self.inference_device
        ).view(1, 1, 3, 1, 1)

    def set_language_instruction(self, instruction: str) -> None:
        self.language_instruction = str(instruction)

    def set_normalizers(
        self,
        *,
        action_mean: Any,
        action_std: Any,
        proprio_mean: Any,
        proprio_std: Any,
    ) -> None:
        def array(value: Any) -> np.ndarray | None:
            if value is None:
                return None
            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()
            return np.asarray(value, dtype=np.float32).copy()

        self.act_mean, self.act_std = array(action_mean), array(action_std)
        self.prop_mean, self.prop_std = array(proprio_mean), array(proprio_std)

    def has_normalizers(self) -> bool:
        return all(
            value is not None
            for value in (self.act_mean, self.act_std, self.prop_mean, self.prop_std)
        )

    def reset_action_chunk(self) -> None:
        self.current_chunk, self.step_in_chunk = None, 0

    def select_action(self, obs: Any, deterministic: bool = False) -> np.ndarray:
        execute_horizon = min(
            self.config.action_horizon, max(1, self.config.execute_horizon)
        )
        if self.current_chunk is None or self.step_in_chunk >= execute_horizon:
            images = [
                np.transpose(
                    _center_crop_resize(
                        np.asarray(obs[name], dtype=np.uint8),
                        self.config.image_size,
                    ).astype(np.float32)
                    / 255.0,
                    (2, 0, 1),
                )
                for name in self.camera_names
            ]
            image_tensor = (
                torch.from_numpy(np.stack(images))
                .unsqueeze(0)
                .to(self.inference_device)
            )
            image_tensor = (image_tensor - self._image_mean) / self._image_std
            proprio = np.asarray(obs["state"], dtype=np.float32)
            if self.prop_mean is not None:
                proprio = (proprio - self.prop_mean) / self.prop_std
            proprio_tensor = torch.from_numpy(proprio).unsqueeze(0).to(
                self.inference_device
            )
            with self._inference_lock:
                sequence = sample_action_sequence(
                    self.inference_model,
                    images=image_tensor,
                    proprio=proprio_tensor,
                    language=[self.language_instruction],
                    action_horizon=self.config.action_horizon,
                    n_steps=self.config.n_ode_steps,
                    deterministic=deterministic,
                )[0].cpu().numpy()
            if self.act_mean is not None:
                sequence = sequence * self.act_std + self.act_mean
            self.current_chunk = sequence.astype(np.float32)
            self.step_in_chunk = 0
        action = self.current_chunk[self.step_in_chunk].copy()
        self.step_in_chunk += 1
        return action

    def update_value(self, batch: AWRStepBatch) -> dict[str, float]:
        batch = batch.to(self.device)
        language = [self.language_instruction] * batch.batch_size
        self.model.eval()
        with torch.no_grad():
            current = self.model.encode_context(
                batch.image_obs, batch.proprio, language
            )["task_scene_cond"].detach()
            next_context = self.model.encode_context(
                batch.next_image_obs, batch.next_proprio, language
            )["task_scene_cond"].detach()
            next_value = self.model.value(next_context)
            target_q = batch.rewards + (
                self.config.discount**self.config.action_horizon
            ) * (1.0 - batch.dones) * next_value
        q1, q2 = self.model.qs(current, batch.actions)
        q1_loss = (q1 - target_q).square().mean()
        q2_loss = (q2 - target_q).square().mean()
        q_loss = q1_loss + q2_loss
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.model.q1.parameters()) + list(self.model.q2.parameters()),
            self.config.grad_clip_norm,
        )
        self.q_optimizer.step()
        with torch.no_grad():
            q1, q2 = self.model.qs(current, batch.actions)
            q_min = torch.minimum(q1, q2)
        value = self.model.value(current)
        value_loss = expectile_loss(
            q_min - value, self.config.expectile
        ).mean()
        self.v_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.value.parameters(), self.config.grad_clip_norm
        )
        self.v_optimizer.step()
        return {
            "q_loss": q_loss.detach().item(),
            "q1_loss": q1_loss.detach().item(),
            "q2_loss": q2_loss.detach().item(),
            "value_loss": value_loss.detach().item(),
            "mean_reward": batch.rewards.mean().detach().item(),
            "mean_target_q": target_q.mean().detach().item(),
            "mean_q": q_min.mean().detach().item(),
            "mean_v": value.mean().detach().item(),
        }

    def update_actor(self, batch: AWRActorBatch) -> dict[str, float]:
        batch = batch.to(self.device)
        self.model.train()
        noise = torch.randn_like(batch.action_sequences)
        time = torch.rand(batch.batch_size, device=self.device)
        x_t = (1.0 - time[:, None, None]) * noise + time[
            :, None, None
        ] * batch.action_sequences
        target = batch.action_sequences - noise
        self.actor_optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            context = self.model.encode_context(
                batch.image_obs,
                batch.proprio,
                [self.language_instruction] * batch.batch_size,
            )
            with torch.no_grad():
                features = context["task_scene_cond"].detach()
                q1, q2 = self.model.qs(features, batch.raw_action_sequences)
                value = self.model.value(features)
                advantage = torch.minimum(q1, q2) - value
                weights = torch.exp(
                    advantage / max(self.config.beta, 1e-6)
                ).clamp(max=self.config.max_adv_weight)
            prediction = self.model.actor(
                x_t.transpose(1, 2), time, context
            ).transpose(1, 2)
            flow_loss = _weighted_mean(
                (prediction - target).square().mean(dim=(1, 2)), weights
            )
            endpoint = x_t + (1.0 - time[:, None, None]) * prediction
            endpoint_loss = _weighted_mean(
                (endpoint - batch.action_sequences)
                .square()
                .mean(dim=(1, 2)),
                weights,
            )
            if endpoint.shape[1] > 1:
                smooth_loss = _weighted_mean(
                    (endpoint[:, 1:] - endpoint[:, :-1])
                    .square()
                    .mean(dim=(1, 2)),
                    weights,
                )
            else:
                smooth_loss = torch.zeros((), device=self.device)
            loss = (
                flow_loss
                + self.config.lambda_endpoint * endpoint_loss
                + self.config.lambda_smooth * smooth_loss
            )
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.actor_optimizer)
        parameters = [
            parameter
            for group in self.actor_optimizer.param_groups
            for parameter in group["params"]
        ]
        torch.nn.utils.clip_grad_norm_(parameters, self.config.grad_clip_norm)
        self.scaler.step(self.actor_optimizer)
        self.scaler.update()
        return {
            "actor_loss": loss.detach().item(),
            "flow_loss": flow_loss.detach().item(),
            "endpoint_loss": endpoint_loss.detach().item(),
            "smooth_loss": smooth_loss.detach().item(),
            "mean_advantage": advantage.mean().detach().item(),
            "mean_weight": weights.mean().detach().item(),
            "max_weight": weights.max().detach().item(),
        }

    def update(
        self, step_batch: AWRStepBatch, actor_batch: AWRActorBatch
    ) -> dict[str, float]:
        metrics = self.update_value(step_batch)
        metrics.update(self.update_actor(actor_batch))
        metrics["loss"] = (
            metrics["q_loss"] + metrics["value_loss"] + metrics["actor_loss"]
        )
        return metrics

    def sync_inference_policy(self) -> None:
        with self._inference_lock:
            self.inference_model.load_state_dict(self.model.state_dict())
            self.inference_model.eval()
        self.reset_action_chunk()

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "normalizers": {
                "act_mean": self.act_mean,
                "act_std": self.act_std,
                "prop_mean": self.prop_mean,
                "prop_std": self.prop_std,
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.model.load_state_dict(state["model"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.q_optimizer.load_state_dict(state["q_optimizer"])
        self.v_optimizer.load_state_dict(state["v_optimizer"])
        self.set_normalizers(**{
            "action_mean": state["normalizers"]["act_mean"],
            "action_std": state["normalizers"]["act_std"],
            "proprio_mean": state["normalizers"]["prop_mean"],
            "proprio_std": state["normalizers"]["prop_std"],
        })
        self.sync_inference_policy()

    def qv_state_dict(self) -> dict[str, Any]:
        return {
            "q1": self.model.q1.state_dict(),
            "q2": self.model.q2.state_dict(),
            "value": self.model.value.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
        }

    def load_qv_state_dict(
        self, state: dict[str, Any], *, load_optimizers: bool = True
    ) -> None:
        self.model.q1.load_state_dict(state["q1"])
        self.model.q2.load_state_dict(state["q2"])
        self.model.value.load_state_dict(state["value"])
        if load_optimizers:
            self.q_optimizer.load_state_dict(state["q_optimizer"])
            self.v_optimizer.load_state_dict(state["v_optimizer"])

    def load_actor_model_state(self, state: dict[str, Any]) -> None:
        current = self.model.state_dict()
        compatible = {
            key: value
            for key, value in state.items()
            if key in current and current[key].shape == value.shape
        }
        current.update(compatible)
        self.model.load_state_dict(current)
        self.sync_inference_policy()

    @staticmethod
    def clone_observation(obs: Any) -> Any:
        return clone_array_tree(obs)


__all__ = [
    "AWRFlowModel",
    "AWRFlowPolicy",
    "expectile_loss",
    "sample_action_sequence",
]
