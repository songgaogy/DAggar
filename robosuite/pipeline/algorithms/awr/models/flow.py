from __future__ import annotations

import copy
import threading
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from robosuite.pipeline.common.utils import clone_array_tree
from robosuite.policy.flow_multi.model import build_flow_policy

from ..common import AWRActorBatch, AWRConfig, AWRStepBatch


def _center_crop_resize(image: np.ndarray, image_size: int) -> np.ndarray:
    height, width = image.shape[:2]
    crop_size = min(height, width)
    y0 = (height - crop_size) // 2
    x0 = (width - crop_size) // 2
    crop = image[y0 : y0 + crop_size, x0 : x0 + crop_size]
    if crop_size == image_size:
        return crop
    ys = np.linspace(0, crop_size - 1, image_size).astype(np.int32)
    xs = np.linspace(0, crop_size - 1, image_size).astype(np.int32)
    return crop[ys][:, xs]


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.reshape(-1)
    values = values.reshape(-1)
    weight_sum = torch.sum(weights).clamp_min(1e-6)
    return torch.sum(values * weights) / weight_sum


def _expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
    weight = torch.where(diff > 0.0, torch.full_like(diff, float(expectile)), torch.full_like(diff, 1.0 - float(expectile)))
    return weight * diff.square()


def _build_mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = int(input_dim)
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(last_dim, int(hidden_dim)))
        layers.append(nn.ReLU(inplace=True))
        last_dim = int(hidden_dim)
    layers.append(nn.Linear(last_dim, int(output_dim)))
    return nn.Sequential(*layers)


@torch.no_grad()
def _sample_action_sequence(
    model: "AWRFlowModel",
    *,
    images: torch.Tensor,
    proprio: torch.Tensor,
    language: list[str],
    action_horizon: int,
    n_steps: int,
    deterministic: bool,
) -> torch.Tensor:
    batch_size = proprio.shape[0]
    context = model.encode_multimodal_context(images=images, proprio=proprio, language=language)
    if deterministic:
        x = torch.zeros(batch_size, model.action_dim, action_horizon, device=proprio.device, dtype=proprio.dtype)
    else:
        x = torch.randn(batch_size, model.action_dim, action_horizon, device=proprio.device, dtype=proprio.dtype)
    dt = 1.0 / float(n_steps)
    for step in range(n_steps):
        t = torch.full((batch_size,), float(step) / float(n_steps), device=proprio.device, dtype=proprio.dtype)
        v = model.forward_actor_from_context(x_t=x, t=t, context=context)
        x = x + dt * v
    return x.transpose(1, 2)


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
        self.model_cfg = copy.deepcopy(model_cfg)
        base_model = build_flow_policy(
            self.model_cfg,
            proprio_dim=int(proprio_dim),
            action_dim=int(action_dim),
            camera_names=[str(name) for name in camera_names],
        )
        self.camera_names = list(base_model.camera_names)
        self.action_dim = int(base_model.action_dim)
        self.feature_dim = int(base_model.feature_dim)
        self.context_dim = int(base_model.condition_aggregator.output_dim)
        self.image_encoder = base_model.image_encoder
        self.proprio_tokenizer = base_model.proprio_tokenizer
        self.language_encoder = base_model.language_encoder
        self.language_guided_modulation = base_model.language_guided_modulation
        self.fusion = base_model.fusion
        self.condition_aggregator = base_model.condition_aggregator
        self.flow_head = base_model.flow_head

        hidden_dims = tuple(int(dim) for dim in critic_hidden_dims)
        self.action_horizon = int(action_horizon)
        self.full_action_dim = self.action_dim * self.action_horizon
        self.q1 = _build_mlp(self.context_dim + self.full_action_dim, hidden_dims, 1)
        self.q2 = _build_mlp(self.context_dim + self.full_action_dim, hidden_dims, 1)
        self.value = _build_mlp(self.context_dim, hidden_dims, 1)

    def encode_multimodal_context(
        self,
        images: torch.Tensor,
        proprio: torch.Tensor,
        language: list[str] | tuple[str, ...] | str,
    ) -> dict[str, torch.Tensor]:
        batch_size, num_cameras, channels, height, width = images.shape
        if num_cameras != len(self.camera_names):
            raise ValueError(f"Expected {len(self.camera_names)} cameras, got {num_cameras}")

        flat_images = images.reshape(batch_size * num_cameras, channels, height, width)
        if flat_images.device.type == "cuda":
            flat_images = flat_images.contiguous(memory_format=torch.channels_last)

        image_tokens = self.image_encoder(flat_images)
        image_tokens = image_tokens.reshape(batch_size, num_cameras * image_tokens.shape[1], image_tokens.shape[2])
        proprio_tokens = self.proprio_tokenizer(proprio)
        language_tokens, language_global, language_mask = self.language_encoder(language)

        image_tokens, proprio_tokens = self.language_guided_modulation(
            visual_tokens=image_tokens,
            proprio_tokens=proprio_tokens,
            language_global=language_global,
        )
        fused_tokens, token_padding_mask = self.fusion(
            language_tokens=language_tokens,
            language_mask=language_mask,
            proprio_tokens=proprio_tokens,
            image_tokens=image_tokens,
            language_global=language_global,
        )
        task_scene_cond = self.condition_aggregator(
            fused_tokens=fused_tokens,
            token_padding_mask=token_padding_mask,
            language_global=language_global,
        )
        context_tokens = torch.cat([language_tokens, fused_tokens], dim=1)
        context_padding_mask = torch.cat([~language_mask, token_padding_mask], dim=1)
        return {
            "task_scene_cond": task_scene_cond,
            "context_tokens": context_tokens,
            "context_padding_mask": context_padding_mask,
        }

    def forward_actor_from_context(
        self,
        *,
        x_t: torch.Tensor,
        t: torch.Tensor,
        context: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.flow_head(
            x_t=x_t,
            timesteps=t,
            task_scene_cond=context["task_scene_cond"],
            context_tokens=context["context_tokens"],
            context_padding_mask=context["context_padding_mask"],
        )

    def forward_qs_from_context(
        self,
        context_features: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if actions.ndim > 2:
            actions = actions.reshape(actions.shape[0], -1)
        critic_input = torch.cat([context_features, actions], dim=-1)
        return self.q1(critic_input), self.q2(critic_input)

    def forward_value_from_context(self, context_features: torch.Tensor) -> torch.Tensor:
        return self.value(context_features)


class AWRFlowPolicy:
    def __init__(
        self,
        *,
        model_cfg: dict[str, Any],
        config: AWRConfig,
        camera_names: list[str],
    ) -> None:
        self.model_cfg = copy.deepcopy(model_cfg)
        self.config = config
        self.camera_names = [str(name) for name in camera_names]
        self.device = torch.device(config.device)
        self.inference_device = torch.device(config.inference_device or config.device)
        self.language_instruction = str(config.language_instruction or config.task_name or "perform the task")

        critic_hidden_dims = tuple(int(dim) for dim in config.critic_hidden_dims)
        self.model = AWRFlowModel(
            model_cfg=self.model_cfg,
            proprio_dim=int(config.proprio_dim),
            action_dim=int(config.action_dim),
            action_horizon=int(config.action_horizon),
            camera_names=self.camera_names,
            critic_hidden_dims=critic_hidden_dims,
        ).to(self.device)
        actor_parameters = list(self.model.image_encoder.parameters())
        actor_parameters += list(self.model.proprio_tokenizer.parameters())
        actor_parameters += list(self.model.language_encoder.parameters())
        actor_parameters += list(self.model.language_guided_modulation.parameters())
        actor_parameters += list(self.model.fusion.parameters())
        actor_parameters += list(self.model.condition_aggregator.parameters())
        actor_parameters += list(self.model.flow_head.parameters())
        self.actor_optimizer = torch.optim.AdamW(
            [param for param in actor_parameters if param.requires_grad],
            lr=float(config.actor_learning_rate),
            weight_decay=float(config.weight_decay),
        )
        critic_parameters = list(self.model.q1.parameters()) + list(self.model.q2.parameters())
        self.q_optimizer = torch.optim.AdamW(
            [param for param in critic_parameters if param.requires_grad],
            lr=float(config.critic_learning_rate),
            weight_decay=float(config.weight_decay),
        )
        self.v_optimizer = torch.optim.AdamW(
            [param for param in self.model.value.parameters() if param.requires_grad],
            lr=float(config.critic_learning_rate),
            weight_decay=float(config.weight_decay),
        )
        self.scaler = torch.amp.GradScaler(enabled=(self.device.type == "cuda"), device=self.device)

        self.inference_model = copy.deepcopy(self.model).to(self.inference_device)
        self.inference_model.eval()
        self._inference_shadow_model = copy.deepcopy(self.inference_model).to(self.inference_device)
        self._inference_shadow_model.eval()
        self._state_lock = threading.RLock()
        self._inference_lock = threading.Lock()

        self.act_mean: np.ndarray | None = None
        self.act_std: np.ndarray | None = None
        self.prop_mean: np.ndarray | None = None
        self.prop_std: np.ndarray | None = None
        self.current_chunk: np.ndarray | None = None
        self.step_in_chunk = 0

        self._inference_image_mean = torch.tensor(
            [0.485, 0.456, 0.406],
            dtype=torch.float32,
            device=self.inference_device,
        ).view(1, 1, 3, 1, 1)
        self._inference_image_std = torch.tensor(
            [0.229, 0.224, 0.225],
            dtype=torch.float32,
            device=self.inference_device,
        ).view(1, 1, 3, 1, 1)
        self.sync_inference_policy()

    def set_language_instruction(self, language_instruction: str) -> None:
        self.language_instruction = str(language_instruction)

    def set_normalizers(
        self,
        *,
        action_mean: np.ndarray | None,
        action_std: np.ndarray | None,
        proprio_mean: np.ndarray | None,
        proprio_std: np.ndarray | None,
    ) -> None:
        self.act_mean = None if action_mean is None else np.asarray(action_mean, dtype=np.float32).copy()
        self.act_std = None if action_std is None else np.asarray(action_std, dtype=np.float32).copy()
        self.prop_mean = None if proprio_mean is None else np.asarray(proprio_mean, dtype=np.float32).copy()
        self.prop_std = None if proprio_std is None else np.asarray(proprio_std, dtype=np.float32).copy()

    def has_normalizers(self) -> bool:
        return self.act_mean is not None and self.act_std is not None and self.prop_mean is not None and self.prop_std is not None

    def reset_action_chunk(self) -> None:
        self.current_chunk = None
        self.step_in_chunk = 0

    def notify_intervention(self) -> None:
        self.reset_action_chunk()

    def select_action(self, obs, deterministic: bool = False) -> np.ndarray:
        """
        Select a single env-step action while internally using an action "chunk".

        The policy generates an action sequence of length `action_horizon`, but at runtime we only
        execute one action per env step. We therefore cache the sampled sequence (`current_chunk`)
        and consume it step-by-step.

        `execute_horizon` controls how many consecutive env steps we take from the cached chunk
        before forcing a re-sample. This lets us trade off compute (fewer model calls) vs. reactivity.
        """
        execute_horizon = max(1, min(int(self.config.execute_horizon), int(self.config.action_horizon)))
        if (
            self.current_chunk is None
            or self.step_in_chunk >= execute_horizon
            or self.step_in_chunk >= len(self.current_chunk)
        ):
            # (Re)sample a fresh action sequence conditioned on the current observation.
            images = []
            for camera_name in self.camera_names:
                image = np.asarray(obs[camera_name], dtype=np.uint8)
                image = _center_crop_resize(image, int(self.config.image_size))
                images.append(np.transpose(image.astype(np.float32) / 255.0, (2, 0, 1)))
            image_tensor = torch.from_numpy(np.stack(images, axis=0)).unsqueeze(0).to(self.inference_device)
            image_tensor = (image_tensor - self._inference_image_mean) / self._inference_image_std

            proprio = np.asarray(obs["state"], dtype=np.float32)
            if self.prop_mean is not None and self.prop_std is not None:
                proprio = (proprio - self.prop_mean) / (self.prop_std + 1e-6)
            proprio_tensor = torch.from_numpy(proprio).unsqueeze(0).to(self.inference_device)

            with self._inference_lock:
                inference_model = self.inference_model
                action_seq = _sample_action_sequence(
                    inference_model,
                    images=image_tensor,
                    proprio=proprio_tensor,
                    language=[self.language_instruction],
                    action_horizon=int(self.config.action_horizon),
                    n_steps=int(self.config.n_ode_steps),
                    deterministic=bool(deterministic),
                )[0].detach().cpu().numpy().astype(np.float32)
            if self.act_mean is not None and self.act_std is not None:
                action_seq = action_seq * self.act_std + self.act_mean
            self.current_chunk = action_seq
            self.step_in_chunk = 0

        # Execute the next action from the cached chunk.
        action = np.asarray(self.current_chunk[self.step_in_chunk], dtype=np.float32)
        self.step_in_chunk += 1
        return action

    def update_value(self, batch: AWRStepBatch) -> dict[str, float]:
        batch = batch.to(self.device)
        self.model.eval()
        language = [self.language_instruction] * batch.batch_size

        with torch.no_grad():
            current_context = self.model.encode_multimodal_context(
                images=batch.image_obs,
                proprio=batch.proprio,
                language=language,
            )["task_scene_cond"].detach()
            next_context = self.model.encode_multimodal_context(
                images=batch.next_image_obs,
                proprio=batch.next_proprio,
                language=language,
            )["task_scene_cond"].detach()
            target_v = self.model.forward_value_from_context(next_context)
            bootstrap_discount = float(self.config.discount) ** int(self.config.action_horizon)
            target_q = batch.rewards + bootstrap_discount * (1.0 - batch.dones) * target_v

        self.q_optimizer.zero_grad(set_to_none=True)
        q1_pred, q2_pred = self.model.forward_qs_from_context(current_context, batch.actions)
        q1_loss = torch.mean((q1_pred - target_q) ** 2)
        q2_loss = torch.mean((q2_pred - target_q) ** 2)
        q_loss = q1_loss + q2_loss
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.model.q1.parameters()) + list(self.model.q2.parameters()), max_norm=float(self.config.grad_clip_norm))
        self.q_optimizer.step()

        with torch.no_grad():
            q1_detached, q2_detached = self.model.forward_qs_from_context(current_context, batch.actions)
            q_min = torch.minimum(q1_detached, q2_detached)
        self.v_optimizer.zero_grad(set_to_none=True)
        v_pred = self.model.forward_value_from_context(current_context)
        value_loss = torch.mean(_expectile_loss(q_min - v_pred, float(self.config.expectile)))
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.value.parameters(), max_norm=float(self.config.grad_clip_norm))
        self.v_optimizer.step()

        return {
            "q_loss": float(q_loss.detach().cpu().item()),
            "q1_loss": float(q1_loss.detach().cpu().item()),
            "q2_loss": float(q2_loss.detach().cpu().item()),
            "value_loss": float(value_loss.detach().cpu().item()),
            "mean_reward": float(batch.rewards.detach().mean().cpu().item()),
            "mean_target_q": float(target_q.detach().mean().cpu().item()),
            "mean_q": float(q_min.detach().mean().cpu().item()),
            "mean_v": float(v_pred.detach().mean().cpu().item()),
            "online_step_ratio": float(batch.is_online.detach().mean().cpu().item()),
        }

    def update_actor(self, batch: AWRActorBatch) -> dict[str, float]:
        batch = batch.to(self.device)
        self.model.train(True)

        noise = torch.randn_like(batch.action_sequences)
        timesteps = torch.rand(batch.batch_size, device=self.device)
        x_t = (
            (1.0 - timesteps).view(-1, 1, 1) * noise
            + timesteps.view(-1, 1, 1) * batch.action_sequences
        )
        v_target = batch.action_sequences - noise
        language = [self.language_instruction] * batch.batch_size

        self.actor_optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(enabled=(self.device.type == "cuda"), device_type=self.device.type):
            context = self.model.encode_multimodal_context(
                images=batch.image_obs,
                proprio=batch.proprio,
                language=language,
            )
            with torch.no_grad():
                context_features = context["task_scene_cond"].detach()
                q1, q2 = self.model.forward_qs_from_context(context_features, batch.raw_action_sequences)
                v = self.model.forward_value_from_context(context_features)
                advantage = torch.minimum(q1, q2) - v
                weights = torch.exp(advantage / max(float(self.config.beta), 1e-6))
                weights = torch.clamp(weights, max=float(self.config.max_adv_weight))

            v_pred = self.model.forward_actor_from_context(
                x_t=x_t.transpose(1, 2),
                t=timesteps,
                context=context,
            ).transpose(1, 2)
            flow_per_sample = torch.mean((v_pred - v_target) ** 2, dim=(1, 2))
            flow_loss = _weighted_mean(flow_per_sample, weights.reshape(-1))
            x1_pred = x_t + (1.0 - timesteps).view(-1, 1, 1) * v_pred
            endpoint_per_sample = torch.mean((x1_pred - batch.action_sequences) ** 2, dim=(1, 2))
            endpoint_loss = _weighted_mean(endpoint_per_sample, weights.reshape(-1))
            if batch.action_sequences.shape[1] > 1:
                smooth_per_sample = torch.mean((x1_pred[:, 1:] - x1_pred[:, :-1]) ** 2, dim=(1, 2))
                smooth_loss = _weighted_mean(smooth_per_sample, weights.reshape(-1))
            else:
                smooth_loss = torch.zeros((), device=self.device, dtype=batch.action_sequences.dtype)
            loss = (
                flow_loss
                + float(self.config.lambda_endpoint) * endpoint_loss
                + float(self.config.lambda_smooth) * smooth_loss
            )

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.actor_optimizer)
        actor_parameters = []
        for group in self.actor_optimizer.param_groups:
            actor_parameters.extend(group["params"])
        torch.nn.utils.clip_grad_norm_(actor_parameters, max_norm=float(self.config.grad_clip_norm))
        self.scaler.step(self.actor_optimizer)
        self.scaler.update()

        return {
            "actor_loss": float(loss.detach().cpu().item()),
            "flow_loss": float(flow_loss.detach().cpu().item()),
            "endpoint_loss": float(endpoint_loss.detach().cpu().item()),
            "smooth_loss": float(smooth_loss.detach().cpu().item()),
            "mean_advantage": float(advantage.detach().mean().cpu().item()),
            "mean_weight": float(weights.detach().mean().cpu().item()),
            "max_weight": float(weights.detach().max().cpu().item()),
            "mean_q_actor": float(torch.minimum(q1, q2).detach().mean().cpu().item()),
            "mean_v_actor": float(v.detach().mean().cpu().item()),
            "online_actor_ratio": float(batch.is_online.detach().mean().cpu().item()),
        }

    def update(self, *, step_batch: AWRStepBatch, actor_batch: AWRActorBatch) -> dict[str, float]:
        value_metrics = self.update_value(step_batch)
        actor_metrics = self.update_actor(actor_batch)
        merged = dict(value_metrics)
        merged.update(actor_metrics)
        merged["loss"] = float(actor_metrics["actor_loss"] + value_metrics["q_loss"] + value_metrics["value_loss"])
        return merged

    def sync_inference_policy(self) -> None:
        with self._state_lock:
            self._inference_shadow_model.load_state_dict(self.model.state_dict())
            self._inference_shadow_model.eval()
        with self._inference_lock:
            self.inference_model, self._inference_shadow_model = self._inference_shadow_model, self.inference_model

    def state_dict(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                "model": self.model.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "q_optimizer": self.q_optimizer.state_dict(),
                "v_optimizer": self.v_optimizer.state_dict(),
                "act_mean": None if self.act_mean is None else torch.as_tensor(self.act_mean),
                "act_std": None if self.act_std is None else torch.as_tensor(self.act_std),
                "prop_mean": None if self.prop_mean is None else torch.as_tensor(self.prop_mean),
                "prop_std": None if self.prop_std is None else torch.as_tensor(self.prop_std),
            }

    def qv_state_dict(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                "q1": self.model.q1.state_dict(),
                "q2": self.model.q2.state_dict(),
                "value": self.model.value.state_dict(),
                "q_optimizer": self.q_optimizer.state_dict(),
                "v_optimizer": self.v_optimizer.state_dict(),
            }

    def load_actor_model_state(self, state_dict: dict[str, Any], *, strict: bool = True) -> None:
        with self._state_lock:
            current_state = self.model.state_dict()
            filtered_state = {
                key: value
                for key, value in state_dict.items()
                if key in current_state and tuple(current_state[key].shape) == tuple(value.shape)
            }
            merged_state = dict(current_state)
            merged_state.update(filtered_state)
            self.model.load_state_dict(merged_state, strict=False if not strict else False)
        self.sync_inference_policy()
        self.reset_action_chunk()

    def load_qv_state_dict(self, state_dict: dict[str, Any], *, load_optimizers: bool = True) -> None:
        with self._state_lock:
            self.model.q1.load_state_dict(state_dict["q1"])
            self.model.q2.load_state_dict(state_dict["q2"])
            self.model.value.load_state_dict(state_dict["value"])
            if load_optimizers:
                q_optimizer_state = state_dict.get("q_optimizer")
                v_optimizer_state = state_dict.get("v_optimizer")
                if q_optimizer_state is not None:
                    self.q_optimizer.load_state_dict(q_optimizer_state)
                if v_optimizer_state is not None:
                    self.v_optimizer.load_state_dict(v_optimizer_state)

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        with self._state_lock:
            self.model.load_state_dict(state_dict["model"])
            actor_optimizer_state = state_dict.get("actor_optimizer")
            q_optimizer_state = state_dict.get("q_optimizer")
            v_optimizer_state = state_dict.get("v_optimizer")
            if actor_optimizer_state is not None:
                self.actor_optimizer.load_state_dict(actor_optimizer_state)
            if q_optimizer_state is not None:
                self.q_optimizer.load_state_dict(q_optimizer_state)
            if v_optimizer_state is not None:
                self.v_optimizer.load_state_dict(v_optimizer_state)
            self.set_normalizers(
                action_mean=state_dict.get("act_mean"),
                action_std=state_dict.get("act_std"),
                proprio_mean=state_dict.get("prop_mean"),
                proprio_std=state_dict.get("prop_std"),
            )
        self.sync_inference_policy()
        self.reset_action_chunk()

    def clone_observation(self, obs: Any) -> Any:
        return clone_array_tree(obs)
