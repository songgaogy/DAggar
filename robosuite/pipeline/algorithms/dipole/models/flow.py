from __future__ import annotations

import copy
import threading
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from robosuite.pipeline.common.utils import clone_array_tree
from robosuite.policy.flow_multi.model import build_flow_policy

from ..common import DipoleBatch, DipoleConfig


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


@torch.no_grad()
def _sample_action_sequence(
    model: "DipoleFlowModel",
    *,
    images: torch.Tensor,
    proprio: torch.Tensor,
    language: list[str],
    action_horizon: int,
    n_steps: int,
    guidance_scale: float,
    deterministic: bool,
) -> torch.Tensor:
    batch_size = proprio.shape[0]
    # Reuse the conditioning context across ODE steps for the same observation.
    context = model.encode_multimodal_context(
        images=images,
        proprio=proprio,
        language=language,
    )
    if deterministic:
        x = torch.zeros(batch_size, model.action_dim, action_horizon, device=proprio.device, dtype=proprio.dtype)
    else:
        x = torch.randn(batch_size, model.action_dim, action_horizon, device=proprio.device, dtype=proprio.dtype)
    dt = 1.0 / float(n_steps)
    for step in range(n_steps):
        t = torch.full((batch_size,), float(step) / float(n_steps), device=proprio.device, dtype=proprio.dtype)
        v_pos, v_neg = model.forward_heads_from_context(
            x_t=x,
            t=t,
            context=context,
        )
        v = (1.0 + float(guidance_scale)) * v_pos - float(guidance_scale) * v_neg
        x = x + dt * v
    return x.transpose(1, 2)


class DipoleFlowModel(nn.Module):
    def __init__(
        self,
        *,
        model_cfg: dict[str, Any],
        proprio_dim: int,
        action_dim: int,
        camera_names: list[str],
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
        self.image_encoder = base_model.image_encoder
        self.proprio_tokenizer = base_model.proprio_tokenizer
        self.language_encoder = base_model.language_encoder
        self.language_guided_modulation = base_model.language_guided_modulation
        self.fusion = base_model.fusion
        self.condition_aggregator = base_model.condition_aggregator
        self.flow_head_pos = base_model.flow_head
        self.flow_head_neg = copy.deepcopy(base_model.flow_head)

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

    def forward_heads(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        images: torch.Tensor,
        proprio: torch.Tensor,
        language: list[str] | tuple[str, ...] | str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context = self.encode_multimodal_context(images=images, proprio=proprio, language=language)
        return self.forward_heads_from_context(x_t=x_t, t=t, context=context)

    def forward_heads_from_context(
        self,
        *,
        x_t: torch.Tensor,
        t: torch.Tensor,
        context: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos = self.flow_head_pos(
            x_t=x_t,
            timesteps=t,
            task_scene_cond=context["task_scene_cond"],
            context_tokens=context["context_tokens"],
            context_padding_mask=context["context_padding_mask"],
        )
        neg = self.flow_head_neg(
            x_t=x_t,
            timesteps=t,
            task_scene_cond=context["task_scene_cond"],
            context_tokens=context["context_tokens"],
            context_padding_mask=context["context_padding_mask"],
        )
        return pos, neg


class DipoleFlowPolicy:
    def __init__(
        self,
        *,
        model_cfg: dict[str, Any],
        config: DipoleConfig,
        camera_names: list[str],
    ) -> None:
        self.model_cfg = copy.deepcopy(model_cfg)
        self.config = config
        self.camera_names = [str(name) for name in camera_names]
        self.device = torch.device(config.device)
        self.inference_device = torch.device(config.inference_device or config.device)
        self.language_instruction = str(config.language_instruction or config.task_name or "perform the task")

        self.model = DipoleFlowModel(
            model_cfg=self.model_cfg,
            proprio_dim=int(config.proprio_dim),
            action_dim=int(config.action_dim),
            camera_names=self.camera_names,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            [param for param in self.model.parameters() if param.requires_grad],
            lr=float(config.learning_rate),
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

        self._image_mean = torch.tensor(
            [0.485, 0.456, 0.406],
            dtype=torch.float32,
            device=self.device,
        ).view(1, 1, 3, 1, 1)
        self._image_std = torch.tensor(
            [0.229, 0.224, 0.225],
            dtype=torch.float32,
            device=self.device,
        ).view(1, 1, 3, 1, 1)
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
        execute_horizon = max(1, min(int(self.config.execute_horizon), int(self.config.action_horizon)))
        if (
            self.current_chunk is None
            or self.step_in_chunk >= execute_horizon
            or self.step_in_chunk >= len(self.current_chunk)
        ):
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
                    guidance_scale=float(self.config.guidance_scale),
                    deterministic=bool(deterministic),
                )[0].detach().cpu().numpy().astype(np.float32)
            if self.act_mean is not None and self.act_std is not None:
                action_seq = action_seq * self.act_std + self.act_mean
            self.current_chunk = action_seq
            self.step_in_chunk = 0

        action = np.asarray(self.current_chunk[self.step_in_chunk], dtype=np.float32)
        self.step_in_chunk += 1
        return action

    def _compute_branch_weights(self, batch: DipoleBatch) -> tuple[torch.Tensor, torch.Tensor]:
        lambda_values = batch.lambda_values
        force_positive = batch.force_positive
        # High discriminator lambda should emphasize the failure-oriented branch.
        negative_weight = torch.sigmoid(float(self.config.beta) * lambda_values)
        negative_weight = negative_weight * (1.0 - force_positive)
        positive_weight = force_positive + (1.0 - force_positive) * (1.0 - negative_weight)
        return positive_weight, negative_weight

    def _branch_loss(
        self,
        *,
        v_pred: torch.Tensor,
        v_target: torch.Tensor,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        action_sequences: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        flat_weights = weights.reshape(-1)
        flow_per_sample = torch.mean((v_pred - v_target) ** 2, dim=(1, 2))
        flow_loss = _weighted_mean(flow_per_sample, flat_weights)
        x1_pred = x_t + (1.0 - timesteps).view(-1, 1, 1) * v_pred
        endpoint_per_sample = torch.mean(
            (x1_pred - action_sequences) ** 2,
            dim=(1, 2),
        )
        endpoint_loss = _weighted_mean(endpoint_per_sample, flat_weights)
        if action_sequences.shape[1] > 1:
            smooth_per_sample = torch.mean((x1_pred[:, 1:] - x1_pred[:, :-1]) ** 2, dim=(1, 2))
            smooth_loss = _weighted_mean(smooth_per_sample, flat_weights)
        else:
            smooth_loss = torch.zeros((), device=self.device, dtype=action_sequences.dtype)
        total = (
            flow_loss
            + float(self.config.lambda_endpoint) * endpoint_loss
            + float(self.config.lambda_smooth) * smooth_loss
        )
        return total, flow_loss, endpoint_loss, smooth_loss

    def update(self, batch: DipoleBatch) -> dict[str, float]:
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
        pos_weight, neg_weight = self._compute_branch_weights(batch)

        self.optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(enabled=(self.device.type == "cuda"), device_type=self.device.type):
            context = self.model.encode_multimodal_context(
                images=batch.image_obs,
                proprio=batch.proprio,
                language=language,
            )
            v_pos, v_neg = self.model.forward_heads_from_context(
                x_t=x_t.transpose(1, 2),
                t=timesteps,
                context=context,
            )
            v_pos = v_pos.transpose(1, 2)
            v_neg = v_neg.transpose(1, 2)
            pos_total, pos_flow, pos_endpoint, pos_smooth = self._branch_loss(
                v_pred=v_pos,
                v_target=v_target,
                x_t=x_t,
                timesteps=timesteps,
                action_sequences=batch.action_sequences,
                weights=pos_weight,
            )
            neg_total, neg_flow, neg_endpoint, neg_smooth = self._branch_loss(
                v_pred=v_neg,
                v_target=v_target,
                x_t=x_t,
                timesteps=timesteps,
                action_sequences=batch.action_sequences,
                weights=neg_weight,
            )

            # DIPOLE loss
            loss = (
                float(self.config.positive_loss_scale) * pos_total
                + float(self.config.negative_loss_scale) * neg_total
            )

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float(self.config.grad_clip_norm))
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return {
            "loss": float(loss.detach().cpu().item()),
            "pos_loss": float(pos_total.detach().cpu().item()),
            "neg_loss": float(neg_total.detach().cpu().item()),
            "pos_flow_loss": float(pos_flow.detach().cpu().item()),
            "neg_flow_loss": float(neg_flow.detach().cpu().item()),
            "pos_endpoint_loss": float(pos_endpoint.detach().cpu().item()),
            "neg_endpoint_loss": float(neg_endpoint.detach().cpu().item()),
            "pos_smooth_loss": float(pos_smooth.detach().cpu().item()),
            "neg_smooth_loss": float(neg_smooth.detach().cpu().item()),
            "mean_lambda": float(batch.lambda_values.detach().mean().cpu().item()),
            "max_lambda": float(batch.lambda_values.detach().max().cpu().item()),
            "mean_pos_weight": float(pos_weight.detach().mean().cpu().item()),
            "mean_neg_weight": float(neg_weight.detach().mean().cpu().item()),
            "force_positive_ratio": float(batch.force_positive.detach().mean().cpu().item()),
            "online_ratio": float(batch.is_online.detach().mean().cpu().item()),
        }

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
                "optimizer": self.optimizer.state_dict(),
                "act_mean": None if self.act_mean is None else torch.as_tensor(self.act_mean),
                "act_std": None if self.act_std is None else torch.as_tensor(self.act_std),
                "prop_mean": None if self.prop_mean is None else torch.as_tensor(self.prop_mean),
                "prop_std": None if self.prop_std is None else torch.as_tensor(self.prop_std),
            }

    def _expand_single_head_state_dict(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        if any(key.startswith("flow_head_pos.") or key.startswith("flow_head_neg.") for key in state_dict.keys()):
            return dict(state_dict)

        expanded = {}
        model_state = self.model.state_dict()
        for key, value in state_dict.items():
            if key.startswith("flow_head."):
                suffix = key[len("flow_head.") :]
                pos_key = f"flow_head_pos.{suffix}"
                neg_key = f"flow_head_neg.{suffix}"
                if pos_key in model_state:
                    expanded[pos_key] = value
                if neg_key in model_state:
                    expanded[neg_key] = value
                continue
            if key in model_state:
                expanded[key] = value
        return expanded

    def load_model_state(self, state_dict: dict[str, Any], *, strict: bool = True) -> None:
        expanded_state = self._expand_single_head_state_dict(state_dict)
        with self._state_lock:
            if strict and not any(key.startswith("flow_head_pos.") for key in state_dict.keys()):
                current_state = self.model.state_dict()
                merged_state = dict(current_state)
                merged_state.update(expanded_state)
                self.model.load_state_dict(merged_state, strict=True)
            else:
                self.model.load_state_dict(expanded_state, strict=strict)
        self.sync_inference_policy()
        self.reset_action_chunk()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        with self._state_lock:
            self.model.load_state_dict(state_dict["model"])
            optimizer_state = state_dict.get("optimizer")
            if optimizer_state is not None:
                self.optimizer.load_state_dict(optimizer_state)
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
