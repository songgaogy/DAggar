from __future__ import annotations

import copy
import threading
from typing import Any

import numpy as np
import torch

from robosuite.pipeline.common.utils import clone_array_tree
from robosuite.policy.flow_multi.model import build_flow_policy

from ..common import FlowDaggerBatch, FlowDaggerConfig


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


@torch.no_grad()
def _sample_action_sequence(
    model: torch.nn.Module,
    *,
    images: torch.Tensor,
    proprio: torch.Tensor,
    language: list[str],
    action_horizon: int,
    n_steps: int,
    deterministic: bool,
) -> torch.Tensor:
    batch_size = proprio.shape[0]
    if deterministic:
        x = torch.zeros(batch_size, model.action_dim, action_horizon, device=proprio.device, dtype=proprio.dtype)
    else:
        x = torch.randn(batch_size, model.action_dim, action_horizon, device=proprio.device, dtype=proprio.dtype)
    dt = 1.0 / float(n_steps)
    for step in range(n_steps):
        t = torch.full((batch_size,), float(step) / float(n_steps), device=proprio.device, dtype=proprio.dtype)
        v = model(x_t=x, t=t, images=images, proprio=proprio, language=language)
        x = x + dt * v
    return x.transpose(1, 2)


class FlowDaggerPolicy:
    def __init__(
        self,
        *,
        model_cfg: dict[str, Any],
        config: FlowDaggerConfig,
        camera_names: list[str],
    ) -> None:
        self.model_cfg = copy.deepcopy(model_cfg)
        self.config = config
        self.camera_names = [str(name) for name in camera_names]
        self.device = torch.device(config.device)
        self.inference_device = torch.device(config.inference_device or config.device)
        self.language_instruction = str(config.language_instruction or config.task_name or "perform the task")

        self.model = build_flow_policy(
            self.model_cfg,
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
        self._freeze_visual_batch_norm = False

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

    def set_freeze_visual_batch_norm(self, freeze: bool) -> None:
        self._freeze_visual_batch_norm = bool(freeze)
        if self._freeze_visual_batch_norm:
            self._set_visual_batch_norm_eval(freeze_affine=True)

    def _set_visual_batch_norm_eval(self, *, freeze_affine: bool = False) -> None:
        image_encoder = getattr(self.model, "image_encoder", None)
        if image_encoder is None:
            return
        for module in image_encoder.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
                if freeze_affine:
                    for param in module.parameters(recurse=False):
                        param.requires_grad = False

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

    def needs_action_chunk(self) -> bool:
        execute_horizon = max(1, min(int(self.config.execute_horizon), int(self.config.action_horizon)))
        return (
            self.current_chunk is None
            or self.step_in_chunk >= execute_horizon
            or self.step_in_chunk >= len(self.current_chunk)
        )

    def notify_intervention(self) -> None:
        self.reset_action_chunk()

    def select_action(self, obs, deterministic: bool = False) -> np.ndarray:
        if self.needs_action_chunk():
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

        action = np.asarray(self.current_chunk[self.step_in_chunk], dtype=np.float32)
        self.step_in_chunk += 1
        return action

    def update(self, batch: FlowDaggerBatch) -> dict[str, float]:
        batch = batch.to(self.device)
        self.model.train(True)
        if self._freeze_visual_batch_norm:
            self._set_visual_batch_norm_eval()

        noise = torch.randn_like(batch.action_sequences)
        timesteps = torch.rand(batch.batch_size, device=self.device)
        x_t = (
            (1.0 - timesteps).view(-1, 1, 1) * noise
            + timesteps.view(-1, 1, 1) * batch.action_sequences
        )
        v_target = batch.action_sequences - noise
        language = [self.language_instruction] * batch.batch_size

        self.optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(enabled=(self.device.type == "cuda"), device_type=self.device.type):
            v_pred = self.model(
                x_t=x_t.transpose(1, 2),
                t=timesteps,
                images=batch.image_obs,
                proprio=batch.proprio,
                language=language,
            ).transpose(1, 2)
            flow_loss = torch.mean((v_pred - v_target) ** 2)
            x1_pred = x_t + (1.0 - timesteps).view(-1, 1, 1) * v_pred
            endpoint_loss = torch.mean((x1_pred - batch.action_sequences) ** 2)
            if batch.action_sequences.shape[1] > 1:
                smooth_loss = torch.mean((x1_pred[:, 1:] - x1_pred[:, :-1]) ** 2)
            else:
                smooth_loss = torch.zeros((), device=self.device, dtype=batch.action_sequences.dtype)
            loss = (
                flow_loss
                + float(self.config.lambda_endpoint) * endpoint_loss
                + float(self.config.lambda_smooth) * smooth_loss
            )

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float(self.config.grad_clip_norm))
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return {
            "actor_loss": float(loss.detach().cpu().item()),
            "flow_loss": float(flow_loss.detach().cpu().item()),
            "endpoint_loss": float(endpoint_loss.detach().cpu().item()),
            "smooth_loss": float(smooth_loss.detach().cpu().item()),
            "mse": float(endpoint_loss.detach().cpu().item()),
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

    def load_model_state(self, state_dict: dict[str, Any], *, strict: bool = True) -> None:
        with self._state_lock:
            self.model.load_state_dict(state_dict, strict=strict)
        self.sync_inference_policy()
        self.reset_action_chunk()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        with self._state_lock:
            self.model.load_state_dict(state_dict["model"])
            optimizer_state = state_dict.get("optimizer")
            if optimizer_state is not None:
                try:
                    self.optimizer.load_state_dict(optimizer_state)
                except ValueError as exc:
                    print(f"[WARN] Skipping incompatible flow optimizer state: {exc}")
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
