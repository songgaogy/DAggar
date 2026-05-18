from __future__ import annotations

import copy
import threading
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from robosuite.pipeline.algorithms.flow_dagger.models.flow import _center_crop_resize
from robosuite.pipeline.common.utils import clone_array_tree
from robosuite.policy.flow_multi.model import MultiModalFlowPolicy, build_flow_policy

from ..common import DipoleBatch, DipoleConfig


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.reshape(-1)
    values = values.reshape(-1)
    weight_sum = torch.sum(weights).clamp_min(1e-6)
    return torch.sum(values * weights) / weight_sum


class DipolePolarityFlowModel(MultiModalFlowPolicy):
    """Shared backbone + learnable polarity embedding for DIPOLE CFG-style branches.

    Index 0 = negative branch (low w_pos), Index 1 = positive branch (high w_pos).
    The polarity embedding is added to ``task_scene_cond`` before the flow head; the
    rest of the backbone is shared and trainable.
    """

    def __init__(self, *args, polarity_embedding_init: str = "small_gaussian", polarity_embedding_init_scale: float = 1e-3, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        cond_dim = int(self.condition_aggregator.output_dim)
        self.polarity_embedding = nn.Embedding(2, cond_dim)
        self._initialize_polarity_embedding(polarity_embedding_init, float(polarity_embedding_init_scale))

    def _initialize_polarity_embedding(self, scheme: str, scale: float) -> None:
        scheme = str(scheme).lower()
        with torch.no_grad():
            self.polarity_embedding.weight.zero_()
            if scheme == "zero_neg":
                # neg branch (row 0) zero, pos branch (row 1) small gaussian
                self.polarity_embedding.weight[1].normal_(mean=0.0, std=scale)
            elif scheme == "antipodal":
                e = torch.randn_like(self.polarity_embedding.weight[0]) * scale
                self.polarity_embedding.weight[0].copy_(-e)
                self.polarity_embedding.weight[1].copy_(e)
            else:
                # default small_gaussian: both rows iid N(0, scale)
                self.polarity_embedding.weight.normal_(mean=0.0, std=scale)

    def forward_from_context(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        context: dict[str, torch.Tensor],
        polarity_idx: int,
    ) -> torch.Tensor:
        cond = context["task_scene_cond"]
        polarity_vec = self.polarity_embedding.weight[int(polarity_idx)].unsqueeze(0)
        cond_polarized = cond + polarity_vec
        return self.flow_head(
            x_t=x_t,
            timesteps=t,
            task_scene_cond=cond_polarized,
            context_tokens=context["context_tokens"],
            context_padding_mask=context["context_padding_mask"],
        )


def build_dipole_flow_policy(
    cfg: Any,
    *,
    proprio_dim: int,
    action_dim: int,
    camera_names: list[str],
    polarity_embedding_init: str,
    polarity_embedding_init_scale: float,
) -> DipolePolarityFlowModel:
    base = build_flow_policy(cfg, proprio_dim=proprio_dim, action_dim=action_dim, camera_names=camera_names)
    # Re-instantiate as DipolePolarityFlowModel sharing the same submodules.
    polar = DipolePolarityFlowModel.__new__(DipolePolarityFlowModel)
    nn.Module.__init__(polar)
    polar.camera_names = list(base.camera_names)
    polar.action_dim = int(base.action_dim)
    polar.feature_dim = int(base.feature_dim)
    polar.image_encoder = base.image_encoder
    polar.proprio_tokenizer = base.proprio_tokenizer
    polar.language_encoder = base.language_encoder
    polar.language_guided_modulation = base.language_guided_modulation
    polar.fusion = base.fusion
    polar.condition_aggregator = base.condition_aggregator
    polar.flow_head = base.flow_head
    cond_dim = int(polar.condition_aggregator.output_dim)
    polar.polarity_embedding = nn.Embedding(2, cond_dim)
    polar._initialize_polarity_embedding(polarity_embedding_init, float(polarity_embedding_init_scale))
    return polar


@torch.no_grad()
def _sample_guided_action_sequence(
    model: DipolePolarityFlowModel,
    *,
    images: torch.Tensor,
    proprio: torch.Tensor,
    language: list[str],
    action_horizon: int,
    n_steps: int,
    omega: float,
    deterministic: bool,
) -> torch.Tensor:
    batch_size = proprio.shape[0]
    if deterministic:
        x = torch.zeros(batch_size, model.action_dim, action_horizon, device=proprio.device, dtype=proprio.dtype)
    else:
        x = torch.randn(batch_size, model.action_dim, action_horizon, device=proprio.device, dtype=proprio.dtype)
    context = model.encode_multimodal_context(images=images, proprio=proprio, language=language)
    dt = 1.0 / float(n_steps)
    for step in range(int(n_steps)):
        t = torch.full(
            (batch_size,),
            float(step) / float(n_steps),
            device=proprio.device,
            dtype=proprio.dtype,
        )
        v_pos = model.forward_from_context(x_t=x, t=t, context=context, polarity_idx=1)
        v_neg = model.forward_from_context(x_t=x, t=t, context=context, polarity_idx=0)
        v = (1.0 + float(omega)) * v_pos - float(omega) * v_neg
        x = x + dt * v
    return x.transpose(1, 2)


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

        self.model = build_dipole_flow_policy(
            self.model_cfg,
            proprio_dim=int(config.proprio_dim),
            action_dim=int(config.action_dim),
            camera_names=self.camera_names,
            polarity_embedding_init=str(config.polarity_embedding_init),
            polarity_embedding_init_scale=float(config.polarity_embedding_init_scale),
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
            [0.485, 0.456, 0.406], dtype=torch.float32, device=self.device,
        ).view(1, 1, 3, 1, 1)
        self._image_std = torch.tensor(
            [0.229, 0.224, 0.225], dtype=torch.float32, device=self.device,
        ).view(1, 1, 3, 1, 1)
        self._inference_image_mean = torch.tensor(
            [0.485, 0.456, 0.406], dtype=torch.float32, device=self.inference_device,
        ).view(1, 1, 3, 1, 1)
        self._inference_image_std = torch.tensor(
            [0.229, 0.224, 0.225], dtype=torch.float32, device=self.inference_device,
        ).view(1, 1, 3, 1, 1)

        # G-statistics state for running_zscore option.
        self._g_running_mean: float = 0.0
        self._g_running_var: float = 1.0
        self._g_running_count: int = 0

        # Optional G provider, injected after construction.
        self.g_provider: Any = None

        self.sync_inference_policy()

    def set_g_provider(self, provider: Any) -> None:
        self.g_provider = provider

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
        return (
            self.act_mean is not None
            and self.act_std is not None
            and self.prop_mean is not None
            and self.prop_std is not None
        )

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
                action_seq = _sample_guided_action_sequence(
                    inference_model,
                    images=image_tensor,
                    proprio=proprio_tensor,
                    language=[self.language_instruction],
                    action_horizon=int(self.config.action_horizon),
                    n_steps=int(self.config.n_ode_steps),
                    omega=float(self.config.guidance_omega),
                    deterministic=bool(deterministic),
                )[0].detach().cpu().numpy().astype(np.float32)
            if self.act_mean is not None and self.act_std is not None:
                action_seq = action_seq * self.act_std + self.act_mean
            self.current_chunk = action_seq
            self.step_in_chunk = 0

        action = np.asarray(self.current_chunk[self.step_in_chunk], dtype=np.float32)
        self.step_in_chunk += 1
        return action

    @torch.no_grad()
    def plan_action_chunk(self, obs, deterministic: bool = False) -> np.ndarray:
        """Plan a fresh action chunk from ``obs`` without mutating ``current_chunk``.

        Mirrors the inference path inside :meth:`select_action` but returns the full
        un-normalized ``(horizon, action_dim)`` chunk, so callers (e.g. the live
        discriminator display) can feed it to a G provider.
        """
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
            action_seq = _sample_guided_action_sequence(
                self.inference_model,
                images=image_tensor,
                proprio=proprio_tensor,
                language=[self.language_instruction],
                action_horizon=int(self.config.action_horizon),
                n_steps=int(self.config.n_ode_steps),
                omega=float(self.config.guidance_omega),
                deterministic=bool(deterministic),
            )[0].detach().cpu().numpy().astype(np.float32)
        if self.act_mean is not None and self.act_std is not None:
            action_seq = action_seq * self.act_std + self.act_mean
        return action_seq

    def _normalize_g(self, raw_g: torch.Tensor) -> torch.Tensor:
        mode = str(self.config.g_normalization).lower()
        if mode == "none":
            return raw_g
        if mode == "batch_zscore":
            return (raw_g - raw_g.mean()) / (raw_g.std() + 1e-6)
        if mode == "running_zscore":
            with torch.no_grad():
                batch_mean = float(raw_g.mean().item())
                batch_var = float(raw_g.var(unbiased=False).item())
                count = int(raw_g.numel())
                if self._g_running_count == 0:
                    self._g_running_mean = batch_mean
                    self._g_running_var = batch_var if batch_var > 0 else 1.0
                else:
                    momentum = 0.1
                    self._g_running_mean = (1 - momentum) * self._g_running_mean + momentum * batch_mean
                    self._g_running_var = (1 - momentum) * self._g_running_var + momentum * batch_var
                self._g_running_count += count
            mean_t = torch.tensor(self._g_running_mean, dtype=raw_g.dtype, device=raw_g.device)
            std_t = torch.tensor(self._g_running_var, dtype=raw_g.dtype, device=raw_g.device).clamp_min(1e-6).sqrt()
            return (raw_g - mean_t) / std_t
        if mode == "minmax":
            # symmetric clip to [-1, 1] using batch min/max
            lo = raw_g.min()
            hi = raw_g.max()
            span = (hi - lo).clamp_min(1e-6)
            return 2.0 * (raw_g - lo) / span - 1.0
        raise ValueError(f"Unknown g_normalization mode: {mode}")

    def _compute_branch_weights(
        self,
        batch: DipoleBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Return (w_pos, w_neg, metrics) tensors of shape (B,)."""
        B = batch.batch_size
        if self.g_provider is None:
            zero_g = torch.zeros(B, dtype=torch.float32, device=self.device)
            logit = (float(self.config.beta) * zero_g + float(self.config.k)).clamp(
                -float(self.config.g_clip), float(self.config.g_clip)
            )
            w_pos = torch.sigmoid(logit)
            w_neg = 1.0 - w_pos
            metrics = {
                "raw_lpb_score_mean": 0.0,
                "raw_lpb_score_std": 0.0,
                "raw_lpb_score_min": 0.0,
                "raw_lpb_score_max": 0.0,
                "G_mean": 0.0,
                "G_std": 0.0,
                "logit_mean": float(logit.mean().item()),
                "logit_std": float(logit.std().item() if B > 1 else 0.0),
            }
        else:
            with torch.no_grad():
                raw = self.g_provider.compute_g_for_batch(batch).to(self.device).reshape(-1)
                if str(self.config.g_sign).lower() == "negate_raw":
                    g = -raw
                else:
                    g = raw
                g_norm = self._normalize_g(g)
                logit = (float(self.config.beta) * g_norm + float(self.config.k)).clamp(
                    -float(self.config.g_clip), float(self.config.g_clip)
                )
                w_pos = torch.sigmoid(logit)
                w_neg = 1.0 - w_pos
                metrics = {
                    "raw_lpb_score_mean": float(raw.mean().item()),
                    "raw_lpb_score_std": float(raw.std().item() if B > 1 else 0.0),
                    "raw_lpb_score_min": float(raw.min().item()),
                    "raw_lpb_score_max": float(raw.max().item()),
                    "G_mean": float(g_norm.mean().item()),
                    "G_std": float(g_norm.std().item() if B > 1 else 0.0),
                    "logit_mean": float(logit.mean().item()),
                    "logit_std": float(logit.std().item() if B > 1 else 0.0),
                }

        # Explicit intervention mask: w_pos=1, w_neg=0 for intervention samples.
        is_int = batch.is_intervention.to(self.device).bool().reshape(-1)
        w_pos = torch.where(is_int, torch.ones_like(w_pos), w_pos)
        w_neg = torch.where(is_int, torch.zeros_like(w_neg), w_neg)
        return w_pos, w_neg, metrics

    def update(self, batch: DipoleBatch) -> dict[str, float]:
        batch = batch.to(self.device)
        self.model.train(True)

        B = batch.batch_size
        noise = torch.randn_like(batch.action_sequences)
        timesteps = torch.rand(B, device=self.device)
        x_t = (
            (1.0 - timesteps).view(-1, 1, 1) * noise
            + timesteps.view(-1, 1, 1) * batch.action_sequences
        )
        v_target = batch.action_sequences - noise
        language = [self.language_instruction] * B

        w_pos, w_neg, weight_metrics = self._compute_branch_weights(batch)

        self.optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(enabled=(self.device.type == "cuda"), device_type=self.device.type):
            context = self.model.encode_multimodal_context(
                images=batch.image_obs,
                proprio=batch.proprio,
                language=language,
            )
            x_t_swapped = x_t.transpose(1, 2)
            v_pos = self.model.forward_from_context(
                x_t=x_t_swapped, t=timesteps, context=context, polarity_idx=1,
            ).transpose(1, 2)
            v_neg = self.model.forward_from_context(
                x_t=x_t_swapped, t=timesteps, context=context, polarity_idx=0,
            ).transpose(1, 2)

            fp_pos = torch.mean((v_pos - v_target) ** 2, dim=(1, 2))
            fp_neg = torch.mean((v_neg - v_target) ** 2, dim=(1, 2))
            x1_pos = x_t + (1.0 - timesteps).view(-1, 1, 1) * v_pos
            x1_neg = x_t + (1.0 - timesteps).view(-1, 1, 1) * v_neg
            ep_pos = torch.mean((x1_pos - batch.action_sequences) ** 2, dim=(1, 2))
            ep_neg = torch.mean((x1_neg - batch.action_sequences) ** 2, dim=(1, 2))
            if batch.action_sequences.shape[1] > 1:
                sm_pos = torch.mean((x1_pos[:, 1:] - x1_pos[:, :-1]) ** 2, dim=(1, 2))
                sm_neg = torch.mean((x1_neg[:, 1:] - x1_neg[:, :-1]) ** 2, dim=(1, 2))
            else:
                sm_pos = torch.zeros(B, device=self.device, dtype=fp_pos.dtype)
                sm_neg = torch.zeros(B, device=self.device, dtype=fp_neg.dtype)

            flow_pos = _weighted_mean(fp_pos, w_pos)
            flow_neg = _weighted_mean(fp_neg, w_neg)
            endpoint_pos = _weighted_mean(ep_pos, w_pos)
            endpoint_neg = _weighted_mean(ep_neg, w_neg)
            smooth_pos = _weighted_mean(sm_pos, w_pos)
            smooth_neg = _weighted_mean(sm_neg, w_neg)

            loss_pos = (
                flow_pos
                + float(self.config.lambda_endpoint) * endpoint_pos
                + float(self.config.lambda_smooth) * smooth_pos
            )
            loss_neg = (
                flow_neg
                + float(self.config.lambda_endpoint) * endpoint_neg
                + float(self.config.lambda_smooth) * smooth_neg
            )
            loss = loss_pos + loss_neg

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float(self.config.grad_clip_norm))
        self.scaler.step(self.optimizer)
        self.scaler.update()

        pos_row = self.model.polarity_embedding.weight[1].detach()
        neg_row = self.model.polarity_embedding.weight[0].detach()
        polarity_pos_norm = float(torch.linalg.vector_norm(pos_row).item())
        polarity_neg_norm = float(torch.linalg.vector_norm(neg_row).item())
        cos_denom = (polarity_pos_norm * polarity_neg_norm) + 1e-6
        polarity_cos = float((pos_row * neg_row).sum().item() / cos_denom)

        return {
            "actor_loss": float(loss.detach().cpu().item()),
            "loss_pos": float(loss_pos.detach().cpu().item()),
            "loss_neg": float(loss_neg.detach().cpu().item()),
            "flow_loss_pos": float(flow_pos.detach().cpu().item()),
            "flow_loss_neg": float(flow_neg.detach().cpu().item()),
            "endpoint_loss_pos": float(endpoint_pos.detach().cpu().item()),
            "endpoint_loss_neg": float(endpoint_neg.detach().cpu().item()),
            "smooth_loss_pos": float(smooth_pos.detach().cpu().item()),
            "smooth_loss_neg": float(smooth_neg.detach().cpu().item()),
            "w_pos_mean": float(w_pos.mean().item()),
            "w_pos_std": float(w_pos.std().item() if B > 1 else 0.0),
            "w_neg_mean": float(w_neg.mean().item()),
            "w_neg_std": float(w_neg.std().item() if B > 1 else 0.0),
            "frac_w_pos_saturated_high": float((w_pos > 0.99).float().mean().item()),
            "frac_w_pos_saturated_low": float((w_pos < 0.01).float().mean().item()),
            "frac_intervention": float(batch.is_intervention.float().mean().item()),
            "polarity_embedding_pos_norm": polarity_pos_norm,
            "polarity_embedding_neg_norm": polarity_neg_norm,
            "polarity_embedding_cos": polarity_cos,
            "flow_loss": float((flow_pos + flow_neg).detach().cpu().item() * 0.5),
            "endpoint_loss": float((endpoint_pos + endpoint_neg).detach().cpu().item() * 0.5),
            "smooth_loss": float((smooth_pos + smooth_neg).detach().cpu().item() * 0.5),
            "mse": float(((endpoint_pos + endpoint_neg) * 0.5).detach().cpu().item()),
            **weight_metrics,
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
                "g_running_mean": float(self._g_running_mean),
                "g_running_var": float(self._g_running_var),
                "g_running_count": int(self._g_running_count),
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
                self.optimizer.load_state_dict(optimizer_state)
            self.set_normalizers(
                action_mean=state_dict.get("act_mean"),
                action_std=state_dict.get("act_std"),
                proprio_mean=state_dict.get("prop_mean"),
                proprio_std=state_dict.get("prop_std"),
            )
            self._g_running_mean = float(state_dict.get("g_running_mean", 0.0))
            self._g_running_var = float(state_dict.get("g_running_var", 1.0))
            self._g_running_count = int(state_dict.get("g_running_count", 0))
        self.sync_inference_policy()
        self.reset_action_chunk()

    def clone_observation(self, obs: Any) -> Any:
        return clone_array_tree(obs)


__all__ = [
    "DipoleFlowPolicy",
    "DipolePolarityFlowModel",
    "build_dipole_flow_policy",
]
