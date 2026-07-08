from __future__ import annotations

import copy
import threading
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from robosuite.pipeline.algorithms.flow_dagger.models.flow import _center_crop_resize
from robosuite.pipeline.common.utils import clone_array_tree
from robosuite.policy.flow_multi_update.model import MultiModalFlowPolicy, build_flow_policy

from ..common import DipoleBatch, DipoleConfig, select_dipole_batch
from .lora import (
    LoRARuntime,
    apply_lora,
    apply_lora_conv1d,
    default_selector,
    freeze_base_params,
    is_lora_param,
    lora_branch,
    lora_health_metrics,
    lora_masked,
    remap_legacy_cond_keys,
)


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.reshape(-1)
    values = values.reshape(-1)
    weight_sum = torch.sum(weights).clamp_min(1e-6)
    return torch.sum(values * weights) / weight_sum


class DipolePolarityFlowModel(MultiModalFlowPolicy):
    """Frozen backbone + dual LoRA adapters for DIPOLE CFG-style guidance.

    The shared backbone is frozen; polarity comes from two symmetric low-rank
    adapters on the condition-pathway ``nn.Linear`` modules of the flow head (and,
    optionally, the condition aggregator):

    - positive policy = ``base + pos_LoRA``
    - negative policy = ``base + neg_LoRA``

    Both adapters start at zero delta (``pos == neg == base``). Branch selection is
    carried by a shared :class:`LoRARuntime` so the existing single 2x-batch
    ``flow_head`` call can apply the pos adapter to the positive half and the neg
    adapter to the negative half via a row mask.
    """

    def __init__(
        self,
        *args,
        lora_rank: int = 16,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        lora_include_aggregator: bool = True,
        lora_include_conv: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._install_lora(
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            include_aggregator=lora_include_aggregator,
            include_conv=lora_include_conv,
        )

    def _install_lora(
        self,
        *,
        rank: int,
        alpha: float,
        dropout: float,
        include_aggregator: bool,
        include_conv: bool = True,
    ) -> None:
        self.lora_runtime = LoRARuntime()
        # (1) Condition-pathway Linears (FiLM/cross-attn/cond MLPs + aggregator).
        self.lora_num_modules = apply_lora(
            self.flow_head,
            self.condition_aggregator if include_aggregator else None,
            rank=int(rank),
            alpha=float(alpha),
            dropout=float(dropout),
            selector=default_selector(),
            runtime=self.lora_runtime,
        )
        # (2) UNet denoising Conv1d pathway (input_proj / conv1,conv2 / up-down
        # samples / output_conv). Without this the frozen convs cannot adapt and
        # only the conditioning can move -- insufficient capacity to fit the data.
        self.lora_num_conv_modules = (
            apply_lora_conv1d(
                self.flow_head,
                rank=int(rank),
                alpha=float(alpha),
                dropout=float(dropout),
                runtime=self.lora_runtime,
            )
            if include_conv
            else 0
        )
        # Freeze the whole backbone; only the two LoRA adapters train.
        self.lora_frozen_numel = freeze_base_params(self)

    def branch_task_scene_cond(
        self, context: dict[str, torch.Tensor], *, branch: str
    ) -> torch.Tensor:
        """Re-run the aggregator under the ``branch`` adapter to get its condition.

        ``encode_multimodal_context`` runs the aggregator base-only, so
        ``context['task_scene_cond']`` is the base condition. The aggregator is
        LoRA-wrapped and sits before the 2x split, so the per-row mask cannot reach
        it; instead we recompute each branch's condition with a cheap aggregator
        pass (once per inference, not per ODE step). Both the positive and negative
        branches diverge from base here.
        """
        with lora_branch(self.lora_runtime, branch):
            return self.condition_aggregator(
                fused_tokens=context["fused_tokens"],
                token_padding_mask=context["token_padding_mask"],
                language_global=context["language_global"],
            )

    def negative_task_scene_cond(self, context: dict[str, torch.Tensor]) -> torch.Tensor:
        """Backwards-compatible alias for ``branch_task_scene_cond(branch='neg')``."""
        return self.branch_task_scene_cond(context, branch="neg")

    def forward_from_context(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        context: dict[str, torch.Tensor],
        *,
        negative: bool,
    ) -> torch.Tensor:
        branch = "neg" if negative else "pos"
        with lora_branch(self.lora_runtime, branch):
            return self.flow_head(
                x_t=x_t,
                timesteps=t,
                task_scene_cond=context["task_scene_cond"],
                context_tokens=context["context_tokens"],
                context_padding_mask=context["context_padding_mask"],
            )


def build_dipole_flow_policy(
    cfg: Any,
    *,
    proprio_dim: int,
    action_dim: int,
    camera_names: list[str],
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float,
    lora_include_aggregator: bool,
    lora_include_conv: bool = True,
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
    polar._install_lora(
        rank=int(lora_rank),
        alpha=float(lora_alpha),
        dropout=float(lora_dropout),
        include_aggregator=bool(lora_include_aggregator),
        include_conv=bool(lora_include_conv),
    )
    return polar


@torch.inference_mode()
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
    # The aggregator is LoRA-wrapped, so ``context['task_scene_cond']`` is base-only.
    # The positive policy is ``base + pos_LoRA`` everywhere, so recompute the
    # positive condition under the pos adapter (used by both the omega==0 fast path
    # and the guided path below).
    context["task_scene_cond"] = model.branch_task_scene_cond(context, branch="pos")
    guided_context: dict[str, torch.Tensor] | None = None
    row_mask: torch.Tensor | None = None
    if float(omega) != 0.0:
        # Positive condition uses the pos adapter; recompute the negative condition
        # once via a neg-adapter aggregator pass (decision: train/eval consistent).
        pos_cond = context["task_scene_cond"]
        neg_cond = model.branch_task_scene_cond(context, branch="neg")
        guided_context = {
            "task_scene_cond": torch.cat([pos_cond, neg_cond], dim=0),
            "context_tokens": context["context_tokens"].repeat(2, 1, 1),
            "context_padding_mask": context["context_padding_mask"].repeat(2, 1),
        }
        # First half = positive (base + pos_LoRA), second half = negative (base + neg_LoRA).
        row_mask = torch.cat(
            [
                torch.zeros(batch_size, dtype=torch.bool, device=proprio.device),
                torch.ones(batch_size, dtype=torch.bool, device=proprio.device),
            ]
        )
    dt = 1.0 / float(n_steps)
    for step in range(int(n_steps)):
        t = torch.full(
            (batch_size,),
            float(step) / float(n_steps),
            device=proprio.device,
            dtype=proprio.dtype,
        )
        if float(omega) == 0.0:
            # omega=0 => v = v_pos exactly (positive policy = base + pos_LoRA); skip
            # the negative branch forward pass (halves the per-ODE-step network cost).
            # This is the online-rollout path (guided=False).
            v = model.forward_from_context(x_t=x, t=t, context=context, negative=False)
        else:
            assert guided_context is not None and row_mask is not None
            with lora_masked(model.lora_runtime, row_mask):
                velocities = model.flow_head(
                    x_t=x.repeat(2, 1, 1),
                    timesteps=t.repeat(2),
                    task_scene_cond=guided_context["task_scene_cond"],
                    context_tokens=guided_context["context_tokens"],
                    context_padding_mask=guided_context["context_padding_mask"],
                )
            v_pos = velocities[:batch_size]
            v_neg = velocities[batch_size:]
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
            lora_rank=int(config.lora_rank),
            lora_alpha=float(config.lora_alpha),
            lora_dropout=float(config.lora_dropout),
            lora_include_aggregator=bool(config.lora_include_aggregator),
            lora_include_conv=bool(config.lora_include_conv),
        ).to(self.device)

        # Frozen backbone + two symmetric adapters: the only trainable parameters are
        # the pos/neg LoRA adapters, both at adapter_lr. The positive loss updates
        # pos_LoRA, the negative loss updates neg_LoRA; base is frozen so there is no
        # gradient combination and base_lr_scale is obsolete.
        lora_params = [
            param
            for name, param in self.model.named_parameters()
            if param.requires_grad and is_lora_param(name)
        ]
        trainable_non_lora = [
            name
            for name, param in self.model.named_parameters()
            if param.requires_grad and not is_lora_param(name)
        ]
        assert not trainable_non_lora, (
            f"backbone must be frozen; found trainable non-LoRA params: {trainable_non_lora[:4]}"
        )
        self._lora_params = lora_params
        self.optimizer = torch.optim.AdamW(
            [
                {"params": lora_params, "lr": float(config.adapter_lr), "weight_decay": 0.0},
            ]
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

    def needs_action_chunk(self) -> bool:
        execute_horizon = max(
            1,
            min(int(self.config.execute_horizon), int(self.config.action_horizon)),
        )
        return bool(
            self.current_chunk is None
            or self.step_in_chunk >= execute_horizon
            or self.step_in_chunk >= len(self.current_chunk)
        )

    def planned_action_chunk(self) -> np.ndarray | None:
        if self.current_chunk is None:
            return None
        return np.asarray(self.current_chunk, dtype=np.float32).copy()

    def select_action(self, obs, deterministic: bool = False, *, guided: bool = True) -> np.ndarray:
        # guided=True  -> two-branch omega guidance (eval): v=(1+w)v_pos - w v_neg
        # guided=False -> positive-only rollout (base + pos_LoRA), single forward
        omega = float(self.config.guidance_omega) if guided else 0.0
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
                action_seq = _sample_guided_action_sequence(
                    inference_model,
                    images=image_tensor,
                    proprio=proprio_tensor,
                    language=[self.language_instruction],
                    action_horizon=int(self.config.action_horizon),
                    n_steps=int(self.config.n_ode_steps),
                    omega=float(omega),
                    deterministic=bool(deterministic),
                )[0].detach().cpu().numpy().astype(np.float32)
            if self.act_mean is not None and self.act_std is not None:
                action_seq = action_seq * self.act_std + self.act_mean
            self.current_chunk = action_seq
            self.step_in_chunk = 0

        action = np.asarray(self.current_chunk[self.step_in_chunk], dtype=np.float32)
        self.step_in_chunk += 1
        return action

    @torch.inference_mode()
    def plan_action_chunk(self, obs, deterministic: bool = False, *, guided: bool = True) -> np.ndarray:
        """Plan a fresh action chunk from ``obs`` without mutating ``current_chunk``.

        Mirrors the inference path inside :meth:`select_action` but returns the full
        un-normalized ``(horizon, action_dim)`` chunk, so callers (e.g. the live
        discriminator display) can feed it to a G provider. ``guided=False`` plans
        with the positive policy only (base + pos_LoRA), matching online rollout.
        """
        omega = float(self.config.guidance_omega) if guided else 0.0
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
                omega=float(omega),
                deterministic=bool(deterministic),
            )[0].detach().cpu().numpy().astype(np.float32)
        if self.act_mean is not None and self.act_std is not None:
            action_seq = action_seq * self.act_std + self.act_mean
        return action_seq

    @staticmethod
    def _resolve_demo_sample_mask(batch: DipoleBatch, device: torch.device) -> torch.Tensor | None:
        """True for rows sampled from demo_buffer (see ``buffer_sources`` metadata)."""
        sources = batch.metadata.get("buffer_sources")
        if sources is None or len(sources) != batch.batch_size:
            return None
        return torch.tensor(
            [str(source) == "demo_buffer" for source in sources],
            device=device,
            dtype=torch.bool,
        )

    def _g_weights_from_raw(self, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        # The G provider already returns the preference G (larger -> more positive branch)
        g = raw
        logit = float(self.config.beta) * g + float(self.config.k)
        w_pos = torch.sigmoid(logit)
        w_neg = 1.0 - w_pos
        n = int(raw.numel())
        metrics = {
            "raw_nnpu_score_mean": float(raw.mean().item()),
            "raw_nnpu_score_std": float(raw.std().item() if n > 1 else 0.0),
            "raw_nnpu_score_min": float(raw.min().item()),
            "raw_nnpu_score_max": float(raw.max().item()),
            "G_mean": float(g.mean().item()),
            "G_std": float(g.std().item() if n > 1 else 0.0),
            "logit_mean": float(logit.mean().item()),
            "logit_std": float(logit.std().item() if n > 1 else 0.0),
        }
        return w_pos, w_neg, metrics

    def _neg_all_branch_weights(
        self,
        batch: DipoleBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Decoupled hard-label weights for ``offline.mode == "neg_all"``.

        - ``w_pos = is_intervention`` (expert + success_rollout -> 1, fail -> 0):
          the positive branch trains on success data only.
        - ``w_neg`` = the attached provider's per-frame membership (offline_data
          rows -> 1, expert -> 0): the negative branch trains on ALL offline_data
          with weight 1.

        Unlike the coupled path, ``w_neg`` is neither ``1 - w_pos`` nor zeroed on
        intervention rows, so success frames drive BOTH branches
        (``w_pos = w_neg = 1``).
        """
        B = batch.batch_size
        w_pos = batch.is_intervention.to(self.device).float().reshape(-1)
        if self.g_provider is None:
            w_neg = torch.ones(B, dtype=torch.float32, device=self.device)
        else:
            raw = self.g_provider.compute_g_for_batch(batch).to(self.device).reshape(-1)
            w_neg = raw.clamp(0.0, 1.0)
        metrics = {"neg_membership_mean": float(w_neg.mean().item())}
        return w_pos, w_neg, metrics

    def _compute_branch_weights(
        self,
        batch: DipoleBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Return (w_pos, w_neg, metrics) tensors of shape (B,).

        When ``buffer_sources`` is set (1:1 online/demo training batch):
        - demo_buffer rows: ``w_pos=1``, ``w_neg=0``; G is not computed
        - online_buffer rows: raw G and sigmoid use online rows only
        """
        if getattr(self.config, "branch_weight_mode", "coupled") == "neg_all":
            return self._neg_all_branch_weights(batch)

        B = batch.batch_size
        demo_mask = self._resolve_demo_sample_mask(batch, self.device)

        if demo_mask is not None:
            w_pos = torch.ones(B, dtype=torch.float32, device=self.device)
            w_neg = torch.zeros(B, dtype=torch.float32, device=self.device)
            online_mask = ~demo_mask
            metrics: dict[str, float] = {
                "frac_demo_buffer": float(demo_mask.float().mean().item()),
                "w_pos_mean_demo": 1.0,
                "raw_nnpu_score_mean": 0.0,
                "raw_nnpu_score_std": 0.0,
                "raw_nnpu_score_min": 0.0,
                "raw_nnpu_score_max": 0.0,
                "G_mean": 0.0,
                "G_std": 0.0,
                "logit_mean": 0.0,
                "logit_std": 0.0,
            }
            if not bool(online_mask.any().item()):
                return w_pos, w_neg, metrics

            online_indices = torch.nonzero(online_mask, as_tuple=False).squeeze(1)
            if self.g_provider is None:
                zero_g = torch.zeros(int(online_indices.numel()), dtype=torch.float32, device=self.device)
                w_online_pos, w_online_neg, g_metrics = self._g_weights_from_raw(zero_g)
            else:
                with torch.no_grad():
                    online_batch = select_dipole_batch(batch, online_indices)
                    raw = self.g_provider.compute_g_for_batch(online_batch).to(self.device).reshape(-1)
                w_online_pos, w_online_neg, g_metrics = self._g_weights_from_raw(raw)

            w_pos[online_mask] = w_online_pos
            w_neg[online_mask] = w_online_neg
            metrics.update(g_metrics)
            metrics["w_pos_mean_online"] = float(w_online_pos.mean().item())
            metrics["w_neg_mean_online"] = float(w_online_neg.mean().item())
            return w_pos, w_neg, metrics

        # Legacy path: demo-only batch without buffer_sources (full-batch G + intervention mask).
        if self.g_provider is None:
            zero_g = torch.zeros(B, dtype=torch.float32, device=self.device)
            w_pos, w_neg, metrics = self._g_weights_from_raw(zero_g)
        else:
            with torch.no_grad():
                raw = self.g_provider.compute_g_for_batch(batch).to(self.device).reshape(-1)
            w_pos, w_neg, metrics = self._g_weights_from_raw(raw)

        is_int = batch.is_intervention.to(self.device).bool().reshape(-1)
        w_pos = torch.where(is_int, torch.ones_like(w_pos), w_pos)
        w_neg = torch.where(is_int, torch.zeros_like(w_neg), w_neg)
        return w_pos, w_neg, metrics

    def _provider_raw_g_for_batch(self, batch: DipoleBatch) -> torch.Tensor:
        """Provider G per sample (before logit/sigmoid weighting)."""
        batch_size = batch.batch_size
        if self.g_provider is None:
            return torch.zeros(batch_size, dtype=torch.float32, device=self.device)

        demo_mask = self._resolve_demo_sample_mask(batch, self.device)
        if demo_mask is not None:
            raw = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            online_mask = ~demo_mask
            if bool(online_mask.any().item()):
                online_indices = torch.nonzero(online_mask, as_tuple=False).squeeze(1)
                online_batch = select_dipole_batch(batch, online_indices)
                raw_online = self.g_provider.compute_g_for_batch(online_batch).to(self.device).reshape(-1)
                raw[online_mask] = raw_online
            return raw

        return self.g_provider.compute_g_for_batch(batch).to(self.device).reshape(-1)

    def update(self, batch: DipoleBatch, *, collect_diagnostics: bool = False) -> dict[str, float]:
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

        def _branch_losses(v_pred: torch.Tensor, weights: torch.Tensor):
            fp = torch.mean((v_pred - v_target) ** 2, dim=(1, 2))
            x1 = x_t + (1.0 - timesteps).view(-1, 1, 1) * v_pred
            ep = torch.mean((x1 - batch.action_sequences) ** 2, dim=(1, 2))
            if batch.action_sequences.shape[1] > 1:
                sm = torch.mean((x1[:, 1:] - x1[:, :-1]) ** 2, dim=(1, 2))
            else:
                sm = torch.zeros(B, device=self.device, dtype=fp.dtype)
            flow = _weighted_mean(fp, weights)
            endpoint = _weighted_mean(ep, weights)
            smooth = _weighted_mean(sm, weights)
            loss = (
                flow
                + float(self.config.lambda_endpoint) * endpoint
                + float(self.config.lambda_smooth) * smooth
            )
            return loss, flow, endpoint, smooth, fp

        x_t_swapped = x_t.transpose(1, 2)
        self.optimizer.zero_grad(set_to_none=True)

        # Shared frozen backbone: encode once. The aggregator is recomputed per branch
        # under its adapter (base task_scene_cond is unused).
        with torch.amp.autocast(enabled=(self.device.type == "cuda"), device_type=self.device.type):
            context = self.model.encode_multimodal_context(
                images=batch.image_obs,
                proprio=batch.proprio,
                language=language,
            )

        # Fused branch pass: false rows use pos LoRA, true rows use neg LoRA.
        with torch.amp.autocast(enabled=(self.device.type == "cuda"), device_type=self.device.type):
            pos_cond = self.model.branch_task_scene_cond(context, branch="pos")
            neg_cond = self.model.branch_task_scene_cond(context, branch="neg")
            row_mask = torch.cat(
                [
                    torch.zeros(B, dtype=torch.bool, device=self.device),
                    torch.ones(B, dtype=torch.bool, device=self.device),
                ],
                dim=0,
            )
            with lora_masked(self.model.lora_runtime, row_mask):
                velocities = self.model.flow_head(
                    x_t=x_t_swapped.repeat(2, 1, 1),
                    timesteps=timesteps.repeat(2),
                    task_scene_cond=torch.cat([pos_cond, neg_cond], dim=0),
                    context_tokens=context["context_tokens"].repeat(2, 1, 1),
                    context_padding_mask=context["context_padding_mask"].repeat(2, 1),
                ).transpose(1, 2)
            v_pos = velocities[:B]
            v_neg = velocities[B:]
            loss_pos, flow_pos, endpoint_pos, smooth_pos, v_pos_mse = _branch_losses(v_pos, w_pos)
            loss_neg, flow_neg, endpoint_neg, smooth_neg, v_neg_mse = _branch_losses(v_neg, w_neg)
            loss = loss_pos + loss_neg

        self.scaler.scale(loss).backward()

        # Base is frozen and pos/neg adapters are disjoint; the fused backward
        # populates each adapter's grads from its own branch rows.
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self._lora_params, max_norm=float(self.config.grad_clip_norm))
        self.scaler.step(self.optimizer)
        self.scaler.update()

        lora_metrics = lora_health_metrics(self.model)

        metrics: dict[str, Any] = {
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
            "flow_loss": float((flow_pos + flow_neg).detach().cpu().item() * 0.5),
            "endpoint_loss": float((endpoint_pos + endpoint_neg).detach().cpu().item() * 0.5),
            "smooth_loss": float((smooth_pos + smooth_neg).detach().cpu().item() * 0.5),
            "mse": float(((endpoint_pos + endpoint_neg) * 0.5).detach().cpu().item()),
            **lora_metrics,
            **weight_metrics,
        }
        if collect_diagnostics:
            with torch.no_grad():
                g_provider_raw = self._provider_raw_g_for_batch(batch)
            metrics["_diag"] = {
                "g_provider_raw": g_provider_raw.detach().cpu().numpy(),
                "w_pos": w_pos.detach().cpu().numpy(),
                "w_neg": w_neg.detach().cpu().numpy(),
                "v_pos_mse": v_pos_mse.detach().cpu().numpy(),
                "v_neg_mse": v_neg_mse.detach().cpu().numpy(),
            }
        return metrics

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
            state_dict = remap_legacy_cond_keys(state_dict, self.model)
            self.model.load_state_dict(state_dict, strict=strict)
        self.sync_inference_policy()
        self.reset_action_chunk()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        with self._state_lock:
            model_state = remap_legacy_cond_keys(state_dict["model"], self.model)
            self.model.load_state_dict(model_state)
            optimizer_state = state_dict.get("optimizer")
            if optimizer_state is not None:
                try:
                    self.optimizer.load_state_dict(optimizer_state)
                except (ValueError, KeyError) as exc:
                    # A legacy 2-group (base + LoRA) optimizer state does not match the
                    # new single LoRA group; skip it and start the optimizer fresh.
                    print(f"[dipole] skipping incompatible optimizer state: {exc}")
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


__all__ = [
    "DipoleFlowPolicy",
    "DipolePolarityFlowModel",
    "build_dipole_flow_policy",
]
