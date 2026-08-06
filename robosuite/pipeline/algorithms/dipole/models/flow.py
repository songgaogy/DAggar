from __future__ import annotations

import copy
import threading
from typing import Any, Optional

import numpy as np
import torch

from robosuite.pipeline.algorithms.flow_dagger.models.flow import _center_crop_resize
from robosuite.pipeline.common.utils import clone_array_tree
from robosuite.policy.flow_multi_update.model import MultiModalFlowPolicy, build_flow_policy

from ..common import DipoleBatch, DipoleConfig, select_dipole_batch


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.reshape(-1)
    values = values.reshape(-1)
    weight_sum = torch.sum(weights).clamp_min(1e-6)
    return torch.sum(values * weights) / weight_sum


@torch.inference_mode()
def _sample_guided_action_sequence(
    model_pos: MultiModalFlowPolicy,
    model_neg: MultiModalFlowPolicy | None,
    *,
    images: torch.Tensor,
    proprio: torch.Tensor,
    language: list[str],
    action_horizon: int,
    n_steps: int,
    omega: float,
    deterministic: bool,
) -> torch.Tensor:
    """CFG-style guided sampling over two independent flow policies.

    ``v = (1 + omega) * v_pos - omega * v_neg``. With ``omega == 0`` only the
    positive policy runs -- the negative policy is never encoded or forwarded,
    halving the per-ODE-step cost.
    """
    batch_size = proprio.shape[0]
    if deterministic:
        x = torch.zeros(batch_size, model_pos.action_dim, action_horizon, device=proprio.device, dtype=proprio.dtype)
    else:
        x = torch.randn(batch_size, model_pos.action_dim, action_horizon, device=proprio.device, dtype=proprio.dtype)

    context_pos = model_pos.encode_multimodal_context(images=images, proprio=proprio, language=language)
    use_guidance = float(omega) != 0.0 and model_neg is not None
    context_neg = (
        model_neg.encode_multimodal_context(images=images, proprio=proprio, language=language)
        if use_guidance
        else None
    )

    dt = 1.0 / float(n_steps)
    for step in range(int(n_steps)):
        t = torch.full(
            (batch_size,),
            float(step) / float(n_steps),
            device=proprio.device,
            dtype=proprio.dtype,
        )
        v_pos = model_pos.flow_head(
            x_t=x,
            timesteps=t,
            task_scene_cond=context_pos["task_scene_cond"],
            context_tokens=context_pos["context_tokens"],
            context_padding_mask=context_pos["context_padding_mask"],
        )
        if use_guidance:
            assert context_neg is not None
            v_neg = model_neg.flow_head(
                x_t=x,
                timesteps=t,
                task_scene_cond=context_neg["task_scene_cond"],
                context_tokens=context_neg["context_tokens"],
                context_padding_mask=context_neg["context_padding_mask"],
            )
            v = (1.0 + float(omega)) * v_pos - float(omega) * v_neg
        else:
            v = v_pos
        x = x + dt * v
    return x.transpose(1, 2)


class DipoleFlowPolicy:
    """Two independent, fully finetuned flow policies (positive + negative).

    Polarity is carried by two separate :class:`MultiModalFlowPolicy` instances
    (``model_pos``/``model_neg``), each trained full-tune under the base
    flow-policy freeze regime (built by ``build_flow_policy``: ResNet layer3/4
    trainable, stem/layer1/layer2 + CLIP frozen). The positive policy is trained
    with per-sample weight ``w_pos``, the negative with ``w_neg`` -- both the
    soft-weight (coupled) and hard-split (naive/neg_all) offline schemes reduce
    to this via :meth:`_compute_branch_weights`.

    Evaluation combines the two policies with CFG-style guidance
    ``v=(1+omega)v_pos-omega v_neg``; ``omega=0`` uses the positive policy only.
    """

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
        if self.device.type != "cuda" or self.inference_device.type != "cuda":
            raise RuntimeError(
                "DIPOLE policy training and inference require CUDA devices; "
                f"got training={self.device}, inference={self.inference_device}."
            )
        if not torch.cuda.is_available():
            raise RuntimeError("DIPOLE policy requires CUDA, but CUDA is unavailable.")
        self.language_instruction = str(config.language_instruction or config.task_name or "perform the task")

        def _build() -> MultiModalFlowPolicy:
            return build_flow_policy(
                self.model_cfg,
                proprio_dim=int(config.proprio_dim),
                action_dim=int(config.action_dim),
                camera_names=self.camera_names,
            ).to(self.device)

        # Two fully independent full-tune policies. The freeze regime (frozen CLIP
        # + ResNet stem/layer1/layer2, trainable layer3/layer4 + heads) is applied
        # inside build_flow_policy from model_cfg, so each instance is correctly
        # frozen on its own.
        self.model_pos = _build()
        self.model_neg = _build()

        def _optimizer(model: MultiModalFlowPolicy) -> torch.optim.Optimizer:
            return torch.optim.AdamW(
                [param for param in model.parameters() if param.requires_grad],
                lr=float(config.learning_rate),
                weight_decay=float(config.weight_decay),
            )

        self.optimizer_pos = _optimizer(self.model_pos)
        self.optimizer_neg = _optimizer(self.model_neg)
        self.scaler_pos = torch.amp.GradScaler(enabled=(self.device.type == "cuda"), device=self.device)
        self.scaler_neg = torch.amp.GradScaler(enabled=(self.device.type == "cuda"), device=self.device)

        # Per-model double-buffered inference copies.
        self.inference_model_pos = copy.deepcopy(self.model_pos).to(self.inference_device)
        self.inference_model_pos.eval()
        self._inference_shadow_model_pos = copy.deepcopy(self.inference_model_pos).to(self.inference_device)
        self._inference_shadow_model_pos.eval()
        self.inference_model_neg = copy.deepcopy(self.model_neg).to(self.inference_device)
        self.inference_model_neg.eval()
        self._inference_shadow_model_neg = copy.deepcopy(self.inference_model_neg).to(self.inference_device)
        self._inference_shadow_model_neg.eval()
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
        # Optional pluggable branch-weight policy (offline DIPOLE). When set it
        # fully overrides _compute_branch_weights; see modules/training/dipole/branch_weights.py.
        self.branch_weight_policy: Any = None

        self.sync_inference_policy()

    def set_g_provider(self, provider: Any) -> None:
        self.g_provider = provider

    def set_branch_weight_policy(self, policy: Any) -> None:
        """Attach a pluggable per-sample branch-weight policy (offline DIPOLE).

        ``policy`` is called as
        ``policy(batch, g_provider=..., sigmoid_fn=self._g_weights_from_raw,
        device=self.device, want_metrics=...) -> (w_pos, w_neg, metrics)`` and,
        when set, short-circuits the built-in coupled/neg_all weighting.
        """
        self.branch_weight_policy = policy

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

    def _prepare_inference_inputs(self, obs) -> tuple[torch.Tensor, torch.Tensor]:
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
        return image_tensor, proprio_tensor

    def select_action(self, obs, deterministic: bool = False, *, guided: bool = True) -> np.ndarray:
        # guided=True  -> two-policy omega guidance (eval): v=(1+w)v_pos - w v_neg
        # guided=False -> positive-policy-only rollout, single forward per ODE step
        omega = float(self.config.guidance_omega) if guided else 0.0
        if self.needs_action_chunk():
            image_tensor, proprio_tensor = self._prepare_inference_inputs(obs)

            with self._inference_lock:
                action_seq = _sample_guided_action_sequence(
                    self.inference_model_pos,
                    self.inference_model_neg,
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
        with the positive policy only.
        """
        omega = float(self.config.guidance_omega) if guided else 0.0
        image_tensor, proprio_tensor = self._prepare_inference_inputs(obs)

        with self._inference_lock:
            action_seq = _sample_guided_action_sequence(
                self.inference_model_pos,
                self.inference_model_neg,
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

    def _g_weights_from_raw(
        self, raw: torch.Tensor, *, want_metrics: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Map provider preference ``G`` to coupled branch weights.

        ``G`` is already the preference signal (larger -> more positive branch).
        Weights use an offset-then-scale logit::

            w_pos = sigmoid(beta * (G + k))
            w_neg = 1 - w_pos

        ``k`` shifts the decision threshold in G-space (``w_pos=0.5`` at ``G=-k``);
        ``beta`` scales the slope after that offset. No normalize / ``g_clip``.
        """
        g = raw
        logit = float(self.config.beta) * (g + float(self.config.k))
        w_pos = torch.sigmoid(logit)
        w_neg = 1.0 - w_pos
        if not want_metrics:
            return w_pos, w_neg, {}
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
        *,
        want_metrics: bool = True,
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
        metrics = {"neg_membership_mean": float(w_neg.mean().item())} if want_metrics else {}
        return w_pos, w_neg, metrics

    def _compute_branch_weights(
        self,
        batch: DipoleBatch,
        *,
        want_metrics: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Return routed branch weights and metrics with shape ``(B,)``.

        ``want_metrics=False`` skips every scalar readback so the training hot
        path incurs no GPU->CPU sync on non-logging steps.
        """
        if self.branch_weight_policy is not None:
            return self.branch_weight_policy(
                batch,
                g_provider=self.g_provider,
                sigmoid_fn=self._g_weights_from_raw,
                device=self.device,
                want_metrics=want_metrics,
            )
        if getattr(self.config, "branch_weight_mode", "coupled") == "neg_all":
            return self._neg_all_branch_weights(batch, want_metrics=want_metrics)

        if self.g_provider is None:
            raw = torch.zeros(batch.batch_size, dtype=torch.float32, device=self.device)
        else:
            with torch.no_grad():
                raw = self.g_provider.compute_g_for_batch(batch).to(self.device).reshape(-1)
        return self._g_weights_from_raw(raw, want_metrics=want_metrics)

    def _provider_raw_g_for_batch(self, batch: DipoleBatch) -> torch.Tensor:
        """Provider G per sample (before logit/sigmoid weighting)."""
        batch_size = batch.batch_size
        if self.g_provider is None:
            return torch.zeros(batch_size, dtype=torch.float32, device=self.device)

        # Offline routed weighting: only "advantage"-route rows have a meaningful
        # provider G; pos_only/neg_only rows are forced-weight and their advantage
        # (though precomputed) must not pollute the diagnostic G histogram.
        if self.branch_weight_policy is not None:
            routes = batch.metadata.get("route")
            raw = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            if routes is not None and len(routes) == batch_size:
                adv_rows = [i for i, r in enumerate(routes) if str(r) == "advantage"]
                if adv_rows:
                    adv_idx = torch.tensor(adv_rows, device=self.device, dtype=torch.long)
                    sub = select_dipole_batch(batch, adv_idx)
                    raw_adv = self.g_provider.compute_g_for_batch(sub).to(self.device).reshape(-1)
                    raw[adv_idx] = raw_adv
            else:
                raw = self.g_provider.compute_g_for_batch(batch).to(self.device).reshape(-1)
            return raw
        return self.g_provider.compute_g_for_batch(batch).to(self.device).reshape(-1)

    def update(
        self,
        batch: DipoleBatch,
        *,
        want_metrics: bool = True,
        collect_diagnostics: bool = False,
    ) -> dict[str, float]:
        batch = batch.to(self.device)
        want_metrics = bool(want_metrics or collect_diagnostics)

        B = batch.batch_size
        noise = torch.randn_like(batch.action_sequences)
        timesteps = torch.rand(B, device=self.device)
        x_t = (
            (1.0 - timesteps).view(-1, 1, 1) * noise
            + timesteps.view(-1, 1, 1) * batch.action_sequences
        )
        v_target = batch.action_sequences - noise
        language = [self.language_instruction] * B
        x_t_swapped = x_t.transpose(1, 2)

        w_pos, w_neg, weight_metrics = self._compute_branch_weights(batch, want_metrics=want_metrics)

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

        def _train_one(
            model: MultiModalFlowPolicy,
            optimizer: torch.optim.Optimizer,
            scaler: torch.amp.GradScaler,
            weights: torch.Tensor,
        ):
            model.train(True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(enabled=(self.device.type == "cuda"), device_type=self.device.type):
                v_pred = model(
                    x_t=x_t_swapped,
                    t=timesteps,
                    images=batch.image_obs,
                    proprio=batch.proprio,
                    language=language,
                ).transpose(1, 2)
                loss, flow, endpoint, smooth, fp = _branch_losses(v_pred, weights)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float(self.config.grad_clip_norm)
            )
            scaler.step(optimizer)
            scaler.update()
            return loss, flow, endpoint, smooth, fp, grad_norm

        # Two independent full-tune policies: the positive policy trains on w_pos,
        # the negative on w_neg. Both branches always run -- rows with weight 0
        # drop out of _weighted_mean, and keeping both forwards busy is what we
        # want for GPU utilization.
        loss_pos, flow_pos, endpoint_pos, smooth_pos, v_pos_mse, gnorm_pos = _train_one(
            self.model_pos, self.optimizer_pos, self.scaler_pos, w_pos
        )
        loss_neg, flow_neg, endpoint_neg, smooth_neg, v_neg_mse, gnorm_neg = _train_one(
            self.model_neg, self.optimizer_neg, self.scaler_neg, w_neg
        )
        loss = loss_pos + loss_neg

        if not want_metrics:
            return {}

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
            "grad_norm_pos": float(gnorm_pos.detach().cpu().item()),
            "grad_norm_neg": float(gnorm_neg.detach().cpu().item()),
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
            self._inference_shadow_model_pos.load_state_dict(self.model_pos.state_dict())
            self._inference_shadow_model_pos.eval()
            self._inference_shadow_model_neg.load_state_dict(self.model_neg.state_dict())
            self._inference_shadow_model_neg.eval()
        with self._inference_lock:
            self.inference_model_pos, self._inference_shadow_model_pos = (
                self._inference_shadow_model_pos,
                self.inference_model_pos,
            )
            self.inference_model_neg, self._inference_shadow_model_neg = (
                self._inference_shadow_model_neg,
                self.inference_model_neg,
            )

    def state_dict(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                "core_pos": {
                    "model": self.model_pos.state_dict(),
                    "optimizer": self.optimizer_pos.state_dict(),
                },
                "core_neg": {
                    "model": self.model_neg.state_dict(),
                    "optimizer": self.optimizer_neg.state_dict(),
                },
                "act_mean": None if self.act_mean is None else torch.as_tensor(self.act_mean),
                "act_std": None if self.act_std is None else torch.as_tensor(self.act_std),
                "prop_mean": None if self.prop_mean is None else torch.as_tensor(self.prop_mean),
                "prop_std": None if self.prop_std is None else torch.as_tensor(self.prop_std),
            }

    def load_model_state(self, state_dict: dict[str, Any], *, strict: bool = True) -> None:
        """Load one base flow-policy state dict into BOTH policies.

        Used for pretrained initialization: the positive and negative policies
        start from the same weights, so ``v_pos == v_neg`` at step 0 and CFG
        guidance is well-conditioned before the two diverge during training.
        """
        with self._state_lock:
            self.model_pos.load_state_dict(state_dict, strict=strict)
            self.model_neg.load_state_dict(state_dict, strict=strict)
        self.sync_inference_policy()
        self.reset_action_chunk()

    def load_dual_model_state(self, state_dict: dict[str, Any]) -> None:
        """Load trained positive/negative policies without optimizer state."""
        with self._state_lock:
            core_pos = state_dict["core_pos"]
            core_neg = state_dict["core_neg"]
            self.model_pos.load_state_dict(core_pos["model"])
            self.model_neg.load_state_dict(core_neg["model"])
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
]
