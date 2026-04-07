from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


def _init_linear(linear: nn.Linear, zero_init: bool = False) -> nn.Linear:
    weight = linear.weight_orig if hasattr(linear, "weight_orig") else linear.weight
    if bool(zero_init):
        if hasattr(linear, "weight_orig"):
            nn.init.normal_(weight, mean=0.0, std=1e-4)
        else:
            nn.init.zeros_(weight)
    else:
        nn.init.xavier_uniform_(weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)
    return linear


def _build_linear(
    input_dim: int,
    output_dim: int,
    use_spectral_norm: bool = False,
    zero_init: bool = False,
) -> nn.Linear:
    linear = nn.Linear(int(input_dim), int(output_dim))
    if bool(use_spectral_norm):
        linear = spectral_norm(linear)
    return _init_linear(linear, zero_init=bool(zero_init))


class SwiGLU(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        self.value = _build_linear(
            input_dim=input_dim,
            output_dim=hidden_dim,
            use_spectral_norm=use_spectral_norm,
        )
        self.gate = _build_linear(
            input_dim=input_dim,
            output_dim=hidden_dim,
            use_spectral_norm=use_spectral_norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.value(x) * F.silu(self.gate(x))


class ResMLPBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        swiglu_hidden_dim: int,
        dropout: float,
        use_spectral_norm: bool = False,
        zero_init_residual: bool = True,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(hidden_dim))
        self.swiglu = SwiGLU(
            input_dim=int(hidden_dim),
            hidden_dim=int(swiglu_hidden_dim),
            use_spectral_norm=use_spectral_norm,
        )
        self.proj = _build_linear(
            input_dim=int(swiglu_hidden_dim),
            output_dim=int(hidden_dim),
            use_spectral_norm=use_spectral_norm,
            zero_init=bool(zero_init_residual),
        )
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(self.swiglu(self.norm(x)))
        return x + self.dropout(residual)


class ResMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_blocks: int,
        dropout: float,
        swiglu_hidden_ratio: float = 2.0 / 3.0,
        use_spectral_norm: bool = False,
        zero_init_residual: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.input_proj = (
            _build_linear(
                input_dim=input_dim,
                output_dim=self.hidden_dim,
                use_spectral_norm=use_spectral_norm,
            )
            if int(input_dim) != self.hidden_dim
            else nn.Identity()
        )
        swiglu_hidden_dim = max(1, int(round(self.hidden_dim * float(swiglu_hidden_ratio))))
        self.blocks = nn.ModuleList(
            [
                ResMLPBlock(
                    hidden_dim=self.hidden_dim,
                    swiglu_hidden_dim=swiglu_hidden_dim,
                    dropout=float(dropout),
                    use_spectral_norm=use_spectral_norm,
                    zero_init_residual=bool(zero_init_residual),
                )
                for _ in range(max(int(num_blocks), 0))
            ]
        )
        self.output_proj = _build_linear(
            input_dim=self.hidden_dim,
            output_dim=output_dim,
            use_spectral_norm=use_spectral_norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output_proj(x)


def build_res_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_blocks: int,
    dropout: float,
    swiglu_hidden_ratio: float = 2.0 / 3.0,
    use_spectral_norm: bool = False,
    zero_init_residual: bool = True,
) -> ResMLP:
    return ResMLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_blocks=num_blocks,
        dropout=dropout,
        swiglu_hidden_ratio=swiglu_hidden_ratio,
        use_spectral_norm=use_spectral_norm,
        zero_init_residual=zero_init_residual,
    )


def build_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    depth: int,
    dropout: float,
    swiglu_hidden_ratio: float = 2.0 / 3.0,
    use_spectral_norm: bool = False,
    zero_init_residual: bool = True,
) -> ResMLP:
    return build_res_mlp(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_blocks=max(int(depth) - 2, 0),
        dropout=dropout,
        swiglu_hidden_ratio=swiglu_hidden_ratio,
        use_spectral_norm=use_spectral_norm,
        zero_init_residual=zero_init_residual,
    )


class ActionSequenceEncoder(nn.Module):
    def __init__(
        self,
        action_dim: int,
        model_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        max_horizon: int,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.model_dim = int(model_dim)
        self.max_horizon = int(max_horizon)
        self.action_proj = nn.Linear(self.action_dim, self.model_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.model_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.max_horizon + 1, self.model_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=int(num_heads),
            dim_feedforward=int(hidden_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        self.norm = nn.LayerNorm(self.model_dim)

    def forward(self, action_sequence: torch.Tensor) -> torch.Tensor:
        if action_sequence.ndim != 3:
            raise ValueError(f"Expected action_sequence to be (B,H,A), got {action_sequence.shape}")
        batch_size, horizon, _ = action_sequence.shape
        if horizon > self.max_horizon:
            raise ValueError(f"horizon {horizon} exceeds max_horizon={self.max_horizon}")
        action_tokens = self.action_proj(action_sequence)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_token, action_tokens], dim=1)
        tokens = tokens + self.pos_embedding[:, : tokens.shape[1], :]
        encoded = self.encoder(tokens)
        return self.norm(encoded[:, 0])


class RunningScalarCalibrator(nn.Module):
    def __init__(self, momentum: float = 0.05, eps: float = 1e-6) -> None:
        super().__init__()
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.register_buffer("running_mean", torch.zeros(1))
        self.register_buffer("running_var", torch.ones(1))
        self.register_buffer("initialized", torch.zeros(1, dtype=torch.bool))

    def _reset_if_corrupted(self) -> None:
        if torch.isfinite(self.running_mean).all() and torch.isfinite(self.running_var).all():
            return
        self.running_mean.zero_()
        self.running_var.fill_(1.0)
        self.initialized.zero_()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        x = values.reshape(-1).float()
        finite_mask = torch.isfinite(x)
        finite_x = x[finite_mask]
        self._reset_if_corrupted()

        if self.training and finite_x.numel() > 0:
            batch_mean = finite_x.mean()
            batch_var = finite_x.var(unbiased=False) if finite_x.numel() > 1 else finite_x.new_ones(())
            if torch.isfinite(batch_mean) and torch.isfinite(batch_var):
                safe_mean = batch_mean.detach().view_as(self.running_mean)
                safe_var = batch_var.detach().view_as(self.running_var).clamp_min(self.eps)
                if not bool(self.initialized.item()):
                    self.running_mean.copy_(safe_mean)
                    self.running_var.copy_(safe_var)
                    self.initialized.fill_(True)
                else:
                    self.running_mean.mul_(1.0 - self.momentum).add_(safe_mean * self.momentum)
                    self.running_var.mul_(1.0 - self.momentum).add_(safe_var * self.momentum)

        safe_var = self.running_var.clamp_min(self.eps)
        safe_std = safe_var.sqrt()
        normalized = (x - self.running_mean) / safe_std
        evidence = torch.sigmoid(normalized)
        if not finite_mask.all():
            evidence = torch.where(finite_mask, evidence, torch.full_like(evidence, 0.5))
        evidence = torch.nan_to_num(evidence, nan=0.5, posinf=1.0, neginf=0.0)
        return evidence


@dataclass
class DynBCEForwardOutput:
    occ_logit: torch.Tensor
    judge_logit: torch.Tensor
    occ_private: torch.Tensor
    dyn_private_mean: torch.Tensor
    ensemble_mean: torch.Tensor
    ensemble_logvar: torch.Tensor
    ema_target: torch.Tensor
    occ_evidence: torch.Tensor
    dyn_evidence: torch.Tensor
    epi_evidence: torch.Tensor
    occ_prob: torch.Tensor
    dyn_residual: torch.Tensor
    epi_variance: torch.Tensor
    task_embedding_debug: torch.Tensor
    current_input_debug: torch.Tensor
    shared_latent_debug: torch.Tensor
    raw_occ_logit_debug: torch.Tensor
    action_feature_debug: torch.Tensor
    dyn_hidden_debug: torch.Tensor
    ensemble_mean_avg_debug: torch.Tensor


class DynBCEModel(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        num_tasks: int,
        shared_dim: int,
        occ_private_dim: int,
        dyn_private_dim: int,
        task_embed_dim: int,
        trunk_hidden_dim: int,
        head_hidden_dim: int,
        action_model_dim: int,
        action_num_layers: int,
        action_num_heads: int,
        action_dropout: float,
        max_action_horizon: int,
        ensemble_size: int,
        judge_hidden_dim: int,
        dyn_model_dim: int = 1536,
        dyn_backbone_num_blocks: int = 3,
        dyn_head_hidden_dim: int = 768,
        dyn_head_num_blocks: int = 1,
        trunk_num_blocks: int = 2,
        head_num_blocks: int = 1,
        judge_num_blocks: int = 2,
        swiglu_hidden_ratio: float = 2.0 / 3.0,
        occupancy_use_spectral_norm: bool = True,
        judge_use_spectral_norm: bool = True,
        occ_logit_scale: float = 8.0,
        occ_logit_temperature: float = 4.0,
        zero_init_residual: bool = True,
        occ_calibrator_momentum: float = 0.05,
        evidence_calibrator_eps: float = 1e-6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.num_tasks = int(num_tasks)
        self.shared_dim = int(shared_dim)
        self.occ_private_dim = int(occ_private_dim)
        self.dyn_private_dim = int(dyn_private_dim)
        self.target_dim = self.shared_dim + self.dyn_private_dim
        self.ensemble_size = int(ensemble_size)
        self.task_embedding = nn.Embedding(self.num_tasks, int(task_embed_dim))     # trainable soft-prompt
        self.input_norm = nn.LayerNorm(self.latent_dim)
        self.dyn_model_dim = int(dyn_model_dim)
        self.dyn_backbone_num_blocks = int(dyn_backbone_num_blocks)
        self.dyn_head_hidden_dim = int(dyn_head_hidden_dim)
        self.dyn_head_num_blocks = int(dyn_head_num_blocks)
        self.trunk_num_blocks = int(trunk_num_blocks)
        self.head_num_blocks = int(head_num_blocks)
        self.judge_num_blocks = int(judge_num_blocks)
        self.swiglu_hidden_ratio = float(swiglu_hidden_ratio)
        self.occupancy_use_spectral_norm = bool(occupancy_use_spectral_norm)
        self.judge_use_spectral_norm = bool(judge_use_spectral_norm)
        self.occ_logit_scale = float(occ_logit_scale)
        self.occ_logit_temperature = max(float(occ_logit_temperature), 1e-4)
        self.zero_init_residual = bool(zero_init_residual)

        trunk_input_dim = self.latent_dim + int(task_embed_dim)

        # light-weight shared encoder
        self.shared_encoder = nn.Sequential(
            nn.LayerNorm(trunk_input_dim),      # pre-norm
            build_res_mlp(
                input_dim=trunk_input_dim,
                hidden_dim=int(trunk_hidden_dim),
                output_dim=self.shared_dim,
                num_blocks=self.trunk_num_blocks,
                dropout=float(dropout),
                swiglu_hidden_ratio=self.swiglu_hidden_ratio,
                zero_init_residual=self.zero_init_residual,
            ),
            nn.LayerNorm(self.shared_dim),
        )

        # occupancy matching encoder
        self.occ_private_encoder = nn.Sequential(
            nn.LayerNorm(trunk_input_dim),
            build_res_mlp(
                input_dim=trunk_input_dim,
                hidden_dim=int(trunk_hidden_dim),
                output_dim=self.occ_private_dim,
                num_blocks=self.trunk_num_blocks,
                dropout=float(dropout),
                swiglu_hidden_ratio=self.swiglu_hidden_ratio,
                zero_init_residual=self.zero_init_residual,
            ),
            nn.LayerNorm(self.occ_private_dim),
        )

        # dynamic model encoder
        self.dyn_private_encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(trunk_input_dim),
                    build_res_mlp(
                        input_dim=trunk_input_dim,
                        hidden_dim=int(trunk_hidden_dim),
                        output_dim=self.dyn_private_dim,
                        num_blocks=self.trunk_num_blocks,
                        dropout=float(dropout),
                        swiglu_hidden_ratio=self.swiglu_hidden_ratio,
                        zero_init_residual=self.zero_init_residual,
                    ),
                    nn.LayerNorm(self.dyn_private_dim),
                )
                for _ in range(self.ensemble_size)
            ]
        )

        # ema settings
        self.ema_shared_encoder = copy.deepcopy(self.shared_encoder)
        self.ema_dyn_private_encoders = copy.deepcopy(self.dyn_private_encoders)
        for param in self.ema_shared_encoder.parameters():
            param.requires_grad = False
        for param in self.ema_dyn_private_encoders.parameters():
            param.requires_grad = False

        # ((s, a), a_t) encoder
        self.action_encoder = ActionSequenceEncoder(
            action_dim=self.action_dim,
            model_dim=int(action_model_dim),
            hidden_dim=int(head_hidden_dim),
            num_layers=int(action_num_layers),
            num_heads=int(action_num_heads),
            dropout=float(action_dropout),
            max_horizon=int(max_action_horizon),
        )

        # occupancy matching BCE head
        occ_input_dim = self.shared_dim + self.occ_private_dim + int(task_embed_dim)
        self.occupancy_head = nn.Sequential(
            nn.LayerNorm(occ_input_dim),
            build_res_mlp(
                input_dim=occ_input_dim,
                hidden_dim=int(head_hidden_dim),
                output_dim=1,
                num_blocks=self.head_num_blocks,
                dropout=float(dropout),
                swiglu_hidden_ratio=self.swiglu_hidden_ratio,
                use_spectral_norm=self.occupancy_use_spectral_norm,
                zero_init_residual=self.zero_init_residual,
            ),
        )

        # dynamic model head
        dyn_input_dim = self.shared_dim + self.dyn_private_dim + int(action_model_dim) + int(task_embed_dim)
        self.dyn_transition_input_projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(dyn_input_dim),
                    _build_linear(dyn_input_dim, self.dyn_model_dim),
                )
                for _ in range(self.ensemble_size)
            ]
        )
        self.dyn_transition_backbone = build_res_mlp(
            input_dim=self.dyn_model_dim,
            hidden_dim=self.dyn_model_dim,
            output_dim=self.dyn_model_dim,
            num_blocks=self.dyn_backbone_num_blocks,
            dropout=float(dropout),
            swiglu_hidden_ratio=self.swiglu_hidden_ratio,
            zero_init_residual=self.zero_init_residual,
        )
        self.dynamics_heads = nn.ModuleList(
            [
                build_res_mlp(
                    input_dim=self.dyn_model_dim,
                    hidden_dim=self.dyn_head_hidden_dim,
                    output_dim=int(self.target_dim * 2),
                    num_blocks=self.dyn_head_num_blocks,
                    dropout=float(dropout),
                    swiglu_hidden_ratio=self.swiglu_hidden_ratio,
                    zero_init_residual=self.zero_init_residual,
                )
                for _ in range(self.ensemble_size)
            ]
        )

        # final judge
        self.judge = nn.Sequential(
            nn.LayerNorm(3),
            build_res_mlp(
                input_dim=3,
                hidden_dim=int(judge_hidden_dim),
                output_dim=1,
                num_blocks=self.judge_num_blocks,
                dropout=float(dropout),
                swiglu_hidden_ratio=self.swiglu_hidden_ratio,
                use_spectral_norm=self.judge_use_spectral_norm,
                zero_init_residual=self.zero_init_residual,
            ),
        )

        self.occ_calibrator = RunningScalarCalibrator(
            momentum=float(occ_calibrator_momentum),
            eps=float(evidence_calibrator_eps),
        )
        self.dyn_calibrator = RunningScalarCalibrator(
            momentum=float(occ_calibrator_momentum),
            eps=float(evidence_calibrator_eps),
        )
        self.epi_calibrator = RunningScalarCalibrator(
            momentum=float(occ_calibrator_momentum),
            eps=float(evidence_calibrator_eps),
        )

    def _concat_task(self, latent: torch.Tensor, task_embedding: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.input_norm(latent), task_embedding], dim=-1)

    def _encode_ema_target(self, next_latent: torch.Tensor, task_embedding: torch.Tensor) -> torch.Tensor:
        ema_input = self._concat_task(next_latent, task_embedding)
        ema_shared = self.ema_shared_encoder(ema_input)
        ema_private_latents = [encoder(ema_input) for encoder in self.ema_dyn_private_encoders]
        ema_private_mean = torch.stack(ema_private_latents, dim=0).mean(dim=0)
        return torch.cat([ema_shared, ema_private_mean], dim=-1)

    def _update_ema_module(self, online_module: nn.Module, ema_module: nn.Module, decay: float) -> None:
        for online_param, ema_param in zip(online_module.parameters(), ema_module.parameters()):
            ema_param.data.mul_(float(decay)).add_(online_param.data * (1.0 - float(decay)))
        for online_buffer, ema_buffer in zip(online_module.buffers(), ema_module.buffers()):
            ema_buffer.data.copy_(online_buffer.data)

    def update_ema(self, decay: float) -> None:
        with torch.no_grad():
            self._update_ema_module(self.shared_encoder, self.ema_shared_encoder, decay=float(decay))
            for online_module, ema_module in zip(self.dyn_private_encoders, self.ema_dyn_private_encoders):
                self._update_ema_module(online_module, ema_module, decay=float(decay))

    def forward(
        self,
        current_latent: torch.Tensor,
        next_latent: torch.Tensor,
        action_sequence: torch.Tensor,
        task_index: torch.Tensor,
    ) -> DynBCEForwardOutput:
        task_embedding = self.task_embedding(task_index)
        current_input = self._concat_task(current_latent, task_embedding)
        shared_latent = self.shared_encoder(current_input)
        occ_private = self.occ_private_encoder(current_input)
        dyn_private = [encoder(current_input) for encoder in self.dyn_private_encoders]
        dyn_private_stack = torch.stack(dyn_private, dim=0)
        dyn_private_mean = dyn_private_stack.mean(dim=0)

        occ_input = torch.cat([shared_latent, occ_private, task_embedding], dim=-1)
        raw_occ_logit = self.occupancy_head(occ_input).squeeze(-1)
        occ_logit = self.occ_logit_scale * torch.tanh(raw_occ_logit / self.occ_logit_temperature)
        occ_positive_prob = torch.sigmoid(occ_logit)
        occ_prob = 1.0 - occ_positive_prob

        action_feature = self.action_encoder(action_sequence)
        ensemble_means = []
        ensemble_logvars = []
        dyn_hidden_debug = None
        for ensemble_idx, head in enumerate(self.dynamics_heads):
            dyn_input = torch.cat(
                [
                    shared_latent,
                    dyn_private_stack[ensemble_idx],
                    action_feature,
                    task_embedding,
                ],
                dim=-1,
            )
            dyn_hidden = self.dyn_transition_input_projs[ensemble_idx](dyn_input)
            dyn_hidden = self.dyn_transition_backbone(dyn_hidden)
            if dyn_hidden_debug is None:
                dyn_hidden_debug = dyn_hidden
            pred = head(dyn_hidden)
            mean, logvar = pred.chunk(2, dim=-1)
            ensemble_means.append(mean)
            ensemble_logvars.append(logvar.clamp(min=-8.0, max=6.0))
        ensemble_mean = torch.stack(ensemble_means, dim=0)
        ensemble_logvar = torch.stack(ensemble_logvars, dim=0)

        with torch.no_grad():
            ema_target = self._encode_ema_target(next_latent=next_latent, task_embedding=task_embedding)

        ensemble_mean_avg = ensemble_mean.mean(dim=0)
        dyn_residual = torch.sqrt(torch.mean((ensemble_mean_avg - ema_target) ** 2, dim=-1) + 1e-6)
        epi_variance = torch.mean(
            torch.var(ensemble_mean, dim=0, unbiased=False),
            dim=-1,
        )

        occ_evidence = self.occ_calibrator(occ_prob).detach()
        dyn_evidence = self.dyn_calibrator(dyn_residual).detach()
        epi_evidence = self.epi_calibrator(epi_variance).detach()

        judge_input = torch.stack([occ_evidence, dyn_evidence, epi_evidence], dim=-1)
        judge_logit = self.judge(judge_input).squeeze(-1)
        
        return DynBCEForwardOutput(
            occ_logit=occ_logit,
            judge_logit=judge_logit,
            occ_private=occ_private,
            dyn_private_mean=dyn_private_mean,
            ensemble_mean=ensemble_mean,
            ensemble_logvar=ensemble_logvar,
            ema_target=ema_target,
            occ_evidence=occ_evidence,
            dyn_evidence=dyn_evidence,
            epi_evidence=epi_evidence,
            occ_prob=occ_prob,
            dyn_residual=dyn_residual,
            epi_variance=epi_variance,
            task_embedding_debug=task_embedding,
            current_input_debug=current_input,
            shared_latent_debug=shared_latent,
            raw_occ_logit_debug=raw_occ_logit,
            action_feature_debug=action_feature,
            dyn_hidden_debug=dyn_hidden_debug if dyn_hidden_debug is not None else shared_latent,
            ensemble_mean_avg_debug=ensemble_mean_avg,
        )
