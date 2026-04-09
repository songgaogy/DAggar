"""Unified task-conditioned DSM model for LPB score transitions."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def _cfg_get(cfg: Any, key: str, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class UnifiedConditionedDSM(nn.Module):
    """Route state, actor, and dynamics denoising through one transformer."""

    TASK_STATE = 0
    TASK_ACTOR = 1
    TASK_DYNAMICS = 2

    def __init__(
        self,
        latent_dim: int,
        action_flat_dim: int,
        embed_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.action_flat_dim = int(action_flat_dim)
        self.embed_dim = int(embed_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.ffn_dim = int(ffn_dim)
        self.dropout = float(dropout)

        if self.embed_dim <= 0:
            raise ValueError("embed_dim must be positive.")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim must be divisible by num_heads, got embed_dim={self.embed_dim}, "
                f"num_heads={self.num_heads}"
            )

        # Shared projections require a common padded width per routed field.
        self.target_max_dim = max(self.latent_dim, self.action_flat_dim)
        self.context_max_dim = self.latent_dim + self.action_flat_dim

        # Each task id owns one learned routing token.
        self.task_embedding = nn.Embedding(3, self.embed_dim)
        self.type_embedding = nn.Embedding(2, self.embed_dim)
        self.context_proj = nn.Sequential(
            nn.Linear(self.context_max_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
        )
        self.target_proj = nn.Sequential(
            nn.Linear(self.target_max_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ffn_dim,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.num_layers,
        )
        self.output_norm = nn.LayerNorm(self.embed_dim)
        self.output_mean = nn.Linear(self.embed_dim, self.target_max_dim)

    def _task_target_dim(self, task_id: int) -> int:
        if int(task_id) == self.TASK_STATE:
            return self.latent_dim
        if int(task_id) == self.TASK_ACTOR:
            return self.action_flat_dim
        if int(task_id) == self.TASK_DYNAMICS:
            return self.latent_dim
        raise ValueError(f"Unsupported task_id: {task_id}")

    def _task_context_dim(self, task_id: int) -> int:
        if int(task_id) == self.TASK_STATE:
            return 0
        if int(task_id) == self.TASK_ACTOR:
            return self.latent_dim
        if int(task_id) == self.TASK_DYNAMICS:
            return self.latent_dim + self.action_flat_dim
        raise ValueError(f"Unsupported task_id: {task_id}")

    @staticmethod
    def _require_2d(name: str, x: torch.Tensor) -> None:
        if x.ndim != 2:
            raise ValueError(f"Expected {name} shape (B,D), got {tuple(x.shape)}")

    def _pad_feature(self, x: torch.Tensor, expected_dim: int, padded_dim: int, name: str) -> torch.Tensor:
        self._require_2d(name, x)
        if int(x.shape[1]) != int(expected_dim):
            raise ValueError(f"{name} dim mismatch: expected {expected_dim}, got {x.shape[1]}")
        if expected_dim > padded_dim:
            raise ValueError(f"{name} padded dim must be >= expected dim, got {padded_dim} < {expected_dim}")
        if expected_dim == padded_dim:
            return x
        # Zero padding keeps the shared linear layers task-agnostic.
        pad = x.new_zeros((x.shape[0], padded_dim - expected_dim))
        return torch.cat([x, pad], dim=-1)

    def forward_task(
        self,
        *,
        task_id: int,
        noisy_target: torch.Tensor,
        context: torch.Tensor,
        traj_type: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run one routed denoising task and slice outputs to its active width.
        
        For each task, the model builds a 3-token sequence:
        1. task token from learnable `nn.Embedding(3, embed_dim)`
        2. context token from the clean condition vector
        3. target token from the noisy target variable
        
        The target and context vectors are zero-padded to shared maximum dimensions before projection:
        - `target_max_dim = max(latent_dim, action_flat_dim)`
        - `context_max_dim = latent_dim + action_flat_dim`
        """
        target_dim = self._task_target_dim(task_id)
        context_dim = self._task_context_dim(task_id)
        
        target_padded = self._pad_feature(
            noisy_target,
            expected_dim=target_dim,
            padded_dim=self.target_max_dim,
            name="noisy_target",
        )
        context_padded = self._pad_feature(
            context,
            expected_dim=context_dim,
            padded_dim=self.context_max_dim,
            name="context",
        )

        batch_size = int(noisy_target.shape[0])
        if traj_type.ndim != 1 or int(traj_type.shape[0]) != batch_size:
            raise ValueError(f"Expected traj_type shape ({batch_size},), got {tuple(traj_type.shape)}")
        task_ids = torch.full(
            (batch_size,),
            int(task_id),
            dtype=torch.long,
            device=noisy_target.device,
        )

        task_token = (self.task_embedding(task_ids) + self.type_embedding(traj_type)).unsqueeze(1)
        context_token = self.context_proj(context_padded).unsqueeze(1)
        target_token = self.target_proj(target_padded).unsqueeze(1)

        # Token order fixes the routing layout seen by the encoder.
        tokens = torch.cat([task_token, context_token, target_token], dim=1)
        tokens = self.transformer(tokens)
        # Only the target token is decoded back to feature space.
        target_feature = self.output_norm(tokens[:, 2, :])

        pred_mean_full = self.output_mean(target_feature)
        return {
            "pred_mean": pred_mean_full[:, :target_dim],
            "target_feature": target_feature,
        }

    def forward(
        self,
        *,
        state_input: torch.Tensor,
        current_latent: torch.Tensor,
        action_input: torch.Tensor,
        action_clean: torch.Tensor,
        next_state_input: torch.Tensor,
        traj_type: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run the three denoising factors with shared backbone parameters.
        
        NOTE: you can regard this as a unified world action model, modeling all three factors with shared backbone parameters.
        """
        batch_size = int(state_input.shape[0])
        state_context = state_input.new_zeros((batch_size, 0))
        # The dynamics task conditions on clean [z_t, a].
        dynamics_context = torch.cat([current_latent, action_clean], dim=-1)

        state_out = self.forward_task(
            task_id=self.TASK_STATE,
            noisy_target=state_input,
            context=state_context,
            traj_type=traj_type,
        )
        action_out = self.forward_task(
            task_id=self.TASK_ACTOR,
            noisy_target=action_input,
            context=current_latent,
            traj_type=traj_type,
        )
        dynamics_out = self.forward_task(
            task_id=self.TASK_DYNAMICS,
            noisy_target=next_state_input,
            context=dynamics_context,
            traj_type=traj_type,
        )
        return {
            "state_hat": state_out["pred_mean"],
            "action_hat": action_out["pred_mean"],
            "next_state_hat": dynamics_out["pred_mean"],
        }


ConditionalManifoldDenoiser = UnifiedConditionedDSM
JointManifoldDenoiser = UnifiedConditionedDSM


class DSMModel(nn.Module):
    """Normalize transitions, inject DSM noise, and compute reconstruction energies."""

    def __init__(
        self,
        predictor: UnifiedConditionedDSM,
        latent_dim: int,
        action_dim: int,
        transition_horizon: int,
        noise_scale: float,
        std_clamp_min: float,
    ) -> None:
        super().__init__()
        self.predictor = predictor
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.transition_horizon = int(transition_horizon)
        self.noise_scale = float(noise_scale)
        self.noise_sigma = self.noise_scale
        self.std_clamp_min = float(std_clamp_min)
        self.action_flat_dim = int(self.action_dim * self.transition_horizon)
        self.tau_dim = int(2 * self.latent_dim + self.action_flat_dim)

        self.state_slice = slice(0, self.latent_dim)
        self.action_slice = slice(self.latent_dim, self.latent_dim + self.action_flat_dim)
        self.next_state_slice = slice(self.latent_dim + self.action_flat_dim, self.tau_dim)

        self.register_buffer("latent_mean", torch.zeros(self.latent_dim, dtype=torch.float32))
        self.register_buffer("latent_var", torch.ones(self.latent_dim, dtype=torch.float32))
        self.register_buffer("action_mean", torch.zeros(self.action_dim, dtype=torch.float32))
        self.register_buffer("action_var", torch.ones(self.action_dim, dtype=torch.float32))

        if int(self.predictor.latent_dim) != self.latent_dim:
            raise ValueError(
                f"predictor latent_dim mismatch: expected {self.latent_dim}, got {self.predictor.latent_dim}"
            )
        if int(self.predictor.action_flat_dim) != self.action_flat_dim:
            raise ValueError(
                "predictor action_flat_dim mismatch: "
                f"expected {self.action_flat_dim}, got {self.predictor.action_flat_dim}"
            )

    def set_normalization_stats(
        self,
        *,
        latent_mean: torch.Tensor,
        latent_var: torch.Tensor,
        action_mean: torch.Tensor,
        action_var: torch.Tensor,
        min_variance: float = 1e-6,
    ) -> None:
        """Load dataset statistics used for latent and action standardization."""
        latent_mean = torch.as_tensor(latent_mean, dtype=torch.float32, device=self.latent_mean.device).reshape(-1)
        latent_var = torch.as_tensor(latent_var, dtype=torch.float32, device=self.latent_var.device).reshape(-1)
        action_mean = torch.as_tensor(action_mean, dtype=torch.float32, device=self.action_mean.device).reshape(-1)
        action_var = torch.as_tensor(action_var, dtype=torch.float32, device=self.action_var.device).reshape(-1)
        if latent_mean.shape[0] != self.latent_dim or latent_var.shape[0] != self.latent_dim:
            raise ValueError(
                f"Latent stats must have shape ({self.latent_dim},), "
                f"got mean={tuple(latent_mean.shape)} var={tuple(latent_var.shape)}"
            )
        if action_mean.shape[0] != self.action_dim or action_var.shape[0] != self.action_dim:
            raise ValueError(
                f"Action stats must have shape ({self.action_dim},), "
                f"got mean={tuple(action_mean.shape)} var={tuple(action_var.shape)}"
            )
        self.latent_mean.copy_(latent_mean)
        self.latent_var.copy_(torch.clamp(latent_var, min=float(min_variance)))
        self.action_mean.copy_(action_mean)
        self.action_var.copy_(torch.clamp(action_var, min=float(min_variance)))

    @property
    def action_var_flat(self) -> torch.Tensor:
        return self.action_var.repeat(self.transition_horizon)

    @property
    def action_mean_flat(self) -> torch.Tensor:
        return self.action_mean.repeat(self.transition_horizon)

    @property
    def latent_std(self) -> torch.Tensor:
        return torch.clamp(torch.sqrt(self.latent_var), min=float(self.std_clamp_min))

    @property
    def action_std_flat(self) -> torch.Tensor:
        return torch.clamp(torch.sqrt(self.action_var_flat), min=float(self.std_clamp_min))

    @property
    def tau_noise_scale(self) -> torch.Tensor:
        return torch.ones(self.tau_dim, dtype=self.latent_var.dtype, device=self.latent_var.device)

    def normalize_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Standardize latent features with clamped positive-pool statistics."""
        latent_mean = self.latent_mean.to(device=latent.device, dtype=latent.dtype)
        latent_std = self.latent_std.to(device=latent.device, dtype=latent.dtype)
        return (latent - latent_mean.unsqueeze(0)) / latent_std.unsqueeze(0)

    def normalize_action_flat(self, action_flat: torch.Tensor) -> torch.Tensor:
        """Standardize flattened action chunks with repeated action statistics."""
        action_mean = self.action_mean_flat.to(device=action_flat.device, dtype=action_flat.dtype)
        action_std = self.action_std_flat.to(device=action_flat.device, dtype=action_flat.dtype)
        return (action_flat - action_mean.unsqueeze(0)) / action_std.unsqueeze(0)

    def flatten_action_sequence(self, action_sequence: torch.Tensor) -> torch.Tensor:
        """Reshape (B, H, A) actions into the routed flat actor target."""
        if action_sequence.ndim != 3:
            raise ValueError(f"Expected action_sequence shape (B,H,A), got {tuple(action_sequence.shape)}")
        if int(action_sequence.shape[1]) != self.transition_horizon:
            raise ValueError(
                "action_sequence horizon mismatch: "
                f"expected {self.transition_horizon}, got {action_sequence.shape[1]}"
            )
        if int(action_sequence.shape[2]) != self.action_dim:
            raise ValueError(
                f"action_sequence dim mismatch: expected {self.action_dim}, got {action_sequence.shape[2]}"
            )
        return action_sequence.reshape(action_sequence.shape[0], self.action_flat_dim)

    def build_tau(
        self,
        current_latent: torch.Tensor,
        action_sequence: torch.Tensor,
        target_latent: torch.Tensor,
    ) -> torch.Tensor:
        """Pack the raw transition tuple into the legacy flat tau layout."""
        if current_latent.ndim != 2:
            raise ValueError(f"Expected current_latent shape (B,D), got {tuple(current_latent.shape)}")
        if action_sequence.ndim != 3:
            raise ValueError(f"Expected action_sequence shape (B,H,A), got {tuple(action_sequence.shape)}")
        if target_latent.ndim != 2:
            raise ValueError(f"Expected target_latent shape (B,D), got {tuple(target_latent.shape)}")
        if int(current_latent.shape[1]) != self.latent_dim:
            raise ValueError(
                f"current_latent dim mismatch: expected {self.latent_dim}, got {current_latent.shape[1]}"
            )
        if int(target_latent.shape[1]) != self.latent_dim:
            raise ValueError(
                f"target_latent dim mismatch: expected {self.latent_dim}, got {target_latent.shape[1]}"
            )
        if int(action_sequence.shape[1]) != self.transition_horizon:
            raise ValueError(
                "action_sequence horizon mismatch: "
                f"expected {self.transition_horizon}, got {action_sequence.shape[1]}"
            )
        if int(action_sequence.shape[2]) != self.action_dim:
            raise ValueError(
                f"action_sequence dim mismatch: expected {self.action_dim}, got {action_sequence.shape[2]}"
            )
        return torch.cat(
            [
                current_latent,
                self.flatten_action_sequence(action_sequence),
                target_latent,
            ],
            dim=-1,
        )

    def split_tau(self, tau: torch.Tensor) -> dict[str, torch.Tensor]:
        """Split a flat tau vector into state, action, and next-state blocks."""
        if tau.ndim != 2 or int(tau.shape[1]) != self.tau_dim:
            raise ValueError(f"Expected tau shape (B,{self.tau_dim}), got {tuple(tau.shape)}")
        return {
            "state": tau[:, self.state_slice],
            "action": tau[:, self.action_slice],
            "next_state": tau[:, self.next_state_slice],
        }

    def add_noise(
        self,
        *,
        state_clean: torch.Tensor,
        action_clean: torch.Tensor,
        next_state_clean: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Add isotropic Gaussian noise to normalized task targets."""
        sigma = float(self.noise_scale)
        state_noise = torch.randn_like(state_clean) * sigma
        action_noise = torch.randn_like(action_clean) * sigma
        next_state_noise = torch.randn_like(next_state_clean) * sigma
        return {
            "state_input": state_clean + state_noise,
            "state_noise": state_noise,
            "action_input": action_clean + action_noise,
            "action_noise": action_noise,
            "next_state_input": next_state_clean + next_state_noise,
            "next_state_noise": next_state_noise,
        }

    @staticmethod
    def _require_traj_type(batch_size: int, traj_type: torch.Tensor | int, device: torch.device) -> torch.Tensor:
        if isinstance(traj_type, int):
            return torch.full((batch_size,), int(traj_type), dtype=torch.long, device=device)
        traj_type_tensor = torch.as_tensor(traj_type, dtype=torch.long, device=device).reshape(-1)
        if int(traj_type_tensor.shape[0]) != int(batch_size):
            raise ValueError(f"Expected traj_type length {batch_size}, got {traj_type_tensor.shape[0]}")
        if torch.any((traj_type_tensor < 0) | (traj_type_tensor > 1)):
            raise ValueError("traj_type must contain only 0 (positive) or 1 (negative).")
        return traj_type_tensor

    def reconstruction_components(
        self,
        *,
        state_clean: torch.Tensor,
        state_hat: torch.Tensor,
        action_clean: torch.Tensor,
        action_hat: torch.Tensor,
        next_state_clean: torch.Tensor,
        next_state_hat: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute per-task MSE diagnostics."""
        state_sq = torch.square(state_hat - state_clean)
        action_sq = torch.square(action_hat - action_clean)
        next_state_sq = torch.square(next_state_hat - next_state_clean)

        error_sq = torch.cat([state_sq, action_sq, next_state_sq], dim=-1)
        state_energy = state_sq.mean(dim=-1)
        action_energy = action_sq.mean(dim=-1)
        next_state_energy = next_state_sq.mean(dim=-1)
        return {
            "error_sq": error_sq,
            "tau_mse_per_sample": error_sq.mean(dim=-1),
            "tau_sse_per_sample": error_sq.sum(dim=-1),
            "state_mse_per_sample": state_sq.mean(dim=-1),
            "action_mse_per_sample": action_sq.mean(dim=-1),
            "next_state_mse_per_sample": next_state_sq.mean(dim=-1),
            "state_sse_per_sample": state_sq.sum(dim=-1),
            "action_sse_per_sample": action_sq.sum(dim=-1),
            "next_state_sse_per_sample": next_state_sq.sum(dim=-1),
            "state_energy_per_sample": state_energy,
            "action_energy_per_sample": action_energy,
            "next_state_energy_per_sample": next_state_energy,
            "score_per_sample": state_energy + action_energy + next_state_energy,
        }

    def fisher_components(
        self,
        *,
        state_pos: torch.Tensor,
        state_neg: torch.Tensor,
        action_pos: torch.Tensor,
        action_neg: torch.Tensor,
        next_state_pos: torch.Tensor,
        next_state_neg: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute Fisher-style L2 energies from dual conditional predictions."""
        state_sq = torch.square(state_pos - state_neg)
        action_sq = torch.square(action_pos - action_neg)
        next_state_sq = torch.square(next_state_pos - next_state_neg)

        state_energy = state_sq.sum(dim=-1)
        action_energy = action_sq.sum(dim=-1)
        next_state_energy = next_state_sq.sum(dim=-1)
        return {
            "state_error_per_sample": state_energy,
            "action_error_per_sample": action_energy,
            "next_state_error_per_sample": next_state_energy,
            "score_per_sample": state_energy + action_energy + next_state_energy,
        }

    def forward(
        self,
        current_latent: torch.Tensor,
        action_sequence: torch.Tensor,
        target_latent: torch.Tensor,
        *,
        traj_type: torch.Tensor | int,
        add_noise: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Build normalized task targets and run the unified denoiser."""
        state_clean_raw = current_latent
        action_clean_raw = self.flatten_action_sequence(action_sequence)
        next_state_clean_raw = target_latent

        # Standardization is shared across train and inference.
        state_clean = self.normalize_latent(state_clean_raw)
        action_clean = self.normalize_action_flat(action_clean_raw)
        next_state_norm_clean = self.normalize_latent(next_state_clean_raw)
        # The dynamics branch reconstructs the normalized residual.
        delta_clean = next_state_norm_clean - state_clean

        # Noise is injected only in normalized space.
        noisy = {
            "state_input": state_clean,
            "state_noise": torch.zeros_like(state_clean),
            "action_input": action_clean,
            "action_noise": torch.zeros_like(action_clean),
            "next_state_input": delta_clean,
            "next_state_noise": torch.zeros_like(delta_clean),
        }
        if add_noise:
            noisy = self.add_noise(
                state_clean=state_clean,
                action_clean=action_clean,
                next_state_clean=delta_clean,
            )

        traj_type_tensor = self._require_traj_type(
            batch_size=int(current_latent.shape[0]),
            traj_type=traj_type,
            device=current_latent.device,
        )

        preds = self.predictor(
            state_input=noisy["state_input"],
            current_latent=state_clean,
            action_input=noisy["action_input"],
            action_clean=action_clean,
            next_state_input=noisy["next_state_input"],
            traj_type=traj_type_tensor,
        )
        return {
            "current_latent": current_latent,
            "traj_type": traj_type_tensor,
            "state_clean": state_clean,
            "state_input": noisy["state_input"],
            "state_hat": preds["state_hat"],
            "state_noise": noisy["state_noise"],
            "action_clean": action_clean,
            "action_input": noisy["action_input"],
            "action_hat": preds["action_hat"],
            "action_noise": noisy["action_noise"],
            "next_state_clean": delta_clean,
            "next_state_norm_clean": next_state_norm_clean,
            "next_state_input": noisy["next_state_input"],
            "next_state_hat": preds["next_state_hat"],
            "next_state_noise": noisy["next_state_noise"],
            "delta_clean": delta_clean,
            "delta_hat": preds["next_state_hat"],
        }

    def compute_dsm_loss(
        self,
        current_latent: torch.Tensor,
        action_sequence: torch.Tensor,
        target_latent: torch.Tensor,
        traj_type: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run noisy denoising and return aggregate training statistics."""
        out = self.forward(
            current_latent=current_latent,
            action_sequence=action_sequence,
            target_latent=target_latent,
            traj_type=traj_type,
            add_noise=True,
        )
        recon = self.reconstruction_components(
            state_clean=out["state_clean"],
            state_hat=out["state_hat"],
            action_clean=out["action_clean"],
            action_hat=out["action_hat"],
            next_state_clean=out["next_state_clean"],
            next_state_hat=out["next_state_hat"],
        )
        loss = recon["score_per_sample"].mean()
        return {
            "loss": loss,
            "score": recon["score_per_sample"].mean(),
            "tau_mse": recon["tau_mse_per_sample"].mean(),
            "state_mse": recon["state_mse_per_sample"].mean(),
            "action_mse": recon["action_mse_per_sample"].mean(),
            "next_state_mse": recon["next_state_mse_per_sample"].mean(),
            "state_energy": recon["state_energy_per_sample"].mean(),
            "action_energy": recon["action_energy_per_sample"].mean(),
            "next_state_energy": recon["next_state_energy_per_sample"].mean(),
            "current_latent": out["current_latent"],
            "traj_type": out["traj_type"],
            "state_clean": out["state_clean"],
            "state_input": out["state_input"],
            "state_hat": out["state_hat"],
            "action_clean": out["action_clean"],
            "action_input": out["action_input"],
            "action_hat": out["action_hat"],
            "next_state_clean": out["next_state_clean"],
            "next_state_input": out["next_state_input"],
            "next_state_hat": out["next_state_hat"],
        }

    def compute_fisher_score(
        self,
        *,
        current_latent: torch.Tensor,
        action_sequence: torch.Tensor,
        target_latent: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run dual conditional inference and return Fisher L2 energies."""
        pos_out = self.forward(
            current_latent=current_latent,
            action_sequence=action_sequence,
            target_latent=target_latent,
            traj_type=0,
            add_noise=False,
        )
        neg_out = self.forward(
            current_latent=current_latent,
            action_sequence=action_sequence,
            target_latent=target_latent,
            traj_type=1,
            add_noise=False,
        )
        fisher = self.fisher_components(
            state_pos=pos_out["state_hat"],
            state_neg=neg_out["state_hat"],
            action_pos=pos_out["action_hat"],
            action_neg=neg_out["action_hat"],
            next_state_pos=pos_out["next_state_hat"],
            next_state_neg=neg_out["next_state_hat"],
        )
        return {
            "score_per_sample": fisher["score_per_sample"],
            "state_error_per_sample": fisher["state_error_per_sample"],
            "action_error_per_sample": fisher["action_error_per_sample"],
            "next_state_error_per_sample": fisher["next_state_error_per_sample"],
            "state_pos_hat": pos_out["state_hat"],
            "state_neg_hat": neg_out["state_hat"],
            "action_pos_hat": pos_out["action_hat"],
            "action_neg_hat": neg_out["action_hat"],
            "next_state_pos_hat": pos_out["next_state_hat"],
            "next_state_neg_hat": neg_out["next_state_hat"],
        }


def build_unified_conditioned_dsm(
    *,
    latent_dim: int,
    action_flat_dim: int,
    cfg_model: Any,
) -> UnifiedConditionedDSM:
    """Build the unified transformer predictor from config values."""
    embed_dim = int(_cfg_get(cfg_model, "embed_dim", _cfg_get(cfg_model, "backbone_dim", 512)))
    return UnifiedConditionedDSM(
        latent_dim=int(latent_dim),
        action_flat_dim=int(action_flat_dim),
        embed_dim=embed_dim,
        num_layers=int(_cfg_get(cfg_model, "num_layers", _cfg_get(cfg_model, "backbone_num_blocks", 4))),
        num_heads=int(_cfg_get(cfg_model, "num_heads", 8)),
        ffn_dim=int(_cfg_get(cfg_model, "ffn_dim", max(4 * embed_dim, embed_dim))),
        dropout=float(_cfg_get(cfg_model, "dropout", 0.1)),
    )


def build_conditional_manifold_denoiser(
    *,
    latent_dim: int,
    action_flat_dim: int,
    cfg_model: Any,
) -> UnifiedConditionedDSM:
    """Backward-compatible alias for the unified predictor builder."""
    return build_unified_conditioned_dsm(
        latent_dim=int(latent_dim),
        action_flat_dim=int(action_flat_dim),
        cfg_model=cfg_model,
    )


def build_joint_manifold_denoiser(
    *,
    tau_dim: int,
    cfg_model: Any,
) -> UnifiedConditionedDSM:
    raise ValueError(
        "build_joint_manifold_denoiser requires the old joint interface. "
        "Use build_unified_conditioned_dsm with latent_dim and action_flat_dim."
    )


def build_dsm_model(
    *,
    latent_dim: int,
    action_dim: int,
    cfg_model: Any,
    transition_horizon: int,
) -> DSMModel:
    """Build the full DSM wrapper around the unified predictor."""
    action_flat_dim = int(action_dim * transition_horizon)
    predictor = build_unified_conditioned_dsm(
        latent_dim=int(latent_dim),
        action_flat_dim=action_flat_dim,
        cfg_model=cfg_model,
    )
    return DSMModel(
        predictor=predictor,
        latent_dim=int(latent_dim),
        action_dim=int(action_dim),
        transition_horizon=int(transition_horizon),
        noise_scale=float(_cfg_get(cfg_model, "noise_scale", _cfg_get(cfg_model, "noise_sigma", 0.08))),
        std_clamp_min=float(_cfg_get(cfg_model, "std_clamp_min", 0.05)),
    )
