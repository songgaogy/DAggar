"""Multitask latent DSM with late-fusion conditioning and AdaLN-Zero DiT blocks.

``UnifiedConditionedDSM`` encodes each routed target/context independently, aggregates route/type/task
conditions into one global vector, and denoises with a lightweight shared MLP backbone.
``DSMModel`` wraps it with normalization, noise, DSM loss, and Fisher scores for the offline detector.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def _cfg_get(cfg: Any, key: str, default=None):
    """Read ``key`` from a dict-like object or attribute-style config."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class AdaLNModulation(nn.Module):
    """AdaLN-Zero modulation for one residual branch."""

    def __init__(self, cond_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(cond_dim, 3 * embed_dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shift, scale, gate = self.linear(cond).chunk(3, dim=-1)
        return shift, scale, gate


class DiTFeedForward(nn.Module):
    """Feedforward block used by the shared DiT-style denoiser."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiTBlock(nn.Module):
    """Two-branch AdaLN-Zero residual MLP block."""

    def __init__(self, embed_dim: int, cond_dim: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.mod1 = AdaLNModulation(cond_dim=cond_dim, embed_dim=embed_dim)
        self.mod2 = AdaLNModulation(cond_dim=cond_dim, embed_dim=embed_dim)
        self.ffn1 = DiTFeedForward(embed_dim=embed_dim, ffn_dim=ffn_dim, dropout=dropout)
        self.ffn2 = DiTFeedForward(embed_dim=embed_dim, ffn_dim=ffn_dim, dropout=dropout)

    @staticmethod
    def _apply_adaln(normed_x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return normed_x * (1.0 + scale) + shift

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift1, scale1, gate1 = self.mod1(cond)
        y = self._apply_adaln(self.norm1(x), shift1, scale1)
        x = x + gate1 * self.ffn1(y)

        shift2, scale2, gate2 = self.mod2(cond)
        y = self._apply_adaln(self.norm2(x), shift2, scale2)
        x = x + gate2 * self.ffn2(y)
        return x


class UnifiedConditionedDSM(nn.Module):
    """Late-fusion multitask DSM with routed encoders and a shared AdaLN-Zero denoiser."""

    TASK_STATE = 0
    TASK_ACTOR = 1
    TASK_DYNAMICS = 2

    def __init__(
        self,
        latent_dim: int,
        action_flat_dim: int,
        num_tasks: int,
        embed_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        """Args: ``num_tasks`` is the multitask vocabulary size (indices align with ``task_to_index``)."""
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.action_flat_dim = int(action_flat_dim)
        self.num_tasks = int(num_tasks)
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
        if self.ffn_dim <= 0:
            raise ValueError("ffn_dim must be positive.")
        if self.num_tasks <= 0:
            raise ValueError("num_tasks must be positive.")
        if self.dropout < 0.0:
            raise ValueError("dropout must be non-negative.")

        # route ∈ {state, actor, dynamics}; type ∈ {non-fail, fail}; task_name ∈ {0..num_tasks-1}.
        self.route_embedding = nn.Embedding(3, self.embed_dim)
        self.type_embedding = nn.Embedding(2, self.embed_dim)
        self.task_name_embedding = nn.Embedding(self.num_tasks, self.embed_dim)

        self.state_target_encoder = self._build_feature_encoder(self.latent_dim)
        self.action_target_encoder = self._build_feature_encoder(self.action_flat_dim)
        self.dynamics_target_encoder = self._build_feature_encoder(self.latent_dim)

        self.state_context_encoder = self._build_feature_encoder(self.latent_dim)
        self.action_context_encoder = self._build_feature_encoder(self.action_flat_dim)
        self.dynamics_context_encoder = self._build_condition_mlp(2 * self.embed_dim, self.embed_dim)
        self.condition_mlp = self._build_condition_mlp(2 * self.embed_dim, self.embed_dim)

        self.shared_blocks = nn.ModuleList(
            [
                DiTBlock(
                    embed_dim=self.embed_dim,
                    cond_dim=self.embed_dim,
                    ffn_dim=self.ffn_dim,
                    dropout=self.dropout,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(self.embed_dim)
        self.state_head = nn.Linear(self.embed_dim, self.latent_dim)
        self.action_head = nn.Linear(self.embed_dim, self.action_flat_dim)
        self.dynamics_head = nn.Linear(self.embed_dim, self.latent_dim)

    def _task_target_dim(self, task_id: int) -> int:
        if int(task_id) == self.TASK_STATE:
            return self.latent_dim
        if int(task_id) == self.TASK_ACTOR:
            return self.action_flat_dim
        if int(task_id) == self.TASK_DYNAMICS:
            return self.latent_dim
        raise ValueError(f"Unsupported task_id: {task_id}")

    def _task_head(self, task_id: int) -> nn.Module:
        if int(task_id) == self.TASK_STATE:
            return self.state_head
        if int(task_id) == self.TASK_ACTOR:
            return self.action_head
        if int(task_id) == self.TASK_DYNAMICS:
            return self.dynamics_head
        raise ValueError(f"Unsupported task_id: {task_id}")

    def _task_target_encoder(self, task_id: int) -> nn.Module:
        if int(task_id) == self.TASK_STATE:
            return self.state_target_encoder
        if int(task_id) == self.TASK_ACTOR:
            return self.action_target_encoder
        if int(task_id) == self.TASK_DYNAMICS:
            return self.dynamics_target_encoder
        raise ValueError(f"Unsupported task_id: {task_id}")

    def _build_feature_encoder(self, input_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

    def _build_condition_mlp(self, input_dim: int, output_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, self.embed_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embed_dim, output_dim),
        )

    @staticmethod
    def _require_2d(name: str, x: torch.Tensor) -> None:
        if x.ndim != 2:
            raise ValueError(f"Expected {name} shape (B,D), got {tuple(x.shape)}")

    @staticmethod
    def _require_feature_dim(name: str, x: torch.Tensor, expected_dim: int) -> None:
        if int(x.shape[1]) != int(expected_dim):
            raise ValueError(f"{name} dim mismatch: expected {expected_dim}, got {x.shape[1]}")

    @staticmethod
    def _require_batch_size(name: str, x: torch.Tensor, expected_batch_size: int) -> None:
        if int(x.shape[0]) != int(expected_batch_size):
            raise ValueError(f"Expected {name} batch size {expected_batch_size}, got {x.shape[0]}")

    def _require_task_index(
        self,
        batch_size: int,
        task_index: torch.Tensor | int,
        device: torch.device,
    ) -> torch.Tensor:
        """Broadcast or validate per-batch task IDs in ``[0, num_tasks)``."""
        if isinstance(task_index, int):
            task_index_tensor = torch.full((batch_size,), int(task_index), dtype=torch.long, device=device)
        else:
            task_index_tensor = torch.as_tensor(task_index, dtype=torch.long, device=device).reshape(-1)
        if int(task_index_tensor.shape[0]) != int(batch_size):
            raise ValueError(f"Expected task_index shape ({batch_size},), got {tuple(task_index_tensor.shape)}")
        if torch.any((task_index_tensor < 0) | (task_index_tensor >= self.num_tasks)):
            raise ValueError(
                f"task_index must be in [0, {self.num_tasks - 1}] for this checkpoint's task vocabulary."
            )
        return task_index_tensor

    @staticmethod
    def _require_traj_type(
        batch_size: int,
        traj_type: torch.Tensor | int,
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(traj_type, int):
            traj_type_tensor = torch.full((batch_size,), int(traj_type), dtype=torch.long, device=device)
        else:
            traj_type_tensor = torch.as_tensor(traj_type, dtype=torch.long, device=device).reshape(-1)
        if int(traj_type_tensor.shape[0]) != int(batch_size):
            raise ValueError(f"Expected traj_type shape ({batch_size},), got {tuple(traj_type_tensor.shape)}")
        if torch.any((traj_type_tensor < 0) | (traj_type_tensor > 1)):
            raise ValueError("traj_type must contain only 0 (positive) or 1 (negative).")
        return traj_type_tensor

    def _build_condition(
        self,
        *,
        task_id: int,
        context_feature: torch.Tensor,
        traj_type: torch.Tensor,
        task_index: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = int(context_feature.shape[0])
        task_ids = torch.full((batch_size,), int(task_id), dtype=torch.long, device=context_feature.device)
        discrete_feature = (
            self.route_embedding(task_ids)
            + self.type_embedding(traj_type)
            + self.task_name_embedding(task_index)
        )
        return self.condition_mlp(torch.cat([context_feature, discrete_feature], dim=-1))

    def _run_shared_backbone(self, target_feature: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        hidden = target_feature
        for block in self.shared_blocks:
            hidden = block(hidden, condition)
        return self.final_norm(hidden)

    @staticmethod
    def _module_parameters(*modules: nn.Module) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for module in modules:
            params.extend(list(module.parameters()))
        return params

    def optimizer_parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        """Return branch-aware parameter groups for optimizer LR scaling."""
        return {
            "shared": self._module_parameters(
                self.route_embedding,
                self.type_embedding,
                self.task_name_embedding,
                self.state_context_encoder,
                self.condition_mlp,
                self.shared_blocks,
                self.final_norm,
            ),
            "state_branch": self._module_parameters(
                self.state_target_encoder,
                self.state_head,
            ),
            "action_branch": self._module_parameters(
                self.action_target_encoder,
                self.action_head,
            ),
            "dynamics_branch": self._module_parameters(
                self.action_context_encoder,
                self.dynamics_context_encoder,
                self.dynamics_target_encoder,
                self.dynamics_head,
            ),
        }

    def forward_task(
        self,
        *,
        task_id: int,
        noisy_target: torch.Tensor,
        context_feature: torch.Tensor,
        traj_type: torch.Tensor,
        task_index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run one routed denoising task with one target token and one global condition."""
        target_dim = self._task_target_dim(task_id)
        self._require_2d("noisy_target", noisy_target)
        self._require_feature_dim("noisy_target", noisy_target, target_dim)
        self._require_2d("context_feature", context_feature)
        self._require_feature_dim("context_feature", context_feature, self.embed_dim)

        target_feature = self._task_target_encoder(task_id)(noisy_target)
        c_global = self._build_condition(
            task_id=task_id,
            context_feature=context_feature,
            traj_type=traj_type,
            task_index=task_index,
        )
        hidden = self._run_shared_backbone(target_feature=target_feature, condition=c_global)
        pred_mean = self._task_head(task_id)(hidden)
        return {
            "pred_mean": pred_mean,
            "target_feature": hidden,
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
        task_index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run state, action-chunk, and residual dynamics heads; ``task_index`` selects multitask embeddings."""
        batch_size = int(state_input.shape[0])
        self._require_2d("state_input", state_input)
        self._require_feature_dim("state_input", state_input, self.latent_dim)
        self._require_2d("current_latent", current_latent)
        self._require_batch_size("current_latent", current_latent, batch_size)
        self._require_feature_dim("current_latent", current_latent, self.latent_dim)
        self._require_2d("action_input", action_input)
        self._require_batch_size("action_input", action_input, batch_size)
        self._require_feature_dim("action_input", action_input, self.action_flat_dim)
        self._require_2d("action_clean", action_clean)
        self._require_batch_size("action_clean", action_clean, batch_size)
        self._require_feature_dim("action_clean", action_clean, self.action_flat_dim)
        self._require_2d("next_state_input", next_state_input)
        self._require_batch_size("next_state_input", next_state_input, batch_size)
        self._require_feature_dim("next_state_input", next_state_input, self.latent_dim)

        traj_type_tensor = self._require_traj_type(
            batch_size=batch_size,
            traj_type=traj_type,
            device=state_input.device,
        )
        task_index_tensor = self._require_task_index(
            batch_size=batch_size,
            task_index=task_index,
            device=state_input.device,
        )

        state_context_feature = state_input.new_zeros((batch_size, self.embed_dim))
        encoded_current_latent = self.state_context_encoder(current_latent)
        actor_context_feature = encoded_current_latent
        dynamics_context_feature = self.dynamics_context_encoder(
            torch.cat(
                [
                    encoded_current_latent,
                    self.action_context_encoder(action_clean),
                ],
                dim=-1,
            )
        )

        state_out = self.forward_task(
            task_id=self.TASK_STATE,
            noisy_target=state_input,
            context_feature=state_context_feature,
            traj_type=traj_type_tensor,
            task_index=task_index_tensor,
        )
        action_out = self.forward_task(
            task_id=self.TASK_ACTOR,
            noisy_target=action_input,
            context_feature=actor_context_feature,
            traj_type=traj_type_tensor,
            task_index=task_index_tensor,
        )
        dynamics_out = self.forward_task(
            task_id=self.TASK_DYNAMICS,
            noisy_target=next_state_input,
            context_feature=dynamics_context_feature,
            traj_type=traj_type_tensor,
            task_index=task_index_tensor,
        )
        return {
            "state_hat": state_out["pred_mean"],
            "action_hat": action_out["pred_mean"],
            "next_state_hat": dynamics_out["pred_mean"],
        }


ConditionalManifoldDenoiser = UnifiedConditionedDSM
JointManifoldDenoiser = UnifiedConditionedDSM


class DSMModel(nn.Module):
    """DSM training/inference: latent and action standardization, noised targets, and Fisher diagnostics."""

    def __init__(
        self,
        predictor: UnifiedConditionedDSM,
        latent_dim: int,
        action_dim: int,
        transition_horizon: int,
        noise_scale: float,
        std_clamp_min: float,
        state_loss_weight: float = 1.0,
        action_loss_weight: float = 0.5,
        dynamics_loss_weight: float = 1.0,
    ) -> None:
        """``predictor.num_tasks`` must match the checkpoint and dataset ``task_index`` range."""
        super().__init__()
        self.predictor = predictor
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.transition_horizon = int(transition_horizon)
        self.noise_scale = float(noise_scale)
        self.noise_sigma = self.noise_scale
        self.std_clamp_min = float(std_clamp_min)
        self.state_loss_weight = float(state_loss_weight)
        self.action_loss_weight = float(action_loss_weight)
        self.dynamics_loss_weight = float(dynamics_loss_weight)
        self.num_tasks = int(self.predictor.num_tasks)
        self.action_flat_dim = int(self.action_dim * self.transition_horizon)
        self.tau_dim = int(2 * self.latent_dim + self.action_flat_dim)

        self.state_slice = slice(0, self.latent_dim)
        self.action_slice = slice(self.latent_dim, self.latent_dim + self.action_flat_dim)
        self.next_state_slice = slice(self.latent_dim + self.action_flat_dim, self.tau_dim)

        self.register_buffer("latent_mean", torch.zeros(self.latent_dim, dtype=torch.float32))
        self.register_buffer("latent_var", torch.ones(self.latent_dim, dtype=torch.float32))
        self.register_buffer("action_mean", torch.zeros(self.action_dim, dtype=torch.float32))
        self.register_buffer("action_var", torch.ones(self.action_dim, dtype=torch.float32))

        if self.state_loss_weight < 0.0:
            raise ValueError(f"state_loss_weight must be non-negative, got {self.state_loss_weight}")
        if self.action_loss_weight < 0.0:
            raise ValueError(f"action_loss_weight must be non-negative, got {self.action_loss_weight}")
        if self.dynamics_loss_weight < 0.0:
            raise ValueError(f"dynamics_loss_weight must be non-negative, got {self.dynamics_loss_weight}")

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

    def _require_task_index(
        self,
        batch_size: int,
        task_index: torch.Tensor | int,
        device: torch.device,
    ) -> torch.Tensor:
        """Delegate task-index validation to the underlying ``UnifiedConditionedDSM``."""
        return self.predictor._require_task_index(
            batch_size=batch_size,
            task_index=task_index,
            device=device,
        )

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

    def weighted_reconstruction_score(self, recon: dict[str, torch.Tensor]) -> torch.Tensor:
        """Weighted reconstruction objective used for DSM training."""
        return (
            self.state_loss_weight * recon["state_energy_per_sample"]
            + self.action_loss_weight * recon["action_energy_per_sample"]
            + self.dynamics_loss_weight * recon["next_state_energy_per_sample"]
        )

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
        task_index: torch.Tensor | int,
        add_noise: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Normalize ``(z_t, a, z_{t+H})``, optionally add DSM noise, and forward through ``predictor``."""
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
        task_index_tensor = self._require_task_index(
            batch_size=int(current_latent.shape[0]),
            task_index=task_index,
            device=current_latent.device,
        )

        preds = self.predictor(
            state_input=noisy["state_input"],
            current_latent=state_clean,
            action_input=noisy["action_input"],
            action_clean=action_clean,
            next_state_input=noisy["next_state_input"],
            traj_type=traj_type_tensor,
            task_index=task_index_tensor,
        )
        return {
            "current_latent": current_latent,
            "traj_type": traj_type_tensor,
            "task_index": task_index_tensor,
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
        task_index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Mean reconstruction energy over noised heads; used by ``Trainer``."""
        out = self.forward(
            current_latent=current_latent,
            action_sequence=action_sequence,
            target_latent=target_latent,
            traj_type=traj_type,
            task_index=task_index,
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
        weighted_score_per_sample = self.weighted_reconstruction_score(recon)
        loss = weighted_score_per_sample.mean()
        return {
            "loss": loss,
            "score": loss,
            "unweighted_score": recon["score_per_sample"].mean(),
            "weighted_score": loss,
            "tau_mse": recon["tau_mse_per_sample"].mean(),
            "state_mse": recon["state_mse_per_sample"].mean(),
            "action_mse": recon["action_mse_per_sample"].mean(),
            "next_state_mse": recon["next_state_mse_per_sample"].mean(),
            "state_energy": recon["state_energy_per_sample"].mean(),
            "action_energy": recon["action_energy_per_sample"].mean(),
            "next_state_energy": recon["next_state_energy_per_sample"].mean(),
            "state_loss_weight": out["state_clean"].new_tensor(self.state_loss_weight),
            "action_loss_weight": out["state_clean"].new_tensor(self.action_loss_weight),
            "dynamics_loss_weight": out["state_clean"].new_tensor(self.dynamics_loss_weight),
            "current_latent": out["current_latent"],
            "traj_type": out["traj_type"],
            "task_index": out["task_index"],
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
        task_index: torch.Tensor | int,
    ) -> dict[str, torch.Tensor]:
        """Compare traj_type 0 vs 1 at fixed ``task_index``; returns Fisher gaps, margins, and branch energies."""
        pos_out = self.forward(
            current_latent=current_latent,
            action_sequence=action_sequence,
            target_latent=target_latent,
            traj_type=0,
            task_index=task_index,
            add_noise=False,
        )
        neg_out = self.forward(
            current_latent=current_latent,
            action_sequence=action_sequence,
            target_latent=target_latent,
            traj_type=1,
            task_index=task_index,
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
        pos_recon = self.reconstruction_components(
            state_clean=pos_out["state_clean"],
            state_hat=pos_out["state_hat"],
            action_clean=pos_out["action_clean"],
            action_hat=pos_out["action_hat"],
            next_state_clean=pos_out["next_state_clean"],
            next_state_hat=pos_out["next_state_hat"],
        )
        neg_recon = self.reconstruction_components(
            state_clean=pos_out["state_clean"],
            state_hat=neg_out["state_hat"],
            action_clean=pos_out["action_clean"],
            action_hat=neg_out["action_hat"],
            next_state_clean=pos_out["next_state_clean"],
            next_state_hat=neg_out["next_state_hat"],
        )
        state_margin = neg_recon["state_energy_per_sample"] - pos_recon["state_energy_per_sample"]
        action_margin = neg_recon["action_energy_per_sample"] - pos_recon["action_energy_per_sample"]
        next_state_margin = neg_recon["next_state_energy_per_sample"] - pos_recon["next_state_energy_per_sample"]
        return {
            "score_per_sample": fisher["score_per_sample"],
            "state_error_per_sample": fisher["state_error_per_sample"],
            "action_error_per_sample": fisher["action_error_per_sample"],
            "next_state_error_per_sample": fisher["next_state_error_per_sample"],
            "state_positive_energy_per_sample": pos_recon["state_energy_per_sample"],
            "action_positive_energy_per_sample": pos_recon["action_energy_per_sample"],
            "next_state_positive_energy_per_sample": pos_recon["next_state_energy_per_sample"],
            "positive_score_per_sample": pos_recon["score_per_sample"],
            "state_negative_energy_per_sample": neg_recon["state_energy_per_sample"],
            "action_negative_energy_per_sample": neg_recon["action_energy_per_sample"],
            "next_state_negative_energy_per_sample": neg_recon["next_state_energy_per_sample"],
            "negative_score_per_sample": neg_recon["score_per_sample"],
            "state_margin_per_sample": state_margin,
            "action_margin_per_sample": action_margin,
            "next_state_margin_per_sample": next_state_margin,
            "margin_score_per_sample": state_margin + action_margin + next_state_margin,
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
    num_tasks: int,
    cfg_model: Any,
) -> UnifiedConditionedDSM:
    """Build the unified late-fusion predictor from config values."""
    embed_dim = int(_cfg_get(cfg_model, "embed_dim", _cfg_get(cfg_model, "backbone_dim", 512)))
    return UnifiedConditionedDSM(
        latent_dim=int(latent_dim),
        action_flat_dim=int(action_flat_dim),
        num_tasks=int(num_tasks),
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
    num_tasks: int,
    cfg_model: Any,
) -> UnifiedConditionedDSM:
    """Backward-compatible alias for the unified predictor builder."""
    return build_unified_conditioned_dsm(
        latent_dim=int(latent_dim),
        action_flat_dim=int(action_flat_dim),
        num_tasks=int(num_tasks),
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
    num_tasks: int,
    cfg_model: Any,
    transition_horizon: int,
) -> DSMModel:
    """Instantiate ``UnifiedConditionedDSM`` with ``num_tasks`` and wrap it in ``DSMModel``."""
    action_flat_dim = int(action_dim * transition_horizon)
    predictor = build_unified_conditioned_dsm(
        latent_dim=int(latent_dim),
        action_flat_dim=action_flat_dim,
        num_tasks=int(num_tasks),
        cfg_model=cfg_model,
    )
    return DSMModel(
        predictor=predictor,
        latent_dim=int(latent_dim),
        action_dim=int(action_dim),
        transition_horizon=int(transition_horizon),
        noise_scale=float(_cfg_get(cfg_model, "noise_scale", _cfg_get(cfg_model, "noise_sigma", 0.08))),
        std_clamp_min=float(_cfg_get(cfg_model, "std_clamp_min", 0.05)),
        state_loss_weight=float(_cfg_get(cfg_model, "state_loss_weight", 1.0)),
        action_loss_weight=float(_cfg_get(cfg_model, "action_loss_weight", 0.5)),
        dynamics_loss_weight=float(_cfg_get(cfg_model, "dynamics_loss_weight", 1.0)),
    )
