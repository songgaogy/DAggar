from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _cfg_get(cfg: Any, key: str, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.hidden_dim = int(hidden_dim)
        self.fc_1 = nn.Linear(dim, 2 * self.hidden_dim)
        self.fc_2 = nn.Linear(self.hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        gate, value = self.fc_1(h).chunk(2, dim=-1)
        h = F.silu(gate) * value
        h = self.fc_2(h)
        return x + h


class MLPHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_blocks: int,
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = int(input_dim)
        blocks = max(1, int(num_blocks))
        for _ in range(blocks - 1):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.GELU(),
                ]
            )
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConditionalDenoisingTower(nn.Module):
    """One conditional denoising tower."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        backbone_dim: int = 1024,
        backbone_num_blocks: int = 4,
        head_hidden_dim: int = 1024,
        head_num_blocks: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(self.input_dim, backbone_dim),
            nn.LayerNorm(backbone_dim),
            nn.GELU(),
        )
        block_hidden_dim = int(2 * head_hidden_dim)
        self.backbone = nn.ModuleList(
            [
                ResidualMLPBlock(
                    dim=backbone_dim,
                    hidden_dim=block_hidden_dim,
                    dropout=dropout,
                )
                for _ in range(int(backbone_num_blocks))
            ]
        )
        self.backbone_norm = nn.LayerNorm(backbone_dim)
        self.head = MLPHead(
            input_dim=backbone_dim,
            hidden_dim=head_hidden_dim,
            output_dim=self.output_dim,
            num_blocks=int(head_num_blocks),
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.backbone:
            h = block(h)
        h = self.backbone_norm(h)
        return self.head(h)


class ConditionalManifoldDenoiser(nn.Module):
    """Three-headed denoiser over state, action, and next state."""

    def __init__(
        self,
        latent_dim: int,
        action_flat_dim: int,
        backbone_dim: int = 1024,
        backbone_num_blocks: int = 4,
        head_hidden_dim: int = 1024,
        head_num_blocks: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.action_flat_dim = int(action_flat_dim)
        self.state_tower = ConditionalDenoisingTower(
            input_dim=self.latent_dim,
            output_dim=self.latent_dim,
            backbone_dim=backbone_dim,
            backbone_num_blocks=backbone_num_blocks,
            head_hidden_dim=head_hidden_dim,
            head_num_blocks=head_num_blocks,
            dropout=dropout,
        )
        self.actor_tower = ConditionalDenoisingTower(
            input_dim=int(self.latent_dim + self.action_flat_dim),
            output_dim=self.action_flat_dim,
            backbone_dim=backbone_dim,
            backbone_num_blocks=backbone_num_blocks,
            head_hidden_dim=head_hidden_dim,
            head_num_blocks=head_num_blocks,
            dropout=dropout,
        )
        self.dynamics_tower = ConditionalDenoisingTower(
            input_dim=int(2 * self.latent_dim + self.action_flat_dim),
            output_dim=self.latent_dim,
            backbone_dim=backbone_dim,
            backbone_num_blocks=backbone_num_blocks,
            head_hidden_dim=head_hidden_dim,
            head_num_blocks=head_num_blocks,
            dropout=dropout,
        )

    def forward(
        self,
        *,
        state_input: torch.Tensor,
        current_latent: torch.Tensor,
        action_input: torch.Tensor,
        action_clean: torch.Tensor,
        next_state_input: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        state_hat = self.state_tower(state_input)
        actor_input = torch.cat([current_latent, action_input], dim=-1)
        action_hat = self.actor_tower(actor_input)
        dynamics_input = torch.cat([current_latent, action_clean, next_state_input], dim=-1)
        next_state_hat = self.dynamics_tower(dynamics_input)
        return {
            "state_hat": state_hat,
            "action_hat": action_hat,
            "next_state_hat": next_state_hat,
        }


JointManifoldDenoiser = ConditionalManifoldDenoiser


class DSMModel(nn.Module):
    """Denoise state, action, and next state with variance scaling."""

    def __init__(
        self,
        predictor: ConditionalManifoldDenoiser,
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
        latent_mean = self.latent_mean.to(device=latent.device, dtype=latent.dtype)
        latent_std = self.latent_std.to(device=latent.device, dtype=latent.dtype)
        return (latent - latent_mean.unsqueeze(0)) / latent_std.unsqueeze(0)

    def normalize_action_flat(self, action_flat: torch.Tensor) -> torch.Tensor:
        action_mean = self.action_mean_flat.to(device=action_flat.device, dtype=action_flat.dtype)
        action_std = self.action_std_flat.to(device=action_flat.device, dtype=action_flat.dtype)
        return (action_flat - action_mean.unsqueeze(0)) / action_std.unsqueeze(0)

    def flatten_action_sequence(self, action_sequence: torch.Tensor) -> torch.Tensor:
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
        state_sq = F.mse_loss(state_hat, state_clean, reduction="none")
        action_sq = F.mse_loss(action_hat, action_clean, reduction="none")
        next_state_sq = F.mse_loss(next_state_hat, next_state_clean, reduction="none")
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

    def forward(
        self,
        current_latent: torch.Tensor,
        action_sequence: torch.Tensor,
        target_latent: torch.Tensor,
        *,
        add_noise: bool = False,
    ) -> dict[str, torch.Tensor]:
        state_clean_raw = current_latent
        action_clean_raw = self.flatten_action_sequence(action_sequence)
        next_state_clean_raw = target_latent
        state_clean = self.normalize_latent(state_clean_raw)
        action_clean = self.normalize_action_flat(action_clean_raw)
        next_state_norm_clean = self.normalize_latent(next_state_clean_raw)
        delta_clean = next_state_norm_clean - state_clean
        noisy = {
            "state_input": state_clean,
            "state_noise": torch.zeros_like(state_clean),
            "action_input": action_clean,
            "action_noise": torch.zeros_like(action_clean),
            "next_state_input": next_state_norm_clean,
            "next_state_noise": torch.zeros_like(next_state_norm_clean),
        }
        if add_noise:
            noisy = self.add_noise(
                state_clean=state_clean,
                action_clean=action_clean,
                next_state_clean=next_state_norm_clean,
            )
        preds = self.predictor(
            state_input=noisy["state_input"],
            current_latent=state_clean,
            action_input=noisy["action_input"],
            action_clean=action_clean,
            next_state_input=noisy["next_state_input"],
        )
        return {
            "current_latent": current_latent,
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
    ) -> dict[str, torch.Tensor]:
        out = self.forward(
            current_latent=current_latent,
            action_sequence=action_sequence,
            target_latent=target_latent,
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


def build_conditional_manifold_denoiser(
    *,
    latent_dim: int,
    action_flat_dim: int,
    cfg_model: Any,
) -> ConditionalManifoldDenoiser:
    return ConditionalManifoldDenoiser(
        latent_dim=int(latent_dim),
        action_flat_dim=int(action_flat_dim),
        backbone_dim=int(_cfg_get(cfg_model, "backbone_dim", 1024)),
        backbone_num_blocks=int(_cfg_get(cfg_model, "backbone_num_blocks", 4)),
        head_hidden_dim=int(_cfg_get(cfg_model, "head_hidden_dim", 1024)),
        head_num_blocks=int(_cfg_get(cfg_model, "head_num_blocks", 2)),
        dropout=float(_cfg_get(cfg_model, "dropout", 0.1)),
    )


def build_joint_manifold_denoiser(
    *,
    tau_dim: int,
    cfg_model: Any,
) -> ConditionalManifoldDenoiser:
    raise ValueError(
        "build_joint_manifold_denoiser requires the old joint interface. "
        "Use build_conditional_manifold_denoiser with latent_dim and action_flat_dim."
    )


def build_dsm_model(
    *,
    latent_dim: int,
    action_dim: int,
    cfg_model: Any,
    transition_horizon: int,
) -> DSMModel:
    action_flat_dim = int(action_dim * transition_horizon)
    predictor = build_conditional_manifold_denoiser(
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
