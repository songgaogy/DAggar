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
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        gate, value = self.fc_1(h).chunk(2, dim=-1)
        h = F.silu(gate) * value
        h = self.dropout(h)
        h = self.fc_2(h)
        h = self.dropout(h)
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
                    nn.Dropout(dropout),
                ]
            )
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class JointManifoldDenoiser(nn.Module):
    """MLP denoiser on the joint transition manifold."""

    def __init__(
        self,
        tau_dim: int,
        backbone_dim: int = 1024,
        backbone_num_blocks: int = 4,
        head_hidden_dim: int = 1024,
        head_num_blocks: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.tau_dim = int(tau_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(self.tau_dim, backbone_dim),
            nn.LayerNorm(backbone_dim),
            nn.GELU(),
            nn.Dropout(dropout),
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
            output_dim=self.tau_dim,
            num_blocks=int(head_num_blocks),
            dropout=dropout,
        )

    def forward(self, tau_noisy: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(tau_noisy)
        for block in self.backbone:
            h = block(h)
        h = self.backbone_norm(h)
        return self.head(h)


class DSMModel(nn.Module):
    """Build and denoise the joint vector [z_t, a_{t:t+H-1}, z_{t+H}]."""

    def __init__(
        self,
        predictor: JointManifoldDenoiser,
        latent_dim: int,
        action_dim: int,
        transition_horizon: int,
        noise_sigma: float,
    ) -> None:
        super().__init__()
        self.predictor = predictor
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.transition_horizon = int(transition_horizon)
        self.noise_sigma = float(noise_sigma)
        self.action_flat_dim = int(self.action_dim * self.transition_horizon)
        self.tau_dim = int(2 * self.latent_dim + self.action_flat_dim)

        self.state_slice = slice(0, self.latent_dim)
        self.action_slice = slice(self.latent_dim, self.latent_dim + self.action_flat_dim)
        self.next_state_slice = slice(self.latent_dim + self.action_flat_dim, self.tau_dim)

        self.register_buffer("latent_mean", torch.zeros(self.latent_dim, dtype=torch.float32))
        self.register_buffer("latent_var", torch.ones(self.latent_dim, dtype=torch.float32))
        self.register_buffer("action_mean", torch.zeros(self.action_dim, dtype=torch.float32))
        self.register_buffer("action_var", torch.ones(self.action_dim, dtype=torch.float32))

        if int(self.predictor.tau_dim) != self.tau_dim:
            raise ValueError(
                f"predictor tau_dim mismatch: expected {self.tau_dim}, got {self.predictor.tau_dim}"
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
    def tau_noise_scale(self) -> torch.Tensor:
        latent_std = torch.sqrt(self.latent_var)
        action_std = torch.sqrt(self.action_var_flat)
        return torch.cat([latent_std, action_std, latent_std], dim=0)

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
                action_sequence.reshape(action_sequence.shape[0], -1),
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

    def add_noise(self, tau: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.noise_sigma <= 0.0:
            eps = torch.zeros_like(tau)
            return tau, eps
        noise_scale = self.tau_noise_scale.to(device=tau.device, dtype=tau.dtype).unsqueeze(0)
        eps = torch.randn_like(tau) * noise_scale * float(self.noise_sigma)
        return tau + eps, eps

    def denoise_tau(self, tau: torch.Tensor, *, add_noise: bool) -> dict[str, torch.Tensor]:
        tau_in = tau
        eps = torch.zeros_like(tau)
        if add_noise:
            tau_in, eps = self.add_noise(tau)
        tau_hat = self.predictor(tau_in)
        return {
            "tau": tau,
            "tau_input": tau_in,
            "tau_hat": tau_hat,
            "noise": eps,
        }

    def reconstruction_components(
        self,
        tau: torch.Tensor,
        tau_hat: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        error_sq = (tau_hat - tau).pow(2)
        state_sq = error_sq[:, self.state_slice]
        action_sq = error_sq[:, self.action_slice]
        next_state_sq = error_sq[:, self.next_state_slice]
        latent_var = self.latent_var.to(device=tau.device, dtype=tau.dtype)
        action_var_flat = self.action_var_flat.to(device=tau.device, dtype=tau.dtype)
        state_energy = (state_sq / latent_var.unsqueeze(0)).mean(dim=-1)
        action_energy = (action_sq / action_var_flat.unsqueeze(0)).mean(dim=-1)
        next_state_energy = (next_state_sq / latent_var.unsqueeze(0)).mean(dim=-1)
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
        tau = self.build_tau(
            current_latent=current_latent,
            action_sequence=action_sequence,
            target_latent=target_latent,
        )
        return self.denoise_tau(tau=tau, add_noise=add_noise)

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
        recon = self.reconstruction_components(tau=out["tau"], tau_hat=out["tau_hat"])
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
            "tau": out["tau"],
            "tau_input": out["tau_input"],
            "tau_hat": out["tau_hat"],
        }


def build_joint_manifold_denoiser(
    *,
    tau_dim: int,
    cfg_model: Any,
) -> JointManifoldDenoiser:
    return JointManifoldDenoiser(
        tau_dim=int(tau_dim),
        backbone_dim=int(_cfg_get(cfg_model, "backbone_dim", 1024)),
        backbone_num_blocks=int(_cfg_get(cfg_model, "backbone_num_blocks", 4)),
        head_hidden_dim=int(_cfg_get(cfg_model, "head_hidden_dim", 1024)),
        head_num_blocks=int(_cfg_get(cfg_model, "head_num_blocks", 2)),
        dropout=float(_cfg_get(cfg_model, "dropout", 0.1)),
    )


def build_dsm_model(
    *,
    latent_dim: int,
    action_dim: int,
    cfg_model: Any,
    transition_horizon: int,
) -> DSMModel:
    tau_dim = int(2 * latent_dim + action_dim * transition_horizon)
    predictor = build_joint_manifold_denoiser(
        tau_dim=tau_dim,
        cfg_model=cfg_model,
    )
    return DSMModel(
        predictor=predictor,
        latent_dim=int(latent_dim),
        action_dim=int(action_dim),
        transition_horizon=int(transition_horizon),
        noise_sigma=float(_cfg_get(cfg_model, "noise_sigma", 0.1)),
    )
