"""ConditionalDynamicsPredictor: LPB-backbone + AdaLN-Zero + (+, -) token.

Input / output shapes match ``lpb.model.DynamicsPredictor`` so the existing
dataset schema and training loop carry over. The only extra input is
``cond_idx: (B,) long`` selecting the per-sample condition token. The network
is zero-initialized in its AdaLN heads, so at step 0 the output is identical
for both condition values — a property relied on by Phase A of the
training loop and by the unit test in ``tests/test_adaln.py``.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .adaln import AdaLNDecoderBlock, AdaLNModulation, ConditionEmbedder


class ConditionalDynamicsPredictor(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        proprio_dim: int,
        action_dim: int,
        d_model: int = 512,
        num_layers: int = 6,
        nhead: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_action_horizon: int = 32,
        d_cond: int = 64,
        adaln_init_std: float = 0.0,
        residual_latent_head: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.proprio_dim = int(proprio_dim)
        self.action_dim = int(action_dim)
        self.d_model = int(d_model)
        self.num_layers = int(num_layers)
        self.nhead = int(nhead)
        self.mlp_ratio = float(mlp_ratio)
        self.dropout = float(dropout)
        self.max_action_horizon = int(max_action_horizon)
        self.d_cond = int(d_cond)
        self.adaln_init_std = float(adaln_init_std)
        self.residual_latent_head = bool(residual_latent_head)

        self.obs_proj = nn.Linear(self.latent_dim, self.d_model)
        self.proprio_proj = nn.Linear(self.proprio_dim, self.d_model)
        self.action_proj = nn.Linear(self.action_dim, self.d_model)

        self.pred_latent_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.pred_proprio_token = nn.Parameter(torch.zeros(1, 1, self.d_model))

        max_seq_len = 2 + self.max_action_horizon + 2
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, self.d_model) * 0.02)
        self.drop = nn.Dropout(self.dropout)

        self.cond_embed = ConditionEmbedder(self.d_cond)

        self.blocks = nn.ModuleList(
            [
                AdaLNDecoderBlock(
                    d_model=self.d_model,
                    nhead=self.nhead,
                    d_cond=self.d_cond,
                    mlp_ratio=self.mlp_ratio,
                    dropout=self.dropout,
                    adaln_init_std=self.adaln_init_std,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.norm_final = nn.LayerNorm(self.d_model, elementwise_affine=False)
        self.mod_final = AdaLNModulation(
            self.d_cond, self.d_model, init_std=self.adaln_init_std
        )
        self.latent_head = nn.Linear(self.d_model, self.latent_dim)
        self.proprio_head = nn.Linear(self.d_model, self.proprio_dim)
        if self.residual_latent_head:
            # Residual pred_latent: pred = z_current + latent_head(...). Zero-
            # init so the model starts as identity (per-elem MSE ~0.01 on this
            # data) and only learns the delta. Rationale in
            # d4disc_v2_bench_and_ckpt_bugs.md §4 — without this, trained
            # models converged WORSE than identity on eval (eval MSE 0.155
            # vs identity 0.010). Old ckpts trained before this flag existed
            # must load with residual_latent_head=False.
            nn.init.zeros_(self.latent_head.weight)
            nn.init.zeros_(self.latent_head.bias)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    def arch_args(self) -> dict:
        return {
            "latent_dim": self.latent_dim,
            "proprio_dim": self.proprio_dim,
            "action_dim": self.action_dim,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "nhead": self.nhead,
            "mlp_ratio": self.mlp_ratio,
            "dropout": self.dropout,
            "max_action_horizon": self.max_action_horizon,
            "d_cond": self.d_cond,
            "adaln_init_std": self.adaln_init_std,
            "residual_latent_head": self.residual_latent_head,
        }

    def forward(
        self,
        obs_token: torch.Tensor,
        proprio_token: torch.Tensor,
        action_tokens: torch.Tensor,
        cond_idx: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if action_tokens.ndim != 3:
            raise ValueError(
                f"Expected action_tokens shape (B,H,A), got {tuple(action_tokens.shape)}"
            )
        bsz, horizon, _ = action_tokens.shape
        if horizon > self.max_action_horizon:
            raise ValueError(
                f"action horizon {horizon} exceeds max_action_horizon={self.max_action_horizon}"
            )
        if cond_idx.shape[0] != bsz:
            raise ValueError(
                f"cond_idx batch {cond_idx.shape} != action batch {bsz}"
            )

        cond = self.cond_embed(cond_idx)                       # (B, d_cond)

        obs = self.obs_proj(obs_token).unsqueeze(1)
        prop = self.proprio_proj(proprio_token).unsqueeze(1)
        act = self.action_proj(action_tokens)
        pz = self.pred_latent_token.expand(bsz, -1, -1)
        ps = self.pred_proprio_token.expand(bsz, -1, -1)

        x = torch.cat([obs, prop, act, pz, ps], dim=1)
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]
        x = self.drop(x)

        # Condition into attention
        mask = self._causal_mask(seq_len, x.device)
        for blk in self.blocks:
            x = blk(x, cond=cond, attn_mask=mask)

        shift_f, scale_f, _gate_f = self.mod_final(cond)
        x = self.norm_final(x) * (1.0 + scale_f.unsqueeze(1)) + shift_f.unsqueeze(1)

        latent_out = self.latent_head(x[:, -2, :])
        pred_latent = obs_token + latent_out if self.residual_latent_head else latent_out
        pred_proprio = self.proprio_head(x[:, -1, :])
        return {"pred_latent": pred_latent, "pred_proprio": pred_proprio}

    @staticmethod
    def residual_sq(pred_latent: torch.Tensor, target_latent: torch.Tensor) -> torch.Tensor:
        return ((pred_latent - target_latent) ** 2).sum(dim=-1)
