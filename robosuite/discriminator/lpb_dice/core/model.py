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


class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.ln_2 = nn.LayerNorm(d_model)
        hidden_dim = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        h = self.ln_1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + h
        x = x + self.mlp(self.ln_2(x))
        return x


class LegacyLatentDynamicsPredictor(nn.Module):
    """
    Original small decoder-only transition model kept for checkpoint compatibility.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        d_model: int = 512,
        num_layers: int = 6,
        nhead: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_action_horizon: int = 32,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.max_action_horizon = int(max_action_horizon)

        self.obs_proj = nn.Linear(self.latent_dim, d_model)
        self.action_proj = nn.Linear(self.action_dim, d_model)
        self.pred_latent_token = nn.Parameter(torch.zeros(1, 1, d_model))

        max_seq_len = 1 + self.max_action_horizon + 1
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    d_model=d_model,
                    nhead=nhead,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.latent_head = nn.Linear(d_model, self.latent_dim)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    def extract_feature(
        self,
        obs_latent: torch.Tensor,
        action_tokens: torch.Tensor,
    ) -> torch.Tensor:
        obs_embed = self.obs_proj(obs_latent)
        act_embed = self.action_proj(action_tokens).mean(dim=1)
        return torch.cat([obs_embed, act_embed], dim=-1)

    def forward(
        self,
        obs_latent: torch.Tensor,
        action_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if action_tokens.ndim != 3:
            raise ValueError(
                f"Expected action_tokens shape (B,H,A), got {tuple(action_tokens.shape)}"
            )
        batch_size, horizon, _ = action_tokens.shape
        if horizon > self.max_action_horizon:
            raise ValueError(
                f"action horizon {horizon} exceeds max_action_horizon={self.max_action_horizon}"
            )

        obs = self.obs_proj(obs_latent).unsqueeze(1)
        act = self.action_proj(action_tokens)
        pred_latent_token = self.pred_latent_token.expand(batch_size, -1, -1)

        x = torch.cat([obs, act, pred_latent_token], dim=1)
        seq_len = int(x.shape[1])
        x = x + self.pos_embedding[:, :seq_len, :]
        x = self.drop(x)

        mask = self._causal_mask(seq_len=seq_len, device=x.device)
        for block in self.blocks:
            x = block(x, attn_mask=mask)
        x = self.norm(x)
        pred_latent = self.latent_head(x[:, -1, :])
        return {"pred_latent": pred_latent}


class ActionSequenceEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_layers: int,
        nhead: int,
        dropout: float,
        max_action_horizon: int,
    ) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.randn(1, max_action_horizon + 1, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=int(4 * d_model),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=int(num_layers),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, action_tokens: torch.Tensor) -> torch.Tensor:
        batch_size, horizon, dim = action_tokens.shape
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, action_tokens], dim=1)
        x = x + self.pos_embedding[:, : horizon + 1, :]
        x = self.encoder(x)
        return self.norm(x[:, 0, :])


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


class LatentWorldModelPredictor(nn.Module):
    """
    Stronger latent transition model with residual delta prediction and uncertainty head.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        d_model: int = 512,
        max_action_horizon: int = 32,
        action_encoder_layers: int = 3,
        action_encoder_heads: int = 8,
        action_encoder_dropout: float = 0.1,
        backbone_dim: int = 1024,
        backbone_num_blocks: int = 4,
        head_hidden_dim: int = 1024,
        head_num_blocks: int = 2,
        dropout: float = 0.1,
        delta_scale: float = 1.0,
        use_uncertainty: bool = True,
        min_logvar: float = -6.0,
        max_logvar: float = 2.0,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.max_action_horizon = int(max_action_horizon)
        self.delta_scale = float(delta_scale)
        self.use_uncertainty = bool(use_uncertainty)
        self.min_logvar = float(min_logvar)
        self.max_logvar = float(max_logvar)

        self.obs_proj = nn.Linear(self.latent_dim, d_model)
        self.action_proj = nn.Linear(self.action_dim, d_model)
        self.action_encoder = ActionSequenceEncoder(
            d_model=d_model,
            num_layers=int(action_encoder_layers),
            nhead=int(action_encoder_heads),
            dropout=float(action_encoder_dropout),
            max_action_horizon=int(max_action_horizon),
        )

        fusion_dim = int(2 * d_model + self.latent_dim)
        self.fusion_proj = nn.Sequential(
            nn.Linear(fusion_dim, backbone_dim),
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
        self.feature_head = nn.Sequential(
            nn.Linear(backbone_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.delta_head = MLPHead(
            input_dim=backbone_dim,
            hidden_dim=head_hidden_dim,
            output_dim=self.latent_dim,
            num_blocks=int(head_num_blocks),
            dropout=dropout,
        )
        self.logvar_head = (
            MLPHead(
                input_dim=backbone_dim,
                hidden_dim=head_hidden_dim,
                output_dim=self.latent_dim,
                num_blocks=int(head_num_blocks),
                dropout=dropout,
            )
            if self.use_uncertainty
            else None
        )

    def _encode_inputs(
        self,
        obs_latent: torch.Tensor,
        action_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if action_tokens.ndim != 3:
            raise ValueError(
                f"Expected action_tokens shape (B,H,A), got {tuple(action_tokens.shape)}"
            )
        batch_size, horizon, _ = action_tokens.shape
        if horizon > self.max_action_horizon:
            raise ValueError(
                f"action horizon {horizon} exceeds max_action_horizon={self.max_action_horizon}"
            )
        obs_embed = self.obs_proj(obs_latent)
        action_embed = self.action_proj(action_tokens)
        action_context = self.action_encoder(action_embed)
        fused = self.fusion_proj(torch.cat([obs_latent, obs_embed, action_context], dim=-1))
        for block in self.backbone:
            fused = block(fused)
        fused = self.backbone_norm(fused)
        return obs_embed, action_context, fused

    def extract_feature(
        self,
        obs_latent: torch.Tensor,
        action_tokens: torch.Tensor,
    ) -> torch.Tensor:
        obs_embed, action_context, fused = self._encode_inputs(obs_latent, action_tokens)
        feature = self.feature_head(fused)
        return torch.cat([obs_embed, action_context, feature], dim=-1)

    def _bound_logvar(self, raw_logvar: torch.Tensor) -> torch.Tensor:
        half_range = 0.5 * (self.max_logvar - self.min_logvar)
        mid = 0.5 * (self.max_logvar + self.min_logvar)
        return mid + half_range * torch.tanh(raw_logvar / max(half_range, 1e-6))

    def forward(
        self,
        obs_latent: torch.Tensor,
        action_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        _, _, fused = self._encode_inputs(obs_latent, action_tokens)
        delta = self.delta_head(fused)
        pred_latent = obs_latent + self.delta_scale * delta
        out = {
            "pred_latent": pred_latent,
            "pred_delta": delta,
        }
        if self.logvar_head is not None:
            out["pred_logvar"] = self._bound_logvar(self.logvar_head(fused))
        return out


class LatentDynamicsModel(nn.Module):
    def __init__(self, predictor: nn.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def extract_feature(
        self,
        current_latent: torch.Tensor,
        action_sequence: torch.Tensor,
    ) -> torch.Tensor:
        if hasattr(self.predictor, "extract_feature"):
            return self.predictor.extract_feature(
                obs_latent=current_latent,
                action_tokens=action_sequence,
            )
        obs_embed = self.predictor.obs_proj(current_latent)
        act_embed = self.predictor.action_proj(action_sequence).mean(dim=1)
        return torch.cat([obs_embed, act_embed], dim=-1)

    def forward(
        self,
        current_latent: torch.Tensor,
        action_sequence: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self.predictor(
            obs_latent=current_latent,
            action_tokens=action_sequence,
        )

    def compute_dynamics_loss(
        self,
        current_latent: torch.Tensor,
        action_sequence: torch.Tensor,
        target_latent: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        pred = self.forward(
            current_latent=current_latent,
            action_sequence=action_sequence,
        )
        error = pred["pred_latent"] - target_latent
        latent_mse = error.pow(2).mean()
        if "pred_logvar" in pred:
            inv_var = torch.exp(-pred["pred_logvar"])
            latent_nll = 0.5 * (error.pow(2) * inv_var + pred["pred_logvar"]).mean()
            loss = latent_nll
        else:
            latent_nll = None
            loss = latent_mse

        out: dict[str, Any] = {
            "loss": loss,
            "latent_mse": latent_mse,
            "pred_latent": pred["pred_latent"],
            "target_latent": target_latent,
        }
        if latent_nll is not None:
            out["latent_nll"] = latent_nll
        if "pred_delta" in pred:
            out["pred_delta"] = pred["pred_delta"]
        if "pred_logvar" in pred:
            out["pred_logvar"] = pred["pred_logvar"]
        return out


def build_latent_dynamics_predictor(
    *,
    latent_dim: int,
    action_dim: int,
    cfg_model: Any,
    transition_horizon: int,
) -> nn.Module:
    predictor_type = str(_cfg_get(cfg_model, "predictor_type", "legacy")).lower()
    common_max_horizon = max(
        int(_cfg_get(cfg_model, "max_action_horizon", transition_horizon)),
        int(transition_horizon),
    )

    if predictor_type == "legacy":
        return LegacyLatentDynamicsPredictor(
            latent_dim=int(latent_dim),
            action_dim=int(action_dim),
            d_model=int(_cfg_get(cfg_model, "d_model", 512)),
            num_layers=int(_cfg_get(cfg_model, "num_layers", 6)),
            nhead=int(_cfg_get(cfg_model, "num_heads", 8)),
            mlp_ratio=float(_cfg_get(cfg_model, "mlp_ratio", 4.0)),
            dropout=float(_cfg_get(cfg_model, "dropout", 0.1)),
            max_action_horizon=int(common_max_horizon),
        )

    if predictor_type == "world_model":
        return LatentWorldModelPredictor(
            latent_dim=int(latent_dim),
            action_dim=int(action_dim),
            d_model=int(_cfg_get(cfg_model, "d_model", 512)),
            max_action_horizon=int(common_max_horizon),
            action_encoder_layers=int(_cfg_get(cfg_model, "action_encoder_layers", 3)),
            action_encoder_heads=int(_cfg_get(cfg_model, "action_encoder_heads", 8)),
            action_encoder_dropout=float(_cfg_get(cfg_model, "action_encoder_dropout", _cfg_get(cfg_model, "dropout", 0.1))),
            backbone_dim=int(_cfg_get(cfg_model, "backbone_dim", 1024)),
            backbone_num_blocks=int(_cfg_get(cfg_model, "backbone_num_blocks", 4)),
            head_hidden_dim=int(_cfg_get(cfg_model, "head_hidden_dim", 1024)),
            head_num_blocks=int(_cfg_get(cfg_model, "head_num_blocks", 2)),
            dropout=float(_cfg_get(cfg_model, "dropout", 0.1)),
            delta_scale=float(_cfg_get(cfg_model, "delta_scale", 1.0)),
            use_uncertainty=bool(_cfg_get(cfg_model, "use_uncertainty", True)),
            min_logvar=float(_cfg_get(cfg_model, "min_logvar", -6.0)),
            max_logvar=float(_cfg_get(cfg_model, "max_logvar", 2.0)),
        )

    raise ValueError(f"Unsupported predictor_type: {predictor_type}")
