from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class Encoder(nn.Module):
    """
    ResNet-18 visual encoder h_theta that outputs a single latent token per image.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        pretrained: bool = True,
        freeze: bool = True,
        normalize_input: bool = True,
    ) -> None:
        super().__init__()
        # Keep torchvision API compatibility across versions.
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet18(weights=weights)
        except Exception:
            resnet = models.resnet18(pretrained=pretrained)

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # (B, 512, 1, 1)
        self.flatten = nn.Flatten()
        self.latent_dim = 512
        self.normalize_input = bool(normalize_input)
        self.register_buffer(
            "rgb_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "rgb_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        if checkpoint_path:
            self._load_external_checkpoint(checkpoint_path)
        if freeze:
            self.freeze()

    def _load_external_checkpoint(self, checkpoint_path: str) -> None:
        state = torch.load(checkpoint_path, map_location="cpu")
        full = models.resnet18(weights=None)
        full_state_keys = set(full.state_dict().keys())
        backbone_keys = {k for k in full_state_keys if not k.startswith("fc.")}

        for candidate in self._iter_state_dict_candidates(state):
            for cleaned in self._candidate_key_variants(candidate):
                try:
                    incompatible = full.load_state_dict(cleaned, strict=False)
                except Exception:
                    continue

                missing = set(incompatible.missing_keys)
                unexpected = set(incompatible.unexpected_keys)
                if not (backbone_keys - missing) == backbone_keys:
                    # Not all backbone params were restored.
                    continue
                # Accept typical case where fc may be missing, but reject junk mappings.
                if any(k not in {"fc.weight", "fc.bias"} for k in missing):
                    continue
                if len(unexpected) > 0:
                    continue

                self.backbone = nn.Sequential(*list(full.children())[:-1])
                return
        raise ValueError(f"Unsupported ResNet checkpoint format: {checkpoint_path}")

    @staticmethod
    def _is_tensor_dict(x: Any) -> bool:
        if not isinstance(x, (dict, OrderedDict)) or len(x) == 0:
            return False
        return all(torch.is_tensor(v) for v in x.values())

    @classmethod
    def _iter_state_dict_candidates(cls, state: Any) -> Iterable[dict]:
        if cls._is_tensor_dict(state):
            yield dict(state)
        if isinstance(state, dict):
            for key in ("state_dict", "model", "encoder", "backbone", "resnet", "net"):
                sub = state.get(key)
                if cls._is_tensor_dict(sub):
                    yield dict(sub)

    @staticmethod
    def _strip_prefix(state_dict: dict, prefix: str) -> dict:
        return {
            (k[len(prefix) :] if k.startswith(prefix) else k): v
            for k, v in state_dict.items()
        }

    @classmethod
    def _candidate_key_variants(cls, state_dict: dict) -> Iterable[dict]:
        prefixes = ("module.", "model.", "encoder.", "backbone.", "resnet.", "net.")
        seen = set()
        queue = [dict(state_dict)]

        while queue:
            cur = queue.pop(0)
            key_sig = tuple(sorted(cur.keys()))
            if key_sig in seen:
                continue
            seen.add(key_sig)
            yield cur
            for p in prefixes:
                if any(k.startswith(p) for k in cur.keys()):
                    nxt = cls._strip_prefix(cur, p)
                    queue.append(nxt)

    @staticmethod
    def _full_resnet_keys() -> list[str]:
        return list(models.resnet18(weights=None).state_dict().keys())

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (B,3,H,W) tensor in [0,1] or [0,255].
        Returns:
            latent token: (B, latent_dim)
        """
        if image.ndim != 4:
            raise ValueError(f"Expected image shape (B,3,H,W), got {tuple(image.shape)}")
        x = image.float()
        if x.max() > 1.5:
            x = x / 255.0
        if self.normalize_input:
            x = (x - self.rgb_mean.to(x.device)) / self.rgb_std.to(x.device)
        z = self.backbone(x)
        return self.flatten(z)


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


class DynamicsPredictor(nn.Module):
    """
    Decoder-only transformer transition model f_phi.
    Input tokens: [z_t, s_t, a_t, ..., a_{t+h-1}, <pred_z>, <pred_s>]
    Output heads read from the final two tokens.
    """

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
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.proprio_dim = int(proprio_dim)
        self.action_dim = int(action_dim)
        self.max_action_horizon = int(max_action_horizon)

        self.obs_proj = nn.Linear(self.latent_dim, d_model)
        self.proprio_proj = nn.Linear(self.proprio_dim, d_model)
        self.action_proj = nn.Linear(self.action_dim, d_model)

        # Query tokens used to decode predicted future latent/proprio.
        self.pred_latent_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pred_proprio_token = nn.Parameter(torch.zeros(1, 1, d_model))

        max_seq_len = 2 + self.max_action_horizon + 2
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
        self.proprio_head = nn.Linear(d_model, self.proprio_dim)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask

    def forward(
        self,
        obs_token: torch.Tensor,
        proprio_token: torch.Tensor,
        action_tokens: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            obs_token: (B, latent_dim)
            proprio_token: (B, proprio_dim)
            action_tokens: (B, H, action_dim)
        Returns:
            dict:
                pred_latent: (B, latent_dim)
                pred_proprio: (B, proprio_dim)
        """
        if action_tokens.ndim != 3:
            raise ValueError(
                f"Expected action_tokens shape (B,H,A), got {tuple(action_tokens.shape)}"
            )
        bsz, horizon, _ = action_tokens.shape
        if horizon > self.max_action_horizon:
            raise ValueError(
                f"action horizon {horizon} exceeds max_action_horizon={self.max_action_horizon}"
            )

        obs = self.obs_proj(obs_token).unsqueeze(1)  # (B,1,D)
        prop = self.proprio_proj(proprio_token).unsqueeze(1)  # (B,1,D)
        act = self.action_proj(action_tokens)  # (B,H,D)

        pred_z_tok = self.pred_latent_token.expand(bsz, -1, -1)
        pred_s_tok = self.pred_proprio_token.expand(bsz, -1, -1)

        x = torch.cat([obs, prop, act, pred_z_tok, pred_s_tok], dim=1)
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]
        x = self.drop(x)

        mask = self._causal_mask(seq_len=seq_len, device=x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask=mask)
        x = self.norm(x)

        pred_latent = self.latent_head(x[:, -2, :])
        pred_proprio = self.proprio_head(x[:, -1, :])
        return {"pred_latent": pred_latent, "pred_proprio": pred_proprio}


class DynamicsModel(nn.Module):
    """
    Visual latent dynamics model d_phi = (h_theta, f_phi).
    """

    def __init__(
        self,
        encoder: Encoder,
        predictor: DynamicsPredictor,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor

    def _encoder_is_trainable(self) -> bool:
        return any(p.requires_grad for p in self.encoder.parameters())

    def encode_observation(self, image: torch.Tensor) -> torch.Tensor:
        if self._encoder_is_trainable() and self.training:
            return self.encoder(image)
        with torch.no_grad():
            return self.encoder(image)

    def forward(
        self,
        current_image: torch.Tensor,
        current_proprio: torch.Tensor,
        action_sequence: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        z_t = self.encode_observation(current_image)
        return self.predictor(
            obs_token=z_t,
            proprio_token=current_proprio,
            action_tokens=action_sequence,
        )

    def compute_dynamics_loss(
        self,
        current_image: torch.Tensor,
        current_proprio: torch.Tensor,
        action_sequence: torch.Tensor,
        target_image: torch.Tensor,
        target_proprio: Optional[torch.Tensor] = None,
        proprio_loss_weight: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        pred = self.forward(current_image, current_proprio, action_sequence)
        z_target = self.encode_observation(target_image).detach()

        latent_mse = F.mse_loss(pred["pred_latent"], z_target)
        total = latent_mse

        proprio_mse: Optional[torch.Tensor] = None
        if target_proprio is not None:
            proprio_mse = F.mse_loss(pred["pred_proprio"], target_proprio)
            total = total + float(proprio_loss_weight) * proprio_mse

        out: Dict[str, Any] = {
            "loss": total,
            "latent_mse": latent_mse,
            "pred_latent": pred["pred_latent"],
            "target_latent": z_target,
            "pred_proprio": pred["pred_proprio"],
        }
        if proprio_mse is not None:
            out["proprio_mse"] = proprio_mse
        return out
