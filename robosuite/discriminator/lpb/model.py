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
        nhead: int = 8,
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
    Input tokens: [z_t, s_t, a_t, ..., a_{t+h-1}, <act_summary>, <pred_z>, <pred_s>]
    Output heads fuse query states with summary/context tokens.
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
        fusion_hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.proprio_dim = int(proprio_dim)
        self.action_dim = int(action_dim)
        self.max_action_horizon = int(max_action_horizon)
        self.fusion_hidden_dim = int(fusion_hidden_dim)

        self.obs_proj = nn.Linear(self.latent_dim, d_model)
        self.proprio_proj = nn.Linear(self.proprio_dim, d_model)
        self.action_proj = nn.Linear(self.action_dim, d_model)
        self.action_summary_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Query tokens used to decode predicted future latent/proprio.
        self.action_summary_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pred_latent_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pred_proprio_token = nn.Parameter(torch.zeros(1, 1, d_model))

        max_seq_len = 2 + self.max_action_horizon + 3
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
        self.shared_feature_dim = d_model * 3
        self.shared_feature_norm = nn.LayerNorm(self.shared_feature_dim)
        self.latent_fusion = nn.Sequential(
            nn.Linear(d_model * 3, self.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(self.fusion_hidden_dim),
            nn.Linear(self.fusion_hidden_dim, d_model),
            nn.GELU(),
        )
        self.proprio_fusion = nn.Sequential(
            nn.Linear(d_model * 3, self.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(self.fusion_hidden_dim),
            nn.Linear(self.fusion_hidden_dim, d_model),
            nn.GELU(),
        )
        self.latent_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, self.latent_dim),
        )
        self.proprio_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, self.proprio_dim),
        )

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
        act_summary = self.action_summary_proj(act.mean(dim=1, keepdim=True))

        summary_tok = self.action_summary_token.expand(bsz, -1, -1) + act_summary
        pred_z_tok = self.pred_latent_token.expand(bsz, -1, -1)
        pred_s_tok = self.pred_proprio_token.expand(bsz, -1, -1)

        x = torch.cat([obs, prop, act, summary_tok, pred_z_tok, pred_s_tok], dim=1)
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]
        x = self.drop(x)

        mask = self._causal_mask(seq_len=seq_len, device=x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask=mask)
        x = self.norm(x)

        summary_state = x[:, -(2 + 1), :]
        shared_feature = self.shared_feature_norm(
            torch.cat([x[:, 0, :], x[:, 1, :], summary_state], dim=-1)
        )
        latent_state = self.latent_fusion(
            torch.cat([x[:, -2, :], summary_state, x[:, 0, :]], dim=-1)
        )
        proprio_state = self.proprio_fusion(
            torch.cat([x[:, -1, :], summary_state, x[:, 1, :]], dim=-1)
        )

        pred_latent = self.latent_head(latent_state)
        pred_proprio = self.proprio_head(proprio_state)
        return {
            "pred_latent": pred_latent,
            "pred_proprio": pred_proprio,
            "shared_feature": shared_feature,
        }


class DynamicsModel(nn.Module):
    """
    Visual latent dynamics model d_phi = (h_theta, f_phi).
    """

    def __init__(
        self,
        encoder: Encoder,
        predictor: DynamicsPredictor,
        projection_dim: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.projection_dim = int(projection_dim)
        self.feature_dim = int(self.predictor.shared_feature_dim)
        if self.projection_dim > 0 and self.projection_dim != self.feature_dim:
            self.contrastive_head: Optional[nn.Module] = nn.Linear(
                self.feature_dim,
                self.projection_dim,
                bias=False,
            )
        else:
            self.contrastive_head = None
        self._expert_feature_queue: Optional[torch.Tensor] = None
        self._negative_feature_queue: Optional[torch.Tensor] = None
        self._negative_weight_queue: Optional[torch.Tensor] = None

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

    def encode_shared_feature(
        self,
        current_image: torch.Tensor,
        current_proprio: torch.Tensor,
        action_sequence: torch.Tensor,
        normalize: bool = False,
    ) -> torch.Tensor:
        pred = self.forward(current_image, current_proprio, action_sequence)
        feat = pred["shared_feature"]
        if normalize:
            feat = F.normalize(feat, p=2.0, dim=-1)
        return feat

    def project_feature(self, feature: torch.Tensor) -> torch.Tensor:
        if self.contrastive_head is not None:
            feature = self.contrastive_head(feature)
        return F.normalize(feature, p=2.0, dim=-1)

    def reset_contrastive_queue(self) -> None:
        self._expert_feature_queue = None
        self._negative_feature_queue = None
        self._negative_weight_queue = None

    def _concat_queue(
        self,
        current: Optional[torch.Tensor],
        queue: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if current is None:
            return queue
        if queue is None:
            return current
        return torch.cat([current, queue.to(device=current.device, dtype=current.dtype)], dim=0)

    def _enqueue_contrastive_features(
        self,
        expert_features: torch.Tensor,
        negative_features: torch.Tensor,
        negative_weights: torch.Tensor,
        queue_size: int,
    ) -> None:
        max_queue = int(queue_size)
        if max_queue <= 0:
            return

        if expert_features.numel() > 0:
            expert_detached = expert_features.detach()
            self._expert_feature_queue = self._concat_queue(
                expert_detached,
                self._expert_feature_queue,
            )
            assert self._expert_feature_queue is not None
            self._expert_feature_queue = self._expert_feature_queue[:max_queue]

        if negative_features.numel() > 0:
            negative_detached = negative_features.detach()
            weight_detached = negative_weights.detach()
            self._negative_feature_queue = self._concat_queue(
                negative_detached,
                self._negative_feature_queue,
            )
            self._negative_weight_queue = self._concat_queue(
                weight_detached,
                self._negative_weight_queue,
            )
            assert self._negative_feature_queue is not None
            assert self._negative_weight_queue is not None
            self._negative_feature_queue = self._negative_feature_queue[:max_queue]
            self._negative_weight_queue = self._negative_weight_queue[:max_queue]

    def _pu_contrastive_loss(
        self,
        shared_feature: torch.Tensor,
        is_expert: Optional[torch.Tensor],
        failure_confidence: Optional[torch.Tensor],
        temperature: float,
        negative_confidence_threshold: float,
        queue_size: int,
    ) -> Dict[str, torch.Tensor]:
        zero = shared_feature.new_zeros(())
        if is_expert is None or failure_confidence is None:
            return {
                "loss": zero,
                "positive_count": zero,
                "negative_count": zero,
                "negative_weight_mean": zero,
                "feature_norm": zero,
            }

        expert_mask = is_expert.reshape(-1).to(dtype=torch.bool, device=shared_feature.device)
        if int(expert_mask.sum().item()) == 0:
            return {
                "loss": zero,
                "positive_count": zero,
                "negative_count": zero,
                "negative_weight_mean": zero,
                "feature_norm": zero,
            }

        features = self.project_feature(shared_feature)
        feature_norm = shared_feature.norm(dim=-1).mean()
        anchors = features[expert_mask]
        current_pos = features[expert_mask].detach()

        neg_weights_cur = failure_confidence.reshape(-1).to(dtype=features.dtype, device=features.device)
        neg_weights_cur = torch.where(
            neg_weights_cur >= float(negative_confidence_threshold),
            neg_weights_cur,
            torch.zeros_like(neg_weights_cur),
        )
        neg_mask = neg_weights_cur > 0
        current_neg = features[neg_mask].detach()
        current_neg_weights = neg_weights_cur[neg_mask].detach()

        use_queue = self.training and int(queue_size) > 0
        pos_bank = current_pos
        neg_bank = current_neg
        neg_weight_bank = current_neg_weights
        if use_queue:
            pos_bank = self._concat_queue(pos_bank, self._expert_feature_queue)
            neg_bank = self._concat_queue(neg_bank, self._negative_feature_queue)
            neg_weight_bank = self._concat_queue(neg_weight_bank, self._negative_weight_queue)

        pos_count = 0 if pos_bank is None else int(pos_bank.shape[0])
        neg_count = 0 if neg_bank is None else int(neg_bank.shape[0])
        if pos_count <= 1 or neg_count <= 0:
            if use_queue:
                self._enqueue_contrastive_features(
                    expert_features=current_pos,
                    negative_features=current_neg,
                    negative_weights=current_neg_weights,
                    queue_size=int(queue_size),
                )
            return {
                "loss": zero,
                "positive_count": shared_feature.new_tensor(float(pos_count)),
                "negative_count": shared_feature.new_tensor(float(neg_count)),
                "negative_weight_mean": (
                    neg_weight_bank.mean() if neg_weight_bank is not None and neg_weight_bank.numel() > 0 else zero
                ),
                "feature_norm": feature_norm,
            }

        logits_pos = anchors @ pos_bank.transpose(0, 1)
        logits_pos = logits_pos / max(float(temperature), 1e-6)
        logits_neg = anchors @ neg_bank.transpose(0, 1)
        logits_neg = logits_neg / max(float(temperature), 1e-6)

        if current_pos.shape[0] > 0:
            diag_n = min(logits_pos.shape[0], current_pos.shape[0])
            diag_idx = torch.arange(diag_n, device=logits_pos.device)
            logits_pos[diag_idx, diag_idx] = logits_pos[diag_idx, diag_idx] - 1e9

        max_logit = torch.maximum(
            logits_pos.max(dim=1, keepdim=True).values,
            logits_neg.max(dim=1, keepdim=True).values,
        )
        pos_mass = torch.exp(logits_pos - max_logit).sum(dim=1)
        neg_mass = (
            torch.exp(logits_neg - max_logit)
            * neg_weight_bank.reshape(1, -1)
        ).sum(dim=1)

        valid = (pos_mass > 0) & (neg_mass > 0)
        if not torch.any(valid):
            if use_queue:
                self._enqueue_contrastive_features(
                    expert_features=current_pos,
                    negative_features=current_neg,
                    negative_weights=current_neg_weights,
                    queue_size=int(queue_size),
                )
            return {
                "loss": zero,
                "positive_count": shared_feature.new_tensor(float(pos_count)),
                "negative_count": shared_feature.new_tensor(float(neg_count)),
                "negative_weight_mean": neg_weight_bank.mean(),
                "feature_norm": feature_norm,
            }

        loss = -torch.log((pos_mass[valid] + 1e-8) / (pos_mass[valid] + neg_mass[valid] + 1e-8))

        if use_queue:
            self._enqueue_contrastive_features(
                expert_features=current_pos,
                negative_features=current_neg,
                negative_weights=current_neg_weights,
                queue_size=int(queue_size),
            )

        return {
            "loss": loss.mean(),
            "positive_count": shared_feature.new_tensor(float(pos_count)),
            "negative_count": shared_feature.new_tensor(float(neg_count)),
            "negative_weight_mean": neg_weight_bank.mean(),
            "feature_norm": feature_norm,
        }

    def compute_dynamics_loss(
        self,
        current_image: torch.Tensor,
        current_proprio: torch.Tensor,
        action_sequence: torch.Tensor,
        target_image: torch.Tensor,
        target_proprio: Optional[torch.Tensor] = None,
        proprio_loss_weight: float = 0.0,
        is_expert: Optional[torch.Tensor] = None,
        failure_confidence: Optional[torch.Tensor] = None,
        contrastive_loss_weight: float = 0.0,
        contrastive_temperature: float = 0.1,
        negative_confidence_threshold: float = 0.0,
        contrastive_queue_size: int = 0,
    ) -> Dict[str, torch.Tensor]:
        pred = self.forward(current_image, current_proprio, action_sequence)
        z_target = self.encode_observation(target_image).detach()

        latent_mse = F.mse_loss(pred["pred_latent"], z_target)
        total = latent_mse

        proprio_mse: Optional[torch.Tensor] = None
        if target_proprio is not None:
            proprio_mse = F.mse_loss(pred["pred_proprio"], target_proprio)
            total = total + float(proprio_loss_weight) * proprio_mse

        contrastive_stats = self._pu_contrastive_loss(
            shared_feature=pred["shared_feature"],
            is_expert=is_expert,
            failure_confidence=failure_confidence,
            temperature=contrastive_temperature,
            negative_confidence_threshold=negative_confidence_threshold,
            queue_size=contrastive_queue_size,
        )
        contrastive_loss = contrastive_stats["loss"]
        total = total + float(contrastive_loss_weight) * contrastive_loss

        out: Dict[str, Any] = {
            "loss": total,
            "latent_mse": latent_mse,
            "contrastive_loss": contrastive_loss,
            "contrastive_positive_count": contrastive_stats["positive_count"],
            "contrastive_negative_count": contrastive_stats["negative_count"],
            "contrastive_negative_weight_mean": contrastive_stats["negative_weight_mean"],
            "shared_feature_norm": contrastive_stats["feature_norm"],
            "pred_latent": pred["pred_latent"],
            "target_latent": z_target,
            "pred_proprio": pred["pred_proprio"],
            "shared_feature": pred["shared_feature"],
        }
        if proprio_mse is not None:
            out["proprio_mse"] = proprio_mse
        return out
