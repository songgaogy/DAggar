from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from robosuite.discriminator.lpb.model import DecoderBlock, Encoder


class ContrastiveContextEncoder(nn.Module):
    """
    Transformer encoder over current observation + proprio + action history.
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

        self.action_summary_token = nn.Parameter(torch.zeros(1, 1, d_model))
        max_seq_len = 2 + self.max_action_horizon + 1
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
        self.shared_feature_fusion = nn.Sequential(
            nn.Linear(d_model * 3, self.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(self.fusion_hidden_dim),
            nn.Linear(self.fusion_hidden_dim, self.shared_feature_dim),
        )

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    def forward(
        self,
        obs_token: torch.Tensor,
        proprio_token: torch.Tensor,
        action_tokens: torch.Tensor,
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

        obs = self.obs_proj(obs_token).unsqueeze(1)
        prop = self.proprio_proj(proprio_token).unsqueeze(1)
        act = self.action_proj(action_tokens)
        act_summary = self.action_summary_proj(act.mean(dim=1, keepdim=True))
        summary_tok = self.action_summary_token.expand(bsz, -1, -1) + act_summary

        x = torch.cat([obs, prop, act, summary_tok], dim=1)
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]
        x = self.drop(x)

        mask = self._causal_mask(seq_len=seq_len, device=x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask=mask)
        x = self.norm(x)

        summary_state = x[:, -1, :]
        fused = self.shared_feature_fusion(
            torch.cat([x[:, 0, :], x[:, 1, :], summary_state], dim=-1)
        )
        shared_feature = self.shared_feature_norm(fused)
        return {
            "shared_feature": shared_feature,
            "summary_state": summary_state,
        }


class PureContrastiveModel(nn.Module):
    """
    Pure contrastive discriminator model using LPB-style context encoding.
    """

    def __init__(
        self,
        encoder: Encoder,
        context_encoder: ContrastiveContextEncoder,
        projection_dim: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.context_encoder = context_encoder
        self.projection_dim = int(projection_dim)
        self.feature_dim = int(self.context_encoder.shared_feature_dim)
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
        return self.context_encoder(
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
        out = self.forward(current_image, current_proprio, action_sequence)
        feat = out["shared_feature"]
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

        neg_weights_cur = failure_confidence.reshape(-1).to(
            dtype=features.dtype,
            device=features.device,
        )
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
                    neg_weight_bank.mean()
                    if neg_weight_bank is not None and neg_weight_bank.numel() > 0
                    else zero
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
            torch.exp(logits_neg - max_logit) * neg_weight_bank.reshape(1, -1)
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

        loss = -torch.log(
            (pos_mass[valid] + 1e-8) / (pos_mass[valid] + neg_mass[valid] + 1e-8)
        )

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

    def compute_contrastive_loss(
        self,
        current_image: torch.Tensor,
        current_proprio: torch.Tensor,
        action_sequence: torch.Tensor,
        is_expert: Optional[torch.Tensor] = None,
        failure_confidence: Optional[torch.Tensor] = None,
        contrastive_temperature: float = 0.1,
        negative_confidence_threshold: float = 0.0,
        contrastive_queue_size: int = 0,
    ) -> Dict[str, torch.Tensor]:
        pred = self.forward(current_image, current_proprio, action_sequence)
        contrastive_stats = self._pu_contrastive_loss(
            shared_feature=pred["shared_feature"],
            is_expert=is_expert,
            failure_confidence=failure_confidence,
            temperature=contrastive_temperature,
            negative_confidence_threshold=negative_confidence_threshold,
            queue_size=contrastive_queue_size,
        )
        return {
            "loss": contrastive_stats["loss"],
            "contrastive_loss": contrastive_stats["loss"],
            "contrastive_positive_count": contrastive_stats["positive_count"],
            "contrastive_negative_count": contrastive_stats["negative_count"],
            "contrastive_negative_weight_mean": contrastive_stats["negative_weight_mean"],
            "shared_feature_norm": contrastive_stats["feature_norm"],
            "shared_feature": pred["shared_feature"],
        }
