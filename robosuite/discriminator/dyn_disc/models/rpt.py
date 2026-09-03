"""RPT-style masked sensorimotor representation model."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RPTModel(nn.Module):
    """Bidirectional masked-token Transformer over an eight-frame context."""

    modality_names = ("agentview", "robot0_eye_in_hand", "proprio", "action")

    def __init__(
        self,
        visual_dim: int = 768,
        proprio_dim: int = 14,
        action_dim: int = 7,
        hidden_dim: int = 192,
        context_length: int = 8,
        depth: int = 4,
        heads: int = 4,
        mlp_dim: int = 384,
        dropout: float = 0.0,
        view_names: Sequence[str] = ("agentview", "robot0_eye_in_hand"),
        slot_position_embedding: str = "learned",
        mask_ratio_min: float = 0.7,
        mask_ratio_max: float = 0.9,
    ) -> None:
        super().__init__()
        if not 0.0 <= mask_ratio_min <= mask_ratio_max <= 1.0:
            raise ValueError("Mask ratio bounds must satisfy 0 <= min <= max <= 1")
        if tuple(view_names) != self.modality_names[:2]:
            raise ValueError(f"RPT requires view order {self.modality_names[:2]}")
        if float(dropout) != 0.0:
            raise ValueError("RPT ablation requires dropout=0")
        if str(slot_position_embedding) != "learned":
            raise ValueError("RPT requires slot_position_embedding='learned'")
        self.visual_dim = int(visual_dim)
        self.proprio_dim = int(proprio_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.context_length = int(context_length)
        self.depth = int(depth)
        self.heads = int(heads)
        self.mlp_dim = int(mlp_dim)
        self.dropout = float(dropout)
        self.view_names = tuple(str(name) for name in view_names)
        self.slot_position_embedding = str(slot_position_embedding)
        self.mask_ratio_min = float(mask_ratio_min)
        self.mask_ratio_max = float(mask_ratio_max)

        self.visual_projection = nn.Linear(self.visual_dim, self.hidden_dim)
        self.proprio_projection = nn.Linear(self.proprio_dim, self.hidden_dim)
        self.action_projection = nn.Linear(self.action_dim, self.hidden_dim)
        self.visual_mask_token = nn.Parameter(torch.zeros(self.hidden_dim))
        self.proprio_mask_token = nn.Parameter(torch.zeros(self.hidden_dim))
        self.action_mask_token = nn.Parameter(torch.zeros(self.hidden_dim))
        self.temporal_position = nn.Parameter(torch.zeros(self.context_length, self.hidden_dim))
        self.slot_position = nn.Parameter(torch.zeros(4, self.hidden_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.heads,
            dim_feedforward=self.mlp_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=self.depth, enable_nested_tensor=False)
        self.visual_head = nn.Linear(self.hidden_dim, self.visual_dim)
        self.proprio_head = nn.Linear(self.hidden_dim, self.proprio_dim)
        self.action_head = nn.Linear(self.hidden_dim, self.action_dim)
        nn.init.normal_(self.visual_mask_token, std=0.02)
        nn.init.normal_(self.proprio_mask_token, std=0.02)
        nn.init.normal_(self.action_mask_token, std=0.02)
        nn.init.normal_(self.temporal_position, std=0.02)
        nn.init.normal_(self.slot_position, std=0.02)

    def architecture_metadata(self) -> dict:
        return {
            "visual_dim": self.visual_dim,
            "proprio_dim": self.proprio_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "context_length": self.context_length,
            "depth": self.depth,
            "heads": self.heads,
            "mlp_dim": self.mlp_dim,
            "dropout": self.dropout,
            "view_names": list(self.view_names),
            "slot_position_embedding": self.slot_position_embedding,
            "mask_ratio_min": self.mask_ratio_min,
            "mask_ratio_max": self.mask_ratio_max,
            "token_order": list(self.modality_names),
        }

    def _validate_inputs(
        self, visual_latents: torch.Tensor, proprio: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[int, int]:
        if visual_latents.ndim != 4 or visual_latents.shape[2:] != (2, self.visual_dim):
            raise ValueError(
                f"Expected visual_latents [B,T,2,{self.visual_dim}], got {tuple(visual_latents.shape)}"
            )
        b, t = visual_latents.shape[:2]
        if t != self.context_length:
            raise ValueError(f"Expected context length {self.context_length}, got {t}")
        if proprio.shape != (b, t, self.proprio_dim):
            raise ValueError(f"Expected proprio {(b, t, self.proprio_dim)}, got {tuple(proprio.shape)}")
        if actions.shape != (b, t, self.action_dim):
            raise ValueError(f"Expected actions {(b, t, self.action_dim)}, got {tuple(actions.shape)}")
        if not (visual_latents.device == proprio.device == actions.device):
            raise ValueError("All RPT inputs must be on the same device")
        return b, t

    def sample_mask(self, batch_size: int, *, device: torch.device) -> torch.Tensor:
        probabilities = torch.empty(batch_size, 1, device=device).uniform_(
            self.mask_ratio_min, self.mask_ratio_max
        )
        mask = torch.rand(batch_size, self.context_length * 4, device=device) < probabilities
        all_visible = ~mask.any(dim=1)
        all_masked = mask.all(dim=1)
        mask[:, 0].logical_or_(all_visible)
        mask[:, 0].logical_and_(~all_masked)
        return mask.view(batch_size, self.context_length, 4)

    def _encode_tokens(
        self,
        visual_latents: torch.Tensor,
        proprio: torch.Tensor,
        actions: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        visual = self.visual_projection(visual_latents)
        tokens = torch.stack(
            (visual[:, :, 0], visual[:, :, 1], self.proprio_projection(proprio), self.action_projection(actions)),
            dim=2,
        )
        if mask is not None:
            mask_tokens = torch.stack(
                (self.visual_mask_token, self.visual_mask_token, self.proprio_mask_token, self.action_mask_token),
                dim=0,
            )
            tokens = torch.where(mask.unsqueeze(-1), mask_tokens.view(1, 1, 4, -1), tokens)
        tokens = (
            tokens
            + self.temporal_position.view(1, self.context_length, 1, self.hidden_dim)
            + self.slot_position.view(1, 1, 4, self.hidden_dim)
        )
        return self.transformer(tokens.flatten(1, 2)).view(
            tokens.shape[0], self.context_length, 4, self.hidden_dim
        )

    @staticmethod
    def _masked_mse(
        prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return masked mean and a scalar validity flag without host sync."""
        per_token = F.mse_loss(prediction.float(), target.float(), reduction="none").mean(dim=-1)
        weights = mask.to(dtype=per_token.dtype)
        count = weights.sum()
        loss = (per_token * weights).sum() / count.clamp_min(1.0)
        return loss, (count > 0).to(dtype=loss.dtype)

    def forward(
        self,
        visual_latents: torch.Tensor,
        proprio: torch.Tensor,
        actions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, _ = self._validate_inputs(visual_latents, proprio, actions)
        sampled_mask = mask is None
        if sampled_mask:
            mask = self.sample_mask(batch_size, device=visual_latents.device)
        elif mask.shape != (batch_size, self.context_length, 4) or mask.dtype != torch.bool:
            raise ValueError(
                f"mask must be bool [{batch_size},{self.context_length},4], got {tuple(mask.shape)} {mask.dtype}"
            )
        if not sampled_mask and (
            not torch.all(mask.flatten(1).any(dim=1))
            or not torch.all((~mask).flatten(1).any(dim=1))
        ):
            raise ValueError("Every sample must contain at least one masked and one visible token")

        hidden = self._encode_tokens(visual_latents, proprio, actions, mask)
        visual_pred = self.visual_head(hidden[:, :, :2])
        proprio_pred = self.proprio_head(hidden[:, :, 2])
        action_pred = self.action_head(hidden[:, :, 3])
        losses_and_validity = {
            "visual": self._masked_mse(visual_pred, visual_latents, mask[:, :, :2]),
            "proprio": self._masked_mse(proprio_pred, proprio, mask[:, :, 2]),
            "action": self._masked_mse(action_pred, actions, mask[:, :, 3]),
        }
        losses = {name: value[0] for name, value in losses_and_validity.items()}
        validity = torch.stack([value[1] for value in losses_and_validity.values()])
        total = (
            torch.stack(list(losses.values())) * validity
        ).sum() / validity.sum().clamp_min(1.0)
        components = {
            "total": total,
            "visual": losses["visual"],
            "proprio": losses["proprio"],
            "action": losses["action"],
            "mask_ratio": mask.float().mean(),
        }
        return total, components

    def extract_features(
        self, visual_latents: torch.Tensor, proprio: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Return the final time step's action-token hidden state ``[B,192]``."""
        self._validate_inputs(visual_latents, proprio, actions)
        return self._encode_tokens(visual_latents, proprio, actions, mask=None)[:, -1, 3]
