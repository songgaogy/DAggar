from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from robosuite.pipeline.algorithms.flow_dagger.common import (
    FlowAugmentationConfig,
    FlowDaggerConfig,
)
from robosuite.pipeline.common.types import (
    EncoderConfig,
    ReplayBufferConfig,
    Transition,
)


@dataclass
class DipoleConfig(FlowDaggerConfig):
    # CFG-style guidance + sigmoid weighting params.
    # Coupled soft weights: w_pos = sigmoid(beta * (G + k)), w_neg = 1 - w_pos.
    # k offsets G before scaling (w_pos=0.5 at G=-k); beta is the post-offset slope.
    beta: float = 2.0
    k: float = 0.0
    guidance_omega: float = 2.0          # eval-only: v=(1+w)v_pos - w v_neg (rollout is pos-only)
    g_clip: float = 10.0                 # unused by current _g_weights_from_raw
    # Two independent, fully finetuned flow policies (positive + negative). Each is
    # trained full-tune under the base flow-policy freeze regime; there are no LoRA
    # adapters. The learning rate / weight decay come from the inherited
    # ``learning_rate`` / ``weight_decay`` (FlowDaggerConfig).
    # Selects the frozen advantage provider attached during the policy stage.
    g_mode: str = "nnpu_frozen"
    # Branch-weight scheme consumed by DipoleFlowPolicy._compute_branch_weights:
    #   "coupled" (default) -> w_neg = 1 - w_pos with the intervention override
    #     (normal / naive offline modes);
    #   "neg_all" -> decoupled hard labels: w_pos = is_intervention, and w_neg is
    #     the attached provider's per-frame membership (0/1), NOT 1 - w_pos and
    #     NOT zeroed on intervention rows. Used by offline.mode == "neg_all" so
    #     success frames drive BOTH branches (w_pos = w_neg = 1).
    branch_weight_mode: str = "coupled"


@dataclass
class TrainerConfig:
    batch_size: int = 64


@dataclass
class DipoleBatch:
    image_obs: torch.Tensor                # ImageNet-normalized for policy
    image_obs_raw: torch.Tensor            # [0,1] floats, for dynamics encoder
    proprio: torch.Tensor                  # policy-normalized
    proprio_raw: torch.Tensor              # un-normalized for dynamics encoder
    action_sequences: torch.Tensor         # policy-normalized
    action_sequences_raw: torch.Tensor     # un-normalized for dynamics encoder
    is_intervention: torch.Tensor          # (B,) bool/float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to(self, device: torch.device | str) -> "DipoleBatch":
        return DipoleBatch(
            image_obs=self.image_obs.to(device),
            image_obs_raw=self.image_obs_raw.to(device),
            proprio=self.proprio.to(device),
            proprio_raw=self.proprio_raw.to(device),
            action_sequences=self.action_sequences.to(device),
            action_sequences_raw=self.action_sequences_raw.to(device),
            is_intervention=self.is_intervention.to(device),
            metadata=self.metadata,
        )

    @property
    def batch_size(self) -> int:
        return int(self.action_sequences.shape[0])


def select_dipole_batch(batch: DipoleBatch, indices: torch.Tensor) -> DipoleBatch:
    """Select routed rows along the batch dimension."""
    if indices.numel() == 0:
        raise ValueError("select_dipole_batch requires at least one index.")
    idx = indices.to(device=batch.image_obs.device, dtype=torch.long).reshape(-1)
    index_list = idx.detach().cpu().tolist()
    metadata: dict[str, Any] = {}
    for key in ("start_indices", "episode_ids", "episode_steps", "route"):
        values = batch.metadata.get(key)
        if values is None:
            continue
        metadata[key] = [values[int(i)] for i in index_list]
    return DipoleBatch(
        image_obs=batch.image_obs.index_select(0, idx),
        image_obs_raw=batch.image_obs_raw.index_select(0, idx),
        proprio=batch.proprio.index_select(0, idx),
        proprio_raw=batch.proprio_raw.index_select(0, idx),
        action_sequences=batch.action_sequences.index_select(0, idx),
        action_sequences_raw=batch.action_sequences_raw.index_select(0, idx),
        is_intervention=batch.is_intervention.index_select(0, idx),
        metadata=metadata,
    )


__all__ = [
    "DipoleBatch",
    "select_dipole_batch",
    "DipoleConfig",
    "EncoderConfig",
    "FlowAugmentationConfig",
    "ReplayBufferConfig",
    "TrainerConfig",
    "Transition",
]
