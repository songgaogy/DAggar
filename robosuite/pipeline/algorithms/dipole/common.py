from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

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
class LPBDetectorConfig:
    """Pre-fitted BCE detector wiring for DIPOLE.

    ``ckpt_path`` points at the artefact produced by
    ``robosuite/discriminator/lpb_v2/scripts/run_bce_robosuite_benchmark.sh``
    (or its visualize_ companion). All other detector-side hyperparameters
    (feature_source, transformer_layer, head shape, thresholds, encoder ckpt
    path, view_names, frameskip, action_dim_per_step) are read from the
    artefact itself. ``camera_to_view`` lets you remap policy camera names to
    encoder view names when they differ.
    """

    ckpt_path: Optional[str] = None
    device: str = "cuda:0"
    camera_to_view: dict[str, str] = field(default_factory=dict)


@dataclass
class DipoleConfig(FlowDaggerConfig):
    # CFG-style guidance + sigmoid weighting params
    beta: float = 2.0
    k: float = 0.0
    guidance_omega: float = 2.0
    g_sign: str = "negate_raw"            # "negate_raw" | "raw"
    g_normalization: str = "batch_zscore" # "batch_zscore" | "running_zscore" | "minmax" | "none"
    g_clip: float = 10.0
    polarity_embedding_init: str = "small_gaussian"   # "small_gaussian" | "zero_neg" | "antipodal"
    polarity_embedding_init_scale: float = 1e-3
    lpb_detector: LPBDetectorConfig = field(default_factory=LPBDetectorConfig)


@dataclass
class TrainerConfig:
    batch_size: int = 64
    warmup_steps: int = 0
    updates_per_step: int = 1
    steps_per_update: int = 50
    random_steps: int = 0
    pretrain_steps: int = 20_000


@dataclass
class DipoleBatch:
    image_obs: torch.Tensor                # ImageNet-normalized for policy
    image_obs_raw: torch.Tensor            # [0,1] floats, for LPB encoder
    proprio: torch.Tensor                  # policy-normalized
    proprio_raw: torch.Tensor              # un-normalized for LPB encoder
    action_sequences: torch.Tensor         # policy-normalized
    action_sequences_raw: torch.Tensor     # un-normalized for LPB encoder
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


__all__ = [
    "DipoleBatch",
    "DipoleConfig",
    "EncoderConfig",
    "FlowAugmentationConfig",
    "LPBDetectorConfig",
    "ReplayBufferConfig",
    "TrainerConfig",
    "Transition",
]
