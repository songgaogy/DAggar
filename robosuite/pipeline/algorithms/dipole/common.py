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
    polarity_embedding_init: str = "zero_pos"   # "zero_pos" | "zero_neg" | "small_gaussian" | "antipodal"
    polarity_embedding_init_scale: float = 1e-3
    lpb_detector: LPBDetectorConfig = field(default_factory=LPBDetectorConfig)
    # DIPOLE-RL: which G provider the trainer should attach.
    # "bce_frozen" -> LPBV2GProvider (legacy); "advantage" -> AdvantageGProvider.
    # Read by train_dipole_rl.py; DipoleFlowPolicy itself does not consume it.
    g_mode: str = "bce_frozen"


@dataclass
class TrainerConfig:
    batch_size: int = 64
    warmup_steps: int = 0
    updates_per_step: int = 1
    steps_per_update: int = 50
    pretrain_steps: int = 20_000
    max_pending_updates: int = 1


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


def select_dipole_batch(batch: DipoleBatch, indices: torch.Tensor) -> DipoleBatch:
    """Row subset along batch dim (for online-only G computation)."""
    if indices.numel() == 0:
        raise ValueError("select_dipole_batch requires at least one index.")
    idx = indices.to(device=batch.image_obs.device, dtype=torch.long).reshape(-1)
    index_list = idx.detach().cpu().tolist()
    metadata: dict[str, Any] = {}
    for key in ("start_indices", "episode_ids", "episode_steps", "buffer_sources"):
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


def concat_dipole_batches(*batches: DipoleBatch) -> DipoleBatch:
    """Concatenate batches along dim=0 (e.g. online + demo halves)."""
    if len(batches) == 0:
        raise ValueError("concat_dipole_batches requires at least one batch.")
    if len(batches) == 1:
        return batches[0]

    def _merge_meta(key: str) -> list[Any]:
        merged: list[Any] = []
        for batch in batches:
            values = batch.metadata.get(key)
            if values is None:
                continue
            merged.extend(list(values))
        return merged

    buffer_sources: list[str] = []
    for batch in batches:
        sources = batch.metadata.get("buffer_sources")
        if sources is not None:
            buffer_sources.extend(list(sources))
        else:
            buffer_sources.extend(["unknown"] * batch.batch_size)

    return DipoleBatch(
        image_obs=torch.cat([b.image_obs for b in batches], dim=0),
        image_obs_raw=torch.cat([b.image_obs_raw for b in batches], dim=0),
        proprio=torch.cat([b.proprio for b in batches], dim=0),
        proprio_raw=torch.cat([b.proprio_raw for b in batches], dim=0),
        action_sequences=torch.cat([b.action_sequences for b in batches], dim=0),
        action_sequences_raw=torch.cat([b.action_sequences_raw for b in batches], dim=0),
        is_intervention=torch.cat([b.is_intervention for b in batches], dim=0),
        metadata={
            "start_indices": _merge_meta("start_indices"),
            "episode_ids": _merge_meta("episode_ids"),
            "episode_steps": _merge_meta("episode_steps"),
            "buffer_sources": buffer_sources,
        },
    )


__all__ = [
    "DipoleBatch",
    "select_dipole_batch",
    "concat_dipole_batches",
    "DipoleConfig",
    "EncoderConfig",
    "FlowAugmentationConfig",
    "LPBDetectorConfig",
    "ReplayBufferConfig",
    "TrainerConfig",
    "Transition",
]
