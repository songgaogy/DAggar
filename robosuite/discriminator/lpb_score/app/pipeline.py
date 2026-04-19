"""Hydra helpers: frozen flow encoder, DSM discriminator wiring, and train/val transition datasets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence

from hydra.utils import to_absolute_path

from robosuite.discriminator.dyn_bce.modules.flow_encoder import FrozenFlowMultitaskEncoder

from ..core.dataset import (
    EncodedTrajectoryRef,
    LatentTrajectory,
    LatentTransitionDataset,
    filter_refs_by_data_types,
    load_latent_trajectories,
)
from ..core.dsm_discriminator import DSMDiscriminator


@dataclass(frozen=True)
class TrainingDatasets:
    """Train/val ``LatentTransitionDataset`` instances plus the underlying encoded trajectory refs."""

    train_dataset: LatentTransitionDataset
    val_dataset: LatentTransitionDataset | None
    train_refs: list[EncodedTrajectoryRef]
    val_refs: list[EncodedTrajectoryRef]


def now_tag() -> str:
    """Return a compact timestamp for output folders and artifacts."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_flow_encoder(cfg: Any) -> FrozenFlowMultitaskEncoder:
    """Construct the frozen policy encoder used across train/eval/visualize."""
    return FrozenFlowMultitaskEncoder(
        checkpoint_path=to_absolute_path(str(cfg.policy.ckpt)),
        device=str(cfg.policy.device),
        image_size=int(cfg.data.image_size),
        batch_size=int(cfg.policy.encoder_batch_size),
    )


def _resolve_detector_branch_weight(
    detector_cfg: Any,
    *,
    explicit_key: str,
    scalar_key: str,
    default: float,
) -> float:
    explicit_value = getattr(detector_cfg, explicit_key, None)
    if explicit_value is not None:
        return float(explicit_value)
    scalar_value = getattr(detector_cfg, scalar_key, None)
    if scalar_value is None:
        return float(default)
    if explicit_key.endswith("_action"):
        return 0.0
    return float(scalar_value)


def build_dsm_discriminator(cfg: Any) -> DSMDiscriminator:
    """Load checkpoint into ``DSMTransitionScorer`` and configure detector thresholds and T3 weights."""
    detector_cfg = cfg.detector
    return DSMDiscriminator(
        checkpoint_path=to_absolute_path(str(cfg.model.dsm_ckpt)),
        feature_device=str(cfg.feature.device),
        feature_batch_size=int(cfg.feature.batch_size),
        action_horizon=int(getattr(cfg.feature, "action_horizon", -1)),
        detector_device=str(detector_cfg.device),
        delta=float(detector_cfg.delta),
        delta_step=float(detector_cfg.delta_step),
        lambda_mode=str(detector_cfg.lambda_mode),
        lambda_window_size=int(detector_cfg.lambda_window_size),
        score_mode=str(getattr(detector_cfg, "score_mode", "t1_positive_energy")),
        alpha_state=_resolve_detector_branch_weight(
            detector_cfg,
            explicit_key="alpha_state",
            scalar_key="alpha",
            default=1.0,
        ),
        alpha_action=_resolve_detector_branch_weight(
            detector_cfg,
            explicit_key="alpha_action",
            scalar_key="alpha",
            default=1.0,
        ),
        alpha_dynamics=_resolve_detector_branch_weight(
            detector_cfg,
            explicit_key="alpha_dynamics",
            scalar_key="alpha",
            default=1.0,
        ),
        beta_state=_resolve_detector_branch_weight(
            detector_cfg,
            explicit_key="beta_state",
            scalar_key="beta",
            default=1.0,
        ),
        beta_action=_resolve_detector_branch_weight(
            detector_cfg,
            explicit_key="beta_action",
            scalar_key="beta",
            default=1.0,
        ),
        beta_dynamics=_resolve_detector_branch_weight(
            detector_cfg,
            explicit_key="beta_dynamics",
            scalar_key="beta",
            default=1.0,
        ),
    )


def select_split_refs(
    cached_splits: dict[str, Sequence[EncodedTrajectoryRef]],
    split_name: str,
    data_types: Sequence[str] | None,
) -> list[EncodedTrajectoryRef]:
    """Select a subset of cached refs for a split and data-type bucket."""
    return filter_refs_by_data_types(
        cached_splits[str(split_name)],
        None if data_types is None else list(data_types),
    )


def load_split_trajectories(
    cached_splits: dict[str, Sequence[EncodedTrajectoryRef]],
    split_name: str,
    data_types: Sequence[str] | None,
) -> list[LatentTrajectory]:
    """Load latent trajectories for one split/data-type selection."""
    return load_latent_trajectories(
        select_split_refs(
            cached_splits=cached_splits,
            split_name=str(split_name),
            data_types=data_types,
        )
    )


def build_training_datasets(
    cfg: Any,
    cached_splits: dict[str, Sequence[EncodedTrajectoryRef]],
) -> TrainingDatasets:
    """Create train/val transition datasets from cached latent trajectories."""
    train_refs = list(cached_splits["train"])
    val_refs = list(cached_splits["val"])

    train_dataset = LatentTransitionDataset(
        trajectory_refs=train_refs,
        horizon=int(cfg.data.transition_horizon),
        preload_to_memory=bool(cfg.data.preload_train_to_memory),
    )
    val_dataset = (
        LatentTransitionDataset(
            trajectory_refs=val_refs,
            horizon=int(cfg.data.transition_horizon),
            preload_to_memory=bool(cfg.data.preload_eval_to_memory),
        )
        if len(val_refs) > 0
        else None
    )
    return TrainingDatasets(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_refs=train_refs,
        val_refs=val_refs,
    )
