"""Hydra helpers: frozen flow encoder, chunk DSM wiring, and train/val datasets."""

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
    resolve_window_size,
)
from ..core.dsm_discriminator import DSMDiscriminator


@dataclass(frozen=True)
class TrainingDatasets:
    """Train/val chunk datasets plus the underlying encoded trajectory refs."""

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


def _resolve_window_size_override(cfg: Any) -> int:
    """Return a validation override only when the config deviates from defaults."""
    dataset_cfg = getattr(cfg, "dataset", None)
    data_cfg = getattr(cfg, "data", None)
    dataset_window = getattr(dataset_cfg, "window_size", None) if dataset_cfg is not None else None
    data_window = getattr(data_cfg, "window_size", None) if data_cfg is not None else None
    if dataset_window is None and data_window is None:
        return -1
    if dataset_window is not None and int(dataset_window) != 8:
        return int(dataset_window)
    if data_window is not None and int(data_window) != 8:
        return int(data_window)
    return -1


def build_dsm_discriminator(cfg: Any) -> DSMDiscriminator:
    """Load checkpoint into the chunk scorer and configure the detector."""
    return DSMDiscriminator(
        checkpoint_path=to_absolute_path(str(cfg.model.dsm_ckpt)),
        feature_device=str(cfg.feature.device),
        feature_batch_size=int(cfg.feature.batch_size),
        window_size=int(_resolve_window_size_override(cfg)),
        detector_device=str(cfg.detector.device),
        delta=float(cfg.detector.delta),
        delta_step=float(cfg.detector.delta_step),
        lambda_mode=str(cfg.detector.lambda_mode),
        lambda_window_size=int(cfg.detector.lambda_window_size),
        alpha=float(getattr(cfg.detector, "alpha", 1.0)),
        beta=float(getattr(cfg.detector, "beta", 1.0)),
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
    """Create train/val chunk datasets from cached latent trajectories."""
    train_refs = list(cached_splits["train"])
    val_refs = list(cached_splits["val"])
    window_size = resolve_window_size(cfg)

    train_dataset = LatentTransitionDataset(
        trajectory_refs=train_refs,
        window_size=window_size,
        preload_to_memory=bool(cfg.data.preload_train_to_memory),
    )
    val_dataset = (
        LatentTransitionDataset(
            trajectory_refs=val_refs,
            window_size=window_size,
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
