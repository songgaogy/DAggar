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
    """Grouped training inputs built from cached latent splits."""

    train_dataset: LatentTransitionDataset
    val_dataset: LatentTransitionDataset | None
    train_refs: list[EncodedTrajectoryRef]
    val_refs: list[EncodedTrajectoryRef]


@dataclass(frozen=True)
class EvalTrajectorySplits:
    """Grouped trajectory buckets used by offline evaluation."""

    bank: list[LatentTrajectory]
    calibration: list[LatentTrajectory]
    expert_eval: list[LatentTrajectory]
    success_eval: list[LatentTrajectory]
    fail_eval: list[LatentTrajectory]


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


def build_dsm_discriminator(cfg: Any) -> DSMDiscriminator:
    """Construct the DSM discriminator from Hydra config."""
    return DSMDiscriminator(
        checkpoint_path=to_absolute_path(str(cfg.model.dsm_ckpt)),
        feature_device=str(cfg.feature.device),
        feature_batch_size=int(cfg.feature.batch_size),
        action_horizon=int(cfg.feature.action_horizon),
        detector_device=str(cfg.detector.device),
        delta=float(cfg.detector.delta),
        delta_step=float(cfg.detector.delta_step),
        lambda_mode=str(cfg.detector.lambda_mode),
        lambda_window_size=int(cfg.detector.lambda_window_size),
        policy_weight=float(getattr(cfg.detector, "policy_weight", 0.2)),
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
    train_refs = select_split_refs(
        cached_splits=cached_splits,
        split_name="train",
        data_types=list(getattr(cfg.data, "train_data_types", ["expert", "success_rollout"])),
    )
    val_refs = select_split_refs(
        cached_splits=cached_splits,
        split_name="val",
        data_types=list(getattr(cfg.data, "val_data_types", ["expert", "success_rollout"])),
    )

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


def load_eval_trajectory_splits(
    cfg: Any,
    cached_splits: dict[str, Sequence[EncodedTrajectoryRef]],
) -> EvalTrajectorySplits:
    """Load the standard bank/calibration/eval buckets used by offline evaluation."""
    return EvalTrajectorySplits(
        bank=load_split_trajectories(
            cached_splits,
            split_name=str(cfg.eval.bank_split),
            data_types=list(cfg.eval.bank_data_types),
        ),
        calibration=load_split_trajectories(
            cached_splits,
            split_name=str(cfg.eval.calibration_split),
            data_types=list(cfg.eval.calibration_data_types),
        ),
        expert_eval=load_split_trajectories(
            cached_splits,
            split_name=str(cfg.eval.expert_eval_split),
            data_types=list(cfg.eval.expert_eval_data_types),
        ),
        success_eval=load_split_trajectories(
            cached_splits,
            split_name=str(cfg.eval.success_eval_split),
            data_types=list(cfg.eval.success_eval_data_types),
        ),
        fail_eval=load_split_trajectories(
            cached_splits,
            split_name=str(cfg.eval.fail_eval_split),
            data_types=list(cfg.eval.fail_eval_data_types),
        ),
    )
