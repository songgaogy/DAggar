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
    filter_refs_by_roles,
    load_latent_trajectories,
)


@dataclass(frozen=True)
class BackboneTrainingDatasets:
    train_dataset: LatentTransitionDataset
    eval_dataset: LatentTransitionDataset | None
    train_refs: list[EncodedTrajectoryRef]
    eval_refs: list[EncodedTrajectoryRef]


@dataclass(frozen=True)
class PUTrajectorySplits:
    train: list[LatentTrajectory]
    eval: list[LatentTrajectory]
    calibration_positive: list[LatentTrajectory]
    calibration_background: list[LatentTrajectory]


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_flow_encoder(cfg: Any) -> FrozenFlowMultitaskEncoder:
    return FrozenFlowMultitaskEncoder(
        checkpoint_path=to_absolute_path(str(cfg.policy.ckpt)),
        device=str(cfg.policy.device),
        image_size=int(cfg.data.image_size),
        batch_size=int(cfg.policy.encoder_batch_size),
    )


def select_split_refs(
    cached_splits: dict[str, Sequence[EncodedTrajectoryRef]],
    split_name: str,
    data_types: Sequence[str] | None = None,
    roles: Sequence[str] | None = None,
) -> list[EncodedTrajectoryRef]:
    refs = list(cached_splits[str(split_name)])
    refs = filter_refs_by_data_types(refs, None if data_types is None else list(data_types))
    refs = filter_refs_by_roles(refs, None if roles is None else list(roles))
    return refs


def load_split_trajectories(
    cached_splits: dict[str, Sequence[EncodedTrajectoryRef]],
    split_name: str,
    data_types: Sequence[str] | None = None,
    roles: Sequence[str] | None = None,
) -> list[LatentTrajectory]:
    refs = select_split_refs(
        cached_splits=cached_splits,
        split_name=split_name,
        data_types=data_types,
        roles=roles,
    )
    return load_latent_trajectories(refs)


def build_backbone_training_datasets(
    cfg: Any,
    cached_splits: dict[str, Sequence[EncodedTrajectoryRef]],
) -> BackboneTrainingDatasets:
    train_refs = select_split_refs(
        cached_splits=cached_splits,
        split_name="train",
        data_types=list(getattr(cfg.data, "train_data_types", ["expert", "success_rollout"])),
    )
    eval_refs = select_split_refs(
        cached_splits=cached_splits,
        split_name="eval",
        data_types=list(getattr(cfg.data, "eval_data_types", ["expert", "success_rollout"])),
    )
    train_dataset = LatentTransitionDataset(
        trajectory_refs=train_refs,
        horizon=int(cfg.data.transition_horizon),
        preload_to_memory=bool(cfg.data.preload_train_to_memory),
    )
    eval_dataset = (
        LatentTransitionDataset(
            trajectory_refs=eval_refs,
            horizon=int(cfg.data.transition_horizon),
            preload_to_memory=bool(cfg.data.preload_eval_to_memory),
        )
        if eval_refs
        else None
    )
    return BackboneTrainingDatasets(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        train_refs=train_refs,
        eval_refs=eval_refs,
    )


def build_pu_trajectory_splits(
    cfg: Any,
    cached_splits: dict[str, Sequence[EncodedTrajectoryRef]],
) -> PUTrajectorySplits:
    train = load_split_trajectories(
        cached_splits=cached_splits,
        split_name="train",
        data_types=list(getattr(cfg.data, "pu_train_data_types", ["expert", "success_rollout", "fail_rollout"])),
    )
    eval_trajectories = load_split_trajectories(
        cached_splits=cached_splits,
        split_name="eval",
        data_types=list(getattr(cfg.data, "pu_eval_data_types", ["expert", "success_rollout", "fail_rollout"])),
    )
    calibration_positive = load_split_trajectories(
        cached_splits=cached_splits,
        split_name=str(cfg.eval.calibration_split),
        data_types=list(cfg.eval.calibration_positive_data_types),
    )
    calibration_background = load_split_trajectories(
        cached_splits=cached_splits,
        split_name=str(cfg.eval.calibration_split),
        data_types=list(cfg.eval.calibration_background_data_types),
    )
    return PUTrajectorySplits(
        train=train,
        eval=eval_trajectories,
        calibration_positive=calibration_positive,
        calibration_background=calibration_background,
    )
