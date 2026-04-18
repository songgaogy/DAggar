"""Hydra helpers for the joint encoder + chunk DSM pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence

from hydra.utils import to_absolute_path

from robosuite.discriminator.dyn_bce.modules.flow_encoder import FlowMultitaskEncoder

from ..core.dataset import (
    CachedTrajectoryRef,
    DemoRef,
    LatentTransitionDataset,
    PreparedTrajectory,
    build_cached_trajectory_refs,
    estimate_trajectories_nbytes,
    filter_refs_by_data_types,
    prepare_trajectories,
    resolve_preprocessed_cache_root,
    resolve_window_size,
)
from ..core.dsm_discriminator import DSMDiscriminator


@dataclass(frozen=True)
class TrainingDatasets:
    train_dataset: LatentTransitionDataset
    val_dataset: LatentTransitionDataset | None
    train_refs: list[DemoRef]
    val_refs: list[DemoRef]
    train_cached_refs: list[CachedTrajectoryRef]
    val_cached_refs: list[CachedTrajectoryRef]


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_flow_encoder(cfg: Any) -> FlowMultitaskEncoder:
    lora_cfg = dict(getattr(cfg.policy, "lora", {}) or {})
    if bool(getattr(cfg.policy, "trainable_encoder", False)):
        lora_cfg["enabled"] = True
    return FlowMultitaskEncoder(
        checkpoint_path=to_absolute_path(str(cfg.policy.ckpt)),
        device=str(cfg.policy.device),
        image_size=int(cfg.data.image_size),
        batch_size=int(cfg.policy.encoder_batch_size),
        trainable=bool(getattr(cfg.policy, "trainable_encoder", False)),
        lora_cfg=lora_cfg,
    )


def _resolve_preprocessed_cache_settings(cfg: Any) -> tuple[str | None, bool, bool]:
    data_cfg = getattr(cfg, "data", None)
    cache_root = getattr(data_cfg, "preprocessed_cache_dir", None) if data_cfg is not None else None
    use_cache = bool(getattr(data_cfg, "use_preprocessed_cache", True)) if data_cfg is not None else True
    refresh_cache = bool(getattr(data_cfg, "refresh_preprocessed_cache", False)) if data_cfg is not None else False
    if cache_root is None or str(cache_root) == "":
        return None, False, refresh_cache
    return resolve_preprocessed_cache_root(data_cfg, str(cache_root)), use_cache, refresh_cache


def _resolve_window_size_override(cfg: Any) -> int:
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
    split_refs: dict[str, Sequence[DemoRef]],
    split_name: str,
    data_types: Sequence[str] | None,
) -> list[DemoRef]:
    return filter_refs_by_data_types(
        split_refs[str(split_name)],
        None if data_types is None else list(data_types),
    )


def load_split_trajectories(
    split_refs: dict[str, Sequence[DemoRef]],
    *,
    split_name: str,
    data_types: Sequence[str] | None,
    encoder: FlowMultitaskEncoder,
    task_to_index: dict[str, int],
    progress_label: str,
    cache_root: str | None = None,
    use_preprocessed_cache: bool = False,
    refresh_preprocessed_cache: bool = False,
) -> list[PreparedTrajectory]:
    refs = select_split_refs(
        split_refs=split_refs,
        split_name=str(split_name),
        data_types=data_types,
    )
    return prepare_trajectories(
        refs=refs,
        encoder=encoder,
        task_to_index=task_to_index,
        progress_label=progress_label,
        cache_root=cache_root,
        use_preprocessed_cache=use_preprocessed_cache,
        refresh_preprocessed_cache=refresh_preprocessed_cache,
    )


def _print_ram_summary(split_name: str, trajectories: Sequence[PreparedTrajectory]) -> None:
    num_bytes = estimate_trajectories_nbytes(trajectories)
    print(
        f"[lpb_score] preloaded split={split_name} "
        f"num_trajectories={len(trajectories)} ram_gb={num_bytes / (1024.0 ** 3):.3f}"
    )


def _print_cache_summary(split_name: str, cached_refs: Sequence[CachedTrajectoryRef]) -> None:
    total_steps = sum(int(ref.num_steps) for ref in cached_refs)
    print(
        f"[lpb_score] indexed_cached_split={split_name} "
        f"num_trajectories={len(cached_refs)} total_steps={total_steps}"
    )


def build_training_datasets(
    cfg: Any,
    *,
    split_refs: dict[str, Sequence[DemoRef]],
    encoder: FlowMultitaskEncoder,
    task_to_index: dict[str, int],
) -> TrainingDatasets:
    train_refs = list(split_refs["train"])
    val_refs = list(split_refs["val"])
    window_size = resolve_window_size(cfg)
    cache_root, use_preprocessed_cache, refresh_preprocessed_cache = _resolve_preprocessed_cache_settings(cfg)

    if use_preprocessed_cache:
        train_cached_refs = build_cached_trajectory_refs(
            refs=train_refs,
            encoder=encoder,
            task_to_index=task_to_index,
            progress_label="preload_train",
            cache_root=cache_root,
            use_preprocessed_cache=use_preprocessed_cache,
            refresh_preprocessed_cache=refresh_preprocessed_cache,
        )
        val_cached_refs = build_cached_trajectory_refs(
            refs=val_refs,
            encoder=encoder,
            task_to_index=task_to_index,
            progress_label="preload_val",
            cache_root=cache_root,
            use_preprocessed_cache=use_preprocessed_cache,
            refresh_preprocessed_cache=refresh_preprocessed_cache,
        )
        _print_cache_summary("train", train_cached_refs)
        _print_cache_summary("val", val_cached_refs)
        train_dataset = LatentTransitionDataset(
            cached_refs=train_cached_refs,
            window_size=window_size,
        )
        val_dataset = (
            LatentTransitionDataset(
                cached_refs=val_cached_refs,
                window_size=window_size,
            )
            if len(val_cached_refs) > 0
            else None
        )
    else:
        train_trajectories = prepare_trajectories(
            refs=train_refs,
            encoder=encoder,
            task_to_index=task_to_index,
            progress_label="preload_train",
            cache_root=cache_root,
            use_preprocessed_cache=False,
            refresh_preprocessed_cache=False,
        )
        val_trajectories = prepare_trajectories(
            refs=val_refs,
            encoder=encoder,
            task_to_index=task_to_index,
            progress_label="preload_val",
            cache_root=cache_root,
            use_preprocessed_cache=False,
            refresh_preprocessed_cache=False,
        )
        _print_ram_summary("train", train_trajectories)
        _print_ram_summary("val", val_trajectories)
        train_cached_refs = []
        val_cached_refs = []
        train_dataset = LatentTransitionDataset(
            trajectories=train_trajectories,
            window_size=window_size,
        )
        val_dataset = (
            LatentTransitionDataset(
                trajectories=val_trajectories,
                window_size=window_size,
            )
            if len(val_trajectories) > 0
            else None
        )
    return TrainingDatasets(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_refs=train_refs,
        val_refs=val_refs,
        train_cached_refs=train_cached_refs,
        val_cached_refs=val_cached_refs,
    )
