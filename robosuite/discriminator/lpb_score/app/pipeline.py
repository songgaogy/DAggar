"""Helpers for assembling the DSM training and visualization pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence

import numpy as np

from hydra.utils import to_absolute_path

from robosuite.discriminator.dyn_bce.modules.flow_encoder import FlowMultitaskEncoder

from ..core.dataset import (
    CachedTrajectoryRef,
    DemoRef,
    LatentTransitionDataset,
    PreparedTrajectory,
    build_cached_trajectory_refs,
    filter_refs_by_data_types,
    prepare_trajectories,
    resolve_preprocessed_cache_root,
    resolve_window_size,
)
from ..core.dsm_discriminator import DSMDiscriminator


@dataclass(frozen=True)
class TrainingDatasets:
    """Bundle containing the datasets and reference lists for a training run."""

    train_dataset: LatentTransitionDataset
    val_dataset: LatentTransitionDataset | None
    train_refs: list[DemoRef]
    val_refs: list[DemoRef]
    train_cached_refs: list[CachedTrajectoryRef]
    val_cached_refs: list[CachedTrajectoryRef]


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_flow_encoder(cfg: Any, *, device_override: str | None = None) -> FlowMultitaskEncoder:
    lora_cfg = dict(getattr(cfg.policy, "lora", {}) or {})
    if bool(getattr(cfg.policy, "trainable_encoder", False)):
        lora_cfg["enabled"] = True
    data_cfg = getattr(cfg, "data", None)
    return FlowMultitaskEncoder(
        checkpoint_path=to_absolute_path(str(cfg.policy.ckpt)),
        device=str(device_override or cfg.policy.device),
        image_size=int(cfg.data.image_size),
        batch_size=int(cfg.policy.encoder_batch_size),
        trainable=bool(getattr(cfg.policy, "trainable_encoder", False)),
        lora_cfg=lora_cfg,
        preprocessed_cache_layout=str(getattr(data_cfg, "preprocessed_cache_layout", "bundle")),
        upgrade_legacy_preprocessed_cache=bool(
            getattr(data_cfg, "upgrade_legacy_preprocessed_cache", True)
        ),
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
    cache_root: str | None = None,
    use_preprocessed_cache: bool = False,
    refresh_preprocessed_cache: bool = False,
) -> list[PreparedTrajectory]:
    """Materialize one split into resident trajectory tensors."""
    refs = select_split_refs(
        split_refs=split_refs,
        split_name=str(split_name),
        data_types=data_types,
    )
    return prepare_trajectories(
        refs=refs,
        encoder=encoder,
        task_to_index=task_to_index,
        cache_root=cache_root,
        use_preprocessed_cache=use_preprocessed_cache,
        refresh_preprocessed_cache=refresh_preprocessed_cache,
    )


def _estimate_cached_refs_nbytes(
    cached_refs: Sequence[CachedTrajectoryRef],
    *,
    image_size: int,
    action_dim: int,
) -> int:
    total = 0
    for ref in cached_refs:
        total += _estimate_cached_ref_nbytes(
            ref,
            image_size=image_size,
            action_dim=action_dim,
        )
    return int(total)


def _resolve_preload_budget_gb(cfg: Any, key: str, default: float = 0.0) -> float:
    training_cfg = getattr(cfg, "training", None)
    if training_cfg is None:
        return float(default)
    return float(getattr(training_cfg, key, default))


def _resolve_data_in_ram(cfg: Any, default: int = 0) -> int:
    training_cfg = getattr(cfg, "training", None)
    if training_cfg is None:
        return int(default)
    return max(0, int(getattr(training_cfg, "data_in_ram", default)))


def _estimate_cached_ref_nbytes(
    ref: CachedTrajectoryRef,
    *,
    image_size: int,
    action_dim: int,
) -> int:
    total = 0
    total += int(ref.num_steps) * int(ref.num_cameras) * 3 * int(image_size) * int(image_size)
    total += int(ref.num_steps) * int(ref.proprio_dim) * 4
    total += int(ref.num_steps) * int(action_dim) * 4
    return int(total)


def _select_resident_indices(
    cached_refs: Sequence[CachedTrajectoryRef],
    *,
    image_size: int,
    action_dim: int,
    minimum_count: int,
    budget_bytes: int,
    seed: int,
) -> list[int]:
    """Choose a RAM-resident subset subject to a hard memory budget.

    `minimum_count` guarantees at least that many resident trajectories when
    available. Remaining candidates are added in a seeded random order until the
    estimated byte budget is exhausted.
    """
    total = int(len(cached_refs))
    minimum_count = min(max(0, int(minimum_count)), total)
    budget_bytes = max(0, int(budget_bytes))
    if total <= 0:
        return []
    if minimum_count >= total:
        return list(range(total))

    rng = np.random.default_rng(int(seed))
    order = np.arange(total, dtype=np.int64)
    rng.shuffle(order)

    selected: list[int] = []
    used_bytes = 0
    for raw_idx in order.tolist():
        idx = int(raw_idx)
        ref_nbytes = _estimate_cached_ref_nbytes(
            cached_refs[idx],
            image_size=image_size,
            action_dim=action_dim,
        )
        if len(selected) < minimum_count:
            selected.append(idx)
            used_bytes += ref_nbytes
            continue
        if budget_bytes <= 0:
            break
        if used_bytes + ref_nbytes <= budget_bytes:
            selected.append(idx)
            used_bytes += ref_nbytes
    return sorted(selected)


def build_training_datasets(
    cfg: Any,
    *,
    split_refs: dict[str, Sequence[DemoRef]],
    encoder: FlowMultitaskEncoder,
    task_to_index: dict[str, int],
    build_val_dataset: bool = True,
    replica_count: int = 1,
) -> TrainingDatasets:
    """Construct train/validation datasets with cache-aware residency decisions."""
    train_refs = list(split_refs["train"])
    val_refs = list(split_refs["val"]) if bool(build_val_dataset) else []
    window_size = resolve_window_size(cfg)
    cache_root, use_preprocessed_cache, refresh_preprocessed_cache = _resolve_preprocessed_cache_settings(cfg)
    replica_count = max(1, int(replica_count))
    # Split RAM budgets across replicas to keep total node memory bounded.
    train_preload_budget_gb = _resolve_preload_budget_gb(cfg, "train_preload_ram_gb", 0.0) / float(replica_count)
    val_preload_budget_gb = _resolve_preload_budget_gb(cfg, "val_preload_ram_gb", 0.0) / float(replica_count)
    data_in_ram = int(np.ceil(float(_resolve_data_in_ram(cfg, 0)) / float(replica_count)))

    if use_preprocessed_cache:
        train_cached_refs = build_cached_trajectory_refs(
            refs=train_refs,
            encoder=encoder,
            task_to_index=task_to_index,
            cache_root=cache_root,
            use_preprocessed_cache=use_preprocessed_cache,
            refresh_preprocessed_cache=refresh_preprocessed_cache,
        )
        val_cached_refs = (
            build_cached_trajectory_refs(
                refs=val_refs,
                encoder=encoder,
                task_to_index=task_to_index,
                cache_root=cache_root,
                use_preprocessed_cache=use_preprocessed_cache,
                refresh_preprocessed_cache=refresh_preprocessed_cache,
            )
            if bool(build_val_dataset)
            else []
        )
        # Keep a deterministic resident subset, lazily reading the rest from cache.
        resident_indices = _select_resident_indices(
            cached_refs=train_cached_refs,
            image_size=int(cfg.data.image_size),
            action_dim=int(encoder.action_dim),
            minimum_count=int(data_in_ram),
            budget_bytes=int(max(0.0, float(train_preload_budget_gb)) * (1024.0 ** 3)),
            seed=int(getattr(cfg, "seed", 0)) + 1337,
        )
        if resident_indices:
            # Keep only a budgeted subset resident; the rest stays indexed on disk.
            resident_cached_refs = [train_cached_refs[idx] for idx in resident_indices]
            resident_refs = [
                DemoRef(
                    task_name=ref.task_name,
                    data_type=ref.data_type,
                    split=ref.split,
                    file_path=ref.file_path,
                    demo_key=ref.demo_key,
                )
                for ref in resident_cached_refs
            ]
            resident_index_by_key = {
                (ref.task_name, ref.file_path, ref.demo_key): int(dataset_index)
                for dataset_index, ref in zip(resident_indices, resident_cached_refs)
            }
            resident_trajectories = prepare_trajectories(
                refs=resident_refs,
                encoder=encoder,
                task_to_index=task_to_index,
                cache_root=cache_root,
                use_preprocessed_cache=use_preprocessed_cache,
                refresh_preprocessed_cache=refresh_preprocessed_cache,
            )
            resident_pairs = sorted(
                (
                    int(resident_index_by_key[(traj.task_name, traj.file_path, traj.demo_key)]),
                    traj,
                )
                for traj in resident_trajectories
            )
        else:
            resident_pairs: list[tuple[int, PreparedTrajectory]] = []

        if len(resident_pairs) >= len(train_cached_refs) and len(train_cached_refs) > 0:
            train_trajectories = [traj for _, traj in resident_pairs]
            train_dataset = LatentTransitionDataset(
                trajectories=train_trajectories,
                window_size=window_size,
            )
        elif resident_pairs:
            train_dataset = LatentTransitionDataset(
                cached_refs=train_cached_refs,
                resident_trajectories=resident_pairs,
                window_size=window_size,
            )
        else:
            train_dataset = LatentTransitionDataset(
                cached_refs=train_cached_refs,
                window_size=window_size,
            )

        val_dataset = None
        if len(val_cached_refs) > 0:
            val_ram_bytes = _estimate_cached_refs_nbytes(
                val_cached_refs,
                image_size=int(cfg.data.image_size),
                action_dim=int(encoder.action_dim),
            )
            val_budget_bytes = int(max(0.0, float(val_preload_budget_gb)) * (1024.0 ** 3))
            # Validation is fully materialized only when the configured budget can hold it.
            if val_budget_bytes > 0 and val_ram_bytes <= val_budget_bytes:
                val_trajectories = prepare_trajectories(
                    refs=val_refs,
                    encoder=encoder,
                    task_to_index=task_to_index,
                    cache_root=cache_root,
                    use_preprocessed_cache=use_preprocessed_cache,
                    refresh_preprocessed_cache=refresh_preprocessed_cache,
                )
                val_dataset = LatentTransitionDataset(
                    trajectories=val_trajectories,
                    window_size=window_size,
                )
            else:
                val_dataset = LatentTransitionDataset(
                    cached_refs=val_cached_refs,
                    window_size=window_size,
                )
    else:
        train_trajectories = prepare_trajectories(
            refs=train_refs,
            encoder=encoder,
            task_to_index=task_to_index,
            cache_root=cache_root,
            use_preprocessed_cache=use_preprocessed_cache,
            refresh_preprocessed_cache=refresh_preprocessed_cache,
        )
        train_dataset = LatentTransitionDataset(
            trajectories=train_trajectories,
            window_size=window_size,
        )
        train_cached_refs = []
        if bool(build_val_dataset) and use_preprocessed_cache:
            val_cached_refs = build_cached_trajectory_refs(
                refs=val_refs,
                encoder=encoder,
                task_to_index=task_to_index,
                cache_root=cache_root,
                use_preprocessed_cache=use_preprocessed_cache,
                refresh_preprocessed_cache=refresh_preprocessed_cache,
            )
            val_dataset = (
                LatentTransitionDataset(
                    cached_refs=val_cached_refs,
                    window_size=window_size,
                )
                if len(val_cached_refs) > 0
                else None
            )
        elif bool(build_val_dataset):
            val_trajectories = prepare_trajectories(
                refs=val_refs,
                encoder=encoder,
                task_to_index=task_to_index,
                cache_root=cache_root,
                use_preprocessed_cache=False,
                refresh_preprocessed_cache=False,
            )
            val_cached_refs = []
            val_dataset = (
                LatentTransitionDataset(
                    trajectories=val_trajectories,
                    window_size=window_size,
                )
                if len(val_trajectories) > 0
                else None
            )
        else:
            val_cached_refs = []
            val_dataset = None
    return TrainingDatasets(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_refs=train_refs,
        val_refs=val_refs,
        train_cached_refs=train_cached_refs,
        val_cached_refs=val_cached_refs,
    )
