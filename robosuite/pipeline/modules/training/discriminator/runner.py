"""Standalone CUDA-only nnPU-replay plus supervised-GT finetuning."""

from __future__ import annotations

import datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from robosuite.pipeline.modules.training.discriminator import (
    BatchOnlineCUDAPoolSampler,
    FinetuneDynamicsEncoder,
    build_finetuned_checkpoint_payload,
    build_gt_negative_windows,
    encode_gt_negative_windows,
    encode_policy_segments,
    feature_tensors,
    finetune_warmstart_detector,
    load_offline_episodes,
    load_pretrain_pools,
    load_round_feature_cache,
    load_warmstart_detector,
    optional_file,
    required_path,
    require_cuda_device,
    resolve_parent_nnpu_semantics,
    resolve_round_mixture_plan,
    resolved_config_dict,
    save_finetuned_checkpoint,
    save_round_feature_cache,
    sha256_file,
    split_policy_segments,
    validate_finetune_contract,
)
from robosuite.pipeline.modules.training.discriminator.finetune_setup import (
    active_training_pool_names,
    configured_loss_terms,
    mapping_config,
    reserved_unlabeled_provenance,
    resolved_sampler_config,
    trajectory_stats,
)
from robosuite.pipeline.utils import (
    maybe_build_metric_logger,
    maybe_log,
    write_resolved_config,
    write_run_info,
)


FINETUNE_METHOD = "nnpu_replay_positive_safety_margin_gt_negative"
SEPARATE_GT_FINETUNE_METHOD = "nnpu_replay_separate_gt_risks"
LEGACY_FINETUNE_METHOD = "nnpu_replay_gt_bce"


def _feature_cache_key(
    *,
    episodes_path: Path,
    encoder: FinetuneDynamicsEncoder,
    camera_names: list[str],
    camera_to_view: dict[str, str],
    action_horizon: int,
    image_height: int,
    image_width: int,
    gt_config: dict[str, Any],
) -> str:
    contract = {
        "episodes_sha256": sha256_file(episodes_path),
        "encoder_sha256": sha256_file(encoder.encoder_checkpoint),
        "normalizer_sha256": sha256_file(encoder.normalizer_checkpoint),
        "camera_names": camera_names,
        "camera_to_view": camera_to_view,
        "feature_source": str(encoder.feature_source),
        "transformer_layer": int(encoder.transformer_layer),
        "proprio_indices": encoder.proprio_indices,
        "use_chunk": bool(encoder.use_chunk),
        "frameskip": int(encoder.inner_encoder.frameskip),
        "action_horizon": int(action_horizon),
        "image_contract": {
            "dtype": "uint8",
            "layout": "time_height_width_channels",
            "height": int(image_height),
            "width": int(image_width),
        },
        "gt_negative": gt_config,
    }
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _apply_effective_round_weights(
    objective_config: dict[str, Any],
    *,
    plans: dict[str, Any],
) -> dict[str, Any]:
    resolved = {
        key: (dict(value) if isinstance(value, dict) else value)
        for key, value in objective_config.items()
    }
    resolved["terms"] = {
        name: dict(term) for name, term in dict(objective_config["terms"]).items()
    }
    configured_terms = configured_loss_terms(objective_config)
    for term in configured_terms:
        matching = [
            plans[pool_name]
            for pool_name in term.pool_batch_sizes
            if pool_name in plans
        ]
        if not matching:
            continue
        if len(matching) != 1:
            raise ValueError(
                f"Loss term {term.name!r} cannot mix multiple recursive pools."
            )
        resolved["terms"][term.name]["weight"] = float(
            matching[0].effective_loss_weight
        )
    return resolved


def _cuda_feature_tensors(
    trajectories: list[Any], *, device: torch.device
) -> list[torch.Tensor]:
    result: list[torch.Tensor] = []
    for trajectory in trajectories:
        tensor = trajectory.features
        if tensor.device.type == "cpu" and not tensor.is_pinned():
            tensor = tensor.pin_memory()
        result.append(
            tensor.to(device=device, dtype=torch.float32, non_blocking=True)
        )
    return result


def run_discriminator_finetune(cfg: DictConfig) -> None:
    """Encode named pools, warm-start the nnPU head, and recalibrate it."""
    task_name = str(cfg.env.environment)
    disc_cfg = cfg.algorithm.discriminator
    finetune_cfg = cfg.offline.discriminator_finetune
    device = require_cuda_device(str(disc_cfg.learner_device))
    torch.cuda.set_device(0 if device.index is None else device.index)
    torch.cuda.manual_seed_all(int(cfg.seed))
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    parent_checkpoint = optional_file(
        disc_cfg.checkpoint,
        name="algorithm.discriminator.checkpoint",
    )
    if parent_checkpoint is None:
        parent_checkpoint = optional_file(
            finetune_cfg.parent_checkpoint,
            name="offline.discriminator_finetune.parent_checkpoint",
        )
    if parent_checkpoint is None:
        raise ValueError(
            "A parent discriminator checkpoint is required via "
            "algorithm.discriminator.checkpoint or "
            "offline.discriminator_finetune.parent_checkpoint; Step 3 supports "
            "warm-start finetuning only."
        )
    encoder_checkpoint = optional_file(
        disc_cfg.encoder_ckpt,
        name="algorithm.discriminator.encoder_ckpt",
    )
    configured_episode_paths = OmegaConf.select(
        cfg,
        "offline.discriminator_finetune.episodes_paths",
        default=None,
    )
    raw_episode_paths = (
        list(configured_episode_paths)
        if configured_episode_paths
        else [finetune_cfg.episodes_path]
    )
    episodes_paths = [
        required_path(
            value,
            name=f"offline.discriminator_finetune.episodes_paths[{index}]",
        )
        for index, value in enumerate(raw_episode_paths)
    ]
    round_index = int(
        OmegaConf.select(
            cfg, "offline.discriminator_finetune.round_index", default=0
        )
    )
    action_horizon = int(
        OmegaConf.select(
            cfg,
            "offline.discriminator_finetune.action_horizon",
            default=8,
        )
    )
    if round_index < 0 or len(episodes_paths) != round_index + 1:
        raise ValueError(
            "episodes_paths must contain exactly rounds [0, current_round], got "
            f"current_round={round_index}, paths={len(episodes_paths)}."
        )
    episodes_path = episodes_paths[-1]
    pretrain_path = required_path(
        finetune_cfg.pretrain_dir,
        name="offline.discriminator_finetune.pretrain_dir",
    )

    epochs = int(finetune_cfg.epochs)
    scheduler_horizon_epochs = int(finetune_cfg.scheduler_horizon_epochs)
    encode_batch_size = int(finetune_cfg.encode_batch_size)
    log_interval = int(finetune_cfg.log_interval)
    if min(epochs, encode_batch_size, log_interval) <= 0:
        raise ValueError("epochs, encode_batch_size, and log_interval must be positive.")
    if scheduler_horizon_epochs < epochs:
        raise ValueError(
            "scheduler_horizon_epochs must be >= epochs, got "
            f"{scheduler_horizon_epochs} < {epochs}."
        )
    if float(finetune_cfg.lr) < 0.0 or float(finetune_cfg.weight_decay) < 0.0:
        raise ValueError("Learning rate and weight decay must be non-negative.")

    objective_config = mapping_config(
        finetune_cfg.objective,
        name="offline.discriminator_finetune.objective",
    )
    loss_terms = configured_loss_terms(objective_config)
    active_term_types = {term.type for term in loss_terms if term.active}
    if "positive_safety_margin" in active_term_types:
        finetune_method = FINETUNE_METHOD
    elif active_term_types & {"positive_logistic", "negative_logistic"}:
        finetune_method = SEPARATE_GT_FINETUNE_METHOD
    else:
        finetune_method = LEGACY_FINETUNE_METHOD
    active_training_pools = active_training_pool_names(loss_terms)
    gt_config = mapping_config(
        finetune_cfg.gt_negative,
        name="offline.discriminator_finetune.gt_negative",
    )
    required_gt_semantics = {
        "action_source": "policy_action",
        "post_boundary": "intervention_block",
        "chunk_boundary": "continuous_window",
    }
    for key, expected in required_gt_semantics.items():
        if str(gt_config.get(key)) != expected:
            raise ValueError(
                f"gt_negative.{key} must be {expected!r}, got {gt_config.get(key)!r}."
            )
    pre_intervention_chunks = int(gt_config.get("pre_intervention_chunks", -1))
    post_intervention_chunks = int(gt_config.get("post_intervention_chunks", -1))
    if pre_intervention_chunks < 0 or post_intervention_chunks < 0:
        raise ValueError(
            "gt_negative pre_intervention_chunks and post_intervention_chunks "
            "must be non-negative."
        )
    pre_end_chunks = int(gt_config.get("pre_end_chunk", 0))
    if pre_end_chunks < 0:
        raise ValueError("gt_negative pre_end_chunk must be non-negative.")

    pretrain_pools, pretrain_manifest = load_pretrain_pools(pretrain_path)
    offline_payloads = [load_offline_episodes(path) for path in episodes_paths]
    offline_payload = offline_payloads[-1]
    detector, parent_payload = load_warmstart_detector(
        parent_checkpoint,
        device=device,
        expected_task=task_name,
    )
    safety_term = next(
        (
            term
            for term in loss_terms
            if term.active and term.type == "positive_safety_margin"
        ),
        None,
    )
    positive_safety_boundary: float | None = None
    resolved_positive_safety_margin: dict[str, Any] | None = None
    if safety_term is not None:
        parent_failure_threshold = float(detector.thresholds[task_name])
        if not math.isfinite(parent_failure_threshold):
            raise ValueError(
                f"Parent threshold for {task_name!r} must be finite, "
                f"got {parent_failure_threshold!r}."
            )
        positive_safety_boundary = -parent_failure_threshold
        assert safety_term.margin_delta is not None
        assert safety_term.safety_margin_weight is not None
        assert safety_term.temperature is not None
        resolved_positive_safety_margin = {
            "task": task_name,
            "boundary_source": safety_term.boundary_source,
            "parent_failure_threshold": parent_failure_threshold,
            "m_k": positive_safety_boundary,
            "margin_delta": float(safety_term.margin_delta),
            "target_logit": (
                positive_safety_boundary + float(safety_term.margin_delta)
            ),
            "safety_margin_weight": float(safety_term.safety_margin_weight),
            "temperature": float(safety_term.temperature),
        }

    manifest_feature = dict(pretrain_manifest.get("feature_contract", {}))
    manifest_camera_map = {
        str(key): str(value)
        for key, value in dict(manifest_feature.get("camera_to_view", {})).items()
    }
    configured_camera_map = {
        str(key): str(value)
        for key, value in dict(disc_cfg.camera_to_view or {}).items()
    }
    if configured_camera_map and configured_camera_map != manifest_camera_map:
        raise ValueError(
            "Configured camera_to_view differs from the extracted pretrain contract: "
            f"config={configured_camera_map}, manifest={manifest_camera_map}."
        )
    manifest_proprio = manifest_feature.get("proprio_indices")
    effective_proprio = (
        None
        if manifest_proprio is None
        else [int(value) for value in manifest_proprio]
    )
    with torch.device(device):
        encoder = FinetuneDynamicsEncoder(
            nnpu_ckpt_path=parent_checkpoint,
            encoder_ckpt=encoder_checkpoint,
            device=device,
            camera_to_view=(configured_camera_map or manifest_camera_map),
            proprio_indices=effective_proprio,
        )
    for payload in offline_payloads:
        validate_finetune_contract(
            task_name=task_name,
            parent_payload=parent_payload,
            pretrain_manifest=pretrain_manifest,
            offline_payload=payload,
            encoder=encoder,
        )

    gt_negative_active = any(
        term.active and term.type in {"supervised_bce", "negative_logistic"}
        for term in loss_terms
    )
    feature_cache_value = OmegaConf.select(
        cfg,
        "offline.discriminator_finetune.feature_cache_dir",
        default=None,
    )
    feature_cache_dir = (
        None
        if feature_cache_value is None
        or str(feature_cache_value).strip().lower() in {"", "none", "null"}
        else Path(to_absolute_path(str(feature_cache_value))).resolve()
    )
    round_positive: dict[int, list[Any]] = {}
    round_gt_negative: dict[int, list[Any]] = {}
    round_records: list[dict[str, Any]] = []
    all_segments: list[Any] = []
    all_reserved_shards: list[dict[str, Any]] = []
    segment_stats = {
        "segments": 0,
        "positive_segments": 0,
        "unlabeled_segments": 0,
        "excluded_human_frames": 0,
    }
    gt_window_stats: dict[str, Any] = {}
    for source_round, (source_path, payload) in enumerate(
        zip(episodes_paths, offline_payloads, strict=True)
    ):
        segments, current_segment_stats = split_policy_segments(payload)
        gt_windows, current_gt_stats = build_gt_negative_windows(
            payload,
            pre_intervention_chunks=pre_intervention_chunks,
            post_intervention_chunks=post_intervention_chunks,
            frameskip=int(encoder.inner_encoder.frameskip),
            pre_end_chunks=pre_end_chunks,
        )
        cache_key = _feature_cache_key(
            episodes_path=source_path,
            encoder=encoder,
            camera_names=list(payload["camera_names"]),
            camera_to_view=(configured_camera_map or manifest_camera_map),
            action_horizon=action_horizon,
            image_height=int(payload["img_height"]),
            image_width=int(payload["img_width"]),
            gt_config=gt_config,
        )
        cache_path = (
            None
            if feature_cache_dir is None
            else feature_cache_dir / f"round_{source_round:03d}.pt"
        )
        cached = (
            None
            if cache_path is None
            else load_round_feature_cache(
                cache_path,
                expected_round_index=source_round,
                expected_cache_key=cache_key,
                device=device,
            )
        )
        if cached is None:
            offline_pools = encode_policy_segments(
                segments,
                encoder=encoder,
                camera_names=list(payload["camera_names"]),
                batch_size=encode_batch_size,
            )
            gt_negative = (
                encode_gt_negative_windows(
                    gt_windows,
                    encoder=encoder,
                    camera_names=list(payload["camera_names"]),
                    batch_size=encode_batch_size,
                )
                if gt_negative_active
                else []
            )
            positive = list(offline_pools.positive)
            negative = list(gt_negative)
            if cache_path is not None:
                save_round_feature_cache(
                    cache_path,
                    round_index=source_round,
                    cache_key=cache_key,
                    offline_positive=positive,
                    offline_gt_negative=negative,
                )
        else:
            positive = list(cached.offline_positive)
            negative = list(cached.offline_gt_negative)
        round_positive[source_round] = positive
        round_gt_negative[source_round] = negative
        all_segments.extend(segments)
        for key, value in current_segment_stats.items():
            if isinstance(value, (int, float)):
                segment_stats[key] = segment_stats.get(key, 0) + value
        for key, value in current_gt_stats.items():
            if isinstance(value, (int, float)):
                gt_window_stats[key] = gt_window_stats.get(key, 0) + value
        all_reserved_shards.extend(
            {
                "round_index": source_round,
                "source_episode_index": int(key[0]),
                "frame_index": int(key[1]),
            }
            for key in current_gt_stats["selected_frame_keys"]
        )
        round_records.append(
            {
                "round_index": source_round,
                "episodes_path": str(source_path),
                "episodes_sha256": sha256_file(source_path),
                "feature_cache_path": None if cache_path is None else str(cache_path),
                "feature_cache_key": cache_key,
                "feature_cache_hit": cached is not None,
                "segment_stats": current_segment_stats,
                "gt_negative_rule": current_gt_stats,
                "offline_positive": trajectory_stats(positive),
                "offline_gt_negative": trajectory_stats(negative),
            }
        )
    gt_window_stats["theory_deviation"] = current_gt_stats["theory_deviation"]
    gt_window_stats["selected_frame_keys"] = [
        [record["source_episode_index"], record["frame_index"]]
        for record in all_reserved_shards
        if record["round_index"] == round_index
    ]

    positive_plan = resolve_round_mixture_plan(
        pool_name="offline_positive",
        current_round=round_index,
        history_mix_beta=float(finetune_cfg.history_mix_beta),
        round_pool_sizes={
            index: int(trajectory_stats(items)["frames"])
            for index, items in round_positive.items()
        },
        configured_loss_weight=max(
            (
                term.weight
                for term in loss_terms
                if "offline_positive" in term.pool_batch_sizes
            ),
            default=0.0,
        ),
    )
    negative_plan = resolve_round_mixture_plan(
        pool_name="offline_gt_negative",
        current_round=round_index,
        history_mix_beta=float(finetune_cfg.history_mix_beta),
        round_pool_sizes={
            index: int(trajectory_stats(items)["frames"])
            for index, items in round_gt_negative.items()
        },
        configured_loss_weight=max(
            (
                term.weight
                for term in loss_terms
                if "offline_gt_negative" in term.pool_batch_sizes
            ),
            default=0.0,
        ),
    )
    round_plans = {
        "offline_positive": positive_plan,
        "offline_gt_negative": negative_plan,
    }
    objective_config = _apply_effective_round_weights(
        objective_config,
        plans=round_plans,
    )
    loss_terms = configured_loss_terms(objective_config)
    active_term_types = {term.type for term in loss_terms if term.active}
    active_training_pools = active_training_pool_names(loss_terms)

    named_trajectories = {
        "pretrain_positive": list(pretrain_pools.positive),
        "pretrain_unlabeled": list(pretrain_pools.unlabeled),
        "pretrain_calibration": list(pretrain_pools.calibration),
        "offline_positive": [
            item for items in round_positive.values() for item in items
        ],
        "offline_gt_negative": [
            item for items in round_gt_negative.values() for item in items
        ],
    }
    named_pool_stats = {
        name: trajectory_stats(items) for name, items in named_trajectories.items()
    }
    reserved_stats = {
        "trajectories": 0,
        "frames": 0,
        "excluded_gt_negative_frames": len(all_reserved_shards),
    }
    reserved_shards = all_reserved_shards
    named_pool_stats["offline_unlabeled_reserved"] = reserved_stats
    empty_active_pools = [
        name
        for name in active_training_pools
        if int(named_pool_stats[name]["frames"]) <= 0
    ]
    if empty_active_pools:
        raise ValueError(
            f"Active objectives require non-empty pools: {empty_active_pools}."
        )
    latent_dims = {
        int(stats["latent_dim"])
        for stats in named_pool_stats.values()
        if int(stats.get("latent_dim", 0)) > 0
    }
    if latent_dims != {int(detector.in_dim)}:
        raise ValueError(
            f"Named pool latent dims {sorted(latent_dims)} != head dim {detector.in_dim}."
        )

    cuda_feature_pools = {
        name: _cuda_feature_tensors(items, device=device)
        for name, items in named_trajectories.items()
        if name in active_training_pools
    }
    cuda_round_positive = {
        index: _cuda_feature_tensors(items, device=device)
        for index, items in round_positive.items()
    }
    cuda_round_gt_negative = {
        index: _cuda_feature_tensors(items, device=device)
        for index, items in round_gt_negative.items()
    }
    pool_sampler = BatchOnlineCUDAPoolSampler(
        {
            name: tensors
            for name, tensors in cuda_feature_pools.items()
            if name in active_training_pools
            and name not in {"offline_positive", "offline_gt_negative"}
        },
        round_pools={
            "offline_positive": cuda_round_positive,
            "offline_gt_negative": cuda_round_gt_negative,
        },
        plans=round_plans,
        device=device,
        seed=int(cfg.seed),
    )

    parent_nnpu = resolve_parent_nnpu_semantics(parent_payload)
    finetune_config: dict[str, Any] = {
        "method": finetune_method,
        "optimizer": "AdamW",
        "schedule": "cosine",
        "scheduler_horizon_epochs": scheduler_horizon_epochs,
        "epochs": epochs,
        "lr": float(finetune_cfg.lr),
        "weight_decay": float(finetune_cfg.weight_decay),
        "encode_batch_size": encode_batch_size,
        "log_interval": log_interval,
        "seed": int(cfg.seed),
        "device": str(device),
        "objective": objective_config,
        "sampler": resolved_sampler_config(
            loss_terms,
            seed=int(cfg.seed),
            device=str(device),
        ),
        "round_mixtures": {
            name: plan.to_record() for name, plan in round_plans.items()
        },
        "gt_negative": gt_config,
        "parent_nnpu": {
            key: parent_nnpu[key]
            for key in ("pi_p", "loss_surrogate", "nn_correction", "beta")
        },
        "delta": float(parent_nnpu["delta"]),
    }
    if resolved_positive_safety_margin is not None:
        finetune_config["resolved_positive_safety_margin"] = dict(
            resolved_positive_safety_margin
        )
    data_provenance = {
        "parent_checkpoint_sha256": sha256_file(parent_checkpoint),
        "pretrain_manifest_path": str(
            pretrain_path / "manifest.json" if pretrain_path.is_dir() else pretrain_path
        ),
        "pretrain_manifest": pretrain_manifest,
        "offline_episodes_path": str(episodes_path),
        "offline_episodes_paths": [str(path) for path in episodes_paths],
        "round_index": round_index,
        "round_feature_caches": round_records,
        "offline_nnpu_checkpoint": offline_payload.get("nnpu_checkpoint"),
        "parent_model_checkpoint": parent_payload.get("model_ckpt"),
        "resolved_encoder_checkpoint": str(encoder.encoder_checkpoint),
        "resolved_encoder_sha256": sha256_file(encoder.encoder_checkpoint),
        "resolved_normalizer_checkpoint": encoder.normalizer_checkpoint,
        "resolved_normalizer_sha256": sha256_file(encoder.normalizer_checkpoint),
        "segment_stats": segment_stats,
        "gt_negative_rule": gt_window_stats,
        "named_pool_stats": named_pool_stats,
        "named_pool_sources": {
            "pretrain_positive": {
                "dataset": "pretrain_manifest",
                "split": "positive_train",
                "role": "nnpu_positive",
            },
            "pretrain_unlabeled": {
                "dataset": "pretrain_manifest",
                "split": "unlabeled_train",
                "role": "nnpu_unlabeled",
            },
            "pretrain_calibration": {
                "dataset": "pretrain_manifest",
                "split": "positive_calib",
                "role": "success_only_recalibration",
            },
            "offline_positive": {
                "dataset": "offline_episodes",
                "observations": "final_policy_segment_direct_success",
                "actions": "executed_action_equal_to_policy_action",
                "role": (
                    "gt_positive_bce_plus_safety_margin"
                    if "positive_safety_margin" in active_term_types
                    else (
                        "gt_positive_logistic_only"
                        if "positive_logistic" in active_term_types
                        else "reserved_not_used_by_active_loss"
                    )
                ),
            },
            "offline_gt_negative": {
                "dataset": "offline_episodes",
                "observations": "policy_prefix_and_human_intervention",
                "actions": "policy_action",
                "role": (
                    "gt_negative_logistic_only"
                    if "negative_logistic" in active_term_types
                    else (
                        "supervised_bce_negative"
                        if "supervised_bce" in active_term_types
                        else "reserved_not_used_by_active_loss"
                    )
                ),
            },
            "offline_unlabeled_reserved": {
                "dataset": "offline_episodes",
                "observations": "policy_only_excluding_gt_selected_frames",
                "role": "reserved_not_used_by_active_loss",
            },
        },
        "active_training_pools": active_training_pools,
        "reserved_pools": ["offline_unlabeled_reserved"],
        "offline_unlabeled_reserved_shards": reserved_shards,
        "offline_segments": [
            {
                "id": segment.identifier,
                "pool": segment.pool,
                "source_episode_index": segment.source_episode_index,
                "frame_start": segment.lo,
                "frame_end": segment.hi,
                "terminal_reason": segment.terminal_reason,
                "ended_by": segment.ended_by,
            }
            for segment in all_segments
        ],
        "calibration_source": "pretrain_positive_calib_only",
        "theory_deviation": gt_window_stats["theory_deviation"],
    }

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    explicit_run_dir = OmegaConf.select(
        cfg, "offline.discriminator_finetune.run_dir", default=None
    )
    if explicit_run_dir is None or str(explicit_run_dir).strip().lower() in {
        "",
        "none",
        "null",
    }:
        directory_name = f"{task_name}_{timestamp}"
        run_root = Path(to_absolute_path(str(finetune_cfg.run_root))).resolve()
        pipeline_run_dir = run_root / directory_name
        run_dir = pipeline_run_dir / "discriminator"
        if pipeline_run_dir.exists():
            raise FileExistsError(
                f"Finetune pipeline directory already exists: {pipeline_run_dir}"
            )
    else:
        run_dir = Path(to_absolute_path(str(explicit_run_dir))).resolve()
        pipeline_run_dir = run_dir.parent
    if run_dir.exists():
        raise FileExistsError(f"Finetune run directory already exists: {run_dir}")
    (run_dir / "checkpoints").mkdir(parents=True)
    run_name = f"{task_name}__discriminator_finetune__{timestamp}"
    print(f"[robosuite][pu_bce] pipeline_run_dir={pipeline_run_dir}", flush=True)
    print(f"[robosuite][pu_bce] stage_run_dir={run_dir}", flush=True)
    write_resolved_config(cfg, run_dir)
    metric_logger = maybe_build_metric_logger(cfg, run_name=run_name, run_dir=run_dir)

    initial_metrics: dict[str, float] = {
        f"data/{name}_frames": float(stats["frames"])
        for name, stats in named_pool_stats.items()
    }
    initial_metrics.update(
        {
            f"data/{name}_trajectories": float(stats["trajectories"])
            for name, stats in named_pool_stats.items()
        }
    )
    for pool_name, plan in round_plans.items():
        initial_metrics[f"round_mix/{pool_name}/configured_weight"] = float(
            plan.configured_loss_weight
        )
        initial_metrics[f"round_mix/{pool_name}/effective_weight"] = float(
            plan.effective_loss_weight
        )
        initial_metrics[f"round_mix/{pool_name}/skipped"] = float(not plan.active)
        for source_round, weight in plan.effective_round_weights.items():
            initial_metrics[
                f"round_mix/{pool_name}/round_{source_round:03d}_weight"
            ] = float(weight)
    initial_metrics.update(
        {
            "data/gt_intervention_events": float(
                gt_window_stats["intervention_events"]
            ),
            "data/gt_negative_windows": float(
                gt_window_stats["gt_negative_windows"]
            ),
            "data/gt_pre_frames": float(gt_window_stats["pre_frames"]),
            "data/gt_post_frames": float(gt_window_stats["post_frames"]),
            "data/gt_pre_truncated_events": float(
                gt_window_stats["pre_truncated_events"]
            ),
            "data/gt_post_truncated_events": float(
                gt_window_stats["post_truncated_events"]
            ),
            "data/gt_deduplicated_frames": float(
                gt_window_stats["deduplicated_frames"]
            ),
            "data/gt_non_success_episodes": float(
                gt_window_stats["non_success_episodes"]
            ),
            "data/gt_pre_end_windows": float(
                gt_window_stats["pre_end_windows"]
            ),
            "data/gt_pre_end_frames": float(
                gt_window_stats["pre_end_frames"]
            ),
            "data/gt_pre_end_truncated_events": float(
                gt_window_stats["pre_end_truncated_events"]
            ),
            "data/excluded_human_frames": float(
                segment_stats["excluded_human_frames"]
            ),
        }
    )
    maybe_log(metric_logger, initial_metrics, step=0)

    last_global_step = 0

    def log_step(metrics: dict[str, float]) -> None:
        nonlocal last_global_step
        last_global_step = int(metrics["global_step"])
        maybe_log(
            metric_logger,
            {
                f"step/{key}": value
                for key, value in metrics.items()
                if key not in {"global_step", "epoch", "epoch_step"}
            },
            step=last_global_step,
        )
        print(
            f"[pu_bce][step] step={last_global_step} "
            f"epoch={int(metrics['epoch']) + 1}/{epochs} "
            f"epoch_step={int(metrics['epoch_step'])} "
            f"loss={metrics['loss/total']:.5f} lr={metrics['lr']:.2e}",
            flush=True,
        )

    def log_epoch(metrics: dict[str, float]) -> None:
        nonlocal last_global_step
        last_global_step = int(metrics["global_step"])
        positive_bce = metrics.get(
            "gt/positive_bce",
            metrics.get("gt/positive_logistic", float("nan")),
        )
        maybe_log(
            metric_logger,
            {
                key: value
                for key, value in metrics.items()
                if key not in {"global_step", "epoch"}
            },
            step=last_global_step,
        )
        print(
            f"[pu_bce][fit] epoch={int(metrics['epoch']) + 1}/{epochs} "
            f"loss={metrics['loss/total']:.5f} "
            f"nnpu={metrics.get('loss/nnpu_replay/raw', float('nan')):.5f}/"
            f"{metrics.get('loss/nnpu_replay/weighted', float('nan')):.5f} "
            f"gt_p={metrics.get('loss/gt_positive/raw', float('nan')):.5f}/"
            f"{metrics.get('loss/gt_positive/weighted', float('nan')):.5f} "
            f"p_bce={positive_bce:.5f} "
            f"p_safe={metrics.get('gt/positive_safety_margin', float('nan')):.5f} "
            f"p_violate={metrics.get('safety/margin_violation_fraction', float('nan')):.3f} "
            f"gt_n={metrics.get('loss/gt_negative/raw', float('nan')):.5f}/"
            f"{metrics.get('loss/gt_negative/weighted', float('nan')):.5f} "
            f"cap={metrics.get('regularization/quadratic_logit_cap', 0.0):.5f}/"
            f"{metrics.get('regularization/quadratic_logit_cap_weighted', 0.0):.5f} "
            f"cap_out={metrics.get('regularization/quadratic_logit_cap_fraction_outside', 0.0):.3f} "
            f"batch={int(metrics.get('batch/pretrain_positive', 0.0))}/"
            f"{int(metrics.get('batch/pretrain_unlabeled', 0.0))}/"
            f"{int(metrics.get('batch/offline_positive', 0.0))}/"
            f"{int(metrics.get('batch/offline_gt_negative', 0.0))} "
            f"clamp_fraction={metrics.get('nnpu/clamp_fraction', 0.0):.3f} "
            f"delta_gt={metrics.get('scores/delta_gt', float('nan')):+.5f} "
            f"lr={metrics['lr']:.2e}",
            flush=True,
        )

    try:
        print(
            f"[robosuite][pu_bce] task={task_name} method={finetune_method} "
            f"epochs={epochs} lr={float(finetune_cfg.lr):.2e} "
            f"scheduler_horizon_epochs={scheduler_horizon_epochs} "
            f"feat_dim={int(detector.in_dim)} "
            f"NpreP={named_pool_stats['pretrain_positive']['frames']} "
            f"NpreU={named_pool_stats['pretrain_unlabeled']['frames']} "
            f"NoffP={named_pool_stats['offline_positive']['frames']} "
            f"NgtN={named_pool_stats['offline_gt_negative']['frames']}",
            flush=True,
        )
        if resolved_positive_safety_margin is not None:
            print(
                "[robosuite][pu_bce] positive_safety_margin "
                f"m_k={resolved_positive_safety_margin['m_k']:.10f} "
                f"target={resolved_positive_safety_margin['target_logit']:.10f} "
                f"mu={resolved_positive_safety_margin['safety_margin_weight']:.6g} "
                f"delta={resolved_positive_safety_margin['margin_delta']:.6g} "
                f"temperature={resolved_positive_safety_margin['temperature']:.6g}",
                flush=True,
            )
        thresholds = finetune_warmstart_detector(
            detector,
            feature_pools=cuda_feature_pools,
            calibration_features=feature_tensors(pretrain_pools.calibration),
            objective_config=objective_config,
            positive_safety_boundary=positive_safety_boundary,
            task_name=task_name,
            parent_payload=parent_payload,
            epochs=epochs,
            scheduler_horizon_epochs=scheduler_horizon_epochs,
            lr=float(finetune_cfg.lr),
            weight_decay=float(finetune_cfg.weight_decay),
            seed=int(cfg.seed),
            log_interval=log_interval,
            metric_callback=log_epoch,
            step_metric_callback=log_step,
            pool_sampler=pool_sampler,
            verbose=False,
        )
        if detector._train_history:  # noqa: SLF001
            finetune_config["resolved_steps_per_epoch"] = int(
                detector._train_history[-1]["steps"]  # noqa: SLF001
            )
        finetune_config["resolved_logit_normalization"] = dict(
            detector._logit_normalization  # noqa: SLF001
        )
        checkpoint_payload = build_finetuned_checkpoint_payload(
            detector,
            parent_payload=parent_payload,
            parent_checkpoint=parent_checkpoint,
            task_name=task_name,
            finetune_config=finetune_config,
            data_provenance=data_provenance,
            encoder_checkpoint=encoder.encoder_checkpoint,
        )
        output_checkpoint = save_finetuned_checkpoint(
            checkpoint_payload,
            run_dir / "checkpoints" / "pu_bce_head_finetuned.pth",
        )
        run_info = {
            "run_name": run_name,
            "run_dir": str(run_dir),
            "pipeline_run_dir": str(pipeline_run_dir),
            "stage_run_dir": str(run_dir),
            "started_at": timestamp,
            "task_name": task_name,
            "output_checkpoint": str(output_checkpoint),
            "parent_checkpoint": str(parent_checkpoint),
            "encoder_checkpoint": str(encoder.encoder_checkpoint),
            "from_init": False,
            "finetune_method": finetune_method,
            "finetune_config": finetune_config,
            "data": data_provenance,
            "thresholds": thresholds,
            "calibration": checkpoint_payload["finetune_recalibration"],
            "resolved_config": resolved_config_dict(cfg),
        }
        write_run_info(run_dir, run_info)
        maybe_log(
            metric_logger,
            {f"calibration/threshold/{key}": value for key, value in thresholds.items()},
            step=max(last_global_step, epochs),
        )
        print(f"[robosuite][pu_bce] saved checkpoint={output_checkpoint}", flush=True)
    finally:
        if metric_logger is not None:
            metric_logger.flush()
            metric_logger.close()
