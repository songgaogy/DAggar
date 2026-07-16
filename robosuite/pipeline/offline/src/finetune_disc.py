"""Standalone CUDA-only nnPU-replay plus supervised-GT finetuning."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from robosuite.pipeline.offline.discriminator import (
    FinetuneDynamicsEncoder,
    build_finetuned_checkpoint_payload,
    build_gt_negative_windows,
    encode_gt_negative_windows,
    encode_policy_segments,
    feature_tensors,
    finetune_warmstart_detector,
    load_offline_episodes,
    load_pretrain_pools,
    load_warmstart_detector,
    optional_file,
    required_path,
    require_cuda_device,
    resolve_parent_nnpu_semantics,
    resolved_config_dict,
    safe_run_suffix,
    save_finetuned_checkpoint,
    sha256_file,
    split_policy_segments,
    validate_finetune_contract,
)
from robosuite.pipeline.offline.discriminator.finetune_setup import (
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


@hydra.main(version_base="1.2", config_path="../../config", config_name="finetune_disc")
def main(cfg: DictConfig) -> None:
    """Encode named pools, warm-start the nnPU head, and recalibrate it."""
    if bool(OmegaConf.select(cfg, "logging.use_wandb", default=False)):
        raise ValueError(
            "Discriminator finetuning supports TensorBoard only; WandB is disabled."
        )

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
        raise ValueError(
            "algorithm.discriminator.checkpoint is required; Step 3 supports "
            "warm-start finetuning only."
        )
    encoder_checkpoint = optional_file(
        disc_cfg.encoder_ckpt,
        name="algorithm.discriminator.encoder_ckpt",
    )
    episodes_path = required_path(
        finetune_cfg.episodes_path,
        name="offline.discriminator_finetune.episodes_path",
    )
    pretrain_path = required_path(
        finetune_cfg.pretrain_dir,
        name="offline.discriminator_finetune.pretrain_dir",
    )

    epochs = int(finetune_cfg.epochs)
    encode_batch_size = int(finetune_cfg.encode_batch_size)
    log_interval = int(finetune_cfg.log_interval)
    if min(epochs, encode_batch_size, log_interval) <= 0:
        raise ValueError("epochs, encode_batch_size, and log_interval must be positive.")
    if float(finetune_cfg.lr) < 0.0 or float(finetune_cfg.weight_decay) < 0.0:
        raise ValueError("Learning rate and weight decay must be non-negative.")

    objective_config = mapping_config(
        finetune_cfg.objective,
        name="offline.discriminator_finetune.objective",
    )
    loss_terms = configured_loss_terms(objective_config)
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
    if pre_intervention_chunks < 0 or post_intervention_chunks <= 0:
        raise ValueError(
            "gt_negative pre_intervention_chunks must be non-negative and "
            "post_intervention_chunks must be positive."
        )
    pre_end_chunks = int(gt_config.get("pre_end_chunk", 0))
    if pre_end_chunks < 0:
        raise ValueError("gt_negative pre_end_chunk must be non-negative.")

    pretrain_pools, pretrain_manifest = load_pretrain_pools(pretrain_path)
    offline_payload = load_offline_episodes(episodes_path)
    detector, parent_payload = load_warmstart_detector(
        parent_checkpoint,
        device=device,
        expected_task=task_name,
    )

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
    validate_finetune_contract(
        task_name=task_name,
        parent_checkpoint=parent_checkpoint,
        parent_payload=parent_payload,
        pretrain_manifest=pretrain_manifest,
        offline_payload=offline_payload,
        encoder=encoder,
        require_parent_checksum_match=True,
    )

    segments, segment_stats = split_policy_segments(offline_payload)
    offline_pools = encode_policy_segments(
        segments,
        encoder=encoder,
        camera_names=list(offline_payload["camera_names"]),
        batch_size=encode_batch_size,
    )
    gt_windows, gt_window_stats = build_gt_negative_windows(
        offline_payload,
        pre_intervention_chunks=pre_intervention_chunks,
        post_intervention_chunks=post_intervention_chunks,
        frameskip=int(encoder.inner_encoder.frameskip),
        pre_end_chunks=pre_end_chunks,
    )
    supervised_gt_active = any(
        term.active and term.type == "supervised_bce" for term in loss_terms
    )
    gt_negative = (
        encode_gt_negative_windows(
            gt_windows,
            encoder=encoder,
            camera_names=list(offline_payload["camera_names"]),
            batch_size=encode_batch_size,
        )
        if supervised_gt_active
        else []
    )

    named_trajectories = {
        "pretrain_positive": list(pretrain_pools.positive),
        "pretrain_unlabeled": list(pretrain_pools.unlabeled),
        "pretrain_calibration": list(pretrain_pools.calibration),
        "offline_positive": list(offline_pools.positive),
        "offline_gt_negative": list(gt_negative),
    }
    named_pool_stats = {
        name: trajectory_stats(items) for name, items in named_trajectories.items()
    }
    reserved_stats, reserved_shards = reserved_unlabeled_provenance(
        offline_pools.unlabeled,
        selected_frame_keys=gt_window_stats["selected_frame_keys"],
    )
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

    parent_nnpu = resolve_parent_nnpu_semantics(parent_payload)
    finetune_config: dict[str, Any] = {
        "method": "nnpu_replay_gt_bce",
        "optimizer": "AdamW",
        "schedule": "cosine",
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
        "gt_negative": gt_config,
        "parent_nnpu": {
            key: parent_nnpu[key]
            for key in ("pi_p", "loss_surrogate", "nn_correction", "beta")
        },
        "delta": float(parent_nnpu["delta"]),
    }
    data_provenance = {
        "parent_checkpoint_sha256": sha256_file(parent_checkpoint),
        "pretrain_manifest_path": str(
            pretrain_path / "manifest.json" if pretrain_path.is_dir() else pretrain_path
        ),
        "pretrain_manifest": pretrain_manifest,
        "offline_episodes_path": str(episodes_path),
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
                "role": "supervised_bce_positive",
            },
            "offline_gt_negative": {
                "dataset": "offline_episodes",
                "observations": "policy_prefix_and_human_intervention",
                "actions": "policy_action",
                "role": "supervised_bce_negative",
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
            for segment in segments
        ],
        "calibration_source": "pretrain_positive_calib_only",
        "theory_deviation": gt_window_stats["theory_deviation"],
    }

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = safe_run_suffix(finetune_cfg.run_subfix)
    directory_name = (
        f"{task_name}_{timestamp}_{suffix}" if suffix else f"{task_name}_{timestamp}"
    )
    run_root = Path(to_absolute_path(str(finetune_cfg.run_root))).resolve()
    run_dir = run_root / directory_name
    if run_dir.exists():
        raise FileExistsError(f"Finetune run directory already exists: {run_dir}")
    (run_dir / "checkpoints").mkdir(parents=True)
    run_name = f"{task_name}__discriminator_finetune__{timestamp}"
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
            f"nnpu={metrics.get('loss/nnpu_replay/raw', float('nan')):.5f} "
            f"gt_bce={metrics.get('loss/supervised_gt_bce/raw', float('nan')):.5f} "
            f"clamp_fraction={metrics.get('nnpu/clamp_fraction', 0.0):.3f} "
            f"delta_gt={metrics.get('scores/delta_gt', float('nan')):+.5f} "
            f"lr={metrics['lr']:.2e}",
            flush=True,
        )

    try:
        print(
            f"[robosuite][pu_bce] task={task_name} method=nnpu_replay_gt_bce "
            f"epochs={epochs} lr={float(finetune_cfg.lr):.2e} "
            f"feat_dim={int(detector.in_dim)} "
            f"NpreP={named_pool_stats['pretrain_positive']['frames']} "
            f"NpreU={named_pool_stats['pretrain_unlabeled']['frames']} "
            f"NoffP={named_pool_stats['offline_positive']['frames']} "
            f"NgtN={named_pool_stats['offline_gt_negative']['frames']}",
            flush=True,
        )
        thresholds = finetune_warmstart_detector(
            detector,
            feature_pools={
                name: feature_tensors(items)
                for name, items in named_trajectories.items()
                if name in active_training_pools
            },
            calibration_features=feature_tensors(pretrain_pools.calibration),
            objective_config=objective_config,
            task_name=task_name,
            parent_payload=parent_payload,
            epochs=epochs,
            lr=float(finetune_cfg.lr),
            weight_decay=float(finetune_cfg.weight_decay),
            seed=int(cfg.seed),
            log_interval=log_interval,
            metric_callback=log_epoch,
            step_metric_callback=log_step,
            verbose=False,
        )
        if detector._train_history:  # noqa: SLF001
            finetune_config["resolved_steps_per_epoch"] = int(
                detector._train_history[-1]["steps"]  # noqa: SLF001
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
            "started_at": timestamp,
            "task_name": task_name,
            "output_checkpoint": str(output_checkpoint),
            "parent_checkpoint": str(parent_checkpoint),
            "encoder_checkpoint": str(encoder.encoder_checkpoint),
            "from_init": False,
            "finetune_method": "nnpu_replay_gt_bce",
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


if __name__ == "__main__":
    main()
