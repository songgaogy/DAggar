"""Standalone CUDA-only warm-start finetuning for the offline nnPU head."""

from __future__ import annotations

import datetime
import tempfile
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from robosuite.pipeline.offline.discriminator import (
    FinetuneDynamicsEncoder,
    build_finetuned_checkpoint_payload,
    combine_pools,
    encode_policy_segments,
    feature_tensors,
    finetune_warmstart_detector,
    PUBCEDiscriminatorFT,
    load_offline_episodes,
    load_pretrain_pools,
    load_warmstart_detector,
    optional_file,
    required_path,
    require_cuda_device,
    resolved_config_dict,
    safe_run_suffix,
    save_finetuned_checkpoint,
    sha256_file,
    split_policy_segments,
    validate_finetune_contract,
)
from robosuite.pipeline.utils import (
    maybe_build_metric_logger,
    maybe_log,
    write_resolved_config,
    write_run_info,
)


DEFAULT_FROM_INIT_HEAD_HIDDEN = 256
DEFAULT_FROM_INIT_HEAD_LAYERS = 2
DEFAULT_FROM_INIT_PI_P = 0.5
DEFAULT_FROM_INIT_LOSS_SURROGATE = "logistic"
DEFAULT_FROM_INIT_NN_CORRECTION = True
DEFAULT_FROM_INIT_BETA = 0.0
DEFAULT_FROM_INIT_DELTA = 10.0


@hydra.main(version_base="1.2", config_path="../../config", config_name="finetune_disc")
def main(cfg: DictConfig) -> None:
    """Encode policy segments, warm-start the nnPU head, and recalibrate it."""
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
    parent_from_scratch = parent_checkpoint is None
    bootstrap_checkpoint: Path | None = None
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
    batch_size = int(finetune_cfg.batch_size)
    encode_batch_size = int(finetune_cfg.encode_batch_size)
    log_interval = int(finetune_cfg.log_interval)
    use_only_offline = bool(finetune_cfg.use_only_offline)
    if min(epochs, batch_size, encode_batch_size, log_interval) <= 0:
        raise ValueError("epochs, batch sizes, and log_interval must all be positive.")
    if batch_size < 2 or batch_size % 2 != 0:
        raise ValueError(
            "offline.discriminator_finetune.batch_size must be even and at least 2 "
            "for exact 50/50 P/U sampling."
        )
    if float(finetune_cfg.lr) < 0.0 or float(finetune_cfg.weight_decay) < 0.0:
        raise ValueError("Learning rate and weight decay must be non-negative.")

    pretrain_pools, pretrain_manifest = load_pretrain_pools(pretrain_path)
    offline_payload = load_offline_episodes(episodes_path)
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

    if parent_checkpoint is None:
        manifest_checkpoint = dict(pretrain_manifest.get("checkpoint", {}))
        if "model_ckpt" not in manifest_checkpoint:
            raise ValueError(
                "from-init mode requires pretrain manifest field checkpoint.model_ckpt."
            )
        feature_source = str(manifest_feature.get("feature_source", "transformer"))
        transformer_layer = int(manifest_feature.get("transformer_layer", 1))
        use_chunk = bool(manifest_feature.get("use_chunk", False))
        model_ckpt = manifest_checkpoint.get("model_ckpt")
        if model_ckpt is None:
            raise ValueError(
                "from-init mode requires pretrain manifest field checkpoint.model_ckpt."
            )
        normalized_model_ckpt = to_absolute_path(str(model_ckpt))
        model_ckpt_sha = manifest_checkpoint.get("model_ckpt_sha256")
        normalizer_ckpt = manifest_checkpoint.get("normalizer_ckpt")
        normalizer_ckpt_sha = manifest_checkpoint.get("normalizer_ckpt_sha256")
        in_dim = int(manifest_feature.get("latent_dim", 0))
        if in_dim <= 0:
            raise ValueError(
                "from-init mode requires feature_contract.latent_dim > 0 in manifest."
            )
        with tempfile.NamedTemporaryFile(suffix=".pth", prefix="robosuite-", delete=False) as temp_file:
            bootstrap_checkpoint = Path(temp_file.name)
            init_probe = PUBCEDiscriminatorFT(
                in_dim=in_dim,
                hidden=DEFAULT_FROM_INIT_HEAD_HIDDEN,
                num_layers=DEFAULT_FROM_INIT_HEAD_LAYERS,
                device=str(device),
            )
            synthetic_payload = {
                "feature_source": feature_source,
                "transformer_layer": transformer_layer,
                "use_chunk": use_chunk,
                "model_ckpt": normalized_model_ckpt,
                "in_dim": in_dim,
                "hidden": DEFAULT_FROM_INIT_HEAD_HIDDEN,
                "num_layers": DEFAULT_FROM_INIT_HEAD_LAYERS,
                "model_ckpt_sha256": model_ckpt_sha,
                "normalizer_ckpt": normalizer_ckpt,
                "normalizer_ckpt_sha256": normalizer_ckpt_sha,
                "pi_p": DEFAULT_FROM_INIT_PI_P,
                "loss_surrogate": DEFAULT_FROM_INIT_LOSS_SURROGATE,
                "nn_correction": DEFAULT_FROM_INIT_NN_CORRECTION,
                "beta": DEFAULT_FROM_INIT_BETA,
                "delta": DEFAULT_FROM_INIT_DELTA,
                "pu_bce_detector": init_probe.state_dict(),
                "proprio_indices": None
                if manifest_feature.get("proprio_indices") is None
                else [int(value) for value in manifest_feature.get("proprio_indices")],
            }
            torch.save(synthetic_payload, temp_file.name)
            parent_checkpoint = bootstrap_checkpoint
            detector = init_probe
            parent_payload: dict[str, Any] = synthetic_payload.copy()
    else:
        detector, parent_payload = load_warmstart_detector(
            parent_checkpoint,
            device=device,
            expected_task=task_name,
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
        require_parent_checksum_match=not parent_from_scratch,
    )
    segments, segment_stats = split_policy_segments(offline_payload)
    online_pools = encode_policy_segments(
        segments,
        encoder=encoder,
        camera_names=list(offline_payload["camera_names"]),
        batch_size=encode_batch_size,
    )
    combined_pools = combine_pools(
        pretrain_pools,
        online_pools,
        use_only_offline=use_only_offline,
    )
    if int(combined_pools.stats["latent_dim"]) != int(detector.in_dim):
        raise ValueError(
            f"Combined latent dim {combined_pools.stats['latent_dim']} "
            f"!= head dim {detector.in_dim}."
        )

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
    parent_checkpoint_tag = "from_init" if parent_from_scratch else str(parent_checkpoint)
    write_resolved_config(cfg, run_dir)
    metric_logger = maybe_build_metric_logger(cfg, run_name=run_name, run_dir=run_dir)

    detector_state = parent_payload["pu_bce_detector"]
    delta_value = parent_payload.get("delta")
    if delta_value is None:
        delta_value = detector_state.get("delta")
    if delta_value is None:
        if parent_from_scratch:
            delta_value = DEFAULT_FROM_INIT_DELTA
        else:
            raise KeyError("Parent nnPU payload is missing calibration delta.")
    pi_p_value = parent_payload.get("pi_p")
    if pi_p_value is None:
        pi_p_value = detector_state.get("pi_p")
    if pi_p_value is None:
        raise KeyError("Parent nnPU payload is missing class prior pi_p.")
    loss_surrogate_value = parent_payload.get("loss_surrogate")
    if loss_surrogate_value is None:
        loss_surrogate_value = detector_state.get("loss_surrogate", "logistic")
    nn_correction_value = parent_payload.get("nn_correction")
    if nn_correction_value is None:
        nn_correction_value = detector_state.get("nn_correction", True)
    beta_value = parent_payload.get("beta")
    if beta_value is None:
        beta_value = detector_state.get("beta", DEFAULT_FROM_INIT_BETA)
    finetune_config = {
        "optimizer": "AdamW",
        "schedule": "cosine",
        "epochs": epochs,
        "lr": float(finetune_cfg.lr),
        "weight_decay": float(finetune_cfg.weight_decay),
        "batch_size": batch_size,
        "encode_batch_size": encode_batch_size,
        "use_only_offline": use_only_offline,
        "seed": int(cfg.seed),
        "device": str(device),
        "pi_p": float(pi_p_value),
        "loss_surrogate": str(loss_surrogate_value),
        "nn_correction": bool(nn_correction_value),
        "beta": float(beta_value),
        "delta": float(delta_value),
    }
    data_provenance = {
        "parent_checkpoint_sha256": sha256_file(parent_checkpoint),
        "pretrain_manifest_path": str(
            pretrain_path / "manifest.json" if pretrain_path.is_dir() else pretrain_path
        ),
        "pretrain_manifest": pretrain_manifest,
        "offline_episodes_path": str(episodes_path),
        "offline_nnpu_checkpoint": offline_payload.get("nnpu_checkpoint"),
        "use_only_offline": use_only_offline,
        "parent_model_checkpoint": parent_payload.get("model_ckpt"),
        "resolved_encoder_checkpoint": str(encoder.encoder_checkpoint),
        "resolved_encoder_sha256": sha256_file(encoder.encoder_checkpoint),
        "resolved_normalizer_checkpoint": encoder.normalizer_checkpoint,
        "resolved_normalizer_sha256": sha256_file(encoder.normalizer_checkpoint),
        "segment_stats": segment_stats,
        "pool_stats": combined_pools.stats,
        "online_segments": [
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
    }

    maybe_log(
        metric_logger,
        {
            "data/positive_frames": combined_pools.stats["positive_frames"],
            "data/unlabeled_frames": combined_pools.stats["unlabeled_frames"],
            "data/calibration_frames": combined_pools.stats["calibration_frames"],
            "data/online_positive_frames": online_pools.stats["positive_frames"],
            "data/online_unlabeled_frames": online_pools.stats["unlabeled_frames"],
            "data/excluded_human_frames": segment_stats["excluded_human_frames"],
        },
        step=0,
    )

    def log_epoch(metrics: dict[str, float]) -> None:
        step = int(metrics["epoch"]) + 1
        maybe_log(
            metric_logger,
            {
                f"finetune/{key}": value
                for key, value in metrics.items()
                if key != "epoch"
            },
            step=step,
        )
        nn_corr_batches = int(metrics["nn_correction_batches"])
        nn_corr_steps = max(1, int(metrics["steps"]))
        print(
            f"[pu_bce][fit] epoch={step}/{epochs} "
            f"risk={metrics['risk']:.5f} neg_risk={metrics['neg_risk']:+.5f} "
            f"nn_corr_batches={nn_corr_batches}/{nn_corr_steps} "
            f"lr={metrics['lr']:.2e} Np={int(metrics['num_positive_frames'])} "
            f"Nu={int(metrics['num_unlabeled_frames'])} "
            f"pi_p={finetune_config['pi_p']:.3f} "
            f"surrogate={finetune_config['loss_surrogate']} "
            f"nn_correction={finetune_config['nn_correction']} "
            f"beta={finetune_config['beta']:.3g}",
            flush=True,
        )

    try:
        print(
            f"[robosuite][pu_bce] task={task_name} epochs={epochs} lr={float(finetune_cfg.lr):.2e} "
            f"init_mode={'from_init' if parent_from_scratch else 'from_checkpoint'} "
            f"batch_size={batch_size} feat_dim={int(detector.in_dim)} "
            f"Np={int(combined_pools.stats['positive_frames'])} "
            f"Nu={int(combined_pools.stats['unlabeled_frames'])} "
            f"N_calib={int(combined_pools.stats['calibration_frames'])}",
            flush=True,
        )
        thresholds = finetune_warmstart_detector(
            detector,
            positive_features=feature_tensors(combined_pools.positive),
            unlabeled_features=feature_tensors(combined_pools.unlabeled),
            calibration_features=feature_tensors(combined_pools.calibration),
            task_name=task_name,
            parent_payload=parent_payload,
            epochs=epochs,
            lr=float(finetune_cfg.lr),
            weight_decay=float(finetune_cfg.weight_decay),
            batch_size=batch_size,
            seed=int(cfg.seed),
            metric_callback=log_epoch,
            verbose=False,
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
            "parent_checkpoint": parent_checkpoint_tag,
            "encoder_checkpoint": str(encoder.encoder_checkpoint),
            "from_init": bool(parent_from_scratch),
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
            step=epochs,
        )
        print(f"[robosuite][pu_bce] saved checkpoint={output_checkpoint}", flush=True)
    finally:
        if bootstrap_checkpoint is not None and bootstrap_checkpoint.exists():
            bootstrap_checkpoint.unlink()
        if metric_logger is not None:
            metric_logger.flush()
            metric_logger.close()


if __name__ == "__main__":
    main()
