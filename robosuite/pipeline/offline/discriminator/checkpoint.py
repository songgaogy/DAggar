"""Warm-start nnPU checkpoint loading and publication."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from .trainer import PUBCEDiscriminatorFT, require_cuda_device


def load_warmstart_detector(
    checkpoint: str | Path,
    *,
    device: str | torch.device,
    expected_task: str | None = None,
) -> tuple[PUBCEDiscriminatorFT, dict[str, Any]]:
    """Load a trainable nnPU head and validate the immutable latent contract."""
    cuda_device = require_cuda_device(device)
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"nnPU checkpoint not found: {path}")
    payload = torch.load(path, map_location=cuda_device, weights_only=False)
    if not isinstance(payload, dict) or "pu_bce_detector" not in payload:
        raise KeyError(f"{path} is not a pu_bce_head.pth checkpoint.")
    if str(payload.get("feature_source")) != "transformer":
        raise ValueError(
            "Offline discriminator finetuning requires feature_source='transformer', "
            f"got {payload.get('feature_source')!r}."
        )
    if int(payload.get("transformer_layer", -1)) != 1:
        raise ValueError(
            "Offline discriminator finetuning requires transformer_layer=1, "
            f"got {payload.get('transformer_layer')!r}."
        )

    state = dict(payload["pu_bce_detector"])
    in_dim = int(state.get("in_dim", payload.get("in_dim", 0)))
    hidden = int(state.get("hidden", payload.get("hidden", 0)))
    num_layers = int(state.get("num_layers", payload.get("num_layers", 0)))
    if min(in_dim, hidden, num_layers) <= 0:
        raise ValueError(
            f"Invalid nnPU architecture in {path}: "
            f"in_dim={in_dim}, hidden={hidden}, num_layers={num_layers}."
        )
    with torch.device(cuda_device):
        detector = PUBCEDiscriminatorFT(
            in_dim=in_dim,
            hidden=hidden,
            num_layers=num_layers,
            device=str(cuda_device),
        )
    detector.load_state_dict(state)
    detector.head.train()
    for parameter in detector.head.parameters():
        parameter.requires_grad_(True)

    if expected_task is not None and expected_task not in detector.thresholds:
        raise KeyError(
            f"Task {expected_task!r} is not calibrated in {path}; available tasks: "
            f"{sorted(detector.thresholds)}."
        )
    if expected_task is not None and set(detector.thresholds) != {str(expected_task)}:
        raise ValueError(
            "Standalone discriminator finetuning is single-task because every updated "
            "shared-head threshold must be recalibrated. Parent checkpoint tasks are "
            f"{sorted(detector.thresholds)}, requested task is {expected_task!r}."
        )
    payload["_resolved_checkpoint"] = str(path)
    return detector, payload


def build_finetuned_checkpoint_payload(
    detector: PUBCEDiscriminatorFT,
    *,
    parent_payload: dict[str, Any],
    parent_checkpoint: str | Path,
    task_name: str,
    finetune_config: dict[str, Any],
    data_provenance: dict[str, Any],
    encoder_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Build a backward-compatible pu_bce_head payload with finetune metadata."""
    parent_path = str(Path(parent_checkpoint).expanduser().resolve())
    payload = {
        key: value
        for key, value in parent_payload.items()
        if not str(key).startswith("_")
    }
    detector_state = detector.state_dict()
    manifest = data_provenance.get("pretrain_manifest", {})
    if isinstance(manifest, dict):
        manifest_checkpoint = manifest.get("checkpoint", {})
        split_config = manifest.get("split_config", {})
        trajectory_ids = manifest.get("trajectory_ids", {})
        if isinstance(split_config, dict):
            if split_config.get("seed") is not None:
                payload.setdefault("seed", int(split_config["seed"]))
            if split_config.get("calibration_fraction") is not None:
                payload.setdefault(
                    "calib_fraction", float(split_config["calibration_fraction"])
                )
        if isinstance(manifest_checkpoint, dict):
            model_sha256 = manifest_checkpoint.get("model_ckpt_sha256")
            if model_sha256 is not None:
                payload.setdefault("model_ckpt_sha256", str(model_sha256))
            normalizer_path = manifest_checkpoint.get("normalizer_ckpt")
            normalizer_sha256 = manifest_checkpoint.get("normalizer_ckpt_sha256")
            if normalizer_path is not None:
                payload.setdefault("normalizer_ckpt", str(normalizer_path))
            if normalizer_sha256 is not None:
                payload.setdefault("normalizer_ckpt_sha256", str(normalizer_sha256))
        feature_contract = manifest.get("feature_contract", {})
        if isinstance(feature_contract, dict):
            camera_map = feature_contract.get("camera_to_view")
            proprio_indices = feature_contract.get("proprio_indices")
            if camera_map is not None and not payload.get("camera_to_view"):
                payload["camera_to_view"] = {
                    str(key): str(value) for key, value in dict(camera_map).items()
                }
            if proprio_indices is not None and payload.get("proprio_indices") is None:
                payload["proprio_indices"] = [
                    int(value) for value in proprio_indices
                ]
        if isinstance(trajectory_ids, dict):
            positive_train_ids = trajectory_ids.get("positive_train")
            positive_calib_ids = trajectory_ids.get("positive_calib")
            unlabeled_ids = trajectory_ids.get("unlabeled_train")
            if positive_train_ids is not None:
                payload.setdefault(
                    "success_train_video_ids",
                    {str(task_name): [str(value) for value in positive_train_ids]},
                )
            if positive_calib_ids is not None:
                payload.setdefault(
                    "success_calib_video_ids",
                    {str(task_name): [str(value) for value in positive_calib_ids]},
                )
            if unlabeled_ids is not None:
                payload.setdefault(
                    "unlabeled_fail_video_ids",
                    [str(value) for value in unlabeled_ids],
                )
    payload.setdefault("delta", detector_state.get("delta"))
    if encoder_checkpoint is not None:
        payload["model_ckpt"] = str(Path(encoder_checkpoint).expanduser().resolve())
    payload.update(
        {
            "epoch": int(finetune_config["epochs"]),
            "in_dim": int(detector.in_dim),
            "hidden": int(detector.hidden),
            "num_layers": int(detector.num_layers),
            "pu_bce_detector": detector_state,
            "parent_nnpu_checkpoint": parent_path,
            "finetuned_offline": True,
            "finetune_schema_version": 1,
            "finetune_task": str(task_name),
            "finetune_config": dict(finetune_config),
            "finetune_data": dict(data_provenance),
            "finetune_history": list(detector_state.get("train_history", [])),
            "finetune_recalibration": {
                "source": "pretrain_positive_calib_only",
                "delta": detector_state.get("delta"),
                "thresholds": dict(detector_state.get("thresholds", {})),
                "calib_stats": dict(detector_state.get("calib_stats", {})),
            },
        }
    )
    return payload


def save_finetuned_checkpoint(payload: dict[str, Any], path: str | Path) -> Path:
    """Atomically save a finetuned checkpoint without exposing partial output."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, output)
    finally:
        if tmp.exists():
            tmp.unlink()
    return output


__all__ = [
    "build_finetuned_checkpoint_payload",
    "load_warmstart_detector",
    "save_finetuned_checkpoint",
]
