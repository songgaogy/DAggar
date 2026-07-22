"""Path, checksum, and feature-contract validation for nnPU finetuning."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf


def required_path(raw: Any, *, name: str) -> Path:
    if raw is None or str(raw).strip().lower() in {"", "none", "null"}:
        raise ValueError(f"{name} must be set.")
    path = Path(to_absolute_path(str(raw))).resolve()
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    return path


def optional_file(raw: Any, *, name: str) -> Path | None:
    if raw is None or str(raw).strip().lower() in {"", "none", "null"}:
        return None
    path = Path(to_absolute_path(str(raw))).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name} not found: {path}")
    return path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolved_config_dict(cfg: DictConfig) -> dict[str, Any]:
    resolved = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("Resolved Hydra configuration must be a mapping.")
    return resolved


def _task_trajectory_ids(
    raw: Any,
    *,
    task_name: str,
    field_name: str,
) -> list[str]:
    values = raw.get(task_name) if isinstance(raw, Mapping) else raw
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(
            f"{field_name} must contain a trajectory ID sequence for task "
            f"{task_name!r}."
        )
    result = [str(value) for value in values]
    if not isinstance(raw, Mapping):
        task_prefix = f"{task_name}/"
        task_values = [value for value in result if value.startswith(task_prefix)]
        if task_values:
            result = task_values
    if not result:
        raise ValueError(f"{field_name} is empty for task {task_name!r}.")
    if len(result) != len(set(result)):
        raise ValueError(f"{field_name} contains duplicate trajectory IDs.")
    return result


def validate_finetune_contract(
    task_name: str,
    parent_payload: Mapping[str, Any],
    pretrain_manifest: Mapping[str, Any],
    offline_payload: Mapping[str, Any],
    encoder: Any,
) -> None:
    manifest_task = str(pretrain_manifest.get("task", ""))
    if manifest_task != task_name:
        raise ValueError(
            f"Pretrain manifest task mismatch: expected {task_name!r}, "
            f"got {manifest_task!r}."
        )
    offline_task = str(offline_payload.get("task_name", ""))
    if offline_task != task_name:
        raise ValueError(
            f"Offline data task mismatch: expected {task_name!r}, got {offline_task!r}."
        )

    manifest_checkpoint = dict(pretrain_manifest.get("checkpoint", {}))

    manifest_ids = dict(pretrain_manifest.get("trajectory_ids", {}))
    id_contract = (
        (
            "positive_train",
            manifest_ids.get("positive_train"),
            parent_payload.get("success_train_video_ids"),
        ),
        (
            "positive_calib",
            manifest_ids.get("positive_calib"),
            parent_payload.get("success_calib_video_ids"),
        ),
        (
            "unlabeled_train",
            manifest_ids.get("unlabeled_train"),
            parent_payload.get("unlabeled_fail_video_ids"),
        ),
    )
    for pool_name, manifest_values, checkpoint_values in id_contract:
        manifest_pool_ids = _task_trajectory_ids(
            manifest_values,
            task_name=task_name,
            field_name=f"pretrain manifest trajectory_ids.{pool_name}",
        )
        checkpoint_pool_ids = _task_trajectory_ids(
            checkpoint_values,
            task_name=task_name,
            field_name=f"parent checkpoint {pool_name} trajectory IDs",
        )
        if manifest_pool_ids != checkpoint_pool_ids:
            raise ValueError(
                f"Pretrain manifest {pool_name} trajectory IDs differ from the "
                f"parent checkpoint for task {task_name!r}."
            )

    expected_model_sha = str(manifest_checkpoint.get("model_ckpt_sha256", ""))
    actual_model_sha = sha256_file(encoder.encoder_checkpoint)
    if not expected_model_sha or expected_model_sha != actual_model_sha:
        raise ValueError(
            "Frozen dynamics encoder does not match the pretrain latent manifest: "
            f"manifest_sha256={expected_model_sha!r}, "
            f"encoder_sha256={actual_model_sha!r}."
        )
    expected_normalizer_sha = str(
        manifest_checkpoint.get("normalizer_ckpt_sha256", "")
    )
    actual_normalizer_sha = sha256_file(encoder.normalizer_checkpoint)
    if not expected_normalizer_sha or expected_normalizer_sha != actual_normalizer_sha:
        raise ValueError(
            "Frozen dynamics normalizer does not match the pretrain latent manifest: "
            f"manifest_sha256={expected_normalizer_sha!r}, "
            f"normalizer_sha256={actual_normalizer_sha!r}."
        )

    feature = dict(pretrain_manifest.get("feature_contract", {}))
    expected_contract = {
        "feature_source": str(parent_payload.get("feature_source")),
        "transformer_layer": int(parent_payload.get("transformer_layer", -1)),
        "use_chunk": bool(parent_payload.get("use_chunk", False)),
        "latent_dim": int(dict(parent_payload["pu_bce_detector"])["in_dim"]),
    }
    actual_contract = {
        "feature_source": str(feature.get("feature_source")),
        "transformer_layer": int(feature.get("transformer_layer", -1)),
        "use_chunk": bool(feature.get("use_chunk", False)),
        "latent_dim": int(feature.get("latent_dim", -1)),
    }
    if actual_contract != expected_contract:
        raise ValueError(
            "Pretrain manifest feature contract differs from the parent checkpoint: "
            f"manifest={actual_contract}, checkpoint={expected_contract}."
        )
    encoder_contract = {
        "feature_source": str(encoder.feature_source),
        "transformer_layer": int(encoder.transformer_layer),
        "use_chunk": bool(encoder.use_chunk),
        "latent_dim": int(encoder.chunk_feature_dim),
        "view_names": list(encoder.inner_encoder.view_names),
        "camera_to_view": dict(encoder.camera_to_view),
        "proprio_indices": encoder.proprio_indices,
        "proprio_input_dim": int(encoder.proprio_input_dim),
        "frameskip": int(encoder.frameskip),
        "action_dim_per_step": int(encoder.action_dim_per_step),
        "action_input_dim": int(encoder.action_input_dim),
    }
    manifest_encoder_contract = {
        **actual_contract,
        "view_names": [str(value) for value in feature.get("view_names", [])],
        "camera_to_view": {
            str(key): str(value)
            for key, value in dict(feature.get("camera_to_view", {})).items()
        },
        "proprio_indices": (
            None
            if feature.get("proprio_indices") is None
            else [int(value) for value in feature["proprio_indices"]]
        ),
        "proprio_input_dim": int(feature.get("proprio_input_dim", -1)),
        "frameskip": int(feature.get("frameskip", -1)),
        "action_dim_per_step": int(feature.get("action_dim_per_step", -1)),
        "action_input_dim": int(feature.get("action_input_dim", -1)),
    }
    expected_encoder_contract = {
        **expected_contract,
        **{
            key: manifest_encoder_contract[key]
            for key in (
                "view_names",
                "camera_to_view",
                "proprio_indices",
                "proprio_input_dim",
                "frameskip",
                "action_dim_per_step",
                "action_input_dim",
            )
        },
    }
    if encoder_contract != expected_encoder_contract:
        raise ValueError(
            "Frozen dynamics encoder contract differs from the pretrain manifest: "
            f"encoder={encoder_contract}, manifest={expected_encoder_contract}."
        )


__all__ = [
    "optional_file",
    "required_path",
    "resolved_config_dict",
    "sha256_file",
    "validate_finetune_contract",
]
