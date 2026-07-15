"""Path, checksum, and feature-contract validation for nnPU finetuning."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping

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


def safe_run_suffix(raw: Any) -> str:
    suffix = str(raw or "").strip()
    if suffix and re.fullmatch(r"[A-Za-z0-9_.-]+", suffix) is None:
        raise ValueError(
            "offline.discriminator_finetune.run_subfix may contain only "
            "letters, digits, underscore, dot, and dash."
        )
    return suffix


def resolved_config_dict(cfg: DictConfig) -> dict[str, Any]:
    resolved = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("Resolved Hydra configuration must be a mapping.")
    return resolved


def validate_finetune_contract(
    task_name: str,
    parent_checkpoint: Path,
    parent_payload: Mapping[str, Any],
    pretrain_manifest: Mapping[str, Any],
    offline_payload: Mapping[str, Any],
    encoder: Any,
    require_parent_checksum_match: bool = True,
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

    if require_parent_checksum_match:
        manifest_checkpoint = dict(pretrain_manifest.get("checkpoint", {}))
        expected_sha = str(manifest_checkpoint.get("sha256", ""))
        actual_sha = sha256_file(parent_checkpoint)
        if not expected_sha or expected_sha != actual_sha:
            raise ValueError(
                "Pretrain manifest was not extracted from the selected parent nnPU "
                f"checkpoint: manifest_sha256={expected_sha!r}, "
                f"checkpoint_sha256={actual_sha!r}."
            )
    else:
        manifest_checkpoint = dict(pretrain_manifest.get("checkpoint", {}))

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
    "safe_run_suffix",
    "sha256_file",
    "validate_finetune_contract",
]
