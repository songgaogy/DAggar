"""Checkpoint provenance checks for finetuned discriminator visualization."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


def _sha256_file(path_value: str | Path) -> str:
    path = Path(path_value).expanduser().resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class FinetunedVisualizationContract:
    """Validated checkpoint payload and effective encoder selectors."""

    payload: dict[str, Any]
    camera_to_view: dict[str, str] | None
    proprio_indices: list[int] | None


def load_finetuned_visualization_contract(
    checkpoint: str | Path,
    *,
    model_checkpoint: str | Path,
    feature_source: str,
    transformer_layer: int,
    camera_to_view: Mapping[str, str] | None,
    proprio_indices: Sequence[int] | None,
) -> FinetunedVisualizationContract:
    """Load a finetuned checkpoint and pin its frozen latent contract."""
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Finetuned nnPU checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "pu_bce_detector" not in payload:
        raise KeyError(f"{path} is not a pu_bce_head_finetuned.pth checkpoint.")
    if not bool(payload.get("finetuned_offline", False)):
        raise ValueError(
            "The standalone finetuned visualizer rejects parent nnPU checkpoints."
        )
    if str(payload.get("feature_source")) != str(feature_source):
        raise ValueError(
            "Visualization feature_source differs from the loaded checkpoint: "
            f"cli={feature_source!r}, checkpoint={payload.get('feature_source')!r}."
        )
    if int(payload.get("transformer_layer", -1)) != int(transformer_layer):
        raise ValueError(
            "Visualization transformer layer differs from the loaded checkpoint: "
            f"cli={transformer_layer}, checkpoint={payload.get('transformer_layer')}."
        )

    checkpoint_camera_raw = payload.get("camera_to_view")
    checkpoint_camera = (
        None
        if checkpoint_camera_raw is None
        else {str(key): str(value) for key, value in dict(checkpoint_camera_raw).items()}
    )
    requested_camera = (
        None
        if camera_to_view is None
        else {str(key): str(value) for key, value in camera_to_view.items()}
    )
    if (
        checkpoint_camera is not None
        and requested_camera is not None
        and checkpoint_camera != requested_camera
    ):
        raise ValueError(
            "Visualization camera_to_view differs from the loaded checkpoint: "
            f"cli={requested_camera}, checkpoint={checkpoint_camera}."
        )
    effective_camera = requested_camera if requested_camera is not None else checkpoint_camera

    checkpoint_proprio_raw = payload.get("proprio_indices")
    checkpoint_proprio = (
        None
        if checkpoint_proprio_raw is None
        else [int(value) for value in checkpoint_proprio_raw]
    )
    requested_proprio = (
        None if proprio_indices is None else [int(value) for value in proprio_indices]
    )
    if (
        checkpoint_proprio is not None
        and requested_proprio is not None
        and checkpoint_proprio != requested_proprio
    ):
        raise ValueError(
            "Visualization proprio_indices differ from the loaded checkpoint: "
            f"cli={requested_proprio}, checkpoint={checkpoint_proprio}."
        )
    effective_proprio = (
        requested_proprio if requested_proprio is not None else checkpoint_proprio
    )

    expected_model_sha = payload.get("model_ckpt_sha256")
    if expected_model_sha is None:
        finetune_data = payload.get("finetune_data", {})
        if isinstance(finetune_data, Mapping):
            manifest = finetune_data.get("pretrain_manifest", {})
            if isinstance(manifest, Mapping):
                manifest_checkpoint = manifest.get("checkpoint", {})
                if isinstance(manifest_checkpoint, Mapping):
                    expected_model_sha = manifest_checkpoint.get("model_ckpt_sha256")
    if expected_model_sha is None:
        raise KeyError(
            "Finetuned checkpoint is missing dynamics checkpoint SHA-256 provenance."
        )
    actual_model_sha = _sha256_file(model_checkpoint)
    if str(expected_model_sha) != actual_model_sha:
        raise ValueError(
            "Visualization dynamics checkpoint does not match finetuned latent provenance: "
            f"expected_sha256={expected_model_sha}, actual_sha256={actual_model_sha}."
        )
    if payload.get("normalizer_ckpt_sha256") is None:
        raise KeyError("Finetuned checkpoint is missing normalizer SHA-256 provenance.")

    return FinetunedVisualizationContract(
        payload=payload,
        camera_to_view=effective_camera,
        proprio_indices=effective_proprio,
    )


def validate_runtime_normalizer(
    discriminator: Any,
    contract: FinetunedVisualizationContract,
) -> None:
    """Ensure the runtime encoder uses the normalizer pinned by finetuning."""
    normalizer_checkpoint = discriminator.encoder.normalizer_checkpoint
    if normalizer_checkpoint is None:
        raise RuntimeError("Visualizer dynamics encoder did not load a saved normalizer.")
    expected_sha = str(contract.payload["normalizer_ckpt_sha256"])
    actual_sha = _sha256_file(normalizer_checkpoint)
    if expected_sha != actual_sha:
        raise ValueError(
            "Visualization normalizer does not match finetuned latent provenance: "
            f"expected_sha256={expected_sha}, actual_sha256={actual_sha}."
        )


__all__ = [
    "FinetunedVisualizationContract",
    "load_finetuned_visualization_contract",
    "validate_runtime_normalizer",
]
