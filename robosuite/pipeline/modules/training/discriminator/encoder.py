"""Finetune-specific frozen dynamics encoder contract."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from robosuite.pipeline.algorithms.discriminator.encoder import (
    SharedDynamicsEncoder,
    resolve_encoder_checkpoint,
)

from .contracts import sha256_file
from .trainer import require_cuda_device


def resolve_saved_normalizer_checkpoint(model_checkpoint: str | Path) -> Path:
    """Locate the saved normalizer used by ``DynEncoder`` without rebuilding it."""
    checkpoint = Path(model_checkpoint).expanduser().resolve()
    run_dirs = [checkpoint.parent, checkpoint.parent.parent]
    for child in checkpoint.parent.iterdir():
        if child.is_dir() and (child / ".hydra" / "hydra.yaml").is_file():
            run_dirs.insert(0, child)
    candidates = [
        candidate
        for run_dir in run_dirs
        for candidate in (
            run_dir / "normalizer.pth",
            run_dir / ".hydra" / "normalizer.pth",
        )
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.expanduser().resolve()
    raise FileNotFoundError(
        "A saved normalizer.pth is required for CUDA-only discriminator finetuning. "
        "The dataset rebuild fallback is disabled. Tried: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def resolve_saved_dynamics_config(model_checkpoint: str | Path) -> Path:
    """Locate the Hydra config required to reconstruct a dynamics checkpoint."""
    checkpoint = Path(model_checkpoint).expanduser().resolve()
    run_dirs = [checkpoint.parent, checkpoint.parent.parent]
    for child in checkpoint.parent.iterdir():
        if child.is_dir() and (child / ".hydra" / "hydra.yaml").is_file():
            run_dirs.insert(0, child)
    candidates = [
        candidate
        for run_dir in run_dirs
        for candidate in (
            run_dir / "hydra.yaml",
            run_dir / ".hydra" / "config.yaml",
            run_dir / ".hydra" / "hydra.yaml",
        )
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.expanduser().resolve()
    raise FileNotFoundError(
        "A saved Hydra config is required to reconstruct the dynamics encoder. Tried: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


class FinetuneDynamicsEncoder(SharedDynamicsEncoder):
    """CUDA-only encoder with immutable offline finetune provenance."""

    def __init__(
        self,
        nnpu_ckpt_path: str | Path,
        *,
        encoder_ckpt: str | Path | None = None,
        device: str | torch.device = "cuda",
        camera_to_view: dict[str, str] | None = None,
        proprio_indices: Sequence[int] | None = None,
    ) -> None:
        cuda_device = require_cuda_device(device)
        checkpoint = Path(nnpu_ckpt_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"nnPU checkpoint not found: {checkpoint}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise TypeError(f"nnPU checkpoint payload must be a dict: {checkpoint}")

        checkpoint_camera_map = payload.get("camera_to_view") or {}
        effective_camera_map = dict(
            checkpoint_camera_map if camera_to_view is None else camera_to_view
        )
        checkpoint_proprio = payload.get("proprio_indices")
        selected_proprio = (
            checkpoint_proprio if proprio_indices is None else proprio_indices
        )
        self.proprio_indices = (
            None
            if selected_proprio is None
            else [int(index) for index in selected_proprio]
        )
        self.use_chunk = bool(payload.get("use_chunk", False))

        raw_model_checkpoint = str(encoder_ckpt or payload["model_ckpt"])
        model_checkpoint = resolve_encoder_checkpoint(
            raw_model_checkpoint,
            hint_root=checkpoint.parent,
        )
        expected_model_sha = payload.get("model_ckpt_sha256")
        if expected_model_sha is not None:
            actual_model_sha = sha256_file(model_checkpoint)
            if str(expected_model_sha) != actual_model_sha:
                raise ValueError(
                    "Resolved dynamics checkpoint does not match nnPU provenance: "
                    f"expected_sha256={expected_model_sha}, "
                    f"actual_sha256={actual_model_sha}."
                )

        normalizer_checkpoint = resolve_saved_normalizer_checkpoint(model_checkpoint)
        expected_normalizer_sha = payload.get("normalizer_ckpt_sha256")
        if expected_normalizer_sha is not None:
            actual_normalizer_sha = sha256_file(normalizer_checkpoint)
            if str(expected_normalizer_sha) != actual_normalizer_sha:
                raise ValueError(
                    "Resolved normalizer does not match nnPU provenance: "
                    f"expected_sha256={expected_normalizer_sha}, "
                    f"actual_sha256={actual_normalizer_sha}."
                )

        super().__init__(
            nnpu_ckpt_path=checkpoint,
            encoder_ckpt=model_checkpoint,
            device=cuda_device,
            camera_to_view=effective_camera_map,
        )
        if self.device.type != "cuda":
            raise RuntimeError(
                f"Dynamics encoder initialized on non-CUDA device {self.device}."
            )
        self.normalizer_checkpoint = str(normalizer_checkpoint)
        self.proprio_input_dim = int(self.inner_encoder.model.proprio_encoder.in_chans)

    def prepare_proprio(self, states: np.ndarray) -> np.ndarray:
        """Validate already-extracted policy proprio against the encoder input."""
        array = np.asarray(states, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError(f"states must be (B, D), got {array.shape}.")
        current_dim = int(array.shape[1])
        if current_dim != self.proprio_input_dim:
            raise ValueError(
                "Collected policy proprio is already extracted and must exactly match "
                f"the frozen encoder input dim {self.proprio_input_dim}; got {current_dim}."
            )
        return np.ascontiguousarray(array, dtype=np.float32)


__all__ = [
    "FinetuneDynamicsEncoder",
    "resolve_saved_dynamics_config",
    "resolve_saved_normalizer_checkpoint",
]
