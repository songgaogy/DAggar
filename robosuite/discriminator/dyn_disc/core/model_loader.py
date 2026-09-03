"""Strict loader for RPT representation checkpoints."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from omegaconf import DictConfig, OmegaConf

from robosuite.discriminator.dyn_disc.models.rpt import RPTModel


CHECKPOINT_VERSION = 1


def _find_run_file(checkpoint: Path, filename: str) -> Path:
    for directory in (checkpoint.parent, checkpoint.parent.parent):
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not locate {filename!r} next to RPT checkpoint {checkpoint}"
    )


def checkpoint_fingerprint(path: str | Path) -> str:
    """Return a stable SHA-256 fingerprint for checkpoint identity validation."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _architecture(cfg: DictConfig, payload: Dict[str, Any]) -> Dict[str, Any]:
    stored = payload.get("architecture")
    if stored is not None:
        architecture = dict(stored)
    else:
        model_cfg = OmegaConf.select(cfg, "model", default={})
        architecture = dict(OmegaConf.to_container(model_cfg, resolve=True) or {})
        architecture.pop("_target_", None)

    defaults = {
        "view_names": list(OmegaConf.select(cfg, "env.view_names")),
        "visual_dim": 768,
        "proprio_dim": int(OmegaConf.select(cfg, "env.proprio_dim", default=14)),
        "action_dim": int(OmegaConf.select(cfg, "env.action_dim", default=7)),
        "context_length": int(OmegaConf.select(cfg, "context_length", default=8)),
        "hidden_dim": 192,
        "depth": 4,
        "heads": 4,
        "mlp_dim": 384,
        "dropout": 0.0,
        "slot_position_embedding": "learned",
        "mask_ratio_min": 0.7,
        "mask_ratio_max": 0.9,
    }
    if "num_heads" in architecture and "heads" not in architecture:
        architecture["heads"] = architecture.pop("num_heads")
    architecture.pop("token_order", None)
    for key, value in defaults.items():
        architecture.setdefault(key, value)
    allowed = set(defaults)
    unknown = sorted(set(architecture) - allowed)
    if unknown:
        raise ValueError(f"Unsupported RPT architecture fields: {unknown}")
    return architecture


def load_rpt_checkpoint(
    model_ckpt: str | Path,
    device: str | torch.device = "cuda",
) -> Tuple[RPTModel, DictConfig, Dict[str, Any]]:
    """Load an RPT model, rejecting legacy dynamics/TACO checkpoints."""
    requested_device = torch.device(device)
    if requested_device.type != "cuda":
        raise ValueError(f"RPT inference is CUDA-only; got device={device!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for RPT inference")

    checkpoint = Path(model_ckpt).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"RPT checkpoint not found: {checkpoint}")
    cfg_path = _find_run_file(checkpoint, "hydra.yaml")
    cfg = OmegaConf.load(cfg_path)
    if str(OmegaConf.select(cfg, "pretraining_method", default="")) != "rpt":
        raise ValueError(f"Config is not RPT-only: {cfg_path}")

    payload = torch.load(checkpoint, map_location=requested_device, weights_only=False)
    if not isinstance(payload, dict) or payload.get("pretraining_method") != "rpt":
        raise ValueError(
            "Unsupported representation checkpoint: expected pretraining_method='rpt'"
        )
    version = int(payload.get("checkpoint_version", -1))
    if version != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported RPT checkpoint_version={version}; expected {CHECKPOINT_VERSION}"
        )
    if "rpt_model" not in payload:
        raise ValueError("Invalid RPT checkpoint: missing 'rpt_model' state dict")

    architecture = _architecture(cfg, payload)
    model = RPTModel(**architecture).to(requested_device)
    model.load_state_dict(payload["rpt_model"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    metadata = {
        "checkpoint_path": str(checkpoint),
        "checkpoint_fingerprint": checkpoint_fingerprint(checkpoint),
        "cache_manifest_fingerprint": payload.get("cache_manifest_fingerprint"),
        "architecture": architecture,
        "view_names": list(OmegaConf.select(cfg, "env.view_names")),
        "epoch": int(payload.get("epoch", 0)),
        "global_step": int(payload.get("global_step", 0)),
    }
    return model, cfg, metadata


# Deliberately retain this name as a strict RPT alias for callers outside this package.
load_model = load_rpt_checkpoint


__all__ = ["CHECKPOINT_VERSION", "checkpoint_fingerprint", "load_model", "load_rpt_checkpoint"]
