"""Checkpoint and trajectory-selection contracts for nnPU latent extraction."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch

from robosuite.discriminator.utils.robosuite_benchmark import (
    RobosuiteBenchmarkTrajectory,
)


@dataclass(frozen=True)
class CheckpointContract:
    """Feature and data-selection contract restored from an nnPU checkpoint."""

    checkpoint_path: Path
    checkpoint_sha256: str
    model_ckpt: str
    model_ckpt_sha256: Optional[str]
    normalizer_ckpt_sha256: Optional[str]
    in_dim: int
    feature_source: str
    transformer_layer: int
    use_chunk: bool
    unlabeled_fail_video_ids: object
    success_train_video_ids: object
    success_calib_video_ids: object
    seed: Optional[int]
    calibration_fraction: Optional[float]
    camera_to_view: Optional[dict[str, str]]
    proprio_indices: Optional[list[int]]
    encode_batch_size: Optional[int]


def require_cuda(device_value: str) -> torch.device:
    device = torch.device(str(device_value))
    if device.type != "cuda":
        raise ValueError(
            f"CPU execution is forbidden for discriminator extraction; got {device_value!r}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but torch.cuda.is_available() is False.")
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA device index {device.index} is unavailable; "
            f"visible device count is {torch.cuda.device_count()}."
        )
    torch.cuda.set_device(0 if device.index is None else device.index)
    return device


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_value(payload: Mapping[str, Any], key: str, default: Any = None) -> Any:
    if key in payload:
        return payload[key]
    provenance = payload.get("provenance")
    if isinstance(provenance, Mapping) and key in provenance:
        return provenance[key]
    return default


def load_checkpoint_contract(path_value: str, device: torch.device) -> CheckpointContract:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"nnPU checkpoint not found: {path}")

    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError(f"nnPU checkpoint must contain a mapping, got {type(payload).__name__}")

    required = ["model_ckpt", "in_dim", "feature_source", "transformer_layer"]
    missing = [name for name in required if _payload_value(payload, name) is None]
    if missing:
        raise KeyError(f"nnPU checkpoint is missing required fields: {missing}")

    feature_source = str(_payload_value(payload, "feature_source"))
    if feature_source != "transformer":
        raise ValueError(
            "Offline discriminator pretraining requires transformer latents; "
            f"checkpoint feature_source is {feature_source!r}."
        )
    transformer_layer = int(_payload_value(payload, "transformer_layer"))
    if transformer_layer != 1:
        raise ValueError(
            "Offline discriminator pretraining requires transformer layer 1; "
            f"checkpoint transformer_layer is {transformer_layer}."
        )

    raw_u_ids = _payload_value(payload, "unlabeled_fail_video_ids")
    if raw_u_ids is None:
        raise KeyError(
            "nnPU checkpoint has no unlabeled_fail_video_ids; the original U pool "
            "cannot be reconstructed safely."
        )

    raw_camera_map = _payload_value(payload, "camera_to_view")
    camera_map = None
    if raw_camera_map is not None:
        if not isinstance(raw_camera_map, Mapping):
            raise TypeError("checkpoint camera_to_view must be a mapping")
        camera_map = {str(key): str(value) for key, value in raw_camera_map.items()}

    raw_proprio = _payload_value(payload, "proprio_indices")
    raw_seed = _payload_value(payload, "seed")
    raw_fraction = _payload_value(payload, "calib_fraction")
    raw_batch_size = _payload_value(payload, "encode_batch_size")
    contract = CheckpointContract(
        checkpoint_path=path,
        checkpoint_sha256=sha256_file(path),
        model_ckpt=str(_payload_value(payload, "model_ckpt")),
        model_ckpt_sha256=(
            None
            if _payload_value(payload, "model_ckpt_sha256") is None
            else str(_payload_value(payload, "model_ckpt_sha256"))
        ),
        normalizer_ckpt_sha256=(
            None
            if _payload_value(payload, "normalizer_ckpt_sha256") is None
            else str(_payload_value(payload, "normalizer_ckpt_sha256"))
        ),
        in_dim=int(_payload_value(payload, "in_dim")),
        feature_source=feature_source,
        transformer_layer=transformer_layer,
        use_chunk=bool(_payload_value(payload, "use_chunk", False)),
        unlabeled_fail_video_ids=raw_u_ids,
        success_train_video_ids=_payload_value(payload, "success_train_video_ids"),
        success_calib_video_ids=_payload_value(payload, "success_calib_video_ids"),
        seed=None if raw_seed is None else int(raw_seed),
        calibration_fraction=None if raw_fraction is None else float(raw_fraction),
        camera_to_view=camera_map,
        proprio_indices=(
            None if raw_proprio is None else [int(value) for value in raw_proprio]
        ),
        encode_batch_size=None if raw_batch_size is None else int(raw_batch_size),
    )
    del payload
    torch.cuda.empty_cache()
    return contract


def resolve_model_checkpoint(path_value: str) -> Path:
    raw = Path(path_value).expanduser()
    if raw.is_absolute() and raw.is_file():
        return raw.resolve()
    repo_root = Path(__file__).resolve().parents[4]
    candidates = [Path.cwd() / raw, repo_root / raw]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    tried = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Dynamics checkpoint from nnPU metadata was not found: {path_value!r}. Tried: {tried}"
    )


def require_saved_normalizer(model_ckpt: Path) -> Path:
    """Reject DynEncoder's dataset-based normalizer rebuild path."""
    run_dirs = [model_ckpt.parent, model_ckpt.parent.parent]
    for child in model_ckpt.parent.iterdir():
        if child.is_dir() and (child / ".hydra" / "hydra.yaml").is_file():
            run_dirs.insert(0, child)
    candidates = [
        candidate
        for run_dir in run_dirs
        for candidate in (run_dir / "normalizer.pth", run_dir / ".hydra" / "normalizer.pth")
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.expanduser().resolve()
    raise FileNotFoundError(
        "A saved normalizer.pth is required for strict CUDA-only extraction; "
        "the CPU-capable dataset rebuild fallback is disabled. Tried: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def ids_for_task(raw_ids: object, task: str, *, field_name: str) -> Optional[list[str]]:
    if raw_ids is None:
        return None
    values: object
    if isinstance(raw_ids, Mapping):
        values = raw_ids.get(task)
        if values is None:
            return []
    else:
        values = raw_ids
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"checkpoint {field_name} must be a sequence or task mapping")
    result = [str(value) for value in values]
    if not isinstance(raw_ids, Mapping):
        prefix = f"{task}/"
        result = [video_id for video_id in result if video_id.startswith(prefix)]
    if len(result) != len(set(result)):
        raise ValueError(f"checkpoint {field_name} contains duplicate IDs for task {task!r}")
    return result


def split_success_trajectories(
    trajectories: Sequence[RobosuiteBenchmarkTrajectory],
    *,
    seed: int,
    calibration_fraction: float,
) -> tuple[list[RobosuiteBenchmarkTrajectory], list[RobosuiteBenchmarkTrajectory]]:
    """Mirror PUBCEBenchmarkDiscriminator's trajectory-level split exactly."""
    if not (0.0 < float(calibration_fraction) < 1.0):
        raise ValueError(
            f"calibration_fraction must be in (0, 1), got {calibration_fraction}"
        )
    if len(trajectories) < 2:
        raise RuntimeError(
            f"Need at least two success trajectories, found {len(trajectories)}."
        )
    rng = np.random.default_rng(int(seed))
    permutation = rng.permutation(len(trajectories))
    n_calib = int(round(float(calibration_fraction) * len(trajectories)))
    n_calib = max(1, min(len(trajectories) - 1, n_calib))
    calib_indices = set(permutation[:n_calib].tolist())
    train = [trajectory for index, trajectory in enumerate(trajectories) if index not in calib_indices]
    calib = [trajectory for index, trajectory in enumerate(trajectories) if index in calib_indices]
    return train, calib


def validate_saved_split_ids(
    trajectories: Sequence[RobosuiteBenchmarkTrajectory],
    expected_ids: Optional[Sequence[str]],
    *,
    field_name: str,
) -> None:
    if expected_ids is None:
        return
    actual = [str(trajectory.video_id) for trajectory in trajectories]
    expected = [str(video_id) for video_id in expected_ids]
    if actual == expected:
        return
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    order_only = not missing and not unexpected
    raise RuntimeError(
        f"Reconstructed {field_name} does not match checkpoint provenance. "
        f"missing={missing}, unexpected={unexpected}, order_only_mismatch={order_only}. "
        "Check --success-split, --success-cap, --seed, and --calibration-fraction."
    )


def select_trajectories_by_id(
    candidates: Sequence[RobosuiteBenchmarkTrajectory],
    video_ids: Sequence[str],
    *,
    field_name: str,
) -> list[RobosuiteBenchmarkTrajectory]:
    by_id = {str(trajectory.video_id): trajectory for trajectory in candidates}
    missing = [str(video_id) for video_id in video_ids if str(video_id) not in by_id]
    if missing:
        raise RuntimeError(
            f"Checkpoint {field_name} IDs are absent from the discovered success pool: "
            f"{missing}. Check --success-split and --success-cap."
        )
    if not video_ids:
        raise RuntimeError(f"Checkpoint {field_name} is empty for the requested task.")
    return [by_id[str(video_id)] for video_id in video_ids]


def select_unlabeled_trajectories(
    candidates: Sequence[RobosuiteBenchmarkTrajectory],
    checkpoint_ids: Sequence[str],
) -> list[RobosuiteBenchmarkTrajectory]:
    by_id = {str(trajectory.video_id): trajectory for trajectory in candidates}
    missing = [video_id for video_id in checkpoint_ids if video_id not in by_id]
    if missing:
        raise RuntimeError(
            "Checkpoint U trajectory IDs are absent from the discovered failure pool: "
            f"{missing}. Check --failure-split and --failure-cap."
        )
    if not checkpoint_ids:
        raise RuntimeError("Checkpoint U trajectory ID list is empty for the requested task.")
    return [by_id[video_id] for video_id in checkpoint_ids]


__all__ = [
    "CheckpointContract",
    "ids_for_task",
    "load_checkpoint_contract",
    "require_cuda",
    "require_saved_normalizer",
    "resolve_model_checkpoint",
    "select_trajectories_by_id",
    "select_unlabeled_trajectories",
    "sha256_file",
    "split_success_trajectories",
    "validate_saved_split_ids",
]
