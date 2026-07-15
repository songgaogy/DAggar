"""CUDA encoding and atomic shard publication for nnPU pretraining latents."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from robosuite.discriminator.dyn_disc.adapters.single_bank import DynBenchmarkDiscriminator
from robosuite.discriminator.utils.robosuite_benchmark import RobosuiteBenchmarkTrajectory

from .pretrain_extract_contract import CheckpointContract


SCHEMA_VERSION = 1


def _prepare_trajectory(
    adapter: DynBenchmarkDiscriminator,
    trajectory: RobosuiteBenchmarkTrajectory,
    *,
    frame_end: Optional[int],
) -> dict[str, Any]:
    """Load source arrays and exactly reproduce the parent feature contract."""
    view_names = list(adapter.encoder.view_names)
    cameras = [adapter._resolve_camera(view) for view in view_names]
    images_by_camera = trajectory.load_images(cameras=cameras)
    states = np.asarray(trajectory.load_states(), dtype=np.float32)
    actions = np.asarray(trajectory.load_actions(), dtype=np.float32)
    lengths = [int(states.shape[0]), int(actions.shape[0])]
    lengths.extend(int(images_by_camera[camera].shape[0]) for camera in cameras)
    full_length = int(min(lengths))
    encoded_length = full_length if frame_end is None else min(full_length, int(frame_end))
    if encoded_length <= 0:
        raise ValueError(f"Trajectory has no encodable frames: {trajectory.describe()}")

    target_proprio_dim = None
    try:
        target_proprio_dim = int(adapter.encoder.model.proprio_encoder.in_chans)
    except Exception:
        pass
    proprio = adapter._slice_proprio(
        states[:encoded_length],
        target_dim=target_proprio_dim,
        task_name=str(trajectory.task_name),
    )
    # The parent prepared complete success trajectories before truncating them
    # to pre-done frames. Preserve that ordering exactly.
    action = adapter._prepare_actions(actions[:full_length], t_len=full_length)[:encoded_length]

    images_by_view: dict[str, np.ndarray] = {}
    for view, camera in zip(view_names, cameras):
        images = np.asarray(images_by_camera[camera][:encoded_length])
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(
                f"Camera {camera!r} must provide (T, H, W, 3), got {images.shape}."
            )
        images_by_view[view] = images
    return {
        "images_by_view": images_by_view,
        "proprio": np.ascontiguousarray(proprio, dtype=np.float32),
        "action": np.ascontiguousarray(action, dtype=np.float32),
        "t_len": int(encoded_length),
    }


def effective_proprio_indices(
    adapter: DynBenchmarkDiscriminator,
    task_name: str,
) -> Optional[list[int]]:
    if adapter.proprio_indices is not None:
        return [int(value) for value in adapter.proprio_indices.tolist()]
    task_config = adapter.proprio_map.get(str(task_name), {})
    indices = task_config.get("indices") if isinstance(task_config, Mapping) else None
    if not indices:
        return None
    return [int(value) for value in indices]


@torch.inference_mode()
def _encode_cuda(
    adapter: DynBenchmarkDiscriminator,
    prepared: Mapping[str, Any],
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    t_len = int(prepared["t_len"])
    target_size = int(adapter.encoder.original_img_size)
    features: list[torch.Tensor] = []
    for start in range(0, t_len, int(batch_size)):
        end = min(start + int(batch_size), t_len)
        image_batch: dict[str, torch.Tensor] = {}
        for view, images in prepared["images_by_view"].items():
            array = np.ascontiguousarray(images[start:end])
            scale = 1.0 / 255.0 if float(np.max(array)) > 1.5 else 1.0
            tensor = torch.from_numpy(array).to(device=device, dtype=torch.float32)
            tensor = tensor.permute(0, 3, 1, 2).contiguous()
            if scale != 1.0:
                tensor.mul_(scale)
            if tuple(tensor.shape[-2:]) != (target_size, target_size):
                tensor = F.interpolate(
                    tensor,
                    size=(target_size, target_size),
                    mode="bilinear",
                    align_corners=False,
                )
            image_batch[str(view)] = tensor

        proprio = torch.from_numpy(prepared["proprio"][start:end]).to(device=device)
        action = torch.from_numpy(prepared["action"][start:end]).to(device=device)
        feature = adapter.encoder.encode_batch(image_batch, proprio, actions=action)
        if feature.device.type != "cuda":
            raise RuntimeError(f"Encoder returned a non-CUDA tensor on {feature.device}.")
        if feature.ndim != 2:
            raise RuntimeError(f"Expected a (T, D) latent, got {tuple(feature.shape)}")
        features.append(feature.to(dtype=torch.float32))

    latent = torch.cat(features, dim=0)
    if latent.device.type != "cuda" or latent.dtype != torch.float32:
        raise RuntimeError(
            f"Extraction contract violation: latent device={latent.device}, dtype={latent.dtype}."
        )
    return latent


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w+b", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def atomic_json_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _trajectory_provenance(
    trajectory: RobosuiteBenchmarkTrajectory,
    *,
    pool: str,
    encoded_frames: int,
    contract: CheckpointContract,
    model_ckpt: Path,
) -> dict[str, Any]:
    return {
        "task": str(trajectory.task_name),
        "video_id": str(trajectory.video_id),
        "pool": str(pool),
        "source_split": str(trajectory.split),
        "source_file": str(Path(trajectory.file_path).resolve()),
        "source_demo_path": str(trajectory.demo_path),
        "source_demo_key": str(trajectory.source_demo_key),
        "original_num_frames": int(trajectory.num_frames),
        "encoded_frame_start": 0,
        "encoded_frame_end": int(encoded_frames),
        "nnpu_checkpoint": str(contract.checkpoint_path),
        "dynamics_checkpoint": str(model_ckpt),
    }


def _safe_shard_name(index: int, video_id: str) -> str:
    digest = hashlib.sha1(video_id.encode("utf-8")).hexdigest()[:12]
    return f"{index:05d}-{digest}.pt"


def write_pool(
    *,
    adapter: DynBenchmarkDiscriminator,
    trajectories: Sequence[RobosuiteBenchmarkTrajectory],
    pool: str,
    success_prefix_only: bool,
    staging_dir: Path,
    contract: CheckpointContract,
    model_ckpt: Path,
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for index, trajectory in enumerate(trajectories):
        frame_end = None
        if success_prefix_only:
            frame_end = int(trajectory.prefix_frames_before_done())
            if frame_end <= 0:
                raise ValueError(
                    f"Success trajectory has no pre-success frames: {trajectory.describe()}"
                )
        prepared = _prepare_trajectory(adapter, trajectory, frame_end=frame_end)
        latent = _encode_cuda(adapter, prepared, device=device, batch_size=batch_size)
        if int(latent.shape[1]) != int(contract.in_dim):
            raise RuntimeError(
                f"Latent dim mismatch for {trajectory.video_id}: extracted "
                f"{int(latent.shape[1])}, checkpoint head expects {contract.in_dim}."
            )

        provenance = _trajectory_provenance(
            trajectory,
            pool=pool,
            encoded_frames=int(latent.shape[0]),
            contract=contract,
            model_ckpt=model_ckpt,
        )
        relative_path = Path(pool) / _safe_shard_name(index, str(trajectory.video_id))
        latent_host = latent.contiguous().to(device="cpu", dtype=torch.float32)
        frame_indices = torch.from_numpy(np.arange(int(latent.shape[0]), dtype=np.int64))
        _atomic_torch_save(
            {
                "schema_version": SCHEMA_VERSION,
                "latent": latent_host,
                "frame_indices": frame_indices,
                "provenance": provenance,
            },
            staging_dir / relative_path,
        )
        entries.append(
            {
                "path": relative_path.as_posix(),
                "video_id": str(trajectory.video_id),
                "num_frames": int(latent.shape[0]),
                "latent_dim": int(latent.shape[1]),
                "provenance": provenance,
            }
        )
        print(
            f"[extract_data] {pool}: {index + 1}/{len(trajectories)} "
            f"video_id={trajectory.video_id} frames={int(latent.shape[0])}",
            flush=True,
        )
        del latent, latent_host, frame_indices
    return entries


def path_exists(path: Path) -> bool:
    return os.path.lexists(str(path))


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def publish_staging(staging_dir: Path, output_dir: Path, *, overwrite: bool) -> None:
    if not path_exists(output_dir):
        os.replace(staging_dir, output_dir)
        return
    if not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_dir}. Pass --overwrite to replace it."
        )
    backup = output_dir.with_name(f".{output_dir.name}.backup-{uuid.uuid4().hex}")
    os.replace(output_dir, backup)
    try:
        os.replace(staging_dir, output_dir)
    except Exception:
        os.replace(backup, output_dir)
        raise
    _remove_path(backup)


def split_statistics(entries: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    dims = {int(entry["latent_dim"]) for entry in entries}
    if len(dims) > 1:
        raise RuntimeError(f"Latent dimensions differ within a split: {sorted(dims)}")
    return {
        "num_trajectories": int(len(entries)),
        "num_frames": int(sum(int(entry["num_frames"]) for entry in entries)),
        "latent_dim": 0 if not dims else int(next(iter(dims))),
    }


__all__ = [
    "SCHEMA_VERSION",
    "atomic_json_save",
    "effective_proprio_indices",
    "path_exists",
    "publish_staging",
    "split_statistics",
    "write_pool",
]
