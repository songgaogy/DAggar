"""CUDA latent encoding for policy-only offline segments."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch
from tqdm import tqdm

from .episodes import PolicySegment
from .pools import DiscriminatorPools, LatentTrajectory, pool_stats


def build_action_windows(
    actions: np.ndarray,
    *,
    horizon: int,
    use_chunk: bool,
    target_dim: int | None = None,
) -> np.ndarray:
    """Build boundary-local action inputs with repeat-tail or legacy zero padding."""
    action = np.asarray(actions, dtype=np.float32)
    if action.ndim != 2 or int(action.shape[0]) <= 0:
        raise ValueError(f"actions must be a non-empty (T, A) array, got {action.shape}.")
    if int(horizon) <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}.")
    length, action_dim = action.shape
    windows = np.zeros((length, int(horizon), action_dim), dtype=np.float32)
    if use_chunk:
        for step in range(length):
            end = min(length, step + int(horizon))
            chunk = action[step:end]
            windows[step, : int(chunk.shape[0])] = chunk
            if int(chunk.shape[0]) < int(horizon):
                windows[step, int(chunk.shape[0]) :] = chunk[-1]
    else:
        windows[:, 0] = action
    flat = windows.reshape(length, -1)
    if target_dim is None or int(target_dim) == int(flat.shape[1]):
        return np.ascontiguousarray(flat, dtype=np.float32)
    if int(target_dim) < int(flat.shape[1]):
        return np.ascontiguousarray(flat[:, : int(target_dim)], dtype=np.float32)
    padding = np.zeros(
        (length, int(target_dim) - int(flat.shape[1])), dtype=np.float32
    )
    return np.ascontiguousarray(np.concatenate([flat, padding], axis=1), dtype=np.float32)


def _image_batch(
    observations: Mapping[str, Any],
    *,
    camera_names: Sequence[str],
    lo: int,
    hi: int,
    device: torch.device,
) -> torch.Tensor:
    arrays = [np.asarray(observations[name][lo:hi]) for name in camera_names]
    stacked = np.stack(arrays, axis=1)
    if stacked.ndim != 5 or int(stacked.shape[-1]) != 3:
        raise ValueError(
            f"camera observations must form (B, V, H, W, 3), got {stacked.shape}."
        )
    tensor = torch.from_numpy(np.ascontiguousarray(stacked)).to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    tensor = tensor.permute(0, 1, 4, 2, 3).contiguous()
    if np.issubdtype(stacked.dtype, np.integer):
        tensor.mul_(1.0 / 255.0)
    elif float(tensor.detach().amax().item()) > 1.5:
        tensor.mul_(1.0 / 255.0)
    return tensor


@torch.inference_mode()
def encode_policy_segments(
    segments: Sequence[PolicySegment],
    *,
    encoder: Any,
    camera_names: Sequence[str],
    batch_size: int,
) -> DiscriminatorPools:
    """Encode each policy segment independently with a frozen CUDA encoder."""
    device = torch.device(encoder.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"Policy-segment encoding requires CUDA, got {device}.")
    if not hasattr(encoder, "use_chunk"):
        raise AttributeError("FinetuneDynamicsEncoder is missing use_chunk provenance.")
    if str(encoder.feature_source) != "transformer" or int(encoder.transformer_layer) != 1:
        raise ValueError(
            "Policy-segment encoding requires checkpoint transformer layer-1 latents."
        )
    if any(parameter.requires_grad for parameter in encoder.inner_encoder.model.parameters()):
        raise RuntimeError("Dynamics encoder must be frozen before discriminator finetuning.")
    if int(batch_size) <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    encoder.bind_policy_cameras(list(camera_names))

    horizon = int(encoder.inner_encoder.frameskip)
    target_action_dim = int(encoder.inner_encoder.action_input_dim)
    pools = DiscriminatorPools()
    for segment in tqdm(
        segments,
        desc="[disc_finetune][encode]",
        unit="seg",
        leave=True,
    ):
        episode = segment.episode
        actions = np.asarray(episode["executed_action"], dtype=np.float32)[
            segment.lo : segment.hi
        ]
        expected_action_dim = int(encoder.inner_encoder.action_dim_per_step)
        if int(actions.shape[1]) != expected_action_dim:
            raise ValueError(
                f"Segment {segment.identifier} action dim {actions.shape[1]} does not match "
                f"the frozen encoder action dim {expected_action_dim}."
            )
        action_windows = build_action_windows(
            actions,
            horizon=horizon,
            use_chunk=bool(encoder.use_chunk),
            target_dim=target_action_dim,
        )
        features_cuda: list[torch.Tensor] = []
        for start in range(0, segment.num_frames, int(batch_size)):
            end = min(segment.num_frames, start + int(batch_size))
            absolute_lo = segment.lo + start
            absolute_hi = segment.lo + end
            images = _image_batch(
                episode["obs"],
                camera_names=camera_names,
                lo=absolute_lo,
                hi=absolute_hi,
                device=device,
            )
            proprio = torch.from_numpy(
                encoder.prepare_proprio(
                    np.asarray(episode["obs"]["state"])[absolute_lo:absolute_hi]
                )
            ).to(device=device, non_blocking=True)
            action_tensor = torch.from_numpy(action_windows[start:end]).to(
                device=device, non_blocking=True
            )
            feature = encoder.encode_chunk(
                image_obs_raw=images,
                proprio_raw=proprio,
                action_chunk=action_tensor,
            )
            if feature.device.type != "cuda" or feature.ndim != 2:
                raise RuntimeError(
                    f"Encoder returned invalid features: device={feature.device}, "
                    f"shape={tuple(feature.shape)}."
                )
            features_cuda.append(feature.to(dtype=torch.float32))
        latent = torch.cat(features_cuda, dim=0)
        latent_host = latent.contiguous().to(device="cpu", dtype=torch.float32)
        item = LatentTrajectory(
            features=latent_host,
            pool=segment.pool,
            source="online",
            identifier=segment.identifier,
            metadata={
                "source_episode_index": segment.source_episode_index,
                "segment_index": segment.segment_index,
                "frame_start": segment.lo,
                "frame_end": segment.hi,
                "terminal_reason": segment.terminal_reason,
                "ended_by": segment.ended_by,
                "use_chunk": bool(encoder.use_chunk),
            },
        )
        getattr(pools, segment.pool).append(item)
        del latent, features_cuda
    pools.stats = pool_stats(pools)
    return pools


__all__ = ["build_action_windows", "encode_policy_segments"]
