"""CUDA latent encoding for policy-only offline segments."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch
from tqdm import tqdm

from .episodes import GTNegativeWindow, PolicySegment
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
            source="offline",
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


@torch.inference_mode()
def encode_gt_negative_windows(
    windows: Sequence[GTNegativeWindow],
    *,
    encoder: Any,
    camera_names: Sequence[str],
    batch_size: int,
) -> list[LatentTrajectory]:
    """Encode intervention windows with event-local ``policy_action`` chunks."""
    device = torch.device(encoder.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"GT-negative window encoding requires CUDA, got {device}.")
    if not hasattr(encoder, "use_chunk"):
        raise AttributeError("FinetuneDynamicsEncoder is missing use_chunk provenance.")
    if str(encoder.feature_source) != "transformer" or int(encoder.transformer_layer) != 1:
        raise ValueError(
            "GT-negative encoding requires checkpoint transformer layer-1 latents."
        )
    if any(parameter.requires_grad for parameter in encoder.inner_encoder.model.parameters()):
        raise RuntimeError("Dynamics encoder must be frozen before discriminator finetuning.")
    if int(batch_size) <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    encoder.bind_policy_cameras(list(camera_names))

    horizon = int(encoder.inner_encoder.frameskip)
    target_action_dim = int(encoder.inner_encoder.action_input_dim)
    expected_action_dim = int(encoder.inner_encoder.action_dim_per_step)
    trajectories: list[LatentTrajectory] = []
    for window in tqdm(
        windows,
        desc="[disc_finetune][encode_gt_negative]",
        unit="event",
        leave=True,
    ):
        if window.hi <= window.lo or window.num_frames <= 0:
            raise ValueError(f"GT-negative window {window.identifier} is empty.")
        if any(index < window.lo or index >= window.hi for index in window.frame_indices):
            raise ValueError(
                f"GT-negative window {window.identifier} has frame indices outside "
                f"[{window.lo}, {window.hi})."
            )
        episode = window.episode
        policy_actions = np.asarray(episode["policy_action"], dtype=np.float32)
        if policy_actions.ndim != 2 or int(policy_actions.shape[1]) != expected_action_dim:
            raise ValueError(
                f"Window {window.identifier} policy_action shape {policy_actions.shape} "
                f"does not match (*, {expected_action_dim})."
            )
        actions = policy_actions[window.lo : window.hi]
        if not bool(np.isfinite(actions).all()):
            raise ValueError(
                f"Window {window.identifier} policy_action contains non-finite values."
            )
        action_windows = build_action_windows(
            actions,
            horizon=horizon,
            use_chunk=bool(encoder.use_chunk),
            target_dim=target_action_dim,
        )
        features_cuda: list[torch.Tensor] = []
        window_length = int(window.hi - window.lo)
        for start in range(0, window_length, int(batch_size)):
            end = min(window_length, start + int(batch_size))
            absolute_lo = window.lo + start
            absolute_hi = window.lo + end
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
        selected_offsets = torch.tensor(
            [index - window.lo for index in window.frame_indices],
            dtype=torch.long,
            device=device,
        )
        latent = latent.index_select(0, selected_offsets)
        latent_host = latent.contiguous().to(device="cpu", dtype=torch.float32)
        trajectories.append(
            LatentTrajectory(
                features=latent_host,
                pool="offline_gt_negative",
                source="offline",
                identifier=window.identifier,
                metadata={
                    "source_episode_index": window.source_episode_index,
                    "gt_negative_kind": window.kind,
                    "intervention_event_index": window.event_index,
                    "intervention_start": window.intervention_lo,
                    "intervention_end": window.intervention_hi,
                    "window_start": window.lo,
                    "window_end": window.hi,
                    "frame_indices": list(window.frame_indices),
                    "pre_frames": window.num_pre_frames,
                    "post_frames": window.num_post_frames,
                    "observation_source": "policy_prefix_and_human_intervention",
                    "action_source": "policy_action",
                    "post_boundary": "intervention_block",
                    "chunk_boundary": "continuous_window",
                    "tail_padding": "repeat_tail",
                    "stored_gt_fail_ignored": True,
                    "theory_deviation": (
                        "Human-control observations mean this pool is not strictly "
                        "a subset of offline policy-only unlabeled data."
                    ),
                    "use_chunk": bool(encoder.use_chunk),
                },
            )
        )
        del latent, features_cuda, selected_offsets
    return trajectories


def exclude_gt_frames_from_unlabeled(
    trajectories: Sequence[LatentTrajectory],
    windows: Sequence[GTNegativeWindow],
) -> list[LatentTrajectory]:
    """Remove selected pre-onset frames from offline-U latent trajectories."""
    selected_keys = {
        (int(window.source_episode_index), int(frame_index))
        for window in windows
        for frame_index in window.frame_indices
    }
    reserved: list[LatentTrajectory] = []
    for trajectory in trajectories:
        if trajectory.features.device.type != "cuda":
            raise ValueError(
                "Reserved-U feature filtering requires CUDA tensors; "
                f"got {trajectory.features.device}."
            )
        metadata = trajectory.metadata
        required = ("source_episode_index", "frame_start", "frame_end")
        missing = [name for name in required if name not in metadata]
        if missing:
            raise KeyError(
                f"Offline-U trajectory {trajectory.identifier} is missing metadata {missing}."
            )
        episode_index = int(metadata["source_episode_index"])
        frame_start = int(metadata["frame_start"])
        frame_end = int(metadata["frame_end"])
        if frame_end - frame_start != int(trajectory.features.shape[0]):
            raise ValueError(
                f"Offline-U trajectory {trajectory.identifier} frame range "
                f"[{frame_start}, {frame_end}) does not match feature length "
                f"{trajectory.features.shape[0]}."
            )
        kept_offsets = [
            offset
            for offset in range(frame_end - frame_start)
            if (episode_index, frame_start + offset) not in selected_keys
        ]
        run_index = 0
        cursor = 0
        while cursor < len(kept_offsets):
            run_start = kept_offsets[cursor]
            run_end = run_start + 1
            cursor += 1
            while cursor < len(kept_offsets) and kept_offsets[cursor] == run_end:
                run_end += 1
                cursor += 1
            absolute_start = frame_start + run_start
            absolute_end = frame_start + run_end
            run_metadata = dict(metadata)
            run_metadata.update(
                {
                    "frame_start": int(absolute_start),
                    "frame_end": int(absolute_end),
                    "gt_negative_frames_excluded": True,
                    "parent_identifier": trajectory.identifier,
                }
            )
            reserved.append(
                LatentTrajectory(
                    features=trajectory.features[run_start:run_end].contiguous(),
                    pool="offline_unlabeled_reserved",
                    source=trajectory.source,
                    identifier=f"{trajectory.identifier}-reserved-{run_index:03d}",
                    metadata=run_metadata,
                )
            )
            run_index += 1
    return reserved


__all__ = [
    "build_action_windows",
    "encode_gt_negative_windows",
    "encode_policy_segments",
    "exclude_gt_frames_from_unlabeled",
]
