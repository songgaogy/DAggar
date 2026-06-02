from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.utils import to_absolute_path

from robosuite.pipeline.algorithms.discriminator.encoder import SharedFrozenEncoder
from robosuite.pipeline.algorithms.discriminator.lpb_v2_scorer import (
    LPBV2OfflineScorer,
    lpb_disc_intrinsic_from_failure_score,
)
from robosuite.pipeline.algorithms.discriminator.online_bce import (
    DiscriminatorConfig,
    OnlineBCEDiscriminator,
)
from robosuite.pipeline.algorithms.q_learning.common import IQLConfig
from robosuite.pipeline.algorithms.q_learning.data_util import (
    aggregate_chunk_reward,
    chunk_done_mask,
)
from robosuite.pipeline.algorithms.q_learning.replay import (
    _stack_views_uint8,
    _to_image_tensor,
)
from robosuite.pipeline.algorithms.q_learning.iql import IQLLearner
from robosuite.pipeline.algorithms.q_learning.utils.vis_discriminator_util import (
    visualize_selected_trajectory_discriminator,
)
from robosuite.pipeline.common import Transition
from robosuite.pipeline.train_dipole import load_hdf5_demos_into_flow_transitions
from robosuite.policy.flow_multi.utils.env_util import RobosuiteProprioExtractor, parse_env_info


DEFAULT_DEMO_ROOT = "data"


@dataclass
class SelectedDemo:
    hdf5_path: Path
    demo_key: str
    length: int
    successful: bool


@dataclass
class PerStepTrainingDisc:
    """Per-frame Online BCE scores aligned with IQL warmup / vis Q/V."""

    bce_logit: np.ndarray
    intrinsic_reward: np.ndarray
    threshold: float
    pred_failure: np.ndarray

    @property
    def num_frames(self) -> int:
        return int(self.bce_logit.shape[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize trained Q-chunking IQL Q/V on HDF5 rollout windows."
    )
    parser.add_argument("--iql-ckpt", required=True)
    parser.add_argument("--task-data-name", default=None)
    parser.add_argument("--split", default="success_rollout")
    parser.add_argument("--demo-root", default=DEFAULT_DEMO_ROOT)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--demo-key", default=None)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--renderer", default="mjviewer")
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument(
        "--disc-ckpt",
        default=None,
        help="Override BCE head checkpoint for r_disc (Online BCE) and discriminator/ LPB viz.",
    )
    parser.add_argument("--no-disc-reward", action="store_true")
    parser.add_argument(
        "--no-disc-viz",
        action="store_true",
        help="Disable per-frame BCE discriminator visualization on the selected Q/V trajectory.",
    )
    parser.add_argument(
        "--disc-viz-camera",
        default="agentview",
        type=str,
        help="Camera used for the discriminator HUD video. Defaults to agentview.",
    )
    parser.add_argument("--disc-viz-border-thickness", type=int, default=10)
    parser.add_argument("--disc-viz-no-flip-vertical", action="store_true")
    return parser.parse_args()


def resolve_device(requested: str | None, checkpoint_device: str) -> str:
    device = str(requested or checkpoint_device or "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device={device}, but torch.cuda.is_available() is False.")
    return device


def load_iql_payload(iql_ckpt: Path) -> dict[str, Any]:
    if not iql_ckpt.exists():
        raise FileNotFoundError(f"IQL checkpoint does not exist: {iql_ckpt}")
    payload = torch.load(iql_ckpt, map_location="cpu", weights_only=False)
    for key in ("iql_state", "cfg", "encoder_meta"):
        if key not in payload:
            raise KeyError(f"IQL checkpoint {iql_ckpt} is missing required key '{key}'.")
    return payload


def resolve_task_data_name(payload: dict[str, Any], override: str | None) -> str:
    if override:
        return str(override)
    encoder_meta = dict(payload.get("encoder_meta", {}))
    task = encoder_meta.get("task") or encoder_meta.get("task_env")
    if not task:
        raise KeyError("IQL checkpoint encoder_meta does not define task/task_env.")
    return str(task)


def resolve_split_dir(args: argparse.Namespace, task_data_name: str) -> Path:
    return (Path(to_absolute_path(str(args.demo_root))) / task_data_name / str(args.split)).resolve()


def select_demo(split_dir: Path, *, seed: int, demo_key: str | None) -> SelectedDemo:
    hdf5_paths = sorted(split_dir.glob("*.hdf5")) + sorted(split_dir.glob("*.h5"))
    if not hdf5_paths:
        raise FileNotFoundError(f"No HDF5 files found under {split_dir}.")

    candidates: list[SelectedDemo] = []
    for hdf5_path in hdf5_paths:
        with h5py.File(hdf5_path, "r") as file_handle:
            demo_root = file_handle["demos"] if "demos" in file_handle else file_handle["data"]
            for key in sorted(demo_root.keys()):
                demo_group = demo_root[key]
                if demo_key is not None and str(key) != str(demo_key):
                    continue
                candidates.append(
                    SelectedDemo(
                        hdf5_path=hdf5_path,
                        demo_key=str(key),
                        length=int(demo_group.attrs.get("length", len(demo_group["actions"]))),
                        successful=bool(demo_group.attrs.get("successful", False)),
                    )
                )

    if not candidates:
        if demo_key is None:
            raise RuntimeError(f"No demos found under {split_dir}.")
        raise RuntimeError(f"Demo key '{demo_key}' was not found under {split_dir}.")
    rng = random.Random(int(seed))
    return rng.choice(candidates)


def build_proprio_extractor(hdf5_path: Path) -> RobosuiteProprioExtractor:
    with h5py.File(hdf5_path, "r") as file_handle:
        env_info = parse_env_info(file_handle.attrs["env_info"])
    return RobosuiteProprioExtractor(
        env_info,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )


def load_demo_transitions(
    selected: SelectedDemo,
    *,
    camera_names: list[str],
    image_size: int,
    renderer: str,
    control_freq: int,
) -> list[Transition]:
    extractor = build_proprio_extractor(selected.hdf5_path)
    try:
        transitions = load_hdf5_demos_into_flow_transitions(
            selected.hdf5_path,
            policy_camera_names=camera_names,
            camera_aliases={},
            img_height=int(image_size),
            img_width=int(image_size),
            proprio_keys=(),
            renderer=str(renderer),
            control_freq=int(control_freq),
            demo_names=[selected.demo_key],
            state_extractor=extractor,
        )
    finally:
        extractor.close()
    if not transitions:
        raise RuntimeError(f"Selected demo produced zero transitions: {selected.hdf5_path}::{selected.demo_key}")
    return transitions


def build_iql_and_encoder(
    payload: dict[str, Any],
    *,
    device: str,
) -> tuple[IQLLearner, SharedFrozenEncoder, IQLConfig, dict[str, Any]]:
    encoder_meta = dict(payload["encoder_meta"])
    cfg_dict = dict(payload["cfg"])
    cfg_dict["device"] = str(device)
    iql_cfg = IQLConfig(**cfg_dict)

    bce_ckpt = str(encoder_meta.get("bce_ckpt") or encoder_meta.get("disc_warm_start_ckpt") or "")
    if not bce_ckpt:
        raise KeyError("IQL checkpoint encoder_meta does not define bce_ckpt.")
    encoder = SharedFrozenEncoder(bce_ckpt_path=bce_ckpt, device=device, camera_to_view={})
    policy_camera_names = [str(name) for name in encoder_meta.get("policy_camera_names", [])]
    if not policy_camera_names:
        raise KeyError("IQL checkpoint encoder_meta does not define policy_camera_names.")
    encoder.bind_policy_cameras(policy_camera_names)

    action_dim = int(encoder_meta.get("policy_action_dim", payload["iql_state"].get("action_dim", 0)))
    if action_dim <= 0:
        raise KeyError("IQL checkpoint does not define a positive policy_action_dim/action_dim.")
    iql = IQLLearner(cfg=iql_cfg, context_dim=int(encoder.context_dim), action_dim=action_dim)
    iql.load_state_dict(payload["iql_state"], strict=True)
    iql.q1.eval()
    iql.q2.eval()
    iql.v.eval()
    iql.target_v.eval()
    return iql, encoder, iql_cfg, encoder_meta


def build_training_discriminator(
    encoder: SharedFrozenEncoder,
    encoder_meta: dict[str, Any],
    *,
    action_dim: int,
    device: str,
    disc_ckpt_override: str | None,
) -> OnlineBCEDiscriminator:
    """Frozen ``OnlineBCEDiscriminator`` matching IQL warmup / replay sampling."""
    raw_ckpt = (
        disc_ckpt_override
        or encoder_meta.get("disc_warm_start_ckpt")
        or encoder_meta.get("bce_ckpt")
        or ""
    )
    if not raw_ckpt:
        raise KeyError(
            "IQL checkpoint encoder_meta does not define bce_ckpt / disc_warm_start_ckpt."
        )
    bce_ckpt = str(Path(to_absolute_path(str(raw_ckpt))).resolve())
    meta_json = encoder_meta.get("meta_json_path")
    initial_threshold = encoder_meta.get("bce_youden_threshold")
    disc_cfg = DiscriminatorConfig(
        device=str(device),
        warm_start_ckpt=bce_ckpt,
        initial_threshold=(
            float(initial_threshold) if initial_threshold is not None else None
        ),
        meta_json_path=(
            str(Path(to_absolute_path(str(meta_json))).resolve())
            if meta_json
            else None
        ),
    )
    discriminator = OnlineBCEDiscriminator(
        cfg=disc_cfg,
        encoder=encoder,
        context_dim=int(encoder.context_dim),
        action_dim=int(action_dim),
    )
    for param in discriminator.head.parameters():
        param.requires_grad_(False)
    discriminator.head.eval()
    return discriminator


def resolve_disc_ckpt_path(
    encoder_meta: dict[str, Any],
    disc_ckpt_override: str | None,
) -> Path:
    if disc_ckpt_override:
        ckpt_path = Path(to_absolute_path(str(disc_ckpt_override))).resolve()
    else:
        raw_path = str(
            encoder_meta.get("disc_warm_start_ckpt")
            or encoder_meta.get("bce_ckpt")
            or ""
        )
        ckpt_path = Path(to_absolute_path(raw_path)).resolve() if raw_path else None
    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError(
            "BCE checkpoint missing. Pass --disc-ckpt or set encoder_meta.bce_ckpt."
        )
    return ckpt_path


def build_lpb_scorer(
    *,
    payload: dict[str, Any],
    task_name: str,
    device: str,
    batch_size: int,
    disc_ckpt_override: str | None,
    disabled: bool,
) -> LPBV2OfflineScorer | None:
    """LPB v2 BCE scorer — same path as visualize_bce_robosuite.sh (discriminator/ viz)."""
    if disabled:
        return None
    encoder_meta = dict(payload.get("encoder_meta", {}))
    ckpt_path = resolve_disc_ckpt_path(encoder_meta, disc_ckpt_override)
    meta_json = encoder_meta.get("meta_json_path")
    return LPBV2OfflineScorer(
        bce_ckpt_path=ckpt_path,
        task_name=str(task_name),
        device=str(device),
        batch_size=max(1, int(batch_size)),
        meta_json_path=(
            str(Path(to_absolute_path(str(meta_json))).resolve())
            if meta_json
            else None
        ),
    )


def center_crop_resize(image: np.ndarray, image_size: int) -> np.ndarray:
    height, width = image.shape[:2]
    crop_size = min(height, width)
    y0 = (height - crop_size) // 2
    x0 = (width - crop_size) // 2
    crop = image[y0 : y0 + crop_size, x0 : x0 + crop_size]
    if crop.shape[0] == image_size and crop.shape[1] == image_size:
        return np.asarray(crop, dtype=np.uint8)
    ys = np.linspace(0, crop_size - 1, int(image_size)).astype(np.int32)
    xs = np.linspace(0, crop_size - 1, int(image_size)).astype(np.int32)
    return np.asarray(crop[ys][:, xs], dtype=np.uint8)


def stack_views(obs: dict[str, Any], camera_names: list[str], image_size: int) -> np.ndarray:
    images = []
    for camera_name in camera_names:
        if camera_name not in obs:
            raise KeyError(f"Observation is missing camera '{camera_name}'.")
        image = center_crop_resize(np.asarray(obs[camera_name], dtype=np.uint8), image_size)
        images.append(np.transpose(image, (2, 0, 1)))
    return np.stack(images, axis=0)


def build_windows(transitions: list[Transition], horizon: int, max_windows: int | None) -> list[tuple[int, list[Transition]]]:
    max_start = len(transitions) - int(horizon) + 1
    windows: list[tuple[int, list[Transition]]] = []
    for start in range(max(0, max_start)):
        sequence = transitions[start : start + int(horizon)]
        windows.append((start, sequence))
        if max_windows is not None and len(windows) >= int(max_windows):
            break
    if not windows:
        raise RuntimeError(f"Need at least action_horizon={horizon} transitions, got {len(transitions)}.")
    return windows


def next_obs_for_window(transitions: list[Transition], start: int, horizon: int) -> tuple[dict[str, Any], bool]:
    next_index = int(start) + int(horizon)
    if next_index < len(transitions):
        return transitions[next_index].obs, False
    return transitions[start + horizon - 1].next_obs, True


def encode_observation_batch(
    encoder: SharedFrozenEncoder,
    observations: list[dict[str, Any]],
    actions: np.ndarray | None,
    *,
    camera_names: list[str],
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """Replay-aligned encoder forward: uint8 views + optional real actions for s'."""
    if actions is not None and len(actions) != len(observations):
        raise ValueError(
            f"actions length {len(actions)} != observations length {len(observations)}"
        )
    chunks: list[torch.Tensor] = []
    for start in range(0, len(observations), int(batch_size)):
        end = min(len(observations), start + int(batch_size))
        batch_obs = observations[start:end]
        image_np = np.stack(
            [_stack_views_uint8(obs, camera_names) for obs in batch_obs],
            axis=0,
        )
        proprio_np = np.stack(
            [np.asarray(obs["state"], dtype=np.float32) for obs in batch_obs],
            axis=0,
        )
        image_tensor = _to_image_tensor(np.ascontiguousarray(image_np), device)
        proprio_tensor = torch.from_numpy(np.ascontiguousarray(proprio_np)).float()
        action_tensor = None
        if actions is not None:
            action_tensor = torch.from_numpy(
                np.ascontiguousarray(actions[start:end], dtype=np.float32)
            ).float()
        with torch.no_grad():
            latent = encoder.encode(
                image_obs_raw=image_tensor,
                proprio_raw=proprio_tensor,
                action_real=action_tensor,
            )
        chunks.append(latent.detach().cpu())
    return torch.cat(chunks, dim=0)


def tensor_to_float(tensor: torch.Tensor, index: int) -> float:
    return float(tensor[index].detach().cpu().item())


def per_step_training_disc_from_chunks(
    *,
    num_transitions: int,
    horizon: int,
    starts: list[int],
    disc_step_cpu: torch.Tensor,
    disc_logit_cpu: torch.Tensor,
    threshold: float,
) -> PerStepTrainingDisc:
    """Map (W, H) chunk disc tensors to per-frame series (length T).

    Frame ``t`` uses window ``start = min(t, T - H)`` and offset ``h = t - start``,
    matching the chunk position used when ``t`` is the sliding-window start in
    ``steps.csv`` (``disc_intrinsic_step0`` at ``step == t``).
    """
    transition_count = int(num_transitions)
    max_start = max(0, transition_count - int(horizon))
    start_to_wi = {int(start): wi for wi, start in enumerate(starts)}
    intrinsic = np.zeros((transition_count,), dtype=np.float32)
    logit = np.zeros((transition_count,), dtype=np.float32)
    for step in range(transition_count):
        start = min(int(step), max_start)
        offset = int(step) - int(start)
        wi = start_to_wi[int(start)]
        intrinsic[step] = float(disc_step_cpu[wi, offset].item())
        logit[step] = float(disc_logit_cpu[wi, offset].item())
    threshold_f = float(threshold)
    # tau is on failure_score = -expert_logit (LPB convention).
    pred_failure = ((-logit) >= threshold_f).astype(np.int64)
    return PerStepTrainingDisc(
        bce_logit=logit,
        intrinsic_reward=intrinsic,
        threshold=threshold_f,
        pred_failure=pred_failure,
    )


def pad_failure_scores_to_length(
    failure_scores: np.ndarray, num_transitions: int
) -> np.ndarray:
    scores = np.asarray(failure_scores, dtype=np.float32).reshape(-1)
    if scores.size == 0:
        raise ValueError("pad_failure_scores_to_length got empty failure_scores.")
    if scores.size < num_transitions:
        pad = np.full((num_transitions - scores.size,), float(scores[-1]), dtype=np.float32)
        scores = np.concatenate([scores, pad], axis=0)
    elif scores.size > num_transitions:
        scores = scores[:num_transitions]
    return scores


def lpb_intrinsic_from_failure_scores(
    failure_scores: np.ndarray, tau: float
) -> np.ndarray:
    """``r_disc = -sigmoid(failure_score - tau)`` (shared with warmup / replay)."""
    out = lpb_disc_intrinsic_from_failure_score(failure_scores, tau)
    return np.asarray(out, dtype=np.float32)


def per_step_disc_from_lpb_scores(
    failure_scores: np.ndarray, *, tau: float, num_transitions: int
) -> PerStepTrainingDisc:
    scores = pad_failure_scores_to_length(failure_scores, num_transitions)
    intrinsic = lpb_intrinsic_from_failure_scores(scores, tau)
    expert_logit = -scores
    threshold_f = float(tau)
    pred_failure = (scores >= threshold_f).astype(np.int64)
    return PerStepTrainingDisc(
        bce_logit=expert_logit,
        intrinsic_reward=intrinsic,
        threshold=threshold_f,
        pred_failure=pred_failure,
    )


def fill_chunk_disc_from_per_frame_intrinsic(
    *,
    starts: list[int],
    horizon: int,
    num_transitions: int,
    intrinsic_per_frame: np.ndarray,
    expert_logit_per_frame: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_windows = len(starts)
    disc_step = torch.zeros((num_windows, horizon), dtype=torch.float32)
    disc_logit = torch.zeros((num_windows, horizon), dtype=torch.float32)
    for wi, start in enumerate(starts):
        for offset in range(horizon):
            frame_idx = int(start) + offset
            if frame_idx >= num_transitions:
                frame_idx = num_transitions - 1
            disc_step[wi, offset] = float(intrinsic_per_frame[frame_idx])
            disc_logit[wi, offset] = float(expert_logit_per_frame[frame_idx])
    return disc_step, disc_logit


def compute_qv_metrics(
    *,
    iql: IQLLearner,
    encoder: SharedFrozenEncoder,
    discriminator: OnlineBCEDiscriminator | None,
    iql_cfg: IQLConfig,
    transitions: list[Transition],
    camera_names: list[str],
    image_size: int,
    max_windows: int | None,
    batch_size: int,
    device: str,
    lpb_failure_scores: np.ndarray | None = None,
    lpb_tau: float | None = None,
) -> tuple[list[dict[str, float]], PerStepTrainingDisc | None]:
    """Compute per-window Q/V/advantage/TD metrics over the selected demo.

    When ``lpb_failure_scores`` is set, ``r_disc`` uses the same LPB benchmark
    scores as ``discriminator/`` viz (``-sigmoid(failure_score - tau)``). Otherwise
    falls back to ``OnlineBCEDiscriminator`` on ``encode_chunk_frames`` latents.
    """
    horizon = int(iql_cfg.action_horizon)
    discount = float(iql_cfg.discount)
    windows = build_windows(transitions, horizon=horizon, max_windows=max_windows)
    starts = [start for start, _ in windows]
    current_obs = [sequence[0].obs for _, sequence in windows]
    next_obs_data = [next_obs_for_window(transitions, start, horizon) for start, _ in windows]
    next_obs = [item[0] for item in next_obs_data]
    forced_done = np.asarray([float(item[1]) for item in next_obs_data], dtype=np.float32)

    action_np = np.stack(
        [
            np.stack([np.asarray(item.action, dtype=np.float32) for item in sequence], axis=0)
            for _, sequence in windows
        ],
        axis=0,
    )
    reward_np = np.asarray(
        [[float(item.reward) if item.reward is not None else 0.0 for item in sequence] for _, sequence in windows],
        dtype=np.float32,
    )
    done_np = np.asarray(
        [[float(item.done) for item in sequence] for _, sequence in windows],
        dtype=np.float32,
    )
    if forced_done.size:
        done_np[:, -1] = np.maximum(done_np[:, -1], forced_done)

    # Chunk tensors for encode_chunk_frames (same uint8 stacking as IQL replay).
    chunk_images_np = np.stack(
        [
            np.stack([_stack_views_uint8(item.obs, camera_names) for item in sequence], axis=0)
            for _, sequence in windows
        ],
        axis=0,
    )
    chunk_proprio_np = np.stack(
        [
            np.stack([np.asarray(item.obs["state"], dtype=np.float32) for item in sequence], axis=0)
            for _, sequence in windows
        ],
        axis=0,
    )
    view_count, channels, img_h, img_w = chunk_images_np.shape[2:]
    num_windows = len(windows)
    context_dim = int(encoder.context_dim)

    actions_cpu = torch.from_numpy(np.ascontiguousarray(action_np)).float()
    rewards_cpu = torch.from_numpy(np.ascontiguousarray(reward_np)).float()
    dones_cpu = torch.from_numpy(np.ascontiguousarray(done_np)).float()
    use_lpb_disc = (
        lpb_failure_scores is not None
        and lpb_tau is not None
        and float(iql_cfg.disc_reward_coef) != 0.0
    )
    use_disc_reward = use_lpb_disc or (
        discriminator is not None and float(iql_cfg.disc_reward_coef) != 0.0
    )
    score_with_disc = use_lpb_disc or discriminator is not None
    intrinsic_per_frame: np.ndarray | None = None
    expert_logit_per_frame: np.ndarray | None = None
    if use_lpb_disc:
        assert lpb_failure_scores is not None and lpb_tau is not None
        scores_padded = pad_failure_scores_to_length(
            lpb_failure_scores, len(transitions)
        )
        intrinsic_per_frame = lpb_intrinsic_from_failure_scores(
            scores_padded, float(lpb_tau)
        )
        expert_logit_per_frame = -scores_padded

    # Match IQLReplayBuffer: chunk-start context from encode_chunk_frames; s' via
    # encode(..., action_real=last chunk action). Do NOT use encode() without actions.
    chunk_ctx_cpu = torch.empty((num_windows, horizon, context_dim), dtype=torch.float32)
    disc_step_cpu: torch.Tensor | None = None
    disc_logit_cpu: torch.Tensor | None = None
    if score_with_disc:
        disc_step_cpu = torch.empty((num_windows, horizon), dtype=torch.float32)
        disc_logit_cpu = torch.empty((num_windows, horizon), dtype=torch.float32)

    for start_idx in range(0, num_windows, int(batch_size)):
        end_idx = min(num_windows, start_idx + int(batch_size))
        batch_size_local = end_idx - start_idx
        batch_actions = actions_cpu[start_idx:end_idx].to(device)
        batch_chunk_images = chunk_images_np[start_idx:end_idx]
        batch_chunk_proprio = chunk_proprio_np[start_idx:end_idx]
        flat_images = np.ascontiguousarray(
            batch_chunk_images.reshape(
                batch_size_local * horizon, view_count, channels, img_h, img_w
            )
        )
        chunk_images_tensor = _to_image_tensor(flat_images, device).view(
            batch_size_local, horizon, view_count, channels, img_h, img_w
        )
        chunk_proprio_tensor = torch.from_numpy(
            np.ascontiguousarray(batch_chunk_proprio)
        ).float()
        with torch.no_grad():
            chunk_ctx = encoder.encode_chunk_frames(
                chunk_images=chunk_images_tensor,
                chunk_proprio=chunk_proprio_tensor,
                chunk_actions=batch_actions,
            )
            chunk_ctx_cpu[start_idx:end_idx] = chunk_ctx.detach().cpu()
            if (
                use_lpb_disc
                and disc_step_cpu is not None
                and disc_logit_cpu is not None
                and intrinsic_per_frame is not None
                and expert_logit_per_frame is not None
            ):
                batch_starts = starts[start_idx:end_idx]
                step_chunk, logit_chunk = fill_chunk_disc_from_per_frame_intrinsic(
                    starts=batch_starts,
                    horizon=horizon,
                    num_transitions=len(transitions),
                    intrinsic_per_frame=intrinsic_per_frame,
                    expert_logit_per_frame=expert_logit_per_frame,
                )
                disc_step_cpu[start_idx:end_idx] = step_chunk
                disc_logit_cpu[start_idx:end_idx] = logit_chunk
            elif score_with_disc and disc_step_cpu is not None and disc_logit_cpu is not None:
                assert discriminator is not None
                flat_ctx = chunk_ctx.reshape(batch_size_local * horizon, -1)
                r_disc_flat = discriminator.intrinsic_reward(context=flat_ctx)
                disc_step_cpu[start_idx:end_idx] = r_disc_flat.view(
                    batch_size_local, horizon
                ).detach().cpu()
                disc_logit_cpu[start_idx:end_idx] = (
                    discriminator.score(context=flat_ctx).logit.view(batch_size_local, horizon).detach().cpu()
                )

    context_cpu = chunk_ctx_cpu[:, 0, :]
    next_actions_np = np.ascontiguousarray(action_np[:, -1, :], dtype=np.float32)
    next_context_cpu = encode_observation_batch(
        encoder,
        next_obs,
        next_actions_np,
        camera_names=camera_names,
        batch_size=batch_size,
        device=device,
    )

    metrics: list[dict[str, float]] = []
    bootstrap_discount = float(discount) ** horizon
    for start_idx in range(0, num_windows, int(batch_size)):
        end_idx = min(num_windows, start_idx + int(batch_size))
        batch_size_local = end_idx - start_idx
        context = context_cpu[start_idx:end_idx].to(device)
        next_context = next_context_cpu[start_idx:end_idx].to(device)
        actions = actions_cpu[start_idx:end_idx].to(device)
        rewards = rewards_cpu[start_idx:end_idx].to(device)
        dones = dones_cpu[start_idx:end_idx].to(device)

        with torch.no_grad():
            if use_disc_reward and disc_step_cpu is not None:
                disc_step = disc_step_cpu[start_idx:end_idx].to(device)
                disc_reward_horizon = aggregate_chunk_reward(disc_step, discount)
                disc_reward_chunk = disc_step.sum(dim=1, keepdim=True)
                disc_intrinsic_step0 = disc_step[:, :1]
                if use_lpb_disc and expert_logit_per_frame is not None:
                    disc_logit = torch.from_numpy(
                        expert_logit_per_frame[starts[start_idx:end_idx]]
                    ).to(device=device, dtype=rewards.dtype)
                else:
                    assert discriminator is not None
                    disc_logit = discriminator.score(
                        context=chunk_ctx_cpu[start_idx:end_idx, 0, :].to(device)
                    ).logit.view(-1)
            else:
                disc_reward_horizon = torch.zeros(
                    (batch_size_local, 1), device=device, dtype=rewards.dtype
                )
                disc_reward_chunk = torch.zeros_like(disc_reward_horizon)
                disc_intrinsic_step0 = torch.zeros_like(disc_reward_horizon)
                disc_logit = torch.zeros((batch_size_local,), device=device, dtype=rewards.dtype)

            q1 = iql.q1(context, actions)
            q2 = iql.q2(context, actions)
            q_min = torch.minimum(q1, q2)
            q_mean = 0.5 * (q1 + q2)
            q_max = torch.maximum(q1, q2)
            v = iql.v(context)
            next_v = iql.target_v(next_context)
            env_reward_horizon = aggregate_chunk_reward(rewards, discount)
            done_horizon = chunk_done_mask(dones).to(device)

            total_reward_horizon = float(iql_cfg.output_reward_coef) * env_reward_horizon + float(iql_cfg.disc_reward_coef) * disc_reward_horizon
            bootstrap_v = bootstrap_discount * (1.0 - done_horizon) * next_v
            td_target = total_reward_horizon + bootstrap_v
            td_residual = td_target - q_min
            advantage = q_min - v
            # Chunk Bellman target minus V (same γ^H bootstrap as IQL training).
            advantage_td1 = td_target - v

        for local_idx, global_idx in enumerate(range(start_idx, end_idx)):
            row = {
                "window_index": float(global_idx),
                "step": float(starts[global_idx]),
                "env_reward_horizon": tensor_to_float(env_reward_horizon, local_idx),
                "disc_logit": tensor_to_float(disc_logit, local_idx),
                "disc_intrinsic_step0": tensor_to_float(disc_intrinsic_step0, local_idx),
                "disc_reward_chunk": tensor_to_float(disc_reward_chunk, local_idx),
                "disc_reward_horizon": tensor_to_float(disc_reward_horizon, local_idx),
                "bootstrap_v": tensor_to_float(bootstrap_v, local_idx),
                "total_reward_horizon": tensor_to_float(total_reward_horizon, local_idx),
                "done_horizon": tensor_to_float(done_horizon, local_idx),
                "q1": tensor_to_float(q1, local_idx),
                "q2": tensor_to_float(q2, local_idx),
                "q_min": tensor_to_float(q_min, local_idx),
                "q_mean": tensor_to_float(q_mean, local_idx),
                "q_max": tensor_to_float(q_max, local_idx),
                "v": tensor_to_float(v, local_idx),
                "next_v": tensor_to_float(next_v, local_idx),
                "td_target": tensor_to_float(td_target, local_idx),
                "td_residual": tensor_to_float(td_residual, local_idx),
                "advantage": tensor_to_float(advantage, local_idx),
                "advantage_td1": tensor_to_float(advantage_td1, local_idx),
            }
            for horizon_index in range(horizon):
                for action_index in range(actions_cpu.shape[-1]):
                    row[f"action_h{horizon_index}_{action_index}"] = float(
                        actions_cpu[global_idx, horizon_index, action_index].item()
                    )
            metrics.append(row)

    per_step_disc = None
    if use_lpb_disc and lpb_failure_scores is not None and lpb_tau is not None:
        per_step_disc = per_step_disc_from_lpb_scores(
            lpb_failure_scores,
            tau=float(lpb_tau),
            num_transitions=len(transitions),
        )
    elif (
        score_with_disc
        and disc_step_cpu is not None
        and disc_logit_cpu is not None
        and discriminator is not None
    ):
        per_step_disc = per_step_training_disc_from_chunks(
            num_transitions=len(transitions),
            horizon=horizon,
            starts=starts,
            disc_step_cpu=disc_step_cpu,
            disc_logit_cpu=disc_logit_cpu,
            threshold=float(discriminator.threshold),
        )
    return metrics, per_step_disc


def write_metrics_csv(path: Path, metrics: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(metrics[0].keys()) if metrics else []
    with path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)


def write_video(path: Path, transitions: list[Transition], image_keys: list[str], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=int(fps), codec="libx264", macro_block_size=1) as writer:
        for transition in transitions:
            frames = []
            for key in image_keys:
                if key not in transition.obs:
                    continue
                frame = np.asarray(transition.obs[key], dtype=np.uint8)
                frames.append(frame[::-1, ...])
            if frames:
                writer.append_data(np.concatenate(frames, axis=1) if len(frames) > 1 else frames[0])


def filter_nonoverlap_chunk_metrics(
    metrics: list[dict[str, float]], action_horizon: int
) -> list[dict[str, float]]:
    """Keep one window per disjoint chunk (steps 0, H, 2H, ...)."""
    stride = int(action_horizon)
    if stride <= 0:
        raise ValueError(f"action_horizon must be positive, got {action_horizon}")
    return [row for row in metrics if int(row["step"]) % stride == 0]


def _save_qv_timeseries_png(
    path_base: Path,
    metrics: list[dict[str, float]],
    title: str,
    *,
    action_horizon: int,
    per_step_disc: PerStepTrainingDisc | None = None,
) -> Path:
    if not metrics:
        raise ValueError("Cannot plot Q/V timeseries with empty metrics.")
    steps = np.asarray([row["step"] for row in metrics], dtype=np.float32)
    q_min = np.asarray([row["q_min"] for row in metrics], dtype=np.float32)
    q_mean = np.asarray([row["q_mean"] for row in metrics], dtype=np.float32)
    q_max = np.asarray([row["q_max"] for row in metrics], dtype=np.float32)
    v = np.asarray([row["v"] for row in metrics], dtype=np.float32)
    next_v = np.asarray([row["next_v"] for row in metrics], dtype=np.float32)
    td_target = np.asarray([row["td_target"] for row in metrics], dtype=np.float32)
    td_residual = np.asarray([row["td_residual"] for row in metrics], dtype=np.float32)
    advantage = np.asarray([row["advantage"] for row in metrics], dtype=np.float32)
    advantage_td1 = np.asarray([row["advantage_td1"] for row in metrics], dtype=np.float32)
    env_rewards = np.asarray([row["env_reward_horizon"] for row in metrics], dtype=np.float32)
    total_rewards = np.asarray([row["total_reward_horizon"] for row in metrics], dtype=np.float32)
    disc_rewards = np.asarray([row["disc_reward_horizon"] for row in metrics], dtype=np.float32)
    disc_step0 = np.asarray([row["disc_intrinsic_step0"] for row in metrics], dtype=np.float32)
    bootstrap_v = np.asarray([row["bootstrap_v"] for row in metrics], dtype=np.float32)

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(title)

    axes[0].plot(steps, q_mean, label="Q mean", color="tab:blue")
    axes[0].fill_between(steps, q_min, q_max, color="tab:blue", alpha=0.18, label="Q min/max")
    axes[0].plot(steps, v, label="V", color="tab:orange")
    axes[0].plot(steps, next_v, label="target next V", color="tab:green", alpha=0.8)
    axes[0].set_ylabel("Q / V")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, td_target, label="TD target (r + γ^H V')", color="tab:purple")
    axes[1].plot(steps, q_min, label="Q min", color="tab:blue", alpha=0.75)
    axes[1].plot(steps, bootstrap_v, label="γ^H · V(s')", color="tab:green", alpha=0.65, linestyle="--")
    axes[1].plot(steps, total_rewards, label="chunk r_total", color="tab:gray", alpha=0.65, linestyle=":")
    axes[1].set_ylabel("Target / Q")
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # axes[2].plot(steps, td_residual, label="TD residual", color="tab:red")
    axes[2].plot(steps, advantage, label="advantage Qmin - V", color="tab:brown")
    axes[2].plot(
        steps,
        advantage_td1,
        label="advantage TD (Σγ^i r + γ^H V' - V)",
        color="tab:red",
        alpha=0.85,
    )
    axes[2].axhline(0.0, color="black", linewidth=1)
    axes[2].set_ylabel("Residual / Adv")
    axes[2].legend(loc="best")
    axes[2].grid(True, alpha=0.3)

    axes[3].step(steps, total_rewards, where="post", label="total chunk reward (Bellman r)", color="tab:gray")
    axes[3].step(steps, env_rewards, where="post", label="env chunk reward", color="tab:olive", alpha=0.75)
    axes[3].plot(
        steps,
        disc_rewards,
        label=f"disc γ-agg ({int(action_horizon)}-step window)",
        color="tab:pink",
        alpha=0.85,
    )
    axes[3].plot(
        steps,
        disc_step0,
        label="disc intrinsic @ chunk start",
        color="tab:red",
        alpha=0.55,
        linestyle="--",
    )
    if per_step_disc is not None:
        frame_steps = np.arange(int(per_step_disc.num_frames), dtype=np.float32)
        axes[3].plot(
            frame_steps,
            per_step_disc.intrinsic_reward,
            label="disc intrinsic (per-frame)",
            color="tab:orange",
            alpha=0.45,
            linewidth=1.0,
        )
    axes[3].set_ylabel("Reward")
    axes[3].set_xlabel("Step")
    axes[3].legend(loc="best")
    axes[3].grid(True, alpha=0.3)

    fig.tight_layout()
    path_base.parent.mkdir(parents=True, exist_ok=True)
    out_path = path_base.with_suffix(".png")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_qv(
    path_base: Path,
    metrics: list[dict[str, float]],
    title: str,
    *,
    action_horizon: int,
    per_step_disc: PerStepTrainingDisc | None = None,
) -> dict[str, Path]:
    """Write overlapping-window and non-overlapping-chunk Q/V plots (PNG only)."""
    plot_paths: dict[str, Path] = {}
    plot_paths["overlapping"] = _save_qv_timeseries_png(
        path_base,
        metrics,
        title,
        action_horizon=int(action_horizon),
        per_step_disc=per_step_disc,
    )
    nonoverlap_metrics = filter_nonoverlap_chunk_metrics(metrics, int(action_horizon))
    nonoverlap_base = path_base.parent / f"{path_base.name}_nonoverlap"
    plot_paths["nonoverlap"] = _save_qv_timeseries_png(
        nonoverlap_base,
        nonoverlap_metrics,
        f"{title} (non-overlapping chunks, stride={int(action_horizon)})",
        action_horizon=int(action_horizon),
        per_step_disc=per_step_disc,
    )
    return plot_paths


def summarize_metrics(metrics: list[dict[str, float]], iql_cfg: IQLConfig) -> dict[str, Any]:
    q_values = np.asarray([row["q_mean"] for row in metrics], dtype=np.float32)
    v_values = np.asarray([row["v"] for row in metrics], dtype=np.float32)
    td_abs = np.abs(np.asarray([row["td_residual"] for row in metrics], dtype=np.float32))
    adv_values = np.asarray([row["advantage"] for row in metrics], dtype=np.float32)
    env_rewards = np.asarray([row["env_reward_horizon"] for row in metrics], dtype=np.float32)
    total_rewards = np.asarray([row["total_reward_horizon"] for row in metrics], dtype=np.float32)
    disc_rewards = np.asarray([row["disc_reward_horizon"] for row in metrics], dtype=np.float32)
    return {
        "num_windows": int(len(metrics)),
        "action_horizon": int(iql_cfg.action_horizon),
        "discount": float(iql_cfg.discount),
        "disc_reward_coef": float(iql_cfg.disc_reward_coef),
        "env_reward_horizon_mean": float(env_rewards.mean()),
        "env_reward_horizon_min": float(env_rewards.min()),
        "env_reward_horizon_max": float(env_rewards.max()),
        "disc_reward_horizon_mean": float(disc_rewards.mean()),
        "total_reward_horizon_mean": float(total_rewards.mean()),
        "total_reward_horizon_min": float(total_rewards.min()),
        "total_reward_horizon_max": float(total_rewards.max()),
        "q_mean_min": float(q_values.min()),
        "q_mean_max": float(q_values.max()),
        "v_min": float(v_values.min()),
        "v_max": float(v_values.max()),
        "advantage_mean": float(adv_values.mean()),
        "advantage_min": float(adv_values.min()),
        "advantage_max": float(adv_values.max()),
        "mean_abs_td_residual": float(td_abs.mean()),
        "max_abs_td_residual": float(td_abs.max()),
    }


def main() -> None:
    args = parse_args()
    iql_ckpt = Path(to_absolute_path(str(args.iql_ckpt))).resolve()
    payload = load_iql_payload(iql_ckpt)
    task_data_name = resolve_task_data_name(payload, args.task_data_name)
    split_dir = resolve_split_dir(args, task_data_name)
    if not split_dir.exists():
        raise FileNotFoundError(f"Demo split directory does not exist: {split_dir}")

    checkpoint_device = str(dict(payload["cfg"]).get("device", "cpu"))
    device = resolve_device(args.device, checkpoint_device)
    iql, encoder, iql_cfg, encoder_meta = build_iql_and_encoder(payload, device=device)
    camera_names = [str(name) for name in encoder_meta["policy_camera_names"]]
    image_size = int(encoder_meta.get("image_size", 128))
    action_dim = int(encoder_meta.get("policy_action_dim", iql.action_dim))

    selected = select_demo(split_dir, seed=int(args.seed), demo_key=args.demo_key)
    transitions = load_demo_transitions(
        selected,
        camera_names=camera_names,
        image_size=image_size,
        renderer=str(args.renderer),
        control_freq=int(args.control_freq),
    )

    lpb_scorer: LPBV2OfflineScorer | None = None
    lpb_failure_scores: np.ndarray | None = None
    lpb_tau: float | None = None
    need_lpb = (not bool(args.no_disc_reward)) or (not bool(args.no_disc_viz))
    if need_lpb:
        lpb_scorer = build_lpb_scorer(
            payload=payload,
            task_name=task_data_name,
            device=device,
            batch_size=max(1, int(args.batch_size)),
            disc_ckpt_override=args.disc_ckpt,
            disabled=False,
        )
        assert lpb_scorer is not None
        lpb_failure_scores = lpb_scorer.score_hdf5_demo(
            selected.hdf5_path,
            selected.demo_key,
            fps=int(args.video_fps),
        )
        lpb_tau = float(lpb_scorer.tau)
        print(
            f"[vis_qv] LPB disc scores: T={lpb_failure_scores.shape[0]} "
            f"tau={lpb_tau:.6f} (source={lpb_scorer.tau_source}) "
            f"output_reward_coef={float(iql_cfg.output_reward_coef)} disc_reward_coef={float(iql_cfg.disc_reward_coef)}"
        )

    discriminator = None
    if not bool(args.no_disc_reward) and lpb_failure_scores is None:
        discriminator = build_training_discriminator(
            encoder,
            encoder_meta,
            action_dim=action_dim,
            device=device,
            disc_ckpt_override=args.disc_ckpt,
        )
        print(
            f"[vis_qv] online disc fallback: threshold={discriminator.threshold:.6f} "
            f"(source={discriminator.threshold_source})"
        )

    metrics, per_step_disc = compute_qv_metrics(
        iql=iql,
        encoder=encoder,
        discriminator=discriminator,
        iql_cfg=iql_cfg,
        transitions=transitions,
        camera_names=camera_names,
        image_size=image_size,
        max_windows=args.max_windows,
        batch_size=max(1, int(args.batch_size)),
        device=device,
        lpb_failure_scores=lpb_failure_scores,
        lpb_tau=lpb_tau,
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_root = Path(to_absolute_path(str(args.output_root))).resolve()
    output_dir = output_root / f"{task_data_name}_iql-qv/{args.split}_seed{args.seed}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "steps.csv"
    video_path = output_dir / "rollout_policy_obs.mp4"
    plot_base = output_dir / "qv_timeseries"
    summary_path = output_dir / "summary.json"

    write_metrics_csv(csv_path, metrics)
    write_video(video_path, transitions, image_keys=camera_names, fps=int(args.video_fps))
    plot_paths = plot_qv(
        plot_base,
        metrics,
        title=f"{task_data_name} {args.split} {selected.hdf5_path.name}::{selected.demo_key}",
        action_horizon=int(iql_cfg.action_horizon),
        per_step_disc=per_step_disc,
    )

    disc_viz_outputs = None
    if not bool(args.no_disc_viz):
        if lpb_scorer is None:
            lpb_scorer = build_lpb_scorer(
                payload=payload,
                task_name=task_data_name,
                device=device,
                batch_size=max(1, int(args.batch_size)),
                disc_ckpt_override=args.disc_ckpt,
                disabled=False,
            )
        disc_ckpt_path = resolve_disc_ckpt_path(encoder_meta, args.disc_ckpt)
        disc_viz_result = visualize_selected_trajectory_discriminator(
            disc_ckpt=str(disc_ckpt_path),
            task_name=task_data_name,
            selected_hdf5_path=selected.hdf5_path,
            selected_demo_key=selected.demo_key,
            transitions=transitions,
            camera_names=camera_names,
            image_size=image_size,
            output_dir=output_dir / "discriminator",
            device=device,
            batch_size=max(1, int(args.batch_size)),
            video_fps=int(args.video_fps),
            camera_name=args.disc_viz_camera,
            border_thickness=int(args.disc_viz_border_thickness),
            flip_vertical=not bool(args.disc_viz_no_flip_vertical),
            scorer=lpb_scorer,
        )
        disc_viz_outputs = {
            "output_dir": str(disc_viz_result.output_dir),
            "scores_csv": str(disc_viz_result.scores_csv),
            "plot_png": str(disc_viz_result.plot_png),
            "plot_pdf": str(disc_viz_result.plot_pdf),
            "video": str(disc_viz_result.video),
            "summary": str(disc_viz_result.summary_json),
            "scoring_path": "LPBV2OfflineScorer.score_hdf5_demo(BCEBenchmarkDiscriminator.score_trajectory)",
        }

    summary = {
        "iql_ckpt": str(iql_ckpt),
        "schema_version": int(payload.get("schema_version", -1)),
        "task_data_name": task_data_name,
        "task_env": str(encoder_meta.get("task_env", task_data_name)),
        "split": str(args.split),
        "selected_hdf5": str(selected.hdf5_path),
        "selected_demo_key": selected.demo_key,
        "selected_demo_length": int(selected.length),
        "selected_demo_successful": bool(selected.successful),
        "loaded_transitions": int(len(transitions)),
        "used_windows": int(len(metrics)),
        "camera_names": camera_names,
        "image_size": int(image_size),
        "device": str(device),
        "q_chunking": True,
        "disc_reward_enabled": (
            (lpb_failure_scores is not None or discriminator is not None)
            and float(iql_cfg.disc_reward_coef) != 0.0
        ),
        "output_reward_coef": float(iql_cfg.output_reward_coef),
        "disc_reward_source": (
            "LPBV2OfflineScorer(-sigmoid(failure_score - tau))"
            if lpb_failure_scores is not None
            else (
                "OnlineBCEDiscriminator.intrinsic_reward(encode_chunk_frames)"
                if discriminator is not None
                else None
            )
        ),
        "disc_threshold": None if discriminator is None else float(discriminator.threshold),
        "disc_threshold_source": None if discriminator is None else str(discriminator.threshold_source),
        "lpb_disc_viz_tau": None if lpb_scorer is None else float(lpb_scorer.tau),
        "lpb_disc_viz_tau_source": None if lpb_scorer is None else str(lpb_scorer.tau_source),
        "per_step_disc_frames": None if per_step_disc is None else int(per_step_disc.num_frames),
        "per_step_disc_first_pred_failure": (
            None
            if per_step_disc is None or not bool(per_step_disc.pred_failure.any())
            else int(np.where(per_step_disc.pred_failure.astype(bool))[0][0])
        ),
        "bce_ckpt": str(encoder_meta.get("bce_ckpt", "")),
        "disc_ckpt_override": None if args.disc_ckpt is None else str(Path(to_absolute_path(str(args.disc_ckpt))).resolve()),
        "outputs": {
            "steps_csv": str(csv_path),
            "video": str(video_path),
            "plot_png": str(plot_paths["overlapping"]),
            "plot_png_nonoverlap": str(plot_paths["nonoverlap"]),
            "discriminator": disc_viz_outputs,
        },
        "metrics_summary": summarize_metrics(metrics, iql_cfg),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"selected_demo={selected.hdf5_path}::{selected.demo_key}")
    print(f"iql_ckpt={iql_ckpt}")
    print(f"output_dir={output_dir}")
    print(f"video={video_path}")
    print(f"plot_png={plot_paths['overlapping']}")
    print(f"plot_png_nonoverlap={plot_paths['nonoverlap']}")
    if disc_viz_outputs is not None:
        print(f"disc_viz_dir={disc_viz_outputs['output_dir']}")
        print(f"disc_viz_video={disc_viz_outputs['video']}")
        print(f"disc_viz_plot_png={disc_viz_outputs['plot_png']}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
