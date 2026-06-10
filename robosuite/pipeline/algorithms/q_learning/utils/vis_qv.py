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


@dataclass
class QCandidateInputs:
    """State/action tensors reused for counterfactual Q scoring."""

    starts: list[int]
    context_cpu: torch.Tensor
    actions_cpu: torch.Tensor


@dataclass
class WindowSelectionInfo:
    candidate_windows: int
    used_windows: int
    dropped_truncated_boundary_windows: int
    dropped_mid_terminal_windows: int


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
    parser.add_argument(
        "--q-candidate-noise-sigmas",
        default="0.05,0.10,0.20",
        help="Comma-separated Gaussian sigmas for full action-chunk perturbation.",
    )
    parser.add_argument(
        "--q-candidate-random-n",
        type=int,
        default=16,
        help="Number of uniform random action chunks to score per window.",
    )
    parser.add_argument(
        "--q-candidate-single-dim-sigma",
        type=float,
        default=0.20,
        help="Gaussian sigma for single-action-dim perturbation diagnostics.",
    )
    parser.add_argument(
        "--q-candidate-single-dim-n",
        type=int,
        default=0,
        help="Number of action dims to perturb one-at-a-time; <=0 means all dims.",
    )
    parser.add_argument(
        "--q-candidate-seed",
        type=int,
        default=None,
        help="RNG seed for candidate action diagnostics. Defaults to --seed.",
    )
    parser.add_argument("--q-candidate-action-low", type=float, default=-1.0)
    parser.add_argument("--q-candidate-action-high", type=float, default=1.0)
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
    reward_mode: str,
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
            reward_mode=str(reward_mode),
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
    iql.q_ensemble.eval()
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


def _is_truncated_boundary(transition: Transition) -> bool:
    info = transition.info or {}
    return bool(info.get("is_truncated_boundary", False)) or (
        str(info.get("episode_terminal_reason", "")).lower() == "truncated"
    )


def next_obs_for_window(
    transitions: list[Transition], start: int, horizon: int
) -> tuple[dict[str, Any] | None, bool, bool, bool]:
    """Return next obs and terminal/truncation flags for an H-step window."""
    next_index = int(start) + int(horizon)
    if next_index < len(transitions):
        return transitions[next_index].obs, False, True, False
    last = transitions[start + horizon - 1]
    if bool(last.done) and not _is_truncated_boundary(last):
        return last.next_obs, True, False, False
    return None, False, False, True


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

    Frame ``t`` uses the nearest available chunk start not greater than ``t``.
    Truncated boundary chunks may be absent from ``steps.csv``, so the final
    frames are clamped to the last available chunk for HUD display only.
    """
    transition_count = int(num_transitions)
    if not starts:
        raise ValueError("per_step_training_disc_from_chunks requires at least one start.")
    sorted_starts = np.asarray([int(start) for start in starts], dtype=np.int64)
    start_to_wi = {int(start): wi for wi, start in enumerate(starts)}
    intrinsic = np.zeros((transition_count,), dtype=np.float32)
    logit = np.zeros((transition_count,), dtype=np.float32)
    for step in range(transition_count):
        pos = int(np.searchsorted(sorted_starts, int(step), side="right") - 1)
        pos = max(0, min(pos, int(sorted_starts.shape[0]) - 1))
        start = int(sorted_starts[pos])
        offset = max(0, min(int(step) - int(start), int(horizon) - 1))
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
) -> tuple[
    list[dict[str, float]],
    PerStepTrainingDisc | None,
    QCandidateInputs,
    WindowSelectionInfo,
]:
    """Compute per-window Q/V/advantage/TD metrics over the selected demo.

    When ``lpb_failure_scores`` is set, ``r_disc`` uses the same LPB benchmark
    scores as ``discriminator/`` viz (``-sigmoid(failure_score - tau)``). Otherwise
    falls back to ``OnlineBCEDiscriminator`` on ``encode_chunk_frames`` latents.
    """
    horizon = int(iql_cfg.action_horizon)
    discount = float(iql_cfg.discount)
    candidate_windows = build_windows(transitions, horizon=horizon, max_windows=max_windows)
    windows: list[tuple[int, list[Transition]]] = []
    next_obs: list[dict[str, Any]] = []
    forced_done_list: list[float] = []
    has_bootstrap_next_obs_list: list[float] = []
    true_terminal_list: list[float] = []
    truncated_boundary_list: list[float] = []
    dropped_truncated = 0
    dropped_mid_terminal = 0
    for start, sequence in candidate_windows:
        if any(bool(item.done) for item in sequence[:-1]):
            dropped_mid_terminal += 1
            continue
        next_obs_item, forced_done_item, has_bootstrap_item, truncated_boundary = next_obs_for_window(
            transitions, start, horizon
        )
        if truncated_boundary:
            dropped_truncated += 1
            continue
        if next_obs_item is None:
            raise RuntimeError(f"Window start={start} has neither bootstrap nor terminal next_obs.")
        windows.append((start, sequence))
        next_obs.append(next_obs_item)
        forced_done_list.append(float(forced_done_item))
        has_bootstrap_next_obs_list.append(float(has_bootstrap_item))
        true_terminal_list.append(float(forced_done_item))
        truncated_boundary_list.append(0.0)
    if not windows:
        raise RuntimeError(
            "No valid Q/V windows remain after dropping truncated boundary windows "
            f"(candidate_windows={len(candidate_windows)})."
        )
    window_selection = WindowSelectionInfo(
        candidate_windows=int(len(candidate_windows)),
        used_windows=int(len(windows)),
        dropped_truncated_boundary_windows=int(dropped_truncated),
        dropped_mid_terminal_windows=int(dropped_mid_terminal),
    )
    starts = [start for start, _ in windows]
    current_obs = [sequence[0].obs for _, sequence in windows]
    forced_done = np.asarray(forced_done_list, dtype=np.float32)
    has_bootstrap_next_obs = np.asarray(has_bootstrap_next_obs_list, dtype=np.float32)
    is_true_terminal_window = np.asarray(true_terminal_list, dtype=np.float32)
    is_truncated_boundary_window = np.asarray(truncated_boundary_list, dtype=np.float32)

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
    # Mirror IQL training-target shaping so the plotted target/residual match.
    mc_blend_lambda = float(getattr(iql_cfg, "mc_blend_lambda", 0.0))
    terminal_undisc = bool(getattr(iql_cfg, "terminal_undiscounted_reward", False))
    output_reward_coef = float(iql_cfg.output_reward_coef)
    starts_arr = np.asarray(starts, dtype=np.int64)
    mc_return_all = np.zeros(num_windows, dtype=np.float32)
    terminal_window_idx = np.nonzero(is_true_terminal_window > 0.5)[0]
    if terminal_window_idx.size:
        t_term_start = int(starts_arr[int(terminal_window_idx[-1])])
        exps = np.maximum(t_term_start - starts_arr, 0)
        mc_return_all = (float(discount) ** exps).astype(np.float32)
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

            q_values = iql._q_values(context, actions)
            q1 = q_values[0]
            q2 = q_values[min(1, q_values.shape[0] - 1)]
            q_min = q_values.min(dim=0).values
            q_mean = q_values.mean(dim=0)
            q_max = q_values.max(dim=0).values
            v = iql.v(context)
            next_v = iql.target_v(next_context)
            env_reward_horizon = aggregate_chunk_reward(rewards, discount)
            done_horizon = chunk_done_mask(dones).to(device)

            total_reward_horizon = float(iql_cfg.output_reward_coef) * env_reward_horizon + float(iql_cfg.disc_reward_coef) * disc_reward_horizon
            # Terminal chunk reward override (mirror IQL replay): undiscounted env
            # reward (=1 for a 0/1 success chunk) so the terminal target -> ~1.
            if terminal_undisc:
                term_reward = output_reward_coef * rewards.sum(dim=1, keepdim=True)
                total_reward_horizon = torch.where(
                    done_horizon > 0.5, term_reward, total_reward_horizon
                )
            bootstrap_v = bootstrap_discount * (1.0 - done_horizon) * next_v
            td_target = total_reward_horizon + bootstrap_v
            # Blended training target y = (1-λ)·TD + λ·MC_return_to_go.
            mc_return = torch.from_numpy(
                mc_return_all[start_idx:end_idx]
            ).to(device=device, dtype=td_target.dtype).unsqueeze(-1)
            value_target = (1.0 - mc_blend_lambda) * td_target + mc_blend_lambda * mc_return
            td_residual = value_target - q_mean
            advantage = q_mean - v
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
                "has_bootstrap_next_obs": float(has_bootstrap_next_obs[global_idx]),
                "is_true_terminal_window": float(is_true_terminal_window[global_idx]),
                "is_truncated_boundary_window": float(is_truncated_boundary_window[global_idx]),
                "q1": tensor_to_float(q1, local_idx),
                "q2": tensor_to_float(q2, local_idx),
                "q_min": tensor_to_float(q_min, local_idx),
                "q_mean": tensor_to_float(q_mean, local_idx),
                "q_max": tensor_to_float(q_max, local_idx),
                "v": tensor_to_float(v, local_idx),
                "next_v": tensor_to_float(next_v, local_idx),
                "td_target": tensor_to_float(td_target, local_idx),
                "mc_return": tensor_to_float(mc_return, local_idx),
                "value_target": tensor_to_float(value_target, local_idx),
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
    q_candidate_inputs = QCandidateInputs(
        starts=list(starts),
        context_cpu=context_cpu.detach().cpu(),
        actions_cpu=actions_cpu.detach().cpu(),
    )
    return metrics, per_step_disc, q_candidate_inputs, window_selection


def write_metrics_csv(path: Path, metrics: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(metrics[0].keys()) if metrics else []
    with path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)


def parse_float_csv(value: str) -> list[float]:
    items: list[float] = []
    for raw in str(value).split(","):
        item = raw.strip()
        if not item:
            continue
        items.append(float(item))
    return items


def selected_single_action_dims(action_dim: int, requested: int) -> list[int]:
    dim = int(action_dim)
    if dim <= 0:
        raise ValueError(f"action_dim must be positive, got {action_dim}")
    count = dim if int(requested) <= 0 else min(dim, int(requested))
    if count >= dim:
        return list(range(dim))
    return sorted({int(x) for x in np.linspace(0, dim - 1, count, dtype=np.int64)})


def build_q_candidate_actions(
    actions_cpu: torch.Tensor,
    *,
    noise_sigmas: list[float],
    random_n: int,
    single_dim_sigma: float,
    single_dim_n: int,
    seed: int,
    action_low: float,
    action_high: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    base = actions_cpu.detach().cpu().numpy().astype(np.float32, copy=True)
    if base.ndim != 3:
        raise ValueError(f"actions_cpu expected (W, H, D), got shape {base.shape}")
    action_dim = int(base.shape[-1])
    low = float(action_low)
    high = float(action_high)
    if not low < high:
        raise ValueError(f"q candidate action bounds must satisfy low < high, got {low}, {high}")
    rng = np.random.default_rng(int(seed))

    candidates: list[np.ndarray] = [base.copy()]
    specs: list[dict[str, Any]] = [
        {
            "candidate_type": "demo",
            "candidate_group": "demo",
            "candidate_label": "demo",
            "noise_sigma": None,
            "action_dim": None,
            "random_index": None,
        }
    ]

    for sigma in noise_sigmas:
        sigma_f = float(sigma)
        if sigma_f <= 0.0:
            continue
        noise = rng.normal(loc=0.0, scale=sigma_f, size=base.shape).astype(np.float32)
        candidates.append(np.clip(base + noise, low, high).astype(np.float32))
        specs.append(
            {
                "candidate_type": "noise",
                "candidate_group": f"noise_sigma_{sigma_f:g}",
                "candidate_label": f"noise_sigma_{sigma_f:g}",
                "noise_sigma": sigma_f,
                "action_dim": None,
                "random_index": None,
            }
        )

    if float(single_dim_sigma) > 0.0:
        for dim in selected_single_action_dims(action_dim, int(single_dim_n)):
            noise = np.zeros_like(base)
            noise[:, :, int(dim)] = rng.normal(
                loc=0.0,
                scale=float(single_dim_sigma),
                size=base.shape[:2],
            ).astype(np.float32)
            candidates.append(np.clip(base + noise, low, high).astype(np.float32))
            specs.append(
                {
                    "candidate_type": "single_dim_noise",
                    "candidate_group": "single_dim_noise",
                    "candidate_label": f"single_dim_noise_sigma_{float(single_dim_sigma):g}_dim_{int(dim)}",
                    "noise_sigma": float(single_dim_sigma),
                    "action_dim": int(dim),
                    "random_index": None,
                }
            )

    for random_index in range(max(0, int(random_n))):
        random_action = rng.uniform(low=low, high=high, size=base.shape).astype(np.float32)
        candidates.append(random_action)
        specs.append(
            {
                "candidate_type": "random_uniform",
                "candidate_group": "random_uniform",
                "candidate_label": f"random_uniform_{int(random_index)}",
                "noise_sigma": None,
                "action_dim": None,
                "random_index": int(random_index),
            }
        )

    return np.stack(candidates, axis=1), specs


def compute_q_candidate_rows(
    *,
    iql: IQLLearner,
    inputs: QCandidateInputs,
    noise_sigmas: list[float],
    random_n: int,
    single_dim_sigma: float,
    single_dim_n: int,
    seed: int,
    action_low: float,
    action_high: float,
    batch_size: int,
    device: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate_actions, specs = build_q_candidate_actions(
        inputs.actions_cpu,
        noise_sigmas=noise_sigmas,
        random_n=int(random_n),
        single_dim_sigma=float(single_dim_sigma),
        single_dim_n=int(single_dim_n),
        seed=int(seed),
        action_low=float(action_low),
        action_high=float(action_high),
    )
    num_windows, num_candidates, horizon, action_dim = candidate_actions.shape
    flat_actions = torch.from_numpy(
        np.ascontiguousarray(candidate_actions.reshape(num_windows * num_candidates, horizon, action_dim))
    ).float()
    flat_context = inputs.context_cpu.repeat_interleave(num_candidates, dim=0).float()

    total = int(flat_actions.shape[0])
    q_min_flat = np.empty((total,), dtype=np.float32)
    q_mean_flat = np.empty((total,), dtype=np.float32)
    q_max_flat = np.empty((total,), dtype=np.float32)
    q_std_flat = np.empty((total,), dtype=np.float32)
    eval_batch_size = max(1, int(batch_size))
    with torch.no_grad():
        for start in range(0, total, eval_batch_size):
            end = min(total, start + eval_batch_size)
            q_values = iql._q_values(
                flat_context[start:end].to(device),
                flat_actions[start:end].to(device),
            )
            q_min_flat[start:end] = q_values.min(dim=0).values.view(-1).detach().cpu().numpy()
            q_mean_flat[start:end] = q_values.mean(dim=0).view(-1).detach().cpu().numpy()
            q_max_flat[start:end] = q_values.max(dim=0).values.view(-1).detach().cpu().numpy()
            q_std_flat[start:end] = q_values.std(dim=0, unbiased=False).view(-1).detach().cpu().numpy()

    q_min = q_min_flat.reshape(num_windows, num_candidates)
    q_mean = q_mean_flat.reshape(num_windows, num_candidates)
    q_max = q_max_flat.reshape(num_windows, num_candidates)
    q_std = q_std_flat.reshape(num_windows, num_candidates)
    # Rank candidates by pessimistic Q (min over the full ensemble) so OOD
    # action overestimation cannot win best-of-n selection.
    ranks = np.empty_like(q_min, dtype=np.int64)
    best_indices = np.empty((num_windows,), dtype=np.int64)
    for window_index in range(num_windows):
        order = np.argsort(-q_min[window_index], kind="mergesort")
        best_indices[window_index] = int(order[0])
        ranks[window_index, order] = np.arange(1, num_candidates + 1, dtype=np.int64)

    rows: list[dict[str, Any]] = []
    for window_index in range(num_windows):
        demo_q_min = float(q_min[window_index, 0])
        for candidate_index, spec in enumerate(specs):
            row = {
                "window_index": int(window_index),
                "step": int(inputs.starts[window_index]),
                "candidate_index": int(candidate_index),
                "candidate_rank_by_q_min": int(ranks[window_index, candidate_index]),
                "is_best": bool(candidate_index == int(best_indices[window_index])),
                "is_demo": bool(candidate_index == 0),
                "q_min": float(q_min[window_index, candidate_index]),
                "q_mean": float(q_mean[window_index, candidate_index]),
                "q_max": float(q_max[window_index, candidate_index]),
                "q_std": float(q_std[window_index, candidate_index]),
                "delta_q_min_vs_demo": float(q_min[window_index, candidate_index] - demo_q_min),
            }
            row.update(spec)
            rows.append(row)

    summary = summarize_q_candidate_rows(
        rows,
        num_windows=num_windows,
        num_candidates=num_candidates,
        noise_sigmas=noise_sigmas,
        random_n=int(random_n),
        single_dim_sigma=float(single_dim_sigma),
        single_dim_n=int(single_dim_n),
        action_low=float(action_low),
        action_high=float(action_high),
        seed=int(seed),
    )
    return rows, summary


def summarize_q_candidate_rows(
    rows: list[dict[str, Any]],
    *,
    num_windows: int,
    num_candidates: int,
    noise_sigmas: list[float],
    random_n: int,
    single_dim_sigma: float,
    single_dim_n: int,
    action_low: float,
    action_high: float,
    seed: int,
) -> dict[str, Any]:
    demo_rows = [row for row in rows if bool(row["is_demo"])]
    best_rows = [row for row in rows if bool(row["is_best"])]
    demo_ranks = np.asarray([int(row["candidate_rank_by_q_min"]) for row in demo_rows], dtype=np.int64)
    best_margin = np.asarray(
        [
            max(float(row["delta_q_min_vs_demo"]) for row in rows if int(row["window_index"]) == window_index)
            for window_index in range(int(num_windows))
        ],
        dtype=np.float32,
    )

    def count_by(key: str, source_rows: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in source_rows:
            value = str(row[key])
            counts[value] = counts.get(value, 0) + 1
        return counts

    def mean_delta_by(key: str) -> dict[str, float]:
        values: dict[str, list[float]] = {}
        for row in rows:
            if bool(row["is_demo"]):
                continue
            value = str(row[key])
            values.setdefault(value, []).append(float(row["delta_q_min_vs_demo"]))
        return {key_value: float(np.mean(items)) for key_value, items in sorted(values.items())}

    return {
        "num_windows": int(num_windows),
        "num_candidates_per_window": int(num_candidates),
        "ranking_score": "q_min",
        "noise_sigmas": [float(item) for item in noise_sigmas],
        "random_n": int(random_n),
        "single_dim_sigma": float(single_dim_sigma),
        "single_dim_n": int(single_dim_n),
        "action_low": float(action_low),
        "action_high": float(action_high),
        "seed": int(seed),
        "demo_rank1_rate": float(np.mean(demo_ranks == 1)) if demo_ranks.size else 0.0,
        "demo_top3_rate": float(np.mean(demo_ranks <= 3)) if demo_ranks.size else 0.0,
        "demo_rank_mean": float(np.mean(demo_ranks)) if demo_ranks.size else 0.0,
        "demo_rank_min": int(demo_ranks.min()) if demo_ranks.size else 0,
        "demo_rank_max": int(demo_ranks.max()) if demo_ranks.size else 0,
        "best_margin_q_min_over_demo_mean": float(best_margin.mean()) if best_margin.size else 0.0,
        "best_margin_q_min_over_demo_max": float(best_margin.max()) if best_margin.size else 0.0,
        "best_candidate_type_counts": count_by("candidate_type", best_rows),
        "best_candidate_group_counts": count_by("candidate_group", best_rows),
        "mean_delta_q_min_by_type": mean_delta_by("candidate_type"),
        "mean_delta_q_min_by_group": mean_delta_by("candidate_group"),
    }


def write_q_candidate_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "window_index",
        "step",
        "candidate_index",
        "candidate_type",
        "candidate_group",
        "candidate_label",
        "candidate_rank_by_q_min",
        "is_best",
        "is_demo",
        "noise_sigma",
        "action_dim",
        "random_index",
        "q_min",
        "q_mean",
        "q_max",
        "q_std",
        "delta_q_min_vs_demo",
    ]
    with path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_q_candidate_rank_hist(path: Path, rows: list[dict[str, Any]]) -> Path:
    demo_ranks = np.asarray(
        [int(row["candidate_rank_by_q_min"]) for row in rows if bool(row["is_demo"])],
        dtype=np.int64,
    )
    if demo_ranks.size == 0:
        raise ValueError("Cannot plot candidate rank histogram with no demo rows.")
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    bins = np.arange(1, int(demo_ranks.max()) + 3) - 0.5
    ax.hist(demo_ranks, bins=bins, color="tab:blue", alpha=0.85)
    ax.set_xlabel("Demo action rank by Q_min")
    ax.set_ylabel("Windows")
    ax.set_title("Best-of-n diagnostic: demo action rank")
    ax.grid(True, alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _save_q_candidate_delta_by_type(path: Path, rows: list[dict[str, Any]]) -> Path:
    groups: dict[str, list[float]] = {}
    for row in rows:
        if bool(row["is_demo"]):
            continue
        groups.setdefault(str(row["candidate_group"]), []).append(float(row["delta_q_min_vs_demo"]))
    if not groups:
        raise ValueError("Cannot plot candidate deltas with no non-demo candidates.")
    labels = list(sorted(groups))
    data = [groups[label] for label in labels]
    fig, ax = plt.subplots(1, 1, figsize=(max(8, len(labels) * 1.3), 5))
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_ylabel("Q_min(candidate) - Q_min(demo)")
    ax.set_title("Counterfactual action Q deltas")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _save_q_candidate_timeseries(path: Path, rows: list[dict[str, Any]]) -> Path:
    by_window: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_window.setdefault(int(row["window_index"]), []).append(row)
    window_ids = sorted(by_window)
    steps = np.asarray([int(by_window[window_id][0]["step"]) for window_id in window_ids], dtype=np.float32)
    demo_q = np.asarray(
        [next(float(row["q_min"]) for row in by_window[window_id] if bool(row["is_demo"])) for window_id in window_ids],
        dtype=np.float32,
    )
    best_rows = [next(row for row in by_window[window_id] if bool(row["is_best"])) for window_id in window_ids]
    best_q = np.asarray([float(row["q_min"]) for row in best_rows], dtype=np.float32)
    demo_rank = np.asarray(
        [
            next(int(row["candidate_rank_by_q_min"]) for row in by_window[window_id] if bool(row["is_demo"]))
            for window_id in window_ids
        ],
        dtype=np.float32,
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(steps, demo_q, label="demo Q_min", color="tab:blue")
    axes[0].plot(steps, best_q, label="best candidate Q_min", color="tab:red", alpha=0.8)
    axes[0].fill_between(steps, demo_q, best_q, color="tab:red", alpha=0.12)
    axes[0].set_ylabel("Q_min")
    axes[0].set_title("Best-of-n diagnostic over rollout windows")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    axes[1].step(steps, demo_rank, where="post", label="demo rank", color="tab:purple")
    axes[1].axhline(1.0, color="black", linewidth=1)
    axes[1].set_ylabel("Rank")
    axes[1].set_xlabel("Step")
    axes[1].invert_yaxis()
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_q_candidate_diagnostics(bon_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Path]:
    bon_dir.mkdir(parents=True, exist_ok=True)
    return {
        "rank_hist": _save_q_candidate_rank_hist(bon_dir / "rank_hist.png", rows),
        "delta_by_type": _save_q_candidate_delta_by_type(bon_dir / "delta_by_type.png", rows),
        "timeseries": _save_q_candidate_timeseries(bon_dir / "timeseries.png", rows),
    }


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
    mc_return = np.asarray([row.get("mc_return", 0.0) for row in metrics], dtype=np.float32)
    value_target = np.asarray([row.get("value_target", row["td_target"]) for row in metrics], dtype=np.float32)
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
    axes[1].plot(steps, value_target, label="value target ((1-λ)TD+λMC)", color="tab:red")
    axes[1].plot(steps, mc_return, label="MC return-to-go", color="tab:brown", alpha=0.7, linestyle="-.")
    axes[1].plot(steps, q_min, label="Q min", color="tab:blue", alpha=0.75)
    axes[1].plot(steps, bootstrap_v, label="γ^H · V(s')", color="tab:green", alpha=0.65, linestyle="--")
    axes[1].plot(steps, total_rewards, label="chunk r_total", color="tab:gray", alpha=0.65, linestyle=":")
    axes[1].set_ylabel("Target / Q")
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # axes[2].plot(steps, td_residual, label="TD residual", color="tab:red")
    axes[2].plot(steps, advantage, label="advantage Q_mean - V", color="tab:brown")
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
    run_bon_diagnostics = str(args.split) == "success_rollout"
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
    print(
        f"[vis_qv] ckpt reward_mode={iql_cfg.reward_mode} "
        f"output_reward_coef={float(iql_cfg.output_reward_coef)} "
        f"disc_reward_coef={float(iql_cfg.disc_reward_coef)}"
    )
    if str(args.split) == "success_rollout" and float(iql_cfg.output_reward_coef) == 0.0:
        print(
            "[vis_qv][WARN] success_rollout is being visualized with output_reward_coef=0. "
            "Q/V scale will be driven only by LPB discriminator reward; successful "
            "trajectories can stay near 0 if LPB scores are already safe."
        )

    selected = select_demo(split_dir, seed=int(args.seed), demo_key=args.demo_key)
    transitions = load_demo_transitions(
        selected,
        camera_names=camera_names,
        image_size=image_size,
        renderer=str(args.renderer),
        control_freq=int(args.control_freq),
        reward_mode=str(iql_cfg.reward_mode),
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

    metrics, per_step_disc, q_candidate_inputs, window_selection = compute_qv_metrics(
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

    bon_outputs: dict[str, Any] | None = None
    q_candidate_summary: dict[str, Any] | None = None
    if run_bon_diagnostics:
        bon_dir = output_dir / "BON"
        bon_dir.mkdir(parents=True, exist_ok=True)
        q_candidate_csv_path = bon_dir / "q_candidates.csv"
        q_candidate_summary_path = bon_dir / "summary.json"
        q_candidate_seed = int(args.seed) if args.q_candidate_seed is None else int(args.q_candidate_seed)
        q_candidate_rows, q_candidate_summary = compute_q_candidate_rows(
            iql=iql,
            inputs=q_candidate_inputs,
            noise_sigmas=parse_float_csv(str(args.q_candidate_noise_sigmas)),
            random_n=int(args.q_candidate_random_n),
            single_dim_sigma=float(args.q_candidate_single_dim_sigma),
            single_dim_n=int(args.q_candidate_single_dim_n),
            seed=q_candidate_seed,
            action_low=float(args.q_candidate_action_low),
            action_high=float(args.q_candidate_action_high),
            batch_size=max(1, int(args.batch_size)),
            device=device,
        )
        write_q_candidate_rows(q_candidate_csv_path, q_candidate_rows)
        q_candidate_plot_paths = plot_q_candidate_diagnostics(bon_dir, q_candidate_rows)
        q_candidate_summary["outputs"] = {
            "candidates_csv": str(q_candidate_csv_path),
            "summary_json": str(q_candidate_summary_path),
            "rank_hist_png": str(q_candidate_plot_paths["rank_hist"]),
            "delta_by_type_png": str(q_candidate_plot_paths["delta_by_type"]),
            "timeseries_png": str(q_candidate_plot_paths["timeseries"]),
        }
        q_candidate_summary_path.write_text(
            json.dumps(q_candidate_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        bon_outputs = {
            "output_dir": str(bon_dir),
            "candidates_csv": str(q_candidate_csv_path),
            "summary": str(q_candidate_summary_path),
            "rank_hist_png": str(q_candidate_plot_paths["rank_hist"]),
            "delta_by_type_png": str(q_candidate_plot_paths["delta_by_type"]),
            "timeseries_png": str(q_candidate_plot_paths["timeseries"]),
        }
    else:
        print(
            f"[vis_qv] skipping Best-of-n Q diagnostics for split={args.split!r} "
            "(only enabled for success_rollout)."
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
        "bon_enabled": bool(run_bon_diagnostics),
        "selected_hdf5": str(selected.hdf5_path),
        "selected_demo_key": selected.demo_key,
        "selected_demo_length": int(selected.length),
        "selected_demo_successful": bool(selected.successful),
        "loaded_transitions": int(len(transitions)),
        "used_windows": int(len(metrics)),
        "candidate_windows": int(window_selection.candidate_windows),
        "dropped_truncated_boundary_windows": int(
            window_selection.dropped_truncated_boundary_windows
        ),
        "dropped_mid_terminal_windows": int(window_selection.dropped_mid_terminal_windows),
        "true_terminal_windows": int(
            sum(float(row.get("is_true_terminal_window", 0.0)) > 0.5 for row in metrics)
        ),
        "camera_names": camera_names,
        "image_size": int(image_size),
        "device": str(device),
        "q_chunking": True,
        "reward_mode": str(iql_cfg.reward_mode),
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
            "bon": bon_outputs,
            "discriminator": disc_viz_outputs,
        },
        "metrics_summary": summarize_metrics(metrics, iql_cfg),
        "q_candidate_summary": q_candidate_summary,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"selected_demo={selected.hdf5_path}::{selected.demo_key}")
    print(f"iql_ckpt={iql_ckpt}")
    print(f"output_dir={output_dir}")
    print(f"video={video_path}")
    print(f"plot_png={plot_paths['overlapping']}")
    print(f"plot_png_nonoverlap={plot_paths['nonoverlap']}")
    if bon_outputs is not None:
        print(f"bon_dir={bon_outputs['output_dir']}")
        print(f"bon_candidates_csv={bon_outputs['candidates_csv']}")
        print(f"bon_summary={bon_outputs['summary']}")
        print(f"bon_rank_hist_png={bon_outputs['rank_hist_png']}")
        print(f"bon_delta_by_type_png={bon_outputs['delta_by_type_png']}")
        print(f"bon_timeseries_png={bon_outputs['timeseries_png']}")
    else:
        print("bon_skipped=1 (split != success_rollout)")
    if disc_viz_outputs is not None:
        print(f"disc_viz_dir={disc_viz_outputs['output_dir']}")
        print(f"disc_viz_video={disc_viz_outputs['video']}")
        print(f"disc_viz_plot_png={disc_viz_outputs['plot_png']}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
