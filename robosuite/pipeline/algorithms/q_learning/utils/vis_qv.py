"""Visualize the nnPU-backed V-only IQL value on one recorded or offline trajectory.

Outputs (per run, under ``<output-root>/<split>_seed<seed>_<ts>/``):
  * ``steps.csv``                    — per-window V / TD-advantage / reward metrics.
  * ``qv_timeseries.png``            — 4-subplot diagnostics over overlapping windows.
  * ``qv_timeseries_nonoverlap.png`` — same plot restricted to disjoint chunks (stride=H).
  * ``rollout_policy_obs.mp4``       — raw policy-camera rollout video.
  * ``discriminator/``               — per-frame nnPU failure scores: CSV + plot + HUD video.
  * ``summary.json``                 — run metadata and output paths.

The value is V-only (no Q head); the per-step advantage is the TD residual
``r + γ^H V(s') - V(s)``. The discriminator path uses the frozen nnPU head.
"""

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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from robosuite.pipeline.algorithms.discriminator.encoder import SharedDynamicsEncoder
from robosuite.pipeline.algorithms.discriminator.nnpu import FrozenNNPUDiscriminator
from robosuite.pipeline.algorithms.q_learning.common import IQLConfig
from robosuite.pipeline.algorithms.q_learning.data_util import (
    aggregate_chunk_reward,
    chunk_done_mask,
)
from robosuite.pipeline.algorithms.q_learning.iql import IQLLearner
from robosuite.pipeline.algorithms.q_learning.utils.vis_discriminator_util import (
    visualize_selected_trajectory_discriminator_nnpu,
    write_rollout_video,
)
from robosuite.pipeline.common.types import Transition
from robosuite.pipeline.train_dipole import load_hdf5_demos_into_flow_transitions
from robosuite.policy.flow_multi_update.utils.env_util import (
    RobosuiteProprioExtractor,
    parse_env_info,
)


DEFAULT_DEMO_ROOT = "data"
# Must match train_dipole.yaml env.img_height when checkpoint omits image_size.
DEFAULT_IMAGE_SIZE = 128
# HUD / MP4 discriminator rollout videos use native dynamics-encoder resolution.
DISC_VIZ_IMAGE_SIZE = 256


def resolve_cli_path(path: str | Path) -> Path:
    """Resolve CLI paths without Hydra original-cwd state."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return Path.cwd() / candidate


@dataclass(frozen=True)
class SelectedDemo:
    hdf5_path: Path
    demo_key: str
    length: int
    successful: bool


@dataclass(frozen=True)
class SelectedOfflineEpisode:
    buffer_path: Path
    episode_index: int
    length: int
    namespace: str
    demo_source: str | None


@dataclass
class PerStepNNPUDisc:
    """Per-frame nnPU failure scores aligned with the recorded trajectory.

    Frame ``t`` carries the score of the action chunk starting at frame ``t``
    (the chunk-start convention used by ``failure_score_start`` in ``steps.csv``).
    Tail frames ``t > T - H`` reuse the last fully-encoded window.
    """

    failure_score: np.ndarray
    intrinsic_reward: np.ndarray
    threshold: float
    pred_failure: np.ndarray

    @property
    def num_frames(self) -> int:
        return int(self.failure_score.shape[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iql-ckpt", required=True)
    parser.add_argument("--disc-ckpt", default=None)
    parser.add_argument("--demo-root", default=DEFAULT_DEMO_ROOT)
    parser.add_argument("--task-data-name", default=None)
    parser.add_argument("--split", default="fail_rollout")
    parser.add_argument(
        "--offline-buffer",
        default=None,
        help=(
            "Optional saved FlowDaggerReplayBuffer transitions (.pt). "
            "If set, --split filters transition.info['episode_namespace']."
        ),
    )
    parser.add_argument("--demo-key", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--output-root", default="outputs/DIPOLE_rl/iql_qv_cache-vis")
    parser.add_argument("--renderer", default=None)
    parser.add_argument("--control-freq", type=int, default=None)
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Override HDF5 resize (H=W). Default: read from IQL checkpoint encoder_meta.",
    )
    parser.add_argument("--no-disc-reward", action="store_true")
    parser.add_argument("--no-disc-viz", action="store_true")
    # Discriminator HUD / rollout video controls.
    parser.add_argument(
        "--disc-viz-image-size",
        type=int,
        default=DISC_VIZ_IMAGE_SIZE,
        help="Square resolution (H=W) for discriminator HUD MP4 frames (default: 256).",
    )
    parser.add_argument("--video-fps", type=int, default=None)
    parser.add_argument("--disc-viz-camera", default=None)
    parser.add_argument("--disc-viz-border-thickness", type=int, default=10)
    parser.add_argument("--no-flip-vertical", action="store_true")
    return parser.parse_args()


def load_iql_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"IQL checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    schema_version = int(payload.get("schema_version", -1))
    if schema_version != 4:
        raise ValueError(
            f"Unsupported IQL checkpoint schema_version={schema_version}. "
            "This visualizer expects the V-only schema (v4); legacy Q-containing "
            "checkpoints (v2/v3) are incompatible — re-run offline V warmup."
        )
    for key in ("iql_state", "cfg", "encoder_meta"):
        if key not in payload:
            raise KeyError(f"IQL checkpoint is missing required key {key!r}")
    return payload


def resolve_device(requested: str | None, checkpoint_device: str) -> str:
    device = str(requested or checkpoint_device or "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device={device}, but CUDA is unavailable")
    return device


@dataclass(frozen=True)
class DemoLoadConfig:
    """HDF5 demo loading settings aligned with IQL warmup / training."""

    img_height: int
    img_width: int
    camera_aliases: dict[str, str]
    reward_mode: str
    renderer: str
    control_freq: int


def resolve_demo_load_config(
    meta: dict[str, Any],
    cfg: IQLConfig,
    *,
    image_size_override: int | None = None,
    renderer_override: str | None = None,
    control_freq_override: int | None = None,
) -> DemoLoadConfig:
    """Resolve demo-loading kwargs to mirror q_learning.warmup data flow."""
    if image_size_override is not None:
        img_height = int(image_size_override)
        img_width = int(image_size_override)
    else:
        img_height = int(meta.get("img_height", meta.get("image_size", DEFAULT_IMAGE_SIZE)))
        img_width = int(meta.get("img_width", img_height))
    aliases_raw = meta.get("camera_aliases", {})
    camera_aliases = {str(k): str(v) for k, v in dict(aliases_raw or {}).items()}
    reward_mode = str(cfg.reward_mode)
    renderer = str(renderer_override or meta.get("renderer", "mjviewer"))
    control_freq = int(control_freq_override or meta.get("control_freq", 20))
    if "image_size" not in meta and "img_height" not in meta and image_size_override is None:
        print(
            f"[vis_qv] warning: checkpoint encoder_meta lacks image_size; "
            f"using default img={img_height}x{img_width} (re-run warmup to persist)."
        )
    return DemoLoadConfig(
        img_height=img_height,
        img_width=img_width,
        camera_aliases=camera_aliases,
        reward_mode=reward_mode,
        renderer=renderer,
        control_freq=control_freq,
    )


def disc_viz_load_config(
    load_cfg: DemoLoadConfig,
    *,
    image_size: int = DISC_VIZ_IMAGE_SIZE,
) -> DemoLoadConfig:
    """Demo load config for discriminator HUD/video frames (default 256px, not training 128)."""
    size = int(image_size)
    return DemoLoadConfig(
        img_height=size,
        img_width=size,
        camera_aliases=dict(load_cfg.camera_aliases),
        reward_mode=str(load_cfg.reward_mode),
        renderer=str(load_cfg.renderer),
        control_freq=int(load_cfg.control_freq),
    )


def align_disc_viz_transitions(
    disc_transitions: list[Transition],
    *,
    viz_end_exclusive: int | None,
) -> list[Transition]:
    """Match Q/V pre-success truncation on the high-res video trajectory."""
    if viz_end_exclusive is None:
        return disc_transitions
    return disc_transitions[: int(viz_end_exclusive)]


def _resize_hwc_image(image: np.ndarray, *, height: int, width: int) -> np.ndarray:
    from PIL import Image

    arr = np.asarray(image, dtype=np.uint8)
    if int(arr.shape[0]) == int(height) and int(arr.shape[1]) == int(width):
        return arr
    return np.asarray(
        Image.fromarray(arr).resize((int(width), int(height)), Image.BILINEAR),
        dtype=np.uint8,
    )


def _resize_obs_dict_images(obs: dict[str, Any], *, height: int, width: int) -> dict[str, Any]:
    resized: dict[str, Any] = {}
    for key, value in obs.items():
        if isinstance(value, np.ndarray) and value.ndim == 3 and int(value.shape[-1]) in (1, 3, 4):
            resized[key] = _resize_hwc_image(value, height=height, width=width)
        else:
            resized[key] = value
    return resized


def upscale_transition_images_for_disc_viz(
    transitions: list[Transition],
    *,
    image_size: int,
) -> list[Transition]:
    """Upscale offline-buffer camera frames when source HDF5 is unavailable."""
    size = int(image_size)
    return [
        Transition(
            obs=_resize_obs_dict_images(dict(transition.obs), height=size, width=size),  # type: ignore[arg-type]
            action=transition.action,
            reward=transition.reward,
            next_obs=_resize_obs_dict_images(dict(transition.next_obs), height=size, width=size),  # type: ignore[arg-type]
            done=transition.done,
            grasp_penalty=transition.grasp_penalty,
            is_intervention=transition.is_intervention,
            info=transition.info,
            reward_source=transition.reward_source,
            demo_source=transition.demo_source,
        )
        for transition in transitions
    ]


def load_offline_disc_viz_transitions(
    transitions: list[Transition],
    *,
    camera_names: list[str],
    load_cfg: DemoLoadConfig,
    viz_end_exclusive: int | None,
    image_size: int = DISC_VIZ_IMAGE_SIZE,
) -> list[Transition]:
    """Load 256px discriminator-video frames for an offline-buffer episode."""
    if not transitions:
        return transitions
    disc_viz_load_cfg = disc_viz_load_config(load_cfg, image_size=int(image_size))
    first_info = transitions[0].info or {}
    hdf5_raw = first_info.get("source_hdf5_path")
    demo_name = first_info.get("demo_name")
    if hdf5_raw and demo_name:
        hdf5_path = Path(str(hdf5_raw))
        if hdf5_path.exists():
            selected = SelectedDemo(
                hdf5_path=hdf5_path,
                demo_key=str(demo_name),
                length=len(transitions),
                successful=bool(first_info.get("demo_success_attr", False)),
            )
            try:
                disc_transitions = load_demo(
                    selected,
                    camera_names=camera_names,
                    load_cfg=disc_viz_load_cfg,
                )
            except KeyError as exc:
                print(
                    f"[vis_qv] disc_viz: source HDF5 reload failed ({hdf5_path}: {exc}); "
                    f"upscaling buffer frames to {disc_viz_load_cfg.img_height}x"
                    f"{disc_viz_load_cfg.img_width}"
                )
            else:
                disc_transitions = align_disc_viz_transitions(
                    disc_transitions,
                    viz_end_exclusive=viz_end_exclusive,
                )
                if viz_end_exclusive is None and len(disc_transitions) > len(transitions):
                    disc_transitions = disc_transitions[: len(transitions)]
                return disc_transitions
        else:
            print(
                f"[vis_qv] disc_viz: source HDF5 missing ({hdf5_path}); "
                f"upscaling buffer frames to {disc_viz_load_cfg.img_height}x"
                f"{disc_viz_load_cfg.img_width}"
            )
    else:
        print(
            f"[vis_qv] disc_viz: offline episode lacks source_hdf5_path/demo_name; "
            f"upscaling buffer frames to {disc_viz_load_cfg.img_height}x"
            f"{disc_viz_load_cfg.img_width}"
        )
    return upscale_transition_images_for_disc_viz(
        transitions,
        image_size=int(disc_viz_load_cfg.img_height),
    )


def is_success_related_split(split: str) -> bool:
    """True for rollout splits that may contain post-success padding (e.g. success_rollout)."""
    return "success" in str(split).lower()


def load_demo_is_success(selected: SelectedDemo) -> np.ndarray | None:
    """Per-frame flag: True if task success already occurred before this step."""
    with h5py.File(selected.hdf5_path, "r") as handle:
        root = handle["demos"] if "demos" in handle else handle["data"]
        group = root[selected.demo_key]
        if "is_success" not in group:
            return None
        return np.asarray(group["is_success"][:], dtype=bool)


def pre_success_exclusive_end(is_success: np.ndarray | None, num_frames: int) -> int:
    """Exclusive end index before the first success frame."""
    if is_success is None:
        return int(num_frames)
    mask = np.asarray(is_success, dtype=bool).reshape(-1)
    n = int(min(int(num_frames), int(mask.shape[0])))
    if n <= 0:
        return 0
    mask = mask[:n]
    if not mask.any():
        return n
    return int(np.argmax(mask))


def round_success_viz_end_to_chunk(
    pre_success_end: int,
    *,
    num_frames: int,
    action_horizon: int,
) -> int:
    """Round the pre-success cutoff up to a chunk boundary, clamped to trajectory length."""
    horizon = int(action_horizon)
    if horizon <= 0:
        raise ValueError(f"action_horizon must be positive, got {action_horizon}")
    raw = max(0, int(pre_success_end))
    rounded = ((raw + horizon - 1) // horizon) * horizon
    return int(min(max(0, rounded), int(num_frames)))


def truncate_transitions_for_success_viz(
    transitions: list[Transition],
    *,
    split: str,
    selected: SelectedDemo,
    action_horizon: int,
) -> tuple[list[Transition], int | None]:
    """Drop post-success frames for success-related HDF5 splits (viz only)."""
    if not is_success_related_split(split):
        return transitions, None
    is_success = load_demo_is_success(selected)
    pre_success_end = pre_success_exclusive_end(is_success, len(transitions))
    viz_end = round_success_viz_end_to_chunk(
        pre_success_end,
        num_frames=len(transitions),
        action_horizon=int(action_horizon),
    )
    if viz_end >= len(transitions):
        return transitions, None
    if viz_end < int(action_horizon):
        raise RuntimeError(
            f"Rounded pre-success segment has {viz_end} frames "
            f"(raw pre_success_end={pre_success_end}), but action_horizon={action_horizon} "
            f"requires at least {action_horizon} frames for Q/V windows."
        )
    print(
        f"[vis_qv] success split: truncate viz to chunk-rounded pre-success frames "
        f"[0, {viz_end}) / {len(transitions)} (raw pre_success_end={pre_success_end})"
    )
    return transitions[:viz_end], int(viz_end)


def truncate_offline_transitions_for_success_viz(
    transitions: list[Transition],
    *,
    split: str,
    action_horizon: int,
) -> tuple[list[Transition], int | None]:
    """Drop post-success offline frames, keeping a chunk-rounded prefix."""
    if not is_success_related_split(split):
        return transitions, None
    success = np.asarray(
        [bool((transition.info or {}).get("success", False)) for transition in transitions],
        dtype=bool,
    )
    pre_success_end = pre_success_exclusive_end(success, len(transitions))
    viz_end = round_success_viz_end_to_chunk(
        pre_success_end,
        num_frames=len(transitions),
        action_horizon=int(action_horizon),
    )
    if viz_end >= len(transitions):
        return transitions, None
    if viz_end < int(action_horizon):
        raise RuntimeError(
            f"Rounded pre-success segment has {viz_end} frames "
            f"(raw pre_success_end={pre_success_end}), but action_horizon={action_horizon} "
            f"requires at least {action_horizon} frames for Q/V windows."
        )
    print(
        f"[vis_qv] offline success split: truncate viz to chunk-rounded pre-success frames "
        f"[0, {viz_end}) / {len(transitions)} (raw pre_success_end={pre_success_end})"
    )
    return transitions[:viz_end], int(viz_end)


def select_demo(split_dir: Path, *, seed: int, demo_key: str | None) -> SelectedDemo:
    candidates: list[SelectedDemo] = []
    for path in sorted(split_dir.glob("*.hdf5")) + sorted(split_dir.glob("*.h5")):
        with h5py.File(path, "r") as handle:
            root = handle["demos"] if "demos" in handle else handle["data"]
            for key in sorted(root.keys()):
                if demo_key is not None and str(key) != str(demo_key):
                    continue
                group = root[key]
                candidates.append(
                    SelectedDemo(
                        path,
                        str(key),
                        int(group.attrs.get("length", len(group["actions"]))),
                        bool(group.attrs.get("successful", False)),
                    )
                )
    if not candidates:
        raise FileNotFoundError(f"No matching HDF5 demo found under {split_dir}")
    return random.Random(int(seed)).choice(candidates)


def _transition_episode_index(transition: Transition) -> int | None:
    info = transition.info or {}
    if "episode_index" not in info:
        return None
    return int(info["episode_index"])


def _transition_episode_step(transition: Transition) -> int:
    return int((transition.info or {}).get("episode_step", 0))


def _transition_namespace(transition: Transition) -> str:
    return str((transition.info or {}).get("episode_namespace", ""))


def select_offline_episode(
    buffer_path: Path,
    *,
    split: str,
    seed: int,
    demo_key: str | None,
    action_horizon: int,
) -> tuple[SelectedOfflineEpisode, list[Transition]]:
    """Select one split-filtered episode from a saved replay-buffer state dict."""
    if not buffer_path.exists():
        raise FileNotFoundError(f"Offline buffer does not exist: {buffer_path}")
    state_dict = torch.load(buffer_path, map_location="cpu", weights_only=False)
    storage = list(state_dict.get("storage", []))
    if not storage:
        raise RuntimeError(f"Offline buffer contains zero transitions: {buffer_path}")

    episodes: dict[int, list[Transition]] = {}
    for transition in storage:
        episode_index = _transition_episode_index(transition)
        if episode_index is None:
            continue
        episodes.setdefault(episode_index, []).append(transition)
    if not episodes:
        raise RuntimeError(
            f"Offline buffer has no transition.info['episode_index']: {buffer_path}"
        )

    split_key = str(split).lower()
    valid: dict[int, list[Transition]] = {}
    for episode_index, transitions in episodes.items():
        ordered = sorted(transitions, key=_transition_episode_step)
        if len(ordered) < int(action_horizon):
            continue
        namespace = _transition_namespace(ordered[0]).lower()
        if split_key and split_key not in namespace:
            continue
        valid[int(episode_index)] = ordered
    if not valid:
        raise RuntimeError(
            f"Offline buffer has no episode matching split={split!r} with at least "
            f"action_horizon={int(action_horizon)} transitions: {buffer_path}"
        )

    if demo_key is not None:
        try:
            selected_index = int(str(demo_key))
        except ValueError as exc:
            raise ValueError("--demo-key must be an integer episode_index in --offline-buffer mode") from exc
        if selected_index not in valid:
            raise KeyError(
                f"Requested episode_index={selected_index} is missing, too short, "
                f"or not in split={split!r}: {buffer_path}"
            )
    else:
        selected_index = random.Random(int(seed)).choice(sorted(valid))

    transitions = valid[selected_index]
    first = transitions[0]
    selected = SelectedOfflineEpisode(
        buffer_path=buffer_path,
        episode_index=int(selected_index),
        length=len(transitions),
        namespace=_transition_namespace(first),
        demo_source=None if first.demo_source is None else str(first.demo_source),
    )
    return selected, transitions


def load_demo(
    selected: SelectedDemo,
    *,
    camera_names: list[str],
    load_cfg: DemoLoadConfig,
) -> list[Transition]:
    with h5py.File(selected.hdf5_path, "r") as handle:
        env_info = parse_env_info(handle.attrs["env_info"])
    extractor = RobosuiteProprioExtractor(
        env_info,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )
    try:
        transitions = load_hdf5_demos_into_flow_transitions(
            selected.hdf5_path,
            policy_camera_names=camera_names,
            camera_aliases=dict(load_cfg.camera_aliases),
            img_height=int(load_cfg.img_height),
            img_width=int(load_cfg.img_width),
            proprio_keys=(),
            renderer=str(load_cfg.renderer),
            control_freq=int(load_cfg.control_freq),
            demo_names=[selected.demo_key],
            state_extractor=extractor,
            reward_mode=str(load_cfg.reward_mode),
        )
    finally:
        extractor.close()
    if not transitions:
        raise RuntimeError("Selected demo produced zero transitions")
    return transitions


def _stack_views(obs: dict[str, Any], camera_names: list[str]) -> np.ndarray:
    return np.stack(
        [np.asarray(obs[name], dtype=np.uint8).transpose(2, 0, 1) for name in camera_names]
    )


def _image_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array)).float().div_(255.0)


def build_models(
    payload: dict[str, Any], *, device: str, disc_override: str | None
) -> tuple[IQLLearner, SharedDynamicsEncoder, FrozenNNPUDiscriminator, IQLConfig, dict[str, Any]]:
    meta = dict(payload["encoder_meta"])
    nnpu_path = disc_override or meta.get("nnpu_checkpoint")
    if not nnpu_path:
        raise KeyError("encoder_meta.nnpu_checkpoint is missing; pass --disc-ckpt")
    camera_names = [str(name) for name in meta.get("policy_camera_names", [])]
    if not camera_names:
        raise KeyError("encoder_meta.policy_camera_names is missing")
    encoder = SharedDynamicsEncoder(nnpu_ckpt_path=nnpu_path, device=device)
    encoder.bind_policy_cameras(camera_names)
    task_name = str(meta.get("nnpu_task") or meta.get("task_env") or meta.get("task"))
    discriminator = FrozenNNPUDiscriminator(
        nnpu_ckpt_path=nnpu_path, task_name=task_name, device=device, encoder=encoder
    )
    cfg_dict = dict(payload["cfg"])
    cfg_dict["device"] = device
    cfg = IQLConfig(**cfg_dict)
    action_dim = int(meta.get("policy_action_dim", payload["iql_state"].get("action_dim", 0)))
    learner = IQLLearner(
        cfg,
        state_feature_dim=encoder.state_feature_dim,
        chunk_feature_dim=encoder.chunk_feature_dim,
        action_dim=action_dim,
        n_tokens=int(encoder.inner_encoder.num_patches),
        proprio_dim=int(encoder.inner_encoder.proprio_emb_dim),
    )
    learner.load_state_dict(payload["iql_state"], strict=True)
    learner.v.eval()
    learner.target_v.eval()
    return learner, encoder, discriminator, cfg, meta


@torch.no_grad()
def compute_metrics(
    transitions: list[Transition],
    *,
    learner: IQLLearner,
    encoder: SharedDynamicsEncoder,
    discriminator: FrozenNNPUDiscriminator,
    cfg: IQLConfig,
    camera_names: list[str],
    batch_size: int,
    max_windows: int | None,
    use_disc_reward: bool,
) -> tuple[list[dict[str, float]], PerStepNNPUDisc]:
    """Compute per-window V/TD metrics and a per-frame nnPU failure series.

    The per-frame discriminator series reuses the per-frame chunk features that
    are already encoded for every sliding window, so it costs no extra encoder
    passes — only additional frozen-head evaluations.
    """
    horizon = int(cfg.action_horizon)
    threshold = float(discriminator.threshold)
    starts = list(range(max(0, len(transitions) - horizon + 1)))
    if max_windows is not None:
        starts = starts[: int(max_windows)]
    if not starts:
        raise RuntimeError(f"Need at least {horizon} transitions, got {len(transitions)}")
    num_windows = len(starts)
    sequences = [transitions[start : start + horizon] for start in starts]
    # First task-success frame (if any). Windows whose s' lands on/after this
    # index have no continuing bootstrap state — V(s') is terminal/OOD.
    success_flags = [
        bool((item.info or {}).get("success", False)) for item in transitions
    ]
    first_success_idx = next(
        (idx for idx, flag in enumerate(success_flags) if flag),
        None,
    )
    images_np = np.stack(
        [np.stack([_stack_views(item.obs, camera_names) for item in seq]) for seq in sequences]
    )
    proprio_cpu = torch.from_numpy(
        np.stack([[np.asarray(item.obs["state"], np.float32) for item in seq] for seq in sequences])
    ).float()
    actions_cpu = torch.from_numpy(
        np.stack([[np.asarray(item.action, np.float32) for item in seq] for seq in sequences])
    ).float()
    rewards_cpu = torch.tensor(
        [[float(item.reward or 0.0) for item in seq] for seq in sequences], dtype=torch.float32
    )
    dones_cpu = torch.tensor(
        [[float(bool((item.info or {}).get("success", False))) for item in seq] for seq in sequences],
        dtype=torch.float32,
    )
    # Per-frame nnPU traces for every window: (W, H).
    disc_failure_wh = np.zeros((num_windows, horizon), dtype=np.float32)
    disc_intrinsic_wh = np.zeros((num_windows, horizon), dtype=np.float32)
    rows: list[dict[str, float]] = []
    bootstrap_discount = float(cfg.discount) ** horizon
    batch_step = max(1, int(batch_size))
    batch_offsets = range(0, len(starts), batch_step)
    for offset in tqdm(
        batch_offsets,
        desc="[vis_qv] Q/V metrics",
        unit="batch",
        total=(len(starts) + batch_step - 1) // batch_step,
    ):
        batch_starts = starts[offset : offset + batch_step]
        batch_images_np = images_np[offset : offset + len(batch_starts)]
        proprio = proprio_cpu[offset : offset + len(batch_starts)]
        actions = actions_cpu[offset : offset + len(batch_starts)]
        rewards = rewards_cpu[offset : offset + len(batch_starts)]
        # Bootstrap terminal mask, mirroring IQL training (replay._build_step_batch):
        # only a task-success frame is a true terminal. A truncated (e.g.
        # fail_rollout) trajectory end keeps the γ^H·V(s') bootstrap, so the tail
        # chunks' TD target stays on the same scale as the interior windows.
        dones = dones_cpu[offset : offset + len(batch_starts)]
        batch, _, views, channels, height, width = batch_images_np.shape
        images = _image_tensor(batch_images_np.reshape(batch * horizon, views, channels, height, width))
        images = images.view(batch, horizon, views, channels, height, width)
        state_features, chunk_features = encoder.encode_features(
            chunk_images=images, chunk_proprio=proprio, chunk_actions=actions
        )
        next_obs = [
            transitions[start + horizon].obs
            if start + horizon < len(transitions)
            else transitions[start + horizon - 1].next_obs
            for start in batch_starts
        ]
        next_state = encoder.encode_state(
            image_obs_raw=_image_tensor(np.stack([_stack_views(obs, camera_names) for obs in next_obs])),
            proprio_raw=torch.from_numpy(
                np.stack([np.asarray(obs["state"], np.float32) for obs in next_obs])
            ).float(),
        )
        # Per-frame nnPU traces over the full chunk: (B, H).
        failure_all = discriminator.failure_score(chunk_feature=chunk_features)
        intrinsic_all = -torch.sigmoid(failure_all - threshold)
        disc_failure_wh[offset : offset + batch] = failure_all.detach().cpu().numpy()
        disc_intrinsic_wh[offset : offset + batch] = intrinsic_all.detach().cpu().numpy()
        disc_steps = (
            intrinsic_all.detach().cpu()
            if use_disc_reward
            else torch.zeros_like(rewards)
        )
        env_aggregated = aggregate_chunk_reward(rewards, float(cfg.discount))
        disc_aggregated = aggregate_chunk_reward(disc_steps, float(cfg.discount))
        total_steps = float(cfg.output_reward_coef) * rewards + float(cfg.disc_reward_coef) * disc_steps
        aggregated = aggregate_chunk_reward(total_steps, float(cfg.discount)).to(learner.cfg.device)
        done = chunk_done_mask(dones).to(learner.cfg.device)
        v = learner.v(state_features[:, 0])
        next_v = learner.target_v(next_state)
        bootstrap_v = bootstrap_discount * (1.0 - done) * next_v
        td_target = aggregated + bootstrap_v
        for index, start in enumerate(batch_starts):
            v_val = float(v[index].item())
            td_target_val = float(td_target[index].item())
            done_flag = float(done[index].item()) > 0.5
            next_idx = int(start) + horizon
            # Valid continuing s' only when bootstrap is used and s' is still a
            # pre-success in-buffer frame. Otherwise V(s') is terminal/OOD and
            # must not be plotted (raw values still stored for CSV).
            if done_flag:
                has_valid_next = False
            elif first_success_idx is not None and next_idx >= int(first_success_idx):
                has_valid_next = False
            elif next_idx >= len(transitions):
                # Traj-end fallback next_obs: keep for fail (matches training
                # truncation bootstrap); success path already rejected above.
                has_valid_next = True
            else:
                has_valid_next = True
            rows.append(
                {
                    "window_start": float(start),
                    "step": float(start),
                    "v": v_val,
                    "next_v": float(next_v[index].item()),
                    "bootstrap_v": float(bootstrap_v[index].item()),
                    "advantage_td1": td_target_val - v_val,
                    "td_target": td_target_val,
                    "env_reward_horizon": float(env_aggregated[index].item()),
                    "disc_reward_horizon": float(disc_aggregated[index].item()),
                    "total_reward_horizon": float(aggregated[index].item()),
                    "disc_intrinsic_step0": float(disc_intrinsic_wh[offset + index, 0]),
                    "failure_score_start": float(disc_failure_wh[offset + index, 0]),
                    "done_chunk": float(done[index].item()),
                    "has_valid_next": float(has_valid_next),
                }
            )
    per_step_disc = _per_step_disc_from_windows(
        starts=starts,
        horizon=horizon,
        num_transitions=len(transitions),
        disc_failure_wh=disc_failure_wh,
        disc_intrinsic_wh=disc_intrinsic_wh,
        threshold=threshold,
    )
    return rows, per_step_disc


def _per_step_disc_from_windows(
    *,
    starts: list[int],
    horizon: int,
    num_transitions: int,
    disc_failure_wh: np.ndarray,
    disc_intrinsic_wh: np.ndarray,
    threshold: float,
) -> PerStepNNPUDisc:
    """Map per-window ``(W, H)`` traces to a per-frame series of length ``L``.

    Frame ``t`` uses window ``start = min(t, last_start)`` and offset ``t - start``
    (``starts`` is contiguous from 0, so the window index equals ``start``). This
    matches the chunk-start position recorded for ``step == t`` in ``steps.csv``.
    """
    last_start = int(starts[-1])
    length = min(int(num_transitions), last_start + int(horizon))
    failure = np.zeros((length,), dtype=np.float32)
    intrinsic = np.zeros((length,), dtype=np.float32)
    for step in range(length):
        start = min(int(step), last_start)
        offset = int(step) - start
        failure[step] = float(disc_failure_wh[start, offset])
        intrinsic[step] = float(disc_intrinsic_wh[start, offset])
    pred_failure = (failure >= float(threshold)).astype(np.int64)
    return PerStepNNPUDisc(
        failure_score=failure,
        intrinsic_reward=intrinsic,
        threshold=float(threshold),
        pred_failure=pred_failure,
    )


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
    per_step_disc: PerStepNNPUDisc | None = None,
) -> Path:
    """Render the 4-subplot Q/V diagnostics (layout from dipole-rl/v0-kingback)."""
    if not metrics:
        raise ValueError("Cannot plot Q/V timeseries with empty metrics.")
    steps = np.asarray([row["step"] for row in metrics], dtype=np.float32)
    v = np.asarray([row["v"] for row in metrics], dtype=np.float32)
    next_v = np.asarray([row["next_v"] for row in metrics], dtype=np.float32)
    td_target = np.asarray([row["td_target"] for row in metrics], dtype=np.float32)
    advantage_td1 = np.asarray([row["advantage_td1"] for row in metrics], dtype=np.float32)
    bootstrap_v = np.asarray([row["bootstrap_v"] for row in metrics], dtype=np.float32)
    env_rewards = np.asarray([row["env_reward_horizon"] for row in metrics], dtype=np.float32)
    total_rewards = np.asarray([row["total_reward_horizon"] for row in metrics], dtype=np.float32)
    disc_rewards = np.asarray([row["disc_reward_horizon"] for row in metrics], dtype=np.float32)
    disc_step0 = np.asarray([row["disc_intrinsic_step0"] for row in metrics], dtype=np.float32)

    # No continuing s' (done=1 bootstrap mask, or s' on/after first success):
    # hide every V(s')-dependent curve so terminal/OOD next_v ≈ 0 does not
    # crush the y-axis. Raw values remain in steps.csv (has_valid_next).
    if any("has_valid_next" in row for row in metrics):
        no_s_prime = np.asarray(
            [float(row.get("has_valid_next", 1.0)) for row in metrics], dtype=np.float32
        ) < 0.5
    else:
        # Backward-compat for older steps.csv without has_valid_next.
        no_s_prime = (
            np.asarray([row.get("done_chunk", 0.0) for row in metrics], dtype=np.float32) > 0.5
        )
    if bool(no_s_prime.any()):
        next_v = next_v.copy()
        td_target = td_target.copy()
        advantage_td1 = advantage_td1.copy()
        bootstrap_v = bootstrap_v.copy()
        next_v[no_s_prime] = np.nan
        td_target[no_s_prime] = np.nan
        advantage_td1[no_s_prime] = np.nan
        bootstrap_v[no_s_prime] = np.nan

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(title)

    axes[0].plot(steps, v, label="V", color="tab:orange")
    axes[0].plot(steps, next_v, label="target next V", color="tab:green", alpha=0.8)
    axes[0].set_ylabel("V")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    # γ^H from valid-bootstrap windows only (nan next_v already excluded).
    safe = (~no_s_prime) & np.isfinite(next_v) & (np.abs(next_v) > 1e-8)
    gamma_h = (
        float(np.median(bootstrap_v[safe] / next_v[safe])) if bool(safe.any()) else 1.0
    )
    g_next_v = gamma_h * next_v
    axes[1].plot(steps, next_v - v, label="V(s') - V(s)", color="tab:purple")
    axes[1].plot(
        steps,
        g_next_v - v,
        label="γ^H V(s') - V(s)",
        color="tab:green",
        alpha=0.85,
    )
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_ylabel("V diff")
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(
        steps,
        advantage_td1,
        label="advantage TD (Σγ^i r + γ^H V' - V)",
        color="tab:red",
        alpha=0.85,
    )
    axes[2].axhline(0.0, color="black", linewidth=1)
    axes[2].set_ylabel("Advantage")
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
    axes[3].set_xlabel("step (window start)")
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
    per_step_disc: PerStepNNPUDisc | None = None,
) -> dict[str, Path]:
    """Write overlapping-window and non-overlapping-chunk Q/V plots (PNG)."""
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


def write_metrics_csv(csv_path: Path, rows: list[dict[str, float]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    checkpoint = resolve_cli_path(args.iql_ckpt)
    payload = load_iql_payload(checkpoint)
    device = resolve_device(args.device, str(payload["cfg"].get("device", "cpu")))
    learner, encoder, discriminator, cfg, meta = build_models(
        payload, device=device, disc_override=args.disc_ckpt
    )
    task = str(args.task_data_name or meta.get("task") or meta.get("task_env"))
    camera_names = [str(name) for name in meta["policy_camera_names"]]
    load_cfg = resolve_demo_load_config(
        meta,
        cfg,
        image_size_override=args.image_size,
        renderer_override=args.renderer,
        control_freq_override=args.control_freq,
    )
    print(
        f"[vis_qv] demo_load img={load_cfg.img_height}x{load_cfg.img_width} "
        f"reward_mode={load_cfg.reward_mode} renderer={load_cfg.renderer} "
        f"control_freq={load_cfg.control_freq} camera_aliases={load_cfg.camera_aliases or '{}'}"
    )

    selected_hdf5: str | None = None
    selected_demo_key: str | None = None
    selected_offline_buffer: str | None = None
    selected_episode_index: int | None = None
    selected_demo_source: str | None = None
    selected_namespace: str | None = None
    viz_end_exclusive: int | None = None
    disc_viz_img_height = int(load_cfg.img_height)
    disc_viz_img_width = int(load_cfg.img_width)

    if args.offline_buffer:
        offline_buffer = resolve_cli_path(args.offline_buffer)
        selected_offline, transitions = select_offline_episode(
            offline_buffer,
            split=str(args.split),
            seed=args.seed,
            demo_key=args.demo_key,
            action_horizon=int(cfg.action_horizon),
        )
        transitions, viz_end_exclusive = truncate_offline_transitions_for_success_viz(
            transitions,
            split=str(args.split),
            action_horizon=int(cfg.action_horizon),
        )
        disc_viz_load_cfg = disc_viz_load_config(
            load_cfg,
            image_size=int(args.disc_viz_image_size),
        )
        disc_viz_transitions = load_offline_disc_viz_transitions(
            transitions,
            camera_names=camera_names,
            load_cfg=load_cfg,
            viz_end_exclusive=viz_end_exclusive,
            image_size=int(args.disc_viz_image_size),
        )
        disc_viz_img_height = int(disc_viz_load_cfg.img_height)
        disc_viz_img_width = int(disc_viz_load_cfg.img_width)
        selected_offline_buffer = str(selected_offline.buffer_path)
        selected_episode_index = int(selected_offline.episode_index)
        selected_demo_source = selected_offline.demo_source
        selected_namespace = selected_offline.namespace
        print(
            f"[vis_qv] offline_buffer={selected_offline.buffer_path} "
            f"episode_index={selected_offline.episode_index} length={selected_offline.length} "
            f"namespace={selected_offline.namespace!r} demo_source={selected_offline.demo_source or '<unknown>'}"
        )
        print(
            f"[vis_qv] disc_viz_frames img={disc_viz_img_height}x"
            f"{disc_viz_img_width} num_frames={len(disc_viz_transitions)}"
        )
    else:
        split_dir = resolve_cli_path(args.demo_root) / task / str(args.split)
        selected = select_demo(split_dir, seed=args.seed, demo_key=args.demo_key)
        transitions = load_demo(
            selected,
            camera_names=camera_names,
            load_cfg=load_cfg,
        )
        transitions, viz_end_exclusive = truncate_transitions_for_success_viz(
            transitions,
            split=str(args.split),
            selected=selected,
            action_horizon=int(cfg.action_horizon),
        )
        disc_viz_load_cfg = disc_viz_load_config(
            load_cfg,
            image_size=int(args.disc_viz_image_size),
        )
        disc_viz_img_height = int(disc_viz_load_cfg.img_height)
        disc_viz_img_width = int(disc_viz_load_cfg.img_width)
        disc_viz_transitions = align_disc_viz_transitions(
            load_demo(
                selected,
                camera_names=camera_names,
                load_cfg=disc_viz_load_cfg,
            ),
            viz_end_exclusive=viz_end_exclusive,
        )
        selected_hdf5 = str(selected.hdf5_path)
        selected_demo_key = selected.demo_key
        selected_namespace = str(args.split)
        print(
            f"[vis_qv] disc_viz_frames img={disc_viz_load_cfg.img_height}x"
            f"{disc_viz_load_cfg.img_width} num_frames={len(disc_viz_transitions)}"
        )

    rows, per_step_disc = compute_metrics(
        transitions,
        learner=learner,
        encoder=encoder,
        discriminator=discriminator,
        cfg=cfg,
        camera_names=camera_names,
        batch_size=args.batch_size,
        max_windows=args.max_windows,
        use_disc_reward=not args.no_disc_reward,
    )
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = (
        resolve_cli_path(args.output_root)
        / f"{args.split}_seed{args.seed}_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    video_fps = int(
        args.video_fps if args.video_fps is not None else load_cfg.control_freq
    )
    flip_vertical = not bool(args.no_flip_vertical)

    write_metrics_csv(output_dir / "steps.csv", rows)
    plot_paths = plot_qv(
        output_dir / "qv_timeseries",
        rows,
        title=(
            f"{task} {args.split} offline episode {selected_episode_index}"
            if selected_episode_index is not None
            else f"{task} {args.split} {selected_hdf5}::{selected_demo_key}"
        ),
        action_horizon=int(cfg.action_horizon),
        per_step_disc=per_step_disc,
    )
    rollout_video = write_rollout_video(
        output_dir / "rollout_policy_obs.mp4",
        transitions,
        camera_names=camera_names,
        camera_name=args.disc_viz_camera,
        fps=video_fps,
        flip_vertical=flip_vertical,
    )

    disc_viz_outputs: dict[str, Any] | None = None
    if not bool(args.no_disc_viz):
        disc_result = visualize_selected_trajectory_discriminator_nnpu(
            output_dir=output_dir / "discriminator",
            transitions=disc_viz_transitions,
            camera_names=camera_names,
            failure_score=per_step_disc.failure_score,
            intrinsic_reward=per_step_disc.intrinsic_reward,
            pred_failure=per_step_disc.pred_failure,
            threshold=per_step_disc.threshold,
            ckpt_path=discriminator.ckpt_path,
            task_name=discriminator.task_name,
            video_fps=video_fps,
            camera_name=args.disc_viz_camera,
            border_thickness=int(args.disc_viz_border_thickness),
            flip_vertical=flip_vertical,
        )
        disc_viz_outputs = {
            "output_dir": str(disc_result.output_dir),
            "scores_csv": str(disc_result.scores_csv),
            "plot_png": str(disc_result.plot_png),
            "plot_pdf": str(disc_result.plot_pdf),
            "video": str(disc_result.video),
            "summary": str(disc_result.summary_json),
        }

    first_pred = np.where(per_step_disc.pred_failure.astype(bool))[0]
    summary = {
        "schema_version": 2,
        "iql_checkpoint": str(checkpoint),
        "nnpu_checkpoint": discriminator.ckpt_path,
        "threshold": float(discriminator.threshold),
        "threshold_source": "checkpoint",
        "state_feature_dim": int(encoder.state_feature_dim),
        "chunk_feature_dim": int(encoder.chunk_feature_dim),
        "split": str(args.split),
        "selected_hdf5": selected_hdf5,
        "selected_demo_key": selected_demo_key,
        "selected_offline_buffer": selected_offline_buffer,
        "selected_episode_index": selected_episode_index,
        "selected_demo_source": selected_demo_source,
        "selected_namespace": selected_namespace,
        "viz_end_exclusive": viz_end_exclusive,
        "num_transitions_viz": len(transitions),
        "num_windows": len(rows),
        "demo_load": {
            "img_height": int(load_cfg.img_height),
            "img_width": int(load_cfg.img_width),
            "reward_mode": str(load_cfg.reward_mode),
            "renderer": str(load_cfg.renderer),
            "control_freq": int(load_cfg.control_freq),
            "camera_aliases": dict(load_cfg.camera_aliases),
        },
        "disc_viz": {
            "img_height": int(disc_viz_img_height),
            "img_width": int(disc_viz_img_width),
            "num_frames": len(disc_viz_transitions),
        },
        "per_step_disc_frames": int(per_step_disc.num_frames),
        "first_pred_failure_frame": None if first_pred.size == 0 else int(first_pred[0]),
        "device": device,
        "outputs": {
            "steps_csv": str(output_dir / "steps.csv"),
            "plot_png": str(plot_paths["overlapping"]),
            "plot_png_nonoverlap": str(plot_paths["nonoverlap"]),
            "video": None if rollout_video is None else str(rollout_video),
            "discriminator": disc_viz_outputs,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[vis_qv] wrote {output_dir}")
    print(f"[vis_qv] plot_png={plot_paths['overlapping']}")
    print(f"[vis_qv] plot_png_nonoverlap={plot_paths['nonoverlap']}")
    if rollout_video is not None:
        print(f"[vis_qv] rollout_video={rollout_video}")
    if disc_viz_outputs is not None:
        print(f"[vis_qv] disc_viz_dir={disc_viz_outputs['output_dir']}")
        print(f"[vis_qv] disc_viz_video={disc_viz_outputs['video']}")


if __name__ == "__main__":
    main()
