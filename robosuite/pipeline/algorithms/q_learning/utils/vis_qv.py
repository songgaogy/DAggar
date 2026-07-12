"""Visualize nnPU-backed IQL Q/V values on one recorded or offline trajectory.

Outputs (per run, under ``<output-root>/<task>_iql-qv/<split>_seed<seed>_<ts>/``):
  * ``steps.csv``                    — per-window Q/V/advantage/reward metrics.
  * ``qv_timeseries.png``            — 4-subplot diagnostics over overlapping windows.
  * ``qv_timeseries_nonoverlap.png`` — same plot restricted to disjoint chunks (stride=H).
  * ``rollout_policy_obs.mp4``       — raw policy-camera rollout video.
  * ``BON/``                         — best-of-n counterfactual Q diagnostics.
  * ``discriminator/``               — per-frame nnPU failure scores: CSV + plot + HUD video.
  * ``summary.json``                 — run metadata and output paths.

The 4-subplot layout and metric semantics mirror the ``dipole-rl/v0-kingback``
branch; the discriminator path is adapted to the frozen nnPU head.
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


@dataclass(frozen=True)
class QCandidateInputs:
    starts: list[int]
    images_np: np.ndarray
    proprio_cpu: torch.Tensor
    actions_cpu: torch.Tensor


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
    parser.add_argument(
        "--no-bon",
        action="store_true",
        help="Skip Best-of-n Q diagnostics even on success-like splits.",
    )
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
    parser.add_argument("--q-candidate-noise-sigmas", default="0.05,0.10,0.20")
    parser.add_argument("--q-candidate-random-n", type=int, default=16)
    parser.add_argument("--q-candidate-single-dim-sigma", type=float, default=0.20)
    parser.add_argument("--q-candidate-single-dim-n", type=int, default=0)
    parser.add_argument("--q-candidate-seed", type=int, default=None)
    parser.add_argument("--q-candidate-action-low", type=float, default=-1.0)
    parser.add_argument("--q-candidate-action-high", type=float, default=1.0)
    return parser.parse_args()


def load_iql_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"IQL checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    schema_version = int(payload.get("schema_version", -1))
    if schema_version not in (2, 3):
        raise ValueError(
            f"Unsupported IQL checkpoint schema_version={schema_version}. "
            "Legacy LPB checkpoints are incompatible with nnPU features; "
            "re-run offline Q/V warmup."
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
    learner.q_ensemble.eval()
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
) -> tuple[list[dict[str, float]], PerStepNNPUDisc, QCandidateInputs]:
    """Compute per-window Q/V metrics and a per-frame nnPU failure series.

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
        q_values = learner._q_values(  # noqa: SLF001
            chunk_features[:, 0], actions.to(learner.cfg.device)
        )
        v = learner.v(state_features[:, 0])
        next_v = learner.target_v(next_state)
        bootstrap_v = bootstrap_discount * (1.0 - done) * next_v
        td_target = aggregated + bootstrap_v
        q_mean = q_values.mean(dim=0)
        for index, start in enumerate(batch_starts):
            q_min = float(q_values[:, index].min().item())
            q_max = float(q_values[:, index].max().item())
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
                    "q_mean": float(q_mean[index].item()),
                    "q_min": q_min,
                    "q_max": q_max,
                    "v": v_val,
                    "next_v": float(next_v[index].item()),
                    "bootstrap_v": float(bootstrap_v[index].item()),
                    "advantage": float(q_mean[index].item()) - v_val,
                    "advantage_td1": td_target_val - v_val,
                    "td_target": td_target_val,
                    "td_residual": td_target_val - q_min,
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
    q_candidate_inputs = QCandidateInputs(
        starts=list(starts),
        images_np=np.ascontiguousarray(images_np),
        proprio_cpu=proprio_cpu.detach().cpu(),
        actions_cpu=actions_cpu.detach().cpu(),
    )
    return rows, per_step_disc, q_candidate_inputs


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
    q_min = np.asarray([row["q_min"] for row in metrics], dtype=np.float32)
    q_mean = np.asarray([row["q_mean"] for row in metrics], dtype=np.float32)
    q_max = np.asarray([row["q_max"] for row in metrics], dtype=np.float32)
    v = np.asarray([row["v"] for row in metrics], dtype=np.float32)
    next_v = np.asarray([row["next_v"] for row in metrics], dtype=np.float32)
    td_target = np.asarray([row["td_target"] for row in metrics], dtype=np.float32)
    advantage = np.asarray([row["advantage"] for row in metrics], dtype=np.float32)
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

    axes[0].plot(steps, q_mean, label="Q mean", color="tab:blue")
    axes[0].fill_between(steps, q_min, q_max, color="tab:blue", alpha=0.18, label="Q min/max")
    axes[0].plot(steps, v, label="V", color="tab:orange")
    axes[0].plot(steps, next_v, label="target next V", color="tab:green", alpha=0.8)
    axes[0].set_ylabel("Q / V")
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

    axes[2].plot(steps, advantage, label="advantage Qmean - V", color="tab:brown")
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


def parse_float_csv(value: str) -> list[float]:
    items: list[float] = []
    for raw in str(value).split(","):
        item = raw.strip()
        if item:
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


@torch.no_grad()
def compute_q_candidate_rows(
    *,
    learner: IQLLearner,
    encoder: SharedDynamicsEncoder,
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
    total = int(num_windows * num_candidates)
    flat_actions_np = np.ascontiguousarray(
        candidate_actions.reshape(total, horizon, action_dim)
    )
    flat_actions = torch.from_numpy(flat_actions_np).float()
    q_min_flat = np.empty((total,), dtype=np.float32)
    q_mean_flat = np.empty((total,), dtype=np.float32)
    q_max_flat = np.empty((total,), dtype=np.float32)
    q_std_flat = np.empty((total,), dtype=np.float32)

    eval_batch_size = max(1, int(batch_size))
    eval_offsets = range(0, total, eval_batch_size)
    for start in tqdm(
        eval_offsets,
        desc="[vis_qv] BON Q candidates",
        unit="batch",
        total=(total + eval_batch_size - 1) // eval_batch_size,
    ):
        end = min(total, start + eval_batch_size)
        flat_indices = np.arange(start, end, dtype=np.int64)
        window_indices = flat_indices // int(num_candidates)
        batch_images_np = inputs.images_np[window_indices]
        batch_proprio = inputs.proprio_cpu.index_select(
            0, torch.from_numpy(window_indices).long()
        )
        batch_actions = flat_actions[start:end]
        batch, _, views, channels, height, width = batch_images_np.shape
        images = _image_tensor(
            batch_images_np.reshape(batch * horizon, views, channels, height, width)
        ).view(batch, horizon, views, channels, height, width)
        _, chunk_features = encoder.encode_features(
            chunk_images=images,
            chunk_proprio=batch_proprio,
            chunk_actions=batch_actions,
        )
        q_values = learner._q_values(  # noqa: SLF001
            chunk_features[:, 0], batch_actions.to(device)
        )
        q_min_flat[start:end] = q_values.min(dim=0).values.view(-1).detach().cpu().numpy()
        q_mean_flat[start:end] = q_values.mean(dim=0).view(-1).detach().cpu().numpy()
        q_max_flat[start:end] = q_values.max(dim=0).values.view(-1).detach().cpu().numpy()
        q_std_flat[start:end] = q_values.std(dim=0, unbiased=False).view(-1).detach().cpu().numpy()

    q_min = q_min_flat.reshape(num_windows, num_candidates)
    q_mean = q_mean_flat.reshape(num_windows, num_candidates)
    q_max = q_max_flat.reshape(num_windows, num_candidates)
    q_std = q_std_flat.reshape(num_windows, num_candidates)
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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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


def main() -> None:
    args = parse_args()
    run_bon_diagnostics = is_success_related_split(str(args.split)) and not bool(args.no_bon)
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

    rows, per_step_disc, q_candidate_inputs = compute_metrics(
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
        / f"{task}_iql-qv"
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

    bon_outputs: dict[str, Any] | None = None
    q_candidate_summary: dict[str, Any] | None = None
    if run_bon_diagnostics:
        bon_dir = output_dir / "BON"
        bon_dir.mkdir(parents=True, exist_ok=True)
        q_candidate_csv_path = bon_dir / "q_candidates.csv"
        q_candidate_summary_path = bon_dir / "summary.json"
        q_candidate_seed = int(args.seed) if args.q_candidate_seed is None else int(args.q_candidate_seed)
        q_candidate_rows, q_candidate_summary = compute_q_candidate_rows(
            learner=learner,
            encoder=encoder,
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
        if bool(args.no_bon):
            print("[vis_qv] skipping Best-of-n Q diagnostics (--no-bon).")
        else:
            print(
                f"[vis_qv] skipping Best-of-n Q diagnostics for split={args.split!r} "
                "(only enabled for success-like splits)."
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
        "bon_enabled": bool(run_bon_diagnostics),
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
            "bon": bon_outputs,
            "discriminator": disc_viz_outputs,
        },
        "q_candidate_summary": q_candidate_summary,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[vis_qv] wrote {output_dir}")
    print(f"[vis_qv] plot_png={plot_paths['overlapping']}")
    print(f"[vis_qv] plot_png_nonoverlap={plot_paths['nonoverlap']}")
    if rollout_video is not None:
        print(f"[vis_qv] rollout_video={rollout_video}")
    if bon_outputs is not None:
        print(f"[vis_qv] bon_dir={bon_outputs['output_dir']}")
        print(f"[vis_qv] bon_candidates_csv={bon_outputs['candidates_csv']}")
    if disc_viz_outputs is not None:
        print(f"[vis_qv] disc_viz_dir={disc_viz_outputs['output_dir']}")
        print(f"[vis_qv] disc_viz_video={disc_viz_outputs['video']}")


if __name__ == "__main__":
    main()
