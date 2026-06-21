"""Visualize nnPU-backed IQL Q/V values on one recorded HDF5 trajectory.

Outputs (per run, under ``<output-root>/<task>_iql-qv/<split>_seed<seed>_<ts>/``):
  * ``steps.csv``                    — per-window Q/V/advantage/reward metrics.
  * ``qv_timeseries.png``            — 4-subplot diagnostics over overlapping windows.
  * ``qv_timeseries_nonoverlap.png`` — same plot restricted to disjoint chunks (stride=H).
  * ``rollout_policy_obs.mp4``       — raw policy-camera rollout video.
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
from hydra.utils import to_absolute_path

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


@dataclass(frozen=True)
class SelectedDemo:
    hdf5_path: Path
    demo_key: str
    length: int
    successful: bool


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
    parser.add_argument("--demo-key", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--output-root", default="outputs/DIPOLE_rl/iql_qv_cache-vis")
    parser.add_argument("--renderer", default="mjviewer")
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("--no-disc-reward", action="store_true")
    parser.add_argument("--no-disc-viz", action="store_true")
    # Discriminator HUD / rollout video controls.
    parser.add_argument("--video-fps", type=int, default=None)
    parser.add_argument("--disc-viz-camera", default=None)
    parser.add_argument("--disc-viz-border-thickness", type=int, default=10)
    parser.add_argument("--no-flip-vertical", action="store_true")
    # Retained so existing launch scripts remain valid. Candidate diagnostics
    # require re-encoding every candidate and are intentionally not generated.
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
    if int(payload.get("schema_version", -1)) != 2:
        raise ValueError(
            "Legacy LPB IQL checkpoints are incompatible with nnPU features. "
            "Re-run offline Q/V warmup."
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


def load_demo(
    selected: SelectedDemo,
    *,
    camera_names: list[str],
    image_size: int,
    renderer: str,
    control_freq: int,
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
) -> tuple[list[dict[str, float]], PerStepNNPUDisc]:
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
    # Per-frame nnPU traces for every window: (W, H).
    disc_failure_wh = np.zeros((num_windows, horizon), dtype=np.float32)
    disc_intrinsic_wh = np.zeros((num_windows, horizon), dtype=np.float32)
    rows: list[dict[str, float]] = []
    bootstrap_discount = float(cfg.discount) ** horizon
    for offset in range(0, len(starts), max(1, int(batch_size))):
        batch_starts = starts[offset : offset + max(1, int(batch_size))]
        sequences = [transitions[start : start + horizon] for start in batch_starts]
        images_np = np.stack(
            [np.stack([_stack_views(item.obs, camera_names) for item in seq]) for seq in sequences]
        )
        proprio = torch.from_numpy(
            np.stack([[np.asarray(item.obs["state"], np.float32) for item in seq] for seq in sequences])
        ).float()
        actions = torch.from_numpy(
            np.stack([[np.asarray(item.action, np.float32) for item in seq] for seq in sequences])
        ).float()
        rewards = torch.tensor(
            [[float(item.reward or 0.0) for item in seq] for seq in sequences], dtype=torch.float32
        )
        # Bootstrap terminal mask, mirroring IQL training (replay._build_step_batch):
        # only a task-success frame is a true terminal. A truncated (e.g.
        # fail_rollout) trajectory end keeps the γ^H·V(s') bootstrap, so the tail
        # chunks' TD target stays on the same scale as the interior windows.
        dones = torch.tensor(
            [[float(bool((item.info or {}).get("success", False))) for item in seq] for seq in sequences],
            dtype=torch.float32,
        )
        batch, _, views, channels, height, width = images_np.shape
        images = _image_tensor(images_np.reshape(batch * horizon, views, channels, height, width))
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
        q_values = learner._q_values(chunk_features[:, 0])  # noqa: SLF001
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
                    "advantage": q_min - v_val,
                    "advantage_td1": td_target_val - v_val,
                    "td_target": td_target_val,
                    "td_residual": td_target_val - q_min,
                    "env_reward_horizon": float(env_aggregated[index].item()),
                    "disc_reward_horizon": float(disc_aggregated[index].item()),
                    "total_reward_horizon": float(aggregated[index].item()),
                    "disc_intrinsic_step0": float(disc_intrinsic_wh[offset + index, 0]),
                    "failure_score_start": float(disc_failure_wh[offset + index, 0]),
                    "done_chunk": float(done[index].item()),
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

    # At a trajectory's terminal chunk the bootstrap γ^H·V(s') is masked out
    # (done=1), so td_target collapses to the chunk's immediate reward while V/Q
    # still carry the bootstrapped tail. The resulting td_target / advantage_td1
    # outlier would crush the y-axis, so hide just those points (gap in the
    # curve); the raw values remain in steps.csv (see the done_chunk column).
    terminal = np.asarray([row.get("done_chunk", 0.0) for row in metrics], dtype=np.float32) > 0.5
    if bool(terminal.any()):
        td_target = td_target.copy()
        advantage_td1 = advantage_td1.copy()
        td_target[terminal] = np.nan
        advantage_td1[terminal] = np.nan

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

    axes[2].plot(steps, advantage, label="advantage Qmin - V", color="tab:brown")
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
    checkpoint = Path(to_absolute_path(str(args.iql_ckpt))).resolve()
    payload = load_iql_payload(checkpoint)
    device = resolve_device(args.device, str(payload["cfg"].get("device", "cpu")))
    learner, encoder, discriminator, cfg, meta = build_models(
        payload, device=device, disc_override=args.disc_ckpt
    )
    task = str(args.task_data_name or meta.get("task") or meta.get("task_env"))
    split_dir = Path(to_absolute_path(str(args.demo_root))) / task / str(args.split)
    selected = select_demo(split_dir.resolve(), seed=args.seed, demo_key=args.demo_key)
    camera_names = [str(name) for name in meta["policy_camera_names"]]
    transitions = load_demo(
        selected,
        camera_names=camera_names,
        image_size=int(meta.get("image_size", 128)),
        renderer=args.renderer,
        control_freq=args.control_freq,
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
        Path(to_absolute_path(str(args.output_root))).resolve()
        / f"{task}_iql-qv"
        / f"{args.split}_seed{args.seed}_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    video_fps = int(args.video_fps if args.video_fps is not None else args.control_freq)
    flip_vertical = not bool(args.no_flip_vertical)

    write_metrics_csv(output_dir / "steps.csv", rows)
    plot_paths = plot_qv(
        output_dir / "qv_timeseries",
        rows,
        title=f"{task} {args.split} {selected.hdf5_path.name}::{selected.demo_key}",
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
            transitions=transitions,
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
        "selected_hdf5": str(selected.hdf5_path),
        "selected_demo_key": selected.demo_key,
        "num_windows": len(rows),
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
