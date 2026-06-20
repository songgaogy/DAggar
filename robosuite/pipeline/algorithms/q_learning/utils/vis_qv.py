"""Visualize nnPU-backed IQL Q/V values on one recorded HDF5 trajectory."""

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
) -> list[dict[str, float]]:
    horizon = int(cfg.action_horizon)
    starts = list(range(max(0, len(transitions) - horizon + 1)))
    if max_windows is not None:
        starts = starts[: int(max_windows)]
    if not starts:
        raise RuntimeError(f"Need at least {horizon} transitions, got {len(transitions)}")
    rows: list[dict[str, float]] = []
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
        dones = torch.tensor(
            [[float(bool(item.done)) for item in seq] for seq in sequences], dtype=torch.float32
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
        disc_steps = (
            discriminator.intrinsic_reward(chunk_feature=chunk_features).cpu()
            if use_disc_reward
            else torch.zeros_like(rewards)
        )
        total_steps = float(cfg.output_reward_coef) * rewards + float(cfg.disc_reward_coef) * disc_steps
        aggregated = aggregate_chunk_reward(total_steps, float(cfg.discount)).to(learner.cfg.device)
        done = chunk_done_mask(dones).to(learner.cfg.device)
        q_values = learner._q_values(chunk_features[:, 0])  # noqa: SLF001
        v = learner.v(state_features[:, 0])
        next_v = learner.target_v(next_state)
        target = aggregated + float(cfg.discount) ** horizon * (1.0 - done) * next_v
        q_mean = q_values.mean(dim=0)
        for index, start in enumerate(batch_starts):
            rows.append(
                {
                    "window_start": float(start),
                    "q_mean": float(q_mean[index].item()),
                    "q_min": float(q_values[:, index].min().item()),
                    "v": float(v[index].item()),
                    "advantage": float((q_mean[index] - v[index]).item()),
                    "target_q": float(target[index].item()),
                    "td_residual": float((q_mean[index] - target[index]).item()),
                    "env_reward_horizon": float(aggregate_chunk_reward(rewards, cfg.discount)[index].item()),
                    "disc_reward_horizon": float(aggregate_chunk_reward(disc_steps, cfg.discount)[index].item()),
                    "total_reward_horizon": float(aggregated[index].item()),
                    "failure_score_start": float(
                        discriminator.failure_score(chunk_features[index, 0]).item()
                    ),
                }
            )
    return rows


def write_outputs(
    output_dir: Path,
    rows: list[dict[str, float]],
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "steps.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    steps = np.asarray([row["window_start"] for row in rows])
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(steps, [row["q_mean"] for row in rows], label="Q mean")
    axes[0].plot(steps, [row["v"] for row in rows], label="V")
    axes[0].legend()
    axes[1].plot(steps, [row["advantage"] for row in rows], label="Q-V")
    axes[1].legend()
    axes[2].plot(steps, [row["total_reward_horizon"] for row in rows], label="chunk reward")
    axes[2].plot(steps, [row["failure_score_start"] for row in rows], label="nnPU failure")
    axes[2].legend()
    axes[2].set_xlabel("window start")
    for axis in axes:
        axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "qv_timeseries.png", dpi=160)
    plt.close(fig)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )


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
    rows = compute_metrics(
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
        "device": device,
    }
    write_outputs(output_dir, rows, summary)
    print(f"[vis_qv] wrote {output_dir}")


if __name__ == "__main__":
    main()
