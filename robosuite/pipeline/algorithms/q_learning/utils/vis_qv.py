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
from robosuite.pipeline.algorithms.discriminator.online_bce import (
    DiscriminatorConfig,
    OnlineBCEDiscriminator,
)
from robosuite.pipeline.algorithms.q_learning.common import IQLConfig
from robosuite.pipeline.algorithms.q_learning.data_util import (
    aggregate_chunk_reward,
    chunk_done_mask,
    disc_logit_to_intrinsic_reward,
)
from robosuite.pipeline.algorithms.q_learning.iql import IQLLearner
from robosuite.pipeline.common import Transition
from robosuite.pipeline.train_dipole import load_hdf5_demos_into_flow_transitions
from robosuite.policy.flow_multi.utils.env_util import RobosuiteProprioExtractor, parse_env_info


DEFAULT_DEMO_ROOT = "data"
DEFAULT_OUTPUT_ROOT = "outputs/DIPOLE_rl/iql_qv_visualization"


@dataclass
class SelectedDemo:
    hdf5_path: Path
    demo_key: str
    length: int
    successful: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize trained Q-chunking IQL Q/V on HDF5 rollout windows."
    )
    parser.add_argument("--iql-ckpt", required=True)
    parser.add_argument("--task-data-name", default=None)
    parser.add_argument("--split", default="success_rollout")
    parser.add_argument("--demo-root", default=DEFAULT_DEMO_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
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
        help="BCE discriminator head checkpoint for plotting discriminator-shaped reward.",
    )
    parser.add_argument("--no-disc-reward", action="store_true")
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


def build_discriminator(
    *,
    payload: dict[str, Any],
    encoder: SharedFrozenEncoder,
    action_dim: int,
    action_horizon: int,
    device: str,
    disc_ckpt_override: str | None,
    disabled: bool,
) -> OnlineBCEDiscriminator | None:
    if disabled:
        return None
    encoder_meta = dict(payload.get("encoder_meta", {}))
    if disc_ckpt_override:
        ckpt_path = Path(to_absolute_path(str(disc_ckpt_override))).resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Discriminator checkpoint does not exist: {ckpt_path}")
    else:
        raw_path = str(
            encoder_meta.get("disc_warm_start_ckpt")
            or encoder_meta.get("bce_ckpt")
            or ""
        )
        ckpt_path = Path(raw_path).resolve() if raw_path else None

    if ckpt_path is None or not ckpt_path.exists():
        print(f"[WARN] BCE checkpoint missing; skipping discriminator reward: {ckpt_path}")
        return None
    disc_cfg = DiscriminatorConfig(warm_start_ckpt=str(ckpt_path), device=device)
    discriminator = OnlineBCEDiscriminator(
        cfg=disc_cfg,
        encoder=encoder,
        context_dim=int(encoder.context_dim),
        action_dim=int(action_dim),
        action_horizon=int(action_horizon),
    )
    for param in discriminator.head.parameters():
        param.requires_grad_(False)
    discriminator.head.eval()
    return discriminator


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


def encode_observations(
    encoder: SharedFrozenEncoder,
    observations: list[dict[str, Any]],
    *,
    camera_names: list[str],
    image_size: int,
    batch_size: int,
) -> torch.Tensor:
    chunks = []
    for start in range(0, len(observations), int(batch_size)):
        batch_obs = observations[start : start + int(batch_size)]
        image_np = np.stack(
            [stack_views(obs, camera_names, image_size) for obs in batch_obs],
            axis=0,
        )
        proprio_np = np.stack(
            [np.asarray(obs["state"], dtype=np.float32) for obs in batch_obs],
            axis=0,
        )
        image_tensor = torch.from_numpy(np.ascontiguousarray(image_np)).float().div_(255.0)
        proprio_tensor = torch.from_numpy(np.ascontiguousarray(proprio_np)).float()
        with torch.no_grad():
            chunks.append(encoder.encode(image_obs_raw=image_tensor, proprio_raw=proprio_tensor).detach().cpu())
    return torch.cat(chunks, dim=0)


def tensor_to_float(tensor: torch.Tensor, index: int) -> float:
    return float(tensor[index].detach().cpu().item())


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
) -> list[dict[str, float]]:
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

    context_cpu = encode_observations(
        encoder,
        current_obs,
        camera_names=camera_names,
        image_size=image_size,
        batch_size=batch_size,
    )
    next_context_cpu = encode_observations(
        encoder,
        next_obs,
        camera_names=camera_names,
        image_size=image_size,
        batch_size=batch_size,
    )
    actions_cpu = torch.from_numpy(np.ascontiguousarray(action_np)).float()
    rewards_cpu = torch.from_numpy(np.ascontiguousarray(reward_np)).float()
    dones_cpu = torch.from_numpy(np.ascontiguousarray(done_np)).float()

    metrics: list[dict[str, float]] = []
    for start_idx in range(0, len(windows), int(batch_size)):
        end_idx = min(len(windows), start_idx + int(batch_size))
        context = context_cpu[start_idx:end_idx].to(device)
        next_context = next_context_cpu[start_idx:end_idx].to(device)
        actions = actions_cpu[start_idx:end_idx].to(device)
        rewards = rewards_cpu[start_idx:end_idx].to(device)
        dones = dones_cpu[start_idx:end_idx].to(device)

        with torch.no_grad():
            q1 = iql.q1(context, actions)
            q2 = iql.q2(context, actions)
            q_min = torch.minimum(q1, q2)
            q_mean = 0.5 * (q1 + q2)
            q_max = torch.maximum(q1, q2)
            v = iql.v(context)
            next_v = iql.target_v(next_context)
            env_reward_horizon = aggregate_chunk_reward(rewards, discount)
            done_horizon = chunk_done_mask(dones).to(device)

            if discriminator is not None and float(iql_cfg.disc_reward_coef) != 0.0:
                disc_logit = discriminator.intrinsic_reward(context=context, action_chunk=actions)
                disc_reward_chunk = disc_logit_to_intrinsic_reward(
                    disc_logit, str(iql_cfg.disc_reward_sign)
                )
                disc_per_step = (disc_reward_chunk.unsqueeze(-1) / float(horizon)).expand(-1, horizon)
                disc_reward_horizon = aggregate_chunk_reward(disc_per_step, discount)
            else:
                disc_logit = torch.zeros((end_idx - start_idx,), device=device, dtype=context.dtype)
                disc_reward_chunk = torch.zeros_like(disc_logit)
                disc_reward_horizon = torch.zeros_like(env_reward_horizon)

            total_reward_horizon = env_reward_horizon + float(iql_cfg.disc_reward_coef) * disc_reward_horizon
            td_target = total_reward_horizon + (discount**horizon) * (1.0 - done_horizon) * next_v
            td_residual = td_target - q_min
            advantage = q_min - v

        for local_idx, global_idx in enumerate(range(start_idx, end_idx)):
            row = {
                "window_index": float(global_idx),
                "step": float(starts[global_idx]),
                "env_reward_horizon": tensor_to_float(env_reward_horizon, local_idx),
                "disc_logit": tensor_to_float(disc_logit, local_idx),
                "disc_reward_chunk": tensor_to_float(disc_reward_chunk, local_idx),
                "disc_reward_horizon": tensor_to_float(disc_reward_horizon, local_idx),
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
            }
            for horizon_index in range(horizon):
                for action_index in range(actions_cpu.shape[-1]):
                    row[f"action_h{horizon_index}_{action_index}"] = float(
                        actions_cpu[global_idx, horizon_index, action_index].item()
                    )
            metrics.append(row)
    return metrics


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


def plot_qv(path_base: Path, metrics: list[dict[str, float]], title: str) -> None:
    steps = np.asarray([row["step"] for row in metrics], dtype=np.float32)
    q_min = np.asarray([row["q_min"] for row in metrics], dtype=np.float32)
    q_mean = np.asarray([row["q_mean"] for row in metrics], dtype=np.float32)
    q_max = np.asarray([row["q_max"] for row in metrics], dtype=np.float32)
    v = np.asarray([row["v"] for row in metrics], dtype=np.float32)
    next_v = np.asarray([row["next_v"] for row in metrics], dtype=np.float32)
    td_target = np.asarray([row["td_target"] for row in metrics], dtype=np.float32)
    td_residual = np.asarray([row["td_residual"] for row in metrics], dtype=np.float32)
    advantage = np.asarray([row["advantage"] for row in metrics], dtype=np.float32)
    env_rewards = np.asarray([row["env_reward_horizon"] for row in metrics], dtype=np.float32)
    total_rewards = np.asarray([row["total_reward_horizon"] for row in metrics], dtype=np.float32)
    disc_rewards = np.asarray([row["disc_reward_horizon"] for row in metrics], dtype=np.float32)

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(title)

    axes[0].plot(steps, q_mean, label="Q mean", color="tab:blue")
    axes[0].fill_between(steps, q_min, q_max, color="tab:blue", alpha=0.18, label="Q min/max")
    axes[0].plot(steps, v, label="V", color="tab:orange")
    axes[0].plot(steps, next_v, label="target next V", color="tab:green", alpha=0.8)
    axes[0].set_ylabel("Q / V")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, td_target, label="TD target", color="tab:purple")
    axes[1].plot(steps, q_min, label="Q min", color="tab:blue", alpha=0.75)
    axes[1].set_ylabel("Target / Q")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps, td_residual, label="TD residual", color="tab:red")
    axes[2].plot(steps, advantage, label="advantage Qmin - V", color="tab:brown")
    axes[2].axhline(0.0, color="black", linewidth=1)
    axes[2].set_ylabel("Residual / Adv")
    axes[2].legend(loc="best")
    axes[2].grid(True, alpha=0.3)

    axes[3].step(steps, total_rewards, where="post", label="total chunk reward", color="tab:gray")
    axes[3].step(steps, env_rewards, where="post", label="env chunk reward", color="tab:olive", alpha=0.75)
    axes[3].plot(steps, disc_rewards, label="disc chunk reward", color="tab:pink", alpha=0.85)
    axes[3].set_ylabel("Reward")
    axes[3].set_xlabel("Step")
    axes[3].legend(loc="best")
    axes[3].grid(True, alpha=0.3)

    fig.tight_layout()
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".png"), dpi=160)
    fig.savefig(path_base.with_suffix(".pdf"))
    plt.close(fig)


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
        "disc_reward_sign": str(iql_cfg.disc_reward_sign),
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
    discriminator = build_discriminator(
        payload=payload,
        encoder=encoder,
        action_dim=action_dim,
        action_horizon=int(iql_cfg.action_horizon),
        device=device,
        disc_ckpt_override=args.disc_ckpt,
        disabled=bool(args.no_disc_reward),
    )

    selected = select_demo(split_dir, seed=int(args.seed), demo_key=args.demo_key)
    transitions = load_demo_transitions(
        selected,
        camera_names=camera_names,
        image_size=image_size,
        renderer=str(args.renderer),
        control_freq=int(args.control_freq),
    )
    metrics = compute_qv_metrics(
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
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_root = Path(to_absolute_path(str(args.output_root))).resolve()
    output_dir = output_root / f"{task_data_name}_iql-qv/{args.split}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "steps.csv"
    video_path = output_dir / "rollout_policy_obs.mp4"
    plot_base = output_dir / "qv_timeseries"
    summary_path = output_dir / "summary.json"

    write_metrics_csv(csv_path, metrics)
    write_video(video_path, transitions, image_keys=camera_names, fps=int(args.video_fps))
    plot_qv(plot_base, metrics, title=f"{task_data_name} {args.split} {selected.hdf5_path.name}::{selected.demo_key}")

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
        "disc_reward_enabled": discriminator is not None and not bool(args.no_disc_reward),
        "bce_ckpt": str(encoder_meta.get("bce_ckpt", "")),
        "disc_ckpt_override": None if args.disc_ckpt is None else str(Path(to_absolute_path(str(args.disc_ckpt))).resolve()),
        "outputs": {
            "steps_csv": str(csv_path),
            "video": str(video_path),
            "plot_png": str(plot_base.with_suffix(".png")),
            "plot_pdf": str(plot_base.with_suffix(".pdf")),
        },
        "metrics_summary": summarize_metrics(metrics, iql_cfg),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"selected_demo={selected.hdf5_path}::{selected.demo_key}")
    print(f"iql_ckpt={iql_ckpt}")
    print(f"output_dir={output_dir}")
    print(f"video={video_path}")
    print(f"plot_png={plot_base.with_suffix('.png')}")
    print(f"plot_pdf={plot_base.with_suffix('.pdf')}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
