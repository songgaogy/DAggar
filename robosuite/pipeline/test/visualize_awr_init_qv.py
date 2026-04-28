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

from robosuite.pipeline.algorithms.awr.replay_buffer import get_transition_awr_fields
from robosuite.pipeline.common import Transition
from robosuite.pipeline.factory import build_algorithm
from robosuite.pipeline.envs.robosuite import load_hdf5_demos_into_transitions
from robosuite.pipeline.train_flow_dagger import load_hdf5_demos_into_flow_transitions
from robosuite.policy.flow_multi.utils.env_util import RobosuiteProprioExtractor, parse_env_info


DEFAULT_CACHE_DIR = "outputs/awr/qv_cache"
DEFAULT_TASK_DATA_NAME = "PickPlaceCereal"
DEFAULT_OUTPUT_ROOT = "outputs/awr/qv_visualization"


@dataclass
class SelectedDemo:
    hdf5_path: Path
    demo_key: str
    length: int
    successful: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize AWR initialization Q/V from a Q/V cache.")
    parser.add_argument("--qv-cache", default=None)
    parser.add_argument("--task-data-name", default=None)
    parser.add_argument("--split", default="success_rollout", choices=("success_rollout", "fail_rollout", "expert"))
    parser.add_argument("--demo-root", default="data")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--demo-key", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--inference-device", default=None)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--renderer", default="mjviewer")
    parser.add_argument("--control-freq", type=int, default=20)
    return parser.parse_args()


def resolve_qv_cache_path(args: argparse.Namespace) -> Path:
    if args.qv_cache is not None:
        return Path(to_absolute_path(str(args.qv_cache))).resolve()
    cache_name = f"{str(args.task_data_name or DEFAULT_TASK_DATA_NAME)}.pt"
    return (Path(to_absolute_path(DEFAULT_CACHE_DIR)) / cache_name).resolve()


def resolve_split_dir(args: argparse.Namespace, task_data_name: str) -> Path:
    return (Path(to_absolute_path(str(args.demo_root))) / task_data_name / str(args.split)).resolve()


def load_cache_payload(qv_cache_path: Path) -> dict[str, Any]:
    if not qv_cache_path.exists():
        raise FileNotFoundError(f"Q/V cache does not exist: {qv_cache_path}")
    payload = torch.load(qv_cache_path, map_location="cpu", weights_only=False)
    if "qv_core" not in payload:
        raise KeyError(f"Cache is missing 'qv_core': {qv_cache_path}")
    return payload


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
                successful = bool(demo_group.attrs.get("successful", False))
                length = int(demo_group.attrs.get("length", len(demo_group["actions"])))
                selected = SelectedDemo(
                    hdf5_path=hdf5_path,
                    demo_key=str(key),
                    length=length,
                    successful=successful,
                )
                if demo_key is None or str(key) == str(demo_key):
                    candidates.append(selected)

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
    camera_aliases: dict[str, str],
    image_size: int,
    renderer: str,
    control_freq: int,
) -> list[Transition]:
    # We keep the original flow loader to ensure observation / proprio formatting
    # matches what the cached AWR model expects, but we override reward/done using
    # env-replay sparse success signals so reward becomes 0 starting at the true
    # success step (not only at the last step of a padded rollout).
    extractor = build_proprio_extractor(selected.hdf5_path)
    try:
        transitions = load_hdf5_demos_into_flow_transitions(
            selected.hdf5_path,
            policy_camera_names=camera_names,
            camera_aliases=camera_aliases,
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

    # Compute env-based sparse success reward/done for the same demo and patch it in.
    # NOTE: we only use reward/done scalars; obs/state from env replay may not match.
    env_transitions = load_hdf5_demos_into_transitions(
        selected.hdf5_path,
        camera_names=tuple(camera_names),
        img_height=int(image_size),
        img_width=int(image_size),
        proprio_keys=(),
        renderer=str(renderer),
        control_freq=int(control_freq),
        demo_names=(str(selected.demo_key),),
    )
    if len(env_transitions) != len(transitions):
        raise RuntimeError(
            "Env replay transitions length mismatch with flow transitions: "
            f"{len(env_transitions)} vs {len(transitions)} for {selected.hdf5_path}::{selected.demo_key}"
        )
    for idx, (flow_t, env_t) in enumerate(zip(transitions, env_transitions)):
        _ = idx
        flow_t.reward = float(env_t.reward)
        flow_t.done = bool(env_t.done)
        if isinstance(flow_t.info, dict):
            flow_t.info["reward_source"] = "env_success"
            flow_t.info["success_from_env_replay"] = True
    return transitions


def resolve_init_checkpoint(payload: dict[str, Any], override: str | None) -> Path:
    if override is not None:
        path = Path(to_absolute_path(str(override))).resolve()
    else:
        metadata = dict(payload.get("metadata", {}))
        init_checkpoint = metadata.get("init_checkpoint", None)
        if init_checkpoint is None:
            raise KeyError("Q/V cache metadata does not contain init_checkpoint. Pass --init-checkpoint.")
        path = Path(to_absolute_path(str(init_checkpoint))).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Actor init checkpoint does not exist: {path}")
    return path


def prepare_algorithm_cfg(payload: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    awr_cfg = dict(payload.get("awr_config", {}))
    awr_cfg["model"] = dict(payload.get("model_cfg", awr_cfg.get("model", {})))
    if args.device is not None:
        awr_cfg["device"] = str(args.device)
    if args.inference_device is not None:
        awr_cfg["inference_device"] = str(args.inference_device)
    return {
        "type": "awr",
        "camera_names": [str(name) for name in payload.get("camera_names", [])],
        "task_name": str(payload.get("task_name", awr_cfg.get("task_name", "task"))),
        "encoder": dict(payload.get("encoder_config", {})),
        "awr": awr_cfg,
        "online_buffer": {"capacity": 1},
        "demo_buffer": {"capacity": 1},
        "trainer": dict(payload.get("trainer_config", {})),
    }


def build_agent(
    payload: dict[str, Any],
    algorithm_cfg: dict[str, Any],
    transitions: list[Transition],
    init_checkpoint: Path,
):
    awr_cfg = dict(payload.get("awr_config", {}))
    action_dim = int(awr_cfg.get("action_dim", np.asarray(transitions[0].action).reshape(-1).shape[0]))
    action_low = -np.ones(action_dim, dtype=np.float32)
    action_high = np.ones(action_dim, dtype=np.float32)
    agent = build_algorithm(
        algorithm_cfg,
        observation_example=transitions[0].obs,
        sample_action=np.zeros(action_dim, dtype=np.float32),
        action_low=action_low,
        action_high=action_high,
    )
    agent.load_flow_policy_checkpoint(init_checkpoint, task_name=str(algorithm_cfg["task_name"]))
    agent.load_qv_cache_payload(payload, load_optimizers=False)
    agent.core.model.eval()
    agent.core.inference_model.eval()
    return agent


def center_crop_resize(image: np.ndarray, image_size: int) -> np.ndarray:
    height, width = image.shape[:2]
    crop_size = min(height, width)
    y0 = (height - crop_size) // 2
    x0 = (width - crop_size) // 2
    crop = image[y0 : y0 + crop_size, x0 : x0 + crop_size]
    if crop_size == image_size:
        return np.asarray(crop, dtype=np.uint8)
    ys = np.linspace(0, crop_size - 1, image_size).astype(np.int32)
    xs = np.linspace(0, crop_size - 1, image_size).astype(np.int32)
    return np.asarray(crop[ys][:, xs], dtype=np.uint8)


def preprocess_observations(
    observations: list[dict[str, Any]],
    *,
    camera_names: list[str],
    image_size: int,
    proprio_mean: np.ndarray | None,
    proprio_std: np.ndarray | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    image_batch = []
    proprio_batch = []
    for obs in observations:
        obs_images = []
        for camera_name in camera_names:
            image = np.asarray(obs[camera_name], dtype=np.uint8)
            resized = center_crop_resize(image, int(image_size))
            obs_images.append(np.transpose(resized, (2, 0, 1)))
        image_batch.append(np.stack(obs_images, axis=0))
        proprio = np.asarray(obs["state"], dtype=np.float32)
        if proprio_mean is not None and proprio_std is not None:
            proprio = (proprio - proprio_mean) / (proprio_std + 1e-6)
        proprio_batch.append(proprio)

    images = torch.from_numpy(np.ascontiguousarray(np.stack(image_batch, axis=0))).to(device=device)
    images = images.to(dtype=torch.float32).div_(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=images.dtype, device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=images.dtype, device=device).view(1, 1, 3, 1, 1)
    images = (images - mean) / std
    proprio = torch.from_numpy(np.ascontiguousarray(np.stack(proprio_batch, axis=0))).to(device=device)
    return images, proprio.float()


def build_windows(transitions: list[Transition], horizon: int, max_windows: int | None) -> list[tuple[int, list[Transition]]]:
    windows = []
    max_start = len(transitions) - int(horizon) + 1
    for start in range(max(0, max_start)):
        sequence = transitions[start : start + int(horizon)]
        windows.append((start, sequence))
        if max_windows is not None and len(windows) >= int(max_windows):
            break
    if not windows:
        raise RuntimeError(f"Need at least action_horizon={horizon} transitions, got {len(transitions)}.")
    return windows


def compute_qv_metrics(agent, transitions: list[Transition], *, max_windows: int | None) -> list[dict[str, float]]:
    core = agent.core
    horizon = int(core.config.action_horizon)
    discount = float(core.config.discount)
    windows = build_windows(transitions, horizon=horizon, max_windows=max_windows)
    starts = [start for start, _ in windows]
    current_obs = [sequence[0].obs for _, sequence in windows]
    next_obs = [sequence[-1].next_obs for _, sequence in windows]
    actions_np = np.stack(
        [np.stack([np.asarray(item.action, dtype=np.float32) for item in sequence], axis=0) for _, sequence in windows],
        axis=0,
    )
    discount_powers = np.asarray([discount**offset for offset in range(horizon)], dtype=np.float32)
    rewards_np = np.asarray(
        [
            float(
                np.sum(
                    np.asarray([float(get_transition_awr_fields(item)["reward"]) for item in sequence], dtype=np.float32)
                    * discount_powers
                )
            )
            for _, sequence in windows
        ],
        dtype=np.float32,
    ).reshape(-1, 1)
    dones_np = np.asarray([float(any(bool(item.done) for item in sequence)) for _, sequence in windows], dtype=np.float32).reshape(-1, 1)

    device = core.device
    image_obs, proprio = preprocess_observations(
        current_obs,
        camera_names=core.camera_names,
        image_size=int(core.config.image_size),
        proprio_mean=core.prop_mean,
        proprio_std=core.prop_std,
        device=device,
    )
    next_image_obs, next_proprio = preprocess_observations(
        next_obs,
        camera_names=core.camera_names,
        image_size=int(core.config.image_size),
        proprio_mean=core.prop_mean,
        proprio_std=core.prop_std,
        device=device,
    )
    actions = torch.as_tensor(actions_np, dtype=torch.float32, device=device)
    rewards = torch.as_tensor(rewards_np, dtype=torch.float32, device=device)
    dones = torch.as_tensor(dones_np, dtype=torch.float32, device=device)
    language = [core.language_instruction] * len(windows)

    with torch.no_grad():
        current_context = core.model.encode_multimodal_context(
            images=image_obs,
            proprio=proprio,
            language=language,
        )["task_scene_cond"]
        next_context = core.model.encode_multimodal_context(
            images=next_image_obs,
            proprio=next_proprio,
            language=language,
        )["task_scene_cond"]
        q1, q2 = core.model.forward_qs_from_context(current_context, actions)
        q_min = torch.minimum(q1, q2)
        q_mean = 0.5 * (q1 + q2)
        q_max = torch.maximum(q1, q2)
        v = core.model.forward_value_from_context(current_context)
        next_v = core.model.forward_value_from_context(next_context)
        bootstrap_discount = discount**horizon
        td_target = rewards + bootstrap_discount * (1.0 - dones) * next_v
        td_residual = td_target - q_min
        advantage = q_min - v

    metrics: list[dict[str, float]] = []
    for row_index, start in enumerate(starts):
        row = {
            "window_index": float(row_index),
            "step": float(start),
            "reward_horizon": float(rewards[row_index].detach().cpu().item()),
            "done_horizon": float(dones[row_index].detach().cpu().item()),
            "q1": float(q1[row_index].detach().cpu().item()),
            "q2": float(q2[row_index].detach().cpu().item()),
            "q_min": float(q_min[row_index].detach().cpu().item()),
            "q_mean": float(q_mean[row_index].detach().cpu().item()),
            "q_max": float(q_max[row_index].detach().cpu().item()),
            "v": float(v[row_index].detach().cpu().item()),
            "next_v": float(next_v[row_index].detach().cpu().item()),
            "td_target": float(td_target[row_index].detach().cpu().item()),
            "td_residual": float(td_residual[row_index].detach().cpu().item()),
            "advantage": float(advantage[row_index].detach().cpu().item()),
        }
        for horizon_index in range(horizon):
            for action_index in range(actions.shape[-1]):
                row[f"action_h{horizon_index}_{action_index}"] = float(actions[row_index, horizon_index, action_index].detach().cpu().item())
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
                frame = frame[::-1, ...]
                frames.append(frame)
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
    rewards = np.asarray([row["reward_horizon"] for row in metrics], dtype=np.float32)

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(title)

    axes[0].plot(steps, q_mean, label="Q mean", color="tab:blue")
    axes[0].fill_between(steps, q_min, q_max, color="tab:blue", alpha=0.18, label="Q min/max")
    axes[0].plot(steps, v, label="V", color="tab:orange")
    axes[0].plot(steps, next_v, label="next V", color="tab:green", alpha=0.8)
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

    axes[3].step(steps, rewards, where="post", label="discounted horizon reward", color="tab:gray")
    axes[3].set_ylabel("Reward")
    axes[3].set_xlabel("Step")
    axes[3].legend(loc="best")
    axes[3].grid(True, alpha=0.3)

    fig.tight_layout()
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".png"), dpi=160)
    fig.savefig(path_base.with_suffix(".pdf"))
    plt.close(fig)


def summarize_metrics(metrics: list[dict[str, float]], payload: dict[str, Any], agent) -> dict[str, Any]:
    q_values = np.asarray([row["q_mean"] for row in metrics], dtype=np.float32)
    v_values = np.asarray([row["v"] for row in metrics], dtype=np.float32)
    td_abs = np.abs(np.asarray([row["td_residual"] for row in metrics], dtype=np.float32))
    adv_values = np.asarray([row["advantage"] for row in metrics], dtype=np.float32)
    rewards = np.asarray([row["reward_horizon"] for row in metrics], dtype=np.float32)
    trainer_state = dict(payload.get("trainer_state", {}))
    return {
        "num_windows": int(len(metrics)),
        "action_horizon": int(agent.core.config.action_horizon),
        "discount": float(agent.core.config.discount),
        "value_warmup_updates": int(trainer_state.get("total_value_warmup_updates", 0)),
        "reward_horizon_mean": float(rewards.mean()),
        "reward_horizon_min": float(rewards.min()),
        "reward_horizon_max": float(rewards.max()),
        "reward_horizon_sum": float(rewards.sum()),
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
    qv_cache_path = resolve_qv_cache_path(args)
    payload = load_cache_payload(qv_cache_path)
    metadata = dict(payload.get("metadata", {}))
    if metadata.get("reward_convention") != "sparse_success_-1_0":
        print(
            "[WARN] Q/V cache metadata does not declare reward_convention=sparse_success_-1_0. "
            "This may be an old cache trained with a different reward convention."
        )
    task_data_name = str(args.task_data_name or metadata.get("task_data_name") or payload.get("task_name"))
    split_dir = resolve_split_dir(args, task_data_name)
    if not split_dir.exists():
        raise FileNotFoundError(f"Demo split directory does not exist: {split_dir}")

    camera_names = [str(name) for name in payload.get("camera_names", [])]
    if not camera_names:
        raise ValueError("Q/V cache payload does not define camera_names.")
    awr_cfg = dict(payload.get("awr_config", {}))
    image_size = int(awr_cfg.get("image_size", metadata.get("img_height", 128)))
    camera_aliases = dict(awr_cfg.get("camera_aliases", {}) or {})

    selected = select_demo(split_dir, seed=int(args.seed), demo_key=args.demo_key)
    transitions = load_demo_transitions(
        selected,
        camera_names=camera_names,
        camera_aliases={str(key): str(value) for key, value in camera_aliases.items()},
        image_size=image_size,
        renderer=str(args.renderer),
        control_freq=int(args.control_freq),
    )
    init_checkpoint = resolve_init_checkpoint(payload, args.init_checkpoint)
    algorithm_cfg = prepare_algorithm_cfg(payload, args)
    agent = build_agent(payload, algorithm_cfg, transitions, init_checkpoint)
    metrics = compute_qv_metrics(agent, transitions, max_windows=args.max_windows)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_root = Path(to_absolute_path(str(args.output_root))).resolve()
    output_dir = output_root / f"{task_data_name}_{args.split}_awr_init_qv_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "steps.csv"
    video_path = output_dir / "rollout_policy_obs.mp4"
    plot_base = output_dir / "qv_timeseries"
    summary_path = output_dir / "summary.json"

    write_metrics_csv(csv_path, metrics)
    write_video(video_path, transitions, image_keys=camera_names, fps=int(args.video_fps))
    plot_qv(plot_base, metrics, title=f"{task_data_name} {args.split} {selected.hdf5_path.name}::{selected.demo_key}")

    summary = {
        "qv_cache": str(qv_cache_path),
        "qv_cache_reward_convention": metadata.get("reward_convention"),
        "init_checkpoint": str(init_checkpoint),
        "task_data_name": task_data_name,
        "task_name": str(payload.get("task_name")),
        "split": str(args.split),
        "selected_hdf5": str(selected.hdf5_path),
        "selected_demo_key": selected.demo_key,
        "selected_demo_length": int(selected.length),
        "selected_demo_successful": bool(selected.successful),
        "used_windows": int(len(metrics)),
        "camera_names": camera_names,
        "image_size": int(image_size),
        "device": str(agent.core.device),
        "inference_device": str(agent.core.inference_device),
        "outputs": {
            "steps_csv": str(csv_path),
            "video": str(video_path),
            "plot_png": str(plot_base.with_suffix(".png")),
            "plot_pdf": str(plot_base.with_suffix(".pdf")),
        },
        "metrics_summary": summarize_metrics(metrics, payload, agent),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"selected_demo={selected.hdf5_path}::{selected.demo_key}")
    print(f"qv_cache={qv_cache_path}")
    print(f"init_checkpoint={init_checkpoint}")
    print(f"output_dir={output_dir}")
    print(f"video={video_path}")
    print(f"plot_png={plot_base.with_suffix('.png')}")
    print(f"plot_pdf={plot_base.with_suffix('.pdf')}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
