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
from omegaconf import OmegaConf

from robosuite.pipeline.common import Transition
from robosuite.pipeline.common.utils import nested_to_torch, stack_tree
from robosuite.pipeline.envs import load_hdf5_demos_into_transitions
from robosuite.pipeline.factory import build_algorithm
from robosuite.pipeline.utils import resolve_algorithm_devices


DEFAULT_CHECKPOINT = (
    "outputs/hil_serl/hil_serl_PickPlaceCereal_2026-04-28_11-28-39/checkpoints/latest.pt"
)
DEFAULT_SUCCESS_DIR = "data/PickPlaceCereal/success_rollout"
DEFAULT_OUTPUT_ROOT = "outputs/hil_serl/qv_visualization"


@dataclass
class SelectedDemo:
    hdf5_path: Path
    demo_key: str
    length: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize HIL-SERL Q/V on an existing success rollout demo."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--success-rollout-dir", default=DEFAULT_SUCCESS_DIR)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--mc-action-samples", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--inference-device", default=None)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--renderer", default="mjviewer")
    parser.add_argument("--control-freq", type=int, default=20)
    return parser.parse_args()


def resolve_run_dir(checkpoint_path: Path) -> Path | None:
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent
    return None


def load_resolved_config(checkpoint_path: Path):
    run_dir = resolve_run_dir(checkpoint_path)
    if run_dir is None:
        return None
    config_path = run_dir / "config_resolved.yaml"
    if not config_path.exists():
        return None
    return OmegaConf.load(config_path)


def checkpoint_algorithm_cfg(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "hil-serl",
        "encoder": dict(payload.get("encoder_config", {})),
        "sac": dict(payload.get("sac_config", {})),
        "trainer": dict(payload.get("trainer_config", {})),
        "online_buffer": {"capacity": 1},
        "demo_buffer": {"capacity": 1},
    }


def prepare_algorithm_cfg(payload: dict[str, Any], checkpoint_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    resolved_cfg = load_resolved_config(checkpoint_path)
    if resolved_cfg is not None and hasattr(resolved_cfg, "algorithm"):
        algorithm_cfg = OmegaConf.to_container(resolved_cfg.algorithm, resolve=True)
    else:
        algorithm_cfg = checkpoint_algorithm_cfg(payload)
    if not isinstance(algorithm_cfg, dict):
        raise TypeError("Algorithm config must resolve to a dictionary.")

    encoder_cfg = algorithm_cfg.get("encoder", None)
    if isinstance(encoder_cfg, dict) and encoder_cfg.get("pretrained_path"):
        encoder_cfg["pretrained_path"] = to_absolute_path(str(encoder_cfg["pretrained_path"]))

    sac_cfg = algorithm_cfg.setdefault("sac", {})
    if args.device is not None:
        sac_cfg["device"] = str(args.device)
    if args.inference_device is not None:
        sac_cfg["inference_device"] = str(args.inference_device)
    resolve_algorithm_devices(algorithm_cfg)
    return algorithm_cfg


def select_random_success_demo(success_rollout_dir: Path, seed: int) -> SelectedDemo:
    hdf5_paths = sorted(success_rollout_dir.glob("*.hdf5")) + sorted(success_rollout_dir.glob("*.h5"))
    if not hdf5_paths:
        raise FileNotFoundError(f"No HDF5 files found under {success_rollout_dir}.")

    candidates: list[SelectedDemo] = []
    for hdf5_path in hdf5_paths:
        with h5py.File(hdf5_path, "r") as file_handle:
            demo_root = file_handle["demos"] if "demos" in file_handle else file_handle["data"]
            for demo_key in sorted(demo_root.keys()):
                demo_group = demo_root[demo_key]
                if not bool(demo_group.attrs.get("successful", True)):
                    continue
                length = int(demo_group.attrs.get("length", len(demo_group["actions"])))
                candidates.append(SelectedDemo(hdf5_path=hdf5_path, demo_key=str(demo_key), length=length))

    if not candidates:
        raise RuntimeError(f"No successful demos found under {success_rollout_dir}.")
    rng = random.Random(int(seed))
    return rng.choice(candidates)


def load_success_transitions(
    selected: SelectedDemo,
    *,
    image_keys: list[str],
    image_size: int,
    renderer: str,
    control_freq: int,
) -> list[Transition]:
    transitions = load_hdf5_demos_into_transitions(
        selected.hdf5_path,
        camera_names=image_keys,
        img_height=int(image_size),
        img_width=int(image_size),
        proprio_keys=(),
        renderer=str(renderer),
        control_freq=int(control_freq),
        demo_names=[selected.demo_key],
    )
    if not transitions:
        raise RuntimeError(f"Selected demo produced zero transitions: {selected.hdf5_path}::{selected.demo_key}")
    return transitions


def crop_at_first_success_or_terminal(transitions: list[Transition]) -> tuple[list[Transition], int | None]:
    for index, transition in enumerate(transitions):
        reward = None if transition.reward is None else float(transition.reward)
        if bool(transition.done) or (reward is not None and reward >= 0.0):
            return transitions[: index + 1], index
    return transitions, None


def build_agent(payload: dict[str, Any], algorithm_cfg: dict[str, Any], transitions: list[Transition]):
    action_dim = int(np.asarray(transitions[0].action).reshape(-1).shape[0])
    action_low = -np.ones(action_dim, dtype=np.float32)
    action_high = np.ones(action_dim, dtype=np.float32)
    agent = build_algorithm(
        algorithm_cfg,
        observation_example=transitions[0].obs,
        sample_action=np.zeros(action_dim, dtype=np.float32),
        action_low=action_low,
        action_high=action_high,
    )
    agent.core.load_state_dict(payload["core"])
    set_eval_mode(agent)
    return agent


def set_eval_mode(agent) -> None:
    modules = [
        agent.core.encoder,
        agent.core.target_encoder,
        agent.core.actor,
        agent.core.critic,
        agent.core.grasp_critic,
        agent.core.target_critic,
        agent.core.target_grasp_critic,
    ]
    for module in modules:
        module.eval()


def compute_qv_metrics(agent, transitions: list[Transition], *, mc_action_samples: int) -> list[dict[str, float]]:
    core = agent.core
    obs = nested_to_torch(stack_tree([transition.obs for transition in transitions]), device=core.device)
    next_obs = nested_to_torch(stack_tree([transition.next_obs for transition in transitions]), device=core.device)
    actions = torch.as_tensor(
        np.stack([np.asarray(transition.action, dtype=np.float32).reshape(-1) for transition in transitions], axis=0),
        device=core.device,
        dtype=torch.float32,
    )
    rewards = torch.as_tensor(
        np.asarray([float(transition.reward) for transition in transitions], dtype=np.float32).reshape(-1, 1),
        device=core.device,
    )
    dones = torch.as_tensor(
        np.asarray([float(transition.done) for transition in transitions], dtype=np.float32).reshape(-1, 1),
        device=core.device,
    )
    grasp_penalty = torch.as_tensor(
        np.asarray(
            [0.0 if transition.grasp_penalty is None else float(transition.grasp_penalty) for transition in transitions],
            dtype=np.float32,
        ).reshape(-1, 1),
        device=core.device,
    )

    with torch.no_grad():
        features = core.encoder(obs)
        next_features = core.encoder(next_obs)
        q_ensemble = core.critic(features, actions[..., :-1])
        q_min = q_ensemble.min(dim=0).values
        q_mean = q_ensemble.mean(dim=0)
        q_max = q_ensemble.max(dim=0).values

        grasp_qs = core.grasp_critic(features)
        grasp_indices = core._grasp_action_indices(actions[..., -1:]).squeeze(-1)
        grasp_q_selected = grasp_qs.gather(-1, grasp_indices.unsqueeze(-1))
        next_grasp_v = core.target_grasp_critic(next_features).max(dim=-1, keepdim=True).values

        v_soft = estimate_soft_value(core, features, samples=int(mc_action_samples))
        next_v_soft = estimate_soft_value(core, next_features, samples=int(mc_action_samples), use_target_critic=True)
        td_target = rewards + float(core.config.discount) * (1.0 - dones) * next_v_soft
        td_residual = td_target - q_min

        grasp_rewards = rewards + grasp_penalty
        grasp_td_target = grasp_rewards + float(core.config.discount) * (1.0 - dones) * next_grasp_v
        grasp_td_residual = grasp_td_target - grasp_q_selected

    metrics: list[dict[str, float]] = []
    grasp_values = core.grasp_action_values_np.tolist()
    for index in range(len(transitions)):
        row = {
            "step": float(index),
            "reward": float(rewards[index].detach().cpu().item()),
            "done": float(dones[index].detach().cpu().item()),
            "q_cont_min": float(q_min[index].detach().cpu().item()),
            "q_cont_mean": float(q_mean[index].detach().cpu().item()),
            "q_cont_max": float(q_max[index].detach().cpu().item()),
            "v_cont_soft": float(v_soft[index].detach().cpu().item()),
            "td_target": float(td_target[index].detach().cpu().item()),
            "td_residual": float(td_residual[index].detach().cpu().item()),
            "grasp_action": float(actions[index, -1].detach().cpu().item()),
            "grasp_action_index": float(grasp_indices[index].detach().cpu().item()),
            "grasp_q_selected": float(grasp_q_selected[index].detach().cpu().item()),
            "grasp_td_target": float(grasp_td_target[index].detach().cpu().item()),
            "grasp_td_residual": float(grasp_td_residual[index].detach().cpu().item()),
        }
        for grasp_index, grasp_value in enumerate(grasp_values):
            row[f"grasp_value_{grasp_index}"] = float(grasp_value)
            row[f"grasp_q_{grasp_index}"] = float(grasp_qs[index, grasp_index].detach().cpu().item())
        for action_index in range(actions.shape[-1]):
            row[f"action_{action_index}"] = float(actions[index, action_index].detach().cpu().item())
        metrics.append(row)
    return metrics


def estimate_soft_value(core, features: torch.Tensor, *, samples: int, use_target_critic: bool = False) -> torch.Tensor:
    samples = max(1, int(samples))
    values = []
    critic = core.target_critic if use_target_critic else core.critic
    for _ in range(samples):
        sampled_actions, log_probs = core.actor.sample(features, deterministic=False)
        q_values = critic(features, sampled_actions).min(dim=0).values
        values.append(q_values - core.alpha.detach() * log_probs)
    return torch.stack(values, dim=0).mean(dim=0)


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
                if key in transition.obs:
                    # in robosuite the view is upside down
                    frames.append(np.asarray(transition.obs[key][::-1, ...], dtype=np.uint8))
            if not frames:
                continue
            frame = np.concatenate(frames, axis=1) if len(frames) > 1 else frames[0]
            writer.append_data(frame)


def plot_qv(path_base: Path, metrics: list[dict[str, float]], title: str) -> None:
    steps = np.asarray([row["step"] for row in metrics], dtype=np.float32)
    q_min = np.asarray([row["q_cont_min"] for row in metrics], dtype=np.float32)
    q_mean = np.asarray([row["q_cont_mean"] for row in metrics], dtype=np.float32)
    q_max = np.asarray([row["q_cont_max"] for row in metrics], dtype=np.float32)
    v_soft = np.asarray([row["v_cont_soft"] for row in metrics], dtype=np.float32)
    td_residual = np.asarray([row["td_residual"] for row in metrics], dtype=np.float32)
    grasp_selected = np.asarray([row["grasp_q_selected"] for row in metrics], dtype=np.float32)
    grasp_td_residual = np.asarray([row["grasp_td_residual"] for row in metrics], dtype=np.float32)
    rewards = np.asarray([row["reward"] for row in metrics], dtype=np.float32)

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(title)

    axes[0].plot(steps, q_mean, label="Q mean", color="tab:blue")
    axes[0].fill_between(steps, q_min, q_max, color="tab:blue", alpha=0.18, label="Q min/max")
    axes[0].plot(steps, v_soft, label="V soft estimate", color="tab:orange")
    axes[0].set_ylabel("Q / V")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, td_residual, label="continuous TD residual", color="tab:red")
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_ylabel("TD residual")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps, grasp_selected, label="selected grasp Q", color="tab:green")
    for grasp_index in range(3):
        values = np.asarray([row[f"grasp_q_{grasp_index}"] for row in metrics], dtype=np.float32)
        axes[2].plot(steps, values, linestyle="--", alpha=0.7, label=f"grasp Q {grasp_index}")
    axes[2].set_ylabel("Grasp Q")
    axes[2].legend(loc="best")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(steps, grasp_td_residual, label="grasp TD residual", color="tab:purple")
    axes[3].step(steps, rewards, where="post", label="reward", color="tab:gray", alpha=0.8)
    axes[3].axhline(0.0, color="black", linewidth=1)
    axes[3].set_ylabel("Residual / reward")
    axes[3].set_xlabel("Step")
    axes[3].legend(loc="best")
    axes[3].grid(True, alpha=0.3)

    fig.tight_layout()
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".png"), dpi=160)
    fig.savefig(path_base.with_suffix(".pdf"))
    plt.close(fig)


def summarize_metrics(metrics: list[dict[str, float]], gamma: float) -> dict[str, Any]:
    q_values = np.asarray([row["q_cont_mean"] for row in metrics], dtype=np.float32)
    v_values = np.asarray([row["v_cont_soft"] for row in metrics], dtype=np.float32)
    td_abs = np.abs(np.asarray([row["td_residual"] for row in metrics], dtype=np.float32))
    lower_bound = -1.0 / max(1e-6, 1.0 - float(gamma))
    upper_bound = 0.0
    combined = np.concatenate([q_values, v_values], axis=0)
    out_of_range = np.logical_or(combined < lower_bound, combined > upper_bound)
    return {
        "num_steps": int(len(metrics)),
        "reward_sum": float(sum(row["reward"] for row in metrics)),
        "q_mean_min": float(q_values.min()),
        "q_mean_max": float(q_values.max()),
        "v_soft_min": float(v_values.min()),
        "v_soft_max": float(v_values.max()),
        "mean_abs_td_residual": float(td_abs.mean()),
        "max_abs_td_residual": float(td_abs.max()),
        "expected_negative_reward_value_range": [float(lower_bound), float(upper_bound)],
        "qv_out_of_expected_range_fraction": float(out_of_range.mean()),
    }


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(to_absolute_path(args.checkpoint)).resolve()
    success_rollout_dir = Path(to_absolute_path(args.success_rollout_dir)).resolve()
    output_root = Path(to_absolute_path(args.output_root)).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    if not success_rollout_dir.exists():
        raise FileNotFoundError(f"Success rollout directory does not exist: {success_rollout_dir}")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    algorithm_cfg = prepare_algorithm_cfg(payload, checkpoint_path, args)
    encoder_cfg = dict(algorithm_cfg.get("encoder", {}))
    image_keys = [str(key) for key in encoder_cfg.get("image_keys", [])]
    if not image_keys:
        raise ValueError("Checkpoint encoder config does not define image_keys.")
    image_size = int(encoder_cfg.get("image_size", 128))

    selected = select_random_success_demo(success_rollout_dir, seed=int(args.seed))
    transitions = load_success_transitions(
        selected,
        image_keys=image_keys,
        image_size=image_size,
        renderer=str(args.renderer),
        control_freq=int(args.control_freq),
    )
    original_transition_count = len(transitions)
    transitions, first_terminal_index = crop_at_first_success_or_terminal(transitions)
    if args.max_steps is not None:
        transitions = transitions[: max(1, int(args.max_steps))]

    agent = build_agent(payload, algorithm_cfg, transitions)
    metrics = compute_qv_metrics(agent, transitions, mc_action_samples=int(args.mc_action_samples))

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = output_root / f"PickPlaceCereal_success_qv_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "steps.csv"
    video_path = output_dir / "success_rollout_policy_obs.mp4"
    plot_base = output_dir / "qv_timeseries"
    summary_path = output_dir / "summary.json"

    write_metrics_csv(csv_path, metrics)
    write_video(video_path, transitions, image_keys=image_keys, fps=int(args.video_fps))
    plot_qv(plot_base, metrics, title=f"{selected.hdf5_path.name}::{selected.demo_key}")

    summary = {
        "checkpoint": str(checkpoint_path),
        "success_rollout_dir": str(success_rollout_dir),
        "selected_hdf5": str(selected.hdf5_path),
        "selected_demo_key": selected.demo_key,
        "selected_demo_length": int(selected.length),
        "original_loaded_steps": int(original_transition_count),
        "first_success_or_terminal_index": (
            None if first_terminal_index is None else int(first_terminal_index)
        ),
        "used_steps": int(len(transitions)),
        "image_keys": image_keys,
        "image_size": int(image_size),
        "mc_action_samples": int(args.mc_action_samples),
        "device": str(algorithm_cfg.get("sac", {}).get("device")),
        "inference_device": str(algorithm_cfg.get("sac", {}).get("inference_device")),
        "outputs": {
            "steps_csv": str(csv_path),
            "video": str(video_path),
            "plot_png": str(plot_base.with_suffix(".png")),
            "plot_pdf": str(plot_base.with_suffix(".pdf")),
        },
        "metrics_summary": summarize_metrics(metrics, gamma=float(agent.core.config.discount)),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"selected_demo={selected.hdf5_path}::{selected.demo_key}")
    print(f"output_dir={output_dir}")
    print(f"video={video_path}")
    print(f"plot_png={plot_base.with_suffix('.png')}")
    print(f"plot_pdf={plot_base.with_suffix('.pdf')}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
