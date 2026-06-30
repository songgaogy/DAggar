"""Open-loop visualization for a pretrained flow-dagger policy on HDF5 demos.

For each selected expert trajectory, the script starts at observation index 0,
plans an action chunk, then advances by ``chunk_size`` and repeats from the
stored expert observation at that next index. It compares the inferred
non-overlapping policy actions against the expert actions on the same timeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from tqdm import tqdm

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from robosuite.pipeline.algorithms.flow_dagger.common import FlowDaggerConfig
from robosuite.pipeline.algorithms.flow_dagger.models.flow import (
    FlowDaggerPolicy,
    _sample_action_sequence,
)
from robosuite.pipeline.envs import build_robosuite_env
from robosuite.pipeline.train_dipole import (
    _center_crop_resize_image,
    _reset_flow_env_for_demo,
    _resolve_hdf5_demo_group,
    bind_flow_proprio_extractor,
    build_flow_runtime_cfg,
    normalize_policy_observation,
    resolve_flow_task_metadata,
)
from robosuite.pipeline.utils import read_hdf5_camera_names, read_hdf5_env_info, resolve_requested_device
from robosuite.policy.flow_multi_update.utils.datasets import DEFAULT_TASK_PROMPTS


@dataclass
class PlotSeries:
    expert: list[list[float]]
    pred: list[list[float]]


@dataclass
class DemoResult:
    demo_name: str
    output_dir: str
    num_chunks: int
    num_action_steps: int
    mse_by_dim: list[float]
    mae_by_dim: list[float]
    rmse_by_dim: list[float]
    mean_mse: float
    mean_mae: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Flow-dagger / flow-multi checkpoint.")
    parser.add_argument("--data", required=True, help="Expert pretrain HDF5 file.")
    parser.add_argument("--task-name", required=True, help="Task name used for language prompt metadata.")
    parser.add_argument("--env-name", default=None, help="Robosuite env name; defaults to --task-name.")
    parser.add_argument("--output-dir", required=True, help="Directory where CSV and plots are written.")
    parser.add_argument("--chunk-size", type=int, default=8, help="Expert/policy chunk length to compare.")
    parser.add_argument("--max-demos", type=int, default=0, help="Limit selected demos (<=0 = all).")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit total chunk starts (<=0 = all).")
    parser.add_argument("--max-plot-points", type=int, default=10000, help="Max trajectory points per plot.")
    parser.add_argument("--device", default=None, help="Inference device override, e.g. cuda:0.")
    parser.add_argument("--deterministic", action="store_true", help="Use zero-noise ODE initialization.")
    parser.add_argument("--seed", type=int, default=42, help="Torch / NumPy seed for stochastic action sampling.")
    return parser.parse_args()


def _as_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _resolve_language_instruction(task_name: str, task_prompt_map: dict[str, Any] | None) -> str:
    prompt_map = {} if task_prompt_map is None else dict(task_prompt_map)
    prompt_value = prompt_map.get(task_name, DEFAULT_TASK_PROMPTS.get(task_name, task_name))
    if isinstance(prompt_value, str):
        return str(prompt_value)
    if isinstance(prompt_value, (list, tuple)) and len(prompt_value) > 0:
        return str(prompt_value[0])
    return str(task_name)


def _infer_first_action_dim(data_path: Path) -> int:
    with h5py.File(data_path, "r") as file_handle:
        demos_group = _resolve_hdf5_demo_group(file_handle)
        for demo_name in sorted(demos_group.keys()):
            actions = np.asarray(demos_group[demo_name]["actions"])
            if actions.ndim == 2 and actions.shape[0] > 0:
                return int(actions.shape[-1])
    raise RuntimeError(f"Unable to infer action_dim from {data_path}: no non-empty demo actions.")


def _build_flow_policy(
    *,
    checkpoint_path: Path,
    data_path: Path,
    task_name: str,
    device_override: str | None,
    chunk_size: int,
) -> tuple[FlowDaggerPolicy, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state = payload.get("ema_model", payload.get("model"))
    if model_state is None:
        raise KeyError(f"Checkpoint {checkpoint_path} is missing both 'ema_model' and 'model'.")
    if "model_cfg" not in payload:
        raise KeyError(f"Checkpoint {checkpoint_path} is missing 'model_cfg'.")

    act_mean = _as_numpy(payload.get("act_mean"))
    act_std = _as_numpy(payload.get("act_std"))
    prop_mean = _as_numpy(payload.get("prop_mean"))
    prop_std = _as_numpy(payload.get("prop_std"))
    action_horizon = int(act_mean.shape[0]) if act_mean is not None else int(chunk_size)
    action_dim = int(act_mean.shape[-1]) if act_mean is not None else _infer_first_action_dim(data_path)
    if int(chunk_size) > action_horizon:
        raise ValueError(
            f"--chunk-size ({chunk_size}) cannot exceed checkpoint action_horizon ({action_horizon})."
        )
    proprio_dim = int(prop_mean.reshape(-1).shape[0]) if prop_mean is not None else 0
    if proprio_dim <= 0:
        raise ValueError(
            f"Checkpoint {checkpoint_path} is missing prop_mean/prop_std; cannot infer proprio_dim safely."
        )

    cfg = dict(payload.get("cfg", {}) or {})
    data_cfg = dict(cfg.get("data", {}) or {})
    train_cfg = dict(cfg.get("train", {}) or {})
    device = resolve_requested_device(
        device_override or train_cfg.get("device", "cpu"),
        fallback="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    camera_names = [str(name) for name in payload.get("camera_names", [])]
    if not camera_names:
        camera_names = read_hdf5_camera_names(data_path)
    if not camera_names:
        raise ValueError("Unable to resolve policy camera names from checkpoint or HDF5 attrs.")

    config = FlowDaggerConfig(
        action_dim=action_dim,
        proprio_dim=proprio_dim,
        action_horizon=action_horizon,
        execute_horizon=action_horizon,
        image_size=int(data_cfg.get("image_size", 128)),
        learning_rate=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-6)),
        grad_clip_norm=float(train_cfg.get("grad_clip_norm", 1.0)),
        lambda_endpoint=float(train_cfg.get("lambda_endpoint", 0.5)),
        lambda_smooth=float(train_cfg.get("lambda_smooth", 0.05)),
        n_ode_steps=int(cfg.get("eval", {}).get("n_ode_steps", 8)) if isinstance(cfg.get("eval", {}), dict) else 8,
        device=str(device),
        inference_device=str(device),
        task_name=str(task_name),
        language_instruction=_resolve_language_instruction(str(task_name), payload.get("task_prompt_map")),
    )
    policy = FlowDaggerPolicy(
        model_cfg=dict(payload["model_cfg"]),
        config=config,
        camera_names=camera_names,
    )
    policy.load_model_state(model_state, strict=True)
    policy.set_normalizers(
        action_mean=act_mean,
        action_std=act_std,
        proprio_mean=prop_mean,
        proprio_std=prop_std,
    )
    policy.set_language_instruction(config.language_instruction or str(task_name))
    policy.reset_action_chunk()
    return policy, payload


def _plan_action_chunk(policy: FlowDaggerPolicy, obs: dict[str, Any], *, deterministic: bool) -> np.ndarray:
    images = []
    for camera_name in policy.camera_names:
        image = np.asarray(obs[camera_name], dtype=np.uint8)
        image = _center_crop_resize_image(
            image,
            img_height=int(policy.config.image_size),
            img_width=int(policy.config.image_size),
        )
        images.append(np.transpose(image.astype(np.float32) / 255.0, (2, 0, 1)))
    image_tensor = torch.from_numpy(np.stack(images, axis=0)).unsqueeze(0).to(policy.inference_device)
    image_tensor = (image_tensor - policy._inference_image_mean) / policy._inference_image_std

    proprio = np.asarray(obs["state"], dtype=np.float32)
    if policy.prop_mean is not None and policy.prop_std is not None:
        proprio = (proprio - policy.prop_mean) / (policy.prop_std + 1e-6)
    proprio_tensor = torch.from_numpy(proprio).unsqueeze(0).to(policy.inference_device)

    with policy._inference_lock:
        action_seq = _sample_action_sequence(
            policy.inference_model,
            images=image_tensor,
            proprio=proprio_tensor,
            language=[policy.language_instruction],
            action_horizon=int(policy.config.action_horizon),
            n_steps=int(policy.config.n_ode_steps),
            deterministic=bool(deterministic),
        )[0].detach().cpu().numpy().astype(np.float32)
    if policy.act_mean is not None and policy.act_std is not None:
        action_seq = action_seq * policy.act_std + policy.act_mean
    return np.asarray(action_seq, dtype=np.float32)


def _resolve_env_metadata(payload: dict[str, Any], data_path: Path, task_name: str) -> dict[str, Any]:
    env_metadata = resolve_flow_task_metadata(payload, task_name)
    if env_metadata is not None:
        return env_metadata
    return read_hdf5_env_info(data_path)


def _build_proprio_env(
    *,
    payload: dict[str, Any],
    data_path: Path,
    task_name: str,
    env_name: str,
    image_size: int,
    camera_names: list[str],
):
    env_metadata = _resolve_env_metadata(payload, data_path, task_name)
    cfg = OmegaConf.create(
        {
            "env": {
                "environment": str(env_name),
                "renderer": "mjviewer",
                "img_height": int(image_size),
                "img_width": int(image_size),
                "proprio_keys": [],
                "control_freq": 20,
                "horizon": None,
            }
        }
    )
    runtime_cfg = build_flow_runtime_cfg(
        cfg,
        env_metadata=env_metadata,
        camera_names=list(camera_names),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        renderer="mjviewer",
    )
    env = build_robosuite_env(runtime_cfg)
    extractor = bind_flow_proprio_extractor(env, env_metadata)
    return env, extractor


def _select_demo_names(
    data_path: Path,
    *,
    max_demos: int,
) -> list[str]:
    with h5py.File(data_path, "r") as file_handle:
        demos_group = _resolve_hdf5_demo_group(file_handle)
        demo_names = sorted(str(name) for name in demos_group.keys())
        if max_demos > 0:
            demo_names = demo_names[: int(max_demos)]
    return demo_names


def _decode_model_xml(demo_group: h5py.Group) -> str | None:
    model_xml = demo_group.attrs.get("model_file", None)
    if isinstance(model_xml, bytes):
        model_xml = model_xml.decode("utf-8")
    return str(model_xml) if model_xml else None


def _chunk_starts(num_actions: int, chunk_size: int) -> list[int]:
    usable = max(0, int(num_actions) - int(chunk_size) + 1)
    return list(range(0, usable, int(chunk_size)))


def _count_chunks(
    *,
    demos_group: h5py.Group,
    demo_names: list[str],
    chunk_size: int,
    max_frames: int,
) -> int:
    count = 0
    for demo_name in demo_names:
        count += len(_chunk_starts(int(len(demos_group[demo_name]["actions"])), int(chunk_size)))
        if max_frames > 0 and count >= int(max_frames):
            return int(max_frames)
    return count


def _load_step_obs(
    *,
    states: np.ndarray,
    obs_group: h5py.Group,
    step_idx: int,
    extractor,
    policy_camera_names: list[str],
    image_size: int,
) -> dict[str, Any]:
    required_hdf5_camera_names = set(policy_camera_names)
    for camera_name in required_hdf5_camera_names:
        if camera_name not in obs_group:
            raise KeyError(f"Missing camera '{camera_name}' in HDF5 observation group.")
    raw_images = {
        camera_name: _center_crop_resize_image(
            np.asarray(obs_group[camera_name]["images"][step_idx], dtype=np.uint8),
            img_height=int(image_size),
            img_width=int(image_size),
        )
        for camera_name in required_hdf5_camera_names
    }
    obs_state = extractor.extract(states[step_idx]).astype(np.float32)
    return normalize_policy_observation(
        {**raw_images, "state": obs_state},
        policy_camera_names=policy_camera_names,
        camera_aliases={},
    )


def _downsample_indices(length: int, max_points: int) -> np.ndarray:
    if max_points <= 0 or length <= max_points:
        return np.arange(length, dtype=np.int64)
    return np.linspace(0, length - 1, int(max_points)).astype(np.int64)


def _save_dim_plot(
    *,
    path: Path,
    expert: np.ndarray,
    pred: np.ndarray,
    dim: int,
    max_plot_points: int,
) -> None:
    indices = _downsample_indices(len(expert), max_plot_points)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(indices, expert[indices], label="expert", linewidth=1.0)
    ax.plot(indices, pred[indices], label="policy", linewidth=1.0, alpha=0.85)
    ax.set_title(f"Action dim {dim}")
    ax.set_xlabel("trajectory action index")
    ax.set_ylabel("action")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_all_dims_plot(
    *,
    path: Path,
    expert_by_dim: np.ndarray,
    pred_by_dim: np.ndarray,
    max_plot_points: int,
) -> None:
    action_dim = int(expert_by_dim.shape[0])
    fig_height = max(3.0, min(2.2 * action_dim, 28.0))
    fig, axes = plt.subplots(action_dim, 1, figsize=(15, fig_height), sharex=True, squeeze=False)
    for dim in range(action_dim):
        expert = expert_by_dim[dim]
        pred = pred_by_dim[dim]
        indices = _downsample_indices(len(expert), max_plot_points)
        ax = axes[dim, 0]
        ax.plot(indices, expert[indices], label="expert", linewidth=0.9)
        ax.plot(indices, pred[indices], label="policy", linewidth=0.9, alpha=0.85)
        rmse = float(np.sqrt(np.mean((pred - expert) ** 2))) if len(expert) else float("nan")
        ax.set_ylabel(f"dim {dim}\nRMSE {rmse:.4f}")
        ax.grid(True, alpha=0.2)
        if dim == 0:
            ax.legend(loc="best")
    axes[-1, 0].set_xlabel("trajectory action index")
    fig.suptitle("Open-loop policy actions vs expert actions")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _series_to_arrays(series: PlotSeries) -> tuple[np.ndarray, np.ndarray]:
    expert_by_dim = np.asarray([np.asarray(values, dtype=np.float32) for values in series.expert])
    pred_by_dim = np.asarray([np.asarray(values, dtype=np.float32) for values in series.pred])
    return expert_by_dim, pred_by_dim


def _metrics_from_arrays(expert_by_dim: np.ndarray, pred_by_dim: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    errors = pred_by_dim - expert_by_dim
    mse_by_dim = np.mean(errors**2, axis=1)
    mae_by_dim = np.mean(np.abs(errors), axis=1)
    return mse_by_dim, mae_by_dim


def _write_demo_summary(
    *,
    demo_dir: Path,
    demo_name: str,
    num_chunks: int,
    chunk_size: int,
    mse_by_dim: np.ndarray,
    mae_by_dim: np.ndarray,
) -> DemoResult:
    result = DemoResult(
        demo_name=str(demo_name),
        output_dir=str(demo_dir),
        num_chunks=int(num_chunks),
        num_action_steps=int(num_chunks) * int(chunk_size),
        mse_by_dim=[float(x) for x in mse_by_dim],
        mae_by_dim=[float(x) for x in mae_by_dim],
        rmse_by_dim=[float(np.sqrt(x)) for x in mse_by_dim],
        mean_mse=float(np.mean(mse_by_dim)),
        mean_mae=float(np.mean(mae_by_dim)),
    )
    (demo_dir / "summary.json").write_text(json.dumps(result.__dict__, indent=2), encoding="utf-8")
    return result


def _write_global_metadata(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    checkpoint_path: Path,
    data_path: Path,
    policy: FlowDaggerPolicy,
    demo_results: list[DemoResult],
    global_mse_by_dim: np.ndarray,
    global_mae_by_dim: np.ndarray,
) -> None:
    num_chunks = sum(result.num_chunks for result in demo_results)
    num_action_steps = sum(result.num_action_steps for result in demo_results)
    summary = {
        "checkpoint": str(checkpoint_path),
        "data": str(data_path),
        "task_name": str(args.task_name),
        "env_name": str(args.env_name or args.task_name),
        "chunk_size": int(args.chunk_size),
        "num_demos": len(demo_results),
        "num_chunks": int(num_chunks),
        "num_action_steps": int(num_action_steps),
        "rollout_mode": "non_overlapping_hdf5_observation_stride",
        "deterministic": bool(args.deterministic),
        "seed": int(args.seed),
        "device": str(policy.inference_device),
        "camera_names": list(policy.camera_names),
        "language_instruction": str(policy.language_instruction),
        "mse_by_dim": [float(x) for x in global_mse_by_dim],
        "mae_by_dim": [float(x) for x in global_mae_by_dim],
        "rmse_by_dim": [float(np.sqrt(x)) for x in global_mse_by_dim],
        "mean_mse": float(np.mean(global_mse_by_dim)),
        "mean_mae": float(np.mean(global_mae_by_dim)),
        "demos": [result.__dict__ for result in demo_results],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    if int(args.chunk_size) <= 0:
        raise ValueError("--chunk-size must be positive.")

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    checkpoint_path = Path(to_absolute_path(args.checkpoint)).resolve()
    data_path = Path(to_absolute_path(args.data)).resolve()
    output_dir = Path(to_absolute_path(args.output_dir)).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"HDF5 data does not exist: {data_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    policy, payload = _build_flow_policy(
        checkpoint_path=checkpoint_path,
        data_path=data_path,
        task_name=str(args.task_name),
        device_override=args.device,
        chunk_size=int(args.chunk_size),
    )
    env, extractor = _build_proprio_env(
        payload=payload,
        data_path=data_path,
        task_name=str(args.task_name),
        env_name=str(args.env_name or args.task_name),
        image_size=int(policy.config.image_size),
        camera_names=list(policy.camera_names),
    )
    selected_demo_names = _select_demo_names(
        data_path,
        max_demos=int(args.max_demos),
    )
    if not selected_demo_names:
        raise RuntimeError(
            f"No demos found in {data_path}; max_demos={args.max_demos}."
        )

    action_dim = int(policy.config.action_dim)
    global_series = PlotSeries(expert=[[] for _ in range(action_dim)], pred=[[] for _ in range(action_dim)])
    demo_results: list[DemoResult] = []
    total_processed_chunks = 0
    try:
        with h5py.File(data_path, "r") as file_handle:
            demos_group = _resolve_hdf5_demo_group(file_handle)
            total_chunks = _count_chunks(
                demos_group=demos_group,
                demo_names=selected_demo_names,
                chunk_size=int(args.chunk_size),
                max_frames=int(args.max_frames),
            )
            if total_chunks <= 0:
                raise RuntimeError(
                    f"No usable non-overlapping chunks found in {data_path}; chunk_size={args.chunk_size}."
                )
            with tqdm(total=total_chunks, desc="openloop") as progress:
                for demo_name in selected_demo_names:
                    demo_group = demos_group[demo_name]
                    states = np.asarray(demo_group["states"])
                    actions = np.asarray(demo_group["actions"], dtype=np.float32)
                    obs_group = demo_group["observations"]
                    starts = _chunk_starts(int(len(actions)), int(args.chunk_size))
                    if not starts:
                        continue
                    demo_dir = output_dir / str(demo_name)
                    demo_dir.mkdir(parents=True, exist_ok=True)
                    demo_series = PlotSeries(expert=[[] for _ in range(action_dim)], pred=[[] for _ in range(action_dim)])
                    demo_chunks = 0
                    _reset_flow_env_for_demo(env, extractor, _decode_model_xml(demo_group))
                    csv_path = demo_dir / "actions.csv"
                    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
                        writer = csv.writer(csv_file)
                        writer.writerow(
                            [
                                "demo",
                                "chunk_start",
                                "step",
                                "action_dim",
                                "expert",
                                "policy",
                                "error",
                                "abs_error",
                                "sq_error",
                            ]
                        )
                        for chunk_start in starts:
                            if int(args.max_frames) > 0 and total_processed_chunks >= int(args.max_frames):
                                break
                            obs = _load_step_obs(
                                states=states,
                                obs_group=obs_group,
                                step_idx=int(chunk_start),
                                extractor=extractor,
                                policy_camera_names=list(policy.camera_names),
                                image_size=int(policy.config.image_size),
                            )
                            expert_chunk = np.asarray(
                                actions[chunk_start : chunk_start + int(args.chunk_size)],
                                dtype=np.float32,
                            )
                            pred_chunk = _plan_action_chunk(policy, obs, deterministic=bool(args.deterministic))[
                                : int(args.chunk_size)
                            ]
                            if pred_chunk.shape != expert_chunk.shape:
                                raise ValueError(
                                    f"Chunk shape mismatch at {demo_name}:{chunk_start}: "
                                    f"policy={pred_chunk.shape}, expert={expert_chunk.shape}"
                                )
                            for offset in range(int(args.chunk_size)):
                                step_idx = int(chunk_start) + int(offset)
                                for dim in range(action_dim):
                                    expert_value = float(expert_chunk[offset, dim])
                                    pred_value = float(pred_chunk[offset, dim])
                                    error = pred_value - expert_value
                                    demo_series.expert[dim].append(expert_value)
                                    demo_series.pred[dim].append(pred_value)
                                    global_series.expert[dim].append(expert_value)
                                    global_series.pred[dim].append(pred_value)
                                    writer.writerow(
                                        [
                                            str(demo_name),
                                            int(chunk_start),
                                            int(step_idx),
                                            int(dim),
                                            expert_value,
                                            pred_value,
                                            float(error),
                                            float(abs(error)),
                                            float(error * error),
                                        ]
                                    )
                            demo_chunks += 1
                            total_processed_chunks += 1
                            progress.update(1)

                    if demo_chunks > 0:
                        demo_expert_by_dim, demo_pred_by_dim = _series_to_arrays(demo_series)
                        demo_mse_by_dim, demo_mae_by_dim = _metrics_from_arrays(
                            demo_expert_by_dim,
                            demo_pred_by_dim,
                        )
                        for dim in range(action_dim):
                            _save_dim_plot(
                                path=demo_dir / f"action_dim_{dim:02d}.png",
                                expert=demo_expert_by_dim[dim],
                                pred=demo_pred_by_dim[dim],
                                dim=dim,
                                max_plot_points=int(args.max_plot_points),
                            )
                        _save_all_dims_plot(
                            path=demo_dir / "actions_all_dims.png",
                            expert_by_dim=demo_expert_by_dim,
                            pred_by_dim=demo_pred_by_dim,
                            max_plot_points=int(args.max_plot_points),
                        )
                        demo_results.append(
                            _write_demo_summary(
                                demo_dir=demo_dir,
                                demo_name=str(demo_name),
                                num_chunks=demo_chunks,
                                chunk_size=int(args.chunk_size),
                                mse_by_dim=demo_mse_by_dim,
                                mae_by_dim=demo_mae_by_dim,
                            )
                        )

                    if int(args.max_frames) > 0 and total_processed_chunks >= int(args.max_frames):
                        break
    finally:
        try:
            env.close()
        finally:
            extractor.close()

    if not demo_results:
        raise RuntimeError("No demo chunks were processed; check --chunk-size / --max-frames.")
    global_expert_by_dim, global_pred_by_dim = _series_to_arrays(global_series)
    global_mse_by_dim, global_mae_by_dim = _metrics_from_arrays(
        global_expert_by_dim,
        global_pred_by_dim,
    )
    _write_global_metadata(
        output_dir=output_dir,
        args=args,
        checkpoint_path=checkpoint_path,
        data_path=data_path,
        policy=policy,
        demo_results=demo_results,
        global_mse_by_dim=global_mse_by_dim,
        global_mae_by_dim=global_mae_by_dim,
    )

    print(f"output_dir: {output_dir}")
    print(f"num_demos: {len(demo_results)}")
    print(f"num_chunks: {total_processed_chunks}")
    print(f"summary_json: {output_dir / 'summary.json'}")
    print(f"mean_rmse: {float(np.sqrt(np.mean(global_mse_by_dim))):.6f}")


if __name__ == "__main__":
    main()
