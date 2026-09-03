"""Visualize frozen policy features with PCA and t-SNE.

CUDA is used only for policy inference. Dimensionality reduction is an offline
analysis step and never affects discriminator training or calibration.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib-policy-robosuite-latent"
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmark.core import BenchmarkTrajectory

from robosuite.discriminator.dyn_disc.adapters.single_bank import (
    PolicyBenchmarkDiscriminator,
)
from robosuite.discriminator.utils.robosuite_benchmark import FailureBenchmark


PHASE_SUCCESS = "success"
PHASE_FAILURE_BEFORE_GT = "failure_before_gt"
PHASE_FAILURE_AFTER_GT = "failure_after_gt"
PHASES = (PHASE_SUCCESS, PHASE_FAILURE_BEFORE_GT, PHASE_FAILURE_AFTER_GT)
PHASE_COLORS = {
    PHASE_SUCCESS: "#9ecae1",
    PHASE_FAILURE_BEFORE_GT: "#2171b5",
    PHASE_FAILURE_AFTER_GT: "#de2d26",
}
DEFAULT_POLICY_CKPT = "checkpoints/multitask_6/flow_multi_ep0100.pt"
DEFAULT_OUTPUT_ROOT = Path("checkpoints/dyn_disc/ablations/policy")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize policy task_scene_cond features with PCA and t-SNE."
    )
    parser.add_argument("--policy-ckpt", default=DEFAULT_POLICY_CKPT)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--fail-split", default="fail_rollout-val-labeled")
    parser.add_argument("--success-split", default="success_rollout-val")
    parser.add_argument("--task", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--max-fail-per-task", type=int, default=25)
    parser.add_argument("--max-success-per-task", type=int, default=25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encode-batch-size", type=int, default=128)
    parser.add_argument("--preload-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--feature-cache-dir",
        default=str(DEFAULT_OUTPUT_ROOT / "feature_cache"),
    )
    parser.add_argument(
        "--reuse-feature-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tsne-max-points", type=int, default=20000)
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-learning-rate", default="auto")
    parser.add_argument("--tsne-init", default="pca")
    parser.add_argument("--tsne-iterations", type=int, default=1000)
    parser.add_argument("--point-size", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.65)
    return parser.parse_args()


def _default_out_dir(task: Optional[str]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_name = str(task) if task else "all"
    safe_task = "".join(c if c.isalnum() or c in "._-" else "_" for c in task_name)
    return (
        DEFAULT_OUTPUT_ROOT
        / "runs"
        / f"run_{timestamp}_{safe_task}"
        / "visualizations"
        / "latent"
    )


def _first_gt_failure_frame(traj: BenchmarkTrajectory) -> Optional[int]:
    first = traj.first_gt_failure_frame()
    if first is not None:
        return int(first)
    mask = traj.load_failure_mask()
    if mask is not None:
        positive = np.flatnonzero(np.asarray(mask).reshape(-1) > 0)
        if positive.size:
            return int(positive[0])
    return None


def _phase_labels(
    traj: BenchmarkTrajectory,
    n_frames: int,
    first_gt: Optional[int] = None,
) -> list[str]:
    if not bool(traj.is_failure):
        return [PHASE_SUCCESS] * n_frames
    gt_start = n_frames if first_gt is None else int(first_gt)
    return [
        PHASE_FAILURE_AFTER_GT if frame_idx >= gt_start else PHASE_FAILURE_BEFORE_GT
        for frame_idx in range(n_frames)
    ]


def _collect_latents(
    discriminator: PolicyBenchmarkDiscriminator,
    trajectories: list[BenchmarkTrajectory],
) -> tuple[np.ndarray, dict[str, np.ndarray], list[dict[str, Any]]]:
    encoded = discriminator.preencode_trajectories(trajectories, desc="load+encode")
    features: list[np.ndarray] = []
    tasks: list[str] = []
    video_ids: list[str] = []
    frame_indices: list[int] = []
    is_failure: list[bool] = []
    phases: list[str] = []
    metadata: list[dict[str, Any]] = []

    for traj_idx, (traj, feat_tensor) in enumerate(zip(trajectories, encoded)):
        feat = feat_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        n_frames = min(int(feat.shape[0]), int(traj.num_frames))
        if n_frames <= 0:
            continue
        feat = feat[:n_frames]
        first_gt = _first_gt_failure_frame(traj) if bool(traj.is_failure) else None
        phase = _phase_labels(traj, n_frames, first_gt=first_gt)
        features.append(feat)
        tasks.extend([str(traj.task_name)] * n_frames)
        video_ids.extend([str(traj.video_id)] * n_frames)
        frame_indices.extend(range(n_frames))
        is_failure.extend([bool(traj.is_failure)] * n_frames)
        phases.extend(phase)
        metadata.append(
            {
                "trajectory_index": traj_idx,
                "task_name": str(traj.task_name),
                "video_id": str(traj.video_id),
                "is_failure": bool(traj.is_failure),
                "num_encoded_frames": n_frames,
                "num_frames": int(traj.num_frames),
                "first_gt_failure_frame": first_gt,
                "failure_segments": list(traj.failure_segments),
            }
        )

    if not features:
        raise RuntimeError("No policy latent features were encoded.")
    labels = {
        "task_name": np.asarray(tasks),
        "video_id": np.asarray(video_ids),
        "frame_idx": np.asarray(frame_indices, dtype=np.int64),
        "is_failure": np.asarray(is_failure, dtype=np.bool_),
        "phase": np.asarray(phases),
    }
    return np.concatenate(features, axis=0), labels, metadata


def _standardize(features: np.ndarray) -> np.ndarray:
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    return (features - mean) / np.where(std < 1e-6, 1.0, std)


def _run_tsne(features: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    from sklearn.manifold import TSNE

    if features.shape[0] < 3:
        raise ValueError("t-SNE requires at least 3 points.")
    perplexity = min(
        float(args.tsne_perplexity),
        max(1.0, (features.shape[0] - 1) / 3.0),
    )
    try:
        learning_rate: str | float = float(args.tsne_learning_rate)
    except ValueError:
        learning_rate = str(args.tsne_learning_rate)
    kwargs = {
        "n_components": 2,
        "perplexity": perplexity,
        "learning_rate": learning_rate,
        "init": str(args.tsne_init),
        "random_state": int(args.seed),
        "verbose": 1,
    }
    try:
        result = TSNE(max_iter=int(args.tsne_iterations), **kwargs).fit_transform(features)
    except TypeError:
        result = TSNE(n_iter=int(args.tsne_iterations), **kwargs).fit_transform(features)
    return result.astype(np.float32)


def _plot(
    coords: np.ndarray,
    phases: np.ndarray,
    path: Path,
    title: str,
    args: argparse.Namespace,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7), dpi=180)
    for phase in PHASES:
        mask = phases == phase
        if np.any(mask):
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=float(args.point_size),
                c=PHASE_COLORS[phase],
                alpha=float(args.alpha),
                linewidths=0,
                label=f"{phase} ({int(mask.sum())})",
            )
    ax.set(title=title, xlabel="dim 1", ylabel="dim 2")
    ax.legend(loc="best", frameon=True)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, facecolor="white", transparent=False)
    plt.close(fig)


def _encoder_metadata(discriminator: PolicyBenchmarkDiscriminator) -> dict[str, Any]:
    encoder = getattr(discriminator, "encoder", None)
    if encoder is None:
        encoder = getattr(discriminator, "_encoder", None)
    raw = encoder.metadata() if encoder is not None else {}
    return {
        "policy_checkpoint_sha256": raw.get(
            "policy_ckpt_hash",
            getattr(discriminator, "policy_ckpt_hash", None),
        ),
        "checkpoint_state": raw.get("policy_weight_source", "ema_model"),
        "feature_source": raw.get("feature_source", "policy_task_scene_cond"),
        "feature_dtype": raw.get("dtype", "float32"),
        "camera_names": list(raw.get("camera_names", ())),
        "prompt_map": dict(raw.get("task_prompts", {})),
        "preprocess_version": raw.get("preprocess_version"),
    }


def main() -> None:
    args = _parse_args()
    if not str(args.device).startswith("cuda"):
        raise ValueError("Policy latent inference is CUDA-only; use --device cuda.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for policy latent inference but is unavailable.")

    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else _default_out_dir(args.task).resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    benchmark = FailureBenchmark(
        data_root=str(args.data_root),
        tasks=[str(args.task)] if args.task else None,
        fail_split=str(args.fail_split),
        success_split=str(args.success_split),
        max_fail_per_task=args.max_fail_per_task,
        max_success_per_task=args.max_success_per_task,
    )
    trajectories = benchmark.trajectories()
    if not trajectories:
        raise RuntimeError("No robosuite trajectories discovered.")

    n_fail = sum(bool(traj.is_failure) for traj in trajectories)
    print(
        f"[dyn_disc][latent][policy] discovered {len(trajectories)} trajectories "
        f"(failure={n_fail}, success={len(trajectories) - n_fail})",
        flush=True,
    )
    discriminator = PolicyBenchmarkDiscriminator(
        policy_ckpt=str(args.policy_ckpt),
        device=str(args.device),
        encode_batch_size=int(args.encode_batch_size),
        preload_workers=int(args.preload_workers),
        prefetch_factor=int(args.prefetch_factor),
        pin_memory=bool(args.pin_memory),
        feature_cache_dir=str(args.feature_cache_dir),
        reuse_feature_cache=bool(args.reuse_feature_cache),
        seed=int(args.seed),
        verbose_fit=True,
    )
    try:
        features, labels, trajectory_metadata = _collect_latents(
            discriminator,
            trajectories,
        )
        encoder_metadata = _encoder_metadata(discriminator)
    finally:
        discriminator.close()

    standardized = _standardize(features)
    from sklearn.decomposition import PCA

    if standardized.shape[0] < 2:
        raise ValueError("PCA requires at least 2 points.")
    pca = PCA(n_components=2, random_state=int(args.seed)).fit_transform(standardized)
    pca = pca.astype(np.float32)
    pca_path = out_dir / "policy_latent_pca.png"
    _plot(
        pca,
        labels["phase"],
        pca_path,
        f"Policy task_scene_cond PCA ({features.shape[0]} frames)",
        args,
    )

    n_points = standardized.shape[0]
    if 0 < int(args.tsne_max_points) < n_points:
        rng = np.random.default_rng(int(args.seed))
        tsne_indices = np.sort(
            rng.choice(n_points, int(args.tsne_max_points), replace=False)
        )
    else:
        tsne_indices = np.arange(n_points, dtype=np.int64)
    tsne = _run_tsne(standardized[tsne_indices], args)
    tsne_path = out_dir / "policy_latent_tsne.png"
    _plot(
        tsne,
        labels["phase"][tsne_indices],
        tsne_path,
        f"Policy task_scene_cond t-SNE ({len(tsne_indices)}/{n_points} frames)",
        args,
    )

    npz_path = out_dir / "policy_latent_points.npz"
    np.savez_compressed(
        npz_path,
        latent=features,
        pca=pca,
        tsne=tsne,
        tsne_indices=tsne_indices,
        **labels,
    )
    meta_path = out_dir / "policy_latent_points_meta.json"
    summary = {
        "num_trajectories": len(trajectories),
        "num_failure_trajectories": int(n_fail),
        "num_success_trajectories": len(trajectories) - int(n_fail),
        "num_frames": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "tasks": sorted({str(traj.task_name) for traj in trajectories}),
        "phase_counts": {
            phase: int(np.sum(labels["phase"] == phase)) for phase in PHASES
        },
        "tsne_num_points": int(len(tsne_indices)),
        "policy_ckpt": str(Path(args.policy_ckpt).expanduser().resolve()),
        "encode_batch_size": int(args.encode_batch_size),
        "preload_workers": int(args.preload_workers),
        "prefetch_factor": int(args.prefetch_factor),
        "pin_memory": bool(args.pin_memory),
        "feature_cache_dir": str(Path(args.feature_cache_dir).expanduser().resolve()),
        "reuse_feature_cache": bool(args.reuse_feature_cache),
        "seed": int(args.seed),
        **encoder_metadata,
        "outputs": {
            "pca_png": str(pca_path),
            "tsne_png": str(tsne_path),
            "latent_npz": str(npz_path),
            "metadata_json": str(meta_path),
        },
        "trajectories": trajectory_metadata,
    }
    with meta_path.open("w") as file:
        json.dump(summary, file, indent=2)

    for path in (pca_path, tsne_path, npz_path, meta_path):
        print(f"[dyn_disc][latent][policy] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
