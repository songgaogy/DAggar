"""Visualize TACO encoder features on robosuite trajectories with PCA and t-SNE.

This entry point only performs CUDA encoder inference and offline dimensionality
reduction. It does not fit or evaluate the nnPU discriminator.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib-taco-robosuite-latent"
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmark.core import BenchmarkTrajectory
from robosuite.discriminator.dyn_disc.adapters.single_bank import DynBenchmarkDiscriminator
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-ckpt", required=True)
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--fail-split", type=str, default="fail_rollout-val-labeled")
    parser.add_argument("--success-split", type=str, default="success_rollout-val")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-fail-per-task", type=int, default=25)
    parser.add_argument("--max-success-per-task", type=int, default=25)
    parser.add_argument("--proprio-indices", type=int, nargs="*", default=None)
    parser.add_argument("--camera-to-view", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument("--preload-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tsne-max-points", type=int, default=20000)
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-learning-rate", type=str, default="auto")
    parser.add_argument("--tsne-init", type=str, default="pca")
    parser.add_argument("--tsne-iterations", type=int, default=1000)
    parser.add_argument("--point-size", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.65)
    return parser.parse_args()


def _parse_camera_to_view(value: Optional[str]) -> Optional[dict[str, str]]:
    if not value:
        return None
    mapping: dict[str, str] = {}
    for chunk in value.split(","):
        camera, view = chunk.split(":", 1)
        mapping[camera.strip()] = view.strip()
    return mapping


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
    if first_gt is None:
        first_gt = n_frames
    return [
        PHASE_FAILURE_AFTER_GT if frame_idx >= first_gt else PHASE_FAILURE_BEFORE_GT
        for frame_idx in range(n_frames)
    ]


def _collect_latents(
    discriminator: DynBenchmarkDiscriminator,
    trajectories: list[BenchmarkTrajectory],
) -> tuple[np.ndarray, dict[str, np.ndarray], list[dict]]:
    encoded = discriminator.preencode_trajectories(
        trajectories,
        desc="load+encode",
    )
    features: list[np.ndarray] = []
    tasks: list[str] = []
    video_ids: list[str] = []
    frame_indices: list[int] = []
    is_failure: list[bool] = []
    phases: list[str] = []
    metadata: list[dict] = []

    for traj_idx, (traj, feat_tensor) in enumerate(zip(trajectories, encoded)):
        feat = (
            feat_tensor
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
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
        raise RuntimeError("No latent features were encoded.")
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
    perplexity = min(float(args.tsne_perplexity), max(1.0, (features.shape[0] - 1) / 3.0))
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


def _plot(coords: np.ndarray, phases: np.ndarray, path: Path, title: str, args: argparse.Namespace) -> None:
    fig, ax = plt.subplots(figsize=(9, 7), dpi=180)
    for phase in PHASES:
        mask = phases == phase
        if np.any(mask):
            ax.scatter(
                coords[mask, 0], coords[mask, 1], s=float(args.point_size),
                c=PHASE_COLORS[phase], alpha=float(args.alpha), linewidths=0,
                label=f"{phase} ({int(mask.sum())})",
            )
    ax.set(title=title, xlabel="dim 1", ylabel="dim 2")
    ax.legend(loc="best", frameon=True)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, facecolor="white", transparent=False)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    if not str(args.device).startswith("cuda"):
        raise ValueError("TACO latent inference is CUDA-only; use --device cuda.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TACO latent inference but is unavailable.")

    out_dir = Path(args.out_dir).expanduser().resolve()
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
        f"[dyn_disc][latent][taco] discovered {len(trajectories)} trajectories "
        f"(failure={n_fail}, success={len(trajectories) - n_fail})",
        flush=True,
    )
    discriminator = DynBenchmarkDiscriminator(
        model_ckpt=str(args.model_ckpt),
        device=str(args.device),
        encode_batch_size=int(args.encode_batch_size),
        preload_workers=int(args.preload_workers),
        prefetch_factor=int(args.prefetch_factor),
        pin_memory=bool(args.pin_memory),
        proprio_indices=list(args.proprio_indices) if args.proprio_indices else None,
        camera_to_view=_parse_camera_to_view(args.camera_to_view),
    )
    discriminator.batch_infer_times = []
    try:
        features, labels, trajectory_metadata = _collect_latents(discriminator, trajectories)
        infer_times = list(discriminator.batch_infer_times or [])
    finally:
        discriminator.close()

    standardized = _standardize(features)
    from sklearn.decomposition import PCA

    if standardized.shape[0] < 2:
        raise ValueError("PCA requires at least 2 points.")
    pca = PCA(n_components=2, random_state=int(args.seed)).fit_transform(standardized).astype(np.float32)
    pca_path = out_dir / "taco_latent_pca.png"
    _plot(pca, labels["phase"], pca_path, f"TACO encoder PCA ({features.shape[0]} frames)", args)

    n_points = standardized.shape[0]
    if 0 < int(args.tsne_max_points) < n_points:
        rng = np.random.default_rng(int(args.seed))
        tsne_indices = np.sort(rng.choice(n_points, int(args.tsne_max_points), replace=False))
    else:
        tsne_indices = np.arange(n_points, dtype=np.int64)
    tsne = _run_tsne(standardized[tsne_indices], args)
    tsne_path = out_dir / "taco_latent_tsne.png"
    _plot(
        tsne, labels["phase"][tsne_indices], tsne_path,
        f"TACO encoder t-SNE ({len(tsne_indices)}/{n_points} frames)", args,
    )

    npz_path = out_dir / "taco_latent_points.npz"
    np.savez_compressed(
        npz_path,
        latent=features,
        pca=pca,
        tsne=tsne,
        tsne_indices=tsne_indices,
        **labels,
    )
    warmup_runs = 3
    measured_times = infer_times[warmup_runs:]
    meta_path = out_dir / "taco_latent_points_meta.json"
    summary = {
        "num_trajectories": len(trajectories),
        "num_failure_trajectories": int(n_fail),
        "num_success_trajectories": len(trajectories) - int(n_fail),
        "num_frames": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "tasks": sorted({str(traj.task_name) for traj in trajectories}),
        "phase_counts": {phase: int(np.sum(labels["phase"] == phase)) for phase in PHASES},
        "tsne_num_points": int(len(tsne_indices)),
        "model_ckpt": str(args.model_ckpt),
        "feature_source": "encoder",
        "encode_batch_size": int(args.encode_batch_size),
        "preload_workers": int(args.preload_workers),
        "prefetch_factor": int(args.prefetch_factor),
        "pin_memory": bool(args.pin_memory),
        "single_infer_mean_s": float(np.mean(measured_times)) if measured_times else None,
        "num_batch_infers": len(infer_times),
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
        print(f"[dyn_disc][latent][taco] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
