"""Visualize frame-level RPT action-token features with PCA and t-SNE."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmark.core import BenchmarkTrajectory
from robosuite.discriminator.dyn_disc.adapters.single_bank import DynBenchmarkDiscriminator
from robosuite.discriminator.utils.robosuite_benchmark import FailureBenchmark


LABEL_NAMES = ("success", "failure-before-GT", "failure-after-GT")
LABEL_COLORS = ("#9ecae1", "#2171b5", "#de2d26")  # light blue, blue, red


def _parse_camera_to_view(value: str | None) -> dict[str, str] | None:
    if not value:
        return None
    mapping: dict[str, str] = {}
    for chunk in value.split(","):
        camera, view = chunk.split(":", 1)
        mapping[camera.strip()] = view.strip()
    return mapping


def _sample_trajectories(
    trajectories: Sequence[BenchmarkTrajectory],
    *,
    num_per_split: int,
    seed: int,
) -> list[BenchmarkTrajectory]:
    rng = random.Random(seed)
    selected: list[BenchmarkTrajectory] = []
    for is_failure in (False, True):
        pool = [trajectory for trajectory in trajectories if bool(trajectory.is_failure) == is_failure]
        if not pool:
            kind = "failure" if is_failure else "success"
            raise RuntimeError(f"No {kind} trajectories are available for latent visualization.")
        selected.extend(rng.sample(pool, min(num_per_split, len(pool))))
    return selected


def _frame_labels(trajectory: BenchmarkTrajectory, length: int) -> np.ndarray:
    if not trajectory.is_failure:
        return np.zeros(length, dtype=np.int64)
    mask = trajectory.load_failure_mask()
    if mask is None:
        raise RuntimeError(f"Failure trajectory {trajectory.video_id!r} has no GT failure mask.")
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    failure_indices = np.flatnonzero(mask[:length])
    if failure_indices.size == 0:
        raise RuntimeError(
            f"Failure trajectory {trajectory.video_id!r} has no positive GT failure frame."
        )
    labels = np.ones(length, dtype=np.int64)
    labels[int(failure_indices[0]) :] = 2
    return labels


def _plot_embedding(points: np.ndarray, labels: np.ndarray, path: Path, title: str) -> None:
    fig, axis = plt.subplots(figsize=(8, 7))
    for label_id, (name, color) in enumerate(zip(LABEL_NAMES, LABEL_COLORS)):
        selected = labels == label_id
        if selected.any():
            axis.scatter(
                points[selected, 0],
                points[selected, 1],
                s=7,
                alpha=0.65,
                c=color,
                label=f"{name} (n={int(selected.sum())})",
                linewidths=0,
            )
    axis.set_title(title)
    axis.set_xlabel("component 1")
    axis.set_ylabel("component 2")
    axis.grid(alpha=0.2)
    axis.legend(markerscale=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-ckpt", required=True, help="RPT representation checkpoint .pth")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--task", required=True)
    parser.add_argument("--fail-split", default="fail_rollout-val-labeled")
    parser.add_argument("--success-split", default="success_rollout-val")
    parser.add_argument("--num-trajs", type=int, default=8, help="Trajectories per split.")
    parser.add_argument("--max-points", type=int, default=10000)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encode-batch-size", type=int, default=128)
    parser.add_argument("--proprio-indices", type=int, nargs="*", default=None)
    parser.add_argument("--camera-to-view", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if str(args.device) != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("RPT latent extraction requires CUDA; CPU fallback is disabled.")
    if int(args.num_trajs) < 1 or int(args.max_points) < 3:
        raise ValueError("--num-trajs must be positive and --max-points must be at least 3.")

    benchmark = FailureBenchmark(
        data_root=args.data_root,
        tasks=[str(args.task)],
        fail_split=args.fail_split,
        success_split=args.success_split,
    )
    sampled = _sample_trajectories(
        benchmark.trajectories(),
        num_per_split=int(args.num_trajs),
        seed=int(args.seed),
    )

    encoder = DynBenchmarkDiscriminator(
        model_ckpt=str(args.model_ckpt),
        device="cuda",
        encode_batch_size=int(args.encode_batch_size),
        proprio_indices=(list(args.proprio_indices) if args.proprio_indices else None),
        camera_to_view=_parse_camera_to_view(args.camera_to_view),
    )
    feature_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    video_ids: list[str] = []
    frame_indices: list[int] = []
    try:
        for trajectory in sampled:
            prepared = encoder.preload_trajectory(trajectory)
            features = (
                encoder.encode_preloaded(trajectory, prepared)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            length = min(features.shape[0], int(trajectory.num_frames))
            feature_chunks.append(features[:length])
            label_chunks.append(_frame_labels(trajectory, length))
            video_ids.extend([str(trajectory.video_id)] * length)
            frame_indices.extend(range(length))
    finally:
        encoder.close()

    features = np.concatenate(feature_chunks, axis=0)
    labels = np.concatenate(label_chunks, axis=0)
    video_ids_array = np.asarray(video_ids)
    frame_indices_array = np.asarray(frame_indices, dtype=np.int64)
    if features.shape[0] < 3:
        raise RuntimeError("At least three frame features are required for PCA/t-SNE.")

    rng = np.random.default_rng(int(args.seed))
    if features.shape[0] > int(args.max_points):
        indices = np.sort(rng.choice(features.shape[0], int(args.max_points), replace=False))
        features = features[indices]
        labels = labels[indices]
        video_ids_array = video_ids_array[indices]
        frame_indices_array = frame_indices_array[indices]

    pca = PCA(n_components=2, random_state=int(args.seed)).fit_transform(features)
    perplexity = min(30.0, max(1.0, float(features.shape[0] - 1) / 3.0))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=int(args.seed),
    ).fit_transform(features)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "rpt_latents.npz",
        features=features,
        labels=labels,
        label_names=np.asarray(LABEL_NAMES),
        pca=pca,
        tsne=tsne,
        video_ids=video_ids_array,
        frame_indices=frame_indices_array,
    )
    metadata = {
        "pretraining_method": "rpt",
        "feature_source": "rpt_action_token",
        "model_ckpt": os.path.abspath(args.model_ckpt),
        "task": str(args.task),
        "seed": int(args.seed),
        "num_trajectories": len(sampled),
        "num_points": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "tsne_perplexity": float(perplexity),
        "labels": {str(index): name for index, name in enumerate(LABEL_NAMES)},
        "gt_usage": "plot labels only; never used by RPT or nnPU training",
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    _plot_embedding(pca, labels, out_dir / "pca.png", f"RPT latent PCA — {args.task}")
    _plot_embedding(tsne, labels, out_dir / "tsne.png", f"RPT latent t-SNE — {args.task}")
    print(f"[rpt][latent-viz] wrote PCA, t-SNE, NPZ, and metadata to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
