"""Visualize DINOv3 dynamic robosuite latent features with PCA and t-SNE."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib-dinov3-robosuite-latent"
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

from robosuite.discriminator.dyn_disc.adapters.single_bank import LPBV2BenchmarkDiscriminator
from robosuite.discriminator.dyn_disc.visualization.vis_latent import (
    PHASE_FAILURE_AFTER_GT,
    PHASE_FAILURE_BEFORE_GT,
    PHASE_SUCCESS,
    _first_gt_failure_frame,
    _parse_camera_to_view,
    _phase_labels,
    _plot_embedding,
    _run_pca,
    _run_tsne,
    _sample_for_tsne,
    _standardize_features,
)
from robosuite.discriminator.utils.robosuite_benchmark import FailureBenchmark


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
    parser.add_argument(
        "--knn-feature-source",
        type=str,
        default="transformer",
        choices=["encoder", "transformer"],
    )
    parser.add_argument("--knn-transformer-layer", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--tsne-max-points", type=int, default=20000)
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-learning-rate", type=str, default="auto")
    parser.add_argument("--tsne-init", type=str, default="pca")
    parser.add_argument("--tsne-iterations", type=int, default=1000)
    parser.add_argument("--point-size", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.65)
    return parser.parse_args()


def _collect_latents_preloaded(
    discriminator: LPBV2BenchmarkDiscriminator,
    trajectories: list,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[dict]]:
    preloaded: list[dict] = []
    bar = tqdm(total=len(trajectories) * 2, desc="preload", unit="traj")
    for traj in trajectories:
        preloaded.append(discriminator.preload_trajectory(traj))
        bar.update(1)

    bar.set_description("encode")
    features: list[np.ndarray] = []
    tasks: list[str] = []
    video_ids: list[str] = []
    frame_indices: list[int] = []
    is_failure: list[bool] = []
    phases: list[str] = []
    meta: list[dict] = []

    for traj_idx, (traj, prepared) in enumerate(zip(trajectories, preloaded)):
        feat = (
            discriminator.encode_preloaded(traj, prepared)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
        n = int(min(feat.shape[0], int(traj.num_frames), int(prepared["t_len"])))
        if n <= 0:
            bar.update(1)
            continue
        feat = feat[:n]
        phase = _phase_labels(traj, n)
        first_gt = _first_gt_failure_frame(traj) if bool(traj.is_failure) else None

        features.append(feat)
        tasks.extend([str(traj.task_name)] * n)
        video_ids.extend([str(traj.video_id)] * n)
        frame_indices.extend(range(n))
        is_failure.extend([bool(traj.is_failure)] * n)
        phases.extend(phase)
        meta.append(
            {
                "trajectory_index": int(traj_idx),
                "task_name": str(traj.task_name),
                "video_id": str(traj.video_id),
                "is_failure": bool(traj.is_failure),
                "num_encoded_frames": int(n),
                "num_frames": int(traj.num_frames),
                "first_gt_failure_frame": first_gt,
                "failure_segments": list(traj.failure_segments),
            }
        )
        bar.update(1)

    bar.close()
    if not features:
        raise RuntimeError("No latent features were encoded.")

    x = np.concatenate(features, axis=0)
    labels = {
        "task_name": np.asarray(tasks),
        "video_id": np.asarray(video_ids),
        "frame_idx": np.asarray(frame_indices, dtype=np.int64),
        "is_failure": np.asarray(is_failure, dtype=np.bool_),
        "phase": np.asarray(phases),
    }
    return x, labels, meta


def main() -> None:
    args = _parse_args()
    out_dir = str(Path(args.out_dir).expanduser().resolve())
    os.makedirs(out_dir, exist_ok=True)

    bench = FailureBenchmark(
        data_root=str(args.data_root),
        tasks=[args.task] if args.task else None,
        fail_split=str(args.fail_split),
        success_split=str(args.success_split),
        max_fail_per_task=args.max_fail_per_task,
        max_success_per_task=args.max_success_per_task,
    )
    trajs = bench.trajectories()
    if not trajs:
        raise RuntimeError("No robosuite trajectories discovered.")

    n_fail = sum(1 for traj in trajs if bool(traj.is_failure))
    n_succ = len(trajs) - n_fail
    print(
        f"[dyn_disc][latent][dinov3][robosuite] discovered {len(trajs)} trajectories "
        f"(failure={n_fail}, success={n_succ}) tasks={sorted({str(t.task_name) for t in trajs})}",
        flush=True,
    )

    discriminator = LPBV2BenchmarkDiscriminator(
        model_ckpt=str(args.model_ckpt),
        device=str(args.device),
        encode_batch_size=int(args.encode_batch_size),
        proprio_indices=(list(args.proprio_indices) if args.proprio_indices else None),
        camera_to_view=_parse_camera_to_view(args.camera_to_view),
        feature_source=str(args.knn_feature_source),
        transformer_layer=int(args.knn_transformer_layer),
    )
    discriminator.batch_infer_times = []
    try:
        features, labels, traj_meta = _collect_latents_preloaded(discriminator, trajs)
    finally:
        discriminator.close()

    batch_infer_times = list(discriminator.batch_infer_times or [])
    discriminator.batch_infer_times = None
    single_infer_mean_s = None
    warmup_runs = 3
    if len(batch_infer_times) > warmup_runs:
        single_infer_mean_s = sum(batch_infer_times[warmup_runs:]) / (
            len(batch_infer_times) - warmup_runs
        )

    features_std = _standardize_features(features)
    pca_coords = _run_pca(features_std)
    pca_path = os.path.join(out_dir, "latent_pca.png")
    _plot_embedding(
        pca_coords,
        labels["phase"],
        pca_path,
        title=f"DINOv3 robosuite latent PCA ({features.shape[0]} frames)",
        point_size=float(args.point_size),
        alpha=float(args.alpha),
    )

    tsne_indices = _sample_for_tsne(features_std.shape[0], int(args.tsne_max_points), int(args.seed))
    tsne_coords = _run_tsne(
        features_std[tsne_indices],
        perplexity=float(args.tsne_perplexity),
        learning_rate=str(args.tsne_learning_rate),
        init=str(args.tsne_init),
        iterations=int(args.tsne_iterations),
        seed=int(args.seed),
    )
    tsne_path = os.path.join(out_dir, "latent_tsne.png")
    _plot_embedding(
        tsne_coords,
        labels["phase"][tsne_indices],
        tsne_path,
        title=f"DINOv3 robosuite latent t-SNE ({len(tsne_indices)}/{features.shape[0]} frames)",
        point_size=float(args.point_size),
        alpha=float(args.alpha),
    )

    npz_path = os.path.join(out_dir, "latent_points.npz")
    np.savez_compressed(
        npz_path,
        pca=pca_coords,
        tsne=tsne_coords,
        tsne_indices=tsne_indices,
        task_name=labels["task_name"],
        video_id=labels["video_id"],
        frame_idx=labels["frame_idx"],
        is_failure=labels["is_failure"],
        phase=labels["phase"],
    )

    meta_path = os.path.join(out_dir, "latent_points_meta.json")
    summary = {
        "num_trajectories": int(len(trajs)),
        "num_failure_trajectories": int(n_fail),
        "num_success_trajectories": int(n_succ),
        "num_frames": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "tasks": sorted({str(t.task_name) for t in trajs}),
        "phase_counts": {
            phase: int(np.sum(labels["phase"] == phase))
            for phase in [PHASE_SUCCESS, PHASE_FAILURE_BEFORE_GT, PHASE_FAILURE_AFTER_GT]
        },
        "tsne_num_points": int(len(tsne_indices)),
        "tsne_max_points": int(args.tsne_max_points),
        "model_ckpt": str(args.model_ckpt),
        "knn_feature_source": str(args.knn_feature_source),
        "knn_transformer_layer": int(args.knn_transformer_layer),
        "single_infer_mean_s": single_infer_mean_s,
        "num_batch_infers": len(batch_infer_times),
        "encode_batch_size": int(args.encode_batch_size),
        "outputs": {
            "pca_png": pca_path,
            "tsne_png": tsne_path,
            "npz": npz_path,
        },
        "trajectories": traj_meta,
    }
    with open(meta_path, "w") as fp:
        json.dump(summary, fp, indent=2)

    print(f"[dyn_disc][latent][dinov3][robosuite] wrote {pca_path}", flush=True)
    print(f"[dyn_disc][latent][dinov3][robosuite] wrote {tsne_path}", flush=True)
    print(f"[dyn_disc][latent][dinov3][robosuite] wrote {npz_path}", flush=True)
    print(f"[dyn_disc][latent][dinov3][robosuite] wrote {meta_path}", flush=True)
    if single_infer_mean_s is not None:
        print(
            f"[dyn_disc][latent][dinov3][robosuite] single model infer mean: "
            f"{single_infer_mean_s:.4f}s per encode_batch() "
            f"(encode_batch_size={args.encode_batch_size}, skip first {warmup_runs})",
            flush=True,
        )


if __name__ == "__main__":
    main()
