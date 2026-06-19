"""Visualize LPB v2 real-world latent features with PCA and t-SNE.

This script only encodes trajectories into latent features.
It does NOT run KNN fitting or threshold calibration.

Example:
    python -m robosuite.discriminator.dyn_disc.utils.vis_latent \
        --model-ckpt checkpoints/dyn_disc/dynamics/agilex_train-20260429_003751/checkpoints/model_49.pth \
        --fail-root data/agilex/failure_annotations/out_by_task \
        --success-root data/agilex \
        --tasks candy_in_plate duck_in_bowl Micky_in_box sausage_in_pot \
        --benchmark-json checkpoints/dyn_disc/real_world_eval/run_20260429_103014/benchmark.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib-lpb-v2-latent"
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmark.core import BenchmarkTrajectory
from benchmark.real_world import FailureBenchmark
from robosuite.discriminator.dyn_disc.adapters.single_bank import LPBV2BenchmarkDiscriminator


PHASE_SUCCESS = "success"
PHASE_FAILURE_BEFORE_GT = "failure_before_gt"
PHASE_FAILURE_AFTER_GT = "failure_after_gt"

PHASE_COLORS = {
    PHASE_SUCCESS: "#9ecae1",           # light blue
    PHASE_FAILURE_BEFORE_GT: "#2171b5", # blue
    PHASE_FAILURE_AFTER_GT: "#de2d26",  # red
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-ckpt", required=True)
    parser.add_argument("--fail-root", required=True)
    parser.add_argument("--success-root", required=True)
    parser.add_argument("--cache-root", type=str, default=None)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--benchmark-json", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)

    parser.add_argument("--max-fail-per-task", type=int, default=None)
    parser.add_argument("--max-success-per-task", type=int, default=None)
    parser.add_argument("--proprio-field", type=str, default="qpos")
    parser.add_argument("--proprio-start", type=int, default=7)
    parser.add_argument("--proprio-stop", type=int, default=14)
    parser.add_argument("--action-start", type=int, default=7)
    parser.add_argument("--action-stop", type=int, default=14)
    parser.add_argument("--proprio-indices", type=int, nargs="*", default=None)
    parser.add_argument("--camera-to-view", type=str, default=None)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument("--visual-weight", type=float, default=1.0)
    parser.add_argument("--proprio-weight", type=float, default=2.0)
    parser.add_argument("--action-weight", type=float, default=0.0)
    parser.add_argument("--delta", type=float, default=10.0)
    parser.add_argument("--knn-chunk-size", type=int, default=2048)
    parser.add_argument("--knn-feature-source", type=str, default="encoder",
                        choices=["encoder", "transformer"])
    parser.add_argument("--knn-transformer-layer", type=int, default=-1)
    parser.add_argument("--calib-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet-fit", action="store_true")

    parser.add_argument(
        "--tsne-max-points",
        type=int,
        default=20000,
        help="Maximum frames used for t-SNE. Use <=0 to run t-SNE on all frames.",
    )
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-learning-rate", type=str, default="auto")
    parser.add_argument("--tsne-init", type=str, default="pca")
    parser.add_argument("--tsne-iterations", type=int, default=1000)
    parser.add_argument("--point-size", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.65)
    return parser.parse_args()


def _parse_camera_to_view(value: str | None) -> Optional[dict[str, str]]:
    if not value:
        return None
    out: dict[str, str] = {}
    for chunk in value.split(","):
        cam, view = chunk.split(":", 1)
        out[cam.strip()] = view.strip()
    return out


def _default_out_dir(benchmark_json: str | None, out_dir: str | None) -> str:
    if out_dir:
        return str(out_dir)
    if benchmark_json:
        return str(Path(benchmark_json).expanduser().resolve().parent)
    return str(Path.cwd() / "latent_viz")


def _tasks_from_benchmark_json(path: str | None) -> Optional[list[str]]:
    if not path:
        return None
    json_path = Path(path)
    if not json_path.is_file():
        return None
    with json_path.open("r") as fp:
        data = json.load(fp)
    per_task = data.get("trajectory_level_per_task", {})
    if not per_task:
        return None
    return sorted(str(task) for task in per_task.keys())


def _first_gt_failure_frame(traj: BenchmarkTrajectory) -> Optional[int]:
    mask = traj.load_failure_mask()
    if mask is not None:
        mask_arr = np.asarray(mask).reshape(-1)
        positive = np.where(mask_arr > 0)[0]
        if positive.size > 0:
            return int(positive[0])
    first = traj.first_gt_failure_frame()
    if first is not None:
        return int(first)
    return None


def _phase_labels(traj: BenchmarkTrajectory, n_frames: int) -> list[str]:
    if not bool(traj.is_failure):
        return [PHASE_SUCCESS] * int(n_frames)

    first_gt = _first_gt_failure_frame(traj)
    if first_gt is None:
        first_gt = int(n_frames)

    labels = []
    for frame_idx in range(int(n_frames)):
        if frame_idx >= first_gt:
            labels.append(PHASE_FAILURE_AFTER_GT)
        else:
            labels.append(PHASE_FAILURE_BEFORE_GT)
    return labels


def _collect_latents(
    discriminator: LPBV2BenchmarkDiscriminator,
    trajectories: list[BenchmarkTrajectory],
) -> tuple[np.ndarray, dict[str, np.ndarray], list[dict]]:
    features: list[np.ndarray] = []
    tasks: list[str] = []
    video_ids: list[str] = []
    frame_indices: list[int] = []
    is_failure: list[bool] = []
    phases: list[str] = []
    meta: list[dict] = []

    for traj_idx, traj in enumerate(trajectories):
        feat = discriminator._encode(traj).detach().cpu().numpy().astype(np.float32, copy=False)
        n = int(min(feat.shape[0], int(traj.num_frames)))
        if n <= 0:
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
        print(
            f"[dyn_disc][latent] encoded {traj_idx + 1}/{len(trajectories)} "
            f"{traj.describe()} -> {tuple(feat.shape)}",
            flush=True,
        )

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


def _standardize_features(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (x - mean) / std


def _run_pca(features: np.ndarray) -> np.ndarray:
    from sklearn.decomposition import PCA

    if features.shape[0] < 2:
        raise ValueError("PCA requires at least 2 points.")
    return PCA(n_components=2, random_state=0).fit_transform(features).astype(np.float32)


def _sample_for_tsne(n_points: int, max_points: int, seed: int) -> np.ndarray:
    if int(max_points) <= 0 or int(max_points) >= int(n_points):
        return np.arange(int(n_points), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(int(n_points), size=int(max_points), replace=False)).astype(np.int64)


def _run_tsne(
    features: np.ndarray,
    *,
    perplexity: float,
    learning_rate: str,
    init: str,
    iterations: int,
    seed: int,
) -> np.ndarray:
    from sklearn.manifold import TSNE

    n = int(features.shape[0])
    if n < 3:
        raise ValueError("t-SNE requires at least 3 points.")
    effective_perplexity = min(float(perplexity), max(1.0, float(n - 1) / 3.0))
    lr: str | float
    try:
        lr = float(learning_rate)
    except ValueError:
        lr = str(learning_rate)
    kwargs = dict(
        n_components=2,
        perplexity=effective_perplexity,
        learning_rate=lr,
        init=init,
        random_state=int(seed),
        verbose=1,
    )
    try:
        return TSNE(max_iter=int(iterations), **kwargs).fit_transform(features).astype(np.float32)
    except TypeError:
        return TSNE(n_iter=int(iterations), **kwargs).fit_transform(features).astype(np.float32)


def _plot_embedding(
    coords: np.ndarray,
    phases: np.ndarray,
    out_path: str,
    *,
    title: str,
    point_size: float,
    alpha: float,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7), dpi=180)
    draw_order = [PHASE_SUCCESS, PHASE_FAILURE_BEFORE_GT, PHASE_FAILURE_AFTER_GT]
    for phase in draw_order:
        mask = phases == phase
        if not np.any(mask):
            continue
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=float(point_size),
            c=PHASE_COLORS[phase],
            alpha=float(alpha),
            linewidths=0,
            label=f"{phase} ({int(mask.sum())})",
        )
    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.legend(loc="best", frameon=True)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, facecolor="white", transparent=False)
    plt.close(fig)
    try:
        from PIL import Image

        with Image.open(out_path) as img:
            if img.mode != "RGB":
                img.convert("RGB").save(out_path)
    except Exception as exc:
        print(f"[dyn_disc][latent] warning: could not convert {out_path} to RGB PNG: {exc}", flush=True)


def main() -> None:
    args = _parse_args()
    out_dir = _default_out_dir(args.benchmark_json, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    tasks = list(args.tasks) if args.tasks else _tasks_from_benchmark_json(args.benchmark_json)
    bench = FailureBenchmark(
        fail_labeled_root=args.fail_root,
        success_root=args.success_root,
        tasks=tasks,
        max_fail_per_task=args.max_fail_per_task,
        max_success_per_task=args.max_success_per_task,
        cache_root=args.cache_root,
        proprio_field=args.proprio_field,
        proprio_slice=slice(int(args.proprio_start), int(args.proprio_stop)),
        action_slice=slice(int(args.action_start), int(args.action_stop)),
    )
    trajs = bench.trajectories()
    if not trajs:
        raise RuntimeError("No real-world trajectories discovered.")

    n_fail = sum(1 for traj in trajs if bool(traj.is_failure))
    n_succ = len(trajs) - n_fail
    print(
        f"[dyn_disc][latent] discovered {len(trajs)} trajectories "
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
    try:
        print("[dyn_disc][latent] encoding trajectories without KNN fitting...", flush=True)
        features, labels, traj_meta = _collect_latents(discriminator, trajs)
    finally:
        discriminator.close()

    features_std = _standardize_features(features)
    pca_coords = _run_pca(features_std)
    pca_path = os.path.join(out_dir, "latent_pca.png")
    _plot_embedding(
        pca_coords,
        labels["phase"],
        pca_path,
        title=f"LPB v2 real-world latent PCA ({features.shape[0]} frames)",
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
        title=f"LPB v2 real-world latent t-SNE ({len(tsne_indices)}/{features.shape[0]} frames)",
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
        "benchmark_json": args.benchmark_json,
        "outputs": {
            "pca_png": pca_path,
            "tsne_png": tsne_path,
            "npz": npz_path,
        },
        "trajectories": traj_meta,
    }
    with open(meta_path, "w") as fp:
        json.dump(summary, fp, indent=2)

    print(f"[dyn_disc][latent] wrote {pca_path}", flush=True)
    print(f"[dyn_disc][latent] wrote {tsne_path}", flush=True)
    print(f"[dyn_disc][latent] wrote {npz_path}", flush=True)
    print(f"[dyn_disc][latent] wrote {meta_path}", flush=True)


if __name__ == "__main__":
    main()
