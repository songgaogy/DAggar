"""Phase-1 latent-separability diagnostic for LPB v2.

Implements PLAN_score_based_discriminator.md §3.1, §3.2 and §3.3 with
pooled-across-tasks tests (single MMD p-value, single Mahalanobis AUROC,
single matched-timestep AUROC). Per-task numbers are reported as a
sanity-check side panel only.

Pipeline:
  1. Encode every frame of `FailureBenchmark.trajectories()` with the LPB v2
     transformer hidden state at the chosen layer.
  2. Persist raw features to ``latents.npz`` (and metadata to
     ``latents_meta.json``) so analysis can be re-run without GPU encoding.
  3. Run pooled MMD permutation test, pooled Ledoit-Wolf Mahalanobis AUROC
     with rollout-level success split, and pooled matched-timestep AUROC.
  4. Emit ``roc.png``, ``hist.png``, ``roc_matched.png`` plus
     ``separability_summary.json``.

Pass ``--load-cache PATH/latents.npz`` to skip encoding entirely.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score, roc_curve

from benchmark.core import BenchmarkTrajectory
from benchmark.real_world import FailureBenchmark
from robosuite.discriminator.lpb_v2.adapters.single_bank import LPBV2BenchmarkDiscriminator
from robosuite.discriminator.lpb_v2.utils.vis_latent import (
    PHASE_FAILURE_AFTER_GT,
    PHASE_FAILURE_BEFORE_GT,
    PHASE_SUCCESS,
)


@dataclass
class EncodedTrajectory:
    trajectory_index: int
    task_name: str
    video_id: str
    is_failure: bool
    features: np.ndarray
    first_gt_failure_frame: Optional[int]
    failure_segments: list[dict]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-ckpt", required=False, default=None)
    parser.add_argument("--fail-root", required=False, default=None)
    parser.add_argument("--success-root", required=False, default=None)
    parser.add_argument("--cache-root", type=str, default=None)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--out-dir", required=True)

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
    parser.add_argument(
        "--feature-source",
        type=str,
        default="transformer",
        choices=["encoder", "transformer"],
    )
    parser.add_argument("--transformer-layer", type=int, default=1)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument(
        "--max-frames-per-class",
        type=int,
        default=4000,
        help="Subsample cap for the O(N^2) MMD kernel (pooled across tasks).",
    )
    parser.add_argument(
        "--median-max-points",
        type=int,
        default=2000,
        help="Maximum pooled points used to estimate the median-heuristic bandwidth.",
    )
    parser.add_argument(
        "--mahalanobis-train-frac",
        type=float,
        default=0.8,
        help="Fraction of success rollouts used to fit Ledoit-Wolf for Mahalanobis AUROC.",
    )
    parser.add_argument(
        "--matched-window",
        type=int,
        default=5,
        help="Half-width (in frames) for §3.3 matched-timestep control.",
    )
    parser.add_argument(
        "--load-cache",
        type=str,
        default=None,
        help="If set, load latents from this npz instead of re-encoding.",
    )
    parser.add_argument(
        "--cache-features-only",
        action="store_true",
        help="Encode, write latents.npz, and exit (skip analysis).",
    )
    return parser.parse_args()


def _parse_camera_to_view(value: str | None) -> Optional[dict[str, str]]:
    if not value:
        return None
    out: dict[str, str] = {}
    for chunk in value.split(","):
        cam, view = chunk.split(":", 1)
        out[cam.strip()] = view.strip()
    return out


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        x = float(value)
        return None if not math.isfinite(x) else x
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _first_gt_failure_frame(traj: BenchmarkTrajectory) -> Optional[int]:
    if not bool(traj.is_failure):
        return None
    mask = traj.load_failure_mask()
    if mask is not None:
        mask_arr = np.asarray(mask).reshape(-1)
        positive = np.where(mask_arr > 0)[0]
        if positive.size > 0:
            return int(positive[0])
    first = traj.first_gt_failure_frame()
    return None if first is None else int(first)


def _phase_labels(is_failure: bool, n_frames: int, first_gt: Optional[int]) -> list[str]:
    if not is_failure:
        return [PHASE_SUCCESS] * int(n_frames)
    cutoff = int(first_gt) if first_gt is not None else int(n_frames)
    labels: list[str] = []
    for frame_idx in range(int(n_frames)):
        labels.append(PHASE_FAILURE_AFTER_GT if frame_idx >= cutoff else PHASE_FAILURE_BEFORE_GT)
    return labels


def _collect_encoded_trajectories(
    discriminator: LPBV2BenchmarkDiscriminator,
    trajectories: list[BenchmarkTrajectory],
) -> list[EncodedTrajectory]:
    encoded: list[EncodedTrajectory] = []
    for traj_idx, traj in enumerate(trajectories):
        feat = discriminator._encode(traj).detach().cpu().numpy().astype(np.float32, copy=False)
        n = int(min(feat.shape[0], int(traj.num_frames)))
        if n <= 0:
            continue
        first_gt = _first_gt_failure_frame(traj)
        item = EncodedTrajectory(
            trajectory_index=int(traj_idx),
            task_name=str(traj.task_name),
            video_id=str(traj.video_id),
            is_failure=bool(traj.is_failure),
            features=feat[:n],
            first_gt_failure_frame=first_gt,
            failure_segments=list(traj.failure_segments),
        )
        encoded.append(item)
        print(
            f"[lpb_v2][sep] encoded {traj_idx + 1}/{len(trajectories)} "
            f"{traj.describe()} -> {tuple(item.features.shape)} first_gt={first_gt}",
            flush=True,
        )
    if not encoded:
        raise RuntimeError("No latent features were encoded.")
    return encoded


def _build_frame_arrays(encoded: list[EncodedTrajectory]) -> dict[str, np.ndarray]:
    features_list: list[np.ndarray] = []
    task_name: list[str] = []
    video_id: list[str] = []
    trajectory_index: list[int] = []
    frame_idx: list[int] = []
    is_failure: list[bool] = []
    phase: list[str] = []
    first_gt_per_frame: list[int] = []
    for item in encoded:
        n = int(item.features.shape[0])
        if n == 0:
            continue
        features_list.append(item.features)
        task_name.extend([item.task_name] * n)
        video_id.extend([item.video_id] * n)
        trajectory_index.extend([int(item.trajectory_index)] * n)
        frame_idx.extend(range(n))
        is_failure.extend([bool(item.is_failure)] * n)
        phase.extend(_phase_labels(item.is_failure, n, item.first_gt_failure_frame))
        sentinel = -1 if item.first_gt_failure_frame is None else int(item.first_gt_failure_frame)
        first_gt_per_frame.extend([sentinel] * n)
    return {
        "features": np.concatenate(features_list, axis=0).astype(np.float32, copy=False),
        "task_name": np.asarray(task_name, dtype=np.str_),
        "video_id": np.asarray(video_id, dtype=np.str_),
        "trajectory_index": np.asarray(trajectory_index, dtype=np.int64),
        "frame_idx": np.asarray(frame_idx, dtype=np.int64),
        "is_failure": np.asarray(is_failure, dtype=bool),
        "phase": np.asarray(phase, dtype=np.str_),
        "first_gt_failure_frame": np.asarray(first_gt_per_frame, dtype=np.int64),
    }


def _save_latents_cache(out_dir: Path, arrays: dict[str, np.ndarray], meta: dict) -> tuple[str, str]:
    npz_path = out_dir / "latents.npz"
    np.savez_compressed(npz_path, **arrays)
    meta_path = out_dir / "latents_meta.json"
    with meta_path.open("w") as fp:
        json.dump(_json_safe(meta), fp, indent=2)
    return str(npz_path), str(meta_path)


def _load_latents_cache(path: str) -> tuple[dict[str, np.ndarray], dict]:
    npz = np.load(path, allow_pickle=False)
    arrays = {key: npz[key] for key in npz.files}
    meta_path = Path(path).with_name("latents_meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return arrays, meta


def _sample_indices(n: int, max_rows: int, rng: np.random.Generator) -> np.ndarray:
    if int(max_rows) <= 0 or n <= int(max_rows):
        return np.arange(n, dtype=np.int64)
    return np.sort(rng.choice(n, size=int(max_rows), replace=False))


def _squared_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    x_norm = np.sum(x64 * x64, axis=1, keepdims=True)
    y_norm = np.sum(y64 * y64, axis=1, keepdims=True).T
    dist2 = x_norm + y_norm - 2.0 * (x64 @ y64.T)
    return np.maximum(dist2, 0.0)


def _median_sq_distance(x: np.ndarray, *, rng: np.random.Generator, max_points: int) -> float:
    n = int(x.shape[0])
    if n < 2:
        return 1.0
    if int(max_points) > 0 and n > int(max_points):
        idx = np.sort(rng.choice(n, size=int(max_points), replace=False))
        work = x[idx]
    else:
        work = x
    dist2 = _squared_distances(work, work)
    upper = dist2[np.triu_indices(dist2.shape[0], k=1)]
    upper = upper[np.isfinite(upper) & (upper > 0.0)]
    if upper.size == 0:
        return 1.0
    return float(max(np.median(upper), 1e-12))


def _mmd_permutation_test(
    success: np.ndarray,
    failure: np.ndarray,
    *,
    permutations: int,
    median_max_points: int,
    seed: int,
) -> dict:
    if success.ndim != 2 or failure.ndim != 2:
        return {"skipped_reason": "success and failure features must be 2D"}
    if success.shape[0] < 2 or failure.shape[0] < 2:
        return {"skipped_reason": "need at least two frames per class"}
    if success.shape[1] != failure.shape[1]:
        return {"skipped_reason": f"feature dim mismatch: {success.shape[1]} vs {failure.shape[1]}"}

    rng = np.random.default_rng(int(seed))
    z = np.concatenate([success, failure], axis=0).astype(np.float32, copy=False)
    n_success = int(success.shape[0])
    n_failure = int(failure.shape[0])
    bandwidth_sq = _median_sq_distance(z, rng=rng, max_points=int(median_max_points))
    kernel = np.exp(-_squared_distances(z, z) / bandwidth_sq).astype(np.float32, copy=False)

    weights = np.concatenate(
        [
            np.full((n_success,), 1.0 / float(n_success), dtype=np.float64),
            np.full((n_failure,), -1.0 / float(n_failure), dtype=np.float64),
        ],
        axis=0,
    )
    observed = float(weights @ (kernel @ weights))

    count_ge = 0
    num_perm = int(permutations)
    for _ in range(num_perm):
        perm = rng.permutation(weights.shape[0])
        w_perm = weights[perm]
        stat = float(w_perm @ (kernel @ w_perm))
        if stat >= observed - 1e-15:
            count_ge += 1

    p_value = float((count_ge + 1) / (num_perm + 1))
    return {
        "mmd2_biased": observed,
        "p_value": p_value,
        "permutations": int(num_perm),
        "num_success_frames": int(n_success),
        "num_failure_after_gt_frames": int(n_failure),
        "kernel": "gaussian",
        "median_sq_distance": float(bandwidth_sq),
        "rbf_gamma": float(1.0 / bandwidth_sq),
    }


def _mmd_decision(p_value: Optional[float]) -> str:
    if p_value is None:
        return "skipped"
    if float(p_value) < 0.01:
        return "different_distributions"
    if float(p_value) > 0.05:
        return "no_frame_level_signal_by_mmd"
    return "inconclusive"


def _split_success_rollouts(
    arrays: dict[str, np.ndarray],
    *,
    train_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_mask, eval_mask) over frames for §3.2 Mahalanobis.

    Splits *success* rollouts globally (over unique video_id) into a fit set
    and an eval set. Frames from failure rollouts are not in either mask;
    they are added separately as evaluation positives downstream.
    """
    is_success = ~arrays["is_failure"]
    success_video_ids = np.unique(arrays["video_id"][is_success])
    if success_video_ids.size < 2:
        raise RuntimeError("Need >=2 success rollouts for a rollout-level split.")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(success_video_ids.size)
    success_video_ids = success_video_ids[order]
    n_train = max(1, int(round(float(train_frac) * success_video_ids.size)))
    n_train = min(n_train, success_video_ids.size - 1)
    train_ids = set(success_video_ids[:n_train].tolist())
    eval_ids = set(success_video_ids[n_train:].tolist())

    video_id = arrays["video_id"]
    train_mask = np.zeros(video_id.shape[0], dtype=bool)
    eval_mask = np.zeros(video_id.shape[0], dtype=bool)
    for i, vid in enumerate(video_id):
        if not is_success[i]:
            continue
        if vid in train_ids:
            train_mask[i] = True
        elif vid in eval_ids:
            eval_mask[i] = True
    return train_mask, eval_mask


def _mahalanobis_fit_and_score(
    features_train: np.ndarray,
    features_eval: np.ndarray,
) -> tuple[LedoitWolf, np.ndarray]:
    lw = LedoitWolf(store_precision=True, assume_centered=False).fit(features_train)
    scores = lw.mahalanobis(features_eval)
    return lw, np.asarray(scores, dtype=np.float64)


def _mahalanobis_auroc(
    arrays: dict[str, np.ndarray],
    train_mask: np.ndarray,
    eval_mask_success: np.ndarray,
) -> dict:
    """Pooled Mahalanobis AUROC.

    Returns the AUROC + ROC curve over pooled
    (success_eval, failure_after_gt) frames. Score is shrinkage-Mahalanobis
    distance from the success-train mean.
    """
    features = arrays["features"]
    phase = arrays["phase"]
    failure_mask = phase == PHASE_FAILURE_AFTER_GT
    if int(train_mask.sum()) < 2:
        return {"skipped_reason": "not enough success-train frames"}
    if int(eval_mask_success.sum()) == 0 or int(failure_mask.sum()) == 0:
        return {"skipped_reason": "missing eval success or failure_after_gt frames"}

    train_X = features[train_mask]
    lw, scores_succ_eval = _mahalanobis_fit_and_score(train_X, features[eval_mask_success])
    scores_fail = lw.mahalanobis(features[failure_mask])

    y_true = np.concatenate(
        [
            np.zeros(scores_succ_eval.shape[0], dtype=np.int64),
            np.ones(scores_fail.shape[0], dtype=np.int64),
        ]
    )
    scores = np.concatenate([scores_succ_eval, scores_fail])
    auroc = float(roc_auc_score(y_true, scores))
    fpr, tpr, _ = roc_curve(y_true, scores)

    return {
        "auroc": auroc,
        "n_train": int(train_X.shape[0]),
        "n_eval_succ": int(scores_succ_eval.shape[0]),
        "n_eval_fail": int(scores_fail.shape[0]),
        "feature_dim": int(features.shape[1]),
        "roc_curve": {"fpr": fpr.astype(np.float32), "tpr": tpr.astype(np.float32)},
        "_scores_succ": scores_succ_eval.astype(np.float32),
        "_scores_fail": scores_fail.astype(np.float32),
        "_lw_mean": lw.location_.astype(np.float32),
        "_lw_precision": lw.precision_.astype(np.float32),
    }


def _matched_timestep_auroc(
    arrays: dict[str, np.ndarray],
    eval_mask_success: np.ndarray,
    lw_mean: np.ndarray,
    lw_precision: np.ndarray,
    *,
    window: int,
) -> dict:
    """Same-task matched-timestep control (§3.3) pooled into one AUROC.

    For every failure_after_gt frame at ``(task=T, t=k)`` we pull every
    success eval frame in task ``T`` with ``|t - k| <= window``, scored with
    the global Mahalanobis. The matched success scores and the failure
    score are concatenated across all (T, k) pairs and a single AUROC is
    computed over the pool. A success frame may appear in multiple matched
    sets, which is intentional under the pooling rule.
    """
    features = arrays["features"]
    task_name = arrays["task_name"]
    frame_idx = arrays["frame_idx"]
    phase = arrays["phase"]
    failure_mask = phase == PHASE_FAILURE_AFTER_GT
    if int(failure_mask.sum()) == 0 or int(eval_mask_success.sum()) == 0:
        return {"skipped_reason": "missing eval success or failure_after_gt frames"}

    succ_by_task: dict[str, dict[int, list[int]]] = {}
    succ_idx = np.where(eval_mask_success)[0]
    for i in succ_idx:
        t_key = str(task_name[i])
        k_key = int(frame_idx[i])
        succ_by_task.setdefault(t_key, {}).setdefault(k_key, []).append(int(i))

    P = lw_precision.astype(np.float64)
    mu = lw_mean.astype(np.float64)

    def _score(rows: np.ndarray) -> np.ndarray:
        diff = features[rows].astype(np.float64) - mu
        return np.einsum("ni,ij,nj->n", diff, P, diff)

    matched_succ_scores: list[float] = []
    matched_fail_scores: list[float] = []
    n_pairs = 0
    fail_indices = np.where(failure_mask)[0]
    for j in fail_indices:
        t_key = str(task_name[j])
        k = int(frame_idx[j])
        by_k = succ_by_task.get(t_key, {})
        matched_rows: list[int] = []
        for off in range(-int(window), int(window) + 1):
            sel = by_k.get(k + off)
            if sel:
                matched_rows.extend(sel)
        if not matched_rows:
            continue
        n_pairs += 1
        succ_scores = _score(np.asarray(matched_rows, dtype=np.int64))
        fail_score = float(_score(np.asarray([int(j)], dtype=np.int64))[0])
        matched_succ_scores.extend(succ_scores.tolist())
        matched_fail_scores.append(fail_score)

    if not matched_succ_scores or not matched_fail_scores:
        return {"skipped_reason": "no matched (T, k) pairs within window"}

    y_true = np.concatenate(
        [
            np.zeros(len(matched_succ_scores), dtype=np.int64),
            np.ones(len(matched_fail_scores), dtype=np.int64),
        ]
    )
    scores = np.concatenate(
        [
            np.asarray(matched_succ_scores, dtype=np.float64),
            np.asarray(matched_fail_scores, dtype=np.float64),
        ]
    )
    auroc = float(roc_auc_score(y_true, scores))
    fpr, tpr, _ = roc_curve(y_true, scores)
    return {
        "auroc": auroc,
        "n_pairs": int(n_pairs),
        "n_matched_succ_scores": int(len(matched_succ_scores)),
        "n_matched_fail_scores": int(len(matched_fail_scores)),
        "window": int(window),
        "roc_curve": {"fpr": fpr.astype(np.float32), "tpr": tpr.astype(np.float32)},
    }


def _plot_roc(curve: dict, auroc: float, out_path: Path, title: str) -> None:
    fpr = np.asarray(curve["fpr"], dtype=np.float64)
    tpr = np.asarray(curve["tpr"], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=120)
    ax.plot(fpr, tpr, label=f"AUROC = {auroc:.3f}", linewidth=1.8)
    ax.plot([0.0, 1.0], [0.0, 1.0], color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_hist(scores_succ: np.ndarray, scores_fail: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 3.5), dpi=120)
    lo = float(min(scores_succ.min(), scores_fail.min()))
    hi = float(max(scores_succ.max(), scores_fail.max()))
    bins = np.linspace(lo, hi, 60)
    ax.hist(
        scores_succ,
        bins=bins,
        alpha=0.55,
        color="#9ecae1",
        label=f"success (n={scores_succ.size})",
        density=True,
    )
    ax.hist(
        scores_fail,
        bins=bins,
        alpha=0.55,
        color="#de2d26",
        label=f"failure_after_gt (n={scores_fail.size})",
        density=True,
    )
    ax.set_xlabel("Mahalanobis score")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _gate(mmd_p: Optional[float], mahal_auroc: Optional[float]) -> dict:
    mmd_pass = bool(mmd_p is not None and float(mmd_p) < 0.01)
    mmd_fail = bool(mmd_p is not None and float(mmd_p) > 0.05)
    auroc_60 = bool(mahal_auroc is not None and float(mahal_auroc) > 0.60)
    auroc_75 = bool(mahal_auroc is not None and float(mahal_auroc) > 0.75)
    if mmd_fail:
        decision = "fix_representation_first"
    elif mahal_auroc is None:
        decision = "skipped"
    elif auroc_75:
        decision = "proceed_to_phase_2_strong"
    elif auroc_60:
        decision = "proceed_to_phase_2_marginal"
    else:
        decision = "fix_representation_first"
    return {
        "mmd_p_lt_0_01": mmd_pass,
        "mmd_p_gt_0_05": mmd_fail,
        "auroc_above_0_60": auroc_60,
        "auroc_above_0_75": auroc_75,
        "decision": decision,
    }


def _per_task_side_panel(
    arrays: dict[str, np.ndarray],
    *,
    permutations: int,
    median_max_points: int,
    max_frames_per_class: int,
    seed: int,
    lw_mean: np.ndarray,
    lw_precision: np.ndarray,
    eval_mask_success: np.ndarray,
) -> dict[str, dict]:
    """Per-task sanity panel. Reports MMD p and Mahalanobis AUROC per task.

    Mahalanobis AUROC reuses the global pooled Ledoit-Wolf fit so per-task
    numbers are comparable to the headline. MMD permutation tests are run
    per task on a subsample to keep cost bounded.
    """
    features = arrays["features"]
    phase = arrays["phase"]
    task_name = arrays["task_name"]
    failure_mask = phase == PHASE_FAILURE_AFTER_GT
    P = lw_precision.astype(np.float64)
    mu = lw_mean.astype(np.float64)

    out: dict[str, dict] = {}
    for i, task in enumerate(sorted({str(t) for t in task_name.tolist()})):
        task_mask = task_name == task
        succ_eval = eval_mask_success & task_mask
        fail = failure_mask & task_mask
        side: dict = {
            "num_success_eval_frames": int(succ_eval.sum()),
            "num_failure_after_gt_frames": int(fail.sum()),
        }

        rng = np.random.default_rng(int(seed) + 9000 + i)
        if int(succ_eval.sum()) >= 2 and int(fail.sum()) >= 2:
            s_idx = np.where(succ_eval)[0]
            f_idx = np.where(fail)[0]
            s_idx = s_idx[_sample_indices(s_idx.size, max_frames_per_class, rng)]
            f_idx = f_idx[_sample_indices(f_idx.size, max_frames_per_class, rng)]
            side["mmd"] = _mmd_permutation_test(
                features[s_idx],
                features[f_idx],
                permutations=int(permutations),
                median_max_points=int(median_max_points),
                seed=int(seed) + 9100 + i,
            )

        if int(succ_eval.sum()) > 0 and int(fail.sum()) > 0:
            diff_s = features[succ_eval].astype(np.float64) - mu
            diff_f = features[fail].astype(np.float64) - mu
            s_scores = np.einsum("ni,ij,nj->n", diff_s, P, diff_s)
            f_scores = np.einsum("ni,ij,nj->n", diff_f, P, diff_f)
            y_true = np.concatenate(
                [np.zeros(s_scores.size, dtype=np.int64), np.ones(f_scores.size, dtype=np.int64)]
            )
            side["mahalanobis_auroc"] = float(
                roc_auc_score(y_true, np.concatenate([s_scores, f_scores]))
            )
        out[task] = side
    return out


def _build_meta(
    encoded: list[EncodedTrajectory],
    args: argparse.Namespace,
    arrays: dict[str, np.ndarray],
) -> dict:
    phases, counts = np.unique(arrays["phase"], return_counts=True)
    phase_counts = {str(p): int(c) for p, c in zip(phases, counts)}
    return {
        "config": {
            "model_ckpt": str(args.model_ckpt) if args.model_ckpt else None,
            "feature_source": str(args.feature_source),
            "transformer_layer": int(args.transformer_layer),
            "fail_root": str(args.fail_root) if args.fail_root else None,
            "success_root": str(args.success_root) if args.success_root else None,
            "tasks": sorted({str(t) for t in arrays["task_name"].tolist()}),
        },
        "num_trajectories": int(len(encoded)),
        "num_failure_trajectories": int(sum(1 for it in encoded if it.is_failure)),
        "num_success_trajectories": int(sum(1 for it in encoded if not it.is_failure)),
        "num_frames": int(arrays["features"].shape[0]),
        "feature_dim": int(arrays["features"].shape[1]),
        "phase_counts": phase_counts,
        "trajectories": [
            {
                "trajectory_index": int(item.trajectory_index),
                "task_name": item.task_name,
                "video_id": item.video_id,
                "is_failure": bool(item.is_failure),
                "num_encoded_frames": int(item.features.shape[0]),
                "first_gt_failure_frame": item.first_gt_failure_frame,
                "failure_segments": item.failure_segments,
            }
            for item in encoded
        ],
    }


def _strip_internal(d: dict) -> dict:
    return {k: v for k, v in d.items() if not str(k).startswith("_")}


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.load_cache:
        arrays, meta = _load_latents_cache(args.load_cache)
        print(
            f"[lpb_v2][sep] loaded cache from {args.load_cache} "
            f"features={tuple(arrays['features'].shape)}",
            flush=True,
        )
        cache_paths = {
            "latents": str(args.load_cache),
            "latents_meta": str(Path(args.load_cache).with_name("latents_meta.json")),
        }
    else:
        if not (args.model_ckpt and args.fail_root and args.success_root):
            raise RuntimeError(
                "--model-ckpt, --fail-root and --success-root are required unless --load-cache is set."
            )
        bench = FailureBenchmark(
            fail_labeled_root=args.fail_root,
            success_root=args.success_root,
            tasks=args.tasks,
            max_fail_per_task=args.max_fail_per_task,
            max_success_per_task=args.max_success_per_task,
            cache_root=args.cache_root,
            proprio_field=args.proprio_field,
            proprio_slice=slice(int(args.proprio_start), int(args.proprio_stop)),
            action_slice=slice(int(args.action_start), int(args.action_stop)),
        )
        trajectories = bench.trajectories()
        if not trajectories:
            raise RuntimeError("No real-world trajectories discovered.")
        n_fail_traj = sum(1 for traj in trajectories if bool(traj.is_failure))
        n_success_traj = int(len(trajectories) - n_fail_traj)
        tasks = sorted({str(traj.task_name) for traj in trajectories})
        print(
            f"[lpb_v2][sep] discovered {len(trajectories)} trajectories "
            f"(failure={n_fail_traj}, success={n_success_traj}) tasks={tasks}",
            flush=True,
        )

        discriminator = LPBV2BenchmarkDiscriminator(
            model_ckpt=str(args.model_ckpt),
            device=str(args.device),
            encode_batch_size=int(args.encode_batch_size),
            proprio_indices=(list(args.proprio_indices) if args.proprio_indices else None),
            camera_to_view=_parse_camera_to_view(args.camera_to_view),
            feature_source=str(args.feature_source),
            transformer_layer=int(args.transformer_layer),
            verbose_fit=False,
        )
        try:
            encoded = _collect_encoded_trajectories(discriminator, trajectories)
        finally:
            discriminator.close()
        arrays = _build_frame_arrays(encoded)
        meta = _build_meta(encoded, args, arrays)
        npz_path, meta_path = _save_latents_cache(out_dir, arrays, meta)
        cache_paths = {"latents": npz_path, "latents_meta": meta_path}
        print(f"[lpb_v2][sep] wrote {npz_path}", flush=True)
        print(f"[lpb_v2][sep] wrote {meta_path}", flush=True)

    if args.cache_features_only:
        print("[lpb_v2][sep] --cache-features-only set; skipping analysis.", flush=True)
        return

    features = arrays["features"]
    phase = arrays["phase"]
    is_success_mask = phase == PHASE_SUCCESS
    is_failure_after_mask = phase == PHASE_FAILURE_AFTER_GT
    n_succ = int(is_success_mask.sum())
    n_fail = int(is_failure_after_mask.sum())
    print(
        f"[lpb_v2][sep] pooled frame counts: success={n_succ} failure_after_gt={n_fail} "
        f"feature_dim={features.shape[1]}",
        flush=True,
    )

    train_mask, eval_succ_mask = _split_success_rollouts(
        arrays, train_frac=float(args.mahalanobis_train_frac), seed=int(args.seed)
    )
    print(
        f"[lpb_v2][sep] rollout-level success split: "
        f"train_frames={int(train_mask.sum())} eval_frames={int(eval_succ_mask.sum())}",
        flush=True,
    )

    # §3.1 pooled MMD on all success vs all failure_after_gt (subsampled).
    rng = np.random.default_rng(int(args.seed))
    succ_idx_all = np.where(is_success_mask)[0]
    fail_idx_all = np.where(is_failure_after_mask)[0]
    s_sub = succ_idx_all[_sample_indices(succ_idx_all.size, int(args.max_frames_per_class), rng)]
    f_sub = fail_idx_all[_sample_indices(fail_idx_all.size, int(args.max_frames_per_class), rng)]
    mmd_result = _mmd_permutation_test(
        features[s_sub],
        features[f_sub],
        permutations=int(args.permutations),
        median_max_points=int(args.median_max_points),
        seed=int(args.seed) + 17,
    )
    mmd_p = mmd_result.get("p_value")
    print(
        f"[lpb_v2][sep] MMD pooled: p={mmd_p} mmd2={mmd_result.get('mmd2_biased')}",
        flush=True,
    )

    # §3.2 pooled Mahalanobis AUROC.
    mahal = _mahalanobis_auroc(arrays, train_mask, eval_succ_mask)
    if "auroc" in mahal:
        _plot_roc(mahal["roc_curve"], mahal["auroc"], out_dir / "roc.png", "Pooled Mahalanobis ROC")
        _plot_hist(
            mahal["_scores_succ"],
            mahal["_scores_fail"],
            out_dir / "hist.png",
            "Mahalanobis score (pooled)",
        )
        print(f"[lpb_v2][sep] Mahalanobis pooled AUROC = {mahal['auroc']:.4f}", flush=True)

    # §3.3 same-task matched-timestep AUROC.
    matched: dict = {}
    if "_lw_mean" in mahal:
        matched = _matched_timestep_auroc(
            arrays,
            eval_succ_mask,
            mahal["_lw_mean"],
            mahal["_lw_precision"],
            window=int(args.matched_window),
        )
        if "auroc" in matched:
            _plot_roc(
                matched["roc_curve"],
                matched["auroc"],
                out_dir / "roc_matched.png",
                f"Matched-timestep ROC (|Δt|≤{int(args.matched_window)})",
            )
            print(
                f"[lpb_v2][sep] Matched-timestep AUROC = {matched['auroc']:.4f} "
                f"(n_pairs={matched.get('n_pairs')})",
                flush=True,
            )

    # Per-task side panel (sanity-only, not part of the gate).
    side_panel: dict = {}
    if "_lw_mean" in mahal:
        side_panel = _per_task_side_panel(
            arrays,
            permutations=min(int(args.permutations), 500),
            median_max_points=int(args.median_max_points),
            max_frames_per_class=int(args.max_frames_per_class),
            seed=int(args.seed),
            lw_mean=mahal["_lw_mean"],
            lw_precision=mahal["_lw_precision"],
            eval_mask_success=eval_succ_mask,
        )

    pooled_block = {
        "mmd": {**mmd_result, "decision": _mmd_decision(mmd_p)},
        "mahalanobis": _strip_internal(mahal),
        "matched_timestep": matched,
        "gate": _gate(mmd_p, mahal.get("auroc")),
    }

    summary = {
        "config": {
            "model_ckpt": str(args.model_ckpt) if args.model_ckpt else None,
            "fail_root": str(args.fail_root) if args.fail_root else None,
            "success_root": str(args.success_root) if args.success_root else None,
            "cache_root": args.cache_root,
            "tasks": (
                meta.get("config", {}).get("tasks")
                if isinstance(meta.get("config"), dict)
                else None
            ),
            "feature_source": str(args.feature_source),
            "transformer_layer": int(args.transformer_layer),
            "seed": int(args.seed),
            "permutations": int(args.permutations),
            "max_frames_per_class": int(args.max_frames_per_class),
            "median_max_points": int(args.median_max_points),
            "mahalanobis_train_frac": float(args.mahalanobis_train_frac),
            "matched_window": int(args.matched_window),
        },
        "dataset": {
            "num_frames": int(arrays["features"].shape[0]),
            "num_success_frames": n_succ,
            "num_failure_after_gt_frames": n_fail,
            "feature_dim": int(arrays["features"].shape[1]),
            "tasks": sorted({str(t) for t in arrays["task_name"].tolist()}),
        },
        "outputs": {
            "out_dir": str(out_dir),
            "latents_npz": cache_paths["latents"],
            "latents_meta": cache_paths["latents_meta"],
        },
        "pooled": pooled_block,
        "per_task_side_panel": side_panel,
    }

    summary_path = out_dir / "separability_summary.json"
    with summary_path.open("w") as fp:
        json.dump(_json_safe(summary), fp, indent=2)
    print(f"[lpb_v2][sep] wrote {summary_path}", flush=True)
    print(
        "[lpb_v2][sep] gate:",
        json.dumps(_json_safe(pooled_block["gate"]), sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
