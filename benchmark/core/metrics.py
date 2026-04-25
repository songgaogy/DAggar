"""Metrics for the failure-detector benchmark.

Two evaluation axes:
    (1) Trajectory-level: success vs failure. Classic binary classification
        on an aggregated trajectory score (max / mean / topk_mean of step_scores).
    (2) Step-level (failure trajectories only): how well per-frame scores or
        binary predictions align with ground-truth failure_frame_mask and
        failure segments.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


# ---------------------------------------------------------------------- #
# Aggregation                                                            #
# ---------------------------------------------------------------------- #


def aggregate_trajectory_score(
    step_scores: np.ndarray,
    mode: str = "max",
    topk: int = 20,
) -> float:
    s = np.asarray(step_scores, dtype=np.float64).reshape(-1)
    if s.size == 0:
        return float("nan")
    if mode == "max":
        return float(np.max(s))
    if mode == "mean":
        return float(np.mean(s))
    if mode == "topk_mean":
        k = max(1, min(int(topk), int(s.shape[0])))
        top = np.partition(s, -k)[-k:]
        return float(np.mean(top))
    raise ValueError(f"Unknown aggregator {mode!r}")


# ---------------------------------------------------------------------- #
# Trajectory-level                                                       #
# ---------------------------------------------------------------------- #


def _tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, fpr_target: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.searchsorted(fpr, fpr_target, side="right") - 1
    idx = max(0, int(idx))
    return float(tpr[idx])


def _fpr_at_tpr(y_true: np.ndarray, y_score: np.ndarray, tpr_target: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    mask = tpr >= tpr_target
    if not np.any(mask):
        return float("nan")
    return float(np.min(fpr[mask]))


def _best_f1(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # sklearn appends a 0/1 recall/precision, drop last point (no threshold).
    precision = precision[:-1]
    recall = recall[:-1]
    if len(thresholds) == 0:
        return float("nan"), float("nan")
    denom = np.where((precision + recall) > 0, precision + recall, 1.0)
    f1 = 2.0 * precision * recall / denom
    best = int(np.argmax(f1))
    return float(f1[best]), float(thresholds[best])


def trajectory_level_metrics(
    traj_scores: np.ndarray,
    traj_labels: np.ndarray,
) -> dict:
    """Binary metrics on aggregated trajectory scores. labels: 1=failure."""
    y_true = np.asarray(traj_labels, dtype=np.int64).reshape(-1)
    y_score = np.asarray(traj_scores, dtype=np.float64).reshape(-1)
    if y_true.size == 0:
        return {"num_trajectories": 0}

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    out: dict = {
        "num_trajectories": int(y_true.size),
        "num_failure": n_pos,
        "num_success": n_neg,
    }
    if n_pos == 0 or n_neg == 0:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
        return out

    finite = np.isfinite(y_score)
    if not np.all(finite):
        y_score = np.where(finite, y_score, np.nanmin(y_score[finite]) - 1.0)

    out["auroc"] = float(roc_auc_score(y_true, y_score))
    out["auprc"] = float(average_precision_score(y_true, y_score))
    out["tpr_at_fpr_0.05"] = _tpr_at_fpr(y_true, y_score, 0.05)
    out["tpr_at_fpr_0.10"] = _tpr_at_fpr(y_true, y_score, 0.10)
    out["fpr_at_tpr_0.95"] = _fpr_at_tpr(y_true, y_score, 0.95)
    f1, thr = _best_f1(y_true, y_score)
    out["best_f1"] = f1
    out["best_f1_threshold"] = thr
    return out


# ---------------------------------------------------------------------- #
# Step-level                                                             #
# ---------------------------------------------------------------------- #


def _pool_frames(
    per_traj_scores: Sequence[np.ndarray],
    per_traj_masks: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    flat_s: list[np.ndarray] = []
    flat_y: list[np.ndarray] = []
    for s, m in zip(per_traj_scores, per_traj_masks):
        s = np.asarray(s, dtype=np.float64).reshape(-1)
        m = np.asarray(m, dtype=np.int64).reshape(-1)
        if s.shape != m.shape:
            raise ValueError(f"per-step shape mismatch: {s.shape} vs {m.shape}")
        flat_s.append(s)
        flat_y.append(m)
    return np.concatenate(flat_s), np.concatenate(flat_y)


def frame_level_auroc_auprc(
    per_traj_scores: Sequence[np.ndarray],
    per_traj_masks: Sequence[np.ndarray],
) -> dict:
    if len(per_traj_scores) == 0:
        return {"frame_auroc": float("nan"), "frame_auprc": float("nan"), "num_frames": 0}
    s, y = _pool_frames(per_traj_scores, per_traj_masks)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    out = {"num_frames": int(s.size), "num_failure_frames": n_pos, "num_normal_frames": n_neg}
    if n_pos == 0 or n_neg == 0:
        out["frame_auroc"] = float("nan")
        out["frame_auprc"] = float("nan")
        return out
    finite = np.isfinite(s)
    if not np.all(finite):
        s = np.where(finite, s, np.nanmin(s[finite]) - 1.0)
    out["frame_auroc"] = float(roc_auc_score(y, s))
    out["frame_auprc"] = float(average_precision_score(y, s))
    return out


def _contiguous_segments(binary: np.ndarray) -> list[tuple[int, int]]:
    b = np.asarray(binary, dtype=np.int64).reshape(-1)
    if b.size == 0:
        return []
    pad = np.concatenate([[0], b, [0]])
    diff = np.diff(pad)
    starts = np.where(diff == 1)[0].tolist()
    ends = np.where(diff == -1)[0].tolist()
    return [(int(s), int(e - 1)) for s, e in zip(starts, ends)]


def _segment_iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
    union = max(a[1], b[1]) - min(a[0], b[0]) + 1
    return float(inter) / float(max(1, union))


def event_level_metrics(
    per_traj_predictions: Sequence[np.ndarray],
    per_traj_gt_segments: Sequence[list[dict]],
    per_traj_num_frames: Sequence[int],
    iou_threshold: float = 0.1,
) -> dict:
    """Segment-level recall / precision on failure trajectories."""
    total_gt = 0
    hit_gt = 0
    total_pred = 0
    hit_pred = 0
    delays: list[int] = []
    for pred, segs, T in zip(per_traj_predictions, per_traj_gt_segments, per_traj_num_frames):
        p = np.asarray(pred, dtype=np.int64).reshape(-1)
        if p.size == 0 or int(p.shape[0]) != int(T):
            continue
        gt_ranges = [(int(s["start"]), int(min(int(s["end"]), int(T) - 1))) for s in segs]
        pred_ranges = _contiguous_segments(p)
        total_gt += len(gt_ranges)
        total_pred += len(pred_ranges)
        for gt in gt_ranges:
            if any(_segment_iou(gt, pr) >= iou_threshold for pr in pred_ranges):
                hit_gt += 1
        for pr in pred_ranges:
            if any(_segment_iou(gt, pr) >= iou_threshold for gt in gt_ranges):
                hit_pred += 1
        if len(gt_ranges) > 0 and p.sum() > 0:
            first_gt = min(g[0] for g in gt_ranges)
            first_pred = int(np.argmax(p > 0))
            delays.append(int(first_pred) - int(first_gt))
    out = {
        "event_recall": float(hit_gt) / float(total_gt) if total_gt > 0 else float("nan"),
        "event_precision": float(hit_pred) / float(total_pred) if total_pred > 0 else float("nan"),
        "num_gt_segments": int(total_gt),
        "num_pred_segments": int(total_pred),
        "detection_delay_n": int(len(delays)),
    }
    if delays:
        arr = np.asarray(delays, dtype=np.float64)
        out["detection_delay_mean"] = float(arr.mean())
        out["detection_delay_median"] = float(np.median(arr))
        out["detection_delay_p25"] = float(np.percentile(arr, 25))
        out["detection_delay_p75"] = float(np.percentile(arr, 75))
    else:
        for k in ("mean", "median", "p25", "p75"):
            out[f"detection_delay_{k}"] = float("nan")
    return out


def frame_binary_metrics(
    per_traj_predictions: Sequence[np.ndarray],
    per_traj_masks: Sequence[np.ndarray],
) -> dict:
    if len(per_traj_predictions) == 0:
        return {
            "frame_f1": float("nan"),
            "frame_iou": float("nan"),
            "frame_precision": float("nan"),
            "frame_recall": float("nan"),
        }
    preds = np.concatenate(
        [np.asarray(p, dtype=np.int64).reshape(-1) for p in per_traj_predictions]
    )
    masks = np.concatenate(
        [np.asarray(m, dtype=np.int64).reshape(-1) for m in per_traj_masks]
    )
    if preds.shape != masks.shape:
        raise ValueError(f"preds / masks length mismatch: {preds.shape} vs {masks.shape}")
    tp = int(((preds == 1) & (masks == 1)).sum())
    fp = int(((preds == 1) & (masks == 0)).sum())
    fn = int(((preds == 0) & (masks == 1)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    iou = tp / max(1, tp + fp + fn)
    f1 = f1_score(masks, preds, zero_division=0)
    return {
        "frame_f1": float(f1),
        "frame_iou": float(iou),
        "frame_precision": float(precision),
        "frame_recall": float(recall),
        "frame_tp": tp,
        "frame_fp": fp,
        "frame_fn": fn,
    }
