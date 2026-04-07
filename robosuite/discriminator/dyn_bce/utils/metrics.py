from __future__ import annotations

import numpy as np

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except Exception:  # pragma: no cover
    average_precision_score = None
    roc_auc_score = None


def _safe_numpy(x) -> np.ndarray:
    arr = np.asarray(x)
    return arr.reshape(-1)


def compute_binary_metrics(
    probs,
    labels,
    threshold: float = 0.5,
) -> dict[str, float]:
    probs_np = _safe_numpy(probs).astype(np.float64)
    labels_np = _safe_numpy(labels).astype(np.int64)
    if probs_np.size == 0:
        return {}
    if probs_np.shape[0] != labels_np.shape[0]:
        raise ValueError(
            f"probs and labels must have the same number of samples, got {probs_np.shape[0]} and {labels_np.shape[0]}"
        )

    finite_mask = np.isfinite(probs_np)
    finite_count = int(finite_mask.sum())
    total_count = int(finite_mask.shape[0])
    if finite_count <= 0:
        return {
            "nonfinite_rate": 1.0,
            "finite_rate": 0.0,
        }

    if finite_count != total_count:
        probs_np = probs_np[finite_mask]
        labels_np = labels_np[finite_mask]

    preds_np = (probs_np >= float(threshold)).astype(np.int64)
    accuracy = float((preds_np == labels_np).mean())

    tp = int(np.sum((preds_np == 1) & (labels_np == 1)))
    fp = int(np.sum((preds_np == 1) & (labels_np == 0)))
    fn = int(np.sum((preds_np == 0) & (labels_np == 1)))
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    f1 = float(2.0 * precision * recall / max(precision + recall, 1e-8))

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "positive_rate": float(preds_np.mean()),
        "label_rate": float(labels_np.mean()),
        "nonfinite_rate": float(1.0 - finite_count / max(total_count, 1)),
        "finite_rate": float(finite_count / max(total_count, 1)),
    }

    if not labels_np.min() == labels_np.max():
        if roc_auc_score is not None:
            metrics["auroc"] = float(roc_auc_score(labels_np, probs_np))
        if average_precision_score is not None:
            metrics["auprc"] = float(average_precision_score(labels_np, probs_np))
    return metrics


def estimate_best_threshold(
    probs,
    labels,
    objective: str = "f1",
    default_threshold: float = 0.5,
) -> dict[str, float]:
    probs_np = _safe_numpy(probs).astype(np.float64)
    labels_np = _safe_numpy(labels).astype(np.int64)
    if probs_np.size == 0:
        return {
            "threshold": float(default_threshold),
            "objective": 0.0,
            "f1": 0.0,
            "balanced_accuracy": 0.0,
            "label_rate": 0.0,
            "positive_rate": 0.0,
        }
    if probs_np.shape[0] != labels_np.shape[0]:
        raise ValueError(
            f"probs and labels must have the same number of samples, got {probs_np.shape[0]} and {labels_np.shape[0]}"
        )

    finite_mask = np.isfinite(probs_np)
    probs_np = probs_np[finite_mask]
    labels_np = labels_np[finite_mask]
    if probs_np.size == 0 or labels_np.size == 0:
        return {
            "threshold": float(default_threshold),
            "objective": 0.0,
            "f1": 0.0,
            "balanced_accuracy": 0.0,
            "label_rate": 0.0,
            "positive_rate": 0.0,
        }

    positive_total = int(np.sum(labels_np == 1))
    negative_total = int(np.sum(labels_np == 0))
    label_rate = float(labels_np.mean())
    if positive_total <= 0 or negative_total <= 0:
        return {
            "threshold": float(default_threshold),
            "objective": 0.0,
            "f1": 0.0,
            "balanced_accuracy": 0.0,
            "label_rate": label_rate,
            "positive_rate": float(label_rate),
        }

    order = np.argsort(-probs_np, kind="mergesort")
    sorted_probs = probs_np[order]
    sorted_labels = labels_np[order]

    positive_cumsum = np.cumsum(sorted_labels == 1)
    negative_cumsum = np.cumsum(sorted_labels == 0)
    group_end = np.nonzero(np.r_[sorted_probs[:-1] != sorted_probs[1:], True])[0]

    tp = positive_cumsum[group_end].astype(np.float64)
    fp = negative_cumsum[group_end].astype(np.float64)
    fn = float(positive_total) - tp
    tn = float(negative_total) - fp
    preds = (group_end + 1).astype(np.float64)

    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / max(float(positive_total), 1.0)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-8)
    balanced_accuracy = 0.5 * (recall + tn / max(float(negative_total), 1.0))
    positive_rate = preds / max(float(labels_np.shape[0]), 1.0)

    if objective == "balanced_accuracy":
        objective_values = balanced_accuracy
    else:
        objective_values = f1

    best_value = float(np.max(objective_values))
    candidate_idx = np.flatnonzero(np.isclose(objective_values, best_value, atol=1e-12, rtol=0.0))
    if candidate_idx.size > 1:
        label_rate_deltas = np.abs(positive_rate[candidate_idx] - label_rate)
        best_idx = int(candidate_idx[int(np.argmin(label_rate_deltas))])
    else:
        best_idx = int(candidate_idx[0])

    return {
        "threshold": float(sorted_probs[group_end][best_idx]),
        "objective": float(objective_values[best_idx]),
        "f1": float(f1[best_idx]),
        "balanced_accuracy": float(balanced_accuracy[best_idx]),
        "label_rate": label_rate,
        "positive_rate": float(positive_rate[best_idx]),
    }


def compute_score_ranking_metrics(
    scores,
    labels,
) -> dict[str, float]:
    scores_np = _safe_numpy(scores).astype(np.float64)
    labels_np = _safe_numpy(labels).astype(np.int64)
    if scores_np.size == 0:
        return {}
    if scores_np.shape[0] != labels_np.shape[0]:
        raise ValueError(
            f"scores and labels must have the same number of samples, got {scores_np.shape[0]} and {labels_np.shape[0]}"
        )

    finite_mask = np.isfinite(scores_np)
    finite_count = int(finite_mask.sum())
    total_count = int(finite_mask.shape[0])
    if finite_count <= 0:
        return {
            "nonfinite_rate": 1.0,
            "finite_rate": 0.0,
        }

    if finite_count != total_count:
        scores_np = scores_np[finite_mask]
        labels_np = labels_np[finite_mask]

    metrics = {
        "mean": float(scores_np.mean()),
        "nonfinite_rate": float(1.0 - finite_count / max(total_count, 1)),
        "finite_rate": float(finite_count / max(total_count, 1)),
        "label_rate": float(labels_np.mean()),
    }
    if not labels_np.min() == labels_np.max():
        if roc_auc_score is not None:
            metrics["auroc"] = float(roc_auc_score(labels_np, scores_np))
        if average_precision_score is not None:
            metrics["auprc"] = float(average_precision_score(labels_np, scores_np))
    return metrics
