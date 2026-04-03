from __future__ import annotations

from typing import Callable, Optional, Sequence, TypeVar

import numpy as np

from .base import OfflineTrajectoryDiscriminator
from .types import DetectorCalibrationSummary, TrajectoryDetectionResult


TTrajectory = TypeVar("TTrajectory")


def _safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return float(num / den)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred shape mismatch: {y_true.shape} vs {y_pred.shape}")

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    acc = _safe_div(tp + tn, y_true.size)
    return {
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": acc,
    }


def evaluate_trajectory_discriminator(
    detector: OfflineTrajectoryDiscriminator[TTrajectory],
    *,
    bank_trajectories: Sequence[TTrajectory],
    calibration_trajectories: Sequence[TTrajectory],
    expert_eval_trajectories: Sequence[TTrajectory],
    success_eval_trajectories: Sequence[TTrajectory],
    fail_eval_trajectories: Sequence[TTrajectory],
    fail_label_builder: Callable[[TTrajectory, TrajectoryDetectionResult], np.ndarray],
    adaptive_threshold: bool,
    delta_min: float,
    delta_max: float,
    warmup_steps: int,
    update_interval: int,
) -> tuple[dict[str, object], DetectorCalibrationSummary, dict[str, list[TrajectoryDetectionResult]]]:
    calibration_summary = detector.fit(
        normal_bank_trajectories=bank_trajectories,
        calibration_trajectories=calibration_trajectories,
    )

    expert_results: list[TrajectoryDetectionResult] = []
    success_results: list[TrajectoryDetectionResult] = []
    fail_results: list[TrajectoryDetectionResult] = []

    expert_false_alarm = 0
    expert_total = 0
    expert_lambda_values: list[float] = []
    for traj in expert_eval_trajectories:
        result = detector.detect_trajectory(traj, adaptive_threshold=False)
        expert_results.append(result)
        expert_false_alarm += int(np.sum(result.predictions))
        expert_total += int(result.predictions.size)
        expert_lambda_values.extend(result.aggregate_scores.tolist())

    success_false_alarm = 0
    success_total = 0
    success_lambda_values: list[float] = []
    for traj in success_eval_trajectories:
        result = detector.detect_trajectory(traj, adaptive_threshold=False)
        success_results.append(result)
        success_false_alarm += int(np.sum(result.predictions))
        success_total += int(result.predictions.size)
        success_lambda_values.extend(result.aggregate_scores.tolist())

    y_true: list[int] = []
    y_pred: list[int] = []
    fail_lambda_values: list[float] = []
    pre_fail_lambda_values: list[float] = []
    for traj in fail_eval_trajectories:
        draft_result = detector.detect_trajectory(traj, adaptive_threshold=False)
        labels = fail_label_builder(traj, draft_result)
        result = detector.detect_trajectory(
            traj,
            labels=labels if adaptive_threshold else None,
            adaptive_threshold=adaptive_threshold,
            delta_min=delta_min,
            delta_max=delta_max,
            warmup_steps=warmup_steps,
            update_interval=update_interval,
        )
        result.labels = np.asarray(labels, dtype=np.int64)
        fail_results.append(result)
        y_true.extend(result.labels.tolist())
        y_pred.extend(result.predictions.tolist())
        for lambda_value, label in zip(result.aggregate_scores.tolist(), result.labels.tolist()):
            if int(label) == 1:
                fail_lambda_values.append(float(lambda_value))
            else:
                pre_fail_lambda_values.append(float(lambda_value))

    fail_metrics = classification_metrics(
        y_true=np.asarray(y_true, dtype=np.int64),
        y_pred=np.asarray(y_pred, dtype=np.int64),
    )
    if fail_results:
        last_fail = fail_results[-1]
        fail_metrics["threshold_final"] = float(last_fail.metadata.get("threshold_final", np.nan))
        fail_metrics["delta_final"] = float(last_fail.metadata.get("delta_final", np.nan))
    else:
        fail_metrics["threshold_final"] = float("nan")
        fail_metrics["delta_final"] = float("nan")

    summary: dict[str, object] = {
        "counts": {
            "bank": len(bank_trajectories),
            "calibration": len(calibration_trajectories),
            "expert_eval": len(expert_eval_trajectories),
            "success_eval": len(success_eval_trajectories),
            "fail_eval": len(fail_eval_trajectories),
        },
        "runtime": {
            "threshold_init": (
                float(calibration_summary.threshold) if calibration_summary.threshold is not None else float("nan")
            ),
            **dict(calibration_summary.metadata),
        },
        "expert_metrics": {
            "prefixes": float(expert_total),
            "false_alarms": float(expert_false_alarm),
            "false_alarm_rate": float(expert_false_alarm / expert_total) if expert_total > 0 else float("nan"),
            "lambda_mean": float(np.mean(expert_lambda_values)) if expert_lambda_values else float("nan"),
        },
        "success_metrics": {
            "prefixes": float(success_total),
            "false_alarms": float(success_false_alarm),
            "false_alarm_rate": float(success_false_alarm / success_total) if success_total > 0 else float("nan"),
            "lambda_mean": float(np.mean(success_lambda_values)) if success_lambda_values else float("nan"),
        },
        "fail_metrics": fail_metrics,
        "lambda_stats": {
            "lambda_fail_mean": float(np.mean(fail_lambda_values)) if fail_lambda_values else float("nan"),
            "lambda_prefail_mean": float(np.mean(pre_fail_lambda_values)) if pre_fail_lambda_values else float("nan"),
            "lambda_gap": (
                float(np.mean(fail_lambda_values) - np.mean(pre_fail_lambda_values))
                if fail_lambda_values and pre_fail_lambda_values
                else float("nan")
            ),
        },
    }
    
    return summary, calibration_summary, {
        "expert": expert_results,
        "success": success_results,
        "fail": fail_results,
    }
