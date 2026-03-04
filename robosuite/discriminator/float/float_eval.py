from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .float_core import FLOATComputer, OnlineDetector, ThresholdCalibrator, Trajectory
from .float_data import fail_prefix_labels


@dataclass
class PrefixPrediction:
    trajectory_id: int
    t: int
    lambda_value: float
    threshold: float
    pred: int
    label: int


def _safe_div(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return float(a / b)


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


def evaluate_fail_rollouts(
    fail_rollouts: list[Trajectory],
    float_computer: FLOATComputer,
    calibrator: ThresholdCalibrator,
    delta: float,
    delta_step: float,
    fail_tail_ratio: float,
    stride: int,
    adaptive_delta: bool,
) -> tuple[dict[str, float], list[PrefixPrediction], float]:
    detector = OnlineDetector(
        float_computer=float_computer,
        calibrator=calibrator,
        delta=delta,
        delta_step=delta_step,
        stride=stride,
        greater_is_failure=True,
    )

    preds: list[PrefixPrediction] = []

    for traj_id, traj in enumerate(fail_rollouts):
        detector.reset()
        labels = fail_prefix_labels(length=traj.obs.shape[0], fail_tail_ratio=fail_tail_ratio)

        for t in range(traj.obs.shape[0]):
            step_out = detector.step(traj.obs[t])
            pred = int(bool(step_out["is_failure"]))
            label = int(labels[t])
            preds.append(
                PrefixPrediction(
                    trajectory_id=traj_id,
                    t=t + 1,
                    lambda_value=float(step_out["lambda"]),
                    threshold=float(step_out["threshold"]),
                    pred=pred,
                    label=label,
                )
            )

            if adaptive_delta:
                detector.feedback(was_failure=bool(label), detector_raised=bool(pred))

    y_true = np.asarray([p.label for p in preds], dtype=np.int64)
    y_pred = np.asarray([p.pred for p in preds], dtype=np.int64)
    metrics = classification_metrics(y_true=y_true, y_pred=y_pred)
    metrics["delta_final"] = float(detector.delta)
    metrics["threshold_final"] = float(calibrator.threshold) if calibrator.threshold is not None else float("nan")

    return metrics, preds, float(detector.delta)


def evaluate_success_rollouts(
    success_rollouts: list[Trajectory],
    float_computer: FLOATComputer,
    threshold: float,
    stride: int,
) -> dict[str, float]:
    detector = OnlineDetector(
        float_computer=float_computer,
        calibrator=ThresholdCalibrator(),
        delta=10.0,
        delta_step=1.0,
        stride=stride,
        greater_is_failure=True,
    )
    detector.calibrator.threshold = float(threshold)

    total = 0
    false_alarms = 0
    lambdas = []

    for traj in success_rollouts:
        detector.reset()
        for t in range(traj.obs.shape[0]):
            out = detector.step(traj.obs[t])
            total += 1
            pred = int(bool(out["is_failure"]))
            false_alarms += pred
            lambdas.append(float(out["lambda"]))

    fa_rate = float(false_alarms / total) if total > 0 else 0.0
    return {
        "success_prefixes": float(total),
        "false_alarms": float(false_alarms),
        "false_alarm_rate": fa_rate,
        "lambda_success_mean": float(np.mean(lambdas)) if lambdas else float("nan"),
    }


def summarize_lambda_separation(preds: list[PrefixPrediction]) -> dict[str, float]:
    fail_values = [p.lambda_value for p in preds if p.label == 1]
    success_values = [p.lambda_value for p in preds if p.label == 0]
    return {
        "lambda_fail_mean": float(np.mean(fail_values)) if fail_values else float("nan"),
        "lambda_success_mean": float(np.mean(success_values)) if success_values else float("nan"),
        "lambda_gap": (
            float(np.mean(fail_values) - np.mean(success_values))
            if fail_values and success_values
            else float("nan")
        ),
    }


def build_float_and_calibrate(
    expert_rollouts: list[Trajectory],
    success_rollouts_for_calibration: list[Trajectory],
    sinkhorn_reg: float,
    max_iter: int,
    tol: float,
    delta: float,
    pad_to: Optional[int],
    use_similarity_cost: bool,
):
    from .float_core import FLOATComputer, IdentityEncoder, ThresholdCalibrator

    float_computer = FLOATComputer(
        experts=expert_rollouts,
        encoder=IdentityEncoder(),
        sinkhorn_reg=sinkhorn_reg,
        max_iter=max_iter,
        tol=tol,
        pad_to=pad_to,
        use_similarity_cost=use_similarity_cost,
    )
    calibrator = ThresholdCalibrator()
    threshold = calibrator.fit(
        success_rollouts=success_rollouts_for_calibration,
        float_computer=float_computer,
        delta=delta,
    )
    return float_computer, calibrator, threshold
