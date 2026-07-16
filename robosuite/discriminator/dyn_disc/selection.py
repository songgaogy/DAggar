"""Deterministic model selection for the dyn-disc logit-health sweep.

This module is intentionally tensor-free.  It consumes the JSON-compatible
records produced by training/evaluation and implements the approved health and
ranking protocol without loading a model or benchmark dataset.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 1
REQUIRED_HEALTH_POOLS = (
    "train_positive",
    "unlabeled_failure",
    "success_calib",
)
DEFAULT_REQUIRED_TASKS = (
    "PickPlaceCereal",
    "NutAssemblyRound",
    "NutAssemblySquare",
    "PickPlaceMilk",
    "Stack",
)
CONFIG_PARAMETER_FIELDS = (
    "threshold_normalization",
    "soft_cap_c",
    "soft_cap_lambda",
    "soft_cap_temperature",
)

_POOL_ALIASES = {
    "train_positive": ("train_positive", "train_p", "positive_train"),
    "unlabeled_failure": (
        "unlabeled_failure",
        "whole_failure_u",
        "failure_unlabeled",
        "unlabeled_train",
    ),
    "success_calib": ("success_calib", "calib_success", "positive_calib"),
}


def _as_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}.")
    return value


def _first(mapping: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    raise KeyError(f"Missing required field; expected one of {tuple(names)!r}.")


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be numeric, not boolean.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric, got {value!r}.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result!r}.")
    return result


def _unit_metric(value: Any, name: str) -> float:
    result = _finite_float(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {result!r}.")
    return result


def _normalization_mode(record: Mapping[str, Any]) -> str:
    for owner in (record, record.get("training_config", {})):
        if isinstance(owner, Mapping) and "threshold_normalization" in owner:
            return str(owner["threshold_normalization"])
    return "none"


def _health_payload(record_or_health: Mapping[str, Any]) -> Mapping[str, Any]:
    value = record_or_health.get("health", record_or_health)
    return _as_mapping(value, "health")


def _pool_stats(
    health: Mapping[str, Any], canonical_name: str
) -> Mapping[str, Any]:
    pools = _as_mapping(health.get("pools", health), "health.pools")
    pool = None
    for alias in _POOL_ALIASES[canonical_name]:
        if alias in pools:
            pool = pools[alias]
            break
    if pool is None:
        raise KeyError(f"Missing health pool {canonical_name!r}.")
    pool = _as_mapping(pool, f"health.pools.{canonical_name}")
    effective = pool.get("effective", pool.get("effective_logits", pool))
    return _as_mapping(effective, f"health.pools.{canonical_name}.effective")


def _health_tau(health: Mapping[str, Any]) -> float:
    for name in ("tau", "final_tau", "threshold_tau", "calibration_tau"):
        if name in health:
            return _finite_float(health[name], f"health.{name}")
    calibration = health.get("calibration")
    if isinstance(calibration, Mapping) and "tau" in calibration:
        return _finite_float(calibration["tau"], "health.calibration.tau")
    raise KeyError("Missing normalization health field 'tau'.")


def _pool_all_finite(stats: Mapping[str, Any], pool_name: str) -> bool:
    if "finite_fraction" in stats:
        fraction = _finite_float(
            stats["finite_fraction"], f"health.{pool_name}.finite_fraction"
        )
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(
                f"health.{pool_name}.finite_fraction must be in [0, 1]."
            )
        return fraction == 1.0
    value = _first(stats, ("all_finite", "finite", "is_finite"))
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(
            f"health.{pool_name}.all_finite must be boolean, got {value!r}."
        )
    return bool(value)


def evaluate_health(
    record_or_health: Mapping[str, Any],
    *,
    threshold_normalization: str | None = None,
    abs_logit_limit: float = 9.21,
    max_saturation_fraction: float = 0.05,
    max_normalized_abs_tau: float = 0.1,
) -> dict[str, Any]:
    """Recompute the approved checkpoint health gate.

    Missing or malformed diagnostics fail the gate and are reported in
    ``reasons``.  The function does not trust a serialized ``health.passed``
    value, which prevents stale thresholds from influencing model selection.
    """

    record_or_health = _as_mapping(record_or_health, "record_or_health")
    limit = _finite_float(abs_logit_limit, "abs_logit_limit")
    max_saturation = _finite_float(
        max_saturation_fraction, "max_saturation_fraction"
    )
    max_tau = _finite_float(max_normalized_abs_tau, "max_normalized_abs_tau")
    if limit <= 0.0:
        raise ValueError("abs_logit_limit must be positive.")
    if not 0.0 <= max_saturation <= 1.0:
        raise ValueError("max_saturation_fraction must be in [0, 1].")
    if max_tau < 0.0:
        raise ValueError("max_normalized_abs_tau must be non-negative.")

    health = _health_payload(record_or_health)
    mode = (
        _normalization_mode(record_or_health)
        if threshold_normalization is None
        else str(threshold_normalization)
    )
    reasons: list[str] = []
    normalized_pools: dict[str, dict[str, Any]] = {}

    for pool_name in REQUIRED_HEALTH_POOLS:
        try:
            stats = _pool_stats(health, pool_name)
            all_finite = _pool_all_finite(stats, pool_name)
            abs_p99 = _finite_float(
                _first(stats, ("abs_p99", "p99_abs", "effective_abs_p99")),
                f"health.{pool_name}.abs_p99",
            )
            saturation = _finite_float(
                _first(
                    stats,
                    (
                        "saturation_fraction",
                        "saturated_fraction",
                        "fraction_above_limit",
                        "abs_gt_9_21_fraction",
                    ),
                ),
                f"health.{pool_name}.saturation_fraction",
            )
            if not 0.0 <= saturation <= 1.0:
                raise ValueError(
                    f"health.{pool_name}.saturation_fraction must be in [0, 1]."
                )
            normalized_pools[pool_name] = {
                "all_finite": all_finite,
                "abs_p99": abs_p99,
                "saturation_fraction": saturation,
            }
            if not all_finite:
                reasons.append(f"{pool_name}: effective logits are not all finite")
            if abs_p99 > limit:
                reasons.append(
                    f"{pool_name}: abs_p99={abs_p99:.12g} exceeds {limit:.12g}"
                )
            if saturation > max_saturation:
                reasons.append(
                    f"{pool_name}: saturation_fraction={saturation:.12g} "
                    f"exceeds {max_saturation:.12g}"
                )
        except (KeyError, TypeError, ValueError) as exc:
            reasons.append(f"{pool_name}: {exc}")

    tau: float | None = None
    if mode == "epoch_boundary":
        try:
            tau = _health_tau(health)
            if abs(tau) > max_tau:
                reasons.append(
                    f"normalization: abs(tau)={abs(tau):.12g} exceeds {max_tau:.12g}"
                )
        except (KeyError, TypeError, ValueError) as exc:
            reasons.append(f"normalization: {exc}")
    elif mode != "none":
        reasons.append(f"normalization: unsupported mode {mode!r}")

    saturation_values = [
        float(stats["saturation_fraction"]) for stats in normalized_pools.values()
    ]
    return {
        "passed": not reasons,
        "reasons": reasons,
        "threshold_normalization": mode,
        "abs_logit_limit": limit,
        "max_saturation_fraction_limit": max_saturation,
        "max_normalized_abs_tau": max_tau,
        "tau": tau,
        "max_observed_saturation_fraction": (
            max(saturation_values) if saturation_values else None
        ),
        "pools": normalized_pools,
    }


def task_score_components(
    benchmark: Mapping[str, Any],
    success_false_alarm: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Extract the five approved task-score components from benchmark JSON."""

    benchmark = _as_mapping(benchmark, "benchmark")
    trajectory = _as_mapping(benchmark.get("trajectory_level"), "trajectory_level")
    step = _as_mapping(benchmark.get("step_level"), "step_level")
    trajectory_auroc = _unit_metric(trajectory.get("auroc"), "trajectory AUROC")
    frame_auroc = _unit_metric(step.get("frame_auroc"), "frame AUROC")
    frame_f1 = _unit_metric(step.get("frame_f1"), "frame F1")
    event_precision = _unit_metric(step.get("event_precision"), "event precision")
    event_recall = _unit_metric(step.get("event_recall"), "event recall")
    denominator = event_precision + event_recall
    event_f1 = (
        0.0
        if denominator == 0.0
        else 2.0 * event_precision * event_recall / denominator
    )

    success = benchmark.get("success_level")
    if not isinstance(success, Mapping) and any(
        name in step
        for name in (
            "success_specificity",
            "success_frame_false_alarm_rate",
            "success_frame_fpr",
        )
    ):
        success = step
    if not isinstance(success, Mapping):
        success = success_false_alarm
    success = _as_mapping(success, "benchmark.success_level")
    if "success_specificity" in success:
        specificity = _unit_metric(
            success["success_specificity"], "success frame specificity"
        )
    elif "frame_specificity" in success:
        specificity = _unit_metric(
            success["frame_specificity"], "success frame specificity"
        )
    else:
        false_alarm = _unit_metric(
            _first(
                success,
                (
                    "success_frame_false_alarm_rate",
                    "success_frame_fpr",
                    "frame_false_alarm_rate",
                    "frame_fpr",
                    "false_alarm_rate",
                ),
            ),
            "success frame false-alarm rate",
        )
        specificity = 1.0 - false_alarm

    return {
        "trajectory_auroc": trajectory_auroc,
        "frame_auroc": frame_auroc,
        "frame_f1": frame_f1,
        "event_f1": event_f1,
        "success_specificity": specificity,
    }


def compute_task_score(
    benchmark: Mapping[str, Any],
    success_false_alarm: Mapping[str, Any] | None = None,
) -> float:
    """Compute the approved unweighted mean of five evaluation metrics."""

    components = task_score_components(benchmark, success_false_alarm)
    return float(np.mean(np.asarray(tuple(components.values()), dtype=np.float64)))


def _record_identity(record: Mapping[str, Any]) -> tuple[str, str, int]:
    try:
        task = str(record["task"])
        config = str(record["config"])
        epoch = int(record["epoch"])
    except KeyError as exc:
        raise KeyError(f"Checkpoint record is missing {exc.args[0]!r}.") from exc
    if not task or not config:
        raise ValueError("Checkpoint record task and config must be non-empty.")
    if epoch <= 0:
        raise ValueError(f"Checkpoint epoch must be positive, got {epoch}.")
    return task, config, epoch


def _prepare_record(record: Mapping[str, Any]) -> dict[str, Any]:
    record = _as_mapping(record, "checkpoint record")
    task, config, epoch = _record_identity(record)
    benchmark = _as_mapping(record.get("benchmark"), "record.benchmark")
    success_false_alarm = record.get("success_false_alarm")
    if success_false_alarm is not None:
        success_false_alarm = _as_mapping(
            success_false_alarm, "record.success_false_alarm"
        )
    components = task_score_components(benchmark, success_false_alarm)
    health = evaluate_health(record)
    prepared = copy.deepcopy(dict(record))
    prepared.update(
        {
            "task": task,
            "config": config,
            "epoch": epoch,
            "health_gate": health,
            "task_score_components": components,
            "task_score": float(np.mean(tuple(components.values()))),
        }
    )
    return prepared


def _checkpoint_rank(record: Mapping[str, Any]) -> tuple[float, float, int, str]:
    saturation = record["health_gate"]["max_observed_saturation_fraction"]
    saturation = float("inf") if saturation is None else float(saturation)
    return (
        -float(record["task_score"]),
        saturation,
        int(record["epoch"]),
        str(record["config"]),
    )


def select_best_healthy_per_config(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Return each config's best healthy checkpoint for a single task."""

    prepared = [_prepare_record(record) for record in records]
    tasks = {record["task"] for record in prepared}
    if len(tasks) > 1:
        raise ValueError(
            "Stage-one selection expects one task, got "
            + ", ".join(sorted(tasks))
            + "."
        )
    seen: set[tuple[str, str, int]] = set()
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in prepared:
        identity = _record_identity(record)
        if identity in seen:
            raise ValueError(f"Duplicate checkpoint record {identity!r}.")
        seen.add(identity)
        if record["health_gate"]["passed"]:
            grouped.setdefault(record["config"], []).append(record)
    return {
        config: min(candidates, key=_checkpoint_rank)
        for config, candidates in sorted(grouped.items())
    }


def select_top_configs(
    records: Iterable[Mapping[str, Any]],
    *,
    top_k: int = 2,
    baseline_configs: Sequence[str] = ("baseline",),
) -> list[dict[str, Any]]:
    """Select distinct non-baseline configs from their best healthy epochs.

    A shorter list means the approved expansion condition was not met; callers
    should stop rather than relax the gate.
    """

    if int(top_k) != top_k or int(top_k) <= 0:
        raise ValueError("top_k must be a positive integer.")
    best = select_best_healthy_per_config(records)
    excluded = {str(name) for name in baseline_configs}
    candidates = [
        record for config, record in best.items() if config not in excluded
    ]
    candidates.sort(key=_checkpoint_rank)
    return candidates[: int(top_k)]


def sha256_file(path: str | Path) -> str:
    """Return a checkpoint's SHA-256 using bounded memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_sha256(record: Mapping[str, Any]) -> str:
    for name in ("checkpoint_sha256", "checkpoint_hash", "sha256"):
        value = record.get(name)
        if value:
            return str(value)
    checkpoint = record.get("checkpoint")
    if checkpoint is None:
        raise KeyError("Selected record has no checkpoint path or checkpoint SHA-256.")
    path = Path(str(checkpoint)).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"Cannot hash selected checkpoint because it does not exist: {path}"
        )
    return sha256_file(path)


def _config_parameters(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: _json_safe(record[name])
        for name in CONFIG_PARAMETER_FIELDS
        if name in record
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Selection output contains a non-finite float.")
    return value


def select_best_bundle(
    records: Iterable[Mapping[str, Any]],
    *,
    required_tasks: Sequence[str] = DEFAULT_REQUIRED_TASKS,
    baseline_configs: Sequence[str] = ("baseline",),
) -> dict[str, Any]:
    """Select the best same-config, same-epoch healthy five-task bundle."""

    required = tuple(str(task) for task in required_tasks)
    if not required or any(not task for task in required):
        raise ValueError("required_tasks must contain non-empty task names.")
    if len(set(required)) != len(required):
        raise ValueError("required_tasks contains duplicates.")
    required_set = set(required)
    excluded = {str(name) for name in baseline_configs}

    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for raw_record in records:
        record = _prepare_record(raw_record)
        if record["task"] not in required_set or record["config"] in excluded:
            continue
        key = (record["config"], record["epoch"])
        task_records = grouped.setdefault(key, {})
        if record["task"] in task_records:
            raise ValueError(
                "Duplicate checkpoint record for "
                f"config={key[0]!r}, epoch={key[1]}, task={record['task']!r}."
            )
        task_records[record["task"]] = record

    candidates: list[dict[str, Any]] = []
    for (config, epoch), task_records in grouped.items():
        if set(task_records) != required_set:
            continue
        ordered_records = [task_records[task] for task in required]
        if not all(record["health_gate"]["passed"] for record in ordered_records):
            continue
        config_parameters = _config_parameters(ordered_records[0])
        if any(
            _config_parameters(record) != config_parameters
            for record in ordered_records[1:]
        ):
            raise ValueError(
                f"Config {config!r} epoch {epoch} has inconsistent parameters "
                "across tasks."
            )
        scores = [float(record["task_score"]) for record in ordered_records]
        saturations = [
            float(record["health_gate"]["max_observed_saturation_fraction"])
            for record in ordered_records
        ]
        macro = float(np.mean(np.asarray(scores, dtype=np.float64)))
        worst = float(min(scores))
        candidates.append(
            {
                "config": config,
                "epoch": epoch,
                "global_score": 0.5 * macro + 0.5 * worst,
                "macro_task_score": macro,
                "worst_task_score": worst,
                "max_saturation_fraction": max(saturations),
                "config_parameters": config_parameters,
                "task_records": task_records,
            }
        )

    if not candidates:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "no_healthy_best",
            "reason": (
                "No non-baseline config and epoch is present and healthy on all "
                f"required tasks: {list(required)!r}."
            ),
            "required_tasks": list(required),
        }

    candidates.sort(
        key=lambda candidate: (
            -candidate["global_score"],
            -candidate["macro_task_score"],
            candidate["max_saturation_fraction"],
            candidate["epoch"],
            candidate["config"],
        )
    )
    best = candidates[0]
    tasks: dict[str, Any] = {}
    for task in required:
        record = best["task_records"][task]
        tasks[task] = {
            "checkpoint": str(record["checkpoint"]),
            "checkpoint_sha256": _checkpoint_sha256(record),
            "task_score": record["task_score"],
            "task_score_components": record["task_score_components"],
            "health": record["health"],
            "health_gate": record["health_gate"],
            "benchmark": record["benchmark"],
        }
        if "success_false_alarm" in record:
            tasks[task]["success_false_alarm"] = record["success_false_alarm"]

    return _json_safe(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "ok",
            "selection_protocol": {
                "task_score": (
                    "mean(trajectory_auroc, frame_auroc, frame_f1, "
                    "event_f1, success_specificity)"
                ),
                "global_score": "0.5 * macro_task_score + 0.5 * worst_task_score",
                "tie_break": [
                    "higher_macro_task_score",
                    "lower_max_saturation_fraction",
                    "earlier_epoch",
                    "lexicographic_config",
                ],
            },
            "required_tasks": list(required),
            "config": best["config"],
            "config_parameters": best["config_parameters"],
            "epoch": best["epoch"],
            "global_score": best["global_score"],
            "macro_task_score": best["macro_task_score"],
            "worst_task_score": best["worst_task_score"],
            "max_saturation_fraction": best["max_saturation_fraction"],
            "tasks": tasks,
        }
    )


def write_best_bundle_json(bundle: Mapping[str, Any], path: str | Path) -> Path:
    """Atomically write a selected bundle as strict, stable JSON."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    payload = _json_safe(_as_mapping(bundle, "bundle"))
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(destination)
    return destination


__all__ = [
    "CONFIG_PARAMETER_FIELDS",
    "DEFAULT_REQUIRED_TASKS",
    "REQUIRED_HEALTH_POOLS",
    "compute_task_score",
    "evaluate_health",
    "select_best_bundle",
    "select_best_healthy_per_config",
    "select_top_configs",
    "sha256_file",
    "task_score_components",
    "write_best_bundle_json",
]
