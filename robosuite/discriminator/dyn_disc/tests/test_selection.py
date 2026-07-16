"""NumPy-only tests for the dyn-disc sweep selection protocol."""

from __future__ import annotations

import json

import numpy as np
import pytest

from robosuite.discriminator.dyn_disc.selection import (
    compute_task_score,
    evaluate_health,
    select_best_bundle,
    select_best_healthy_per_config,
    select_top_configs,
    sha256_file,
    task_score_components,
    write_best_bundle_json,
)


TASKS = ("Cereal", "Round", "Square", "Milk", "Stack")


def _health(
    *, saturation: float = 0.01, abs_p99: float = 4.0, tau: float = 0.0
) -> dict:
    return {
        "passed": False,
        "reasons": ["stale serialized result"],
        "tau": tau,
        "pools": {
            name: {
                "effective": {
                    "all_finite": True,
                    "abs_p99": abs_p99,
                    "saturation_fraction": saturation,
                }
            }
            for name in ("train_positive", "unlabeled_failure", "success_calib")
        },
    }


def _benchmark(score: float, *, false_alarm: float | None = None) -> dict:
    value = float(score)
    benchmark = {
        "trajectory_level": {"auroc": value},
        "step_level": {
            "frame_auroc": value,
            "frame_f1": value,
            "event_precision": value,
            "event_recall": value,
        },
    }
    benchmark["success_level"] = (
        {"frame_specificity": value}
        if false_alarm is None
        else {"frame_false_alarm_rate": false_alarm}
    )
    return benchmark


def _record(
    task: str,
    config: str,
    epoch: int,
    score: float,
    *,
    saturation: float = 0.01,
    healthy_abs_p99: float = 4.0,
    normalization: str = "none",
    tau: float = 0.0,
) -> dict:
    return {
        "task": task,
        "config": config,
        "epoch": epoch,
        "checkpoint": f"/{task}/{config}/epoch_{epoch}.pth",
        "checkpoint_sha256": f"sha-{task}-{config}-{epoch}",
        "threshold_normalization": normalization,
        "benchmark": _benchmark(score),
        "health": _health(
            saturation=saturation, abs_p99=healthy_abs_p99, tau=tau
        ),
    }


def test_health_gate_recomputes_all_limits_and_tau() -> None:
    healthy = _record(
        "Cereal", "normalized", 1, 0.8, normalization="epoch_boundary", tau=0.1
    )
    result = evaluate_health(healthy)
    assert result["passed"] is True
    assert result["reasons"] == []
    assert result["max_observed_saturation_fraction"] == pytest.approx(0.01)

    unhealthy = _record(
        "Cereal",
        "normalized",
        1,
        0.8,
        saturation=0.05001,
        healthy_abs_p99=9.21001,
        normalization="epoch_boundary",
        tau=-0.10001,
    )
    unhealthy["health"]["pools"]["train_positive"]["effective"][
        "all_finite"
    ] = False
    result = evaluate_health(unhealthy)
    assert result["passed"] is False
    assert any("not all finite" in reason for reason in result["reasons"])
    assert any("abs_p99" in reason for reason in result["reasons"])
    assert any("saturation_fraction" in reason for reason in result["reasons"])
    assert any("abs(tau)" in reason for reason in result["reasons"])


def test_health_gate_fails_closed_on_missing_diagnostics() -> None:
    result = evaluate_health(
        {
            "threshold_normalization": "epoch_boundary",
            "health": {"pools": {}, "passed": True},
        }
    )
    assert result["passed"] is False
    assert len(result["reasons"]) == 4


def test_health_gate_accepts_runner_finite_fraction_layout() -> None:
    record = _record("Cereal", "cap", 1, 0.8)
    for pool in record["health"]["pools"].values():
        stats = pool.pop("effective")
        stats.pop("all_finite")
        stats["finite_fraction"] = 1.0
        pool.update(stats)
    assert evaluate_health(record)["passed"] is True

    record["health"]["pools"]["success_calib"]["finite_fraction"] = 0.999
    result = evaluate_health(record)
    assert result["passed"] is False
    assert any("not all finite" in reason for reason in result["reasons"])


def test_task_score_uses_event_f1_and_success_specificity() -> None:
    benchmark = {
        "trajectory_level": {"auroc": np.float64(0.9)},
        "step_level": {
            "frame_auroc": 0.8,
            "frame_f1": 0.7,
            "event_precision": 0.25,
            "event_recall": 1.0,
        },
        "success_level": {
            "frame_false_alarm_rate": 0.1,
            "trajectory_alarm_rate": 0.5,
        },
    }
    components = task_score_components(benchmark)
    assert components["event_f1"] == pytest.approx(0.4)
    assert components["success_specificity"] == pytest.approx(0.9)
    assert compute_task_score(benchmark) == pytest.approx(
        np.mean([0.9, 0.8, 0.7, 0.4, 0.9])
    )


def test_task_score_supports_legacy_success_report() -> None:
    benchmark = _benchmark(0.8)
    del benchmark["success_level"]
    assert compute_task_score(benchmark, {"frame_fpr": 0.2}) == pytest.approx(0.8)


def test_task_score_supports_additive_step_success_metrics() -> None:
    benchmark = _benchmark(0.8)
    del benchmark["success_level"]
    benchmark["step_level"]["success_specificity"] = 0.6
    assert task_score_components(benchmark)["success_specificity"] == pytest.approx(0.6)


def test_stage_one_selects_best_healthy_epoch_and_excludes_baseline() -> None:
    records = [
        _record("Cereal", "baseline", 1, 0.99, healthy_abs_p99=10.0),
        _record("Cereal", "norm", 1, 0.80, saturation=0.02),
        _record("Cereal", "norm", 2, 0.85, healthy_abs_p99=10.0),
        _record("Cereal", "cap", 1, 0.80, saturation=0.01),
        _record("Cereal", "cap", 2, 0.80, saturation=0.01),
    ]
    best = select_best_healthy_per_config(records)
    assert set(best) == {"norm", "cap"}
    assert best["norm"]["epoch"] == 1
    assert best["cap"]["epoch"] == 1
    assert [record["config"] for record in select_top_configs(records)] == [
        "cap",
        "norm",
    ]


def test_stage_one_returns_short_list_instead_of_relaxing_gate() -> None:
    records = [
        _record("Cereal", "only", 1, 0.8),
        _record("Cereal", "bad", 1, 0.9, healthy_abs_p99=10.0),
    ]
    selected = select_top_configs(records, top_k=2)
    assert len(selected) == 1
    assert selected[0]["config"] == "only"


def test_bundle_requires_same_config_epoch_and_all_tasks() -> None:
    records = [
        _record(task, "cap", 1 if task != "Stack" else 2, 0.9)
        for task in TASKS
    ]
    result = select_best_bundle(records, required_tasks=TASKS)
    assert result["status"] == "no_healthy_best"

    records = [_record(task, "cap", 1, 0.9) for task in TASKS]
    records[-1]["health"] = _health(abs_p99=10.0)
    result = select_best_bundle(records, required_tasks=TASKS)
    assert result["status"] == "no_healthy_best"


def test_bundle_global_score_and_deterministic_ties() -> None:
    records = []
    # Same global/macro/saturation: earlier epoch wins, then lexical config.
    for config, epoch in (("zeta", 1), ("alpha", 1), ("earlier", 1), ("later", 2)):
        records.extend(_record(task, config, epoch, 0.8) for task in TASKS)
    result = select_best_bundle(records, required_tasks=TASKS)
    assert result["status"] == "ok"
    assert result["config"] == "alpha"
    assert result["config_parameters"] == {"threshold_normalization": "none"}
    assert result["epoch"] == 1
    assert result["macro_task_score"] == pytest.approx(0.8)
    assert result["worst_task_score"] == pytest.approx(0.8)
    assert result["global_score"] == pytest.approx(0.8)
    assert list(result["tasks"]) == list(TASKS)


def test_bundle_prefers_global_then_macro_then_saturation() -> None:
    records = []
    for task, score in zip(TASKS, (0.9, 0.9, 0.9, 0.9, 0.5)):
        records.append(_record(task, "macro_high", 5, score, saturation=0.02))
    for task, score in zip(TASKS, (0.82, 0.82, 0.82, 0.82, 0.7)):
        records.append(_record(task, "worst_high", 5, score, saturation=0.01))
    result = select_best_bundle(records, required_tasks=TASKS)
    assert result["config"] == "worst_high"

    tied = []
    tied.extend(_record(task, "high_sat", 5, 0.8, saturation=0.02) for task in TASKS)
    tied.extend(_record(task, "low_sat", 10, 0.8, saturation=0.01) for task in TASKS)
    result = select_best_bundle(tied, required_tasks=TASKS)
    assert result["config"] == "low_sat"


def test_bundle_rejects_inconsistent_parameters_under_same_config_name() -> None:
    records = [_record(task, "cap", 1, 0.8) for task in TASKS]
    records[-1]["soft_cap_lambda"] = 0.01
    with pytest.raises(ValueError, match="inconsistent parameters"):
        select_best_bundle(records, required_tasks=TASKS)


def test_bundle_json_helper_is_strict_and_hashes_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "head.pth"
    checkpoint.write_bytes(b"checkpoint")
    records = [_record(task, "cap", 1, 0.8) for task in TASKS]
    records[0].pop("checkpoint_sha256")
    records[0]["checkpoint"] = checkpoint
    bundle = select_best_bundle(records, required_tasks=TASKS)
    assert bundle["tasks"][TASKS[0]]["checkpoint_sha256"] == sha256_file(checkpoint)

    path = write_best_bundle_json(bundle, tmp_path / "nested" / "best_bundle.json")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == bundle
    assert not (path.parent / f".{path.name}.tmp").exists()
