"""Summarize omega=0 positive-policy matrix evaluations without tensor loading."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


CONTRASTS = (
    ("C1", "C0", "hard_success_coupled"),
    ("C2", "C0", "independent_without_success"),
    ("C3", "C1", "independent_with_success"),
    ("C5", "C4", "success_within_filtered_bc"),
    ("C5", "C3", "filtered_bc_vs_soft_policy_rows"),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix-root",
        default="outputs/debug/offline_disc_positive_matrix",
    )
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def _completed_steps(summary: dict[str, Any]) -> int:
    match = re.search(r"step_(\d+)\.pt$", str(summary.get("checkpoint", "")))
    if match is None:
        raise ValueError(f"Cannot parse checkpoint step: {summary.get('checkpoint')}")
    return int(match.group(1))


def _mcnemar_exact(success_a: list[bool], success_b: list[bool]) -> dict[str, Any]:
    if len(success_a) != len(success_b):
        raise ValueError("Paired evaluation lists have different lengths.")
    a_only = sum(a and not b for a, b in zip(success_a, success_b, strict=True))
    b_only = sum(b and not a for a, b in zip(success_a, success_b, strict=True))
    discordant = a_only + b_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, k) for k in range(min(a_only, b_only) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "a_only": int(a_only),
        "b_only": int(b_only),
        "discordant": int(discordant),
        "mcnemar_exact_p": float(p_value),
    }


def _success_step_distribution(
    episode_results: list[dict[str, Any]],
) -> dict[str, float | int | None]:
    steps = [
        int(result["steps"])
        for result in episode_results
        if bool(result["success"])
    ]
    if not steps:
        return {
            "success_steps_count": 0,
            "success_steps_min": None,
            "success_steps_median": None,
            "success_steps_mean": None,
            "success_steps_max": None,
        }
    return {
        "success_steps_count": len(steps),
        "success_steps_min": min(steps),
        "success_steps_median": float(statistics.median(steps)),
        "success_steps_mean": float(statistics.fmean(steps)),
        "success_steps_max": max(steps),
    }


def main() -> None:
    args = _parse_args()
    matrix_root = Path(args.matrix_root).resolve()
    records: dict[tuple[str, int], dict[str, Any]] = {}
    for condition_dir in sorted(matrix_root.glob("C[0-5]")):
        for summary_path in sorted((condition_dir / "eval_matrix").glob("*/summary.json")):
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if float(summary["omega"]) != 0.0:
                continue
            steps = _completed_steps(summary)
            key = (condition_dir.name, steps)
            if key in records:
                raise RuntimeError(f"Duplicate evaluation for {key}: {summary_path}")
            records[key] = {
                "condition": condition_dir.name,
                "completed_steps": steps,
                "episodes": int(summary["episodes"]),
                "success_count": int(summary["success_count"]),
                "success_rate": float(summary["success_rate"]),
                "success_rate_ci95_low": float(summary["success_rate_ci95_low"]),
                "success_rate_ci95_high": float(summary["success_rate_ci95_high"]),
                "mean_return": float(summary["mean_return"]),
                "mean_steps": float(summary["mean_steps"]),
                "successes": [
                    bool(result["success"])
                    for result in summary["episode_results"]
                ],
                "episode_seeds": [
                    result["episode_seed"] for result in summary["episode_results"]
                ],
                **_success_step_distribution(summary["episode_results"]),
                "summary_path": str(summary_path),
            }

    comparisons: list[dict[str, Any]] = []
    for steps in (5000, 10000, 15000):
        for condition_a, condition_b, effect in CONTRASTS:
            a = records.get((condition_a, steps))
            b = records.get((condition_b, steps))
            if a is None or b is None or a["episodes"] != b["episodes"]:
                continue
            if a["episode_seeds"] != b["episode_seeds"]:
                raise RuntimeError(
                    "Paired comparison has mismatched episode seeds: "
                    f"{condition_a} vs {condition_b} at {steps} steps."
                )
            comparisons.append(
                {
                    "effect": effect,
                    "completed_steps": steps,
                    "condition_a": condition_a,
                    "condition_b": condition_b,
                    "success_rate_delta": a["success_rate"] - b["success_rate"],
                    **_mcnemar_exact(a["successes"], b["successes"]),
                }
            )

    serializable_records = []
    for record in sorted(records.values(), key=lambda item: (item["completed_steps"], item["condition"])):
        serializable_records.append(
            {
                key: value
                for key, value in record.items()
                if key not in {"successes", "episode_seeds"}
            }
        )
    report = {"records": serializable_records, "paired_comparisons": comparisons}
    out_path = Path(args.out).resolve() if args.out else matrix_root / "analysis.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    markdown_path = out_path.with_suffix(".md")
    lines = [
        "# Positive-policy matrix",
        "",
        "| Steps | Condition | Success | Mean return | Success steps (min/median/mean/max) | 95% Wilson CI |",
        "|---:|:---:|---:|---:|:---:|:---:|",
    ]
    for record in serializable_records:
        lines.append(
            f"| {record['completed_steps']} | {record['condition']} | "
            f"{record['success_count']}/{record['episodes']} "
            f"({record['success_rate']:.3f}) | {record['mean_return']:.2f} | "
            f"{record['success_steps_min']}/{record['success_steps_median']:.1f}/"
            f"{record['success_steps_mean']:.1f}/{record['success_steps_max']} | "
            f"[{record['success_rate_ci95_low']:.3f}, "
            f"{record['success_rate_ci95_high']:.3f}] |"
        )
    lines.extend(["", "## Paired contrasts", ""])
    for comparison in comparisons:
        lines.append(
            f"- {comparison['completed_steps']} {comparison['effect']}: "
            f"delta={comparison['success_rate_delta']:+.3f}, "
            f"a_only={comparison['a_only']}, b_only={comparison['b_only']}, "
            f"p={comparison['mcnemar_exact_p']:.6g}"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_path)
    print(markdown_path)


if __name__ == "__main__":
    main()
