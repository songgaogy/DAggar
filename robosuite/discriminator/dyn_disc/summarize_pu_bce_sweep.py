"""Summarize PU-BCE trick trials and select approved checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robosuite.discriminator.dyn_disc.selection import (
    DEFAULT_REQUIRED_TASKS,
    select_best_bundle,
    select_top_configs,
    write_best_bundle_json,
)


def _load_records(root: Path) -> list[dict]:
    records: list[dict] = []
    for path in sorted(root.rglob("trial_summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for record in payload.get("records", []):
            if isinstance(record, dict):
                records.append(record)
    if not records:
        raise RuntimeError(f"No trial_summary.json records found under {root}")
    return records


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=["stage1", "final"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="PickPlaceCereal")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = args.root.expanduser().resolve()
    records = _load_records(root)
    if args.stage == "stage1":
        task_records = [record for record in records if record.get("task") == args.task]
        top = select_top_configs(task_records, top_k=2)
        payload = {
            "schema": "pu_bce_stage1_selection_v1",
            "status": "ok" if len(top) == 2 else "insufficient_healthy_configs",
            "task": str(args.task),
            "top_configs": [
                {
                    "config": record["config"],
                    "best_cereal_epoch": record["epoch"],
                    "task_score": record["task_score"],
                    "health_gate": record["health_gate"],
                }
                for record in top
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        if len(top) != 2:
            raise SystemExit(2)
        return

    bundle = select_best_bundle(records, required_tasks=DEFAULT_REQUIRED_TASKS)
    write_best_bundle_json(bundle, args.output)
    print(json.dumps(bundle, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
