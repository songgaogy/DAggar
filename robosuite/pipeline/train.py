"""CLI for batch-online discriminator, VAST, and policy stages."""

from __future__ import annotations

import argparse
from pathlib import Path

from robosuite.pipeline.workflow import RunLayout, rollback_run_for_retrain
from robosuite.pipeline.workflow.runner import train


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("round index must be non-negative")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root")
    parser.add_argument("stage", nargs="?", default="all", choices=("all", "disc", "vast", "policy"))
    parser.add_argument("--round-index", type=_non_negative_int, default=None)
    return parser.parse_args()


def _confirm_in_place_retrain(
    layout: RunLayout,
    *,
    round_index: int,
    stage: str,
) -> bool:
    print("[batch-online] WARNING: retrain will permanently roll back this run in place.")
    print(f"[batch-online] run: {layout.root}")
    print(f"[batch-online] rollback target: round={round_index:03d} stage={stage}")
    try:
        response = input("[batch-online] Press ENTER to continue; type anything to cancel: ")
    except EOFError:
        print("[batch-online] retrain cancelled: no interactive confirmation received.")
        return False
    if response:
        print("[batch-online] retrain cancelled.")
        return False
    return True


def main() -> None:
    args = _parse_args()
    layout = RunLayout(Path(args.run_root))
    if args.round_index is not None:
        retrain_stage = "disc" if args.stage == "all" else args.stage
        if not _confirm_in_place_retrain(
            layout,
            round_index=args.round_index,
            stage=retrain_stage,
        ):
            return
        rollback_run_for_retrain(
            layout,
            round_index=args.round_index,
            stage=retrain_stage,
        )
        print(
            f"[batch-online] in-place retrain start: run={layout.root} "
            f"round={args.round_index:03d} stage={retrain_stage}"
        )
    train(layout, requested_stage=args.stage)
    print(f"[batch-online] training stage(s) complete: {layout.root}")


if __name__ == "__main__":
    main()
