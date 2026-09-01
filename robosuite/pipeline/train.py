"""CLI for batch-online discriminator, VAST, and policy stages."""

from __future__ import annotations

import argparse
from pathlib import Path

from robosuite.pipeline.workflow import RunLayout, clone_run_for_retrain
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
    parser.add_argument("--branch-beta", type=float, default=None)
    parser.add_argument("--branch-k", type=float, default=None)
    parser.add_argument("--branch-eta", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    layout = RunLayout(Path(args.run_root))
    if args.round_index is not None:
        source = layout
        retrain_stage = "disc" if args.stage == "all" else args.stage
        layout = clone_run_for_retrain(
            source,
            round_index=args.round_index,
            stage=retrain_stage,
        )
        print(f"[batch-online] retrain source: {source.root}")
        print(f"[batch-online] retrain run: {layout.root}")
        print(
            f"[batch-online] retrain start: round={args.round_index:03d} "
            f"stage={retrain_stage}"
        )
    branch_weight_overrides = {
        key: value
        for key, value in {
            "beta": args.branch_beta,
            "k": args.branch_k,
            "eta": args.branch_eta,
        }.items()
        if value is not None
    }
    train_kwargs = {"requested_stage": args.stage}
    if branch_weight_overrides:
        train_kwargs["branch_weight_overrides"] = branch_weight_overrides
    train(layout, **train_kwargs)
    print(f"[batch-online] training stage(s) complete: {layout.root}")


if __name__ == "__main__":
    main()
