"""CLI for batch-online discriminator, VAST, and policy stages."""

from __future__ import annotations

import argparse
from pathlib import Path

from robosuite.pipeline.workflow import RunLayout
from robosuite.pipeline.workflow.runner import train


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root")
    parser.add_argument("stage", nargs="?", default="all", choices=("all", "disc", "vast", "policy"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    layout = RunLayout(Path(args.run_root))
    train(layout, requested_stage=args.stage)
    print(f"[batch-online] training stage(s) complete: {layout.root}")


if __name__ == "__main__":
    main()
