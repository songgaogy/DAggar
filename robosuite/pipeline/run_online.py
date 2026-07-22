"""CLI for one batch-online collection round."""

from __future__ import annotations

import argparse
from pathlib import Path

from robosuite.pipeline.workflow import RunLayout
from robosuite.pipeline.workflow.runner import (
    initialize_run,
    load_run_config,
    run_online,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="'new' or an existing run root")
    parser.add_argument("run_name", nargs="?", default="run")
    parser.add_argument("--task", required=True, help="Hydra task config name")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.run == "new":
        layout = initialize_run(run_name=args.run_name, task=args.task)
        print(f"[batch-online] created run: {layout.root}")
    else:
        if args.run_name != "run":
            raise ValueError("run_name is accepted only with the 'new' command.")
        layout = RunLayout(Path(args.run))
        configured_task = str(load_run_config(layout).task.name)
        if configured_task != args.task:
            raise ValueError(
                f"Requested task {args.task!r} does not match run task "
                f"{configured_task!r}."
            )
    episodes = run_online(layout)
    print(f"[batch-online] collection complete: {episodes}")
    print(f"[batch-online] run root: {layout.root}")


if __name__ == "__main__":
    main()
