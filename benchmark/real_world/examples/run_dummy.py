"""Smoke-test the real-world (agilex) benchmark with a placeholder discriminator.

Exists so we can validate the loader / FailureBenchmark wiring end-to-end
without an LPB dynamics checkpoint trained on agilex. Two simple modes:

    score-mode = "random"       per-frame scores ~ U[0, 1].
    score-mode = "linear-time"  score = t / (T - 1); rises monotonically along
                                the trajectory (mimics a "things go wrong as
                                time passes" prior).

With ``random``, trajectory AUROC should hover near 0.5; with ``linear-time``
it should be exactly 0.5 (max(score) = 1 for every trajectory, ties broken
arbitrarily) — the value is uninformative on purpose; what matters is that
the pipeline runs end-to-end and frame-level metrics populate.

Example:
    python -m benchmark.real_world.examples.run_dummy \
        --fail-root data/agilex/failure_annotations/out_by_task \
        --success-root data/agilex \
        --tasks candy_in_plate \
        --save-json /tmp/agilex_dummy.json \
        --max-fail-per-task 5 --max-success-per-task 5
"""

from __future__ import annotations

import argparse

import numpy as np

from benchmark.core import DiscriminatorOutput, EvalConfig
from benchmark.real_world import FailureBenchmark


class DummyDiscriminator:
    name = "dummy"

    def __init__(self, mode: str = "random", seed: int = 0) -> None:
        self.mode = mode
        self._rng = np.random.default_rng(int(seed))

    def score_trajectory(self, trajectory) -> DiscriminatorOutput:
        T = int(trajectory.num_frames)
        if self.mode == "linear-time":
            scores = np.linspace(0.0, 1.0, num=T, dtype=np.float64)
        elif self.mode == "random":
            scores = self._rng.random(size=T).astype(np.float64)
        else:
            raise ValueError(f"unknown dummy mode {self.mode!r}")
        return DiscriminatorOutput(step_scores=scores)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fail-root", required=True,
                        help="data/agilex/failure_annotations/out_by_task")
    parser.add_argument("--success-root", required=True,
                        help="data/agilex (parent of <task>/success_rollout/)")
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--save-json", type=str, default=None)
    parser.add_argument("--max-fail-per-task", type=int, default=None)
    parser.add_argument("--max-success-per-task", type=int, default=None)

    parser.add_argument("--proprio-field", type=str, default="qpos")
    parser.add_argument("--proprio-start", type=int, default=7)
    parser.add_argument("--proprio-stop", type=int, default=14)
    parser.add_argument("--action-start", type=int, default=7)
    parser.add_argument("--action-stop", type=int, default=14)
    parser.add_argument("--camera-name", type=str, default="cam_high",
                        help="Recorded in metadata only; the dummy discriminator "
                             "does not actually load images.")

    parser.add_argument("--score-mode", type=str, default="random",
                        choices=["random", "linear-time"])
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    bench = FailureBenchmark(
        fail_labeled_root=args.fail_root,
        success_root=args.success_root,
        tasks=args.tasks,
        max_fail_per_task=args.max_fail_per_task,
        max_success_per_task=args.max_success_per_task,
        proprio_field=args.proprio_field,
        proprio_slice=slice(int(args.proprio_start), int(args.proprio_stop)),
        action_slice=slice(int(args.action_start), int(args.action_stop)),
    )
    trajs = bench.trajectories()
    n_fail = sum(1 for t in trajs if t.is_failure)
    n_succ = len(trajs) - n_fail
    print(f"[real_world] discovered {len(trajs)} trajectories "
          f"(failure={n_fail}, success={n_succ})")

    discriminator = DummyDiscriminator(mode=args.score_mode, seed=int(args.seed))
    result = bench.evaluate(discriminator, EvalConfig())
    print(result.summary())
    if args.save_json:
        result.save_json(args.save_json)
        print(f"[real_world] wrote {args.save_json}")


if __name__ == "__main__":
    main()
