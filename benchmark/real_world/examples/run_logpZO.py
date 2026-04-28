"""Run logpZO through the real-world Agilex benchmark API."""

from __future__ import annotations

import argparse

from benchmark.core import EvalConfig
from benchmark.real_world import FailureBenchmark
from robosuite.discriminator.logpZO.logpZO_benchmark import LogpZOBenchmarkDiscriminator


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fail-root", required=True)
    parser.add_argument("--success-root", required=True)
    parser.add_argument("--cache-root", type=str, default=None,
                        help="Optional raw array cache root built by benchmark.real_world.examples.build_cache.")
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--save-json", type=str, default=None)
    parser.add_argument("--max-fail-per-task", type=int, default=None)
    parser.add_argument("--max-success-per-task", type=int, default=None)

    parser.add_argument("--proprio-field", type=str, default="qpos")
    parser.add_argument("--proprio-start", type=int, default=7)
    parser.add_argument("--proprio-stop", type=int, default=14)
    parser.add_argument("--action-start", type=int, default=7)
    parser.add_argument("--action-stop", type=int, default=14)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--encoder-batch-size", type=int, default=64)
    parser.add_argument("--camera-name", type=str, default="cam_high")
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--scale-clamp", type=float, default=3.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--calib-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet-fit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    bench = FailureBenchmark(
        fail_labeled_root=args.fail_root,
        success_root=args.success_root,
        tasks=args.tasks,
        max_fail_per_task=args.max_fail_per_task,
        max_success_per_task=args.max_success_per_task,
        cache_root=args.cache_root,
        proprio_field=args.proprio_field,
        proprio_slice=slice(int(args.proprio_start), int(args.proprio_stop)),
        action_slice=slice(int(args.action_start), int(args.action_stop)),
    )
    trajs = bench.trajectories()
    n_fail = sum(1 for t in trajs if t.is_failure)
    print(
        f"[real_world][logpZO] discovered {len(trajs)} trajectories "
        f"(failure={n_fail}, success={len(trajs) - n_fail})"
    )

    discriminator = LogpZOBenchmarkDiscriminator(
        device=args.device,
        image_size=int(args.image_size),
        encoder_batch_size=int(args.encoder_batch_size),
        camera_name=str(args.camera_name),
        num_layers=int(args.num_layers),
        hidden_dim=int(args.hidden_dim),
        scale_clamp=float(args.scale_clamp),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        val_fraction=float(args.val_fraction),
        early_stop_patience=int(args.early_stop_patience),
        alpha=float(args.alpha),
        calib_fraction=float(args.calib_fraction),
        seed=int(args.seed),
        verbose_fit=not bool(args.quiet_fit),
    )
    try:
        discriminator.fit_on_benchmark(trajs)
        result = bench.evaluate(
            discriminator,
            EvalConfig(step_binarize_strategy="provided"),
        )
        print(result.summary())
        print(
            "[real_world][logpZO] calibration summary:",
            discriminator.calibration_summary(),
        )
        if args.save_json:
            result.save_json(args.save_json)
            print(f"[real_world][logpZO] wrote {args.save_json}")
    finally:
        discriminator.close()


if __name__ == "__main__":
    main()
