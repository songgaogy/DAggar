"""Run LPB v2 KNN through the real-world Agilex benchmark API."""

from __future__ import annotations

import argparse

from benchmark.core import EvalConfig
from benchmark.real_world import FailureBenchmark
from robosuite.discriminator.lpb_v2.benchmark import LPBV2BenchmarkDiscriminator


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-ckpt", required=True)
    parser.add_argument("--fail-root", required=True)
    parser.add_argument("--success-root", required=True)
    parser.add_argument(
        "--cache-root",
        type=str,
        default=None,
        help="Optional Agilex raw-array cache built by benchmark.real_world.examples.build_cache.",
    )
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
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument(
        "--proprio-indices",
        type=int,
        nargs="*",
        default=None,
        help="Optional state indices after benchmark proprio slicing.",
    )
    parser.add_argument(
        "--camera-to-view",
        type=str,
        default=None,
        help="Comma-separated camera:view pairs, e.g. cam_high:agentview.",
    )

    parser.add_argument("--visual-weight", type=float, default=1.0)
    parser.add_argument("--proprio-weight", type=float, default=2.0)
    parser.add_argument("--delta", type=float, default=10.0)
    parser.add_argument("--knn-chunk-size", type=int, default=2048)
    parser.add_argument("--calib-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet-fit", action="store_true")
    return parser.parse_args()


def _parse_camera_to_view(value: str | None):
    if not value:
        return None
    out = {}
    for chunk in value.split(","):
        cam, view = chunk.split(":", 1)
        out[cam.strip()] = view.strip()
    return out


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
    n_fail = sum(1 for t in trajs if bool(t.is_failure))
    n_succ = len(trajs) - n_fail
    tasks = sorted({str(t.task_name) for t in trajs})
    print(
        f"[real_world][lpb_v2] discovered {len(trajs)} trajectories "
        f"(failure={n_fail}, success={n_succ}) tasks={tasks}",
        flush=True,
    )

    discriminator = LPBV2BenchmarkDiscriminator(
        model_ckpt=str(args.model_ckpt),
        device=str(args.device),
        encode_batch_size=int(args.encode_batch_size),
        proprio_indices=(list(args.proprio_indices) if args.proprio_indices else None),
        camera_to_view=_parse_camera_to_view(args.camera_to_view),
        visual_weight=float(args.visual_weight),
        proprio_weight=float(args.proprio_weight),
        delta=float(args.delta),
        knn_chunk_size=int(args.knn_chunk_size),
        calib_fraction=float(args.calib_fraction),
        seed=int(args.seed),
        verbose_fit=not bool(args.quiet_fit),
    )
    try:
        print("[real_world][lpb_v2] fitting success KNN banks...", flush=True)
        discriminator.fit_on_benchmark(trajs)
        result = bench.evaluate(
            discriminator,
            EvalConfig(step_binarize_strategy="provided"),
        )
        print(result.summary())
        print("[real_world][lpb_v2] calibration summary:", discriminator.calibration_summary())
        if args.save_json:
            result.save_json(args.save_json)
            print(f"[real_world][lpb_v2] wrote {args.save_json}")
    finally:
        discriminator.close()


if __name__ == "__main__":
    main()
