"""Run the single-bank KNN discriminator through the robosuite benchmark API.

Uses the current robosuite benchmark (``robosuite.discriminator.utils.robosuite_benchmark``,
``data/<task>/<split>`` layout) under robosuite.discriminator.dyn_disc.

Example:
    python -m robosuite.discriminator.dyn_disc.sim_benchmark \
        --model-ckpt /abs/path/checkpoints/dyn_disc/dynamics/<run_name-timestamp>/checkpoints/model_50.pth \
        --data-root /abs/path/data \
        --tasks PickPlaceCan \
        --save-json /tmp/dyn_disc_bench.json
"""

from __future__ import annotations

import argparse

from robosuite.discriminator.utils.robosuite_benchmark import FailureBenchmark
from robosuite.discriminator.dyn_disc.adapters.single_bank import LPBV2BenchmarkDiscriminator


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-ckpt", required=True,
                        help="Path to an LPB v2/original-compatible dynamics checkpoint.")
    parser.add_argument("--data-root", type=str, default="data",
                        help="Root containing data/<task>/<split> directories.")
    parser.add_argument("--fail-split", type=str, default="fail_rollout-val-labeled")
    parser.add_argument("--success-split", type=str, default="success_rollout-val")
    parser.add_argument("--fail-root", type=str, default=None,
                        help="Deprecated; use --data-root/--fail-split.")
    parser.add_argument("--success-root", type=str, default=None,
                        help="Deprecated; use --data-root/--success-split.")
    parser.add_argument("--success-cache-root", type=str, default=None,
                        help="Deprecated; ignored by the new robosuite benchmark.")
    parser.add_argument("--metadata-cache-root", type=str, default=None,
                        help="Deprecated; ignored by the new robosuite benchmark.")
    parser.add_argument("--cache-camera-names", nargs="*", default=None,
                        help="Deprecated; ignored by the new robosuite benchmark.")
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--save-json", type=str, default=None)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument("--proprio-indices", type=int, nargs="*", default=None,
                        help="State indices to slice as proprio (default: use all).")
    parser.add_argument("--camera-to-view", type=str, default=None,
                        help="Comma-separated camera:view pairs, e.g. agentview:agentview,sideview:sideview")

    parser.add_argument("--visual-weight", type=float, default=1.0)
    parser.add_argument("--proprio-weight", type=float, default=2.0)
    parser.add_argument("--action-weight", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=10.0,
                        help="Percentile-based false-alarm budget (0-100).")
    parser.add_argument("--knn-chunk-size", type=int, default=2048)
    parser.add_argument("--knn-feature-source", type=str, default="encoder",
                        choices=["encoder", "transformer"],
                        help="KNN feature source: encoder baseline or transformer hidden state.")
    parser.add_argument("--knn-transformer-layer", type=int, default=-1,
                        help="Transformer layer for KNN when --knn-feature-source=transformer.")
    parser.add_argument("--calib-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet-fit", action="store_true")
    parser.add_argument("--max-fail-per-task", type=int, default=None)
    parser.add_argument("--max-success-per-task", type=int, default=None)
    return parser.parse_args()


def _parse_camera_to_view(s):
    if not s:
        return None
    out = {}
    for chunk in s.split(","):
        cam, view = chunk.split(":", 1)
        out[cam.strip()] = view.strip()
    return out


def main() -> None:
    args = _parse_args()
    print("[dyn_disc] building FailureBenchmark...", flush=True)
    bench = FailureBenchmark(
        data_root=args.data_root,
        tasks=args.tasks,
        fail_split=args.fail_split,
        success_split=args.success_split,
        max_fail_per_task=args.max_fail_per_task,
        max_success_per_task=args.max_success_per_task,
    )
    print("[dyn_disc] discovering trajectories (this may take a while on slow disks)...", flush=True)
    trajs = bench.trajectories()
    n_fail = sum(1 for t in trajs if bool(t.is_failure))
    n_succ = sum(1 for t in trajs if not bool(t.is_failure))
    tasks = sorted({str(t.task_name) for t in trajs})
    print(
        f"[dyn_disc] discovered num_trajectories={len(trajs)} (fail={n_fail}, succ={n_succ}) "
        f"tasks={tasks}",
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
        action_weight=float(args.action_weight),
        delta=float(args.delta),
        knn_chunk_size=int(args.knn_chunk_size),
        feature_source=str(args.knn_feature_source),
        transformer_layer=int(args.knn_transformer_layer),
        calib_fraction=float(args.calib_fraction),
        seed=int(args.seed),
        verbose_fit=not bool(args.quiet_fit),
    )
    try:
        print("[dyn_disc] starting fit_on_benchmark (encoding success demos)...", flush=True)
        discriminator.fit_on_benchmark(bench.trajectories())
        print("[dyn_disc] fit done; starting evaluate()...", flush=True)
        result = bench.evaluate(discriminator)
        print(result.summary())
        print("[dyn_disc] calibration summary:", discriminator.calibration_summary())
        if args.save_json:
            result.save_json(args.save_json)
            print(f"[benchmark] wrote {args.save_json}")
    finally:
        discriminator.close()


if __name__ == "__main__":
    main()
