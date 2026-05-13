"""Run the LPB v2 KNN discriminator through the shared benchmark API.

Mirrors data/utils/benchmark/examples/run_lpb_original.py but uses the cleaned
LPB v2 module under robosuite.discriminator.lpb_v2.

Example:
    python -m data.utils.benchmark.examples.run_lpb_v2 \
        --model-ckpt /abs/path/checkpoints/lpb_v2/dynamics/<run_name-timestamp>/checkpoints/model_50.pth \
        --fail-root  /abs/path/data/utils/fail_rollout \
        --success-root /abs/path/data/utils/success_rollout \
        --tasks PickPlaceCan \
        --save-json /tmp/lpb_v2_bench.json
"""

from __future__ import annotations

import argparse

from benchmark.robosuite import FailureBenchmark
from robosuite.discriminator.lpb_v2.adapters.single_bank import LPBV2BenchmarkDiscriminator


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-ckpt", required=True,
                        help="Path to an LPB v2/original-compatible dynamics checkpoint.")
    parser.add_argument("--fail-root", required=True, help="data/utils/fail_rollout")
    parser.add_argument("--success-root", required=True, help="data/utils/success_rollout")
    parser.add_argument("--success-cache-root", type=str, default=None,
                        help="Optional LPB preprocessed cache root for success trajectories.")
    parser.add_argument("--metadata-cache-root", type=str, default=None,
                        help="Optional LPB metadata cache root used to identify success cache entries.")
    parser.add_argument("--cache-camera-names", nargs="*", default=None,
                        help="Camera names for cache view slots, e.g. agentview birdview frontview.")
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
    print("[lpb_v2] building FailureBenchmark...", flush=True)
    bench = FailureBenchmark(
        fail_labeled_root=args.fail_root,
        success_root=args.success_root,
        tasks=args.tasks,
        max_fail_per_task=args.max_fail_per_task,
        max_success_per_task=args.max_success_per_task,
        success_cache_root=args.success_cache_root,
        metadata_cache_root=args.metadata_cache_root,
        cache_camera_names=args.cache_camera_names,
    )
    print("[lpb_v2] discovering trajectories (this may take a while on slow disks)...", flush=True)
    trajs = bench.trajectories()
    n_fail = sum(1 for t in trajs if bool(t.is_failure))
    n_succ = sum(1 for t in trajs if not bool(t.is_failure))
    tasks = sorted({str(t.task_name) for t in trajs})
    print(
        f"[lpb_v2] discovered num_trajectories={len(trajs)} (fail={n_fail}, succ={n_succ}) "
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
        print("[lpb_v2] starting fit_on_benchmark (encoding success demos)...", flush=True)
        discriminator.fit_on_benchmark(bench.trajectories())
        print("[lpb_v2] fit done; starting evaluate()...", flush=True)
        result = bench.evaluate(discriminator)
        print(result.summary())
        print("[lpb_v2] calibration summary:", discriminator.calibration_summary())
        if args.save_json:
            result.save_json(args.save_json)
            print(f"[benchmark] wrote {args.save_json}")
    finally:
        discriminator.close()


if __name__ == "__main__":
    main()
