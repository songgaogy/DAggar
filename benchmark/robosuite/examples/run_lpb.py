"""Run the LPB KNN discriminator through the robosuite benchmark API.

Example:
    python -m benchmark.robosuite.examples.run_lpb \
        --lpb-ckpt /abs/path/checkpoints/lpb/dynamics/model.pt \
        --fail-root /abs/path/data/utils/fail_rollout \
        --success-root /abs/path/data/utils/success_rollout \
        --tasks PickPlaceCan \
        --save-json /tmp/lpb_bench.json
"""

from __future__ import annotations

import argparse

from benchmark.robosuite import FailureBenchmark
from robosuite.discriminator.lpb.lpb_benchmark import LPBBenchmarkDiscriminator


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lpb-ckpt", required=True, help="Path to LPB dynamics checkpoint (.pt)")
    parser.add_argument("--fail-root", required=True, help="data/utils/fail_rollout")
    parser.add_argument("--success-root", required=True, help="data/utils/success_rollout")
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--save-json", type=str, default=None)

    # Feature extractor.
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--action-horizon", type=int, default=-1,
                        help="Override action horizon; <=0 means use checkpoint value.")
    parser.add_argument("--camera-name", type=str, default="agentview")
    parser.add_argument("--proprio-indices", type=int, nargs="*", default=None,
                        help="Optional state indices to use as proprio (default: use all).")
    parser.add_argument("--no-normalize-feature", action="store_true",
                        help="Disable L2 feature normalization.")
    parser.add_argument("--use-transition-error", action="store_true",
                        help="Also compute transition (latent + proprio) prediction error as aux signal.")
    parser.add_argument("--transition-proprio-error-weight", type=float, default=0.1)

    # Detector.
    parser.add_argument("--delta", type=float, default=10.0,
                        help="Percentile-based false-alarm budget (0-100).")
    parser.add_argument("--delta-step", type=float, default=1.0)
    parser.add_argument("--knn-chunk-size", type=int, default=8192)
    parser.add_argument("--lambda-mode", type=str, default="mean", choices=["mean", "max"])
    parser.add_argument("--lambda-window-size", type=int, default=-1,
                        help="Rolling window; -1 = full prefix.")
    parser.add_argument("--transition-aux-weight", type=float, default=0.0,
                        help="Linear weight on transition-error aux signal. 0 = disabled.")

    # Calibration split + misc.
    parser.add_argument("--calib-fraction", type=float, default=0.2,
                        help="Fraction of success trajectories held out as disjoint calibration set.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet-fit", action="store_true")

    # Data caps.
    parser.add_argument("--max-fail-per-task", type=int, default=None)
    parser.add_argument("--max-success-per-task", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    bench = FailureBenchmark(
        fail_labeled_root=args.fail_root,
        success_root=args.success_root,
        tasks=args.tasks,
        max_fail_per_task=args.max_fail_per_task,
        max_success_per_task=args.max_success_per_task,
    )
    discriminator = LPBBenchmarkDiscriminator(
        checkpoint_path=str(args.lpb_ckpt),
        device=str(args.device),
        feature_batch_size=int(args.feature_batch_size),
        action_horizon=int(args.action_horizon),
        camera_name=str(args.camera_name),
        proprio_indices=(list(args.proprio_indices) if args.proprio_indices else None),
        normalize_feature=not bool(args.no_normalize_feature),
        use_transition_error=bool(args.use_transition_error),
        transition_proprio_error_weight=float(args.transition_proprio_error_weight),
        delta=float(args.delta),
        delta_step=float(args.delta_step),
        knn_chunk_size=int(args.knn_chunk_size),
        lambda_mode=str(args.lambda_mode),
        lambda_window_size=int(args.lambda_window_size),
        transition_aux_weight=float(args.transition_aux_weight),
        calib_fraction=float(args.calib_fraction),
        seed=int(args.seed),
        verbose_fit=not bool(args.quiet_fit),
    )
    try:
        discriminator.fit_on_benchmark(bench.trajectories())
        result = bench.evaluate(discriminator)
        print(result.summary())
        print("[lpb_knn] calibration summary:", discriminator.calibration_summary())
        if args.save_json:
            result.save_json(args.save_json)
            print(f"[benchmark] wrote {args.save_json}")
    finally:
        discriminator.close()


if __name__ == "__main__":
    main()
