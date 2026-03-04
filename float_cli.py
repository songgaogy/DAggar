from __future__ import annotations

import argparse
import json
from typing import Optional


def _add_common_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--expert-dir", required=True, help="Path to expert trajectory HDF5 directory")
    parser.add_argument("--obs-key", default="states", choices=["states", "images", "actions"])
    parser.add_argument("--camera-name", default="agentview")
    parser.add_argument("--pad-to", type=int, default=-1, help="Pad trajectories to this length; <=0 disables")
    parser.add_argument("--sinkhorn-reg", type=float, default=0.05)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--delta", type=float, default=10.0, help="Delta in percent [0,100]")


def _resolve_camera_name(camera_name: str, obs_key: str) -> Optional[str]:
    if obs_key != "images":
        return None
    return camera_name


def cmd_calibrate(args: argparse.Namespace) -> None:
    from robosuite.discriminator.float_data import load_trajectories
    from robosuite.discriminator.float_eval import build_float_and_calibrate

    expert = load_trajectories(
        data_dir=args.expert_dir,
        obs_key=args.obs_key,
        camera_name=_resolve_camera_name(args.camera_name, args.obs_key),
    )

    if args.success_dir:
        success = load_trajectories(
            data_dir=args.success_dir,
            obs_key=args.obs_key,
            camera_name=_resolve_camera_name(args.camera_name, args.obs_key),
        )
    else:
        success = expert

    float_computer, calibrator, threshold = build_float_and_calibrate(
        expert_rollouts=expert,
        success_rollouts_for_calibration=success,
        sinkhorn_reg=float(args.sinkhorn_reg),
        max_iter=int(args.max_iter),
        tol=float(args.tol),
        delta=float(args.delta),
        pad_to=(None if int(args.pad_to) <= 0 else int(args.pad_to)),
        use_similarity_cost=bool(args.use_similarity_cost),
    )

    result = {
        "threshold": float(threshold),
        "delta": float(args.delta),
        "num_expert": len(expert),
        "num_success": len(success),
        "num_success_lambdas": len(calibrator.success_lambdas),
        "expert_embedding_dims": [list(e.shape) for e in float_computer.expert_embeddings],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


def cmd_eval(args: argparse.Namespace) -> None:
    from robosuite.discriminator.float_data import load_trajectories
    from robosuite.discriminator.float_eval import (
        build_float_and_calibrate,
        evaluate_fail_rollouts,
        summarize_lambda_separation,
    )

    expert = load_trajectories(
        data_dir=args.expert_dir,
        obs_key=args.obs_key,
        camera_name=_resolve_camera_name(args.camera_name, args.obs_key),
    )
    fail = load_trajectories(
        data_dir=args.fail_dir,
        obs_key=args.obs_key,
        camera_name=_resolve_camera_name(args.camera_name, args.obs_key),
    )

    if args.success_dir:
        success = load_trajectories(
            data_dir=args.success_dir,
            obs_key=args.obs_key,
            camera_name=_resolve_camera_name(args.camera_name, args.obs_key),
        )
    else:
        success = expert

    float_computer, calibrator, threshold = build_float_and_calibrate(
        expert_rollouts=expert,
        success_rollouts_for_calibration=success,
        sinkhorn_reg=float(args.sinkhorn_reg),
        max_iter=int(args.max_iter),
        tol=float(args.tol),
        delta=float(args.delta),
        pad_to=(None if int(args.pad_to) <= 0 else int(args.pad_to)),
        use_similarity_cost=bool(args.use_similarity_cost),
    )

    fail_metrics, preds, _ = evaluate_fail_rollouts(
        fail_rollouts=fail,
        float_computer=float_computer,
        calibrator=calibrator,
        delta=float(args.delta),
        delta_step=float(args.delta_step),
        fail_tail_ratio=float(args.fail_tail_ratio),
        stride=int(args.stride),
        adaptive_delta=bool(args.adaptive_delta),
    )

    result = {
        "threshold": float(threshold),
        "fail_metrics": fail_metrics,
        "lambda_separation": summarize_lambda_separation(preds),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="float_cli",
        description="FLOAT: failure detection based on OT over policy embeddings",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    calibrate = sub.add_parser("calibrate", help="Calibrate FLOAT threshold")
    _add_common_dataset_args(calibrate)
    calibrate.add_argument("--success-dir", default="", help="Optional rollout-success directory")
    calibrate.add_argument("--use-similarity-cost", action="store_true")
    calibrate.set_defaults(func=cmd_calibrate)

    eval_parser = sub.add_parser("eval", help="Evaluate FLOAT on fail rollouts")
    _add_common_dataset_args(eval_parser)
    eval_parser.add_argument("--fail-dir", required=True, help="Path to fail rollout HDF5 directory")
    eval_parser.add_argument("--success-dir", default="", help="Optional rollout-success directory")
    eval_parser.add_argument("--fail-tail-ratio", type=float, default=0.2)
    eval_parser.add_argument("--stride", type=int, default=1)
    eval_parser.add_argument("--delta-step", type=float, default=1.0)
    eval_parser.add_argument("--adaptive-delta", action="store_true")
    eval_parser.add_argument("--use-similarity-cost", action="store_true")
    eval_parser.set_defaults(func=cmd_eval)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
