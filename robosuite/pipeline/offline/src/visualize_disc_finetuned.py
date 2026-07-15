"""CUDA-only visualization entry point for an offline-finetuned nnPU head."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from robosuite.discriminator.dyn_disc.visualization.visualize_pu_bce import (
    _bootstrap_from_ckpt,
    _parse_camera_to_view,
    _sample_trajectories_for_viz,
)
from robosuite.discriminator.utils.robosuite_benchmark import FailureBenchmark
from robosuite.pipeline.offline.visualization import (
    FinetunedPUBCEBenchmarkDiscriminator,
    FinetunedPUBCEVisualizer,
    load_finetuned_visualization_contract,
    sample_offline_trajectories,
    validate_runtime_normalizer,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize one offline-finetuned nnPU checkpoint."
    )
    parser.add_argument("--model-ckpt", required=True)
    parser.add_argument("--load-ckpt", required=True)
    parser.add_argument("--offline-episodes", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--fail-split", default="fail_rollout-val-labeled")
    parser.add_argument("--success-split", default="success_rollout-val")
    parser.add_argument(
        "--split",
        default="both",
        choices=["success_rollout", "fail_rollout", "both"],
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--num-trajs", type=int, default=4)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--pdf-name", default="finetuned_scores.pdf")
    parser.add_argument("--offline-video-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--border-thickness", type=int, default=10)
    parser.add_argument("--max-fail-per-task", type=int, default=None)
    parser.add_argument("--max-success-per-task", type=int, default=None)
    parser.add_argument("--no-flip-vertical", action="store_true")
    parser.add_argument("--no-debug-score-stats", action="store_true")
    parser.add_argument("--proprio-indices", type=int, nargs="*", default=None)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument("--camera-to-view", default=None)
    parser.add_argument("--camera-name", default="agentview")
    parser.add_argument("--visual-weight", type=float, default=1.0)
    parser.add_argument("--proprio-weight", type=float, default=2.0)
    parser.add_argument("--action-weight", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=10.0)
    parser.add_argument(
        "--feature-source",
        default="transformer",
        choices=["encoder", "transformer"],
    )
    parser.add_argument("--transformer-layer", type=int, default=1)
    parser.add_argument("--calib-fraction", type=float, default=0.2)
    return parser.parse_args()


def _require_cuda(device_value: str) -> torch.device:
    device = torch.device(str(device_value))
    if device.type != "cuda":
        raise ValueError(
            f"Finetuned nnPU visualization requires a CUDA device, got {device_value!r}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Finetuned nnPU visualization requires CUDA, but "
            "torch.cuda.is_available() is False."
        )
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise ValueError(
            f"CUDA device {device} is unavailable; visible device count is "
            f"{torch.cuda.device_count()}."
        )
    torch.cuda.set_device(0 if device.index is None else device.index)
    return device


def _eval_trajectories(args: argparse.Namespace):
    benchmark = FailureBenchmark(
        data_root=str(args.data_root),
        tasks=[str(args.task)],
        fail_split=str(args.fail_split),
        success_split=str(args.success_split),
        max_fail_per_task=args.max_fail_per_task,
        max_success_per_task=args.max_success_per_task,
    )
    trajectories = benchmark.trajectories()
    if not trajectories:
        raise RuntimeError(f"No trajectories discovered for task {args.task!r}.")
    return trajectories


def main() -> None:
    args = _parse_args()
    if int(args.num_trajs) <= 0:
        raise ValueError(f"--num-trajs must be positive, got {args.num_trajs}.")
    device = _require_cuda(str(args.device))
    requested_camera_map = _parse_camera_to_view(args.camera_to_view)
    contract = load_finetuned_visualization_contract(
        args.load_ckpt,
        model_checkpoint=args.model_ckpt,
        feature_source=str(args.feature_source),
        transformer_layer=int(args.transformer_layer),
        camera_to_view=requested_camera_map,
        proprio_indices=args.proprio_indices,
    )

    eval_trajectories = _eval_trajectories(args)
    sampled_eval = _sample_trajectories_for_viz(
        eval_trajectories,
        task=str(args.task),
        split=str(args.split),
        num_trajs=int(args.num_trajs),
        seed=int(args.seed),
    )
    sampled_offline = sample_offline_trajectories(
        args.offline_episodes,
        task=str(args.task),
        num_trajs=int(args.num_trajs),
        seed=int(args.seed),
        fps=int(args.fps),
        video_size=int(args.offline_video_size),
    )

    with torch.device(device):
        discriminator = FinetunedPUBCEBenchmarkDiscriminator(
            model_ckpt=str(Path(args.model_ckpt).expanduser().resolve()),
            unlabeled_fail_trajectories=[],
            save_ckpt_dir=None,
            device=str(device),
            encode_batch_size=int(args.encode_batch_size),
            proprio_indices=contract.proprio_indices,
            camera_to_view=contract.camera_to_view,
            visual_weight=float(args.visual_weight),
            proprio_weight=float(args.proprio_weight),
            action_weight=float(args.action_weight),
            delta=float(args.delta),
            feature_source=str(args.feature_source),
            transformer_layer=int(args.transformer_layer),
            calib_fraction=float(args.calib_fraction),
            seed=int(args.seed),
            verbose_fit=False,
        )
    validate_runtime_normalizer(discriminator, contract)

    try:
        _bootstrap_from_ckpt(discriminator, str(args.load_ckpt))
        if str(args.task) not in discriminator._detectors_per_task:
            raise RuntimeError(
                f"Task {args.task!r} is absent from the finetuned checkpoint; "
                f"available tasks: {sorted(discriminator._detectors_per_task)}."
            )
        visualizer = FinetunedPUBCEVisualizer(
            discriminator,
            camera_name=str(args.camera_name),
            fps=int(args.fps),
            border_thickness=int(args.border_thickness),
            debug_score_stats=not bool(args.no_debug_score_stats),
            flip_vertical=not bool(args.no_flip_vertical),
        )
        eval_paths = visualizer.visualize(
            sampled_eval,
            out_dir=str(args.out_dir),
            pdf_name=str(args.pdf_name),
            split=str(args.split),
        )
        offline_paths = visualizer.visualize(
            sampled_offline,
            out_dir=str(args.out_dir),
            pdf_name="finetuned_scores_offline.pdf",
            split="offline",
        )
        print(
            f"[pu_bce][viz] done. eval_videos={len(eval_paths['videos'])} "
            f"eval_pdf={eval_paths['pdf']} "
            f"offline_videos={len(offline_paths['videos'])} "
            f"offline_pdf={offline_paths['pdf']}",
            flush=True,
        )
    finally:
        discriminator.close()


if __name__ == "__main__":
    main()
