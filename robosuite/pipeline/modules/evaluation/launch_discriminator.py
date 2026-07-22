"""Run discriminator evaluation and visualization for one online round."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from robosuite.pipeline.workflow import RunLayout


def _select(cfg: DictConfig, key: str, default: Any) -> Any:
    return OmegaConf.select(cfg, key, default=default)


def _require_file(path: Path, *, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _append_optional(args: list[str], option: str, value: Any) -> None:
    if value is not None:
        args.extend([option, str(value)])


def build_commands(run_root: str | Path, round_index: int) -> list[list[str]]:
    """Build the ordered load-only evaluation and visualization commands."""

    if isinstance(round_index, bool) or round_index < 0:
        raise ValueError("round_index must be a non-negative integer")
    layout = RunLayout(Path(run_root))
    cfg = OmegaConf.load(_require_file(layout.config_path, label="Resolved config"))
    checkpoint = _require_file(
        layout.round_dir(round_index)
        / "disc"
        / "checkpoints"
        / "pu_bce_head_finetuned.pth",
        label="Discriminator checkpoint",
    )
    encoder = _require_file(
        Path(str(cfg.task.inputs.dynamics_encoder_checkpoint)),
        label="Dynamics encoder checkpoint",
    )
    episodes = _require_file(
        layout.round_data_dir(round_index) / "episodes.pt",
        label="Round episodes",
    )
    repository_root = Path(__file__).resolve().parents[4]
    default_data_root = repository_root / "data"
    data_root = Path(
        str(_select(cfg, "task.evaluation.data_root", default_data_root))
    )
    if not data_root.is_absolute():
        data_root = repository_root / data_root
    if not data_root.is_dir():
        raise FileNotFoundError(f"Evaluation data root does not exist: {data_root}")

    task = str(cfg.task.name)
    fail_split = str(
        _select(cfg, "task.evaluation.fail_split", "fail_rollout-val-labeled")
    )
    success_split = str(
        _select(cfg, "task.evaluation.success_split", "success_rollout-val")
    )
    device = str(_select(cfg, "evaluation.discriminator.device", cfg.cuda.training_device))
    seed = int(_select(cfg, "evaluation.discriminator.seed", 0))
    encode_batch_size = int(
        _select(cfg, "evaluation.discriminator.encode_batch_size", cfg.task.discriminator.encode_batch_size)
    )
    offline_fps = int(
        _select(cfg, "visualization.discriminator.fps", cfg.environment.control_frequency)
    )
    offline_video_size = int(
        _select(cfg, "visualization.discriminator.offline_video_size", 256)
    )
    common = [
        "--model-ckpt", str(encoder),
        "--load-ckpt", str(checkpoint),
        "--task", task,
        "--offline-episodes", str(episodes),
        "--data-root", str(data_root),
        "--fail-split", fail_split,
        "--success-split", success_split,
        "--device", device,
        "--encode-batch-size", str(encode_batch_size),
        "--seed", str(seed),
        "--offline-video-size", str(offline_video_size),
    ]
    max_fail = _select(cfg, "evaluation.discriminator.max_fail_per_task", None)
    max_success = _select(cfg, "evaluation.discriminator.max_success_per_task", None)

    evaluation_args = [
        *common,
        "--out-dir", str(layout.disc_eval_vis_dir(round_index) / "evaluation" / f"seed_{seed:03d}"),
        "--offline-fps", str(offline_fps),
    ]
    _append_optional(evaluation_args, "--max-fail-per-task", max_fail)
    _append_optional(evaluation_args, "--max-success-per-task", max_success)

    visualization_args = [
        *common,
        "--out-dir", str(layout.disc_eval_vis_dir(round_index) / "visualization" / f"seed_{seed:03d}"),
        "--split", str(_select(cfg, "visualization.discriminator.split", "both")),
        "--num-trajs", str(_select(cfg, "visualization.discriminator.num_trajectories", 10)),
        "--fps", str(offline_fps),
        "--camera-name", str(_select(cfg, "visualization.discriminator.camera_name", "agentview")),
        "--border-thickness", str(_select(cfg, "visualization.discriminator.border_thickness", 10)),
        "--pdf-name", "finetuned_scores.pdf",
    ]
    _append_optional(visualization_args, "--max-fail-per-task", max_fail)
    _append_optional(visualization_args, "--max-success-per-task", max_success)
    return [
        [sys.executable, "-m", "robosuite.pipeline.modules.evaluation.discriminator", *evaluation_args],
        [sys.executable, "-m", "robosuite.pipeline.modules.visualization.discriminator.cli", *visualization_args],
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root")
    parser.add_argument("round_index", type=int)
    args = parser.parse_args()
    if args.round_index < 0:
        parser.error("round_index must be non-negative")
    return args


def main() -> None:
    args = _parse_args()
    for command in build_commands(args.run_root, args.round_index):
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
