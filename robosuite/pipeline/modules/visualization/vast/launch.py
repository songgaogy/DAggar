"""Visualize one round's finetuned VAST checkpoint for configured seeds."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from robosuite.pipeline.workflow import RunLayout
from robosuite.pipeline.workflow.state import load_json


def _select(cfg: DictConfig, key: str, default: Any) -> Any:
    return OmegaConf.select(cfg, key, default=default)


def _require_file(path: Path, *, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _resolve_warmup_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_dir():
        path = path / "vast_offline_transitions.pt"
    return _require_file(path, label="VAST warmup replay")


def _resolve_online_episode_paths(
    layout: RunLayout,
    round_index: int,
) -> list[Path]:
    state = load_json(_require_file(layout.state_path, label="Run state"))
    rounds = state.get("rounds", {})
    paths: list[Path] = []
    for index in range(int(round_index) + 1):
        key = f"{index:03d}"
        try:
            collection = rounds[key]["stages"]["collection"]
            relative = collection["outputs"]["episodes"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                f"Round {key} has no completed collection episodes in {layout.state_path}."
            ) from exc
        path = Path(str(relative))
        if not path.is_absolute():
            path = layout.root / path
        paths.append(_require_file(path, label=f"Round {key} online episodes"))
    return paths


def build_commands(
    run_root: str | Path,
    round_index: int,
    *,
    seeds: list[int] | None = None,
    split: str | None = None,
) -> list[list[str]]:
    """Build one VAST visualization command for each configured seed."""

    if isinstance(round_index, bool) or round_index < 0:
        raise ValueError("round_index must be a non-negative integer")
    layout = RunLayout(Path(run_root))
    cfg = OmegaConf.load(_require_file(layout.config_path, label="Resolved config"))
    round_dir = layout.round_dir(round_index)
    vast_checkpoint = _require_file(
        round_dir / "vast" / "checkpoints" / "vast_state_finetuned.pt",
        label="VAST checkpoint",
    )
    disc_checkpoint = _require_file(
        round_dir / "disc" / "checkpoints" / "pu_bce_head_finetuned.pth",
        label="Discriminator checkpoint",
    )
    seeds = (
        [int(seed) for seed in seeds]
        if seeds is not None
        else [
            int(seed)
            for seed in _select(
                cfg, "visualization.vast.seeds", [1, 2, 3, 4, 5, 6]
            )
        ]
    )
    if not seeds:
        raise ValueError("VAST visualization seeds must not be empty")

    device = str(_select(cfg, "visualization.vast.device", cfg.cuda.training_device))
    split = str(
        split
        if split is not None
        else _select(cfg, "visualization.vast.split", "fail_rollout")
    )
    is_online = split.strip().lower() == "online"
    warmup = (
        None
        if is_online
        else _resolve_warmup_path(str(cfg.task.inputs.vast_warmup_transitions))
    )
    online_episodes = (
        _resolve_online_episode_paths(layout, round_index) if is_online else []
    )
    gae_lambda = float(
        _select(
            cfg,
            "task.policy.advantage.gae_lambda"
            if is_online
            else "visualization.vast.gae_lambda",
            0.6 if is_online else 0.9,
        )
    )
    output_root = layout.vast_eval_vis_dir(round_index)
    commands: list[list[str]] = []
    for seed in seeds:
        command = [
            sys.executable,
            "-m",
            (
                "robosuite.pipeline.modules.visualization.vast.online"
                if is_online
                else "robosuite.pipeline.modules.visualization.vast.cli"
            ),
            "--vast-ckpt", str(vast_checkpoint),
            "--disc-ckpt", str(disc_checkpoint),
            "--task-data-name", str(cfg.task.name),
            "--split", split,
            "--seed", str(seed),
            "--device", device,
            "--output-root", str(output_root),
            "--gae-lambda", str(gae_lambda),
            "--video-fps",
            str(
                _select(
                    cfg,
                    "visualization.vast.video_fps",
                    cfg.environment.control_frequency,
                )
            ),
            "--disc-viz-image-size",
            str(_select(cfg, "visualization.vast.discriminator_image_size", 256)),
            "--disc-viz-border-thickness",
            str(
                _select(
                    cfg,
                    "visualization.vast.discriminator_border_thickness",
                    10,
                )
            ),
        ]
        if is_online:
            command.extend(
                [
                    "--online-episodes",
                    *[str(path) for path in online_episodes],
                    "--advantage-estimator",
                    str(
                        _select(cfg, "task.policy.advantage.estimator", "td1")
                    ).strip().lower(),
                    "--reward-success",
                    str(_select(cfg, "task.policy.reward_success", 0.0)),
                    "--reward-fail",
                    str(_select(cfg, "task.policy.reward_failure", -1.0)),
                ]
            )
        else:
            assert warmup is not None
            command.extend(["--offline-buffer", str(warmup)])
        max_windows = _select(cfg, "visualization.vast.max_windows", None)
        if max_windows is not None:
            command.extend(["--max-windows", str(max_windows)])
        sampling_seed = _select(cfg, "visualization.vast.sampling_seed", None)
        if sampling_seed is not None:
            command.extend(["--vast-sampling-seed", str(sampling_seed)])
        commands.append(command)
    return commands


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root")
    parser.add_argument("round_index", type=int)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--split", default=None)
    args = parser.parse_args()
    if args.round_index < 0:
        parser.error("round_index must be non-negative")
    return args


def main() -> None:
    args = _parse_args()
    for command in build_commands(
        args.run_root,
        args.round_index,
        seeds=args.seeds,
        split=args.split,
    ):
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
