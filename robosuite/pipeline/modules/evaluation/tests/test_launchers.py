"""Tensor-free contracts for round-scoped evaluation launchers."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from omegaconf import OmegaConf

from robosuite.pipeline.modules.evaluation.launch_discriminator import (
    build_commands as build_discriminator_commands,
)
from robosuite.pipeline.modules.visualization.vast.launch import (
    build_commands as build_vast_commands,
)
from robosuite.pipeline.workflow import RunLayout


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fixture")
    return path


def _run_fixture(tmp_path: Path) -> tuple[RunLayout, Path]:
    layout = RunLayout(tmp_path / "run")
    encoder = _touch(layout.checkpoints_dir / "dynamics" / "model.pth")
    warmup = _touch(layout.data_dir / "vast_warmup" / "vast_offline_transitions.pt")
    benchmark = tmp_path / "benchmark"
    benchmark.mkdir()
    _touch(layout.round_data_dir(7) / "episodes.pt")
    _touch(layout.round_dir(7) / "disc" / "checkpoints" / "pu_bce_head_finetuned.pth")
    _touch(layout.round_dir(7) / "vast" / "checkpoints" / "vast_state_finetuned.pt")
    cfg = OmegaConf.create(
        {
            "task": {
                "name": "PickPlaceCereal",
                "inputs": {
                    "dynamics_encoder_checkpoint": str(encoder),
                    "vast_warmup_transitions": str(warmup.parent),
                },
                "evaluation": {
                    "data_root": str(benchmark),
                    "fail_split": "fail_rollout-val-labeled",
                    "success_split": "success_rollout-val",
                },
                "discriminator": {"encode_batch_size": 32},
            },
            "cuda": {"training_device": "cuda:0"},
            "environment": {"control_frequency": 20},
            "evaluation": {"discriminator": {"seed": 0}},
            "visualization": {
                "discriminator": {"split": "both", "num_trajectories": 4},
                "vast": {
                    "seeds": [1, 2, 3, 4, 5, 6],
                    "split": "fail_rollout",
                    "gae_lambda": 0.9,
                },
            },
        }
    )
    layout.root.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, layout.config_path, resolve=True)
    layout.state_path.write_text(json.dumps({"sentinel": "unchanged"}), encoding="utf-8")
    return layout, warmup


def _value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def test_discriminator_commands_are_round_scoped_and_ordered(tmp_path: Path) -> None:
    layout, _ = _run_fixture(tmp_path)
    original_state = layout.state_path.read_bytes()

    commands = build_discriminator_commands(layout.root, 7)

    assert [command[2] for command in commands] == [
        "robosuite.pipeline.modules.evaluation.discriminator",
        "robosuite.pipeline.modules.visualization.discriminator.cli",
    ]
    for command in commands:
        assert _value(command, "--load-ckpt").endswith(
            "rounds/007/disc/checkpoints/pu_bce_head_finetuned.pth"
        )
        assert _value(command, "--offline-episodes").endswith(
            "rounds/007/data/episodes.pt"
        )
        assert _value(command, "--model-ckpt").endswith(
            "inputs/checkpoints/dynamics/model.pth"
        )
    assert _value(commands[0], "--out-dir").endswith(
        "rounds/007/eval_vis/disc/evaluation/seed_000"
    )
    assert _value(commands[1], "--out-dir").endswith(
        "rounds/007/eval_vis/disc/visualization/seed_000"
    )
    assert layout.state_path.read_bytes() == original_state


def test_discriminator_commands_support_pre_task2_run_config(tmp_path: Path) -> None:
    layout, _ = _run_fixture(tmp_path)
    cfg = OmegaConf.load(layout.config_path)
    del cfg.task["evaluation"]
    OmegaConf.save(cfg, layout.config_path, resolve=True)

    commands = build_discriminator_commands(layout.root, 7)

    assert _value(commands[0], "--data-root").endswith("/data")
    assert _value(commands[0], "--fail-split") == "fail_rollout-val-labeled"
    assert _value(commands[0], "--success-split") == "success_rollout-val"


def test_vast_commands_use_confirmed_visualization_defaults(tmp_path: Path) -> None:
    layout, warmup = _run_fixture(tmp_path)
    original_state = layout.state_path.read_bytes()

    commands = build_vast_commands(layout.root, 7)

    assert len(commands) == 6
    assert [_value(command, "--seed") for command in commands] == [
        "1", "2", "3", "4", "5", "6"
    ]
    for command in commands:
        assert command[2] == "robosuite.pipeline.modules.visualization.vast.cli"
        assert _value(command, "--split") == "fail_rollout"
        assert _value(command, "--gae-lambda") == "0.9"
        assert _value(command, "--offline-buffer") == str(warmup)
        assert _value(command, "--output-root").endswith(
            "rounds/007/eval_vis/vast"
        )
    assert layout.state_path.read_bytes() == original_state


def test_vast_commands_support_seed_and_split_overrides(tmp_path: Path) -> None:
    layout, _ = _run_fixture(tmp_path)

    commands = build_vast_commands(
        layout.root,
        7,
        seeds=[8, 13],
        split="success_rollout",
    )

    assert [_value(command, "--seed") for command in commands] == ["8", "13"]
    assert all(
        _value(command, "--split") == "success_rollout" for command in commands
    )


def test_bash_entrypoints_validate_round_and_forward_two_arguments(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[5]
    run_root = tmp_path / "run"
    run_root.mkdir()
    log = tmp_path / "fake-python.log"
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"$FAKE_PYTHON_LOG\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    env = {**os.environ, "PY": str(fake_python), "FAKE_PYTHON_LOG": str(log)}

    eval_script = repository / "robosuite" / "pipeline" / "scripts" / "eval_disc.sh"
    result = subprocess.run(
        [str(eval_script), str(run_root), "007"], env=env, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr
    assert log.read_text(encoding="utf-8").splitlines() == [
        "-m",
        "robosuite.pipeline.modules.evaluation.launch_discriminator",
        str(run_root.resolve()),
        "007",
    ]

    vast_script = repository / "robosuite" / "pipeline" / "scripts" / "vis_vast.sh"
    result = subprocess.run(
        [str(vast_script), str(run_root), "-1"], env=env, text=True, capture_output=True
    )
    assert result.returncode == 2
    assert "non-negative integer" in result.stderr
    result = subprocess.run(
        [str(vast_script), str(run_root), "abc"], env=env, text=True, capture_output=True
    )
    assert result.returncode == 2
    assert "non-negative integer" in result.stderr
    result = subprocess.run(
        [str(vast_script), str(run_root)], env=env, text=True, capture_output=True
    )
    assert result.returncode == 2
    assert "Usage:" in result.stderr
    result = subprocess.run(
        [str(vast_script), str(run_root), "7", "extra"],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert "Usage:" in result.stderr

    result = subprocess.run(
        [str(vast_script)],
        env={
            **env,
            "RUN_ROOT": str(run_root),
            "ROUND_INDEX": "7",
            "SEEDS": "8 13",
            "SPLIT": "success_rollout",
        },
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert log.read_text(encoding="utf-8").splitlines() == [
        "-m",
        "robosuite.pipeline.modules.visualization.vast.launch",
        str(run_root.resolve()),
        "7",
        "--seeds",
        "8",
        "13",
        "--split",
        "success_rollout",
    ]
