import json
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from robosuite.pipeline.utils.logging import ConsoleLogCapture, JsonlEventLogger, TensorBoardLogger
from robosuite.pipeline.utils.runtime import create_run_directory


def test_tensorboard_logger_writes_awr_metric_groups(tmp_path: Path) -> None:
    log_dir = tmp_path / "tensorboard"
    with TensorBoardLogger(log_dir, flush_secs=1) as logger:
        assert logger.log(
            {"q_loss": 1.25, "invalid": float("nan"), "text": "skip"},
            step=7,
            prefix="train",
        ) == 1
        logger.log({"value_loss": 0.5}, step=7, prefix="warmup")
        logger.log({"success": 1.0, "intervention_steps": 2.0}, step=1, prefix="episode")
        logger.log({"env_fps": 12.0, "online_size": 64}, step=7, prefix="runtime")
        logger.log({"saved": 1.0}, step=7, prefix="checkpoint")
        logger.flush()

    accumulator = EventAccumulator(str(log_dir))
    accumulator.Reload()
    assert {
        "train/q_loss",
        "warmup/value_loss",
        "episode/success",
        "episode/intervention_steps",
        "runtime/env_fps",
        "runtime/online_size",
        "checkpoint/saved",
    } <= set(accumulator.Tags()["scalars"])
    event = accumulator.Scalars("train/q_loss")[0]
    assert event.step == 7
    assert event.value == 1.25


def test_disabled_tensorboard_logger_has_no_side_effects(tmp_path: Path) -> None:
    _, run_dir = create_run_directory(tmp_path, "PickPlaceCereal", "disabled_tb")
    log_dir = run_dir / "tensorboard"
    with TensorBoardLogger(log_dir, enabled=False) as logger:
        assert logger.log({"loss": 1.0}, step=1) == 0
    assert not log_dir.exists()


def test_jsonl_logger_appends_sorted_events(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    with JsonlEventLogger(path) as logger:
        logger.log({"step": 2, "group": "train", "q_loss": 1.5})
        logger.log({"step": 3, "group": "runtime", "env_fps": 10.0})

    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert events == [
        {"group": "train", "q_loss": 1.5, "step": 2},
        {"env_fps": 10.0, "group": "runtime", "step": 3},
    ]


def test_console_capture_mirrors_stdout_and_stderr(tmp_path: Path, capsys) -> None:
    path = tmp_path / "console.log"
    with ConsoleLogCapture(path):
        print("stdout event")
        print("stderr event", file=__import__("sys").stderr)

    captured = capsys.readouterr()
    contents = path.read_text(encoding="utf-8")
    assert "stdout event" in captured.out
    assert "stderr event" in captured.err
    assert "stdout event" in contents
    assert "stderr event" in contents
