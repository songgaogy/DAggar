import json
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from robosuite.pipeline.utils import (
    ConsoleLogCapture,
    JsonlEventLogger,
    TensorBoardLogger,
    create_run_directory,
)


def test_run_directory_uses_task_timestamp_contract(tmp_path: Path) -> None:
    name, path = create_run_directory(tmp_path, "PickPlaceCereal")
    assert name.startswith("PickPlaceCereal_")
    assert path.parent.name == "PickPlaceCereal"
    assert path.name == name


def test_tensorboard_writes_only_finite_scalars(tmp_path: Path) -> None:
    with TensorBoardLogger(tmp_path, flush_secs=1) as logger:
        assert logger.log(
            {"qa_loss": 1.25, "invalid": float("nan"), "text": "skip"},
            step=7,
            prefix="train",
        ) == 1
        logger.flush()
    accumulator = EventAccumulator(str(tmp_path))
    accumulator.Reload()
    assert accumulator.Scalars("train/qa_loss")[0].value == 1.25


def test_jsonl_and_console_are_persisted(tmp_path: Path, capsys) -> None:
    metrics = tmp_path / "metrics.jsonl"
    with JsonlEventLogger(metrics) as logger:
        logger.log({"event": "episode", "worker": 0, "success": 1})
    assert json.loads(metrics.read_text()) == {
        "event": "episode",
        "success": 1,
        "worker": 0,
    }

    console = tmp_path / "console.log"
    with ConsoleLogCapture(console):
        print("dsrl event")
    assert "dsrl event" in capsys.readouterr().out
    assert "dsrl event" in console.read_text()
