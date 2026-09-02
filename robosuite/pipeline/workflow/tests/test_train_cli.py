from __future__ import annotations

import argparse
import sys

import pytest

import robosuite.pipeline.train as train_cli
from robosuite.pipeline.workflow import RunLayout


def test_round_index_with_all_rolls_back_in_place_from_discriminator(
    tmp_path, monkeypatch, capsys
) -> None:
    layout = RunLayout(tmp_path / "source")
    calls = {}

    def rollback(target, *, round_index: int, stage: str):
        calls["rollback"] = (target, round_index, stage)

    def train(target, *, requested_stage: str):
        calls["train"] = (target, requested_stage)

    monkeypatch.setattr("builtins.input", lambda _prompt: "")
    monkeypatch.setattr(train_cli, "rollback_run_for_retrain", rollback)
    monkeypatch.setattr(train_cli, "train", train)
    monkeypatch.setattr(
        sys,
        "argv",
        ["train", str(layout.root), "all", "--round-index", "3"],
    )

    train_cli.main()

    assert calls["rollback"] == (layout, 3, "disc")
    assert calls["train"] == (layout, "all")
    output = capsys.readouterr().out
    assert "WARNING" in output
    assert f"run: {layout.root}" in output
    assert "round=003 stage=disc" in output


@pytest.mark.parametrize("confirmation", ["cancel", EOFError()])
def test_retrain_cancellation_does_not_roll_back_or_train(
    tmp_path, monkeypatch, confirmation
) -> None:
    layout = RunLayout(tmp_path / "source")

    def confirm(_prompt):
        if isinstance(confirmation, BaseException):
            raise confirmation
        return confirmation

    monkeypatch.setattr("builtins.input", confirm)
    monkeypatch.setattr(
        train_cli,
        "rollback_run_for_retrain",
        lambda *_args, **_kwargs: pytest.fail("cancelled retrain must not roll back"),
    )
    monkeypatch.setattr(
        train_cli,
        "train",
        lambda *_args, **_kwargs: pytest.fail("cancelled retrain must not train"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["train", str(layout.root), "policy", "--round-index", "1"],
    )

    train_cli.main()


def test_retrain_keyboard_interrupt_does_not_roll_back_or_train(
    tmp_path, monkeypatch
) -> None:
    layout = RunLayout(tmp_path / "source")

    def interrupt(_prompt):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", interrupt)
    monkeypatch.setattr(
        train_cli,
        "rollback_run_for_retrain",
        lambda *_args, **_kwargs: pytest.fail("interrupted retrain must not roll back"),
    )
    monkeypatch.setattr(
        train_cli,
        "train",
        lambda *_args, **_kwargs: pytest.fail("interrupted retrain must not train"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["train", str(layout.root), "policy", "--round-index", "1"],
    )

    with pytest.raises(KeyboardInterrupt):
        train_cli.main()


def test_empty_round_index_continues_existing_run(tmp_path, monkeypatch) -> None:
    source = RunLayout(tmp_path / "source")
    calls = {}
    monkeypatch.setattr(
        train_cli,
        "rollback_run_for_retrain",
        lambda *_args, **_kwargs: pytest.fail("continue mode must not roll back"),
    )
    monkeypatch.setattr(
        train_cli,
        "train",
        lambda layout, *, requested_stage: calls.update(
            layout=layout,
            requested_stage=requested_stage,
        ),
    )
    monkeypatch.setattr(sys, "argv", ["train", str(source.root), "policy"])

    train_cli.main()

    assert calls == {"layout": source, "requested_stage": "policy"}


def test_round_index_must_be_non_negative() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="non-negative"):
        train_cli._non_negative_int("-1")
