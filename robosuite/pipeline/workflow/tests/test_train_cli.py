from __future__ import annotations

import argparse
import sys

import pytest

import robosuite.pipeline.train as train_cli
from robosuite.pipeline.workflow import RunLayout


def test_round_index_with_all_clones_from_discriminator(tmp_path, monkeypatch) -> None:
    source = RunLayout(tmp_path / "source")
    cloned = RunLayout(tmp_path / "cloned")
    calls = {}

    def clone(layout, *, round_index: int, stage: str):
        calls["clone"] = (layout, round_index, stage)
        return cloned

    def train(layout, *, requested_stage: str):
        calls["train"] = (layout, requested_stage)

    monkeypatch.setattr(train_cli, "clone_run_for_retrain", clone)
    monkeypatch.setattr(train_cli, "train", train)
    monkeypatch.setattr(
        sys,
        "argv",
        ["train", str(source.root), "all", "--round-index", "3"],
    )

    train_cli.main()

    assert calls["clone"] == (source, 3, "disc")
    assert calls["train"] == (cloned, "all")


def test_empty_round_index_continues_existing_run(tmp_path, monkeypatch) -> None:
    source = RunLayout(tmp_path / "source")
    calls = {}
    monkeypatch.setattr(
        train_cli,
        "clone_run_for_retrain",
        lambda *_args, **_kwargs: pytest.fail("continue mode must not clone"),
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
