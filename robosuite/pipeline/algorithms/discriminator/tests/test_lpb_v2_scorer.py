"""CPU-only unit tests for the LPB v2 offline scorer plumbing.

Covers LPB annotation helpers used by IQL warmup / replay and `vis_qv`.

We do NOT exercise the real `BCEBenchmarkDiscriminator` (it requires a
checkpoint + a 256x256 image bank). Instead, we test the two pieces that
the visualization path still depends on:

  - `annotate_transitions_with_lpb_scores` correctly writes
    `info["lpb_failure_score"] / lpb_margin_reward / lpb_tau` per transition,
    padding when the LPB encoder returned slightly fewer scores than transitions.
  - `_resolve_meta_json_tau` reads `bce_youden_threshold` from a JSON sibling
    of the checkpoint and falls back gracefully on missing / malformed input.
  - `aggregate_chunk_reward` applies γ-discount to a per-step margin
    sequence (the same path vis_qv uses to summarize a chunk window).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from robosuite.pipeline.algorithms.discriminator.lpb_v2_scorer import (
    _resolve_meta_json_tau,
    load_bce_youden_threshold,
    annotate_transitions_with_lpb_scores,
    lpb_disc_intrinsic_from_failure_score,
)
from robosuite.pipeline.algorithms.q_learning.data_util import aggregate_chunk_reward
from robosuite.pipeline.common import Transition


def _make_transition(*, action_dim: int = 3, info: dict[str, Any] | None = None) -> Transition:
    return Transition(
        obs={"agentview": np.zeros((128, 128, 3), dtype=np.uint8), "state": np.zeros((6,), dtype=np.float32)},
        action=np.zeros((action_dim,), dtype=np.float32),
        reward=-1.0,
        next_obs={"agentview": np.zeros((128, 128, 3), dtype=np.uint8), "state": np.zeros((6,), dtype=np.float32)},
        done=False,
        is_intervention=False,
        info=info,
    )


def test_annotate_writes_failure_score_margin_and_tau() -> None:
    transitions = [_make_transition() for _ in range(4)]
    failure = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    annotate_transitions_with_lpb_scores(transitions, failure_scores=failure, tau=2.0)
    margins = [t.info["lpb_margin_reward"] for t in transitions]
    failures = [t.info["lpb_failure_score"] for t in transitions]
    taus = [t.info["lpb_tau"] for t in transitions]
    intrinsics = [t.info["lpb_disc_intrinsic"] for t in transitions]
    assert failures == [0.0, 1.0, 2.0, 3.0]
    assert margins == [2.0, 1.0, 0.0, -1.0]
    assert taus == [2.0, 2.0, 2.0, 2.0]
    expected_intrinsics = [
        float(lpb_disc_intrinsic_from_failure_score(v, 2.0)) for v in failures
    ]
    np.testing.assert_allclose(intrinsics, expected_intrinsics, rtol=0, atol=1e-6)


def test_annotate_pads_shorter_scores_with_last_value() -> None:
    transitions = [_make_transition() for _ in range(5)]
    failure = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    annotate_transitions_with_lpb_scores(transitions, failure_scores=failure, tau=0.0)
    failures = [t.info["lpb_failure_score"] for t in transitions]
    assert failures == [1.0, 2.0, 3.0, 3.0, 3.0]  # last value repeated


def test_annotate_truncates_longer_scores() -> None:
    transitions = [_make_transition() for _ in range(2)]
    failure = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    annotate_transitions_with_lpb_scores(transitions, failure_scores=failure, tau=0.0)
    assert [t.info["lpb_failure_score"] for t in transitions] == [1.0, 2.0]


def test_resolve_meta_json_tau_reads_youden(tmp_path: Path) -> None:
    ckpt = tmp_path / "bce_head.pth"
    ckpt.write_bytes(b"placeholder")
    (tmp_path / "meta.json").write_text(json.dumps({"bce_youden_threshold": 2.0867}))
    tau, used = _resolve_meta_json_tau(ckpt, meta_json_path=None)
    assert tau == pytest.approx(2.0867)
    assert used is not None and used.name == "meta.json"


def test_load_bce_youden_threshold_required(tmp_path: Path) -> None:
    ckpt = tmp_path / "bce_head.pth"
    ckpt.write_bytes(b"placeholder")
    (tmp_path / "meta.json").write_text(json.dumps({"bce_youden_threshold": 3.8183}))
    tau, used = load_bce_youden_threshold(ckpt)
    assert tau == pytest.approx(3.8183)
    assert used.name == "meta.json"


def test_load_bce_youden_threshold_raises_when_missing(tmp_path: Path) -> None:
    ckpt = tmp_path / "bce_head.pth"
    ckpt.write_bytes(b"placeholder")
    with pytest.raises(FileNotFoundError):
        load_bce_youden_threshold(ckpt)


def test_resolve_meta_json_tau_missing_file(tmp_path: Path) -> None:
    ckpt = tmp_path / "bce_head.pth"
    ckpt.write_bytes(b"placeholder")
    tau, used = _resolve_meta_json_tau(ckpt, meta_json_path=None)
    assert tau is None
    assert used is None


def test_resolve_meta_json_tau_malformed(tmp_path: Path) -> None:
    ckpt = tmp_path / "bce_head.pth"
    ckpt.write_bytes(b"placeholder")
    (tmp_path / "meta.json").write_text("not a json {")
    tau, used = _resolve_meta_json_tau(ckpt, meta_json_path=None)
    assert tau is None


def test_aggregate_chunk_reward_applies_to_lpb_margins() -> None:
    """The replay aggregates per-step LPB margins with the same γ-discount as r_env."""
    margins = torch.tensor([[1.0, 2.0]])
    discount = 0.9
    out = aggregate_chunk_reward(margins, discount)
    expected = 1.0 + 0.9 * 2.0
    assert out.shape == (1, 1)
    assert float(out.item()) == pytest.approx(expected)
