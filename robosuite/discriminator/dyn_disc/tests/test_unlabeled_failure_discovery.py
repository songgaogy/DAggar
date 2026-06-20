"""Tests for mask-optional PU unlabeled failure discovery."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robosuite.discriminator.utils.robosuite_benchmark.loader import (  # noqa: E402
    discover_failure_bank,
    discover_unlabeled_failures,
)


DATA_ROOT = REPO_ROOT / "data"
TASK = "PickPlaceCereal"


@pytest.mark.skipif(not (DATA_ROOT / TASK).is_dir(), reason="PickPlaceCereal data missing")
def test_discover_unlabeled_failures_from_fail_rollout() -> None:
    trajs = discover_unlabeled_failures(
        data_root=str(DATA_ROOT),
        tasks=[TASK],
        split="fail_rollout",
    )
    assert len(trajs) > 0
    assert all(t.is_failure for t in trajs)
    assert all(len(t.failure_segments) == 0 for t in trajs)


@pytest.mark.skipif(not (DATA_ROOT / TASK).is_dir(), reason="PickPlaceCereal data missing")
def test_discover_failure_bank_still_requires_mask() -> None:
    raw = discover_unlabeled_failures(
        data_root=str(DATA_ROOT),
        tasks=[TASK],
        split="fail_rollout",
    )
    labeled = discover_failure_bank(
        data_root=str(DATA_ROOT),
        tasks=[TASK],
        split="fail_rollout-labeled",
    )
    assert len(raw) > len(labeled) or len(raw) >= 50
    assert len(labeled) > 0
