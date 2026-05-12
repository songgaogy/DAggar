"""Unit tests for the Phase-A BCE-WAM discriminator.

Run with:
    python -m pytest robosuite/discriminator/lpb_v2/tests/test_bce_discriminator.py -v
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pytest
import torch

from robosuite.discriminator.lpb_v2.bce_discriminator import (
    BCEDiscriminator,
    BCEHead,
)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _gaussian_class(n: int, dim: int, mean: float, std: float, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn((n, dim), generator=g, dtype=torch.float32) * std + mean


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Tiny AUROC implementation (positive class label = 1)."""
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    n_pos = int((y_sorted == 1).sum())
    n_neg = int((y_sorted == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    cum_pos = np.cumsum(y_sorted == 1)
    rank_neg_positions = np.where(y_sorted == 0)[0]
    # Each negative ranked at position i contributes cum_pos[i] true-positives above it
    # (the number of positives ranked strictly higher). Mean over negatives gives the
    # probability that a random positive scores above a random negative.
    above = cum_pos[rank_neg_positions]
    return float(above.mean() / float(n_pos))


@dataclass
class _StubTraj:
    video_id: str
    task_name: str = "t"
    is_failure: bool = False
    num_frames: int = 1


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #


def test_bce_head_shapes() -> None:
    head = BCEHead(in_dim=32, hidden=16, num_layers=2)
    z = torch.randn(8, 32)
    g = head(z)
    assert g.shape == (8,)


def test_bce_fit_synthetic_auroc() -> None:
    """Two well-separated Gaussian classes -> head should hit near-perfect AUROC."""
    in_dim = 16
    Ne, No = 400, 400
    Z_e = _gaussian_class(Ne, in_dim, mean=+1.5, std=0.5, seed=1)
    Z_o = _gaussian_class(No, in_dim, mean=-1.5, std=0.5, seed=2)
    # Held-out test set
    Z_e_test = _gaussian_class(200, in_dim, mean=+1.5, std=0.5, seed=3)
    Z_o_test = _gaussian_class(200, in_dim, mean=-1.5, std=0.5, seed=4)
    # Per-task calib pool (use a slice of D_e disjoint from training set;
    # for synthetic data we just use fresh samples).
    Z_calib = _gaussian_class(200, in_dim, mean=+1.5, std=0.5, seed=5)

    det = BCEDiscriminator(in_dim=in_dim, hidden=32, num_layers=2, device="cpu")
    thresholds = det.fit(
        expert_features=[Z_e],
        other_features=[Z_o],
        expert_calib_per_task={"t": [Z_calib]},
        epochs=8,
        lr=3e-3,
        batch_size=128,
        delta=10.0,
        seed=0,
        verbose=False,
    )
    assert "t" in thresholds
    assert isinstance(thresholds["t"], float)
    assert "t" in det.calib_stats

    # AUROC on held-out: failure_score = -g(z), positive class = "other"
    g_e = det._logits_np(Z_e_test)
    g_o = det._logits_np(Z_o_test)
    failure_scores = np.concatenate([-g_e, -g_o], axis=0)
    labels = np.concatenate([
        np.zeros(g_e.size, dtype=np.int64),
        np.ones(g_o.size, dtype=np.int64),
    ], axis=0)
    auroc = _auroc(failure_scores, labels)
    assert auroc >= 0.9, f"Expected synthetic AUROC >= 0.9, got {auroc}"


def test_bce_threshold_determinism() -> None:
    """Same seed + same data -> identical per-task threshold."""
    in_dim = 8
    Z_e = _gaussian_class(200, in_dim, mean=+1.0, std=0.5, seed=10)
    Z_o = _gaussian_class(200, in_dim, mean=-1.0, std=0.5, seed=11)
    Z_calib = _gaussian_class(80, in_dim, mean=+1.0, std=0.5, seed=12)

    def _run() -> float:
        det = BCEDiscriminator(in_dim=in_dim, hidden=16, num_layers=2, device="cpu")
        thr = det.fit(
            expert_features=[Z_e],
            other_features=[Z_o],
            expert_calib_per_task={"t": [Z_calib]},
            epochs=4,
            lr=3e-3,
            batch_size=64,
            delta=10.0,
            seed=42,
            verbose=False,
        )
        return float(thr["t"])

    tau_a = _run()
    tau_b = _run()
    assert tau_a == pytest.approx(tau_b, abs=1e-6), f"Non-deterministic threshold: {tau_a} vs {tau_b}"


def test_bce_score_emits_detection_result() -> None:
    in_dim = 8
    Z_e = _gaussian_class(100, in_dim, mean=+1.0, std=0.5, seed=20)
    Z_o = _gaussian_class(100, in_dim, mean=-1.0, std=0.5, seed=21)
    Z_calib = _gaussian_class(40, in_dim, mean=+1.0, std=0.5, seed=22)

    det = BCEDiscriminator(in_dim=in_dim, hidden=16, num_layers=2, device="cpu")
    det.fit(
        expert_features=[Z_e],
        other_features=[Z_o],
        expert_calib_per_task={"t": [Z_calib]},
        epochs=2,
        lr=3e-3,
        batch_size=32,
        seed=0,
        verbose=False,
    )

    # Score expert features: most should NOT be flagged as failure.
    q = det.score(Z_e[:50], task="t")
    assert q.step_scores.shape == (50,)
    assert q.thresholds.shape == (50,)
    assert q.preds.shape == (50,)
    # The threshold was set to flag the top ~10% of calib failure_scores, so on
    # a separate batch of expert frames we expect well under 50% positive.
    pos_rate = float(q.preds.sum()) / float(q.preds.size)
    assert pos_rate < 0.5, f"Expert frames flagged at high rate: {pos_rate}"

    # Asking for an uncalibrated task should raise.
    with pytest.raises(KeyError):
        det.score(Z_e[:5], task="missing-task")


def test_assert_disjoint_raises_on_overlap() -> None:
    """The disjointness static method must raise when video_ids overlap."""
    # Import here to avoid forcing benchmark/torchvision setup for the pure-head tests.
    from robosuite.discriminator.lpb_v2.benchmark_bce import BCEBenchmarkDiscriminator

    eval_trajs = [_StubTraj(video_id="v1"), _StubTraj(video_id="v2")]
    bank_trajs = [_StubTraj(video_id="v3"), _StubTraj(video_id="v1")]  # v1 overlap!
    calib_trajs: List[_StubTraj] = []

    with pytest.raises(RuntimeError) as excinfo:
        BCEBenchmarkDiscriminator._assert_disjoint(
            eval_trajs=eval_trajs,
            fail_bank_trajs=bank_trajs,
            fail_calib_trajs=calib_trajs,
        )
    assert "v1" in str(excinfo.value)

    # No overlap -> should not raise.
    BCEBenchmarkDiscriminator._assert_disjoint(
        eval_trajs=eval_trajs,
        fail_bank_trajs=[_StubTraj(video_id="v3"), _StubTraj(video_id="v4")],
        fail_calib_trajs=calib_trajs,
    )


def test_state_dict_roundtrip() -> None:
    in_dim = 8
    Z_e = _gaussian_class(80, in_dim, mean=+1.0, std=0.5, seed=30)
    Z_o = _gaussian_class(80, in_dim, mean=-1.0, std=0.5, seed=31)
    Z_calib = _gaussian_class(30, in_dim, mean=+1.0, std=0.5, seed=32)

    det = BCEDiscriminator(in_dim=in_dim, hidden=16, num_layers=2, device="cpu")
    det.fit(
        expert_features=[Z_e],
        other_features=[Z_o],
        expert_calib_per_task={"t": [Z_calib]},
        epochs=2,
        seed=0,
        verbose=False,
    )
    sd = det.state_dict()

    det2 = BCEDiscriminator(in_dim=in_dim, hidden=16, num_layers=2, device="cpu")
    det2.load_state_dict(sd)
    assert det2.thresholds["t"] == pytest.approx(det.thresholds["t"], abs=1e-6)

    z_probe = _gaussian_class(20, in_dim, mean=0.0, std=1.0, seed=33)
    g_a = det._logits_np(z_probe)
    g_b = det2._logits_np(z_probe)
    np.testing.assert_allclose(g_a, g_b, atol=1e-6)
