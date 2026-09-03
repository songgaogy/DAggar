"""Unit tests for the nnPU (PU-BCE) discriminator.

Run with:
    python -m pytest robosuite/discriminator/dyn_disc/tests/test_pu_bce_discriminator.py -v
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pytest
import torch

from robosuite.discriminator.dyn_disc.detectors.pu_bce import (
    BCEHead,
    PUBCEDiscriminator,
    pu_risk,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Discriminator tensor tests require CUDA.",
)
CUDA = torch.device("cuda")


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _gaussian(n: int, dim: int, mean: float, std: float, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=CUDA).manual_seed(int(seed))
    return torch.randn(
        (n, dim), generator=generator, dtype=torch.float32, device=CUDA
    ) * std + mean


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Tiny AUROC (positive class label = 1)."""
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
    above = cum_pos[rank_neg_positions]
    return float(above.mean() / float(n_pos))


@dataclass
class _StubTraj:
    video_id: str
    task_name: str = "t"
    is_failure: bool = False
    num_frames: int = 1


# --------------------------------------------------------------------------- #
# Head                                                                        #
# --------------------------------------------------------------------------- #


def test_head_shapes() -> None:
    head = BCEHead(in_dim=32, hidden=16, num_layers=2).to(CUDA)
    z = torch.randn(8, 32, device=CUDA)
    g = head(z)
    assert g.shape == (8,)


# --------------------------------------------------------------------------- #
# nnPU risk: non-negative correction                                          #
# --------------------------------------------------------------------------- #


def test_nnpu_correction_triggers_and_clamps() -> None:
    """When the negative-risk term goes below -beta, nn_correction must clamp it.

    neg_risk = E_u[ell(-1,g)] - pi * E_p[ell(-1,g)] with ell(-1,g)=sigmoid(g).
    Make the unlabeled set score very 'negative' (small sigmoid(g_u)~0) and the
    positives very 'positive' (sigmoid(g_p)~1): then neg_risk ~ 0 - pi*1 < 0.
    With nn_correction the reported `neg_risk_used` is clamped to >= -beta and
    differs from the raw `neg_risk`; uPU (no correction) leaves it negative.
    """
    torch.cuda.manual_seed_all(0)
    # Positives extremely positive => ell(-1,g_p)=sigmoid(g_p) ~ 1 => big subtracted term.
    g_p = torch.full((64,), 8.0, device=CUDA)
    # Unlabeled extremely negative => ell(-1,g_u)=sigmoid(g_u) ~ 0.
    g_u = torch.full((64,), -8.0, device=CUDA)
    pi = 0.5

    parts_nn = pu_risk(g_p, g_u, pi_p=pi, surrogate="sigmoid", nn_correction=True, beta=0.0)
    parts_upu = pu_risk(g_p, g_u, pi_p=pi, surrogate="sigmoid", nn_correction=False)

    # Raw negative risk is negative here.
    assert float(parts_nn["neg_risk"]) < 0.0
    # nnPU clamps the *used* term to >= -beta (=0).
    assert float(parts_nn["neg_risk_used"]) >= 0.0 - 1e-9
    # uPU does not clamp: used == raw (still negative).
    assert float(parts_upu["neg_risk_used"]) == pytest.approx(float(parts_upu["neg_risk"]), abs=1e-6)
    assert float(parts_upu["neg_risk_used"]) < 0.0
    # The corrected risk is therefore strictly greater than the uncorrected one.
    assert float(parts_nn["risk"]) > float(parts_upu["risk"])


def test_nnpu_correction_inactive_when_positive() -> None:
    """When neg_risk >= -beta the correction is a no-op (used == raw)."""
    g_p = torch.full(
        (32,), -2.0, device=CUDA
    )  # ell(-1,g_p)=sigmoid(g_p) small => subtracted term small
    g_u = torch.full(
        (32,), 2.0, device=CUDA
    )  # ell(-1,g_u)=sigmoid(g_u) large => neg_risk positive
    parts = pu_risk(g_p, g_u, pi_p=0.5, surrogate="sigmoid", nn_correction=True, beta=0.0)
    assert float(parts["neg_risk"]) > 0.0
    assert float(parts["neg_risk_used"]) == pytest.approx(float(parts["neg_risk"]), abs=1e-6)


# --------------------------------------------------------------------------- #
# Synthetic separability                                                      #
# --------------------------------------------------------------------------- #


def test_pu_synthetic_separability() -> None:
    """Positives vs unlabeled-with-hidden-negatives should recover a useful ranking.

    P = pure positive Gaussian. U = mixture (pi_p positives + (1-pi_p) negatives).
    After nnPU training, failure_score = -g(z) should rank held-out negatives
    above held-out positives (AUROC well above chance).
    """
    in_dim = 16
    pi_true = 0.5
    n_p = 600
    n_u = 600
    n_u_pos = int(pi_true * n_u)
    n_u_neg = n_u - n_u_pos

    Z_p = _gaussian(n_p, in_dim, mean=+1.5, std=0.6, seed=1)
    U_pos = _gaussian(n_u_pos, in_dim, mean=+1.5, std=0.6, seed=2)
    U_neg = _gaussian(n_u_neg, in_dim, mean=-1.5, std=0.6, seed=3)
    Z_u = torch.cat([U_pos, U_neg], dim=0)

    Z_calib = _gaussian(200, in_dim, mean=+1.5, std=0.6, seed=4)

    det = PUBCEDiscriminator(in_dim=in_dim, hidden=32, num_layers=2, device="cuda")
    thresholds = det.fit(
        positive_features=[Z_p],
        unlabeled_features=[Z_u],
        success_calib_per_task={"t": [Z_calib]},
        pi_p=pi_true,
        epochs=12,
        lr=3e-3,
        batch_size=128,
        delta=10.0,
        seed=0,
        pin_memory=False,
        verbose=False,
    )
    assert "t" in thresholds
    assert "t" in det.calib_stats

    # Held-out positives vs negatives; failure_score = -g; label 1 == negative (failure).
    Z_pos_test = _gaussian(200, in_dim, mean=+1.5, std=0.6, seed=5)
    Z_neg_test = _gaussian(200, in_dim, mean=-1.5, std=0.6, seed=6)
    g_pos = det._logits_np(Z_pos_test)
    g_neg = det._logits_np(Z_neg_test)
    failure_scores = np.concatenate([-g_pos, -g_neg], axis=0)
    labels = np.concatenate([
        np.zeros(g_pos.size, dtype=np.int64),
        np.ones(g_neg.size, dtype=np.int64),
    ], axis=0)
    auroc = _auroc(failure_scores, labels)
    assert auroc >= 0.85, f"Expected PU synthetic AUROC >= 0.85, got {auroc}"


# --------------------------------------------------------------------------- #
# Deterministic threshold                                                     #
# --------------------------------------------------------------------------- #


def test_pu_threshold_determinism() -> None:
    in_dim = 8
    Z_p = _gaussian(200, in_dim, mean=+1.0, std=0.5, seed=10)
    U_pos = _gaussian(100, in_dim, mean=+1.0, std=0.5, seed=11)
    U_neg = _gaussian(100, in_dim, mean=-1.0, std=0.5, seed=12)
    Z_u = torch.cat([U_pos, U_neg], dim=0)
    Z_calib = _gaussian(80, in_dim, mean=+1.0, std=0.5, seed=13)

    def _run() -> float:
        det = PUBCEDiscriminator(in_dim=in_dim, hidden=16, num_layers=2, device="cuda")
        thr = det.fit(
            positive_features=[Z_p],
            unlabeled_features=[Z_u],
            success_calib_per_task={"t": [Z_calib]},
            pi_p=0.5,
            epochs=4,
            lr=3e-3,
            batch_size=64,
            delta=10.0,
            seed=42,
            pin_memory=False,
            verbose=False,
        )
        return float(thr["t"])

    tau_a = _run()
    tau_b = _run()
    assert tau_a == pytest.approx(tau_b, abs=1e-6), f"Non-deterministic threshold: {tau_a} vs {tau_b}"


# --------------------------------------------------------------------------- #
# Score output + bad-prior guard                                              #
# --------------------------------------------------------------------------- #


def test_score_emits_detection_result() -> None:
    in_dim = 8
    Z_p = _gaussian(100, in_dim, mean=+1.0, std=0.5, seed=20)
    Z_u = torch.cat([
        _gaussian(50, in_dim, mean=+1.0, std=0.5, seed=21),
        _gaussian(50, in_dim, mean=-1.0, std=0.5, seed=22),
    ], dim=0)
    Z_calib = _gaussian(40, in_dim, mean=+1.0, std=0.5, seed=23)

    det = PUBCEDiscriminator(in_dim=in_dim, hidden=16, num_layers=2, device="cuda")
    det.fit(
        positive_features=[Z_p],
        unlabeled_features=[Z_u],
        success_calib_per_task={"t": [Z_calib]},
        pi_p=0.5,
        epochs=3,
        lr=3e-3,
        batch_size=32,
        seed=0,
        pin_memory=False,
        verbose=False,
    )

    q = det.score(Z_p[:50], task="t")
    assert q.step_scores.shape == (50,)
    assert q.thresholds.shape == (50,)
    assert q.preds.shape == (50,)
    # delta=10 => roughly <= ~10% of *calib* positives exceed tau; on a fresh
    # batch of positive frames we expect well under 50% flagged.
    pos_rate = float(q.preds.sum()) / float(q.preds.size)
    assert pos_rate < 0.5, f"Positive frames flagged at high rate: {pos_rate}"

    with pytest.raises(KeyError):
        det.score(Z_p[:5], task="missing-task")


def test_invalid_prior_raises() -> None:
    in_dim = 8
    Z_p = _gaussian(40, in_dim, mean=+1.0, std=0.5, seed=70)
    Z_u = _gaussian(40, in_dim, mean=0.0, std=0.5, seed=71)
    Z_calib = _gaussian(20, in_dim, mean=+1.0, std=0.5, seed=72)
    det = PUBCEDiscriminator(in_dim=in_dim, hidden=8, num_layers=1, device="cuda")
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            det.fit(
                positive_features=[Z_p],
                unlabeled_features=[Z_u],
                success_calib_per_task={"t": [Z_calib]},
                pi_p=bad,
                epochs=1,
                seed=0,
                pin_memory=False,
                verbose=False,
            )


# --------------------------------------------------------------------------- #
# Disjointness invariant                                                      #
# --------------------------------------------------------------------------- #


def test_assert_disjoint_raises_on_overlap() -> None:
    from robosuite.discriminator.dyn_disc.adapters.pu_bce import PUBCEBenchmarkDiscriminator

    eval_trajs = [
        _StubTraj(video_id="v1", is_failure=True),
        _StubTraj(video_id="v2"),
    ]
    unlabeled = [_StubTraj(video_id="v3", is_failure=True), _StubTraj(video_id="v1", is_failure=True)]

    with pytest.raises(RuntimeError) as excinfo:
        PUBCEBenchmarkDiscriminator._assert_disjoint_unlabeled_eval_fail(
            eval_trajs,
            unlabeled_fail_trajs=unlabeled,
        )
    assert "v1" in str(excinfo.value)

    # No overlap -> should not raise.
    PUBCEBenchmarkDiscriminator._assert_disjoint_unlabeled_eval_fail(
        eval_trajs,
        unlabeled_fail_trajs=[_StubTraj(video_id="v3", is_failure=True),
                              _StubTraj(video_id="v4", is_failure=True)],
    )


# --------------------------------------------------------------------------- #
# State-dict roundtrip                                                        #
# --------------------------------------------------------------------------- #


def test_state_dict_roundtrip() -> None:
    in_dim = 8
    Z_p = _gaussian(80, in_dim, mean=+1.0, std=0.5, seed=30)
    Z_u = torch.cat([
        _gaussian(40, in_dim, mean=+1.0, std=0.5, seed=31),
        _gaussian(40, in_dim, mean=-1.0, std=0.5, seed=32),
    ], dim=0)
    Z_calib = _gaussian(30, in_dim, mean=+1.0, std=0.5, seed=33)

    det = PUBCEDiscriminator(in_dim=in_dim, hidden=16, num_layers=2, device="cuda")
    det.fit(
        positive_features=[Z_p],
        unlabeled_features=[Z_u],
        success_calib_per_task={"t": [Z_calib]},
        pi_p=0.4,
        epochs=2,
        seed=0,
        pin_memory=False,
        verbose=False,
    )
    sd = det.state_dict()
    assert sd["pi_p"] == pytest.approx(0.4, abs=1e-9)

    det2 = PUBCEDiscriminator(in_dim=in_dim, hidden=16, num_layers=2, device="cuda")
    det2.load_state_dict(sd)
    assert det2.thresholds["t"] == pytest.approx(det.thresholds["t"], abs=1e-6)
    assert det2._pi_p == pytest.approx(0.4, abs=1e-9)

    z_probe = _gaussian(20, in_dim, mean=0.0, std=1.0, seed=34)
    g_a = det._logits_np(z_probe)
    g_b = det2._logits_np(z_probe)
    np.testing.assert_allclose(g_a, g_b, atol=1e-6)
