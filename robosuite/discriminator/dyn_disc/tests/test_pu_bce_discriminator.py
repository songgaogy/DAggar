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
    soft_logit_cap_penalty,
)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.fail("CUDA is required for dyn_disc tensor tests")
    return torch.device("cuda")


def test_requested_cuda_unavailable_does_not_fall_back_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        PUBCEDiscriminator(in_dim=4, hidden=8, num_layers=1, device="cuda")


def _gaussian(n: int, dim: int, mean: float, std: float, seed: int) -> torch.Tensor:
    device = _cuda_device()
    g = torch.Generator(device=device).manual_seed(int(seed))
    return torch.randn(
        (n, dim), generator=g, dtype=torch.float32, device=device,
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
    device = _cuda_device()
    head = BCEHead(in_dim=32, hidden=16, num_layers=2).to(device)
    z = torch.randn(8, 32, device=device)
    g = head(z)
    assert g.shape == (8,)
    assert g.is_cuda


def test_head_raw_and_effective_logits_use_detached_center() -> None:
    device = _cuda_device()
    head = BCEHead(in_dim=8, hidden=16, num_layers=2).to(device)
    z = torch.randn(12, 8, device=device, requires_grad=True)

    raw_before = head.raw_forward(z)
    center_source = raw_before.mean() * 0.0 + 1.25
    head.set_logit_center(center_source)

    raw_after = head.raw_forward(z)
    effective = head(z)
    torch.testing.assert_close(raw_after, raw_before)
    torch.testing.assert_close(effective, raw_after - 1.25)
    assert head.logit_center.is_cuda
    assert not head.logit_center.requires_grad
    assert head.logit_center.grad_fn is None


def test_soft_logit_cap_penalty_matches_formula_and_has_linear_tail_gradient() -> None:
    device = _cuda_device()
    logits = torch.tensor(
        [-100.0, -8.0, 0.0, 8.0, 100.0],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    cap = 5.0
    temperature = 1.0

    penalty = soft_logit_cap_penalty(logits, cap=cap, temperature=temperature)
    expected = (
        temperature
        * torch.nn.functional.softplus((logits.abs() - cap) / temperature)
    ).mean()
    torch.testing.assert_close(penalty, expected)

    penalty.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad[0]) == pytest.approx(-1.0 / logits.numel(), abs=1e-6)
    assert float(logits.grad[-1]) == pytest.approx(1.0 / logits.numel(), abs=1e-6)


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
    device = _cuda_device()
    torch.cuda.manual_seed_all(0)
    # Positives extremely positive => ell(-1,g_p)=sigmoid(g_p) ~ 1 => big subtracted term.
    g_p = torch.full((64,), 8.0, device=device)
    # Unlabeled extremely negative => ell(-1,g_u)=sigmoid(g_u) ~ 0.
    g_u = torch.full((64,), -8.0, device=device)
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
    device = _cuda_device()
    g_p = torch.full((32,), -2.0, device=device)  # ell(-1,g_p)=sigmoid(g_p) small
    g_u = torch.full((32,), 2.0, device=device)   # ell(-1,g_u)=sigmoid(g_u) large
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
            verbose=False,
        )
        return float(thr["t"])

    tau_a = _run()
    tau_b = _run()
    assert tau_a == pytest.approx(tau_b, abs=1e-6), f"Non-deterministic threshold: {tau_a} vs {tau_b}"


def test_default_fit_retains_sweep_scheduler_and_health_history() -> None:
    in_dim = 8
    Z_p = _gaussian(128, in_dim, mean=+1.0, std=0.5, seed=80)
    Z_u = torch.cat([
        _gaussian(64, in_dim, mean=+1.0, std=0.5, seed=81),
        _gaussian(64, in_dim, mean=-1.0, std=0.5, seed=82),
    ])
    Z_calib = _gaussian(60, in_dim, mean=+1.0, std=0.5, seed=83)

    det = PUBCEDiscriminator(in_dim=in_dim, device="cuda")
    thresholds = det.fit(
        positive_features=[Z_p],
        unlabeled_features=[Z_u],
        success_calib_per_task={"t": [Z_calib]},
        seed=0,
        verbose=False,
    )

    assert "t" in thresholds
    assert det.hidden == 512
    assert det.num_layers == 3
    assert det._pi_p == pytest.approx(0.3)
    assert det._surrogate == "logistic"
    assert det._threshold_normalization == "none"
    assert det._soft_cap_c == pytest.approx(5.0)
    assert det._soft_cap_lambda == pytest.approx(1e-2)
    assert det._soft_cap_temperature == pytest.approx(1.0)
    assert det._scheduler_horizon_epochs == 20
    assert det._completed_epochs == 1
    assert len(det._train_history) == 1
    expected_lr = 3e-4 * (1.0 + np.cos(np.pi / 20.0)) / 2.0
    assert det._train_history[-1]["lr"] == pytest.approx(expected_lr)
    assert det._train_history[-1]["scheduler_horizon_epochs"] == 20
    assert det._train_history[-1]["logit_center"] == pytest.approx(0.0)
    assert det._train_history[-1]["soft_cap_penalty"] > 0.0
    assert set(det._train_history[-1]["pools"]) == {
        "train_positive",
        "unlabeled_failure",
        "success_calib",
    }


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
                verbose=False,
            )


# --------------------------------------------------------------------------- #
# Disjointness invariant                                                      #
# --------------------------------------------------------------------------- #


def test_assert_disjoint_raises_on_overlap() -> None:
    from robosuite.discriminator.dyn_disc.adapters.pu_bce import PUBCEBenchmarkDiscriminator

    eval_trajs = [_StubTraj(video_id="v1"), _StubTraj(video_id="v2")]
    unlabeled = [_StubTraj(video_id="v3", is_failure=True), _StubTraj(video_id="v1", is_failure=True)]

    with pytest.raises(RuntimeError) as excinfo:
        PUBCEBenchmarkDiscriminator._assert_disjoint(
            eval_trajs=eval_trajs,
            unlabeled_fail_trajs=unlabeled,
        )
    assert "v1" in str(excinfo.value)

    # No overlap -> should not raise.
    PUBCEBenchmarkDiscriminator._assert_disjoint(
        eval_trajs=eval_trajs,
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
        verbose=False,
    )
    sd = det.state_dict()
    assert sd["pi_p"] == pytest.approx(0.4, abs=1e-9)
    assert sd["scheduler_horizon_epochs"] == 20

    det.head.set_logit_center(1.75)
    sd = det.state_dict()
    assert float(sd["head"]["logit_center"]) == pytest.approx(1.75, abs=1e-6)

    det2 = PUBCEDiscriminator(in_dim=in_dim, hidden=16, num_layers=2, device="cuda")
    det2.load_state_dict(sd)
    assert det2.thresholds["t"] == pytest.approx(det.thresholds["t"], abs=1e-6)
    assert det2._pi_p == pytest.approx(0.4, abs=1e-9)
    assert det2._scheduler_horizon_epochs == 20

    z_probe = _gaussian(20, in_dim, mean=0.0, std=1.0, seed=34)
    g_a = det._logits_np(z_probe)
    g_b = det2._logits_np(z_probe)
    np.testing.assert_allclose(g_a, g_b, atol=1e-6)


def test_legacy_state_without_logit_center_loads_with_zero_center() -> None:
    device = _cuda_device()
    in_dim = 8
    source = PUBCEDiscriminator(in_dim=in_dim, hidden=16, num_layers=2, device="cuda")
    z_probe = _gaussian(20, in_dim, mean=0.0, std=1.0, seed=40)
    expected_raw = source.head.raw_forward(z_probe).detach().clone()

    legacy_state = source.state_dict()
    legacy_head_state = dict(legacy_state["head"])
    legacy_head_state.pop("logit_center")
    legacy_state["head"] = legacy_head_state

    restored = PUBCEDiscriminator(in_dim=in_dim, hidden=16, num_layers=2, device="cuda")
    restored.head.set_logit_center(torch.tensor(9.0, device=device))
    restored.load_state_dict(legacy_state)

    assert float(restored.head.logit_center) == pytest.approx(0.0, abs=1e-9)
    torch.testing.assert_close(restored.head(z_probe), expected_raw)
