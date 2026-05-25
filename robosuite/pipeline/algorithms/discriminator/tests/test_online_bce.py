"""CPU-only smoke tests for OnlineBCEDiscriminator and friends.

These tests stand up tiny fakes for SharedFrozenEncoder and a base
replay buffer so the discriminator can be exercised without CUDA or the
heavy LPB v2 pipeline. The single warm-start test reaches for a real
lpb_v2 BCE checkpoint and skips cleanly if none is available.
"""

from __future__ import annotations

import glob
import os
import threading
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest
import torch

from robosuite.pipeline.algorithms.discriminator.base import DiscriminatorBatch
from robosuite.pipeline.algorithms.discriminator.bce_head import TrainableBCEHead
from robosuite.pipeline.algorithms.discriminator.losses import bce_with_logits_loss
from robosuite.pipeline.algorithms.discriminator.online_bce import (
    DiscriminatorConfig,
    OnlineBCEDiscriminator,
)
from robosuite.pipeline.algorithms.discriminator.replay import (
    DiscriminatorReplayBuffer,
)
from robosuite.pipeline.common.types import Transition

REPO_ROOT = Path(__file__).resolve().parents[5]
CKPT_GLOBS = [
    str(REPO_ROOT / "checkpoints" / "lpb_v2" / "bce_viz_robosuite"
        / "*" / "checkpoints" / "bce_head.pth"),
    str(REPO_ROOT / "checkpoints" / "lpb_v2" / "bce_eval_robosuite"
        / "run_*" / "checkpoints" / "bce_head.pth"),
]


def _resolve_bce_ckpt() -> str | None:
    env = os.environ.get("DIPOLE_BCE_CKPT")
    if env and Path(env).exists():
        return env
    matches: list[str] = []
    for pattern in CKPT_GLOBS:
        matches.extend(glob.glob(pattern))
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


class _FakeEncoder:
    """Stand-in for SharedFrozenEncoder.

    Returns a context derived from proprio (and action_real, if supplied)
    so tests can verify that the encoder receives the right inputs.
    """

    def __init__(self, context_dim: int) -> None:
        self.context_dim = int(context_dim)
        self._policy_camera_idx = [0]
        self._view_names = ["frontview_image"]
        self._original_img_size = (4, 4)
        self.device = "cpu"

    @property
    def view_names(self) -> list[str]:
        return list(self._view_names)

    @property
    def original_img_size(self) -> tuple[int, int]:
        return self._original_img_size

    def bind_policy_cameras(self, cams: Sequence[str]) -> None:  # pragma: no cover
        return None

    @torch.no_grad()
    def encode(
        self,
        *,
        image_obs_raw: torch.Tensor,
        proprio_raw: torch.Tensor,
        action_real: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = int(image_obs_raw.shape[0])
        return torch.zeros(B, self.context_dim)


class _FakeBaseBuffer:
    """Minimal FlowDaggerReplayBuffer surface used by DiscriminatorReplayBuffer."""

    def __init__(self, camera_names: list[str], image_size: int, action_horizon: int) -> None:
        self.camera_names = list(camera_names)
        self.image_size = int(image_size)
        self.action_horizon = int(action_horizon)
        self._storage: list[Transition] = []
        self._lock = threading.RLock()

    def add(self, transition: Transition) -> None:
        with self._lock:
            self._storage.append(transition)

    def _get_valid_start_indices_locked(self) -> list[int]:
        H = self.action_horizon
        return [i for i in range(len(self._storage) - H + 1)]


def _make_transition(
    *,
    proprio_dim: int,
    image_size: int,
    action_dim: int,
    is_intervention: bool,
    episode_index: int = 0,
    episode_step: int = 0,
) -> Transition:
    obs = {
        "frontview_image": np.random.randint(0, 255, size=(image_size, image_size, 3), dtype=np.uint8),
        "state": np.random.randn(proprio_dim).astype(np.float32),
    }
    next_obs = dict(obs)
    return Transition(
        obs=obs,
        action=np.random.randn(action_dim).astype(np.float32),
        reward=0.0,
        next_obs=next_obs,
        done=False,
        is_intervention=is_intervention,
        info={"episode_index": episode_index, "episode_step": episode_step},
    )


# --------------------------------------------------------------------------- #
# 1. Head forward shape                                                       #
# --------------------------------------------------------------------------- #


def test_trainable_head_forward_shape() -> None:
    head = TrainableBCEHead(in_dim=24, hidden=32, num_layers=2)
    out = head(torch.randn(7, 24))
    assert out.shape == (7,)
    assert torch.allclose(out, torch.zeros(7), atol=1e-6)


# --------------------------------------------------------------------------- #
# 2. Warm start                                                                #
# --------------------------------------------------------------------------- #


def test_initial_threshold_from_config() -> None:
    cfg = DiscriminatorConfig(
        device="cpu",
        hidden=16,
        num_layers=1,
        initial_threshold=2.0867,
        warm_start_ckpt=None,
    )
    encoder = _FakeEncoder(context_dim=4)
    disc = OnlineBCEDiscriminator(cfg, encoder=encoder, context_dim=4, action_dim=2)
    assert disc.threshold == pytest.approx(2.0867)
    assert disc.threshold_source == "config.initial_threshold"


def test_warm_start_loads_first_linear(tmp_path: Path) -> None:
    """With in_dim == ckpt['in_dim'] (single-frame head), warm-start must
    load EVERY layer including `net.0.weight`. The new strict assertion
    inside `warm_start_from_lpb_bce_ckpt` enforces this so we never fall
    back silently to a Kaiming-random first projection."""
    ckpt_path = _resolve_bce_ckpt()
    if ckpt_path is None:
        pytest.skip("No lpb_v2 BCE checkpoint available for warm-start test.")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    in_dim_ckpt = int(ckpt.get("in_dim", 0))
    hidden = int(ckpt.get("hidden", 256))
    num_layers = int(ckpt.get("num_layers", 2))
    if in_dim_ckpt <= 0:
        pytest.skip("ckpt missing 'in_dim'; can't size head to match.")
    head = TrainableBCEHead(in_dim=in_dim_ckpt, hidden=hidden, num_layers=num_layers)
    before_first = head.state_dict()["net.0.weight"].detach().clone()
    n = head.warm_start_from_lpb_bce_ckpt(ckpt_path)
    assert n >= 1
    after_first = head.state_dict()["net.0.weight"]
    # The first Linear must have been updated by warm-start.
    assert not torch.equal(before_first, after_first), (
        "warm_start did not change net.0.weight; the single-frame head is "
        "supposed to load this layer verbatim from the lpb v2 ckpt."
    )
    # Exact match against the ckpt's stored weight.
    ckpt_first = ckpt["bce_detector"]["head"]["net.0.weight"]
    assert torch.allclose(after_first, ckpt_first.to(after_first.dtype), atol=0)


# --------------------------------------------------------------------------- #
# 3. Score shape / no_grad                                                    #
# --------------------------------------------------------------------------- #


def test_score_shape_and_no_grad() -> None:
    cfg = DiscriminatorConfig(device="cpu", hidden=16, num_layers=1)
    encoder = _FakeEncoder(context_dim=8)
    disc = OnlineBCEDiscriminator(cfg, encoder=encoder, context_dim=8, action_dim=4)
    ctx = torch.randn(5, 8)
    out = disc.score(context=ctx)
    assert out.logit.shape == (5,)
    assert out.prob_failure.shape == (5,)
    assert out.decision.dtype == torch.bool
    assert not out.logit.requires_grad


def test_featurize_rejects_wrong_context_dim() -> None:
    cfg = DiscriminatorConfig(device="cpu", hidden=16, num_layers=1)
    encoder = _FakeEncoder(context_dim=8)
    disc = OnlineBCEDiscriminator(cfg, encoder=encoder, context_dim=8, action_dim=2)
    with pytest.raises(ValueError, match="context must be"):
        disc.score(context=torch.randn(3, 7))


# --------------------------------------------------------------------------- #
# 4. Update reduces loss on synthetic separable batch                          #
# --------------------------------------------------------------------------- #


def test_update_reduces_loss_on_synthetic_batch() -> None:
    torch.manual_seed(0)
    cfg = DiscriminatorConfig(device="cpu", hidden=32, num_layers=2, lr=3e-3)
    encoder = _FakeEncoder(context_dim=8)
    disc = OnlineBCEDiscriminator(cfg, encoder=encoder, context_dim=8, action_dim=2)

    n_each = 32
    expert_ctx = torch.randn(n_each, 8) + 2.0
    policy_ctx = torch.randn(n_each, 8) - 2.0
    ctx = torch.cat([expert_ctx, policy_ctx], dim=0)
    labels = torch.cat([torch.ones(n_each), torch.zeros(n_each)], dim=0)
    batch = DiscriminatorBatch(context=ctx, label=labels)

    initial = disc.update(batch)["disc_loss"]
    for _ in range(60):
        last = disc.update(batch)
    assert last["disc_loss"] < initial * 0.7, (
        f"loss did not drop enough: initial={initial:.4f}, final={last['disc_loss']:.4f}"
    )


# --------------------------------------------------------------------------- #
# 5. Intrinsic reward sign / formula                                           #
# --------------------------------------------------------------------------- #


def test_intrinsic_reward_sign() -> None:
    """`intrinsic_reward` = -sigmoid(failure_score - tau) ∈ (-1, 0).

    Train with LPB-aligned targets (1 = expert). Failure-like ctx should
    receive a lower (more negative) reward than expert-like ctx.
    """
    torch.manual_seed(0)
    cfg = DiscriminatorConfig(device="cpu", hidden=32, num_layers=2, lr=3e-3)
    encoder = _FakeEncoder(context_dim=8)
    disc = OnlineBCEDiscriminator(cfg, encoder=encoder, context_dim=8, action_dim=2)

    n_each = 32
    expert_ctx = torch.randn(n_each, 8) + 2.0           # expert target 1
    failure_ctx = torch.randn(n_each, 8) - 2.0          # expert target 0
    ctx = torch.cat([expert_ctx, failure_ctx], dim=0)
    # Replay-style labels: 1 = failure (maps to expert_target = 0).
    labels = torch.cat([torch.zeros(n_each), torch.ones(n_each)], dim=0)
    batch = DiscriminatorBatch(context=ctx, label=labels)
    for _ in range(80):
        disc.update(batch)

    r_failure = disc.intrinsic_reward(context=failure_ctx)
    r_non_failure = disc.intrinsic_reward(context=expert_ctx)
    assert r_failure.shape == (n_each,)
    assert r_failure.min().item() >= -1.0 and r_failure.max().item() <= 0.0
    assert r_non_failure.min().item() >= -1.0 and r_non_failure.max().item() <= 0.0
    assert r_failure.mean().item() < r_non_failure.mean().item()


def test_intrinsic_reward_formula_matches_sigmoid_threshold() -> None:
    """`intrinsic_reward(ctx)` must equal `-sigmoid(failure_score - tau)` with
    ``failure_score = -head(ctx)``.
    """
    torch.manual_seed(7)
    cfg = DiscriminatorConfig(device="cpu", hidden=16, num_layers=2)
    encoder = _FakeEncoder(context_dim=4)
    disc = OnlineBCEDiscriminator(cfg, encoder=encoder, context_dim=4, action_dim=2)
    disc.threshold = 0.75

    ctx = torch.randn(6, 4)
    r = disc.intrinsic_reward(context=ctx)

    with torch.no_grad():
        head_in = disc._featurize(ctx)
        expert_logit = disc.head(head_in)
        failure_score = -expert_logit
        expected = -torch.sigmoid(failure_score - 0.75)
    assert torch.allclose(r, expected, atol=1e-6)
    assert r.shape == (6,)


# --------------------------------------------------------------------------- #
# 6. state_dict roundtrip                                                      #
# --------------------------------------------------------------------------- #


def test_state_dict_roundtrip() -> None:
    torch.manual_seed(0)
    cfg = DiscriminatorConfig(device="cpu", hidden=16, num_layers=1, lr=1e-3)
    encoder = _FakeEncoder(context_dim=8)
    a = OnlineBCEDiscriminator(cfg, encoder=encoder, context_dim=8, action_dim=2)
    n_each = 4
    ctx = torch.cat([torch.randn(n_each, 8) + 1.0, torch.randn(n_each, 8) - 1.0], dim=0)
    labels = torch.cat([torch.ones(n_each), torch.zeros(n_each)], dim=0)
    for _ in range(5):
        a.update(DiscriminatorBatch(context=ctx, label=labels))

    sd = a.state_dict()

    b = OnlineBCEDiscriminator(cfg, encoder=encoder, context_dim=8, action_dim=2)
    b.load_state_dict(sd, strict=True)

    fixed_ctx = torch.randn(3, 8)
    out_a = a.score(context=fixed_ctx).logit
    out_b = b.score(context=fixed_ctx).logit
    assert torch.allclose(out_a, out_b, atol=1e-6)
    assert b._step == a._step
    assert b.threshold == pytest.approx(a.threshold)


def test_state_dict_dim_mismatch_raises() -> None:
    cfg = DiscriminatorConfig(device="cpu", hidden=16, num_layers=1)
    encoder = _FakeEncoder(context_dim=8)
    a = OnlineBCEDiscriminator(cfg, encoder=encoder, context_dim=8, action_dim=2)
    sd = a.state_dict()
    b = OnlineBCEDiscriminator(
        cfg, encoder=_FakeEncoder(9), context_dim=9, action_dim=2
    )
    with pytest.raises(ValueError, match="context_dim"):
        b.load_state_dict(sd, strict=True)


def test_state_dict_ignores_legacy_action_horizon() -> None:
    """Older checkpoints carried `action_horizon`; the single-frame
    discriminator just ignores it on load (no hard-assert)."""
    cfg = DiscriminatorConfig(device="cpu", hidden=16, num_layers=1)
    encoder = _FakeEncoder(context_dim=8)
    a = OnlineBCEDiscriminator(cfg, encoder=encoder, context_dim=8, action_dim=2)
    sd = a.state_dict()
    sd["action_horizon"] = 4  # legacy field
    b = OnlineBCEDiscriminator(cfg, encoder=encoder, context_dim=8, action_dim=2)
    b.load_state_dict(sd, strict=True)


# --------------------------------------------------------------------------- #
# 7. Loss helper                                                               #
# --------------------------------------------------------------------------- #


def test_bce_loss_with_smoothing() -> None:
    logits = torch.tensor([0.0, 0.0])
    labels = torch.tensor([1.0, 0.0])
    plain = bce_with_logits_loss(logits, labels, label_smoothing=0.0).item()
    smoothed = bce_with_logits_loss(logits, labels, label_smoothing=0.1).item()
    assert plain == pytest.approx(0.6931471, abs=1e-4)
    assert smoothed == pytest.approx(0.6931471, abs=1e-4)


# --------------------------------------------------------------------------- #
# 8. Replay buffer balance (per-frame)                                         #
# --------------------------------------------------------------------------- #


def test_replay_buffer_balance() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    H = 4
    image_size = 4
    proprio_dim = 3
    action_dim = 2
    base = _FakeBaseBuffer(
        camera_names=["frontview_image"], image_size=image_size, action_horizon=H
    )
    # 40 transitions: alternating per-frame intervention so the buffer has
    # exactly 20 failure and 20 non-failure frames.
    for i in range(40):
        is_intv = (i % 2 == 0)
        base.add(_make_transition(
            proprio_dim=proprio_dim, image_size=image_size, action_dim=action_dim,
            is_intervention=is_intv, episode_index=0, episode_step=i,
        ))

    cfg = DiscriminatorConfig(
        device="cpu", batch_size=8, balance_ratio=1.0, hidden=8, num_layers=1
    )
    encoder = _FakeEncoder(context_dim=6)
    rb = DiscriminatorReplayBuffer(cfg, base, encoder=encoder)
    assert rb.ready(8)
    batch = rb.sample(8, device="cpu")
    assert batch.context.shape == (8, encoder.context_dim)
    assert batch.label.shape == (8,)
    # Per-frame label: half from failure pool, half from non-failure pool.
    n_failure = int((batch.label > 0.5).sum().item())
    n_non_failure = int((batch.label < 0.5).sum().item())
    assert n_failure == 4
    assert n_non_failure == 4

    # bootstrap_from_demos: demos are NON-intervention, so they grow the
    # non-failure pool (label=0), matching the lpb_v2 failure-detector
    # warm-start convention.
    demos = [
        _make_transition(
            proprio_dim=proprio_dim, image_size=image_size, action_dim=action_dim,
            is_intervention=False, episode_index=1, episode_step=i,
        )
        for i in range(H)
    ]
    pre_non_failure = rb.num_non_failure
    rb.bootstrap_from_demos(demos)
    assert rb.num_non_failure == pre_non_failure + H
