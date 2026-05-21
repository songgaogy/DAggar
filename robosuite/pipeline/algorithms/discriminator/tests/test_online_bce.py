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
from types import SimpleNamespace
from typing import Any, Sequence

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
    """Stand-in for SharedFrozenEncoder. Returns a zero context."""

    def __init__(self, context_dim: int) -> None:
        self.context_dim = int(context_dim)
        self._policy_camera_idx = [0]
        self._view_names = ["frontview_image"]
        self._original_img_size = (4, 4)

    @property
    def view_names(self) -> list[str]:
        return list(self._view_names)

    @property
    def original_img_size(self) -> tuple[int, int]:
        return self._original_img_size

    def bind_policy_cameras(self, cams: Sequence[str]) -> None:  # pragma: no cover
        return None

    @torch.no_grad()
    def encode(self, *, image_obs_raw: torch.Tensor, proprio_raw: torch.Tensor) -> torch.Tensor:
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
        # Allow any window where start + H <= len(storage) AND no episode
        # boundary in the middle. For our synthetic data we just use a
        # single contiguous episode so every aligned start is valid.
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
    # Zero-init of final Linear means first forward returns zeros.
    assert torch.allclose(out, torch.zeros(7), atol=1e-6)


# --------------------------------------------------------------------------- #
# 2. Warm start                                                                #
# --------------------------------------------------------------------------- #


def test_warm_start_copies_at_least_one_linear(tmp_path: Path) -> None:
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
    before = {k: v.detach().clone() for k, v in head.state_dict().items()}
    n = head.warm_start_from_lpb_bce_ckpt(ckpt_path)
    assert n >= 1
    after = head.state_dict()
    changed = sum(1 for k, v in before.items() if not torch.equal(v, after[k]))
    assert changed >= 1


# --------------------------------------------------------------------------- #
# 3. Score shape / no_grad                                                    #
# --------------------------------------------------------------------------- #


def test_score_shape_and_no_grad() -> None:
    cfg = DiscriminatorConfig(device="cpu", hidden=16, num_layers=1)
    encoder = _FakeEncoder(context_dim=8)
    disc = OnlineBCEDiscriminator(
        cfg, encoder=encoder, context_dim=8, action_dim=4, action_horizon=3
    )
    ctx = torch.randn(5, 8)
    ach = torch.randn(5, 3, 4)
    out = disc.score(context=ctx, action_chunk=ach)
    assert out.logit.shape == (5,)
    assert out.prob_failure.shape == (5,)
    assert out.decision.dtype == torch.bool
    assert not out.logit.requires_grad


# --------------------------------------------------------------------------- #
# 4. Update reduces loss on synthetic separable batch                          #
# --------------------------------------------------------------------------- #


def test_update_reduces_loss_on_synthetic_batch() -> None:
    torch.manual_seed(0)
    cfg = DiscriminatorConfig(device="cpu", hidden=32, num_layers=2, lr=3e-3)
    encoder = _FakeEncoder(context_dim=8)
    disc = OnlineBCEDiscriminator(
        cfg, encoder=encoder, context_dim=8, action_dim=2, action_horizon=2
    )

    n_each = 32
    expert_ctx = torch.randn(n_each, 8) + 2.0
    policy_ctx = torch.randn(n_each, 8) - 2.0
    ctx = torch.cat([expert_ctx, policy_ctx], dim=0)
    ach = torch.randn(2 * n_each, 2, 2)
    labels = torch.cat([torch.ones(n_each), torch.zeros(n_each)], dim=0)
    batch = DiscriminatorBatch(context=ctx, action_chunk=ach, label=labels)

    initial = disc.update(batch)["disc_loss"]
    for _ in range(60):
        last = disc.update(batch)
    assert last["disc_loss"] < initial * 0.7, (
        f"loss did not drop enough: initial={initial:.4f}, final={last['disc_loss']:.4f}"
    )


# --------------------------------------------------------------------------- #
# 5. Intrinsic reward sign after training                                      #
# --------------------------------------------------------------------------- #


def test_intrinsic_reward_sign() -> None:
    """After training, label=1 (failure) ctx must produce a higher logit
    than label=0 (non-failure / expert demo) ctx. The IQL boundary then
    negates this so reward is high for non-failure behavior.
    """
    torch.manual_seed(0)
    cfg = DiscriminatorConfig(device="cpu", hidden=32, num_layers=2, lr=3e-3)
    encoder = _FakeEncoder(context_dim=8)
    disc = OnlineBCEDiscriminator(
        cfg, encoder=encoder, context_dim=8, action_dim=2, action_horizon=2
    )

    n_each = 32
    failure_ctx = torch.randn(n_each, 8) + 2.0          # label=1
    non_failure_ctx = torch.randn(n_each, 8) - 2.0      # label=0
    ctx = torch.cat([failure_ctx, non_failure_ctx], dim=0)
    ach = torch.randn(2 * n_each, 2, 2)
    labels = torch.cat([torch.ones(n_each), torch.zeros(n_each)], dim=0)
    batch = DiscriminatorBatch(context=ctx, action_chunk=ach, label=labels)
    for _ in range(80):
        disc.update(batch)

    r_failure = disc.intrinsic_reward(context=failure_ctx, action_chunk=ach[:n_each])
    r_non_failure = disc.intrinsic_reward(context=non_failure_ctx, action_chunk=ach[n_each:])
    assert r_failure.mean().item() > r_non_failure.mean().item()
    assert r_failure.shape == (n_each,)


# --------------------------------------------------------------------------- #
# 6. state_dict roundtrip                                                      #
# --------------------------------------------------------------------------- #


def test_state_dict_roundtrip() -> None:
    torch.manual_seed(0)
    cfg = DiscriminatorConfig(device="cpu", hidden=16, num_layers=1, lr=1e-3)
    encoder = _FakeEncoder(context_dim=8)
    a = OnlineBCEDiscriminator(
        cfg, encoder=encoder, context_dim=8, action_dim=2, action_horizon=2
    )
    n_each = 4
    ctx = torch.cat([torch.randn(n_each, 8) + 1.0, torch.randn(n_each, 8) - 1.0], dim=0)
    ach = torch.randn(2 * n_each, 2, 2)
    labels = torch.cat([torch.ones(n_each), torch.zeros(n_each)], dim=0)
    for _ in range(5):
        a.update(DiscriminatorBatch(context=ctx, action_chunk=ach, label=labels))

    sd = a.state_dict()

    b = OnlineBCEDiscriminator(
        cfg, encoder=encoder, context_dim=8, action_dim=2, action_horizon=2
    )
    b.load_state_dict(sd, strict=True)

    fixed_ctx = torch.randn(3, 8)
    fixed_ach = torch.randn(3, 2, 2)
    out_a = a.score(context=fixed_ctx, action_chunk=fixed_ach).logit
    out_b = b.score(context=fixed_ctx, action_chunk=fixed_ach).logit
    assert torch.allclose(out_a, out_b, atol=1e-6)
    assert b._step == a._step
    assert b.threshold == pytest.approx(a.threshold)


def test_state_dict_dim_mismatch_raises() -> None:
    cfg = DiscriminatorConfig(device="cpu", hidden=16, num_layers=1)
    encoder = _FakeEncoder(context_dim=8)
    a = OnlineBCEDiscriminator(
        cfg, encoder=encoder, context_dim=8, action_dim=2, action_horizon=2
    )
    sd = a.state_dict()
    b = OnlineBCEDiscriminator(
        cfg, encoder=_FakeEncoder(9), context_dim=9, action_dim=2, action_horizon=2
    )
    with pytest.raises(ValueError, match="context_dim"):
        b.load_state_dict(sd, strict=True)


# --------------------------------------------------------------------------- #
# 7. Loss helper                                                               #
# --------------------------------------------------------------------------- #


def test_bce_loss_with_smoothing() -> None:
    logits = torch.tensor([0.0, 0.0])
    labels = torch.tensor([1.0, 0.0])
    plain = bce_with_logits_loss(logits, labels, label_smoothing=0.0).item()
    smoothed = bce_with_logits_loss(logits, labels, label_smoothing=0.1).item()
    # Smoothing leaves the symmetric-zero-logit loss unchanged in mean
    # (both halves of label values share the same logit), so compare to
    # the analytical value -log(0.5) ≈ 0.6931 instead.
    assert plain == pytest.approx(0.6931471, abs=1e-4)
    assert smoothed == pytest.approx(0.6931471, abs=1e-4)


# --------------------------------------------------------------------------- #
# 8. Replay buffer balance                                                     #
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
    # 40 transitions, alternating 4-step blocks of intervention / not.
    for i in range(40):
        block = i // H
        is_intv = (block % 2 == 0)
        base.add(_make_transition(
            proprio_dim=proprio_dim, image_size=image_size, action_dim=action_dim,
            is_intervention=is_intv, episode_index=0, episode_step=i,
        ))

    cfg = DiscriminatorConfig(
        device="cpu", batch_size=8, balance_ratio=1.0, hidden=8, num_layers=1
    )
    encoder = _FakeEncoder(context_dim=6)
    rb = DiscriminatorReplayBuffer(
        cfg, base, encoder=encoder, action_horizon=H
    )
    assert rb.ready(8)
    batch = rb.sample(8, device="cpu")
    assert batch.context.shape == (8, encoder.context_dim)
    assert batch.action_chunk.shape == (8, H, action_dim)
    assert batch.label.shape == (8,)
    # Intervention chunks → label=1 (failure); non-intervention → label=0.
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
    assert rb.num_non_failure >= pre_non_failure + 1
