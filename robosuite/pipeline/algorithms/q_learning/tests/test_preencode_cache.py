"""Results-neutrality test for the warmup preencode replay cache.

The cache only memoizes the frozen-encoder forward; it must produce *exactly*
the same training batches as the live `sample_step_batch` path when the
np.random stream is identical. This test asserts bit-for-bit equivalence over a
sequence of draws, plus the supporting invariants (no RNG consumed at build
time, cache row order == valid-start order).
"""

from __future__ import annotations

import threading

import numpy as np
import torch

# Import the discriminator package first to initialize it before
# q_learning.replay (they have a known circular dependency that only bites when
# replay.py is imported first; see test_replay_lpb_disc.py).
import robosuite.pipeline.algorithms.discriminator  # noqa: F401
from robosuite.pipeline.algorithms.q_learning.common import IQLConfig
from robosuite.pipeline.algorithms.q_learning.replay import (
    IQLPreencodedReplayCache,
    IQLReplayBuffer,
)
from robosuite.pipeline.common.types import Transition

H = 2
CONTEXT_DIM = 5
CAMERA = "agentview"
VALID_STARTS = [0, 1, 2, 3, 4]


class _FakeEncoder:
    """Deterministic, pure encoder (mimics a frozen encoder under no_grad).

    Distinct chunks -> distinct latents, so equivalence implies the same chunks
    were selected, not just the same shapes.
    """

    context_dim = CONTEXT_DIM
    view_names = [CAMERA]

    def encode_chunk_frames(self, *, chunk_images, chunk_proprio, chunk_actions):
        base = chunk_actions.sum(dim=-1) + chunk_proprio.sum(dim=-1)  # (B, H)
        ctx = base.unsqueeze(-1).repeat(1, 1, CONTEXT_DIM)
        return ctx + torch.arange(CONTEXT_DIM, dtype=ctx.dtype).view(1, 1, -1)

    def encode(self, *, image_obs_raw, proprio_raw, action_real):
        base = action_real.sum(dim=-1) + proprio_raw.sum(dim=-1)  # (B,)
        ctx = base.unsqueeze(-1).repeat(1, CONTEXT_DIM)
        return ctx + torch.arange(CONTEXT_DIM, dtype=ctx.dtype).view(1, -1) * 10.0


class _FakeBase:
    """Minimal stand-in for FlowDaggerReplayBuffer used by IQLReplayBuffer."""

    action_horizon = H
    camera_names = [CAMERA]
    image_size = 4

    def __init__(self, n: int) -> None:
        self._lock = threading.Lock()
        self._storage = [self._make_transition(i) for i in range(n)]

    @staticmethod
    def _make_transition(i: int) -> Transition:
        obs = {
            CAMERA: np.full((4, 4, 3), i % 7, dtype=np.uint8),
            "state": np.array([i, i + 1, i + 2], dtype=np.float32),
        }
        nxt = {
            CAMERA: np.full((4, 4, 3), (i + 1) % 7, dtype=np.uint8),
            "state": np.array([i + 1, i + 2, i + 3], dtype=np.float32),
        }
        return Transition(
            obs=obs,
            action=np.array([float(i), float(-i)], dtype=np.float32),
            reward=float(i) * 0.1,
            next_obs=nxt,
            done=False,
            info={"episode_index": 0, "episode_step": i, "lpb_disc_intrinsic": -0.1 * i},
        )

    def _get_valid_start_indices_locked(self):
        return list(VALID_STARTS)

    def num_valid_sequences(self) -> int:
        return len(VALID_STARTS)

    def __len__(self) -> int:
        return len(self._storage)


def _cfg() -> IQLConfig:
    return IQLConfig(
        action_horizon=H,
        disc_reward_coef=1.0,
        output_reward_coef=1.0,
        device="cpu",
    )


def _assert_batches_equal(a, b) -> None:
    for name in (
        "context",
        "next_context",
        "action_chunk",
        "rewards",
        "dones",
        "is_online",
        "is_intervention",
    ):
        ta, tb = getattr(a, name), getattr(b, name)
        assert torch.equal(ta, tb), f"field '{name}' differs:\n{ta}\nvs\n{tb}"


def test_cache_matches_live_path_under_identical_rng() -> None:
    base = _FakeBase(n=8)
    replay = IQLReplayBuffer(base_buffer=base, cfg=_cfg())

    # Live path: re-encode every draw.
    np.random.seed(1234)
    live = [replay.sample_step_batch(3, encoder=_FakeEncoder(), device="cpu") for _ in range(6)]

    # Cache path: build once (must consume no np.random), then sample.
    np.random.seed(1234)
    cache = replay.preencode_step_cache(
        encoder=_FakeEncoder(), device="cpu", encode_batch_size=2, cache_device="cpu"
    )
    cached = [cache.sample_step_batch(3, device="cpu") for _ in range(6)]

    assert isinstance(cache, IQLPreencodedReplayCache)
    assert len(cache) == len(VALID_STARTS)
    for live_b, cache_b in zip(live, cached):
        _assert_batches_equal(live_b, cache_b)


def test_preencode_does_not_consume_rng() -> None:
    """Building the cache must not advance the np.random stream, otherwise the
    training loop's sampling would diverge from the live path."""
    base = _FakeBase(n=8)
    replay = IQLReplayBuffer(base_buffer=base, cfg=_cfg())

    np.random.seed(7)
    before = np.random.get_state()[1].copy()
    replay.preencode_step_cache(
        encoder=_FakeEncoder(), device="cpu", encode_batch_size=2, cache_device="cpu"
    )
    after = np.random.get_state()[1]
    assert np.array_equal(before, after)


def test_cache_row_order_matches_valid_starts() -> None:
    """Cache row i must correspond to valid_starts[i] so that the shared
    randint draw selects the same chunk in both paths."""
    base = _FakeBase(n=8)
    replay = IQLReplayBuffer(base_buffer=base, cfg=_cfg())
    cache = replay.preencode_step_cache(
        encoder=_FakeEncoder(), device="cpu", encode_batch_size=2, cache_device="cpu"
    )
    enc = _FakeEncoder()
    for i, start in enumerate(VALID_STARTS):
        seq = [base._storage[start + k] for k in range(H)]
        one = replay._build_step_batch([seq], [start], encoder=enc, device="cpu")
        assert torch.equal(cache.context[i : i + 1], one.context)
        assert torch.equal(cache.rewards[i : i + 1], one.rewards)
