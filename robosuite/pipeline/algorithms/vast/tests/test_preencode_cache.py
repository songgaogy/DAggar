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
import pytest
import torch

from robosuite.pipeline.algorithms.vast.common import VASTConfig
from robosuite.pipeline.algorithms.vast.replay import (
    VASTPreencodedReplayCache,
    VASTReplayBuffer,
)
from robosuite.pipeline.common.types import Transition

H = 2
CONTEXT_DIM = 5
STATE_DIM = 4
CAMERA = "agentview"
VALID_STARTS = [0, 1, 2, 3, 4]
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Replay tensor tests require CUDA and never fall back to CPU.",
)
DEVICE = "cuda:0"


class _FakeEncoder:
    """Deterministic, pure encoder (mimics a frozen encoder under no_grad).

    Distinct chunks -> distinct latents, so equivalence implies the same chunks
    were selected, not just the same shapes.
    """

    state_feature_dim = STATE_DIM
    chunk_feature_dim = CONTEXT_DIM
    view_names = [CAMERA]

    def encode_features(self, *, chunk_images, chunk_proprio, chunk_actions):
        base = chunk_actions.sum(dim=-1) + chunk_proprio.sum(dim=-1)  # (B, H)
        chunk = base.unsqueeze(-1).repeat(1, 1, CONTEXT_DIM)
        chunk = chunk + torch.arange(
            CONTEXT_DIM, device=chunk.device, dtype=chunk.dtype
        ).view(1, 1, -1)
        state = base.unsqueeze(-1).repeat(1, 1, STATE_DIM)
        state = state + torch.arange(
            STATE_DIM, device=state.device, dtype=state.dtype
        ).view(1, 1, -1)
        return state, chunk

    def encode_state_and_chunk(self, *, image_obs_raw, proprio_raw, action_chunk):
        # frame-0 fast path: identical to encode_features step 0 (base uses the
        # frame-0 action + frame-0 proprio).
        base0 = action_chunk[:, 0].sum(dim=-1) + proprio_raw.sum(dim=-1)  # (B,)
        state = base0.unsqueeze(-1).repeat(1, STATE_DIM) + torch.arange(
            STATE_DIM, device=base0.device, dtype=base0.dtype
        ).view(1, -1)
        chunk = base0.unsqueeze(-1).repeat(1, CONTEXT_DIM) + torch.arange(
            CONTEXT_DIM, device=base0.device, dtype=base0.dtype
        ).view(1, -1)
        return state, chunk

    def encode_state(self, *, image_obs_raw, proprio_raw):
        base = proprio_raw.sum(dim=-1)
        state = base.unsqueeze(-1).repeat(1, STATE_DIM)
        return state + torch.arange(
            STATE_DIM, device=state.device, dtype=state.dtype
        ).view(1, -1) * 10.0


class _FakeBase:
    """Minimal stand-in for FlowDaggerReplayBuffer used by VASTReplayBuffer."""

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
            info={"episode_index": 0, "episode_step": i, "nnpu_disc_intrinsic": -0.1 * i},
        )

    def _get_valid_start_indices_locked(self):
        return list(VALID_STARTS)

    def num_valid_sequences(self) -> int:
        return len(VALID_STARTS)

    def __len__(self) -> int:
        return len(self._storage)


def _cfg() -> VASTConfig:
    return VASTConfig(
        action_horizon=H,
        disc_reward_coef=1.0,
        output_reward_coef=1.0,
        device=DEVICE,
    )


def _assert_batches_equal(a, b) -> None:
    for name in (
        "chunk_feature",
        "v_state_feature",
        "next_v_state_feature",
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
    replay = VASTReplayBuffer(base_buffer=base, cfg=_cfg())

    # Live path: re-encode every draw.
    np.random.seed(1234)
    live = [replay.sample_step_batch(3, encoder=_FakeEncoder(), device=DEVICE) for _ in range(6)]

    # Cache path: build once (must consume no np.random), then sample.
    np.random.seed(1234)
    cache = replay.preencode_step_cache(
        encoder=_FakeEncoder(), device=DEVICE, encode_batch_size=2, cache_device=DEVICE
    )
    cached = [cache.sample_step_batch(3, device=DEVICE) for _ in range(6)]

    assert isinstance(cache, VASTPreencodedReplayCache)
    assert len(cache) == len(VALID_STARTS)
    for live_b, cache_b in zip(live, cached):
        _assert_batches_equal(live_b, cache_b)


def test_preencode_does_not_consume_rng() -> None:
    """Building the cache must not advance the np.random stream, otherwise the
    training loop's sampling would diverge from the live path."""
    base = _FakeBase(n=8)
    replay = VASTReplayBuffer(base_buffer=base, cfg=_cfg())

    np.random.seed(7)
    before = np.random.get_state()[1].copy()
    replay.preencode_step_cache(
        encoder=_FakeEncoder(), device=DEVICE, encode_batch_size=2, cache_device=DEVICE
    )
    after = np.random.get_state()[1]
    assert np.array_equal(before, after)


def test_cache_row_order_matches_valid_starts() -> None:
    """Cache row i must correspond to valid_starts[i] so that the shared
    randint draw selects the same chunk in both paths."""
    base = _FakeBase(n=8)
    replay = VASTReplayBuffer(base_buffer=base, cfg=_cfg())
    cache = replay.preencode_step_cache(
        encoder=_FakeEncoder(), device=DEVICE, encode_batch_size=2, cache_device=DEVICE
    )
    enc = _FakeEncoder()
    for i, start in enumerate(VALID_STARTS):
        seq = [base._storage[start + k] for k in range(H)]
        one = replay._build_step_batch([seq], [start], encoder=enc, device=DEVICE)
        assert torch.equal(cache.chunk_feature[i : i + 1], one.chunk_feature)
        assert torch.equal(cache.v_state_feature[i : i + 1], one.v_state_feature)
        assert torch.equal(cache.rewards[i : i + 1], one.rewards)


def test_live_replay_scores_nnpu_when_metadata_is_absent() -> None:
    class Discriminator:
        def __init__(self) -> None:
            self.features = None

        def intrinsic_reward(self, *, chunk_feature):
            self.features = chunk_feature.detach().clone()
            values = torch.tensor([-0.25, -0.5], dtype=chunk_feature.dtype)
            return values.view(1, H).expand(chunk_feature.shape[0], -1)

    base = _FakeBase(n=8)
    for transition in base._storage:
        transition.info.pop("nnpu_disc_intrinsic", None)
    replay = VASTReplayBuffer(base_buffer=base, cfg=_cfg())
    discriminator = Discriminator()

    np.random.seed(1)
    batch = replay.sample_step_batch(
        2,
        encoder=_FakeEncoder(),
        discriminator=discriminator,
        device=DEVICE,
    )

    assert discriminator.features is not None
    assert discriminator.features.shape == (2, H, CONTEXT_DIM)
    starts = batch.metadata["start_indices"]
    expected = []
    for start in starts:
        env = torch.tensor([0.1 * start, 0.1 * (start + 1)], device=DEVICE)
        disc = torch.tensor([-0.25, -0.5], device=DEVICE)
        expected.append(((env + disc) * torch.tensor([1.0, 0.99], device=DEVICE)).sum())
    torch.testing.assert_close(batch.rewards[:, 0], torch.stack(expected))
