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
from robosuite.pipeline.algorithms.vast.data_util import (
    clone_with_absorbing_success_tail,
    clone_with_vast_absorbing_tails,
    mask_absorbing_tail_rewards,
)
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
        "future_v_state_feature",
        "intermediate_v_state_feature",
        "k",
        "j",
        "k_step_returns",
        "mc_mask",
        "future_dones",
    ):
        ta, tb = getattr(a, name), getattr(b, name)
        assert ta is not None and tb is not None
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


def test_success_metadata_distinguishes_truncation_from_terminal() -> None:
    base = _FakeBase(n=8)
    base._storage[-1].done = True
    for transition in base._storage[-H:]:
        transition.info = {**(transition.info or {}), "success": False}
    replay = VASTReplayBuffer(base_buffer=base, cfg=_cfg())
    sequence = [base._storage[-H:]]

    truncated = replay._build_step_batch(
        sequence,
        [len(base._storage) - H],
        encoder=_FakeEncoder(),
        device=DEVICE,
    )
    assert truncated.dones.item() == 0.0
    assert replay._raw_chunk_done_locked(len(base._storage) - H) is False

    base._storage[-1].info = {**(base._storage[-1].info or {}), "success": True}
    terminal = replay._build_step_batch(
        sequence,
        [len(base._storage) - H],
        encoder=_FakeEncoder(),
        device=DEVICE,
    )
    assert terminal.dones.item() == 1.0
    assert replay._raw_chunk_done_locked(len(base._storage) - H) is True


def test_absorbing_success_tail_masks_env_and_live_disc_rewards() -> None:
    horizon = 4
    base = _FakeBase(n=5)
    base.action_horizon = horizon
    for step, transition in enumerate(base._storage):
        transition.reward = 0.0 if step == 4 else -1.0
        transition.done = step == 4
        transition.info = {
            "episode_index": 0,
            "episode_step": step,
            "success": step == 4,
        }
    expanded, synthetic_count, padded_count = clone_with_absorbing_success_tail(
        base._storage,
        horizon,
    )
    base._storage = expanded

    class ConstantDiscriminator:
        @staticmethod
        def intrinsic_reward(*, chunk_feature):
            return torch.full(
                chunk_feature.shape[:2],
                -2.0,
                device=chunk_feature.device,
                dtype=chunk_feature.dtype,
            )

    cfg = VASTConfig(
        action_horizon=horizon,
        discount=0.5,
        output_reward_coef=1.0,
        disc_reward_coef=0.5,
        vast_max_k=1,
        device=DEVICE,
    )
    replay = VASTReplayBuffer(base_buffer=base, cfg=cfg)
    starts = list(range(5))
    sequences = [base._storage[start : start + horizon] for start in starts]
    discriminator = ConstantDiscriminator()
    batch = replay._build_step_batch(
        sequences,
        starts,
        encoder=_FakeEncoder(),
        discriminator=discriminator,
        device=DEVICE,
    )

    assert synthetic_count == horizon - 1
    assert padded_count == 1
    torch.testing.assert_close(
        batch.dones[:, 0],
        torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0], device=DEVICE),
    )
    torch.testing.assert_close(
        batch.rewards[:, 0],
        torch.tensor([-3.75, -3.625, -3.25, -2.5, -1.0], device=DEVICE),
    )

    for transition in base._storage:
        info = dict(transition.info or {})
        info["nnpu_disc_intrinsic"] = -2.0
        transition.info = info
    cached_batch = replay._build_step_batch(
        sequences,
        starts,
        encoder=_FakeEncoder(),
        device=DEVICE,
    )
    torch.testing.assert_close(cached_batch.rewards, batch.rewards)
    cache = replay.preencode_step_cache(
        encoder=_FakeEncoder(),
        device=DEVICE,
        encode_batch_size=5,
        cache_device=DEVICE,
    )
    torch.testing.assert_close(cache.rewards, batch.rewards)

    masked = mask_absorbing_tail_rewards(
        torch.tensor([[-3.0, -3.0]], device=DEVICE),
        torch.tensor([[1.0, 1.0]], device=DEVICE),
        torch.tensor([[1.0, 1.0]], device=DEVICE),
    )
    torch.testing.assert_close(masked, torch.zeros_like(masked))


def test_absorbing_failure_tail_pads_env_and_disc_rewards() -> None:
    horizon = 2
    base = _FakeBase(n=3)
    base.action_horizon = horizon
    for step, transition in enumerate(base._storage):
        transition.reward = -1.0
        transition.done = step == 2
        transition.info = {
            "episode_index": 0,
            "episode_step": step,
            "success": False,
            "policy_section_end_reason": "ended_human_intervention",
        }
    expanded, _, _, synthetic_count, padded_count = (
        clone_with_vast_absorbing_tails(base._storage, horizon)
    )
    base._storage = expanded

    class ConstantDiscriminator:
        @staticmethod
        def intrinsic_reward(*, chunk_feature):
            return torch.full(
                chunk_feature.shape[:2],
                -2.0,
                device=chunk_feature.device,
                dtype=chunk_feature.dtype,
            )

    cfg = VASTConfig(
        action_horizon=horizon,
        discount=0.5,
        output_reward_coef=1.0,
        disc_reward_coef=0.5,
        vast_max_k=1,
        device=DEVICE,
    )
    replay = VASTReplayBuffer(base_buffer=base, cfg=cfg)
    starts = [1, 2, 3]
    sequences = [base._storage[start : start + horizon] for start in starts]
    batch = replay._build_step_batch(
        sequences,
        starts,
        encoder=_FakeEncoder(),
        discriminator=ConstantDiscriminator(),
        device=DEVICE,
    )

    assert synthetic_count == horizon
    assert padded_count == 1
    torch.testing.assert_close(batch.dones, torch.zeros_like(batch.dones))
    torch.testing.assert_close(
        batch.rewards[:, 0],
        torch.full((3,), -3.0, device=DEVICE),
    )
    failure_value = (-1.0 + cfg.disc_reward_coef * -2.0) / (
        1.0 - cfg.discount
    )
    boundary_target = batch.rewards[-1, 0] + (cfg.discount**horizon) * failure_value
    assert boundary_target.item() == pytest.approx(failure_value)
    assert torch.equal(
        batch.next_v_state_feature[1],
        batch.next_v_state_feature[2],
    )

    for transition in base._storage:
        info = dict(transition.info or {})
        info["nnpu_disc_intrinsic"] = -2.0
        transition.info = info
    cached = replay._build_step_batch(
        sequences,
        starts,
        encoder=_FakeEncoder(),
        device=DEVICE,
    )
    torch.testing.assert_close(cached.rewards, batch.rewards)

    base._get_valid_start_indices_locked = lambda: [0, 1, 2, 3]
    cache = replay.preencode_step_cache(
        encoder=_FakeEncoder(),
        device=DEVICE,
        encode_batch_size=4,
        cache_device=DEVICE,
    )
    torch.testing.assert_close(
        cache.rewards[1:, 0],
        torch.full((3,), -3.0, device=DEVICE),
    )
