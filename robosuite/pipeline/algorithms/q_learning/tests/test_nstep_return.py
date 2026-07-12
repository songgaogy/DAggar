"""Chain-walk correctness tests for the n-step (multi chunk-macro-step) value
target produced by :meth:`IQLReplayBuffer._build_step_batch`.

CPU-only, deterministic fake encoder + hand-built transition storage. Covers:
    - value_n_step=1 reduces EXACTLY to the legacy 1-step fields
      (nstep_rewards==rewards, nstep_bootstrap_feature==next_v_state_feature,
      nstep_dones==dones, nstep_discount==γ^H).
    - value_n_step=3 accumulates Σ_{k<n_eff} (γ^H)^k·R_k and bootstraps from the
      correct s_{+n_eff} state (brute-force reference).
    - variable n_eff truncation at the episode boundary.
    - a genuine success terminal stops the chain and masks the bootstrap.
"""

from __future__ import annotations

import threading

import numpy as np
import torch

from robosuite.pipeline.algorithms.q_learning.common import IQLConfig
from robosuite.pipeline.algorithms.q_learning.replay import IQLReplayBuffer
from robosuite.pipeline.common.types import Transition

H = 2
STATE_DIM = 4
CONTEXT_DIM = 5
CAMERA = "agentview"
GAMMA = 0.9


class _FakeEncoder:
    """Deterministic pure encoder; state feature is a function of proprio, so a
    bootstrap feature uniquely identifies the storage frame it came from."""

    state_feature_dim = STATE_DIM
    chunk_feature_dim = CONTEXT_DIM
    view_names = [CAMERA]

    def encode_state_and_chunk(self, *, image_obs_raw, proprio_raw, action_chunk):
        state = self.encode_state(image_obs_raw=image_obs_raw, proprio_raw=proprio_raw)
        chunk = state.sum(dim=-1, keepdim=True).repeat(1, CONTEXT_DIM)
        return state, chunk

    def encode_state(self, *, image_obs_raw, proprio_raw):
        base = proprio_raw.sum(dim=-1)  # (B,)
        return base.unsqueeze(-1).repeat(1, STATE_DIM) + torch.arange(
            STATE_DIM, dtype=base.dtype
        ).view(1, -1) * 10.0


class _FakeBase:
    """Minimal FlowDaggerReplayBuffer stand-in with explicit episode layout."""

    action_horizon = H
    camera_names = [CAMERA]
    image_size = 4

    def __init__(self, episode_ids, rewards, successes) -> None:
        assert len(episode_ids) == len(rewards) == len(successes)
        self._lock = threading.Lock()
        self._storage = [
            self._make_transition(i, episode_ids[i], rewards[i], successes[i])
            for i in range(len(episode_ids))
        ]

    @staticmethod
    def _make_transition(i, ep, reward, success) -> Transition:
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
            reward=float(reward),
            next_obs=nxt,
            done=bool(success),
            info={"episode_index": int(ep), "episode_step": i, "success": bool(success)},
        )

    def _get_valid_start_indices_locked(self):
        # Not used directly by _build_step_batch tests; return a permissive set.
        return list(range(len(self._storage)))

    def __len__(self) -> int:
        return len(self._storage)


def _cfg(value_n_step: int) -> IQLConfig:
    return IQLConfig(
        action_horizon=H,
        discount=GAMMA,
        value_n_step=value_n_step,
        disc_reward_coef=0.0,
        output_reward_coef=1.0,
        device="cpu",
    )


def _chunk_reward(rewards, chunk_start):
    """Brute-force R_k = Σ_{i<H} γ^i·r(chunk_start+i)."""
    return sum(GAMMA ** i * rewards[chunk_start + i] for i in range(H))


def _feature_of_state(state_vec):
    base = float(np.asarray(state_vec, dtype=np.float32).sum())
    return base + torch.arange(STATE_DIM, dtype=torch.float32) * 10.0


def test_nstep_one_reduces_to_one_step() -> None:
    # Single 8-frame episode; every field must match its 1-step counterpart.
    n_frames = 8
    base = _FakeBase([0] * n_frames, [i + 1 for i in range(n_frames)], [False] * n_frames)
    replay = IQLReplayBuffer(base_buffer=base, cfg=_cfg(value_n_step=1))
    starts = [0, 1, 2, 3, 4, 5]
    seqs = [base._storage[s : s + H] for s in starts]
    batch = replay._build_step_batch(seqs, starts, encoder=_FakeEncoder(), device="cpu")

    assert torch.allclose(batch.nstep_rewards, batch.rewards)
    assert torch.allclose(batch.nstep_bootstrap_feature, batch.next_v_state_feature)
    assert torch.allclose(batch.nstep_dones, batch.dones)
    assert torch.allclose(batch.nstep_discount, torch.full_like(batch.rewards, GAMMA ** H))


def test_nstep_three_bruteforce_interior() -> None:
    # Single 10-frame episode; start=0 walks 3 full chunks (0,2,4).
    n_frames = 10
    rewards = [float(i + 1) for i in range(n_frames)]
    base = _FakeBase([0] * n_frames, rewards, [False] * n_frames)
    replay = IQLReplayBuffer(base_buffer=base, cfg=_cfg(value_n_step=3))
    starts = [0]
    seqs = [base._storage[s : s + H] for s in starts]
    batch = replay._build_step_batch(seqs, starts, encoder=_FakeEncoder(), device="cpu")

    gamma_h = GAMMA ** H
    expected = (
        _chunk_reward(rewards, 0)
        + gamma_h * _chunk_reward(rewards, 2)
        + gamma_h ** 2 * _chunk_reward(rewards, 4)
    )
    assert np.isclose(float(batch.nstep_rewards[0]), expected, rtol=1e-6)
    # n_eff = 3 -> discount γ^{3H}.
    assert np.isclose(float(batch.nstep_discount[0]), gamma_h ** 3, rtol=1e-6)
    assert float(batch.nstep_dones[0]) == 0.0
    # Bootstrap state is s_{+3H} = frame index 6 (chunk starting at 4 -> next 6).
    assert torch.allclose(batch.nstep_bootstrap_feature[0], _feature_of_state([6, 7, 8]))


def test_nstep_truncates_at_episode_boundary() -> None:
    # 10-frame episode; start=6 can only take chunks at 6 and 8 (10 is OOB).
    n_frames = 10
    rewards = [float(i + 1) for i in range(n_frames)]
    base = _FakeBase([0] * n_frames, rewards, [False] * n_frames)
    replay = IQLReplayBuffer(base_buffer=base, cfg=_cfg(value_n_step=3))
    starts = [6]
    seqs = [base._storage[s : s + H] for s in starts]
    batch = replay._build_step_batch(seqs, starts, encoder=_FakeEncoder(), device="cpu")

    gamma_h = GAMMA ** H
    expected = _chunk_reward(rewards, 6) + gamma_h * _chunk_reward(rewards, 8)
    assert np.isclose(float(batch.nstep_rewards[0]), expected, rtol=1e-6)
    # n_eff = 2 (truncated), not 3.
    assert np.isclose(float(batch.nstep_discount[0]), gamma_h ** 2, rtol=1e-6)
    assert float(batch.nstep_dones[0]) == 0.0
    # Truncation bootstrap = last chunk's forced next_obs = storage[9].next_obs.
    assert torch.allclose(batch.nstep_bootstrap_feature[0], _feature_of_state([10, 11, 12]))


def test_nstep_success_terminal_stops_and_masks_bootstrap() -> None:
    # 10-frame episode with success at frame 3; a chain reaching it stops there.
    n_frames = 10
    rewards = [float(i + 1) for i in range(n_frames)]
    successes = [False] * n_frames
    successes[3] = True
    base = _FakeBase([0] * n_frames, rewards, successes)
    replay = IQLReplayBuffer(base_buffer=base, cfg=_cfg(value_n_step=3))
    # start=0: chunk0=(0,1) no success, chunk1=(2,3) hits success -> n_eff=2, done.
    starts = [0]
    seqs = [base._storage[s : s + H] for s in starts]
    batch = replay._build_step_batch(seqs, starts, encoder=_FakeEncoder(), device="cpu")

    gamma_h = GAMMA ** H
    expected = _chunk_reward(rewards, 0) + gamma_h * _chunk_reward(rewards, 2)
    assert np.isclose(float(batch.nstep_rewards[0]), expected, rtol=1e-6)
    assert np.isclose(float(batch.nstep_discount[0]), gamma_h ** 2, rtol=1e-6)
    assert float(batch.nstep_dones[0]) == 1.0
