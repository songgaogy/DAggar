from __future__ import annotations

import numpy as np
import torch

from robosuite.pipeline.algorithms.dipole.replay_buffer import DipoleReplayBuffer
from robosuite.pipeline.common.types import ReplayBufferConfig, Transition
from robosuite.pipeline.offline.utils.advantage import OfflineAdvantageGProvider


CAMERA = "agentview"
H = 2


def _make_buffer(n: int = 6) -> DipoleReplayBuffer:
    buffer = DipoleReplayBuffer(
        ReplayBufferConfig(capacity=32, batch_size=4),
        name="test_dipole",
        camera_names=[CAMERA],
        action_horizon=H,
        image_size=4,
    )
    for i in range(n):
        obs = {
            CAMERA: np.full((4, 4, 3), i, dtype=np.uint8),
            "state": np.array([i, i + 1, i + 2], dtype=np.float32),
        }
        next_obs = {
            CAMERA: np.full((4, 4, 3), i + 1, dtype=np.uint8),
            "state": np.array([i + 1, i + 2, i + 3], dtype=np.float32),
        }
        buffer.add(
            Transition(
                obs=obs,
                action=np.array([float(i), float(-i)], dtype=np.float32),
                reward=float(i),
                next_obs=next_obs,
                done=(i == n - 1),
                is_intervention=(i % 2 == 0),
                info={"episode_index": 0, "episode_step": i},
            )
        )
    return buffer


def test_static_cache_matches_replay_sample_without_augmentation() -> None:
    buffer = _make_buffer()
    cache = buffer.build_static_cache(pin_memory=True)
    kwargs = {
        "action_mean": np.zeros((H, 2), dtype=np.float32),
        "action_std": np.ones((H, 2), dtype=np.float32),
        "proprio_mean": np.zeros(3, dtype=np.float32),
        "proprio_std": np.ones(3, dtype=np.float32),
        "device": "cpu",
        "augment": False,
    }

    np.random.seed(123)
    live = buffer.sample(4, **kwargs)
    np.random.seed(123)
    cached = cache.sample(4, **kwargs)

    assert cached.metadata == live.metadata
    torch.testing.assert_close(cached.image_obs, live.image_obs)
    torch.testing.assert_close(cached.image_obs_raw, live.image_obs_raw)
    torch.testing.assert_close(cached.proprio, live.proprio)
    torch.testing.assert_close(cached.proprio_raw, live.proprio_raw)
    torch.testing.assert_close(cached.action_sequences, live.action_sequences)
    torch.testing.assert_close(cached.action_sequences_raw, live.action_sequences_raw)
    assert torch.equal(cached.is_intervention, live.is_intervention)


def test_static_cache_batch_supports_offline_g_provider_lookup() -> None:
    buffer = _make_buffer()
    cache = buffer.build_static_cache(pin_memory=False)
    with buffer._lock:  # noqa: SLF001 - test verifies provider's start-index contract.
        valid_starts = list(buffer._get_valid_start_indices_locked())  # noqa: SLF001
    start_to_row = {int(start): row for row, start in enumerate(valid_starts)}
    advantage_raw = torch.arange(len(valid_starts), dtype=torch.float32)
    failure_raw = torch.zeros(len(valid_starts), dtype=torch.float32)
    provider = OfflineAdvantageGProvider(
        vast_learner=None,
        discriminator=None,
        encoder=None,
        alpha=2.0,
        beta=0.0,
        advantage_raw=advantage_raw,
        failure_raw=failure_raw,
        start_to_row=start_to_row,
    )

    np.random.seed(7)
    batch = cache.sample(5, device="cpu", augment=False)
    g = provider.compute_g_for_batch(batch)
    expected = torch.tensor(
        [2.0 * float(start_to_row[int(start)]) for start in batch.metadata["start_indices"]],
        dtype=torch.float32,
    )

    torch.testing.assert_close(g, expected)
