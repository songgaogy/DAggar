from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from robosuite.pipeline.src.data import (
    CudaBatchPrefetcher,
    OnlineReplay,
    UniformReplay,
    WarmupReplay,
    WarmupValidationError,
    build_warmup_fingerprint,
    save_warmup_cache,
)


class ArrayReplay:
    def __init__(self, values: np.ndarray) -> None:
        self.values = np.asarray(values, dtype=np.float32).reshape(-1, 1)

    def __len__(self) -> int:
        return len(self.values)

    def gather(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        values = self.values[indices]
        return {"value": values}


def _compact(value: float) -> dict:
    feature = np.asarray([value, value + 1], dtype=np.float32)
    return {
        "visual_features": feature,
        "proprio": feature[:1],
        "next_visual_features": feature + 1,
        "next_proprio": feature[:1] + 1,
        "actions": np.full((2, 2), value, dtype=np.float32),
        "reward": float(value > 0),
        "done": False,
        "executed_length": 2,
    }


def _warmup_arrays(count: int = 4) -> dict[str, np.ndarray]:
    values = np.arange(count, dtype=np.float32)
    return {
        "visual_features": np.stack((values, values + 1), axis=1),
        "proprio": values[:, None],
        "next_visual_features": np.stack((values + 1, values + 2), axis=1),
        "next_proprio": (values + 1)[:, None],
        "actions": np.zeros((count, 2, 2), dtype=np.float32),
        "rewards": np.asarray([0, 1, 0, 1], dtype=np.float32)[:count, None],
        "dones": np.asarray([False, True, False, True], dtype=np.bool_)[:count, None],
        "executed_length": np.full((count, 1), 2, dtype=np.uint8),
    }


def test_warmup_cache_is_fingerprinted_immutable_and_strict(tmp_path: Path) -> None:
    checkpoint = tmp_path / "flow.pt"
    checkpoint.write_bytes(b"frozen flow checkpoint")
    fingerprint, ingredients = build_warmup_fingerprint(
        task_name="PickPlaceCereal",
        warmup_seed=42,
        base_checkpoint=checkpoint,
        env_metadata={"robots": "Panda"},
        camera_names=("agentview", "robot0_eye_in_hand"),
        action_horizon=2,
        ode_config={"steps": 10},
        feature_schema={"visual_features": [2], "proprio": [1]},
        reward_schema="binary_success",
    )
    replay = save_warmup_cache(
        tmp_path / "warmup",
        fingerprint=fingerprint,
        fingerprint_ingredients=ingredients,
        arrays=_warmup_arrays(),
        episode_boundaries=(0, 2, 4),
        metadata={
            "completed_by_worker": [1, 1],
            "primitive_steps": 8,
            "macro_steps": 4,
            "vector_steps": 2,
        },
    )
    assert isinstance(replay, WarmupReplay)
    assert len(replay) == 4
    assert replay.manifest["episode_count"] == 2
    assert replay.manifest["completed_by_worker"] == [1, 1]
    np.testing.assert_array_equal(
        replay.gather(np.asarray([3, 0]))["proprio"], np.asarray([[3.0], [0.0]])
    )

    replacement = _warmup_arrays()
    replacement["proprio"][:] = 99
    reused = save_warmup_cache(
        tmp_path / "warmup",
        fingerprint=fingerprint,
        fingerprint_ingredients=ingredients,
        arrays=replacement,
        episode_boundaries=(0, 2, 4),
    )
    np.testing.assert_array_equal(reused.gather(np.asarray([0]))["proprio"], [[0.0]])
    with pytest.raises(WarmupValidationError, match="fingerprint mismatch"):
        WarmupReplay(replay.path, expected_fingerprint="wrong")


def test_warmup_cache_rejects_non_binary_rewards(tmp_path: Path) -> None:
    arrays = _warmup_arrays()
    arrays["rewards"][0, 0] = 0.5
    with pytest.raises(WarmupValidationError, match="binary"):
        save_warmup_cache(
            tmp_path,
            fingerprint=hashlib.sha256(b"{}").hexdigest(),
            fingerprint_ingredients={},
            arrays=arrays,
            episode_boundaries=(0, 2, 4),
        )


def test_online_replay_ring_and_snapshot_round_trip(tmp_path: Path) -> None:
    replay = OnlineReplay(capacity=3)
    for value in range(5):
        replay.add(_compact(float(value)))
    assert len(replay) == 3
    assert replay.position == 2
    np.testing.assert_allclose(
        np.sort(replay.gather(np.asarray([0, 1, 2]))["visual_features"][:, 0]),
        [2.0, 3.0, 4.0],
    )

    manifest = replay.snapshot(tmp_path / "online")
    restored = OnlineReplay.load_snapshot(tmp_path / "online")
    assert manifest["position"] == restored.position == 2
    assert restored.capacity == 3
    for key, value in replay.gather(np.asarray([0, 1, 2])).items():
        np.testing.assert_array_equal(value, restored.gather(np.asarray([0, 1, 2]))[key])


def test_uniform_replay_samples_union_by_transition_count() -> None:
    offline = ArrayReplay(np.zeros(3))
    online = ArrayReplay(np.ones(1))
    replay = UniformReplay(offline, online, seed=7)  # type: ignore[arg-type]
    samples = replay.sample(20_000)["value"][:, 0]
    online_fraction = float(samples.sum()) / float(len(samples))
    assert online_fraction == pytest.approx(0.25, abs=0.02)


def test_uniform_replay_rng_state_restores_exact_batch_sequence() -> None:
    offline = ArrayReplay(np.arange(8))
    online = OnlineReplay(2)
    replay = UniformReplay(offline, online, seed=42)  # type: ignore[arg-type]
    replay.sample(3)
    state = replay.state_dict()
    expected = replay.sample(16)

    restored = UniformReplay(offline, online, seed=999)  # type: ignore[arg-type]
    restored.load_state_dict(state)
    actual = restored.sample(16)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])


def test_cuda_prefetch_is_pinned_and_nonblocking() -> None:
    if not torch.cuda.is_available():
        pytest.fail("CUDA is required by the DSRL test suite.")
    replay = UniformReplay(ArrayReplay(np.arange(8)), OnlineReplay(2), seed=1)  # type: ignore[arg-type]
    with CudaBatchPrefetcher(replay, batch_size=4, device="cuda:0") as prefetcher:
        batch = prefetcher.next()
    assert batch["value"].is_cuda
    assert batch["value"].device == torch.device("cuda:0")


def test_closed_prefetcher_rng_state_restarts_at_same_batch() -> None:
    replay = UniformReplay(ArrayReplay(np.arange(32)), OnlineReplay(2), seed=5)  # type: ignore[arg-type]
    first = CudaBatchPrefetcher(replay, batch_size=8, device="cuda:0", depth=2)
    first.close()
    state = replay.state_dict()
    expected = replay.sample(8)["value"]
    replay.load_state_dict(state)
    with CudaBatchPrefetcher(replay, batch_size=8, device="cuda:0", depth=2) as restarted:
        actual = restarted.next()["value"].cpu().numpy()
    np.testing.assert_array_equal(actual, expected)
