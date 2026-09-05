from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from robosuite.pipeline.src.data import CudaBatchPrefetcher, OnlineReplay, UniformReplay


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
        "dino_features": feature,
        "proprio": feature[:1],
        "task_scene_cond": feature,
        "context_tokens": feature[None, :],
        "context_padding_mask": np.asarray([False]),
        "next_dino_features": feature + 1,
        "next_proprio": feature[:1] + 1,
        "next_task_scene_cond": feature + 1,
        "next_context_tokens": (feature + 1)[None, :],
        "next_context_padding_mask": np.asarray([False]),
        "actions": np.full((2, 2), value, dtype=np.float32),
        "reward": value,
        "done": False,
        "executed_length": 2,
    }


def test_online_replay_ring_and_snapshot_round_trip(tmp_path: Path) -> None:
    replay = OnlineReplay(capacity=3)
    for value in range(5):
        replay.add(_compact(float(value)))
    assert len(replay) == 3
    assert replay.position == 2
    np.testing.assert_allclose(
        np.sort(replay.gather(np.asarray([0, 1, 2]))["rewards"][:, 0]), [2.0, 3.0, 4.0]
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
