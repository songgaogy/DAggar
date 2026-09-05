from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from robosuite.pipeline.src.data import (
    CacheDataset,
    CacheValidationError,
    build_cache_fingerprint,
    build_feature_cache,
    load_legacy_buffer,
)


def _transition(episode: int, step: int, *, done: bool, reward: float) -> dict:
    return {
        "obs": {"value": np.asarray([episode, step], dtype=np.float32)},
        "action": np.asarray([episode, step], dtype=np.float32),
        "reward": reward,
        "next_obs": {"value": np.asarray([episode, step + 1], dtype=np.float32)},
        "done": done,
        "info": {"episode_index": episode, "episode_step": step},
    }


def _write_legacy(tmp_path: Path, storage: list[dict], *, horizon: int = 2) -> tuple[Path, Path]:
    source = tmp_path / "legacy.pt"
    metadata = tmp_path / "legacy.meta.json"
    torch.save(
        {
            "storage": storage,
            "action_horizon": horizon,
            "camera_names": ["a", "b", "c"],
        },
        source,
    )
    metadata.write_text(
        json.dumps(
            {
                "task": "Task",
                "n_transitions": len(storage),
                "action_horizon": horizon,
                "camera_names": ["a", "b", "c"],
            }
        ),
        encoding="utf-8",
    )
    return source, metadata


def _features(observations) -> dict[str, np.ndarray]:
    values = np.stack([observation["value"] for observation in observations])
    return {
        "dino_cls": np.repeat(values[:, None, :], 3, axis=1),
        "proprio": values,
        "task_scene_cond": values[:, :1],
        "context_tokens": values[:, None, :],
        "context_padding_mask": np.zeros((len(values), 1), dtype=np.bool_),
    }


def _identity_actions(actions: np.ndarray) -> np.ndarray:
    return actions


def _offset_actions(actions: np.ndarray) -> np.ndarray:
    return actions + 100.0


def test_legacy_load_keeps_storage_reference_and_validates_metadata(tmp_path: Path) -> None:
    source, metadata = _write_legacy(tmp_path, [_transition(0, 0, done=True, reward=1.0)])
    loaded = load_legacy_buffer(
        source,
        metadata_path=metadata,
        expected_task="Task",
        expected_horizon=2,
        expected_cameras=["a", "b", "c"],
    )
    assert loaded.storage is loaded.payload["storage"]

    bad = json.loads(metadata.read_text())
    bad["n_transitions"] = 2
    metadata.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="transition count mismatch"):
        load_legacy_buffer(source, metadata_path=metadata)


def test_cache_windows_do_not_cross_done_and_rewards_are_undiscounted(tmp_path: Path) -> None:
    storage = [
        _transition(0, 0, done=False, reward=1.0),
        _transition(0, 1, done=False, reward=2.0),
        _transition(0, 2, done=True, reward=3.0),
        _transition(1, 0, done=False, reward=10.0),
        _transition(1, 1, done=False, reward=20.0),
        _transition(1, 2, done=False, reward=30.0),
        _transition(1, 3, done=True, reward=40.0),
    ]
    source, metadata = _write_legacy(tmp_path, storage)
    legacy = load_legacy_buffer(source, metadata_path=metadata)
    cache = build_feature_cache(
        tmp_path / "cache",
        legacy,
        fingerprint="synthetic",
        fingerprint_ingredients={"source": "synthetic"},
        action_horizon=2,
        feature_extractor=_features,
        action_normalizer=_offset_actions,
        feature_batch_size=3,
    )
    assert len(cache) == 5
    np.testing.assert_allclose(cache.arrays["rewards"][:, 0], [3.0, 5.0, 30.0, 50.0, 70.0])
    np.testing.assert_array_equal(cache.arrays["dones"][:, 0], [False, True, False, False, True])
    np.testing.assert_array_equal(cache.arrays["executed_length"][:, 0], [2, 2, 2, 2, 2])
    batch = cache.gather(np.asarray([1, 2]))
    np.testing.assert_array_equal(batch["dino_features"][:, 0], [[0, 1], [1, 0]])
    np.testing.assert_array_equal(batch["next_dino_features"][:, 0], [[0, 3], [1, 2]])
    assert isinstance(cache.arrays["actions"], np.memmap)
    assert cache.manifest["actions_normalized"] is True
    np.testing.assert_allclose(cache.arrays["actions"][0, :, 1], [100.0, 101.0])


def test_cache_is_reused_and_manifest_corruption_is_rebuilt(tmp_path: Path) -> None:
    source, metadata = _write_legacy(
        tmp_path,
        [_transition(0, 0, done=False, reward=1.0), _transition(0, 1, done=True, reward=2.0)],
    )
    legacy = load_legacy_buffer(source, metadata_path=metadata)
    calls = 0

    def extractor(observations):
        nonlocal calls
        calls += 1
        return _features(observations)

    arguments = dict(
        cache_root=tmp_path / "cache",
        legacy=legacy,
        fingerprint="stable",
        fingerprint_ingredients={},
        action_horizon=2,
        feature_extractor=extractor,
        action_normalizer=_identity_actions,
    )
    first = build_feature_cache(**arguments)
    initial_calls = calls
    second = build_feature_cache(**arguments)
    assert second.path == first.path
    assert calls == initial_calls

    (first.path / "manifest.json").write_text("{}", encoding="utf-8")
    rebuilt = build_feature_cache(**arguments)
    assert calls > initial_calls
    assert CacheDataset(rebuilt.path, expected_fingerprint="stable").manifest["transition_count"] == 1
    assert not list((tmp_path / "cache").glob(".*.tmp-*"))


def test_cache_fingerprint_tracks_all_inputs(tmp_path: Path) -> None:
    files = {}
    for name in ("source.pt", "source.meta.json", "dino.pt", "flow.pt"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        files[name] = path
    common = dict(
        source_path=files["source.pt"],
        metadata_path=files["source.meta.json"],
        dino_weights=files["dino.pt"],
        flow_checkpoint=files["flow.pt"],
        camera_names=["a", "b", "c"],
        image_size=224,
        normalizer={"mean": [0.0]},
        prompt="pick cereal",
        action_horizon=8,
        ode_steps=8,
    )
    first, ingredients = build_cache_fingerprint(**common)
    second, _ = build_cache_fingerprint(**{**common, "prompt": "changed"})
    assert first != second
    assert ingredients["schema_version"] == 1


def test_cache_rejects_wrong_expected_fingerprint(tmp_path: Path) -> None:
    path = tmp_path / "cache"
    path.mkdir()
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fingerprint": "a",
                "actions_normalized": True,
                "arrays": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(CacheValidationError, match="fingerprint"):
        CacheDataset(path, expected_fingerprint="b")
