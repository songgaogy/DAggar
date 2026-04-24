"""Tests for the legacy preprocessed cache reader and dataset scan."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from robosuite.discriminator.d4disc.data import LatentFlowDynamicsDatasetD4, PreprocessedCacheReader


def _write_demo_hdf5(path: Path, lengths: dict[str, int], *, action_dim: int = 2, state_dim: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        demos = handle.create_group("demos")
        for demo_key, length in lengths.items():
            demo = demos.create_group(demo_key)
            demo.create_dataset("states", data=np.zeros((length, state_dim), dtype=np.float32))
            demo.create_dataset("actions", data=np.zeros((length, action_dim), dtype=np.float32))


def _write_cache(
    reader: PreprocessedCacheReader,
    *,
    task: str,
    file_path: Path,
    demo_key: str,
    images: np.ndarray,
    proprio: np.ndarray,
    actions: np.ndarray,
) -> None:
    cache_path = Path(reader.cache_path(task, str(file_path), demo_key))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, images_chw=images, proprio=proprio, actions=actions)


def test_legacy_cache_key_matches_reference_recipe(tmp_path: Path) -> None:
    file_path = tmp_path / "data" / "PandaLift" / "expert" / "demo.hdf5"
    _write_demo_hdf5(file_path, {"demo_0": 4})

    reader = PreprocessedCacheReader(cache_root=str(tmp_path / "cache"), image_size=128, camera_index=0)
    global_token = "|".join(
        [
            "lpb_score_preprocessed_v2",
            "128",
            "agentview,robot0_robotview,robot0_eye_in_hand",
            "d42f3c06cffef9a91d1bdbe7cca70e2329adba99",
            "3620749c11ed8f06d3eef4d02ba27ccdd3b732ec",
        ]
    )
    key = "|".join(
        [
            global_token,
            "PandaLift",
            "bfb38787161ae540355ab5d20c5c7ba75c890dbe",
            str(file_path.resolve()),
            "demo_0",
            str(os.path.getmtime(file_path)),
        ]
    )
    expected = hashlib.sha1(key.encode("utf-8")).hexdigest()
    assert reader.cache_key("PandaLift", str(file_path), "demo_0") == expected


def test_cache_reader_slices_camera_zero_and_preserves_dtypes(tmp_path: Path) -> None:
    file_path = tmp_path / "data" / "PandaLift" / "expert" / "demo.hdf5"
    _write_demo_hdf5(file_path, {"demo_0": 4})
    reader = PreprocessedCacheReader(cache_root=str(tmp_path / "cache"), image_size=128, camera_index=0)

    images = np.zeros((4, 3, 3, 128, 128), dtype=np.uint8)
    images[:, 0] = 11
    images[:, 1] = 22
    images[:, 2] = 33
    proprio = np.arange(16, dtype=np.float64).reshape(4, 4)
    actions = np.arange(8, dtype=np.float64).reshape(4, 2)
    _write_cache(
        reader,
        task="PandaLift",
        file_path=file_path,
        demo_key="demo_0",
        images=images,
        proprio=proprio,
        actions=actions,
    )

    demo = reader.load("PandaLift", str(file_path), "demo_0")
    assert demo.images_chw.shape == (4, 3, 128, 128)
    assert demo.images_chw.dtype == np.uint8
    assert np.all(demo.images_chw == 11)
    assert demo.proprio.dtype == np.float32
    assert demo.actions.dtype == np.float32


def test_cache_reader_raises_on_missing_cache(tmp_path: Path) -> None:
    file_path = tmp_path / "data" / "PandaLift" / "expert" / "demo.hdf5"
    _write_demo_hdf5(file_path, {"demo_0": 4})
    reader = PreprocessedCacheReader(cache_root=str(tmp_path / "cache"), image_size=128, camera_index=0)
    with pytest.raises(FileNotFoundError):
        reader.load("PandaLift", str(file_path), "demo_0")


def test_dataset_filters_uncached_demos_up_front(tmp_path: Path) -> None:
    file_path = tmp_path / "data" / "PandaLift" / "success_rollout" / "demo.hdf5"
    _write_demo_hdf5(file_path, {"demo_0": 4, "demo_1": 4})
    reader = PreprocessedCacheReader(cache_root=str(tmp_path / "cache"), image_size=128, camera_index=0)

    images = np.zeros((4, 3, 3, 128, 128), dtype=np.uint8)
    proprio = np.zeros((4, 4), dtype=np.float32)
    actions = np.zeros((4, 2), dtype=np.float32)
    _write_cache(
        reader,
        task="PandaLift",
        file_path=file_path,
        demo_key="demo_0",
        images=images,
        proprio=proprio,
        actions=actions,
    )

    dataset = LatentFlowDynamicsDatasetD4(
        cache_reader=reader,
        rollout_paths=[str(file_path.parent)],
        horizon=1,
        image_size=128,
        max_success_rollout_trajectories=10_000,
    )
    assert len(dataset) == 3
    assert dataset.num_rollout_trajectories == 1
    assert dataset.num_rollout_samples == 3
    assert dataset.preload_preprocessed() == 1
    sample = dataset[0]
    assert sample["current_image"].dtype == torch.uint8


def test_dataset_success_rollout_cap_is_per_kind_not_shared_with_expert(tmp_path: Path) -> None:
    # Two cached success demos; cap success_rollout to 1 trajectory => only the first
    # lexicographic demo key should contribute transitions.
    succ_path = tmp_path / "data" / "PandaLift" / "success_rollout" / "demo.hdf5"
    _write_demo_hdf5(succ_path, {"demo_a": 4, "demo_b": 4})

    # Two cached expert demos; expert cap is unlimited (0) even though success cap is 1.
    expert_path = tmp_path / "data" / "PandaLift" / "expert" / "demo.hdf5"
    _write_demo_hdf5(expert_path, {"demo_x": 4, "demo_y": 4})

    reader = PreprocessedCacheReader(cache_root=str(tmp_path / "cache"), image_size=128, camera_index=0)
    images = np.zeros((4, 3, 3, 128, 128), dtype=np.uint8)
    proprio = np.zeros((4, 4), dtype=np.float32)
    actions = np.zeros((4, 2), dtype=np.float32)

    for fp, key in (
        (succ_path, "demo_a"),
        (succ_path, "demo_b"),
        (expert_path, "demo_x"),
        (expert_path, "demo_y"),
    ):
        _write_cache(reader, task="PandaLift", file_path=fp, demo_key=key, images=images, proprio=proprio, actions=actions)

    dataset = LatentFlowDynamicsDatasetD4(
        cache_reader=reader,
        expert_paths=[str(expert_path.parent)],
        rollout_paths=[str(succ_path.parent)],
        horizon=1,
        image_size=128,
        max_expert_trajectories=10_000,
        max_success_rollout_trajectories=1,
        max_fail_rollout_trajectories=10_000,
    )
    assert dataset.num_expert_trajectories == 2
    assert dataset.num_rollout_trajectories == 1
    assert len(dataset) == 9  # 2 experts * 3 + 1 success * 3
