"""Tests for bounded robosuite trajectory input loading."""

from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from robosuite.discriminator.dyn_disc.adapters.single_bank import (
    _load_trajectory_inputs,
    _TrajectoryInputsDataset,
)
from robosuite.discriminator.utils.robosuite_benchmark.trajectory import (
    RobosuiteBenchmarkTrajectory,
)


def _write_demo(path) -> None:
    with h5py.File(path, "w") as handle:
        demo = handle.create_group("demos/demo_000001")
        observations = demo.create_group("observations")
        for camera, offset in (("agentview", 0), ("robot0_eye_in_hand", 100)):
            camera_group = observations.create_group(camera)
            values = np.arange(6 * 4 * 4 * 3, dtype=np.uint8).reshape(6, 4, 4, 3)
            camera_group.create_dataset("images", data=values + offset)
        demo.create_dataset("states", data=np.arange(24, dtype=np.float64).reshape(6, 4))
        demo.create_dataset("actions", data=np.arange(42, dtype=np.float32).reshape(6, 7))


def _trajectory(path) -> RobosuiteBenchmarkTrajectory:
    return RobosuiteBenchmarkTrajectory(
        task_name="PickPlaceCereal",
        num_frames=6,
        is_failure=False,
        video_id="demo_000001",
        file_path=str(path),
        demo_path="demos/demo_000001",
        split="success_rollout",
        available_cameras=("agentview", "robot0_eye_in_hand"),
    )


def test_load_model_inputs_uses_one_open_and_slices_prefix(tmp_path, monkeypatch) -> None:
    path = tmp_path / "demo.hdf5"
    _write_demo(path)
    trajectory = _trajectory(path)

    real_file = h5py.File
    open_count = 0

    def counting_file(*args, **kwargs):
        nonlocal open_count
        open_count += 1
        return real_file(*args, **kwargs)

    monkeypatch.setattr(
        "robosuite.discriminator.utils.robosuite_benchmark.trajectory.h5py.File",
        counting_file,
    )
    loaded = trajectory.load_model_inputs(
        ("agentview", "robot0_eye_in_hand"),
        frame_end=3,
        action_frame_end=5,
    )

    assert open_count == 1
    assert loaded["images"]["agentview"].shape == (3, 4, 4, 3)
    assert loaded["images"]["robot0_eye_in_hand"].shape == (3, 4, 4, 3)
    assert loaded["images"]["agentview"].dtype == np.uint8
    assert loaded["states"].shape == (3, 4)
    assert loaded["actions"].shape == (5, 7)


def test_load_model_inputs_without_prefix_reads_full_demo(tmp_path) -> None:
    path = tmp_path / "demo.hdf5"
    _write_demo(path)
    loaded = _trajectory(path).load_model_inputs(("agentview",))

    assert loaded["images"]["agentview"].shape[0] == 6
    assert loaded["states"].shape[0] == 6
    assert loaded["actions"].shape[0] == 6


def test_preloader_rejects_nonpositive_frame_end(tmp_path) -> None:
    path = tmp_path / "demo.hdf5"
    _write_demo(path)
    with pytest.raises(ValueError, match="frame_end must be positive"):
        _load_trajectory_inputs(
            _trajectory(path),
            ("agentview",),
            frame_end=0,
            action_lookahead=1,
        )


def test_spawned_preloader_preserves_order_and_pins_variable_prefixes(tmp_path) -> None:
    assert torch.cuda.is_available(), "CUDA is required for pinned-memory verification"
    path = tmp_path / "demo.hdf5"
    _write_demo(path)
    trajectory = _trajectory(path)
    dataset = _TrajectoryInputsDataset(
        [(0, trajectory, 2), (1, trajectory, 4), (2, trajectory, None)],
        ("agentview", "robot0_eye_in_hand"),
        action_lookahead=1,
    )
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=2,
        multiprocessing_context="spawn",
        prefetch_factor=1,
        pin_memory=True,
    )

    loaded = list(loader)
    assert [item["output_index"] for item in loaded] == [0, 1, 2]
    assert [item["images"]["agentview"].shape[0] for item in loaded] == [2, 4, 6]
    assert [item["actions"].shape[0] for item in loaded] == [3, 5, 6]
    assert all(item["images"]["agentview"].dtype == torch.uint8 for item in loaded)
    assert all(item["images"]["agentview"].is_pinned() for item in loaded)
