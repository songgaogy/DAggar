"""Tests for the optional uint8 HDF5 image output path."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from robosuite.discriminator.dyn_disc.data.hdf5_dynamics_dataset import (
    HDF5DynamicsModelDataset,
)


def _write_demo(path: Path) -> None:
    images = np.arange(6 * 2 * 2 * 3, dtype=np.uint8).reshape(6, 2, 2, 3)
    with h5py.File(path, "w") as handle:
        demo = handle.create_group("demos/demo_0")
        demo.create_dataset("states", data=np.zeros((6, 4), dtype=np.float32))
        demo.create_dataset("actions", data=np.zeros((6, 7), dtype=np.float32))
        view = demo.create_group("observations/agentview")
        view.create_dataset("images", data=images)
        eye_view = demo.create_group("observations/robot0_eye_in_hand")
        eye_view.create_dataset("images", data=images + np.uint8(80))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_uint8_output_preserves_pixels_and_uses_separate_cache(tmp_path: Path) -> None:
    hdf5_path = tmp_path / "demo.hdf5"
    cache_dir = tmp_path / "cache"
    _write_demo(hdf5_path)
    common = {
        "zarr_path": str(hdf5_path),
        "frameskip": 2,
        "original_img_size": 2,
        "cropped_img_size": 2,
        "use_cache": True,
        "cache_dir": str(cache_dir),
    }

    float_dataset = HDF5DynamicsModelDataset(**common)
    uint8_dataset = HDF5DynamicsModelDataset(**common, return_uint8_images=True)

    float_images = float_dataset[0][0]["visual"]["agentview"]
    uint8_images = uint8_dataset[0][0]["visual"]["agentview"]
    assert float_images.dtype == torch.float32
    assert uint8_images.dtype == torch.uint8
    assert float_dataset.cache_path != uint8_dataset.cache_path

    device = torch.device("cuda")
    expected = uint8_images.to(device=device).float().div_(255.0)
    actual = float_images.to(device=device)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-7)
    torch.testing.assert_close(
        actual.mul(255.0).round().to(torch.uint8),
        uint8_images.to(device=device),
        rtol=0,
        atol=0,
    )

    cached_uint8_dataset = HDF5DynamicsModelDataset(
        **common,
        return_uint8_images=True,
    )
    assert cached_uint8_dataset.return_uint8_images is True
    assert cached_uint8_dataset[0][0]["visual"]["agentview"].dtype == torch.uint8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_per_view_frame_counts_omit_unused_future_frame(tmp_path: Path) -> None:
    hdf5_path = tmp_path / "demo.hdf5"
    cache_dir = tmp_path / "cache"
    _write_demo(hdf5_path)
    common = {
        "zarr_path": str(hdf5_path),
        "frameskip": 2,
        "view_names": ["agentview", "robot0_eye_in_hand"],
        "original_img_size": 2,
        "cropped_img_size": 2,
        "return_uint8_images": True,
        "use_cache": True,
        "cache_dir": str(cache_dir),
    }

    default_dataset = HDF5DynamicsModelDataset(**common)
    selective_dataset = HDF5DynamicsModelDataset(
        **common,
        view_frame_counts={"agentview": 2, "robot0_eye_in_hand": 1},
    )
    default_visual = default_dataset[0][0]["visual"]
    selective_visual = selective_dataset[0][0]["visual"]

    assert default_visual["agentview"].shape[0] == 2
    assert default_visual["robot0_eye_in_hand"].shape[0] == 2
    assert selective_visual["agentview"].shape[0] == 2
    assert selective_visual["robot0_eye_in_hand"].shape[0] == 1
    assert default_dataset.cache_path != selective_dataset.cache_path

    device = torch.device("cuda")
    torch.testing.assert_close(
        selective_visual["agentview"].to(device),
        default_visual["agentview"].to(device),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        selective_visual["robot0_eye_in_hand"].to(device),
        default_visual["robot0_eye_in_hand"][:1].to(device),
        rtol=0,
        atol=0,
    )

    cached_dataset = HDF5DynamicsModelDataset(
        **common,
        view_frame_counts={"agentview": 2, "robot0_eye_in_hand": 1},
    )
    assert cached_dataset.view_frame_counts == {
        "agentview": 2,
        "robot0_eye_in_hand": 1,
    }
