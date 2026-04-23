"""Checkpoint compatibility tests for the refactored D4 benchmark adapter."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from data.utils.benchmark import BenchmarkTrajectory
from robosuite.discriminator.d3disc.detector import DetectionResult
from robosuite.discriminator.d4disc.inference.benchmark import D4BenchmarkDiscriminator
from robosuite.discriminator.d4disc.models.dynamics import ConditionalDynamicsPredictor
from robosuite.discriminator.d4disc.models.encoder import Encoder


def test_benchmark_loads_new_512d_checkpoint(tmp_path: Path) -> None:
    encoder = Encoder(pretrained=False, freeze=True)
    predictor = ConditionalDynamicsPredictor(
        latent_dim=512,
        proprio_dim=4,
        action_dim=2,
        d_model=64,
        num_layers=1,
        nhead=4,
        dropout=0.0,
        max_action_horizon=4,
        d_cond=16,
    )
    ckpt_path = tmp_path / "d4_refactored.pt"
    torch.save(
        {
            "encoder_state_dict": encoder.state_dict(),
            "encoder_config": {"normalize_input": True, "freeze": True},
            "arch_args": predictor.arch_args(),
            "ema_state_dict": predictor.state_dict(),
            "config": {"horizon": 1, "proprio_indices": [0, 1, 2, 3]},
            "sigma_sq": 0.5,
        },
        ckpt_path,
    )

    discriminator = D4BenchmarkDiscriminator(
        d4_ckpt_path=str(ckpt_path),
        preprocessed_cache_root=str(tmp_path / "cache"),
        device="cpu",
        encoder_pretrained=False,
        encoder_freeze=True,
    )
    assert discriminator.arch_args["latent_dim"] == 512
    discriminator.close()


def test_benchmark_uses_aggregated_lambda_for_predictions(tmp_path: Path) -> None:
    encoder = Encoder(pretrained=False, freeze=True)
    predictor = ConditionalDynamicsPredictor(
        latent_dim=512,
        proprio_dim=4,
        action_dim=2,
        d_model=64,
        num_layers=1,
        nhead=4,
        dropout=0.0,
        max_action_horizon=4,
        d_cond=16,
    )
    ckpt_path = tmp_path / "d4_refactored.pt"
    torch.save(
        {
            "encoder_state_dict": encoder.state_dict(),
            "encoder_config": {"normalize_input": True, "freeze": True},
            "arch_args": predictor.arch_args(),
            "ema_state_dict": predictor.state_dict(),
            "config": {"horizon": 1, "proprio_indices": [0, 1, 2, 3]},
            "sigma_sq": 0.5,
        },
        ckpt_path,
    )

    discriminator = D4BenchmarkDiscriminator(
        d4_ckpt_path=str(ckpt_path),
        preprocessed_cache_root=str(tmp_path / "cache"),
        device="cpu",
        encoder_pretrained=False,
        encoder_freeze=True,
    )

    class _FakeDetector:
        def score(self, frames, tau):
            return DetectionResult(
                step_scores=np.array([0.2, 0.9, 0.3], dtype=np.float32),
                lambda_values=np.array([0.2, 0.4, 0.7], dtype=np.float32),
                thresholds=np.array([0.5, 0.5, 0.5], dtype=np.float32),
                preds=np.array([0, 0, 1], dtype=np.int64),
                d_pos_sq=np.array([1.0, 2.0, 3.0], dtype=np.float32),
                d_neg_sq=np.array([1.5, 2.5, 3.5], dtype=np.float32),
            )

    discriminator.detector = _FakeDetector()
    discriminator._tau_per_task["PickPlaceBread"] = 0.5
    discriminator._frames = lambda trajectory: SimpleNamespace(length=3)

    trajectory = BenchmarkTrajectory(
        task_name="PickPlaceBread",
        num_frames=3,
        is_failure=True,
        video_id="traj_0",
        file_path="dummy.hdf5",
        demo_path="demos/demo_0",
    )
    out = discriminator.score_trajectory(trajectory)
    np.testing.assert_allclose(out.step_scores, np.array([0.2, 0.4, 0.7], dtype=np.float32))
    np.testing.assert_array_equal(out.predictions, np.array([0, 0, 1], dtype=np.int64))
    assert out.first_failure_frame == 2
    np.testing.assert_allclose(
        np.asarray(out.aux["raw_step_scores"]),
        np.array([0.2, 0.9, 0.3], dtype=np.float32),
    )
    discriminator.close()


def test_benchmark_filters_trajectories_without_cache(tmp_path: Path) -> None:
    encoder = Encoder(pretrained=False, freeze=True)
    predictor = ConditionalDynamicsPredictor(
        latent_dim=512,
        proprio_dim=4,
        action_dim=2,
        d_model=64,
        num_layers=1,
        nhead=4,
        dropout=0.0,
        max_action_horizon=4,
        d_cond=16,
    )
    ckpt_path = tmp_path / "d4_refactored.pt"
    torch.save(
        {
            "encoder_state_dict": encoder.state_dict(),
            "encoder_config": {"normalize_input": True, "freeze": True},
            "arch_args": predictor.arch_args(),
            "ema_state_dict": predictor.state_dict(),
            "config": {"horizon": 1, "proprio_indices": [0, 1, 2, 3]},
            "sigma_sq": 0.5,
        },
        ckpt_path,
    )

    discriminator = D4BenchmarkDiscriminator(
        d4_ckpt_path=str(ckpt_path),
        preprocessed_cache_root=str(tmp_path / "cache"),
        device="cpu",
        encoder_pretrained=False,
        encoder_freeze=True,
    )
    discriminator.cache_reader.exists = lambda task, file_path, demo_key: file_path != "missing.hdf5"

    trajectories = [
        BenchmarkTrajectory(
            task_name="PickPlaceBread",
            num_frames=3,
            is_failure=False,
            video_id="succ_keep",
            file_path="keep_success.hdf5",
            demo_path="demos/demo_0",
        ),
        BenchmarkTrajectory(
            task_name="PickPlaceBread",
            num_frames=3,
            is_failure=False,
            video_id="succ_skip",
            file_path="missing.hdf5",
            demo_path="demos/demo_1",
        ),
        BenchmarkTrajectory(
            task_name="PickPlaceBread",
            num_frames=3,
            is_failure=True,
            video_id="fail_keep",
            file_path="fail_out.hdf5",
            demo_path="demos/demo_2",
            source_hdf5_path="keep_fail.hdf5",
            source_demo_key="demo_src",
        ),
    ]

    kept, stats = discriminator.filter_cached_trajectories(trajectories)
    assert [traj.video_id for traj in kept] == ["succ_keep", "fail_keep"]
    assert stats["PickPlaceBread"] == {
        "kept_success": 1,
        "kept_fail": 1,
        "skipped_success": 1,
        "skipped_fail": 0,
    }
    discriminator.close()


def test_benchmark_rejects_old_256d_checkpoint(tmp_path: Path) -> None:
    ckpt_path = tmp_path / "d4_old.pt"
    torch.save(
        {
            "encoder_state_dict": {},
            "arch_args": {
                "latent_dim": 256,
                "proprio_dim": 4,
                "action_dim": 2,
                "d_model": 64,
                "num_layers": 1,
                "nhead": 4,
                "mlp_ratio": 4.0,
                "dropout": 0.0,
                "max_action_horizon": 4,
                "d_cond": 16,
            },
            "ema_state_dict": {},
        },
        ckpt_path,
    )

    with pytest.raises(ValueError, match="512-d LPB-parity checkpoints"):
        D4BenchmarkDiscriminator(
            d4_ckpt_path=str(ckpt_path),
            preprocessed_cache_root=str(tmp_path / "cache"),
            device="cpu",
        )
