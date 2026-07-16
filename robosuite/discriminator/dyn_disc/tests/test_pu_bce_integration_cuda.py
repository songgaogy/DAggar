"""CUDA integration tests for nnPU snapshots, logging, and frozen loading."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from robosuite.discriminator.dyn_disc.adapters.pu_bce import (
    PUBCEBenchmarkDiscriminator,
)
from robosuite.discriminator.dyn_disc.detectors.pu_bce import PUBCEDiscriminator
from robosuite.pipeline.algorithms.discriminator.nnpu import FrozenNNPUDiscriminator


def _cuda() -> torch.device:
    if not torch.cuda.is_available():
        pytest.fail("CUDA is required for dyn_disc integration tests")
    return torch.device("cuda:0")


def _detector(device: torch.device, *, center: float) -> PUBCEDiscriminator:
    detector = PUBCEDiscriminator(
        in_dim=6,
        hidden=8,
        num_layers=1,
        device=str(device),
    )
    detector.head.set_logit_center(center)
    detector.thresholds = {"Task": 0.25}
    detector._completed_epochs = 2
    return detector


def _adapter_shell(
    tmp_path: Path,
    detector: PUBCEDiscriminator,
    device: torch.device,
) -> PUBCEBenchmarkDiscriminator:
    model_checkpoint = tmp_path / "model.pth"
    model_checkpoint.write_bytes(b"test model checkpoint")

    adapter = object.__new__(PUBCEBenchmarkDiscriminator)
    adapter.device = device
    adapter.save_ckpt_dir = str(tmp_path / "snapshots")
    adapter._shared_detector = detector
    adapter._detectors_per_task = {"Task": detector}
    adapter.epoch_checkpoint_paths = {}
    adapter.checkpoint_epochs = (1, 2, 5, 10, 20)
    adapter.feature_source = "transformer"
    adapter.transformer_layer = 1
    adapter.use_chunk = True
    adapter.model_ckpt = str(model_checkpoint)
    adapter.encoder = SimpleNamespace(normalizer_checkpoint=None)
    adapter.pi_p = 0.3
    adapter.loss_surrogate = "logistic"
    adapter.nn_correction = True
    adapter.beta = 0.0
    adapter.threshold_normalization = "epoch_boundary"
    adapter.soft_cap_c = 5.0
    adapter.soft_cap_lambda = 1e-2
    adapter.soft_cap_temperature = 1.0
    adapter.seed = 0
    adapter.calib_fraction = 0.2
    adapter.delta = 10.0
    adapter.camera_to_view = {}
    adapter.proprio_indices = None
    adapter.encode_batch_size = 32
    adapter._success_train_video_ids = {"Task": ["train-success"]}
    adapter._success_calib_video_ids = {"Task": ["calib-success"]}
    adapter.unlabeled_fail_trajectories = [SimpleNamespace(video_id="train-failure")]
    adapter.verbose_fit = False
    return adapter


def test_adapter_epoch_snapshot_activation_reproduces_cuda_prediction(
    tmp_path: Path,
) -> None:
    device = _cuda()
    detector = _detector(device, center=1.75)
    adapter = _adapter_shell(tmp_path, detector, device)
    generator = torch.Generator(device=device).manual_seed(17)
    probe = torch.randn((13, 6), generator=generator, device=device)

    expected_scores = detector.failure_score_tensor(probe).detach().clone()
    expected_decisions = expected_scores >= detector.threshold_tensor(probe, "Task")
    adapter._snapshot_epoch(2, detector)
    checkpoint = Path(adapter.epoch_checkpoint_paths[2])

    detector.head.set_logit_center(-9.0)
    with torch.no_grad():
        next(detector.head.parameters()).add_(3.0)
    payload = adapter.activate_checkpoint(checkpoint)

    restored = adapter._shared_detector
    assert restored is not None
    restored_scores = restored.failure_score_tensor(probe)
    restored_decisions = restored_scores >= restored.threshold_tensor(probe, "Task")
    assert payload["epoch"] == 2
    assert restored_scores.is_cuda
    assert float(restored.head.logit_center) == pytest.approx(1.75)
    torch.testing.assert_close(restored_scores, expected_scores)
    torch.testing.assert_close(restored_decisions, expected_decisions)


def test_tensorboard_callbacks_write_train_and_benchmark_events(
    tmp_path: Path,
) -> None:
    device = _cuda()
    log_dir = tmp_path / "tensorboard"
    adapter = object.__new__(PUBCEBenchmarkDiscriminator)
    adapter._writer = SummaryWriter(log_dir=str(log_dir))
    cuda_values = torch.tensor([1.25, 0.875], device=device)

    adapter._record_epoch_metrics(
        2,
        {
            "loss": {"total": float(cuda_values[0].item())},
            "pools": {"train_positive": {"effective": {"abs_p99": 4.5}}},
            "all_finite": True,
        },
    )
    adapter.log_benchmark_metrics(
        2,
        {
            "task_score": float(cuda_values[1].item()),
            "healthy": True,
        },
    )
    adapter._writer.close()
    adapter._writer = None

    event_files = list(log_dir.glob("events.out.tfevents.*"))
    assert event_files
    accumulator = EventAccumulator(str(log_dir))
    accumulator.Reload()
    scalar_tags = set(accumulator.Tags()["scalars"])
    assert {
        "train/loss/total",
        "train/pools/train_positive/effective/abs_p99",
        "benchmark/task_score",
    } <= scalar_tags
    assert "train/all_finite" not in scalar_tags
    assert "benchmark/healthy" not in scalar_tags

    train_event = accumulator.Scalars("train/loss/total")[-1]
    benchmark_event = accumulator.Scalars("benchmark/task_score")[-1]
    assert train_event.step == 2
    assert train_event.value == pytest.approx(1.25)
    assert benchmark_event.step == 2
    assert benchmark_event.value == pytest.approx(0.875)


def test_frozen_nnpu_cuda_roundtrip_preserves_center_and_legacy_zero_center(
    tmp_path: Path,
) -> None:
    device = _cuda()
    source = _detector(device, center=1.5)
    generator = torch.Generator(device=device).manual_seed(23)
    probe = torch.randn((11, 6), generator=generator, device=device)

    checkpoint = tmp_path / "centered.pth"
    torch.save({"pu_bce_detector": source.state_dict()}, checkpoint)
    frozen = FrozenNNPUDiscriminator(
        nnpu_ckpt_path=checkpoint,
        task_name="Task",
        device=device,
    )
    centered_expected = source.failure_score_tensor(probe)
    centered_actual = frozen.failure_score(probe)
    assert centered_actual.is_cuda
    assert float(frozen.detector.head.logit_center) == pytest.approx(1.5)
    torch.testing.assert_close(centered_actual, centered_expected)

    legacy_state = source.state_dict()
    legacy_head = dict(legacy_state["head"])
    legacy_head.pop("logit_center")
    legacy_state["head"] = legacy_head
    legacy_checkpoint = tmp_path / "legacy.pth"
    torch.save({"pu_bce_detector": legacy_state}, legacy_checkpoint)
    legacy_frozen = FrozenNNPUDiscriminator(
        nnpu_ckpt_path=legacy_checkpoint,
        task_name="Task",
        device=device,
    )
    legacy_expected = -source.raw_logits_tensor(probe)
    legacy_actual = legacy_frozen.failure_score(probe)
    assert float(legacy_frozen.detector.head.logit_center) == pytest.approx(0.0)
    torch.testing.assert_close(legacy_actual, legacy_expected)
