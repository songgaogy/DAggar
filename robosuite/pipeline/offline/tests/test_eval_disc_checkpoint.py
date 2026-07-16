"""Tensor-free tests for load-only nnPU benchmark reporting helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from benchmark.core import DiscriminatorOutput
from robosuite.pipeline.offline.src.eval_disc_checkpoint import (
    _RecordingDiscriminator,
    build_success_false_alarm_report,
    load_checkpoint_contract,
)


class _SuccessTrajectory:
    def __init__(self, video_id: str, num_frames: int, valid_frames: int) -> None:
        self.task_name = "PickPlaceCereal"
        self.video_id = video_id
        self.num_frames = num_frames
        self.is_failure = False
        self._valid_frames = valid_frames

    def prefix_frames_before_done(self) -> int:
        return self._valid_frames


def _output(predictions: list[int]) -> DiscriminatorOutput:
    return DiscriminatorOutput(
        step_scores=np.zeros((len(predictions),), dtype=np.float32),
        predictions=np.asarray(predictions, dtype=np.int64),
    )


def test_success_false_alarm_excludes_post_completion_padding() -> None:
    first = _SuccessTrajectory("success-0", num_frames=5, valid_frames=3)
    second = _SuccessTrajectory("success-1", num_frames=4, valid_frames=4)
    report = build_success_false_alarm_report(
        [first, second],
        {
            id(first): _output([0, 1, 0, 1, 1]),
            id(second): _output([0, 0, 0, 0]),
        },
        task="PickPlaceCereal",
        checkpoint="head.pth",
        checkpoint_kind="finetuned",
    )

    assert report["schema_version"] == 1
    assert report["valid_frames"] == 7
    assert report["false_positive_frames"] == 1
    assert report["frame_fpr"] == pytest.approx(1.0 / 7.0)
    assert report["trajectories_with_alarm"] == 1
    assert report["trajectory_alarm_rate"] == pytest.approx(0.5)
    assert report["per_trajectory"][0] == {
        "task_name": "PickPlaceCereal",
        "video_id": "success-0",
        "num_frames": 5,
        "valid_frames": 3,
        "false_positive_frames": 1,
        "frame_fpr": pytest.approx(1.0 / 3.0),
        "any_alarm": True,
        "first_alarm_frame": 1,
    }


def test_success_false_alarm_rejects_missing_or_misaligned_predictions() -> None:
    trajectory = _SuccessTrajectory("success-0", num_frames=3, valid_frames=2)
    with pytest.raises(KeyError, match="Missing recorded output"):
        build_success_false_alarm_report(
            [trajectory],
            {},
            task="PickPlaceCereal",
            checkpoint="head.pth",
            checkpoint_kind="parent",
        )
    with pytest.raises(ValueError, match="length 2, expected 3"):
        build_success_false_alarm_report(
            [trajectory],
            {id(trajectory): _output([0, 1])},
            task="PickPlaceCereal",
            checkpoint="head.pth",
            checkpoint_kind="parent",
        )


def test_recording_discriminator_delegates_and_caches_by_identity() -> None:
    trajectory = SimpleNamespace(video_id="success-0")
    expected = _output([0, 1])
    inner = SimpleNamespace(
        name="inner",
        score_trajectory=lambda value: expected,
    )
    recording = _RecordingDiscriminator(inner)

    assert recording.score_trajectory(trajectory) is expected
    assert recording.name == "inner"
    assert recording.outputs[id(trajectory)] is expected


def test_parent_checkpoint_contract_detects_legacy_checkpoint(tmp_path) -> None:
    model = tmp_path / "model.pth"
    model.write_bytes(b"model")
    checkpoint = tmp_path / "pu_bce_head.pth"
    torch.save(
        {
            "pu_bce_detector": {},
            "feature_source": "transformer",
            "transformer_layer": 1,
            "model_ckpt": str(model),
            "camera_to_view": {"agentview": "front"},
            "proprio_indices": [0, 2],
        },
        checkpoint,
    )

    contract = load_checkpoint_contract(
        checkpoint,
        model_checkpoint=model,
        feature_source="transformer",
        transformer_layer=1,
        camera_to_view=None,
        proprio_indices=None,
    )

    assert contract.checkpoint_kind == "parent"
    assert contract.camera_to_view == {"agentview": "front"}
    assert contract.proprio_indices == [0, 2]
