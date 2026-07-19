"""Tensor-free tests for load-only nnPU benchmark reporting helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from benchmark.core import DiscriminatorOutput
from robosuite.pipeline.offline.src.eval_disc_checkpoint import (
    _RecordingDiscriminator,
    build_offline_success_false_alarm_report,
    build_success_false_alarm_report,
    load_checkpoint_contract,
)
from robosuite.pipeline.offline.discriminator.gt_fail_evaluation import (
    build_not_applicable_report,
)
from robosuite.pipeline.offline.visualization.disc_episodes import offline_trajectory


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


def _offline_success(index: int, length: int):
    success = np.zeros((length,), dtype=np.bool_)
    success[-1] = True
    episode = {
        "obs": {
            "state": np.zeros((length, 3), dtype=np.float32),
            "agentview": np.zeros((length, 4, 4, 3), dtype=np.uint8),
        },
        "executed_action": np.zeros((length, 2), dtype=np.float32),
        "is_intervention": np.zeros((length,), dtype=np.bool_),
        "success": success,
        "terminal_reason": "success",
    }
    return offline_trajectory(
        episode,
        episode_index=index,
        task="PickPlaceCereal",
        camera_names=["agentview"],
        fps=20,
        source_path="offline_episodes.pt",
    )


def _offline_source(path, *, indices: list[int]) -> dict[str, object]:
    return {
        "source_path": str(path),
        "schema_version": 1,
        "payload_task_name": "PickPlaceCereal",
        "camera_names": ["agentview"],
        "source_episode_count": 4,
        "eligible_episode_indices": indices,
    }


def test_parent_gt_fail_report_is_explicitly_not_applicable(tmp_path) -> None:
    report = build_not_applicable_report(
        checkpoint_path=tmp_path / "parent.pth",
        task_name="PickPlaceCereal",
    )
    assert report["schema_version"] == 2
    assert report["applicable"] is False
    assert report["hard_gate"] is False
    assert report["reason"] == "parent_checkpoint_has_no_offline_gt_negative_training_pool"


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


def test_offline_success_false_alarm_uses_pre_success_frames_and_provenance(
    tmp_path,
) -> None:
    checkpoint = tmp_path / "head.pth"
    source = tmp_path / "offline_episodes.pt"
    checkpoint.write_bytes(b"head")
    source.write_bytes(b"episodes")
    first = _offline_success(1, length=5)
    second = _offline_success(3, length=3)

    report = build_offline_success_false_alarm_report(
        [first, second],
        {
            id(first): _output([0, 1, 0, 0, 1]),
            id(second): _output([0, 0, 1]),
        },
        task="PickPlaceCereal",
        checkpoint=str(checkpoint),
        checkpoint_kind="finetuned",
        source_metadata=_offline_source(source, indices=[1, 3]),
    )

    assert report["valid_frame_policy"] == "frames_before_first_success_true"
    assert report["aggregate"] == {
        "source_episode_count": 4,
        "eligible_episode_count": 2,
        "valid_frames": 6,
        "false_alarm_frames": 1,
        "frame_false_alarm_rate": pytest.approx(1.0 / 6.0),
        "episodes_with_alarm": 1,
        "trajectory_alarm_rate": pytest.approx(0.5),
    }
    assert report["first_alarm"]["frames"] == [1]
    assert report["per_episode"][0]["first_success_frame"] == 4
    assert report["per_episode"][0]["first_alarm_frame"] == 1
    assert report["per_episode"][1]["first_alarm_frame"] is None
    assert report["provenance"]["eligible_episode_indices"] == [1, 3]
    assert len(report["provenance"]["checkpoint_sha256"]) == 64
    assert len(report["provenance"]["offline_episodes_sha256"]) == 64


def test_offline_success_false_alarm_rejects_empty_or_misaligned_inputs(
    tmp_path,
) -> None:
    checkpoint = tmp_path / "head.pth"
    source = tmp_path / "offline_episodes.pt"
    checkpoint.write_bytes(b"head")
    source.write_bytes(b"episodes")
    metadata = _offline_source(source, indices=[])
    with pytest.raises(RuntimeError, match="No offline success"):
        build_offline_success_false_alarm_report(
            [],
            {},
            task="PickPlaceCereal",
            checkpoint=str(checkpoint),
            checkpoint_kind="parent",
            source_metadata=metadata,
        )

    trajectory = _offline_success(0, length=3)
    with pytest.raises(ValueError, match="length 2, expected 3"):
        build_offline_success_false_alarm_report(
            [trajectory],
            {id(trajectory): _output([0, 1])},
            task="PickPlaceCereal",
            checkpoint=str(checkpoint),
            checkpoint_kind="parent",
            source_metadata=_offline_source(source, indices=[0]),
        )

    trajectory.intervention_mask[0] = True
    with pytest.raises(ValueError, match="contains intervention"):
        build_offline_success_false_alarm_report(
            [trajectory],
            {id(trajectory): _output([0, 0, 0])},
            task="PickPlaceCereal",
            checkpoint=str(checkpoint),
            checkpoint_kind="parent",
            source_metadata=_offline_source(source, indices=[0]),
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
