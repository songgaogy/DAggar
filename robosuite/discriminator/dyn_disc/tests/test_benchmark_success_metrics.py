"""NumPy-only tests for benchmark success operational metrics."""

from __future__ import annotations

import numpy as np
import pytest

from benchmark.core import (
    BenchmarkTrajectory,
    DiscriminatorOutput,
    EvalConfig,
    FailureBenchmark,
)


class _Trajectory(BenchmarkTrajectory):
    def __init__(
        self,
        *,
        video_id: str,
        is_failure: bool,
        num_frames: int,
        failure_mask: list[int] | None = None,
        pre_done_frames: int | None = None,
    ) -> None:
        segments = []
        if is_failure:
            segments = [{"start": 2, "end": 3, "mode": "test"}]
        super().__init__(
            task_name="PickPlaceCereal",
            num_frames=num_frames,
            is_failure=is_failure,
            video_id=video_id,
            failure_segments=segments,
        )
        self._failure_mask = failure_mask
        self._pre_done_frames = pre_done_frames

    def load_failure_mask(self) -> np.ndarray | None:
        if self._failure_mask is None:
            return None
        return np.asarray(self._failure_mask, dtype=np.int64)

    def prefix_frames_before_done(self) -> int:
        if self._pre_done_frames is None:
            return int(self.num_frames)
        return int(self._pre_done_frames)


class _Discriminator:
    name = "provided-predictions"

    def __init__(self, outputs: dict[str, DiscriminatorOutput]) -> None:
        self._outputs = outputs

    def score_trajectory(self, trajectory: BenchmarkTrajectory) -> DiscriminatorOutput:
        return self._outputs[trajectory.video_id]


def _output(
    scores: list[float],
    predictions: list[int] | None,
    *,
    scored_frames: int | None = None,
) -> DiscriminatorOutput:
    aux = None if scored_frames is None else {"success_prefix_frames": scored_frames}
    return DiscriminatorOutput(
        step_scores=np.asarray(scores, dtype=np.float64),
        predictions=(
            None
            if predictions is None
            else np.asarray(predictions, dtype=np.int64)
        ),
        aux=aux,
    )


def _failure() -> _Trajectory:
    return _Trajectory(
        video_id="failure-0",
        is_failure=True,
        num_frames=4,
        failure_mask=[0, 0, 1, 1],
    )


def test_success_metrics_use_provided_predictions_on_scored_pre_done_frames() -> None:
    first = _Trajectory(
        video_id="success-0",
        is_failure=False,
        num_frames=5,
        pre_done_frames=3,
    )
    second = _Trajectory(
        video_id="success-1",
        is_failure=False,
        num_frames=4,
        pre_done_frames=4,
    )
    failure = _failure()
    discriminator = _Discriminator(
        {
            # High scores prove success metrics do not re-threshold step_scores.
            "success-0": _output(
                [10.0] * 5,
                [0, 1, 0, 1, 1],
                scored_frames=4,
            ),
            # Alarms after the two actually scored frames must be ignored.
            "success-1": _output(
                [10.0] * 4,
                [0, 0, 1, 1],
                scored_frames=2,
            ),
            "failure-0": _output([0.0, 0.0, 1.0, 1.0], [0, 0, 0, 0]),
        }
    )

    result = FailureBenchmark([first, second, failure]).evaluate(
        discriminator,
        EvalConfig(step_binarize_strategy="fixed", step_fixed_threshold=0.5),
        progress=False,
    )

    assert result.step_level["frame_f1"] == pytest.approx(1.0)
    assert result.step_level["success_num_trajectories"] == 2
    assert result.step_level["success_valid_frames"] == 5
    assert result.step_level["success_false_positive_frames"] == 1
    assert result.step_level["success_frame_false_alarm_rate"] == pytest.approx(0.2)
    assert result.step_level["success_specificity"] == pytest.approx(0.8)
    assert result.step_level["success_trajectories_with_alarm"] == 1
    assert result.step_level["success_trajectory_alarm_rate"] == pytest.approx(0.5)
    assert result.step_level_per_task["PickPlaceCereal"][
        "success_specificity"
    ] == pytest.approx(0.8)

    first_detail = result.per_trajectory[0]
    assert first_detail["success_valid_frames"] == 3
    assert first_detail["success_false_positive_frames"] == 1
    assert first_detail["success_frame_false_alarm_rate"] == pytest.approx(1.0 / 3.0)
    assert first_detail["success_any_alarm"] is True
    assert first_detail["success_first_alarm_frame"] == 1
    second_detail = result.per_trajectory[1]
    assert second_detail["success_valid_frames"] == 2
    assert second_detail["success_any_alarm"] is False
    assert second_detail["success_first_alarm_frame"] is None


def test_missing_success_predictions_preserves_score_only_benchmark_behavior() -> None:
    success = _Trajectory(
        video_id="success-0",
        is_failure=False,
        num_frames=4,
        pre_done_frames=3,
    )
    failure = _failure()
    discriminator = _Discriminator(
        {
            "success-0": _output([0.0] * 4, None, scored_frames=3),
            "failure-0": _output([0.0, 0.0, 1.0, 1.0], None),
        }
    )

    result = FailureBenchmark([success, failure]).evaluate(
        discriminator,
        EvalConfig(step_binarize_strategy="fixed", step_fixed_threshold=0.5),
        progress=False,
    )

    assert result.step_level["frame_f1"] == pytest.approx(1.0)
    assert not any(key.startswith("success_") for key in result.step_level)
    assert not any(
        key.startswith("success_") for key in result.per_trajectory[0]
    )


def test_invalid_scored_success_boundary_is_rejected() -> None:
    success = _Trajectory(
        video_id="success-0",
        is_failure=False,
        num_frames=3,
        pre_done_frames=3,
    )
    failure = _failure()
    discriminator = _Discriminator(
        {
            "success-0": _output([0.0] * 3, [0, 0, 0], scored_frames=4),
            "failure-0": _output([0.0, 0.0, 1.0, 1.0], [0, 0, 1, 1]),
        }
    )

    with pytest.raises(ValueError, match="not in \\[0, 3\\]"):
        FailureBenchmark([success, failure]).evaluate(
            discriminator,
            EvalConfig(step_binarize_strategy="provided"),
            progress=False,
        )
