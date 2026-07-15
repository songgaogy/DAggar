"""NumPy-only tests for offline nnPU policy-segment routing."""

from __future__ import annotations

import json
import numpy as np
import pytest

from robosuite.pipeline.offline.discriminator import (
    DiscriminatorPools,
    LatentTrajectory,
    build_action_windows,
    combine_pools,
    load_pretrain_pools,
    split_policy_segments,
    validate_offline_payload,
)


def _episode(
    interventions: list[bool],
    *,
    terminal_reason: str,
    action_dim: int = 2,
    human_success: bool = False,
) -> dict[str, object]:
    length = len(interventions)
    policy = np.arange(length * action_dim, dtype=np.float32).reshape(length, action_dim)
    executed = policy.copy()
    for step, intervention in enumerate(interventions):
        if intervention:
            executed[step] = -100.0 - step
    success = np.zeros(length, dtype=np.bool_)
    if terminal_reason == "success":
        success[-1] = True
        if human_success:
            assert interventions[-1]
    done = np.zeros(length, dtype=np.bool_)
    done[-1] = True
    state = np.arange(length * 3, dtype=np.float32).reshape(length, 3)
    images = np.zeros((length, 4, 5, 3), dtype=np.uint8)
    return {
        "obs": {"state": state, "agentview": images},
        "next_obs": {"state": state + 1.0, "agentview": images},
        "executed_action": executed,
        "policy_action": policy,
        "is_intervention": np.asarray(interventions, dtype=np.bool_),
        "success": success,
        "done": done,
        "terminal_reason": terminal_reason,
    }


def _payload(*episodes: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "task_name": "Task",
        "camera_names": ["agentview"],
        "episodes": list(episodes),
    }


def test_policy_human_policy_success_routes_only_final_policy_run_to_positive() -> None:
    payload = validate_offline_payload(
        _payload(_episode([False, False, True, True, False, False], terminal_reason="success"))
    )
    segments, stats = split_policy_segments(payload)

    assert [(segment.lo, segment.hi, segment.pool) for segment in segments] == [
        (0, 2, "unlabeled"),
        (4, 6, "positive"),
    ]
    assert stats["excluded_human_frames"] == 2
    assert stats["positive_frames"] == 2
    assert stats["unlabeled_frames"] == 2


def test_human_completed_success_does_not_create_positive_segment() -> None:
    payload = validate_offline_payload(
        _payload(
            _episode(
                [False, False, True],
                terminal_reason="success",
                human_success=True,
            )
        )
    )
    segments, _ = split_policy_segments(payload)

    assert [(segment.lo, segment.hi, segment.pool) for segment in segments] == [
        (0, 2, "unlabeled")
    ]


def test_all_human_episode_produces_no_training_segment() -> None:
    payload = validate_offline_payload(
        _payload(_episode([True, True, True], terminal_reason="manual_reset"))
    )
    segments, stats = split_policy_segments(payload)

    assert segments == []
    assert stats["excluded_human_frames"] == 3


@pytest.mark.parametrize(
    "terminal_reason",
    ["manual_reset", "max_steps", "env_done", "interrupted"],
)
def test_non_success_terminal_policy_segments_are_unlabeled(terminal_reason: str) -> None:
    payload = validate_offline_payload(
        _payload(_episode([False, False], terminal_reason=terminal_reason))
    )
    segments, _ = split_policy_segments(payload)

    assert len(segments) == 1
    assert segments[0].pool == "unlabeled"
    assert segments[0].ended_by == terminal_reason


def test_short_segment_chunk_repeats_tail_without_crossing_boundary() -> None:
    first_segment = np.asarray([[1.0, 2.0]], dtype=np.float32)
    second_segment = np.asarray([[7.0, 8.0], [9.0, 10.0]], dtype=np.float32)

    first = build_action_windows(first_segment, horizon=3, use_chunk=True)
    second = build_action_windows(second_segment, horizon=3, use_chunk=True)

    np.testing.assert_array_equal(first, [[1.0, 2.0, 1.0, 2.0, 1.0, 2.0]])
    np.testing.assert_array_equal(
        second,
        [
            [7.0, 8.0, 9.0, 10.0, 9.0, 10.0],
            [9.0, 10.0, 9.0, 10.0, 9.0, 10.0],
        ],
    )


def test_legacy_single_action_mode_zero_pads() -> None:
    actions = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    windows = build_action_windows(actions, horizon=3, use_chunk=False)

    np.testing.assert_array_equal(
        windows,
        [[1.0, 2.0, 0.0, 0.0, 0.0, 0.0], [3.0, 4.0, 0.0, 0.0, 0.0, 0.0]],
    )


def test_combined_calibration_remains_pretrain_only() -> None:
    feature = np.zeros((2, 3), dtype=np.float32)
    pretrain_calibration = LatentTrajectory(
        features=feature,  # type: ignore[arg-type]
        pool="calibration",
        source="pretrain",
        identifier="held-out",
    )
    pretrain = DiscriminatorPools(
        positive=[
            LatentTrajectory(feature, "positive", "pretrain", "pretrain-positive")  # type: ignore[arg-type]
        ],
        unlabeled=[
            LatentTrajectory(feature, "unlabeled", "pretrain", "pretrain-unlabeled")  # type: ignore[arg-type]
        ],
        calibration=[pretrain_calibration],
    )
    online = DiscriminatorPools(
        positive=[
            LatentTrajectory(feature, "positive", "online", "online-positive")  # type: ignore[arg-type]
        ],
        unlabeled=[
            LatentTrajectory(feature, "unlabeled", "online", "online-unlabeled")  # type: ignore[arg-type]
        ],
    )

    combined = combine_pools(pretrain, online)

    assert combined.calibration == [pretrain_calibration]
    assert [item.identifier for item in combined.positive] == [
        "pretrain-positive",
        "online-positive",
    ]
    assert [item.identifier for item in combined.unlabeled] == [
        "pretrain-unlabeled",
        "online-unlabeled",
    ]


def test_use_only_offline_drops_pretrain_unlabeled_pool() -> None:
    feature = np.zeros((2, 3), dtype=np.float32)
    pretrain_calibration = LatentTrajectory(
        features=feature,  # type: ignore[arg-type]
        pool="calibration",
        source="pretrain",
        identifier="held-out",
    )
    pretrain = DiscriminatorPools(
        positive=[
            LatentTrajectory(feature, "positive", "pretrain", "pretrain-positive")  # type: ignore[arg-type]
        ],
        unlabeled=[
            LatentTrajectory(feature, "unlabeled", "pretrain", "pretrain-unlabeled")  # type: ignore[arg-type]
        ],
        calibration=[pretrain_calibration],
    )
    online = DiscriminatorPools(
        positive=[
            LatentTrajectory(feature, "positive", "online", "online-positive")  # type: ignore[arg-type]
        ],
        unlabeled=[
            LatentTrajectory(feature, "unlabeled", "online", "online-unlabeled")  # type: ignore[arg-type]
        ],
    )

    combined = combine_pools(pretrain, online, use_only_offline=True)

    assert combined.calibration == [pretrain_calibration]
    assert [item.identifier for item in combined.positive] == [
        "pretrain-positive",
        "online-positive",
    ]
    assert [item.identifier for item in combined.unlabeled] == ["online-unlabeled"]
    assert combined.stats["use_only_offline"] is True


def test_non_intervention_action_mismatch_is_rejected() -> None:
    episode = _episode([False, True], terminal_reason="manual_reset")
    episode["executed_action"][0, 0] += 0.1  # type: ignore[index]

    with pytest.raises(ValueError, match="executed/policy action mismatch"):
        validate_offline_payload(_payload(episode))


def test_pretrain_calibration_overlap_is_rejected_before_loading_shards(tmp_path) -> None:
    def entry(split_name: str, video_id: str) -> dict[str, object]:
        return {
            "path": f"{split_name}/missing.pt",
            "video_id": video_id,
            "num_frames": 1,
            "latent_dim": 2,
        }

    manifest = {
        "schema_version": 1,
        "splits": {
            "positive_train": [entry("positive_train", "shared-success")],
            "positive_calib": [entry("positive_calib", "shared-success")],
            "unlabeled_train": [entry("unlabeled_train", "failure")],
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="overlap in video IDs"):
        load_pretrain_pools(tmp_path)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda episode: episode.pop("policy_action"), "missing fields"),
        (
            lambda episode: episode.__setitem__("done", np.asarray([True, True])),
            "exactly one terminal done",
        ),
        (
            lambda episode: episode.__setitem__(
                "success", np.asarray([True, False], dtype=np.bool_)
            ),
            "contains success flags",
        ),
    ],
)
def test_schema_inconsistency_is_rejected(mutation, error: str) -> None:
    episode = _episode([False, False], terminal_reason="manual_reset")
    mutation(episode)

    with pytest.raises((KeyError, ValueError), match=error):
        validate_offline_payload(_payload(episode))
