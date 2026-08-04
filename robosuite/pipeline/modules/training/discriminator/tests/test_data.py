"""NumPy-only tests for offline nnPU policy-segment routing."""

from __future__ import annotations

import json
import numpy as np
import pytest

from robosuite.pipeline.modules.training.discriminator import (
    build_action_windows,
    load_pretrain_pools,
    resolve_parent_nnpu_semantics,
    resolve_parent_success_boundary,
    split_policy_segments,
    validate_offline_payload,
)
from robosuite.pipeline.modules.training.discriminator.episodes import build_gt_negative_windows


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


def test_gt_negative_window_uses_chunk_scaled_contiguous_pre_and_post_frames() -> None:
    episode = _episode(
        [False] * 10 + [True] * 9 + [False] * 2,
        terminal_reason="manual_reset",
    )
    payload = validate_offline_payload(_payload(episode))

    windows, stats = build_gt_negative_windows(
        payload,
        pre_intervention_chunks=2,
        post_intervention_chunks=3,
        frameskip=3,
    )

    assert len(windows) == 1
    window = windows[0]
    assert (window.lo, window.onset, window.hi) == (4, 10, 19)
    assert window.frame_indices == tuple(range(4, 19))
    assert window.num_pre_frames == 6
    assert window.num_post_frames == 9
    assert stats["gt_negative_frames"] == 15
    assert stats["post_truncated_events"] == 0
    assert stats["stored_gt_fail_ignored"] is True


def test_gt_negative_window_allows_zero_post_intervention_chunks() -> None:
    episode = _episode(
        [False] * 10 + [True] * 9 + [False] * 2,
        terminal_reason="manual_reset",
    )
    payload = validate_offline_payload(_payload(episode))

    windows, stats = build_gt_negative_windows(
        payload,
        pre_intervention_chunks=2,
        post_intervention_chunks=0,
        frameskip=3,
    )

    assert len(windows) == 1
    window = windows[0]
    assert (window.lo, window.onset, window.hi) == (4, 10, 10)
    assert window.frame_indices == tuple(range(4, 10))
    assert window.num_pre_frames == 6
    assert window.num_post_frames == 0
    assert stats["gt_negative_frames"] == 6
    assert stats["post_frames"] == 0
    assert stats["post_truncated_events"] == 0


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ([True, True], (0, 0, 2)),
        ([False, True, True], (0, 1, 3)),
        ([False, False, True], (0, 2, 3)),
    ],
)
def test_gt_negative_window_truncates_at_episode_and_intervention_boundaries(
    flags: list[bool], expected: tuple[int, int, int]
) -> None:
    payload = validate_offline_payload(
        _payload(_episode(flags, terminal_reason="manual_reset"))
    )

    windows, stats = build_gt_negative_windows(
        payload,
        pre_intervention_chunks=4,
        post_intervention_chunks=4,
        frameskip=2,
    )

    assert [(window.lo, window.onset, window.hi) for window in windows] == [expected]
    assert stats["pre_truncated_events"] == 1
    assert stats["post_truncated_events"] == 1


def test_gt_negative_multiple_events_stay_local_and_are_globally_unique() -> None:
    flags = [False, False, True, True, False, False, False, True, False]
    payload = validate_offline_payload(
        _payload(_episode(flags, terminal_reason="manual_reset"))
    )

    windows, stats = build_gt_negative_windows(
        payload,
        pre_intervention_chunks=2,
        post_intervention_chunks=2,
        frameskip=1,
    )

    assert [(window.lo, window.onset, window.hi) for window in windows] == [
        (0, 2, 4),
        (5, 7, 8),
    ]
    keys = [
        (window.source_episode_index, frame)
        for window in windows
        for frame in window.frame_indices
    ]
    assert len(keys) == len(set(keys))
    assert stats["intervention_events"] == 2
    assert stats["deduplicated_frames"] == 0
    assert [event["event_index"] for event in stats["events"]] == [0, 1]


def test_gt_negative_action_chunks_cross_onset_but_not_event_window() -> None:
    episode = _episode(
        [False, False, True, True, False], terminal_reason="manual_reset", action_dim=1
    )
    payload = validate_offline_payload(_payload(episode))
    windows, _ = build_gt_negative_windows(
        payload,
        pre_intervention_chunks=2,
        post_intervention_chunks=2,
        frameskip=1,
    )
    window = windows[0]
    policy = np.asarray(episode["policy_action"])[window.lo : window.hi]

    chunks = build_action_windows(policy, horizon=3, use_chunk=True)

    np.testing.assert_array_equal(
        chunks,
        [
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 3.0],
            [3.0, 3.0, 3.0],
        ],
    )
    assert not np.any(chunks == np.asarray(episode["executed_action"])[2, 0])


def test_gt_negative_routing_ignores_stored_gt_fail() -> None:
    episode_a = _episode([False, True, True], terminal_reason="manual_reset")
    episode_b = _episode([False, True, True], terminal_reason="manual_reset")
    episode_a["gt_fail"] = np.asarray([False, False, False], dtype=np.bool_)
    episode_b["gt_fail"] = np.asarray([True, False, True], dtype=np.bool_)

    windows_a, stats_a = build_gt_negative_windows(
        validate_offline_payload(_payload(episode_a)),
        pre_intervention_chunks=1,
        post_intervention_chunks=1,
        frameskip=2,
    )
    windows_b, stats_b = build_gt_negative_windows(
        validate_offline_payload(_payload(episode_b)),
        pre_intervention_chunks=1,
        post_intervention_chunks=1,
        frameskip=2,
    )

    assert [(item.lo, item.hi, item.frame_indices) for item in windows_a] == [
        (item.lo, item.hi, item.frame_indices) for item in windows_b
    ]
    assert stats_a == stats_b


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


def test_non_intervention_action_mismatch_is_rejected() -> None:
    episode = _episode([False, True], terminal_reason="manual_reset")
    episode["executed_action"][0, 0] += 0.1  # type: ignore[index]

    with pytest.raises(ValueError, match="executed/policy action mismatch"):
        validate_offline_payload(_payload(episode))


def test_non_finite_policy_action_is_rejected() -> None:
    episode = _episode([False, True], terminal_reason="manual_reset")
    episode["policy_action"][1, 0] = np.nan  # type: ignore[index]

    with pytest.raises(ValueError, match="policy_action contains non-finite"):
        validate_offline_payload(_payload(episode))


def test_parent_nnpu_semantics_are_strictly_inherited() -> None:
    semantics = resolve_parent_nnpu_semantics(
        {
            "pu_bce_detector": {
                "pi_p": 0.3,
                "delta": 5.0,
                "loss_surrogate": "logistic",
            },
            "nn_correction": True,
            "beta": 0.0,
        }
    )

    assert semantics == {
        "pi_p": 0.3,
        "loss_surrogate": "logistic",
        "nn_correction": True,
        "beta": 0.0,
        "delta": 5.0,
    }


def test_normalization_center_is_negative_parent_failure_threshold() -> None:
    payload = {
        "pu_bce_detector": {
            "thresholds": {"PickPlaceCereal": -1.75},
        }
    }

    center = resolve_parent_success_boundary(payload, "PickPlaceCereal")

    assert center == pytest.approx(1.75)
    with pytest.raises(KeyError, match="no threshold"):
        resolve_parent_success_boundary(payload, "OtherTask")


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda payload: payload.pop("beta"), "missing beta"),
        (
            lambda payload: payload.__setitem__("nn_correction", "false"),
            "nn_correction must be bool",
        ),
        (
            lambda payload: payload.__setitem__("beta", True),
            "beta must be a real number",
        ),
        (
            lambda payload: payload.__setitem__("pi_p", "0.3"),
            "pi_p must be a real number",
        ),
        (
            lambda payload: payload.__setitem__("delta", "10.0"),
            "delta must be a real number",
        ),
    ],
)
def test_invalid_parent_nnpu_semantics_fail_fast(mutation, error: str) -> None:
    payload = {
        "pi_p": 0.3,
        "delta": 10.0,
        "loss_surrogate": "logistic",
        "nn_correction": True,
        "beta": 0.0,
        "pu_bce_detector": {},
    }
    mutation(payload)

    with pytest.raises((KeyError, TypeError, ValueError), match=error):
        resolve_parent_nnpu_semantics(payload)


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

