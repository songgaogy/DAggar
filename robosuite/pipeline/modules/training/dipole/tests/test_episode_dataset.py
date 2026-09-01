from __future__ import annotations

import numpy as np
import pytest

from robosuite.pipeline.modules.training.dipole.episode_dataset import (
    build_offline_transitions,
)


def _episode(
    interventions: list[bool],
    *,
    terminal_reason: str = "success",
) -> dict:
    length = len(interventions)
    obs = {
        "state": np.zeros((length, 3), dtype=np.float32),
        "agentview": np.zeros((length, 4, 4, 3), dtype=np.uint8),
    }
    success = np.zeros(length, dtype=np.bool_)
    success[-1] = terminal_reason == "success"
    actions = np.zeros((length, 2), dtype=np.float32)
    done = np.zeros(length, dtype=np.bool_)
    done[-1] = True
    return {
        "obs": obs,
        "next_obs": {key: value.copy() for key, value in obs.items()},
        "executed_action": actions,
        "policy_action": actions.copy(),
        "is_intervention": np.asarray(interventions, dtype=np.bool_),
        "success": success,
        "done": done,
        "terminal_reason": terminal_reason,
    }


def _build(*episodes: dict):
    return build_offline_transitions(
        {"camera_names": ["agentview"], "episodes": list(episodes)},
        action_horizon=2,
        reward_success=0.0,
        reward_fail=-1.0,
        include_policy_action_neg=False,
    )


def test_pure_success_policy_uses_per_frame_sparse_reward() -> None:
    streams = _build(_episode([False, False, False, False]))

    assert [transition.reward for transition in streams.policy_bc] == [
        -1.0,
        -1.0,
        -1.0,
        0.0,
    ]
    assert [
        bool((transition.info or {})["success"])
        for transition in streams.policy_bc
    ] == [False, False, False, True]
    assert streams.stats["policy_success_reward_frames"] == 1
    assert streams.stats["policy_failure_reward_frames"] == 3
    assert all(
        bool((transition.info or {})["is_success_trajectory"])
        for transition in streams.policy_bc
    )


def test_intervention_boundary_keeps_success_metadata_for_truncation() -> None:
    streams = _build(
        _episode([False, False, False, True, True, False, False, False])
    )
    sections: dict[int, list] = {}
    for transition in streams.policy_bc:
        sections.setdefault(int((transition.info or {})["episode_index"]), []).append(
            transition
        )

    before, after = [sections[index] for index in sorted(sections)]
    assert all(
        not bool((transition.info or {})["is_success_trajectory"])
        for transition in streams.policy_bc + streams.human_pos
    )
    assert [int((transition.info or {})["source_frame_index"]) for transition in before] == [
        0,
        1,
        2,
    ]
    assert [transition.reward for transition in before] == [-1.0, -1.0, -1.0]
    assert all((transition.info or {})["success"] is False for transition in before)
    assert all(
        (transition.info or {})["policy_section_end_reason"]
        == "ended_human_intervention"
        for transition in before
    )
    assert [transition.done for transition in before] == [False, False, True]

    assert [int((transition.info or {})["source_frame_index"]) for transition in after] == [
        5,
        6,
        7,
    ]
    assert [transition.reward for transition in after] == [-1.0, -1.0, 0.0]
    assert [(transition.info or {})["success"] for transition in after] == [
        False,
        False,
        True,
    ]
    assert all(
        (transition.info or {})["policy_section_end_reason"] == "success"
        for transition in after
    )


@pytest.mark.parametrize(
    "terminal_reason",
    ["env_done", "max_steps", "manual_reset", "worker_exception"],
)
def test_non_success_final_policy_section_is_kept_as_failure(
    terminal_reason: str,
) -> None:
    streams = _build(
        _episode([False, False, False], terminal_reason=terminal_reason)
    )

    assert len(streams.policy_bc) == 3
    assert [transition.reward for transition in streams.policy_bc] == [-1.0] * 3
    assert all(
        (transition.info or {})["success"] is False
        for transition in streams.policy_bc
    )
    assert all(
        not bool((transition.info or {})["is_success_trajectory"])
        for transition in streams.policy_bc
    )
    assert all(
        (transition.info or {})["policy_section_end_reason"] == terminal_reason
        for transition in streams.policy_bc
    )
    assert streams.stats["keep_reasons"][terminal_reason] == 1


def test_short_non_success_policy_section_is_still_dropped() -> None:
    streams = _build(_episode([False], terminal_reason="env_done"))

    assert streams.policy_bc == []
    assert streams.stats["policy_sections_dropped_short"] == 1
