from __future__ import annotations

import numpy as np
import torch

from robosuite.pipeline.common.episodes import load_round_episode_payloads, merge_episode_payloads
from robosuite.pipeline.modules.training.dipole.episode_dataset import build_offline_transitions


def _payload(*, value: float, episode_index: int) -> dict:
    length = 3
    actions = np.full((length, 2), value, dtype=np.float32)
    state = np.full((length, 4), value, dtype=np.float32)
    images = np.full((length, 4, 4, 3), int(value), dtype=np.uint8)
    return {
        "schema_version": 1,
        "task_name": "PickPlaceCereal",
        "camera_names": ["agentview"],
        "img_height": 4,
        "img_width": 4,
        "action_dim": 2,
        "episodes": [
            {
                "episode_index": episode_index,
                "obs": {"state": state, "agentview": images},
                "next_obs": {"state": state.copy(), "agentview": images.copy()},
                "executed_action": actions,
                "policy_action": actions.copy(),
                "is_intervention": np.zeros(length, dtype=np.bool_),
                "success": np.array([False, False, True]),
                "done": np.array([False, False, True]),
                "terminal_reason": "success",
            }
        ],
    }


def test_cumulative_payload_preserves_round_and_episode_boundaries() -> None:
    merged = merge_episode_payloads(
        [_payload(value=1.0, episode_index=7), _payload(value=2.0, episode_index=9)],
        source_rounds=[0, 1],
    )
    streams = build_offline_transitions(
        merged,
        action_horizon=2,
        reward_success=0.0,
        reward_fail=-1.0,
        include_policy_action_neg=True,
    )

    assert merged["round_stats"] == [
        {"round": 0, "episodes": 1, "transitions": 3},
        {"round": 1, "episodes": 1, "transitions": 3},
    ]
    assert [episode["source_round"] for episode in merged["episodes"]] == [0, 1]
    assert [episode["source_episode_index"] for episode in merged["episodes"]] == [7, 9]
    assert {transition.info["source_round"] for transition in streams.policy_bc} == {0, 1}
    section_ids = [transition.info["episode_index"] for transition in streams.policy_bc]
    assert section_ids[:3] == [0, 0, 0]
    assert section_ids[3:] == [1, 1, 1]


def test_repeated_stage_load_reuses_validated_payload(tmp_path, monkeypatch) -> None:
    source = tmp_path / "episodes.pt"
    torch.save(_payload(value=1.0, episode_index=0), source)
    original_load = torch.load
    calls = []

    def counted_load(*args, **kwargs):
        calls.append(args[0])
        return original_load(*args, **kwargs)

    monkeypatch.setattr("robosuite.pipeline.common.episodes.torch.load", counted_load)

    first = load_round_episode_payloads([source])
    second = load_round_episode_payloads([source])

    assert first is second
    assert calls == [source.resolve()]

