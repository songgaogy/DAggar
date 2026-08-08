from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from robosuite.pipeline.collection import (
    _EPISODE_SHARD_FORMAT,
    _EpisodeBuilder,
    _checkpoint_has_negative_policy,
    _EpisodeStats,
    _finalize_collection_checkpoint,
    _load_collection_checkpoint,
    _metadata_from_stats,
    _save_episode_shard,
    _select_action_candidate,
    _validate_resume_guidance,
    _write_collection_manifest,
)
from robosuite.pipeline.common.episodes import validate_offline_payload


def _episode(index: int, *, intervention: bool = False) -> dict:
    length = 2
    action = np.full((length, 2), float(index), dtype=np.float32)
    return {
        "episode_index": index,
        "episode_seed": 100 + index,
        "episode_step": np.arange(length, dtype=np.int32),
        "terminal_reason": "success",
        "obs": {
            "state": np.full((length, 3), index, dtype=np.float32),
            "agentview": np.full((length, 4, 4, 3), index, dtype=np.uint8),
        },
        "next_obs": {
            "state": np.full((length, 3), index + 1, dtype=np.float32),
            "agentview": np.full((length, 4, 4, 3), index + 1, dtype=np.uint8),
        },
        "executed_action": action.copy(),
        "policy_action": action.copy(),
        "human_action": np.zeros_like(action),
        "is_intervention": np.asarray([False, intervention], dtype=np.bool_),
        "gt_fail": np.asarray([False, intervention], dtype=np.bool_),
        "reward": np.asarray([0.0, 1.0], dtype=np.float32),
        "done": np.asarray([False, True], dtype=np.bool_),
        "success": np.asarray([False, True], dtype=np.bool_),
        "nnpu_pred": np.asarray([0, 0], dtype=np.int8),
        "nnpu_score": np.asarray([0.1, 0.2], dtype=np.float32),
        "nnpu_threshold": np.asarray([0.0, 0.0], dtype=np.float32),
        "grasp_penalty": np.asarray([0.0, 0.0], dtype=np.float32),
    }


def _metadata(output_path: Path, episodes: list[dict]) -> dict:
    return _metadata_from_stats(
        payload_path=output_path,
        task_name="PickPlaceCereal",
        policy_checkpoint=Path("/tmp/policy.pt"),
        nnpu_checkpoint=Path("/tmp/discriminator.pth"),
        camera_names=["agentview"],
        img_height=4,
        img_width=4,
        action_dim=2,
        stats=_EpisodeStats.from_episodes(episodes),
        started_at="2026-08-05T00:00:00",
        finished_at="2026-08-05T00:01:00",
        seed=42,
    )


def _runtime_payload(metadata: dict, *, final: bool = False) -> dict:
    return {
        **metadata,
        "policy_device": "cuda:0",
        "inference_device": "cuda:0",
        "disc_device": "cuda:0",
        "deterministic": False,
        "final": final,
    }


def test_episode_builder_records_candidate_selection_per_step() -> None:
    builder = _EpisodeBuilder(
        episode_index=0, episode_seed=42, camera_names=["agentview"]
    )
    obs = {
        "state": np.zeros(3, dtype=np.float32),
        "agentview": np.zeros((4, 4, 3), dtype=np.uint8),
    }
    common = {
        "obs": obs,
        "next_obs": obs,
        "executed_action": np.zeros(2, dtype=np.float32),
        "policy_action": np.zeros(2, dtype=np.float32),
        "human_action": np.zeros(2, dtype=np.float32),
        "is_intervention": False,
        "reward": 0.0,
        "done": False,
        "success": False,
        "nnpu_pred": 0,
        "nnpu_score": 0.1,
        "nnpu_threshold": 0.0,
        "grasp_penalty": None,
        "policy_candidate_omegas": np.asarray([0.0, 0.1], dtype=np.float32),
        "policy_candidate_nnpu_scores": np.asarray([0.5, -0.2], dtype=np.float32),
        "policy_selected_candidate_index": 1,
        "policy_selected_guidance_omega": 0.1,
        "policy_latency": {
            "context_ms": 1.0,
            "ode_ms": 2.0,
            "d2h_ms": 3.0,
            "policy_total_ms": 6.0,
            "discriminator_ms": 4.0,
            "selector_total_ms": 10.0,
        },
    }
    builder.append(**common, policy_chunk_start=True)
    builder.append(**common, policy_chunk_start=False)

    episode = builder.to_payload("max_steps")

    assert episode["policy_chunk_start"].tolist() == [True, False]
    assert episode["policy_candidate_omegas"].shape == (2, 2)
    assert episode["policy_candidate_nnpu_scores"].shape == (2, 2)
    assert episode["policy_selected_candidate_index"].tolist() == [1, 1]
    np.testing.assert_allclose(
        episode["policy_selected_guidance_omega"], [0.1, 0.1]
    )
    np.testing.assert_allclose(episode["policy_context_latency_ms"], [1.0, 1.0])
    np.testing.assert_allclose(episode["policy_ode_latency_ms"], [2.0, 2.0])
    np.testing.assert_allclose(episode["policy_d2h_latency_ms"], [3.0, 3.0])
    np.testing.assert_allclose(episode["policy_total_latency_ms"], [6.0, 6.0])
    np.testing.assert_allclose(episode["discriminator_latency_ms"], [4.0, 4.0])
    np.testing.assert_allclose(episode["selector_total_latency_ms"], [10.0, 10.0])


def test_candidate_selection_uses_lowest_score_and_first_tie() -> None:
    candidates = np.arange(24, dtype=np.float32).reshape(3, 4, 2)
    omegas = np.asarray([0.0, 0.1, 0.2], dtype=np.float32)

    selected, chunk, omega = _select_action_candidate(
        candidates, omegas, np.asarray([0.5, -0.2, -0.2], dtype=np.float32)
    )

    assert selected == 1
    np.testing.assert_array_equal(chunk, candidates[1])
    assert omega == pytest.approx(0.1)


def test_candidate_selection_supports_positive_only_and_rejects_nonfinite() -> None:
    candidates = np.zeros((1, 4, 2), dtype=np.float32)
    selected, chunk, omega = _select_action_candidate(
        candidates,
        np.asarray([0.0], dtype=np.float32),
        np.asarray([0.25], dtype=np.float32),
    )

    assert selected == 0
    np.testing.assert_array_equal(chunk, candidates[0])
    assert omega == 0.0
    with pytest.raises(RuntimeError, match="non-finite"):
        _select_action_candidate(candidates, np.asarray([0.0]), np.asarray([np.nan]))


def test_negative_policy_requires_dual_core_checkpoint() -> None:
    assert not _checkpoint_has_negative_policy({"model": {}})
    assert not _checkpoint_has_negative_policy({"core": {"core_pos": {}}})
    assert _checkpoint_has_negative_policy(
        {"core": {"core_pos": {}, "core_neg": {}}}
    )


def test_resume_guidance_requires_matching_protocol() -> None:
    checkpoint = {
        "configured_guidance_omegas": [0.0, 0.1],
        "effective_guidance_omegas": [0.0, 0.1],
    }
    _validate_resume_guidance(
        checkpoint,
        has_episodes=True,
        configured=[0.0, 0.1],
        effective=[0.0, 0.1],
    )

    with pytest.raises(ValueError, match="configured guidance"):
        _validate_resume_guidance(
            checkpoint,
            has_episodes=True,
            configured=[0.0, 0.2],
            effective=[0.0, 0.1],
        )
    with pytest.raises(ValueError, match="legacy partial"):
        _validate_resume_guidance(
            {}, has_episodes=True, configured=[0.0], effective=[0.0]
        )


def test_episode_checkpoints_serialize_only_the_new_episode(tmp_path, monkeypatch) -> None:
    output = tmp_path / "episodes.partial.pt"
    saved_payloads = []
    original_save = torch.save

    def recording_save(payload, path) -> None:
        saved_payloads.append(payload)
        original_save(payload, path)

    monkeypatch.setattr("robosuite.pipeline.collection.torch.save", recording_save)
    first = _episode(0)
    second = _episode(1)

    _save_episode_shard(output, shard_index=0, episode=first)
    _save_episode_shard(output, shard_index=1, episode=second)

    assert [payload["shard_index"] for payload in saved_payloads] == [0, 1]
    assert saved_payloads[0]["episode"] is first
    assert saved_payloads[1]["episode"] is second
    assert all("episodes" not in payload for payload in saved_payloads)


def test_manifest_resume_adopts_atomic_orphan_shard(tmp_path) -> None:
    output = tmp_path / "episodes.partial.pt"
    episodes = [_episode(0), _episode(1, intervention=True)]
    _save_episode_shard(output, shard_index=0, episode=episodes[0])
    metadata = _metadata(output, episodes[:1])
    _write_collection_manifest(
        output,
        payload=_runtime_payload(metadata),
        metadata=metadata,
        num_shards=1,
    )
    _save_episode_shard(output, shard_index=1, episode=episodes[1])

    manifest, resumed, final = _load_collection_checkpoint(output)

    assert manifest["checkpoint_format"] == _EPISODE_SHARD_FORMAT
    assert manifest["num_episodes"] == 1
    assert manifest["num_shards"] == 1
    assert "episodes" not in manifest
    assert [episode["episode_index"] for episode in resumed] == [0, 1]
    assert final is False
    assert not list(tmp_path.rglob("*.tmp"))


def test_resume_rejects_missing_or_noncontiguous_shards(tmp_path) -> None:
    output = tmp_path / "episodes.partial.pt"
    episodes = [_episode(0), _episode(1)]
    metadata = _metadata(output, episodes)
    _write_collection_manifest(
        output,
        payload=_runtime_payload(metadata),
        metadata=metadata,
        num_shards=2,
    )
    _save_episode_shard(output, shard_index=1, episode=episodes[1])

    with pytest.raises(RuntimeError, match="only 1 shards"):
        _load_collection_checkpoint(output)


def test_final_checkpoint_preserves_schema_and_cleanup_failure_is_nonfatal(
    tmp_path, monkeypatch, capsys
) -> None:
    output = tmp_path / "episodes.partial.pt"
    episodes = [_episode(0), _episode(1, intervention=True)]
    for index, episode in enumerate(episodes):
        _save_episode_shard(output, shard_index=index, episode=episode)
    metadata = _metadata(output, episodes)

    def fail_cleanup(_path) -> None:
        raise OSError("busy")

    monkeypatch.setattr("robosuite.pipeline.collection.shutil.rmtree", fail_cleanup)
    _, output_bytes = _finalize_collection_checkpoint(
        output,
        payload=_runtime_payload(metadata, final=True),
        metadata=metadata,
        episodes=episodes,
    )

    payload = torch.load(output, map_location="cpu", weights_only=False)
    validated = validate_offline_payload(payload)
    assert output_bytes == output.stat().st_size
    assert validated["num_episodes"] == 2
    assert validated["num_transitions"] == 4
    assert validated["num_intervention_transitions"] == 1
    assert json.loads(output.with_suffix(".meta.json").read_text())["num_episodes"] == 2
    assert "Failed to remove completed episode shards" in capsys.readouterr().out


def test_final_checkpoint_removes_temporary_shards(tmp_path) -> None:
    output = tmp_path / "episodes.partial.pt"
    episodes = [_episode(0)]
    _save_episode_shard(output, shard_index=0, episode=episodes[0])
    metadata = _metadata(output, episodes)

    _finalize_collection_checkpoint(
        output,
        payload=_runtime_payload(metadata, final=True),
        metadata=metadata,
        episodes=episodes,
    )

    assert output.is_file()
    assert not output.with_suffix(".shards").exists()


def test_legacy_incomplete_checkpoint_is_rejected(tmp_path) -> None:
    output = tmp_path / "episodes.partial.pt"
    torch.save({"final": False, "episodes": [_episode(0)]}, output)

    with pytest.raises(RuntimeError, match="Legacy monolithic partial"):
        _load_collection_checkpoint(output)


def test_completed_monolithic_checkpoint_remains_resumable(tmp_path) -> None:
    output = tmp_path / "episodes.partial.pt"
    episodes = [_episode(0)]
    metadata = _metadata(output, episodes)
    torch.save({**_runtime_payload(metadata, final=True), "episodes": episodes}, output)

    _, resumed, final = _load_collection_checkpoint(output)

    assert resumed[0]["episode_index"] == 0
    assert final is True
