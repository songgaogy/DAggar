from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from robosuite.pipeline.collection import (
    _EPISODE_SHARD_FORMAT,
    _EpisodeStats,
    _finalize_collection_checkpoint,
    _load_collection_checkpoint,
    _metadata_from_stats,
    _save_episode_shard,
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
