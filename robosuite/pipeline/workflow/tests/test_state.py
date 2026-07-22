from __future__ import annotations

import json
from pathlib import Path

import pytest

from robosuite.pipeline.workflow import (
    InputSnapshot,
    RunLayout,
    complete_stage,
    create_run,
    fail_stage,
    load_json,
    open_next_round,
    record_artifact,
    recover_interrupted_stage,
    snapshot_inputs,
    start_stage,
    write_json_atomic,
)


def test_create_run_has_canonical_layout(tmp_path: Path) -> None:
    layout = RunLayout(tmp_path / "run")

    state = create_run(layout, task_name="PickPlaceCereal")

    assert state["active_round"] == 0
    assert state["active_stage"] == "collection"
    assert layout.state_path.is_file()
    assert layout.manifest_path.is_file()
    assert layout.round_data_dir(0).is_dir()
    assert layout.stage_dir(0, "disc").is_dir()
    assert layout.stage_dir(0, "vast").is_dir()
    assert layout.stage_dir(0, "policy").is_dir()


def test_eval_vis_paths_are_read_only_and_round_scoped(tmp_path: Path) -> None:
    layout = RunLayout(tmp_path / "run")
    create_run(layout, task_name="PickPlaceCereal")
    state_before = layout.state_path.read_bytes()
    manifest_before = layout.manifest_path.read_bytes()

    assert layout.eval_vis_dir(2) == layout.root / "rounds" / "002" / "eval_vis"
    assert layout.disc_eval_vis_dir(2) == (
        layout.root / "rounds" / "002" / "eval_vis" / "disc"
    )
    assert layout.vast_eval_vis_dir(2) == (
        layout.root / "rounds" / "002" / "eval_vis" / "vast"
    )
    assert not layout.eval_vis_dir(2).exists()
    assert layout.state_path.read_bytes() == state_before
    assert layout.manifest_path.read_bytes() == manifest_before


def test_stage_order_retry_and_next_round(tmp_path: Path) -> None:
    layout = RunLayout(tmp_path / "run")
    create_run(layout, task_name="PickPlaceCereal")

    with pytest.raises(RuntimeError, match="Expected round 0 stage collection"):
        start_stage(layout, 0, "disc")

    start_stage(layout, 0, "collection")
    complete_stage(layout, 0, "collection", outputs={"episodes": "rounds/000/data/episodes.pt"})
    first_attempt = start_stage(
        layout,
        0,
        "disc",
        inputs={"episodes": "rounds/000/data/episodes.pt"},
    )
    assert first_attempt["attempt"] == 1
    assert layout.attempt_dir(0, "disc", 1).is_dir()
    fail_stage(layout, 0, "disc", error="interrupted")
    second_attempt = start_stage(layout, 0, "disc")
    assert second_attempt["attempt"] == 2

    complete_stage(
        layout,
        0,
        "disc",
        outputs={"checkpoint": "rounds/000/disc/checkpoints/head.pth"},
        parent_checkpoint="inputs/checkpoints/parent_discriminator.pth",
        sampling_weights={"gt_positive": {"000": 1.0}},
        effective_losses={"gt_positive": 0.2, "gt_negative": 0.01},
        cache_keys={"encoder_features": "abc123"},
    )
    for stage in ("vast", "policy"):
        start_stage(layout, 0, stage)
        complete_stage(layout, 0, stage)

    round_one = open_next_round(layout)
    assert round_one["index"] == 1
    state = load_json(layout.state_path)
    assert state["active_round"] == 1
    assert state["active_stage"] == "collection"
    disc_state = state["rounds"]["000"]["stages"]["disc"]
    assert disc_state["inputs"]["episodes"] == "rounds/000/data/episodes.pt"
    assert [attempt["status"] for attempt in disc_state["attempts"]] == [
        "failed",
        "completed",
    ]
    assert disc_state["effective_losses"]["gt_positive"] == 0.2


def test_cannot_open_round_before_policy_completion(tmp_path: Path) -> None:
    layout = RunLayout(tmp_path / "run")
    create_run(layout, task_name="PickPlaceCereal")

    with pytest.raises(RuntimeError, match="before policy completion"):
        open_next_round(layout)


def test_snapshot_inputs_copies_files_and_records_per_file_hashes(tmp_path: Path) -> None:
    source_checkpoint = tmp_path / "source" / "policy.pt"
    source_checkpoint.parent.mkdir()
    source_checkpoint.write_bytes(b"checkpoint")
    source_data = tmp_path / "source" / "expert"
    source_data.mkdir()
    (source_data / "manifest.json").write_text('{"version": 1}', encoding="utf-8")
    (source_data / "shard.bin").write_bytes(b"data")
    layout = RunLayout(tmp_path / "run")
    create_run(layout, task_name="PickPlaceCereal")

    records = snapshot_inputs(
        layout,
        [
            InputSnapshot(
                "base_policy",
                source_checkpoint,
                "checkpoints/base_policy.pt",
            ),
            InputSnapshot("expert_data", source_data, "data/expert"),
        ],
        prefer_reflink=False,
    )

    assert (layout.checkpoints_dir / "base_policy.pt").read_bytes() == b"checkpoint"
    assert (layout.data_dir / "expert" / "shard.bin").read_bytes() == b"data"
    assert records[0]["files"][0]["copy_mode"] == "copy"
    assert {item["path"] for item in records[1]["files"]} == {
        "manifest.json",
        "shard.bin",
    }
    assert all(len(item["sha256"]) == 64 for record in records for item in record["files"])
    source_checkpoint.unlink()
    for child in source_data.iterdir():
        child.unlink()
    source_data.rmdir()
    assert (layout.checkpoints_dir / "base_policy.pt").is_file()
    assert (layout.data_dir / "expert" / "shard.bin").is_file()


def test_snapshot_inputs_falls_back_when_reflink_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.pt"
    source.write_bytes(b"checkpoint")
    layout = RunLayout(tmp_path / "run")
    create_run(layout, task_name="PickPlaceCereal")

    class FailedCopy:
        returncode = 1

    monkeypatch.setattr(
        "robosuite.pipeline.workflow.state.subprocess.run",
        lambda *args, **kwargs: FailedCopy(),
    )

    records = snapshot_inputs(
        layout,
        [InputSnapshot("checkpoint", source, "checkpoints/checkpoint.pt")],
    )

    assert records[0]["files"][0]["copy_mode"] == "copy"
    assert (layout.checkpoints_dir / "checkpoint.pt").read_bytes() == b"checkpoint"


def test_atomic_json_never_leaves_temporary_file(tmp_path: Path) -> None:
    layout = RunLayout(tmp_path / "run")
    create_run(layout, task_name="PickPlaceCereal")

    payload = json.loads(layout.state_path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert not list(layout.root.glob(".state.json.*"))


def test_published_artifact_hash_is_recorded(tmp_path: Path) -> None:
    layout = RunLayout(tmp_path / "run")
    create_run(layout, task_name="PickPlaceCereal")
    artifact = layout.round_data_dir(0) / "episodes.pt"
    artifact.write_bytes(b"episodes")

    record = record_artifact(
        layout,
        artifact,
        round_index=0,
        stage="collection",
    )

    manifest = load_json(layout.manifest_path)
    assert record["path"] == "rounds/000/data/episodes.pt"
    assert len(record["sha256"]) == 64
    assert manifest["artifacts"] == [record]


def test_artifact_record_is_idempotent_for_failed_publish_retry(tmp_path: Path) -> None:
    layout = RunLayout(tmp_path / "run")
    create_run(layout, task_name="PickPlaceCereal")
    artifact = layout.round_data_dir(0) / "episodes.pt"
    artifact.write_bytes(b"first")
    record_artifact(layout, artifact, round_index=0, stage="collection")
    artifact.write_bytes(b"retry")

    latest = record_artifact(layout, artifact, round_index=0, stage="collection")

    manifest = load_json(layout.manifest_path)
    assert len(manifest["artifacts"]) == 1
    assert manifest["artifacts"][0] == latest


def test_stale_running_attempt_is_recovered_as_failed(tmp_path: Path) -> None:
    layout = RunLayout(tmp_path / "run")
    create_run(layout, task_name="PickPlaceCereal")
    start_stage(layout, 0, "collection")
    state = load_json(layout.state_path)
    state["rounds"]["000"]["stages"]["collection"]["attempts"][-1][
        "process_id"
    ] = 999_999_999
    write_json_atomic(layout.state_path, state)

    assert recover_interrupted_stage(layout) is True
    recovered = load_json(layout.state_path)
    stage = recovered["rounds"]["000"]["stages"]["collection"]
    assert stage["status"] == "failed"
    assert stage["attempts"][-1]["status"] == "failed"


@pytest.mark.parametrize(
    "destination",
    ["/absolute.pt", "../outside.pt", "cache/not-an-input.pt"],
)
def test_snapshot_destination_must_stay_below_inputs(destination: str) -> None:
    with pytest.raises(ValueError):
        InputSnapshot("unsafe", Path("source.pt"), destination)
