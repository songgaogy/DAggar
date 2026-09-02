from __future__ import annotations

import json
from pathlib import Path

import pytest

from robosuite.pipeline.workflow import (
    InputReference,
    InputSnapshot,
    RunLayout,
    complete_stage,
    create_run,
    fail_stage,
    load_json,
    open_next_round,
    record_artifact,
    recover_interrupted_stage,
    rollback_run_for_retrain,
    snapshot_inputs,
    start_stage,
    write_json_atomic,
)


def _completed_two_round_run(tmp_path: Path) -> RunLayout:
    layout = RunLayout(tmp_path / "trial_20260101_000000")
    create_run(layout, task_name="PickPlaceCereal")
    layout.config_path.write_text(
        f"run_root: {layout.root}\n",
        encoding="utf-8",
    )
    (layout.inputs_dir / "metadata.json").write_text(
        json.dumps({"immutable_source": str(layout.root)}),
        encoding="utf-8",
    )

    for round_index in range(2):
        for stage in ("collection", "disc", "vast", "policy"):
            start_stage(
                layout,
                round_index,
                stage,
                inputs={"run_root": str(layout.root)},
            )
            if stage == "collection":
                artifact = layout.round_data_dir(round_index) / "episodes.pt"
                output_name = "episodes"
            else:
                artifact = layout.stage_dir(round_index, stage) / "checkpoints" / f"{stage}.pt"
                output_name = "checkpoint"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes(f"{round_index}:{stage}".encode())
            record_artifact(
                layout,
                artifact,
                round_index=round_index,
                stage=stage,
            )
            complete_stage(
                layout,
                round_index,
                stage,
                outputs={output_name: artifact.relative_to(layout.root).as_posix()},
                parent_checkpoint=str(layout.root / "inputs" / "parent.pt"),
            )
        if round_index == 0:
            open_next_round(layout)

    for index in range(3):
        cache = layout.cache_dir / "discriminator_features" / f"round_{index:03d}.pt"
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(b"cache")
    for stage in ("disc", "vast"):
        marker = layout.eval_vis_dir(1) / stage / "summary.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"run_root": str(layout.root)}), encoding="utf-8")
    for stage in ("disc", "vast"):
        metadata = layout.stage_dir(1, stage) / "run_info.json"
        metadata.write_text(json.dumps({"run_root": str(layout.root)}), encoding="utf-8")
    (layout.round_data_dir(1) / "episodes.meta.json").write_text(
        json.dumps({"run_root": str(layout.root)}),
        encoding="utf-8",
    )
    return layout


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


@pytest.mark.parametrize(
    ("stage", "retained_stages", "keep_disc_eval", "keep_vast_eval", "keep_round_cache"),
    [
        ("disc", ("collection",), False, False, False),
        ("vast", ("collection", "disc"), True, False, True),
        ("policy", ("collection", "disc", "vast"), True, True, True),
    ],
)
def test_rollback_run_for_retrain_retains_strict_prefix_in_place(
    tmp_path: Path,
    stage: str,
    retained_stages: tuple[str, ...],
    keep_disc_eval: bool,
    keep_vast_eval: bool,
    keep_round_cache: bool,
) -> None:
    layout = _completed_two_round_run(tmp_path)
    config_before = layout.config_path.read_bytes()
    immutable_metadata = (layout.inputs_dir / "metadata.json").read_bytes()
    state_before = load_json(layout.state_path)
    manifest_before = load_json(layout.manifest_path)
    state_before["forked_from"] = {"run_root": "/previous/run"}
    manifest_before["forked_from"] = {"run_root": "/previous/run"}
    write_json_atomic(layout.state_path, state_before)
    write_json_atomic(layout.manifest_path, manifest_before)

    rollback_run_for_retrain(layout, round_index=1, stage=stage)

    state = load_json(layout.state_path)
    assert state["active_round"] == 1
    assert state["active_stage"] == stage
    assert state["created_at"] == state_before["created_at"]
    assert state["forked_from"] == {"run_root": "/previous/run"}
    assert set(state["rounds"]) == {"000", "001"}
    assert state["rounds"]["001"]["stages"]["collection"]["inputs"][
        "run_root"
    ] == str(layout.root)
    for stage_name in retained_stages:
        assert state["rounds"]["001"]["stages"][stage_name]["status"] == "completed"
    for stage_name in ("disc", "vast", "policy"):
        if stage_name not in retained_stages:
            stage_state = state["rounds"]["001"]["stages"][stage_name]
            assert stage_state == {
                "status": "pending",
                "attempts": [],
                "inputs": {},
                "outputs": {},
                "parent_checkpoint": None,
                "sampling_weights": {},
                "effective_losses": {},
                "cache_keys": {},
            }
            assert list(layout.stage_dir(1, stage_name).iterdir()) == []

    assert not layout.round_dir(2).exists()
    assert layout.disc_eval_vis_dir(1).exists() is keep_disc_eval
    assert layout.vast_eval_vis_dir(1).exists() is keep_vast_eval
    assert (
        layout.cache_dir / "discriminator_features" / "round_001.pt"
    ).exists() is keep_round_cache
    assert not (layout.cache_dir / "discriminator_features" / "round_002.pt").exists()
    assert layout.config_path.read_bytes() == config_before
    assert (layout.inputs_dir / "metadata.json").read_bytes() == immutable_metadata

    manifest = load_json(layout.manifest_path)
    assert manifest["created_at"] == manifest_before["created_at"]
    assert manifest["forked_from"] == {"run_root": "/previous/run"}
    assert all(
        int(record["round"]) < 1
        or record["stage"] in retained_stages
        for record in manifest["artifacts"]
    )


def test_rollback_run_for_retrain_removes_later_rounds(tmp_path: Path) -> None:
    layout = _completed_two_round_run(tmp_path)

    rollback_run_for_retrain(layout, round_index=0, stage="policy")

    state = load_json(layout.state_path)
    manifest = load_json(layout.manifest_path)
    assert set(state["rounds"]) == {"000"}
    assert not layout.round_dir(1).exists()
    assert all(int(record["round"]) == 0 for record in manifest["artifacts"])
    assert all(record["stage"] != "policy" for record in manifest["artifacts"])


def test_rollback_run_for_retrain_rejects_running_run(tmp_path: Path) -> None:
    layout = RunLayout(tmp_path / "trial_20260101_000000")
    create_run(layout, task_name="PickPlaceCereal")
    layout.config_path.write_text("task: PickPlaceCereal\n", encoding="utf-8")
    start_stage(layout, 0, "collection")

    with pytest.raises(RuntimeError, match="running stage"):
        rollback_run_for_retrain(layout, round_index=0, stage="disc")


def test_rollback_run_for_retrain_rejects_missing_round(tmp_path: Path) -> None:
    layout = RunLayout(tmp_path / "trial_20260101_000000")
    create_run(layout, task_name="PickPlaceCereal")
    layout.config_path.write_text("task: PickPlaceCereal\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Round 001 does not exist"):
        rollback_run_for_retrain(layout, round_index=1, stage="disc")


@pytest.mark.parametrize("missing_name", ["config_resolved.yaml", "manifest.json", "state.json"])
def test_rollback_run_for_retrain_rejects_missing_metadata(
    tmp_path: Path,
    missing_name: str,
) -> None:
    layout = _completed_two_round_run(tmp_path)
    (layout.root / missing_name).unlink()

    with pytest.raises(FileNotFoundError, match="Run metadata does not exist"):
        rollback_run_for_retrain(layout, round_index=1, stage="policy")


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


def test_snapshot_inputs_records_external_input_without_copying(
    tmp_path: Path,
) -> None:
    source_checkpoint = tmp_path / "source" / "policy.pt"
    source_checkpoint.parent.mkdir()
    source_checkpoint.write_bytes(b"checkpoint")
    external_warmup = tmp_path / "source" / "warmup.pt"
    external_warmup.write_bytes(b"warmup-data")
    layout = RunLayout(tmp_path / "run")
    create_run(layout, task_name="PickPlaceCereal")

    records = snapshot_inputs(
        layout,
        [InputSnapshot("base_policy", source_checkpoint, "checkpoints/base_policy.pt")],
        external_inputs=[InputReference("vast_warmup_transitions", external_warmup)],
        prefer_reflink=False,
    )

    snapshot_record, external_record = records
    assert snapshot_record["storage_mode"] == "snapshot"
    assert external_record["storage_mode"] == "external"
    assert external_record["source"] == str(external_warmup.resolve())
    assert external_record["snapshot_path"] is None
    assert external_record["size_bytes"] == len(b"warmup-data")
    assert len(external_record["sha256"]) == 64
    assert not (layout.data_dir / "vast_warmup").exists()


def test_snapshot_inputs_reports_copy_and_hash_byte_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_checkpoint = tmp_path / "source.pt"
    source_checkpoint.write_bytes(b"checkpoint")
    external_warmup = tmp_path / "warmup.pt"
    external_warmup.write_bytes(b"warmup")
    layout = RunLayout(tmp_path / "run")
    create_run(layout, task_name="PickPlaceCereal")
    progress_calls: dict[str, object] = {"updates": []}

    class FakeProgress:
        def __init__(self, **kwargs) -> None:
            progress_calls["total"] = kwargs["total"]

        def update(self, amount: int) -> None:
            progress_calls["updates"].append(amount)

        def close(self) -> None:
            progress_calls["closed"] = True

    monkeypatch.setattr(
        "robosuite.pipeline.workflow.state.tqdm",
        lambda **kwargs: FakeProgress(**kwargs),
    )

    snapshot_inputs(
        layout,
        [InputSnapshot("checkpoint", source_checkpoint, "checkpoints/checkpoint.pt")],
        external_inputs=[InputReference("warmup", external_warmup)],
        prefer_reflink=False,
    )

    expected_total = 3 * source_checkpoint.stat().st_size + external_warmup.stat().st_size
    assert progress_calls["total"] == expected_total
    assert sum(progress_calls["updates"]) == expected_total
    assert progress_calls["closed"] is True


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


def test_snapshot_hash_mismatch_removes_created_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.pt"
    source.write_bytes(b"checkpoint")
    layout = RunLayout(tmp_path / "run")
    create_run(layout, task_name="PickPlaceCereal")

    monkeypatch.setattr(
        "robosuite.pipeline.workflow.state.sha256_file",
        lambda path, **_kwargs: "source" if Path(path) == source else "destination",
    )

    with pytest.raises(OSError, match="SHA-256 mismatch"):
        snapshot_inputs(
            layout,
            [InputSnapshot("checkpoint", source, "checkpoints/checkpoint.pt")],
            prefer_reflink=False,
        )

    assert not (layout.checkpoints_dir / "checkpoint.pt").exists()
    assert load_json(layout.manifest_path)["inputs"] == []


def test_external_hash_failure_preserves_source_and_removes_created_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.pt"
    source.write_bytes(b"checkpoint")
    external = tmp_path / "warmup.pt"
    external.write_bytes(b"warmup")
    layout = RunLayout(tmp_path / "run")
    create_run(layout, task_name="PickPlaceCereal")

    def fail_external_hash(*_args, **_kwargs):
        raise OSError("external hash failed")

    monkeypatch.setattr(
        "robosuite.pipeline.workflow.state._hash_input", fail_external_hash
    )

    with pytest.raises(OSError, match="external hash failed"):
        snapshot_inputs(
            layout,
            [InputSnapshot("checkpoint", source, "checkpoints/checkpoint.pt")],
            external_inputs=[InputReference("warmup", external)],
            prefer_reflink=False,
        )

    assert external.read_bytes() == b"warmup"
    assert not (layout.checkpoints_dir / "checkpoint.pt").exists()
    assert load_json(layout.manifest_path)["inputs"] == []


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
