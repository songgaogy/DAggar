from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from robosuite.pipeline.workflow import RunLayout, create_run, load_json, write_json_atomic
from robosuite.pipeline.workflow import runner
from robosuite.pipeline.modules.training.discriminator.encoder import (
    resolve_saved_dynamics_config,
)


def _layout(tmp_path: Path) -> RunLayout:
    layout = RunLayout(tmp_path / "run")
    create_run(layout, task_name="PickPlaceCereal")
    cfg = runner.load_default_config(task="PickPlaceCereal")
    cfg.task.inputs.base_policy_checkpoint = str(layout.checkpoints_dir / "base.pt")
    cfg.task.inputs.parent_discriminator_checkpoint = str(layout.checkpoints_dir / "disc.pth")
    cfg.task.inputs.dynamics_encoder_checkpoint = str(layout.checkpoints_dir / "encoder.pth")
    cfg.task.inputs.initial_vast_checkpoint = str(layout.checkpoints_dir / "vast.pt")
    cfg.task.inputs.expert_data = str(layout.data_dir / "expert")
    cfg.task.inputs.discriminator_pretrain_data = str(layout.data_dir / "pretrain")
    cfg.task.inputs.vast_warmup_transitions = str(tmp_path / "warmup.pt")
    cfg.task.policy.flow.model.image_encoder.pretrained_path = str(
        layout.checkpoints_dir / "flow_image_encoder.pth"
    )
    cfg.task.policy.flow.model.language_encoder.pretrained_name = str(
        layout.data_dir / "language_encoder"
    )
    for path in (
        Path(cfg.task.inputs.base_policy_checkpoint),
        Path(cfg.task.inputs.parent_discriminator_checkpoint),
        Path(cfg.task.inputs.dynamics_encoder_checkpoint),
        Path(cfg.task.inputs.initial_vast_checkpoint),
        Path(cfg.task.inputs.vast_warmup_transitions),
        Path(cfg.task.policy.flow.model.image_encoder.pretrained_path),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"input")
    Path(cfg.task.inputs.expert_data).mkdir(parents=True)
    Path(cfg.task.inputs.discriminator_pretrain_data).mkdir(parents=True)
    Path(cfg.task.policy.flow.model.language_encoder.pretrained_name).mkdir(parents=True)
    OmegaConf.save(cfg, layout.config_path, resolve=True)
    manifest = load_json(layout.manifest_path)
    manifest["inputs"] = [
        {
            "name": name,
            "storage_mode": "snapshot",
            "snapshot_path": path.relative_to(layout.root).as_posix(),
            "source": str(path),
        }
        for name, path in (
            ("base_policy", Path(cfg.task.inputs.base_policy_checkpoint)),
            ("parent_discriminator", Path(cfg.task.inputs.parent_discriminator_checkpoint)),
            ("dynamics_encoder", Path(cfg.task.inputs.dynamics_encoder_checkpoint)),
            ("initial_vast", Path(cfg.task.inputs.initial_vast_checkpoint)),
            ("expert_data", Path(cfg.task.inputs.expert_data)),
            ("discriminator_pretrain", Path(cfg.task.inputs.discriminator_pretrain_data)),
            (
                "flow_image_encoder",
                Path(cfg.task.policy.flow.model.image_encoder.pretrained_path),
            ),
            (
                "language_encoder",
                Path(cfg.task.policy.flow.model.language_encoder.pretrained_name),
            ),
        )
    ]
    manifest["inputs"].append(
        {
            "name": "vast_warmup_transitions",
            "storage_mode": "external",
            "snapshot_path": None,
            "source": str(Path(cfg.task.inputs.vast_warmup_transitions)),
        }
    )
    write_json_atomic(layout.manifest_path, manifest)
    return layout


def _fake_collection(cfg) -> None:
    output = Path(cfg.offline_collect.output_dir) / str(cfg.offline_collect.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"final": True}, output)


def _fake_disc(cfg) -> None:
    run_dir = Path(cfg.offline.discriminator_finetune.run_dir)
    checkpoint = run_dir / "checkpoints" / "pu_bce_head_finetuned.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"disc")
    (run_dir / "run_info.json").write_text(
        json.dumps(
            {
                "finetune_config": {"round_mixtures": {}},
                "data": {"round_feature_caches": []},
            }
        ),
        encoding="utf-8",
    )


def _fake_offline(cfg) -> None:
    run_dir = Path(cfg.offline.run_dir)
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True)
    if cfg.offline.execution_stage == "vast":
        (checkpoints / "vast_state_finetuned.pt").write_bytes(b"vast")
        info = {"vast_buffer_stats": {"valid_windows": 8}}
    else:
        (checkpoints / "latest.pt").write_bytes(b"policy")
        info = {"stream_stats": {"policy_bc_transitions": 8}}
    (run_dir / "run_info.json").write_text(json.dumps(info), encoding="utf-8")


def test_two_round_orchestration_and_dependency_order(tmp_path, monkeypatch) -> None:
    layout = _layout(tmp_path)
    latest_cfg = runner.load_default_config(task="PickPlaceCereal")
    latest_cfg.collection.num_episodes = 3

    def latest_config(*, task: str):
        assert task == "PickPlaceCereal"
        return OmegaConf.create(OmegaConf.to_container(latest_cfg, resolve=True))

    monkeypatch.setattr(runner, "load_default_config", latest_config)
    collection_seeds = []

    def collect(cfg) -> None:
        collection_seeds.append(int(cfg.seed))
        _fake_collection(cfg)

    monkeypatch.setattr(runner, "run_collection", collect)
    monkeypatch.setattr(runner, "run_discriminator_finetune", _fake_disc)
    monkeypatch.setattr(runner, "run_offline_training", _fake_offline)

    runner.run_online(layout)
    with pytest.raises(RuntimeError, match="next stage is 'disc'"):
        runner.train(layout, requested_stage="policy")
    runner.train(layout, requested_stage="all")
    runner.run_online(layout)

    state = load_json(layout.state_path)
    assert state["active_round"] == 1
    assert state["active_stage"] == "disc"
    assert state["rounds"]["000"]["stages"]["policy"]["status"] == "completed"
    assert state["rounds"]["001"]["stages"]["collection"]["status"] == "completed"
    assert collection_seeds == [42, 45]
    assert state["rounds"]["000"]["stages"]["collection"]["inputs"]["collection_seed"] == 42
    assert state["rounds"]["001"]["stages"]["collection"]["inputs"]["collection_seed"] == 45
    assert (
        state["rounds"]["001"]["stages"]["collection"]["inputs"]["policy_checkpoint"]
        == str(layout.round_dir(0) / "policy" / "checkpoints" / "latest.pt")
    )


def test_refresh_run_config_uses_latest_settings_and_manifest_inputs(
    tmp_path, monkeypatch
) -> None:
    layout = _layout(tmp_path)
    latest = runner.load_default_config(task="PickPlaceCereal")
    latest.seed = 777
    latest.task.policy.num_steps = 1234
    latest.task.inputs.base_policy_checkpoint = "/stale/base.pt"
    latest.task.inputs.vast_warmup_transitions = "/stale/warmup.pt"

    monkeypatch.setattr(
        runner,
        "load_default_config",
        lambda *, task: OmegaConf.create(OmegaConf.to_container(latest, resolve=True)),
    )

    refreshed = runner.refresh_run_config(layout)

    assert refreshed.seed == 777
    assert refreshed.task.policy.num_steps == 1234
    assert Path(refreshed.task.inputs.base_policy_checkpoint) == (
        layout.checkpoints_dir / "base.pt"
    )
    assert Path(refreshed.task.inputs.vast_warmup_transitions) == tmp_path / "warmup.pt"
    assert Path(refreshed.task.evaluation.data_root).is_absolute()
    persisted = runner.load_run_config(layout)
    assert persisted.seed == 777
    assert persisted.task.inputs.base_policy_checkpoint == str(
        layout.checkpoints_dir / "base.pt"
    )


def test_collection_guidance_omegas_reach_stage_config() -> None:
    cfg = runner.load_default_config(task="PickPlaceCereal")
    assert list(cfg.collection.guidance_omegas) == [0.0, 0.1, 0.2, 0.5]

    cfg.collection.guidance_omegas = [0.75, 1.25]
    stage_cfg = runner.collection_stage_config(cfg, round_index=0)

    assert float(stage_cfg.algorithm.dipole.guidance_omega) == 0.75
    assert list(stage_cfg.offline_collect.guidance_omegas) == [0.75, 1.25]


@pytest.mark.parametrize("guidance_omegas", [[], [0.0, float("inf")], [float("nan")]])
def test_collection_guidance_omegas_must_be_non_empty_and_finite(
    guidance_omegas,
) -> None:
    cfg = runner.load_default_config(task="PickPlaceCereal")
    cfg.collection.guidance_omegas = guidance_omegas

    with pytest.raises(ValueError, match="collection.guidance_omegas"):
        runner.collection_stage_config(cfg, round_index=0)


def test_all_training_stages_refresh_latest_config(tmp_path, monkeypatch) -> None:
    layout = _layout(tmp_path)
    base = runner.load_default_config(task="PickPlaceCereal")
    refresh_calls = 0

    def latest_config(*, task: str):
        nonlocal refresh_calls
        assert task == "PickPlaceCereal"
        refresh_calls += 1
        cfg = OmegaConf.create(OmegaConf.to_container(base, resolve=True))
        cfg.task.discriminator.epochs = 10 + refresh_calls
        cfg.task.vast.num_steps = 100 + refresh_calls
        cfg.task.policy.num_steps = 1000 + refresh_calls
        return cfg

    seen = {}

    def disc(cfg) -> None:
        seen["disc_epochs"] = int(cfg.offline.discriminator_finetune.epochs)
        _fake_disc(cfg)

    def offline(cfg) -> None:
        if cfg.offline.execution_stage == "vast":
            seen["vast_steps"] = int(cfg.offline.vast_finetune.num_steps)
        else:
            seen["policy_steps"] = int(cfg.offline.num_train_steps)
        _fake_offline(cfg)

    monkeypatch.setattr(runner, "load_default_config", latest_config)
    monkeypatch.setattr(runner, "run_collection", _fake_collection)
    monkeypatch.setattr(runner, "run_discriminator_finetune", disc)
    monkeypatch.setattr(runner, "run_offline_training", offline)

    runner.run_online(layout)
    runner.train(layout, requested_stage="all")

    assert refresh_calls == 4
    assert seen == {
        "disc_epochs": 12,
        "vast_steps": 103,
        "policy_steps": 1004,
    }


def test_partial_collection_is_retained_and_reused(tmp_path, monkeypatch) -> None:
    layout = _layout(tmp_path)
    attempts = []
    collection_seeds = []

    def collect(cfg) -> None:
        output = Path(cfg.offline_collect.output_dir) / str(cfg.offline_collect.output_file)
        attempts.append(output.exists())
        collection_seeds.append(int(cfg.seed))
        torch.save({"final": len(attempts) > 1}, output)

    monkeypatch.setattr(runner, "run_collection", collect)

    with pytest.raises(RuntimeError, match="stopped before"):
        runner.run_online(layout)
    assert layout.round_data_dir(0).joinpath("episodes.partial.pt").is_file()
    runner.run_online(layout)

    state = load_json(layout.state_path)
    attempts_state = state["rounds"]["000"]["stages"]["collection"]["attempts"]
    assert attempts == [False, True]
    assert collection_seeds == [42, 42]
    assert [item["status"] for item in attempts_state] == ["failed", "completed"]


def test_dynamics_config_resolves_from_checkpoint_run_root(tmp_path) -> None:
    checkpoint = tmp_path / "dynamics" / "checkpoint" / "model.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"weights")
    config = checkpoint.parent.parent / "hydra.yaml"
    config.write_text("frameskip: 8\n", encoding="utf-8")

    assert resolve_saved_dynamics_config(checkpoint) == config.resolve()


def test_collection_reuses_final_file_after_interrupted_publish(tmp_path, monkeypatch) -> None:
    layout = _layout(tmp_path)
    final_path = layout.round_data_dir(0) / "episodes.pt"
    torch.save({"final": True}, final_path)
    monkeypatch.setattr(
        runner,
        "run_collection",
        lambda _cfg: pytest.fail("completed collection must not run again"),
    )

    assert runner.run_online(layout) == final_path
    assert load_json(layout.state_path)["active_stage"] == "disc"


def test_checkpoint_publish_overwrites_failed_attempt_atomically(tmp_path) -> None:
    layout = _layout(tmp_path)
    for attempt, content in ((1, b"first"), (2, b"retry")):
        work_dir = layout.attempt_dir(0, "disc", attempt) / "work"
        checkpoint = work_dir / "checkpoints" / "head.pth"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(content)
        published = runner._publish_attempt(
            layout,
            0,
            "disc",
            attempt,
            work_dir,
            checkpoint,
        )

    assert published.read_bytes() == b"retry"


def test_initialize_run_records_dependencies_and_references_warmup(tmp_path) -> None:
    source_root = tmp_path / "source"
    evaluation_root = tmp_path / "benchmark"
    evaluation_root.mkdir()
    dynamics_root = source_root / "dynamics"
    encoder = dynamics_root / "checkpoint" / "model.pth"
    normalizer = dynamics_root / "normalizer.pth"
    backbone = source_root / "dinov3"
    directories = {
        "expert": source_root / "expert",
        "pretrain": source_root / "pretrain",
        "language": source_root / "clip",
        "backbone": backbone,
    }
    for directory in directories.values():
        directory.mkdir(parents=True)
        (directory / "asset.bin").write_bytes(b"asset")
    encoder.parent.mkdir(parents=True)
    encoder.write_bytes(b"encoder")
    normalizer.write_bytes(b"normalizer")
    (dynamics_root / "hydra.yaml").write_text(
        f"encoder:\n  model_path: {backbone}\n",
        encoding="utf-8",
    )

    def file(name: str) -> Path:
        path = source_root / name
        path.write_bytes(name.encode())
        return path

    cfg = runner.load_default_config(task="PickPlaceCereal")
    cfg.storage.output_root = str(tmp_path / "outputs")
    cfg.storage.input_copy.prefer_reflink = False
    cfg.task.inputs.base_policy_checkpoint = str(file("base.pt"))
    cfg.task.inputs.parent_discriminator_checkpoint = str(file("disc.pth"))
    cfg.task.inputs.dynamics_encoder_checkpoint = str(encoder)
    cfg.task.inputs.initial_vast_checkpoint = str(file("vast.pt"))
    cfg.task.inputs.expert_data = str(directories["expert"])
    cfg.task.inputs.discriminator_pretrain_data = str(directories["pretrain"])
    warmup_source = file("warmup.pt").resolve()
    cfg.task.inputs.vast_warmup_transitions = str(warmup_source)
    cfg.task.policy.flow.model.image_encoder.pretrained_path = str(file("resnet.pth"))
    cfg.task.policy.flow.model.language_encoder.pretrained_name = str(directories["language"])
    cfg.task.evaluation.data_root = str(evaluation_root)

    layout = runner.initialize_run(run_name="self_contained", cfg=cfg)
    resolved = runner.load_run_config(layout)

    assert Path(resolved.task.inputs.vast_warmup_transitions) == warmup_source
    assert not (layout.data_dir / "vast_warmup").exists()
    assert Path(resolved.task.inputs.base_policy_checkpoint).is_file()
    assert Path(resolved.task.inputs.dynamics_encoder_checkpoint).is_file()
    assert Path(resolved.task.policy.flow.model.image_encoder.pretrained_path).is_file()
    assert Path(resolved.task.policy.flow.model.language_encoder.pretrained_name).is_dir()
    assert Path(resolved.task.evaluation.data_root) == evaluation_root.resolve()
    assert (layout.checkpoints_dir / "dynamics" / "hydra.yaml").is_file()
    assert (layout.checkpoints_dir / "dynamics" / "normalizer.pth").is_file()
    assert (layout.checkpoints_dir / "dynamics" / "backbone" / "asset.bin").is_file()
    manifest_inputs = load_json(layout.manifest_path)["inputs"]
    assert len(manifest_inputs) == 12
    warmup_record = next(
        item for item in manifest_inputs if item["name"] == "vast_warmup_transitions"
    )
    assert warmup_record["storage_mode"] == "external"
    assert warmup_record["source"] == str(warmup_source)
    assert warmup_record["snapshot_path"] is None

    source_root.rename(tmp_path / "source-moved")
    assert Path(resolved.task.inputs.base_policy_checkpoint).is_file()
    assert not warmup_source.exists()
