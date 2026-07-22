from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from robosuite.pipeline.workflow import RunLayout, create_run, load_json
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
    cfg.task.inputs.vast_warmup_transitions = str(layout.data_dir / "warmup.pt")
    for path in (
        Path(cfg.task.inputs.base_policy_checkpoint),
        Path(cfg.task.inputs.parent_discriminator_checkpoint),
        Path(cfg.task.inputs.dynamics_encoder_checkpoint),
        Path(cfg.task.inputs.initial_vast_checkpoint),
        Path(cfg.task.inputs.vast_warmup_transitions),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"input")
    Path(cfg.task.inputs.expert_data).mkdir(parents=True)
    Path(cfg.task.inputs.discriminator_pretrain_data).mkdir(parents=True)
    OmegaConf.save(cfg, layout.config_path, resolve=True)
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
    monkeypatch.setattr(runner, "run_collection", _fake_collection)
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
    assert (
        state["rounds"]["001"]["stages"]["collection"]["inputs"]["policy_checkpoint"]
        == str(layout.round_dir(0) / "policy" / "checkpoints" / "latest.pt")
    )


def test_partial_collection_is_retained_and_reused(tmp_path, monkeypatch) -> None:
    layout = _layout(tmp_path)
    attempts = []

    def collect(cfg) -> None:
        output = Path(cfg.offline_collect.output_dir) / str(cfg.offline_collect.output_file)
        attempts.append(output.exists())
        torch.save({"final": len(attempts) > 1}, output)

    monkeypatch.setattr(runner, "run_collection", collect)

    with pytest.raises(RuntimeError, match="stopped before"):
        runner.run_online(layout)
    assert layout.round_data_dir(0).joinpath("episodes.partial.pt").is_file()
    runner.run_online(layout)

    state = load_json(layout.state_path)
    attempts_state = state["rounds"]["000"]["stages"]["collection"]["attempts"]
    assert attempts == [False, True]
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


def test_initialize_run_snapshots_every_runtime_dependency(tmp_path) -> None:
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
    cfg.task.inputs.vast_warmup_transitions = str(file("warmup.pt"))
    cfg.task.policy.flow.model.image_encoder.pretrained_path = str(file("resnet.pth"))
    cfg.task.policy.flow.model.language_encoder.pretrained_name = str(directories["language"])
    cfg.task.evaluation.data_root = str(evaluation_root)

    layout = runner.initialize_run(run_name="self_contained", cfg=cfg)
    source_root.rename(tmp_path / "source-moved")
    resolved = runner.load_run_config(layout)

    assert Path(resolved.task.inputs.base_policy_checkpoint).is_file()
    assert Path(resolved.task.inputs.dynamics_encoder_checkpoint).is_file()
    assert Path(resolved.task.policy.flow.model.image_encoder.pretrained_path).is_file()
    assert Path(resolved.task.policy.flow.model.language_encoder.pretrained_name).is_dir()
    assert Path(resolved.task.evaluation.data_root) == evaluation_root.resolve()
    assert (layout.checkpoints_dir / "dynamics" / "hydra.yaml").is_file()
    assert (layout.checkpoints_dir / "dynamics" / "normalizer.pth").is_file()
    assert (layout.checkpoints_dir / "dynamics" / "backbone" / "asset.bin").is_file()
    assert len(load_json(layout.manifest_path)["inputs"]) == 12
