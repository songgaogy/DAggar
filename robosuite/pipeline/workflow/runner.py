"""Batch-online DIPOLE orchestration entrypoints."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from robosuite.pipeline.modules.training.discriminator.encoder import (
    resolve_saved_dynamics_config,
    resolve_saved_normalizer_checkpoint,
)
from robosuite.pipeline.collection import run_collection
from robosuite.pipeline.modules.training.discriminator.runner import run_discriminator_finetune
from robosuite.pipeline.modules.training.dipole.runner import run_offline_training

from robosuite.pipeline.config.adapters import (
    collection_stage_config,
    discriminator_stage_config,
    offline_stage_config,
)
from .state import (
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
    snapshot_inputs,
    start_stage,
    write_json_atomic,
    write_text_atomic,
)


_RUN_BOUND_CONFIG_PATHS = {
    "base_policy": "task.inputs.base_policy_checkpoint",
    "parent_discriminator": "task.inputs.parent_discriminator_checkpoint",
    "dynamics_encoder": "task.inputs.dynamics_encoder_checkpoint",
    "initial_vast": "task.inputs.initial_vast_checkpoint",
    "expert_data": "task.inputs.expert_data",
    "discriminator_pretrain": "task.inputs.discriminator_pretrain_data",
    "vast_warmup_transitions": "task.inputs.vast_warmup_transitions",
    "flow_image_encoder": "task.policy.flow.model.image_encoder.pretrained_path",
    "language_encoder": "task.policy.flow.model.language_encoder.pretrained_name",
}


def load_default_config(*, task: str) -> DictConfig:
    config_dir = Path(__file__).resolve().parents[1] / "config"
    with initialize_config_dir(version_base="1.2", config_dir=str(config_dir)):
        return compose(config_name="overall", overrides=[f"task={task}"])


def load_run_config(layout: RunLayout) -> DictConfig:
    if not layout.config_path.is_file():
        raise FileNotFoundError(f"Run configuration does not exist: {layout.config_path}")
    return OmegaConf.load(layout.config_path)


def refresh_run_config(layout: RunLayout) -> DictConfig:
    """Refresh mutable settings while preserving run-bound input locations."""

    state = load_json(layout.state_path)
    task_name = str(state.get("task_name", "")).strip()
    if not task_name:
        raise ValueError(f"Run state has no task_name: {layout.state_path}")
    cfg = load_default_config(task=task_name)
    configured_task = str(cfg.task.name)
    if configured_task != task_name:
        raise ValueError(
            f"Latest task config resolved to {configured_task!r}, expected {task_name!r}."
        )

    manifest = load_json(layout.manifest_path)
    inputs = manifest.get("inputs")
    if not isinstance(inputs, list):
        raise ValueError(f"Run manifest has malformed inputs: {layout.manifest_path}")
    records: dict[str, dict[str, Any]] = {}
    for record in inputs:
        if not isinstance(record, dict) or not record.get("name"):
            raise ValueError(f"Run manifest contains a malformed input record: {record!r}")
        name = str(record["name"])
        if name in records:
            raise ValueError(f"Run manifest contains duplicate input {name!r}")
        records[name] = record
    missing = sorted(set(_RUN_BOUND_CONFIG_PATHS) - set(records))
    if missing:
        raise ValueError(f"Run manifest is missing bound inputs: {', '.join(missing)}")

    for name, config_path in _RUN_BOUND_CONFIG_PATHS.items():
        record = records[name]
        snapshot_path = record.get("snapshot_path")
        if snapshot_path not in (None, ""):
            if record.get("storage_mode") not in (None, "snapshot"):
                raise ValueError(f"Input {name!r} has inconsistent snapshot metadata")
            value = (layout.root / str(snapshot_path)).resolve()
            try:
                value.relative_to(layout.root)
            except ValueError as exc:
                raise ValueError(
                    f"Input {name!r} snapshot escapes the run: {snapshot_path}"
                ) from exc
            if not value.exists():
                raise FileNotFoundError(f"Run-bound input does not exist: {value}")
        else:
            if record.get("storage_mode") not in (None, "external"):
                raise ValueError(f"Input {name!r} has inconsistent external metadata")
            source = record.get("source")
            if source in (None, ""):
                raise ValueError(f"External input {name!r} has no source path")
            value = Path(str(source)).expanduser().resolve()
            if not value.exists():
                raise FileNotFoundError(f"External input does not exist: {value}")
        OmegaConf.update(cfg, config_path, str(value), merge=False)

    evaluation_data_root = OmegaConf.select(
        cfg,
        "task.evaluation.data_root",
        default=None,
    )
    if evaluation_data_root not in (None, ""):
        evaluation_path = Path(str(evaluation_data_root)).expanduser()
        if not evaluation_path.is_absolute():
            evaluation_path = Path(__file__).resolve().parents[3] / evaluation_path
        cfg.task.evaluation.data_root = str(evaluation_path.resolve())

    resolved_yaml = OmegaConf.to_yaml(cfg, resolve=True)
    write_text_atomic(
        layout.config_path,
        resolved_yaml if resolved_yaml.endswith("\n") else f"{resolved_yaml}\n",
    )
    return cfg


def initialize_run(
    *,
    run_name: str = "run",
    task: str | None = None,
    cfg: DictConfig | None = None,
) -> RunLayout:
    if cfg is None:
        if task is None:
            raise ValueError("task is required when initializing a run from overall.yaml")
        cfg = load_default_config(task=task)
    repository_root = Path(__file__).resolve().parents[3]
    safe_name = str(run_name).strip()
    if re.fullmatch(r"[A-Za-z0-9_.-]+", safe_name) is None:
        raise ValueError("run_name may contain only letters, digits, underscore, dot, and dash.")
    output_root = Path(str(cfg.storage.output_root)).expanduser()
    if not output_root.is_absolute():
        output_root = repository_root / output_root
    root = (
        output_root.resolve()
        / str(cfg.environment.name)
        / f"{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    layout = RunLayout(root)

    def source(raw: Any) -> Path:
        path = Path(str(raw)).expanduser()
        return (path if path.is_absolute() else repository_root / path).resolve()

    inputs = cfg.task.inputs
    encoder_source = source(inputs.dynamics_encoder_checkpoint)
    dynamics_config_source = resolve_saved_dynamics_config(encoder_source)
    dynamics_source_cfg = OmegaConf.load(dynamics_config_source)
    dynamics_backbone_path = OmegaConf.select(
        dynamics_source_cfg,
        "encoder.model_path",
        default=None,
    )
    if dynamics_backbone_path in (None, ""):
        raise ValueError(
            f"Dynamics config has no encoder.model_path: {dynamics_config_source}"
        )
    dynamics_backbone_source = source(dynamics_backbone_path)
    normalizer_source = resolve_saved_normalizer_checkpoint(encoder_source)
    vast_warmup_source = source(inputs.vast_warmup_transitions)
    snapshots = [
        InputSnapshot("base_policy", source(inputs.base_policy_checkpoint), "checkpoints/base_policy.pt"),
        InputSnapshot(
            "parent_discriminator",
            source(inputs.parent_discriminator_checkpoint),
            "checkpoints/parent_discriminator.pth",
        ),
        InputSnapshot("dynamics_encoder", encoder_source, "checkpoints/dynamics/model.pth"),
        InputSnapshot(
            "dynamics_config",
            dynamics_config_source,
            "checkpoints/dynamics/hydra.yaml",
        ),
        InputSnapshot(
            "dynamics_backbone",
            dynamics_backbone_source,
            "checkpoints/dynamics/backbone",
        ),
        InputSnapshot("dynamics_normalizer", normalizer_source, "checkpoints/dynamics/normalizer.pth"),
        InputSnapshot("initial_vast", source(inputs.initial_vast_checkpoint), "checkpoints/initial_vast.pt"),
        InputSnapshot("expert_data", source(inputs.expert_data), "data/expert"),
        InputSnapshot(
            "discriminator_pretrain",
            source(inputs.discriminator_pretrain_data),
            "data/discriminator_pretrain",
        ),
        InputSnapshot(
            "flow_image_encoder",
            source(cfg.task.policy.flow.model.image_encoder.pretrained_path),
            "checkpoints/flow_image_encoder.pth",
        ),
        InputSnapshot(
            "language_encoder",
            source(cfg.task.policy.flow.model.language_encoder.pretrained_name),
            "data/language_encoder",
        ),
    ]
    create_run(layout, task_name=str(cfg.environment.name))
    try:
        snapshot_inputs(
            layout,
            snapshots,
            external_inputs=[
                InputReference("vast_warmup_transitions", vast_warmup_source)
            ],
            prefer_reflink=bool(cfg.storage.input_copy.prefer_reflink),
        )
        cfg.task.inputs.base_policy_checkpoint = str(layout.checkpoints_dir / "base_policy.pt")
        cfg.task.inputs.parent_discriminator_checkpoint = str(
            layout.checkpoints_dir / "parent_discriminator.pth"
        )
        cfg.task.inputs.dynamics_encoder_checkpoint = str(
            layout.checkpoints_dir / "dynamics" / "model.pth"
        )
        cfg.task.inputs.initial_vast_checkpoint = str(
            layout.checkpoints_dir / "initial_vast.pt"
        )
        cfg.task.inputs.expert_data = str(layout.data_dir / "expert")
        cfg.task.inputs.discriminator_pretrain_data = str(
            layout.data_dir / "discriminator_pretrain"
        )
        cfg.task.inputs.vast_warmup_transitions = str(vast_warmup_source)
        cfg.task.policy.flow.model.image_encoder.pretrained_path = str(
            layout.checkpoints_dir / "flow_image_encoder.pth"
        )
        cfg.task.policy.flow.model.language_encoder.pretrained_name = str(
            layout.data_dir / "language_encoder"
        )
        evaluation_data_root = OmegaConf.select(
            cfg,
            "task.evaluation.data_root",
            default=None,
        )
        if evaluation_data_root not in (None, ""):
            cfg.task.evaluation.data_root = str(source(evaluation_data_root))
        OmegaConf.save(cfg, layout.config_path, resolve=True)
    except BaseException as exc:
        state = load_json(layout.state_path)
        state["active_stage"] = None
        state["initialization_error"] = f"{type(exc).__name__}: {exc}"
        write_json_atomic(layout.state_path, state)
        raise
    return layout


def run_online(layout: RunLayout) -> Path:
    recover_interrupted_stage(layout)
    state = load_json(layout.state_path)
    if state.get("active_stage") is None:
        open_next_round(layout)
        state = load_json(layout.state_path)
    if state.get("active_stage") != "collection":
        raise RuntimeError(
            f"run_online requires collection as the active stage, got {state.get('active_stage')!r}."
        )
    cfg = refresh_run_config(layout)
    round_index = int(state["active_round"])
    if round_index == 0:
        policy_checkpoint = str(cfg.task.inputs.base_policy_checkpoint)
        discriminator_checkpoint = str(cfg.task.inputs.parent_discriminator_checkpoint)
    else:
        previous = state["rounds"][f"{round_index - 1:03d}"]["stages"]
        policy_checkpoint = _run_path(layout, previous["policy"]["outputs"]["checkpoint"])
        discriminator_checkpoint = _run_path(layout, previous["disc"]["outputs"]["checkpoint"])

    partial_path = layout.round_data_dir(round_index) / "episodes.partial.pt"
    final_path = layout.round_data_dir(round_index) / "episodes.pt"
    stage_cfg = collection_stage_config(
        cfg,
        round_index=round_index,
        policy_checkpoint=policy_checkpoint,
        discriminator_checkpoint=discriminator_checkpoint,
        encoder_checkpoint=str(cfg.task.inputs.dynamics_encoder_checkpoint),
        output_path=str(partial_path),
    )
    start_stage(
        layout,
        round_index,
        "collection",
        inputs={
            "policy_checkpoint": policy_checkpoint,
            "discriminator_checkpoint": discriminator_checkpoint,
            "collection_seed": int(stage_cfg.seed),
        },
    )
    try:
        if final_path.is_file():
            payload = torch.load(final_path, map_location="cpu", weights_only=False)
        else:
            run_collection(stage_cfg)
            payload = torch.load(partial_path, map_location="cpu", weights_only=False)
        if not bool(payload.get("final", False)):
            raise RuntimeError("Collection stopped before the configured episode target.")
        if partial_path.is_file():
            os.replace(partial_path, final_path)
            partial_meta = partial_path.with_suffix(".meta.json")
            if partial_meta.exists():
                os.replace(partial_meta, final_path.with_suffix(".meta.json"))
        artifact = record_artifact(
            layout,
            final_path,
            round_index=round_index,
            stage="collection",
        )
        complete_stage(
            layout,
            round_index,
            "collection",
            outputs={
                "episodes": _relative(layout, final_path),
                "episodes_sha256": artifact["sha256"],
            },
            parent_checkpoint=policy_checkpoint,
        )
        return final_path
    except BaseException as exc:
        fail_stage(layout, round_index, "collection", error=f"{type(exc).__name__}: {exc}")
        raise


def train(layout: RunLayout, *, requested_stage: str = "all") -> None:
    if requested_stage not in {"all", "disc", "vast", "policy"}:
        raise ValueError("stage must be one of all, disc, vast, or policy.")
    recover_interrupted_stage(layout)
    while True:
        state = load_json(layout.state_path)
        active = state.get("active_stage")
        if active not in {"disc", "vast", "policy"}:
            if requested_stage == "all" and active is None:
                return
            raise RuntimeError(f"No requested training stage is ready; active_stage={active!r}.")
        if requested_stage != "all" and requested_stage != active:
            raise RuntimeError(f"Cannot run {requested_stage!r}; the next stage is {active!r}.")
        _run_training_stage(layout, stage=str(active))
        if requested_stage != "all":
            return


def _run_training_stage(layout: RunLayout, *, stage: str) -> None:
    cfg = refresh_run_config(layout)
    state = load_json(layout.state_path)
    round_index = int(state["active_round"])
    episodes_paths = [
        _run_path(
            layout,
            state["rounds"][f"{index:03d}"]["stages"]["collection"]["outputs"]["episodes"],
        )
        for index in range(round_index + 1)
    ]
    parent_policy = (
        str(cfg.task.inputs.base_policy_checkpoint)
        if round_index == 0
        else _run_path(
            layout,
            state["rounds"][f"{round_index - 1:03d}"]["stages"]["policy"]["outputs"]["checkpoint"],
        )
    )
    parent_disc = (
        str(cfg.task.inputs.parent_discriminator_checkpoint)
        if round_index == 0
        else _run_path(
            layout,
            state["rounds"][f"{round_index - 1:03d}"]["stages"]["disc"]["outputs"]["checkpoint"],
        )
    )
    parent_vast = (
        str(cfg.task.inputs.initial_vast_checkpoint)
        if round_index == 0
        else _run_path(
            layout,
            state["rounds"][f"{round_index - 1:03d}"]["stages"]["vast"]["outputs"]["checkpoint"],
        )
    )
    stage_inputs = {
        "episodes": [_relative(layout, Path(path)) for path in episodes_paths],
        "parent_checkpoint": {
            "disc": parent_disc,
            "vast": parent_vast,
            "policy": parent_policy,
        }[stage],
    }
    attempt_record = start_stage(
        layout,
        round_index,
        stage,
        inputs=stage_inputs,
    )
    attempt = int(attempt_record["attempt"])
    attempt_root = layout.attempt_dir(round_index, stage, attempt)
    work_dir = attempt_root / "work"
    try:
        if stage == "disc":
            stage_cfg = discriminator_stage_config(
                cfg,
                parent_checkpoint=parent_disc,
                encoder_checkpoint=str(cfg.task.inputs.dynamics_encoder_checkpoint),
                pretrain_dir=str(cfg.task.inputs.discriminator_pretrain_data),
                episodes_paths=episodes_paths,
                run_dir=str(work_dir),
                round_index=round_index,
                feature_cache_dir=str(layout.cache_dir / "discriminator_features"),
            )
            run_discriminator_finetune(stage_cfg)
            checkpoint = work_dir / "checkpoints" / "pu_bce_head_finetuned.pth"
        elif stage == "vast":
            current_disc = _run_path(
                layout,
                load_json(layout.state_path)["rounds"][f"{round_index:03d}"]["stages"]["disc"]["outputs"]["checkpoint"],
            )
            stage_cfg = offline_stage_config(
                cfg,
                stage="vast",
                policy_checkpoint=parent_policy,
                discriminator_checkpoint=current_disc,
                encoder_checkpoint=str(cfg.task.inputs.dynamics_encoder_checkpoint),
                vast_checkpoint=parent_vast,
                episodes_paths=episodes_paths,
                expert_data=str(cfg.task.inputs.expert_data),
                vast_warmup_transitions=_vast_transition_file(cfg),
                run_dir=str(work_dir),
            )
            run_offline_training(stage_cfg)
            checkpoint = work_dir / "checkpoints" / "vast_state_finetuned.pt"
        else:
            current_state = load_json(layout.state_path)
            current_stages = current_state["rounds"][f"{round_index:03d}"]["stages"]
            current_disc = _run_path(layout, current_stages["disc"]["outputs"]["checkpoint"])
            current_vast = _run_path(layout, current_stages["vast"]["outputs"]["checkpoint"])
            stage_cfg = offline_stage_config(
                cfg,
                stage="policy",
                policy_checkpoint=parent_policy,
                discriminator_checkpoint=current_disc,
                encoder_checkpoint=str(cfg.task.inputs.dynamics_encoder_checkpoint),
                vast_checkpoint=current_vast,
                episodes_paths=episodes_paths,
                expert_data=str(cfg.task.inputs.expert_data),
                vast_warmup_transitions=_vast_transition_file(cfg),
                run_dir=str(work_dir),
            )
            run_offline_training(stage_cfg)
            checkpoint = work_dir / "checkpoints" / "latest.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Stage did not produce its checkpoint: {checkpoint}")
        sampling_weights, effective_losses, cache_keys = _stage_state_metadata(
            work_dir / "run_info.json",
            stage=stage,
        )
        published = _publish_attempt(
            layout,
            round_index,
            stage,
            attempt,
            work_dir,
            checkpoint,
        )
        artifact = record_artifact(
            layout,
            published,
            round_index=round_index,
            stage=stage,
        )
        complete_stage(
            layout,
            round_index,
            stage,
            outputs={
                "checkpoint": _relative(layout, published),
                "checkpoint_sha256": artifact["sha256"],
            },
            parent_checkpoint={"disc": parent_disc, "vast": parent_vast, "policy": parent_policy}[stage],
            sampling_weights=sampling_weights,
            effective_losses=effective_losses,
            cache_keys=cache_keys,
        )
    except BaseException as exc:
        fail_stage(layout, round_index, stage, error=f"{type(exc).__name__}: {exc}")
        raise


def _publish_attempt(
    layout: RunLayout,
    round_index: int,
    stage: str,
    attempt: int,
    work_dir: Path,
    checkpoint: Path,
) -> Path:
    stage_root = layout.stage_dir(round_index, stage)
    published_checkpoints = stage_root / "checkpoints"
    published_checkpoints.mkdir(parents=True, exist_ok=True)
    published = published_checkpoints / checkpoint.name
    os.replace(checkpoint, published)
    for name in ("run_info.json", "config_resolved.yaml"):
        source = work_dir / name
        if source.exists():
            os.replace(source, stage_root / name)
    tensorboard_source = work_dir / "tensorboard"
    if tensorboard_source.exists():
        tensorboard_root = stage_root / "tensorboard"
        tensorboard_root.mkdir(exist_ok=True)
        os.replace(
            tensorboard_source,
            tensorboard_root / f"attempt_{attempt:03d}",
        )
    return published


def _stage_state_metadata(
    run_info_path: Path,
    *,
    stage: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not run_info_path.is_file():
        return {}, {}, {}
    run_info = json.loads(run_info_path.read_text(encoding="utf-8"))
    if stage == "disc":
        mixtures = dict(run_info.get("finetune_config", {}).get("round_mixtures", {}))
        sampling_weights = {
            name: dict(record.get("effective_round_weights", {}))
            for name, record in mixtures.items()
        }
        effective_losses = {
            name: {
                "configured_weight": record.get("configured_loss_weight"),
                "effective_weight": record.get("effective_loss_weight"),
                "skip_reason": record.get("skip_reason"),
                "pool_sizes": record.get("round_pool_sizes", {}),
            }
            for name, record in mixtures.items()
        }
        cache_keys = {
            f"round_{int(record['round_index']):03d}": record["feature_cache_key"]
            for record in run_info.get("data", {}).get("round_feature_caches", [])
        }
        return sampling_weights, effective_losses, cache_keys
    if stage == "vast":
        stats = dict(run_info.get("vast_buffer_stats", {}))
        return {"strategy": "uniform_transition", "buffer": stats}, {}, {}
    streams = dict(run_info.get("stream_stats", {}))
    return {"strategy": "uniform_valid_window", "streams": streams}, {}, {}


def _vast_transition_file(cfg: DictConfig) -> str:
    path = Path(str(cfg.task.inputs.vast_warmup_transitions))
    if path.is_file():
        return str(path)
    candidate = path / "vast_offline_transitions.pt"
    if not candidate.is_file():
        raise FileNotFoundError(f"VAST warmup transitions do not exist: {candidate}")
    return str(candidate)


def _relative(layout: RunLayout, path: Path) -> str:
    return path.resolve().relative_to(layout.root).as_posix()


def _run_path(layout: RunLayout, value: str) -> str:
    path = Path(str(value))
    return str(path if path.is_absolute() else layout.root / path)


__all__ = [
    "initialize_run",
    "load_default_config",
    "load_run_config",
    "refresh_run_config",
    "run_online",
    "train",
]
