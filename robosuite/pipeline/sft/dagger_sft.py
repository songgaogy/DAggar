from __future__ import annotations

import datetime
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from robosuite.pipeline.algorithms.flow_dagger import FlowDaggerTrainer
from robosuite.pipeline.common.types import Transition
from robosuite.pipeline.sft.flow_sft import (
    _build_ema_model,
    build_agent_and_env,
    build_cfg,
    configure_finetune_policy,
    load_sft_demo_transitions,
    resolve_output_dir,
    save_ema_checkpoint,
)
from robosuite.pipeline.train_flow_dagger import load_init_checkpoint_payload


_DAGGER_DATA_KEYS = ("demo_buffer", "warmup_split_name", "warmup_num")
_DEFAULT_OUTPUT_ROOT = "./outputs/flow-sft-dagger"


def _normalize_hydra_additive_overrides(argv: list[str]) -> list[str]:
    """Allow data.demo_buffer=... without editing the shared sft.yaml config."""
    normalized = [argv[0]]
    for arg in argv[1:]:
        if arg.startswith(("+", "++", "~")):
            normalized.append(arg)
            continue
        if any(arg.startswith(f"data.{key}=") for key in _DAGGER_DATA_KEYS):
            normalized.append(f"+{arg}")
        else:
            normalized.append(arg)
    return normalized


def _mutable_cfg(cfg: DictConfig) -> DictConfig:
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.set_struct(cfg, False)
    return cfg


def normalize_dagger_sft_cfg(sft_cfg: DictConfig) -> DictConfig:
    cfg = _mutable_cfg(sft_cfg)
    data_cfg = cfg.data

    if getattr(data_cfg, "demo_buffer", None) is None:
        raise ValueError("data.demo_buffer must point to a DAgger demo chunk file, demo_chunks directory, or run dir.")

    warmup_split = getattr(data_cfg, "warmup_split_name", None)
    if warmup_split is None:
        warmup_split = getattr(data_cfg, "pretrain_split_name", None)
    if warmup_split is None:
        raise ValueError("data.warmup_split_name or data.pretrain_split_name must be set.")

    warmup_num = getattr(data_cfg, "warmup_num", None)
    if warmup_num is None:
        warmup_num = int(getattr(data_cfg.sft_data, "pretrain_data", 0) or 0)
    warmup_num = int(warmup_num)
    if warmup_num <= 0:
        raise ValueError(f"data.warmup_num must be > 0, got {warmup_num}.")

    data_cfg.pretrain_split_name = str(warmup_split)
    data_cfg.warmup_split_name = str(warmup_split)
    data_cfg.warmup_num = int(warmup_num)
    data_cfg.sft_data.expert = 0
    data_cfg.sft_data.pretrain_data = int(warmup_num)
    data_cfg.sft_data.success_rollout = 0
    data_cfg.sft_data.fail_rollout = 0

    if str(cfg.output.root).strip() in {"", "./outputs/flow-sft", "outputs/flow-sft"}:
        cfg.output.root = _DEFAULT_OUTPUT_ROOT
    return cfg


def resolve_demo_chunk_paths(demo_buffer: str | Path) -> tuple[Path, list[Path]]:
    source = Path(to_absolute_path(str(demo_buffer))).resolve()
    if source.is_file():
        return source, [source]
    if not source.exists():
        raise FileNotFoundError(f"DAgger demo buffer path does not exist: {source}")
    if not source.is_dir():
        raise ValueError(f"DAgger demo buffer path must be a file or directory: {source}")

    if (source / "buffers" / "demo_chunks").is_dir():
        chunk_dir = source / "buffers" / "demo_chunks"
    else:
        chunk_dir = source
    chunk_paths = sorted(chunk_dir.glob("chunk_*.pt"))
    if not chunk_paths:
        raise FileNotFoundError(f"No chunk_*.pt files found under DAgger demo buffer directory: {chunk_dir}")
    return chunk_dir.resolve(), [path.resolve() for path in chunk_paths]


def load_dagger_demo_transitions(demo_buffer: str | Path) -> tuple[list[Transition], list[dict[str, Any]]]:
    _, chunk_paths = resolve_demo_chunk_paths(demo_buffer)
    transitions: list[Transition] = []
    chunk_records: list[dict[str, Any]] = []

    for chunk_path in chunk_paths:
        payload = torch.load(chunk_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise TypeError(f"DAgger demo chunk must be a dict payload: {chunk_path}")
        if "transitions" not in payload:
            raise KeyError(f"DAgger demo chunk missing 'transitions': {chunk_path}")
        chunk_transitions = payload["transitions"]
        if not isinstance(chunk_transitions, list):
            raise TypeError(f"DAgger demo chunk 'transitions' must be a list: {chunk_path}")

        normalized_chunk: list[Transition] = []
        for index, transition in enumerate(chunk_transitions):
            if not isinstance(transition, Transition):
                raise TypeError(
                    f"DAgger demo chunk contains non-Transition item at {chunk_path}:{index}: "
                    f"{type(transition)!r}"
                )
            if transition.demo_source is None:
                transition = replace(transition, demo_source="dagger_demo")
            normalized_chunk.append(transition)

        chunk_records.append(
            {
                "path": str(chunk_path),
                "chunk_index": int(payload.get("chunk_index", len(chunk_records))),
                "saved_at": payload.get("saved_at"),
                "declared_transition_count": int(payload.get("transition_count", len(normalized_chunk))),
                "loaded_transitions": int(len(normalized_chunk)),
            }
        )
        transitions.extend(normalized_chunk)
        print(f"[dagger_demo] chunk={chunk_path} transitions={len(normalized_chunk)}")

    if len(transitions) == 0:
        raise RuntimeError(f"DAgger demo buffer loaded zero transitions from {demo_buffer}.")
    return transitions, chunk_records


def _auto_output_postfix(sft_cfg: DictConfig, dagger_transition_count: int) -> str:
    explicit = getattr(sft_cfg.output, "postfix", None)
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    return f"{int(sft_cfg.data.warmup_num)}warmup_dagger-{int(dagger_transition_count)}"


@hydra.main(version_base="1.2", config_path="../config", config_name="sft")
def main(raw_sft_cfg: DictConfig) -> None:
    sft_cfg = normalize_dagger_sft_cfg(raw_sft_cfg)
    cfg = build_cfg(sft_cfg)

    if getattr(cfg, "seed", None) is not None:
        seed = int(cfg.seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    init_checkpoint, init_payload = load_init_checkpoint_payload(cfg)
    if init_checkpoint is None or init_payload is None:
        raise FileNotFoundError(
            "Could not load the original flow checkpoint from runtime.init_checkpoint="
            f"{getattr(cfg.runtime, 'init_checkpoint', None)}."
        )
    print(f"[init] original checkpoint = {init_checkpoint}")

    task_name = str(cfg.env.environment)
    agent, env, proprio_extractor, policy_camera_names, camera_aliases = build_agent_and_env(cfg, init_payload)
    trainer = FlowDaggerTrainer(agent)
    print(f"[env] task={task_name} policy_cameras={policy_camera_names}")

    agent.load_flow_policy_checkpoint(init_checkpoint, task_name=task_name)
    print("[init] warm-started policy weights and normalizers from checkpoint.")
    finetune_config = configure_finetune_policy(
        agent,
        train_scope=str(sft_cfg.train.train_scope),
        learning_rate=float(sft_cfg.train.learning_rate),
        freeze_visual_bn=bool(sft_cfg.train.freeze_visual_bn),
    )
    ema_model = _build_ema_model(agent, ema_decay=float(sft_cfg.train.ema_decay))
    print(
        "[train] scope={train_scope} lr={learning_rate:g} trainable_params={trainable_parameter_count} "
        "freeze_visual_bn={freeze_visual_bn} ema_decay={ema_decay:g}".format(
            **finetune_config,
            ema_decay=float(sft_cfg.train.ema_decay),
        )
    )

    dagger_source, dagger_chunk_paths = resolve_demo_chunk_paths(str(sft_cfg.data.demo_buffer))
    dagger_transitions, dagger_chunk_records = load_dagger_demo_transitions(str(sft_cfg.data.demo_buffer))
    warmup_transitions, warmup_split_records, warmup_sample_seed = load_sft_demo_transitions(
        cfg,
        sft_cfg,
        policy_camera_names=policy_camera_names,
        camera_aliases=camera_aliases,
        proprio_extractor=proprio_extractor,
    )
    transitions = list(dagger_transitions) + list(warmup_transitions)
    if len(transitions) == 0:
        raise RuntimeError("DAgger SFT demo loading produced zero transitions.")
    trainer.bootstrap_demo_buffer(transitions, demo_source="dagger_sft")
    n_demo_episodes = trainer._offline_bootstrap_episodes
    print(
        f"[demo] dagger_transitions={len(dagger_transitions)} warmup_transitions={len(warmup_transitions)} "
        f"episodes={n_demo_episodes} demo_buffer={len(agent.demo_buffer)}"
    )

    if not agent.has_normalizers():
        agent.fit_normalizers_from_transitions(transitions)
        print("[init] fitted normalizers from DAgger SFT demos (checkpoint had none).")
    else:
        print("[init] reusing normalizers from checkpoint.")

    if not agent.ready_for_update():
        raise RuntimeError("Agent not ready for update: not enough valid demo sequences or missing normalizers.")

    output_dir = resolve_output_dir(
        str(sft_cfg.output.root),
        task_name,
        _auto_output_postfix(sft_cfg, len(dagger_transitions)),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[output] writing to {output_dir}")

    ckpt_tag = (
        f"flow_dagger_sft_{task_name}_dagger{len(dagger_transitions):08d}_"
        f"warmup{int(sft_cfg.data.warmup_num):05d}_steps{int(sft_cfg.train.steps):08d}"
    )
    ckpt_path = output_dir / f"{ckpt_tag}.pt"
    loss_log_path = output_dir / f"{ckpt_tag}.loss.jsonl"
    saved_config = OmegaConf.create(
        {
            "dagger_sft": OmegaConf.to_container(sft_cfg, resolve=True),
            "flow_runtime": OmegaConf.to_container(cfg, resolve=True),
        }
    )
    OmegaConf.save(saved_config, output_dir / f"{ckpt_tag}.config.yaml")

    metadata = {
        "env_name": task_name,
        "data_type": "dagger_sft_mixed",
        "init_checkpoint": str(init_checkpoint),
        "demo_buffer": str(Path(to_absolute_path(str(sft_cfg.data.demo_buffer))).resolve()),
        "dagger_demo_source": str(dagger_source),
        "dagger_demo_chunk_paths": [str(path) for path in dagger_chunk_paths],
        "dagger_demo_chunks": dagger_chunk_records,
        "dagger_transition_count": int(len(dagger_transitions)),
        "warmup_split": str(sft_cfg.data.warmup_split_name),
        "warmup_requested_trajectories": int(sft_cfg.data.warmup_num),
        "warmup_transition_count": int(len(warmup_transitions)),
        "warmup_sample_seed": int(warmup_sample_seed),
        "warmup_splits": warmup_split_records,
        "demo_transition_count": int(len(transitions)),
        "demo_episode_count": int(n_demo_episodes),
        "pretrain_steps": int(sft_cfg.train.steps),
        "learning_rate": float(sft_cfg.train.learning_rate),
        "train_scope": str(sft_cfg.train.train_scope),
        "freeze_visual_bn": bool(sft_cfg.train.freeze_visual_bn),
        "ema_decay": float(sft_cfg.train.ema_decay),
        "checkpoint_weight_type": "ema",
        "finetune_config": finetune_config,
        "hil": False,
    }
    print(
        f"[demo] data_type=dagger_sft_mixed warmup_split={sft_cfg.data.warmup_split_name} "
        f"warmup_num={int(sft_cfg.data.warmup_num)} dagger_chunks={len(dagger_chunk_records)}"
    )

    total_steps = int(sft_cfg.train.steps)
    log_interval = max(1, int(sft_cfg.train.log_interval))
    print(f"[train] starting DAgger demo SFT for {total_steps} steps...")
    started_at = time.monotonic()
    with open(loss_log_path, "w") as loss_log:
        loss_log.write(
            json.dumps({"event": "run_start", "meta": metadata, "wall_time": datetime.datetime.now().isoformat()})
            + "\n"
        )
        loss_log.flush()
        save_interval = int(sft_cfg.train.save_interval)
        if save_interval > 0:
            initial_path = output_dir / f"{ckpt_tag}_at{0:08d}.pt"
            save_ema_checkpoint(
                agent,
                initial_path,
                ema_model=ema_model,
                include_buffers=False,
                extra={"trainer_state": trainer.state_dict(), "metadata": dict(metadata), "step": 0},
            )
            print(f"[ckpt] saved initial {initial_path}")
        for local_step in range(total_steps):
            metrics = trainer.pretrain(1)[-1]
            ema_model.update_parameters(agent.core.model)
            step = local_step + 1
            if step % log_interval == 0 or step == total_steps:
                elapsed = time.monotonic() - started_at
                sps = step / max(elapsed, 1e-6)
                record = {
                    "step": step,
                    "loss": float(metrics.get("actor_loss", float("nan"))),
                    "flow": float(metrics.get("flow_loss", float("nan"))),
                    "endpoint": float(metrics.get("endpoint_loss", float("nan"))),
                    "smooth": float(metrics.get("smooth_loss", float("nan"))),
                    "steps_per_sec": float(sps),
                }
                loss_log.write(json.dumps(record) + "\n")
                loss_log.flush()
                print(
                    f"[train] step={step}/{total_steps} loss={record['loss']:.4f} "
                    f"flow={record['flow']:.4f} endpoint={record['endpoint']:.4f} "
                    f"smooth={record['smooth']:.4f} ({sps:.1f} it/s)"
                )
            if save_interval > 0 and step % save_interval == 0 and step != total_steps:
                interim_path = output_dir / f"{ckpt_tag}_at{step:08d}.pt"
                save_ema_checkpoint(
                    agent,
                    interim_path,
                    ema_model=ema_model,
                    include_buffers=False,
                    extra={"trainer_state": trainer.state_dict(), "metadata": dict(metadata), "step": step},
                )
                print(f"[ckpt] saved interim {interim_path}")

    save_ema_checkpoint(
        agent,
        ckpt_path,
        ema_model=ema_model,
        include_buffers=False,
        extra={"trainer_state": trainer.state_dict(), "metadata": dict(metadata), "step": total_steps},
    )
    print(f"[done] saved fine-tuned checkpoint -> {ckpt_path}")
    print(f"[done] loss log -> {loss_log_path}")

    try:
        if proprio_extractor is not None and hasattr(proprio_extractor, "close"):
            proprio_extractor.close()
    finally:
        if hasattr(env, "close"):
            env.close()


if __name__ == "__main__":
    sys.argv = _normalize_hydra_additive_overrides(sys.argv)
    main()
