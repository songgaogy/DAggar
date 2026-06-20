from __future__ import annotations

import datetime
import json
import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from robosuite.pipeline.flow_dagger import FlowDaggerTrainer
from robosuite.pipeline.factory import build_algorithm
from robosuite.pipeline.envs import build_robosuite_env
from robosuite.pipeline.utils import load_demo_paths, resolve_camera_names, resolve_task_demo_paths

# Reuse the flow-dagger helpers directly so the data/model setup cannot drift.
from robosuite.pipeline.train_flow_dagger import (
    bind_flow_proprio_extractor,
    build_flow_runtime_cfg,
    load_hdf5_demos_into_flow_transitions,
    load_init_checkpoint_payload,
    reset_flow_policy_observation,
    resolve_flow_task_metadata,
)

# Canonical flow-dagger config. We load it and override only the offline-relevant knobs.
_DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "train_flow_dagger.yaml"
_SFT_SPLIT_ORDER = ("expert", "pretrain_data", "success_rollout", "fail_rollout")


def build_cfg(sft_cfg: DictConfig) -> Any:
    """Load the canonical flow-dagger config and apply offline overrides."""
    base_config = getattr(sft_cfg.runtime, "base_flow_config", None) or str(_DEFAULT_CONFIG)
    cfg = OmegaConf.load(to_absolute_path(str(base_config)))
    OmegaConf.set_struct(cfg, False)

    # --- task / data overrides ---------------------------------------------------
    task_name = str(sft_cfg.env.environment)
    cfg.env.environment = task_name
    cfg.data.task_name = str(sft_cfg.data.task_name) if sft_cfg.data.task_name is not None else task_name
    cfg.data.demo_root = str(sft_cfg.data.demo_root)
    cfg.data.demo_split = "sft_mixed"
    cfg.data.sft_data = OmegaConf.to_container(sft_cfg.data.sft_data, resolve=True)
    cfg.data.pretrain_split_name = str(sft_cfg.data.pretrain_split_name)
    cfg.data.num_trajectories = int(total_requested_trajectories(sft_cfg))

    # --- training overrides ------------------------------------------------------
    cfg.algorithm.trainer.pretrain_steps = int(sft_cfg.train.steps)
    cfg.algorithm.flow.learning_rate = float(sft_cfg.train.learning_rate)
    # No rollout / HIL in this probe.
    cfg.intervention.enabled = False
    cfg.runtime.interactive = False
    cfg.runtime.viewer_enabled = False
    cfg.runtime.online_updates_enabled = False

    if sft_cfg.seed is not None:
        cfg.seed = int(sft_cfg.seed)
    if sft_cfg.train.device is not None:
        cfg.algorithm.flow.device = str(sft_cfg.train.device)
        cfg.algorithm.flow.inference_device = str(sft_cfg.train.device)
    if sft_cfg.runtime.init_checkpoint is not None:
        cfg.runtime.init_checkpoint = to_absolute_path(str(sft_cfg.runtime.init_checkpoint))
    return cfg


def resolve_sft_split_name(logical_split: str, sft_cfg: DictConfig) -> str:
    """Map logical SFT data keys to on-disk demo split names."""
    if logical_split == "pretrain_data":
        return str(sft_cfg.data.pretrain_split_name)
    return str(logical_split)


def resolve_sft_data_counts(sft_cfg: DictConfig) -> dict[str, int]:
    counts: dict[str, int] = {}
    raw_counts = sft_cfg.data.sft_data
    for split_name in _SFT_SPLIT_ORDER:
        count = int(getattr(raw_counts, split_name, 0) or 0)
        if count < 0:
            raise ValueError(f"data.sft_data.{split_name} must be >= 0, got {count}.")
        if count > 0:
            counts[split_name] = count
    unknown_splits = sorted(set(raw_counts.keys()) - set(_SFT_SPLIT_ORDER))
    if unknown_splits:
        raise ValueError(f"Unsupported data.sft_data keys: {unknown_splits}.")
    if not counts:
        raise ValueError("At least one data.sft_data split must request >0 trajectories.")
    return counts


def total_requested_trajectories(sft_cfg: DictConfig) -> int:
    return int(sum(resolve_sft_data_counts(sft_cfg).values()))


def resolve_output_dir(output_root: str, task_name: str, postfix: str = None) -> Path:
    """Build <output-root>/<env> or <output-root>/<env>_<postfix> when postfix is set."""
    output_subdir = task_name
    if postfix is not None:
        postfix = str(postfix).strip()
        if postfix:
            output_subdir = f"{task_name}_{postfix}"
        return Path(to_absolute_path(output_root)) / output_subdir
    else:
        return Path(to_absolute_path(output_root)) / task_name


def build_agent_and_env(cfg: Any, init_payload: dict[str, Any] | None):
    """Mirror train_flow_dagger.main(): headless env -> obs example -> agent.

    Returns (agent, env, proprio_extractor, policy_camera_names, camera_aliases).
    """
    task_name = str(cfg.env.environment)
    requested_camera_names = resolve_camera_names(cfg)
    flow_env_metadata = resolve_flow_task_metadata(init_payload, task_name)

    # Camera names: prefer the checkpoint's, exactly like the flow-dagger run.
    if bool(getattr(cfg.runtime, "use_init_checkpoint_camera_names", True)) and init_payload is not None:
        policy_camera_names = [str(name) for name in init_payload.get("camera_names", [])]
        if len(policy_camera_names) == 0:
            policy_camera_names = list(requested_camera_names)
    else:
        policy_camera_names = list(requested_camera_names)
    cfg.algorithm.camera_names = list(policy_camera_names)
    camera_aliases = {
        str(key): str(value)
        for key, value in dict(getattr(cfg.algorithm.flow, "camera_aliases", {}) or {}).items()
    }

    # Headless env: no window, offscreen rendering on, camera obs feeding the policy.
    runtime_cfg = build_flow_runtime_cfg(
        cfg,
        env_metadata=flow_env_metadata,
        camera_names=policy_camera_names,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        renderer="mjviewer",
    )
    runtime_cfg.env_name = task_name
    env = build_robosuite_env(runtime_cfg)
    proprio_extractor = bind_flow_proprio_extractor(env, flow_env_metadata)
    initial_obs, _ = reset_flow_policy_observation(
        env,
        preserve_mjviewer=False,
        extractor=proprio_extractor,
        policy_camera_names=policy_camera_names,
        camera_aliases=camera_aliases,
        img_height=int(cfg.env.img_height),
        img_width=int(cfg.env.img_width),
    )

    action_low, action_high = env.action_spec
    action_low = np.asarray(action_low, dtype=np.float32)
    action_high = np.asarray(action_high, dtype=np.float32)

    # --- algorithm cfg: inject the checkpoint's model_cfg / horizon / paths -------
    algorithm_cfg = OmegaConf.to_container(cfg.algorithm, resolve=True)
    assert isinstance(algorithm_cfg, dict)
    algorithm_cfg["camera_names"] = list(policy_camera_names)
    algorithm_cfg["task_name"] = task_name
    flow_cfg = algorithm_cfg.setdefault("flow", {})
    flow_cfg.setdefault("image_size", int(cfg.env.img_height))
    if bool(getattr(cfg.runtime, "use_init_checkpoint_model", True)) and init_payload is not None:
        if "model_cfg" in init_payload:
            flow_cfg["model"] = init_payload["model_cfg"]
        if "task_prompt_map" in init_payload:
            flow_cfg["task_prompt_map"] = init_payload["task_prompt_map"]
        if init_payload.get("act_mean") is not None:
            flow_cfg["action_horizon"] = int(np.asarray(init_payload["act_mean"]).shape[0])
            flow_cfg.setdefault("execute_horizon", 1)
    model_cfg = flow_cfg.setdefault("model", {})
    image_encoder_cfg = model_cfg.get("image_encoder", None)
    if isinstance(image_encoder_cfg, dict) and image_encoder_cfg.get("pretrained_path"):
        image_encoder_cfg["pretrained_path"] = to_absolute_path(str(image_encoder_cfg["pretrained_path"]))
    language_encoder_cfg = model_cfg.get("language_encoder", None)
    if isinstance(language_encoder_cfg, dict) and language_encoder_cfg.get("pretrained_name"):
        language_encoder_cfg["pretrained_name"] = to_absolute_path(str(language_encoder_cfg["pretrained_name"]))

    agent = build_algorithm(
        algorithm_cfg,
        observation_example=initial_obs,
        sample_action=np.zeros_like(action_low, dtype=np.float32),
        action_low=action_low,
        action_high=action_high,
    )
    return agent, env, proprio_extractor, policy_camera_names, camera_aliases


def _set_module_trainable(module: torch.nn.Module, trainable: bool) -> None:
    for param in module.parameters():
        param.requires_grad = bool(trainable)


def configure_finetune_policy(agent, *, train_scope: str, learning_rate: float, freeze_visual_bn: bool) -> dict[str, Any]:
    model = agent.core.model
    scope = str(train_scope)
    if scope not in {"flow_head", "flow_head_aggregator", "all"}:
        raise ValueError(
            "train.train_scope must be one of {'flow_head', 'flow_head_aggregator', 'all'}, "
            f"got {scope!r}."
        )
    if scope != "all":
        _set_module_trainable(model, False)
        _set_module_trainable(model.flow_head, True)
        if scope == "flow_head_aggregator":
            _set_module_trainable(model.condition_aggregator, True)
    if bool(freeze_visual_bn):
        agent.core.set_freeze_visual_batch_norm(True)

    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError(f"train_scope={scope} left zero trainable parameters.")
    agent.core.optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(learning_rate),
        weight_decay=float(agent.flow_config.weight_decay),
    )
    return {
        "train_scope": scope,
        "learning_rate": float(learning_rate),
        "freeze_visual_bn": bool(freeze_visual_bn),
        "trainable_parameter_tensors": int(len(trainable_params)),
        "trainable_parameter_count": int(sum(param.numel() for param in trainable_params)),
        "trainable_parameter_names": trainable_names,
    }


def _build_ema_model(agent, ema_decay: float) -> AveragedModel:
    ema_model = AveragedModel(
        agent.core.model,
        multi_avg_fn=get_ema_multi_avg_fn(float(ema_decay)),
    ).to(agent.core.device)
    ema_model.eval()
    return ema_model


def save_ema_checkpoint(
    agent,
    path: Path,
    *,
    ema_model: AveragedModel,
    include_buffers: bool,
    extra: dict[str, Any],
) -> None:
    payload = agent.build_checkpoint_payload(include_buffers=include_buffers, extra=extra)
    core_state = payload["core"]
    core_state["raw_model"] = core_state["model"]
    core_state["ema_model"] = ema_model.module.state_dict()
    # Deployment/eval uses core.model via FlowDaggerPolicy.load_state_dict().
    core_state["model"] = core_state["ema_model"]
    payload["checkpoint_weight_type"] = "ema"
    payload["ema_decay"] = float(extra.get("metadata", {}).get("ema_decay", 0.0))
    agent.write_checkpoint_payload(path, payload)


def load_sft_demo_transitions(
    cfg: Any,
    sft_cfg: DictConfig,
    *,
    policy_camera_names: list[str],
    camera_aliases: dict[str, str],
    proprio_extractor,
) -> tuple[list[Any], list[dict[str, Any]], int]:
    task_data_name = str(cfg.data.task_name or cfg.env.environment)
    sft_counts = resolve_sft_data_counts(sft_cfg)
    if sft_cfg.data.demo_sample_seed is not None:
        base_sample_seed = int(sft_cfg.data.demo_sample_seed)
    elif getattr(cfg, "seed", None) is not None:
        base_sample_seed = int(cfg.seed)
    else:
        base_sample_seed = 0
    split_records: list[dict[str, Any]] = []
    all_transitions: list[Any] = []

    for split_index, (logical_split, requested_trajectories) in enumerate(sft_counts.items()):
        actual_split = resolve_sft_split_name(logical_split, sft_cfg)
        demo_paths = resolve_task_demo_paths(
            task_name=task_data_name,
            data_root=to_absolute_path(str(sft_cfg.data.demo_root)),
            split=actual_split,
        )
        if not demo_paths:
            raise FileNotFoundError(
                f"No demo files found for SFT split '{logical_split}' "
                f"(resolved split '{actual_split}') under data/{task_data_name}/{actual_split}."
            )

        selected_demo_records: list[dict[str, Any]] = []

        def _record_selected_demos(path: Path, demo_names: list[str]) -> None:
            selected_demo_records.append({"path": str(path), "demo_names": list(demo_names)})

        split_sample_seed = int(base_sample_seed + split_index)
        split_transitions = load_demo_paths(
            demo_paths,
            hdf5_loader=lambda path, demo_names=None: load_hdf5_demos_into_flow_transitions(
                path,
                policy_camera_names=policy_camera_names,
                camera_aliases=camera_aliases,
                img_height=int(cfg.env.img_height),
                img_width=int(cfg.env.img_width),
                proprio_keys=tuple(cfg.env.proprio_keys or []),
                renderer=str(cfg.env.renderer),
                control_freq=int(cfg.env.control_freq),
                demo_names=demo_names,
                state_extractor=proprio_extractor,
            ),
            max_num_trajectories=int(requested_trajectories),
            check_legacy_rewards=False,
            random_sample=True,
            random_seed=split_sample_seed,
            selected_demo_callback=_record_selected_demos,
        )
        if len(split_transitions) == 0:
            raise RuntimeError(
                f"SFT split '{logical_split}' loaded zero transitions from resolved split '{actual_split}'."
            )
        for transition in split_transitions:
            if transition.demo_source is None:
                transition.demo_source = logical_split
        selected_trajectory_count = int(sum(len(record["demo_names"]) for record in selected_demo_records))
        if selected_trajectory_count != int(requested_trajectories):
            raise RuntimeError(
                f"SFT split '{logical_split}' requested {requested_trajectories} trajectories but selected "
                f"{selected_trajectory_count} from resolved split '{actual_split}'."
            )
        split_record = {
            "logical_split": logical_split,
            "resolved_split": actual_split,
            "requested_trajectories": int(requested_trajectories),
            "selected_trajectories": selected_trajectory_count,
            "loaded_transitions": int(len(split_transitions)),
            "sample_seed": split_sample_seed,
            "demo_paths": [str(path) for path in demo_paths],
            "selected_demos": selected_demo_records,
        }
        split_records.append(split_record)
        all_transitions.extend(split_transitions)
        print(
            f"[demo] split={logical_split} resolved={actual_split} requested={requested_trajectories} "
            f"selected={selected_trajectory_count} transitions={len(split_transitions)} seed={split_sample_seed}"
        )

    return all_transitions, split_records, base_sample_seed


@hydra.main(version_base="1.2", config_path="../config", config_name="sft")
def main(sft_cfg: DictConfig) -> None:
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
            f"{getattr(cfg.runtime, 'init_checkpoint', None)}. This probe requires a warm-start checkpoint."
        )
    print(f"[init] original checkpoint = {init_checkpoint}")

    task_name = str(cfg.env.environment)
    agent, env, proprio_extractor, policy_camera_names, camera_aliases = build_agent_and_env(cfg, init_payload)
    trainer = FlowDaggerTrainer(agent)
    print(f"[env] task={task_name} policy_cameras={policy_camera_names}")

    # --- warm-start weights + reuse the checkpoint's normalizers -----------------
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

    # --- load requested SFT splits and bootstrap the demo buffer ------------------
    transitions, split_records, demo_sample_seed = load_sft_demo_transitions(
        cfg,
        sft_cfg,
        policy_camera_names=policy_camera_names,
        camera_aliases=camera_aliases,
        proprio_extractor=proprio_extractor,
    )
    if len(transitions) == 0:
        raise RuntimeError("Demo loading produced zero transitions.")
    trainer.bootstrap_demo_buffer(transitions, demo_source="offline_demo")
    n_demo_episodes = trainer._offline_bootstrap_episodes
    requested_total_trajectories = total_requested_trajectories(sft_cfg)
    print(
        f"[demo] loaded {len(transitions)} transitions from {n_demo_episodes} "
        f"trajectories (requested {requested_total_trajectories}); demo_buffer={len(agent.demo_buffer)}"
    )

    # Normalizers came from the checkpoint (matches flow-dagger "Reusing normalizers from checkpoint").
    if not agent.has_normalizers():
        agent.fit_normalizers_from_transitions(transitions)
        print("[init] fitted normalizers from demos (checkpoint had none).")
    else:
        print("[init] reusing normalizers from checkpoint.")

    if not agent.ready_for_update():
        raise RuntimeError(
            "Agent not ready for update: not enough valid demo sequences or missing normalizers."
        )

    # --- output paths -------------------------------------------------------------
    postfix = sft_cfg.output.postfix
    postfix = None if postfix is None else str(postfix).strip()
    output_dir = resolve_output_dir(str(sft_cfg.output.root), task_name, postfix)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[output] writing to {output_dir}")
    ckpt_tag = f"flow_offline_{task_name}_traj{requested_total_trajectories:05d}_steps{int(sft_cfg.train.steps):08d}"
    ckpt_path = output_dir / f"{ckpt_tag}.pt"
    loss_log_path = output_dir / f"{ckpt_tag}.loss.jsonl"
    saved_config = OmegaConf.create(
        {
            "sft": OmegaConf.to_container(sft_cfg, resolve=True),
            "flow_runtime": OmegaConf.to_container(cfg, resolve=True),
        }
    )
    OmegaConf.save(saved_config, output_dir / f"{ckpt_tag}.config.yaml")

    metadata = {
        "env_name": task_name,
        "data_type": "sft_mixed",
        "num_trajectories": int(requested_total_trajectories),
        "sft_data": OmegaConf.to_container(sft_cfg.data.sft_data, resolve=True),
        "sft_splits": split_records,
        "demo_transition_count": int(len(transitions)),
        "demo_episode_count": int(n_demo_episodes),
        "pretrain_steps": int(sft_cfg.train.steps),
        "init_checkpoint": str(init_checkpoint),
        "demo_sample_seed": int(demo_sample_seed),
        "selected_demos": split_records,
        "learning_rate": float(sft_cfg.train.learning_rate),
        "train_scope": str(sft_cfg.train.train_scope),
        "freeze_visual_bn": bool(sft_cfg.train.freeze_visual_bn),
        "ema_decay": float(sft_cfg.train.ema_decay),
        "checkpoint_weight_type": "ema",
        "finetune_config": finetune_config,
        "hil": False,
    }
    print(f"[demo] data_type=sft_mixed sample_seed={demo_sample_seed} splits={split_records}")

    # --- offline fine-tuning loop (mirrors the base_policy builder) ----------------
    total_steps = int(sft_cfg.train.steps)
    log_interval = max(1, int(sft_cfg.train.log_interval))
    print(f"[train] starting offline fine-tuning for {total_steps} steps...")
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

    # --- final checkpoint ---------------------------------------------------------
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
    main()
