from __future__ import annotations

import argparse
import datetime
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from robosuite.pipeline.algorithms.flow_dagger import FlowDaggerTrainer
from robosuite.pipeline.factory import build_algorithm
from robosuite.pipeline.envs import build_robosuite_env
from robosuite.pipeline.utils import load_demo_paths, resolve_camera_names
from robosuite.pipeline.utils.train_utils import resolve_demo_inputs

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="PickPlaceBread", help="Robosuite environment / task name.")
    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=20,
        help="Number of demo trajectories to fine-tune on (NUM_TRAJECTORIES).",
    )
    parser.add_argument("--steps", type=int, default=2000, help="Number of gradient (tuning) steps.")
    parser.add_argument("--learning-rate", type=float, default=3e-6, help="Offline fine-tune learning rate.")
    parser.add_argument(
        "--train-scope",
        choices=["flow_head", "flow_head_aggregator", "all"],
        default="flow_head",
        help="Which model parameters to update during offline fine-tuning.",
    )
    parser.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay for deployable fine-tuned weights.")
    parser.add_argument(
        "--freeze-visual-bn",
        dest="freeze_visual_bn",
        action="store_true",
        default=True,
        help="Keep image-encoder BatchNorm stats fixed during fine-tuning.",
    )
    parser.add_argument(
        "--no-freeze-visual-bn",
        dest="freeze_visual_bn",
        action="store_false",
        help="Allow image-encoder BatchNorm modules to run in train mode.",
    )
    parser.add_argument(
        "--demo-sample-seed",
        type=int,
        default=None,
        help="Seed for random demo subset selection. Defaults to cfg.seed.",
    )
    parser.add_argument("--device", default=None, help="Override learner device, e.g. cuda:0. Defaults to config.")
    parser.add_argument("--seed", type=int, default=None, help="Override seed. Defaults to config seed (42).")
    parser.add_argument(
        "--config",
        default=str(_DEFAULT_CONFIG),
        help="Base flow-dagger config to mimic training settings from.",
    )
    parser.add_argument(
        "--output-root",
        default="./outputs/flow_dagger_no-hil",
        help="Output root; checkpoint is written under <output-root>/<env>[_<postfix>]/.",
    )
    parser.add_argument("--log-interval", type=int, default=100, help="Console / loss-log interval in steps.")
    parser.add_argument(
        "--save-interval",
        type=int,
        default=0,
        help="If >0, also save an intermediate checkpoint every N steps (for tracking the overfitting curve).",
    )
    parser.add_argument("--output-postfix", default="", help="Postfix for the output directory: <output-root>/<env>_<postfix>")
    parser.add_argument(
        "--data-type",
        default="expert",
        help="Demo split under data/<env>/ (maps to cfg.data.demo_split), e.g. expert or expert-pretrain-data.",
    )
    return parser.parse_args()


def build_cfg(args: argparse.Namespace) -> Any:
    """Load the canonical flow-dagger config and apply offline overrides."""
    cfg = OmegaConf.load(to_absolute_path(args.config))
    OmegaConf.set_struct(cfg, False)

    # --- task / data overrides ---------------------------------------------------
    cfg.env.environment = str(args.env)
    cfg.data.task_name = str(args.env)
    cfg.data.demo_split = str(getattr(args, "data_type", "expert"))
    cfg.data.num_trajectories = int(args.num_trajectories)

    # --- training overrides ------------------------------------------------------
    cfg.algorithm.trainer.pretrain_steps = int(args.steps)
    if getattr(args, "learning_rate", None) is not None:
        cfg.algorithm.flow.learning_rate = float(args.learning_rate)
    # No rollout / HIL in this probe.
    cfg.intervention.enabled = False
    cfg.runtime.interactive = False
    cfg.runtime.viewer_enabled = False
    cfg.runtime.online_updates_enabled = False

    if args.seed is not None:
        cfg.seed = int(args.seed)
    if args.device is not None:
        cfg.algorithm.flow.device = str(args.device)
        cfg.algorithm.flow.inference_device = str(args.device)
    return cfg


def resolve_output_dir(output_root: str, task_name: str, postfix: str) -> Path:
    """Build <output-root>/<env> or <output-root>/<env>_<postfix> when postfix is set."""
    output_subdir = task_name
    postfix = str(postfix).strip()
    if postfix:
        output_subdir = f"{task_name}_{postfix}"
    return Path(to_absolute_path(output_root)) / output_subdir


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


def main() -> None:
    args = parse_args()
    cfg = build_cfg(args)

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
        train_scope=str(args.train_scope),
        learning_rate=float(args.learning_rate),
        freeze_visual_bn=bool(args.freeze_visual_bn),
    )
    ema_model = _build_ema_model(agent, ema_decay=float(args.ema_decay))
    print(
        "[train] scope={train_scope} lr={learning_rate:g} trainable_params={trainable_parameter_count} "
        "freeze_visual_bn={freeze_visual_bn} ema_decay={ema_decay:g}".format(
            **finetune_config,
            ema_decay=float(args.ema_decay),
        )
    )

    # --- load demos (N trajectories) and bootstrap the demo buffer ----------------
    data_type = str(cfg.data.demo_split)
    demo_source_name, demo_paths, max_num_trajectories = resolve_demo_inputs(cfg)
    if not demo_paths:
        raise FileNotFoundError(
            f"No demo files found for '{demo_source_name}'. Expected hdf5 demos under "
            f"data/{task_name}/{data_type}."
        )
    selected_demo_records: list[dict[str, Any]] = []

    def _record_selected_demos(path: Path, demo_names: list[str]) -> None:
        selected_demo_records.append({"path": str(path), "demo_names": list(demo_names)})

    demo_sample_seed = int(cfg.seed) if args.demo_sample_seed is None else int(args.demo_sample_seed)
    transitions = load_demo_paths(
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
        max_num_trajectories=max_num_trajectories,
        check_legacy_rewards=False,
        random_sample=True,
        random_seed=demo_sample_seed,
        selected_demo_callback=_record_selected_demos,
    )
    if len(transitions) == 0:
        raise RuntimeError("Demo loading produced zero transitions.")
    trainer.bootstrap_demo_buffer(transitions, demo_source="offline_demo")
    n_demo_episodes = trainer._offline_bootstrap_episodes
    print(
        f"[demo] loaded {len(transitions)} transitions from {n_demo_episodes} "
        f"trajectories (requested {max_num_trajectories}); demo_buffer={len(agent.demo_buffer)}"
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
    output_dir = resolve_output_dir(args.output_root, task_name, args.output_postfix)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[output] writing to {output_dir}")
    ckpt_tag = f"flow_offline_{task_name}_traj{int(args.num_trajectories):05d}_steps{int(args.steps):08d}"
    ckpt_path = output_dir / f"{ckpt_tag}.pt"
    loss_log_path = output_dir / f"{ckpt_tag}.loss.jsonl"
    OmegaConf.save(cfg, output_dir / f"{ckpt_tag}.config.yaml")

    metadata = {
        "env_name": task_name,
        "data_type": data_type,
        "num_trajectories": int(args.num_trajectories),
        "demo_transition_count": int(len(transitions)),
        "demo_episode_count": int(n_demo_episodes),
        "pretrain_steps": int(args.steps),
        "init_checkpoint": str(init_checkpoint),
        "demo_sample_seed": int(demo_sample_seed),
        "selected_demos": selected_demo_records,
        "learning_rate": float(args.learning_rate),
        "train_scope": str(args.train_scope),
        "freeze_visual_bn": bool(args.freeze_visual_bn),
        "ema_decay": float(args.ema_decay),
        "checkpoint_weight_type": "ema",
        "finetune_config": finetune_config,
        "hil": False,
    }
    print(f"[demo] data_type={data_type} sample_seed={demo_sample_seed} selected={selected_demo_records}")

    # --- offline fine-tuning loop (mirrors the base_policy builder) ----------------
    total_steps = int(args.steps)
    log_interval = max(1, int(args.log_interval))
    print(f"[train] starting offline fine-tuning for {total_steps} steps...")
    started_at = time.monotonic()
    with open(loss_log_path, "w") as loss_log:
        loss_log.write(
            json.dumps({"event": "run_start", "meta": metadata, "wall_time": datetime.datetime.now().isoformat()})
            + "\n"
        )
        loss_log.flush()
        if int(args.save_interval) > 0:
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
            if int(args.save_interval) > 0 and step % int(args.save_interval) == 0 and step != total_steps:
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
