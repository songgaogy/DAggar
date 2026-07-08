"""Success-only SFT for the DIPOLE flow policy (positive-branch only).

A deliberately stripped-down sibling of ``train_offline_dipole.py`` used as a
control experiment:

- Loads **only the ``success_rollout`` split** (the 50 success trajectories the
  IQL warmup exported into ``iql_offline_transitions.pt``), keeping only the
  pre-success frames (``info["success"]`` is False) — no expert ``pretrain_data``
  and no ``fail_rollout``.
- Every kept frame is marked ``is_intervention=True`` so the DIPOLE update forces
  ``w_pos=1, w_neg=0``: this is a plain positive-branch flow-matching SFT. The
  negative adapter (``neg_LoRA``) stays at its zero init, so ``neg == base`` and
  the checkpoint remains fully compatible with ``eval_offline_dipole`` (at
  ``--omega 0`` only the positive policy is used).
- The whole IQL / nnPU / discriminator / TD-advantage stack is skipped — no
  ``g_provider`` is attached (the update's legacy path uses ``zero_g`` and the
  ``is_intervention`` override pins ``w_pos=1`` regardless).
- Uses the **same LoRA config** as the main run (inherited via config).

Run as a Hydra module (config inherits ``offline``)::

    python -m robosuite.pipeline.offline.train_success_only \\
        env.environment=PickPlaceCereal \\
        runtime.init_checkpoint=checkpoints/.../flow.pt
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from robosuite.pipeline.envs import build_robosuite_env
from robosuite.pipeline.factory import build_algorithm
from robosuite.pipeline.offline.train_offline_dipole import _print_policy_param_summary
from robosuite.pipeline.offline.utils import (
    load_offline_data_transitions,
    populate_replay_buffer,
)
from robosuite.pipeline.train_dipole import (
    bind_flow_proprio_extractor,
    build_flow_runtime_cfg,
    load_hdf5_demos_into_flow_transitions,
    load_init_checkpoint_payload,
    maybe_set_seed,
    resolve_camera_names,
    resolve_flow_task_metadata,
)
from robosuite.pipeline.utils import (
    checkpoint_path,
    maybe_build_metric_logger,
    maybe_log,
    write_resolved_config,
    write_run_info,
)

logger = logging.getLogger(__name__)


def _build_success_only_agent(cfg: DictConfig):
    """Build the DIPOLE agent + load the pretrained flow policy (no critics).

    Mirrors the env/agent setup in ``build_offline_pipeline`` but omits the
    IQL/nnPU/discriminator/advantage machinery. Returns
    ``(agent, policy_camera_names, init_checkpoint, img_height)``.
    """
    maybe_set_seed(getattr(cfg, "seed", None))
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    init_checkpoint, init_payload = load_init_checkpoint_payload(cfg)
    if init_checkpoint is None:
        raise RuntimeError(
            "Success-only SFT requires runtime.init_checkpoint (the pretrained "
            "flow policy the LoRA adapters attach to) to be set."
        )
    task_name = str(cfg.env.environment)
    requested_camera_names = resolve_camera_names(cfg)
    flow_env_metadata = resolve_flow_task_metadata(init_payload, task_name)
    if bool(getattr(cfg.runtime, "use_init_checkpoint_camera_names", True)) and init_payload is not None:
        policy_camera_names = [str(name) for name in init_payload.get("camera_names", [])]
        if len(policy_camera_names) == 0:
            policy_camera_names = list(requested_camera_names)
    else:
        policy_camera_names = list(requested_camera_names)

    print(f"[success_only] init_checkpoint={init_checkpoint}")

    img_height = int(cfg.env.img_height)
    img_width = int(cfg.env.img_width)

    # Headless env for HDF5 proprio extraction only (no rendering), like warmup.
    main_runtime_cfg = build_flow_runtime_cfg(
        cfg,
        env_metadata=flow_env_metadata,
        camera_names=policy_camera_names,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        renderer=str(cfg.env.renderer),
    )
    env = build_robosuite_env(main_runtime_cfg)
    try:
        env.reset()
        extractor = bind_flow_proprio_extractor(env, flow_env_metadata)
        proprio_vec = np.asarray(
            extractor.extract(env.sim.get_state().flatten()), dtype=np.float32
        )
        observation_example: dict[str, Any] = {"state": proprio_vec}
        for camera_name in policy_camera_names:
            observation_example[camera_name] = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        action_low, action_high = env.action_spec
        action_low = np.asarray(action_low, dtype=np.float32)
        action_high = np.asarray(action_high, dtype=np.float32)

        algorithm_cfg = OmegaConf.to_container(cfg.algorithm, resolve=True)
        assert isinstance(algorithm_cfg, dict)
        algorithm_cfg["camera_names"] = list(policy_camera_names)
        algorithm_cfg["task_name"] = task_name
        flow_cfg = algorithm_cfg.setdefault("flow", {})
        flow_cfg.setdefault("image_size", img_height)
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
            observation_example=observation_example,
            sample_action=np.zeros_like(action_low, dtype=np.float32),
            action_low=action_low,
            action_high=action_high,
        )
        agent.load_flow_policy_checkpoint(init_checkpoint, task_name=task_name)
        print(f"[success_only] loaded pretrained flow policy from {init_checkpoint}")
        _print_policy_param_summary(agent.core)

        # ------------------------------------------------------------------ #
        # Load ONLY success_rollout trajectories, forced to positive branch. #
        # ------------------------------------------------------------------ #
        success_transitions, _, filter_stats = load_offline_data_transitions(
            data_root=str(cfg.offline.data_root),
            task_name=task_name,
            offline_data_dir=str(cfg.offline.offline_data_dir),
            action_horizon=int(agent.flow_config.action_horizon),
            camera_names=policy_camera_names,
            image_size=img_height,
            episode_index_base=0,
            keep_kinds={"success"},
            mark_intervention=True,
        )
    finally:
        env.close()

    if not success_transitions:
        raise RuntimeError(
            "Success-only SFT loaded zero success transitions; check that "
            f"{cfg.offline.data_root}/{task_name}/{cfg.offline.offline_data_dir}/"
            "iql_offline_transitions.pt exists and contains success_rollout data."
        )
    n_valid = populate_replay_buffer(agent.online_buffer, success_transitions)
    print(
        f"[success_only] replay_buffer: {len(agent.online_buffer)} transitions "
        f"(success only, all w_pos=1), {n_valid} valid windows; "
        f"demo_buffer empty ({len(agent.demo_buffer)})"
    )

    if not agent.has_normalizers():
        agent.fit_normalizers_from_transitions(success_transitions)
        print("[success_only] fitted flow normalizers from success transitions.")
    else:
        print("[success_only] reusing flow normalizers from the pretrained checkpoint.")

    batch_size = int(cfg.algorithm.trainer.batch_size)
    if agent.online_buffer.num_valid_sequences() < batch_size:
        raise RuntimeError(
            f"replay_buffer has only {agent.online_buffer.num_valid_sequences()} valid "
            f"windows (< batch_size={batch_size}); add more data or lower batch_size."
        )

    return (
        agent,
        list(policy_camera_names),
        str(init_checkpoint),
        img_height,
        filter_stats,
        int(n_valid),
        int(len(success_transitions)),
        batch_size,
    )


@hydra.main(version_base="1.2", config_path="../config", config_name="success_only")
def main(cfg: DictConfig) -> None:
    (
        agent,
        policy_camera_names,
        init_checkpoint,
        _img_height,
        filter_stats,
        n_valid,
        num_success_transitions,
        batch_size,
    ) = _build_success_only_agent(cfg)
    task_name = str(cfg.env.environment)

    # ------------------------------------------------------------------ #
    # Run directory: outputs/dipole_success_only/<task>/<timestamp>/     #
    # ------------------------------------------------------------------ #
    output_root = Path(to_absolute_path(str(cfg.logging.output_root)))
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_subfix = str(OmegaConf.select(cfg, "offline.run_subfix", default="") or "").strip()
    dir_name = f"{timestamp}_{run_subfix}" if run_subfix else timestamp
    run_name = f"{task_name}__success_only__{dir_name}"
    run_dir = output_root / task_name / dir_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    print(f"[success_only] run={run_name}")
    print(f"[success_only] run_dir={run_dir}")

    metric_logger = maybe_build_metric_logger(cfg, run_name=run_name, run_dir=run_dir)
    write_resolved_config(cfg, run_dir)
    write_run_info(
        run_dir,
        {
            "run_name": run_name,
            "run_dir": str(run_dir),
            "started_at": timestamp,
            "env_name": task_name,
            "task_name": task_name,
            "policy_camera_names": list(policy_camera_names),
            # eval resolves env metadata from this base flow checkpoint.
            "initialized_checkpoint": str(init_checkpoint),
            "g_mode": "success_only_sft",
            "algorithm_type": "dipole_success_only",
            "filter_stats": filter_stats,
            "replay_valid_windows": int(n_valid),
            "num_success_transitions": int(num_success_transitions),
        },
    )

    def _save(tag: str, step: int) -> Path:
        payload = agent.build_checkpoint_payload(
            include_buffers=False,
            extra={"global_step": int(step), "run_name": run_name},
        )
        path = checkpoint_path(run_dir, tag)
        agent.write_checkpoint_payload(path, payload)
        return path

    # ------------------------------------------------------------------ #
    # Positive-branch-only SFT loop (no critics, no advantage).          #
    # ------------------------------------------------------------------ #
    num_steps = int(cfg.offline.num_train_steps)
    log_interval = max(1, int(cfg.offline.log_interval))
    checkpoint_interval = max(1, int(cfg.offline.checkpoint_interval))

    sample_kwargs = {
        "action_mean": agent.core.act_mean,
        "action_std": agent.core.act_std,
        "proprio_mean": agent.core.prop_mean,
        "proprio_std": agent.core.prop_std,
        "device": agent.core.device,
        "augment": True,
    }
    print(f"[success_only] training for {num_steps} steps (batch_size={batch_size})")
    try:
        for step in range(num_steps):
            batch = agent.online_buffer.sample(batch_size, **sample_kwargs)
            # No g_provider attached + all rows is_intervention=True => w_pos=1,
            # w_neg=0: plain positive-branch flow-matching SFT.
            metrics = agent.core.update(batch=batch, collect_diagnostics=False)
            metrics.pop("_diag", None)
            if step % log_interval == 0 or step == num_steps - 1:
                maybe_log(
                    metric_logger,
                    {f"train/{key}": float(value) for key, value in metrics.items()},
                    step=step,
                )
                print(
                    f"[success_only][step {step:6d}] "
                    f"loss={metrics.get('actor_loss', 0.0):.4f} "
                    f"flow_loss_pos={metrics.get('flow_loss_pos', 0.0):.4f} "
                    f"w_pos={metrics.get('w_pos_mean', 0.0):.3f}"
                )
            if step > 0 and step % checkpoint_interval == 0:
                step_path = _save(f"step_{step:08d}", step)
                _save("latest", step)
                print(f"[success_only][ckpt] step={step} -> {step_path.name} (+latest)")
    finally:
        final_step = max(0, num_steps - 1)
        final_path = _save("latest", final_step)
        _save(f"step_{final_step:08d}", final_step)
        if metric_logger is not None:
            metric_logger.close()
        print(f"[success_only] done. final checkpoint -> {final_path}")


if __name__ == "__main__":
    main()
