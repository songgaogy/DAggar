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

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from robosuite.pipeline.offline.utils import (
    build_agent_env,
    finalize_normalizers,
    load_offline_data_transitions,
    populate_replay_buffer,
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
    """Build the DIPOLE agent + load ONLY the success_rollout split (no critics).

    Reuses the shared env/agent setup (:func:`build_agent_env`) and skips the
    IQL/nnPU/discriminator/advantage machinery entirely; every kept frame is
    forced to the positive branch (``mark_intervention=True``).
    """
    ctx = build_agent_env(cfg, log_tag="success_only")
    agent = ctx.agent
    try:
        # ------------------------------------------------------------------ #
        # Load ONLY success_rollout trajectories, forced to positive branch. #
        # ------------------------------------------------------------------ #
        success_transitions, _, filter_stats = load_offline_data_transitions(
            data_root=str(cfg.offline.data_root),
            task_name=ctx.task_name,
            offline_data_dir=str(cfg.offline.offline_data_dir),
            action_horizon=int(agent.flow_config.action_horizon),
            camera_names=ctx.policy_camera_names,
            image_size=ctx.img_height,
            episode_index_base=0,
            keep_kinds={"success"},
            mark_intervention=True,
        )
    finally:
        ctx.env.close()

    if not success_transitions:
        raise RuntimeError(
            "Success-only SFT loaded zero success transitions; check that "
            f"{cfg.offline.data_root}/{ctx.task_name}/{cfg.offline.offline_data_dir}/"
            "iql_offline_transitions.pt exists and contains success_rollout data."
        )
    n_valid = populate_replay_buffer(agent.online_buffer, success_transitions)
    print(
        f"[success_only] replay_buffer: {len(agent.online_buffer)} transitions "
        f"(success only, all w_pos=1), {n_valid} valid windows; "
        f"demo_buffer empty ({len(agent.demo_buffer)})"
    )

    batch_size = finalize_normalizers(
        agent, cfg, success_transitions,
        log_tag="success_only", norm_desc="success transitions",
    )

    return (
        agent,
        list(ctx.policy_camera_names),
        str(ctx.init_checkpoint),
        ctx.img_height,
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
