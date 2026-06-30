"""Offline DIPOLE training entry (README step-4).

Fine-tunes the pretrained DIPOLE flow policy on disk data using the DIPOLE-RL
branch-weighting update, but **without any environment rollout**:

- ``pretrain_data`` (expert) + filtered ``offline_data`` are fully mixed in one
  replay buffer (``agent.online_buffer``) and sampled uniformly; the demo buffer
  stays empty.
- ``offline_data`` rows are weighted by the **TD advantage**
  ``A = V(s) - gamma^H V(s') - r`` (see :mod:`offline.utils.advantage`) instead
  of the online ``Q - V``.
- ``pretrain_data`` rows carry ``is_intervention=True`` so the policy forces
  ``w_pos=1, w_neg=0`` on them (positive-branch-only update).
- The IQL critics are **frozen** (loaded from the warmup checkpoint, never
  updated). They only supply the advantage.

Run as a Hydra module (config inherits ``train_dipole_rl``)::

    python -m robosuite.pipeline.offline.train_offline_dipole \\
        env.environment=PickPlaceCereal \\
        runtime.init_checkpoint=checkpoints/.../flow.pt \\
        algorithm.discriminator.checkpoint=.../pu_bce_head.pth \\
        algorithm.q_learning.warmup_ckpt=.../iql_state.pt
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

from robosuite.pipeline.algorithms.discriminator.encoder import SharedDynamicsEncoder
from robosuite.pipeline.algorithms.discriminator.nnpu import FrozenNNPUDiscriminator
from robosuite.pipeline.algorithms.q_learning.common import IQLConfig
from robosuite.pipeline.algorithms.q_learning.iql import IQLLearner
from robosuite.pipeline.envs import build_robosuite_env
from robosuite.pipeline.factory import build_algorithm
from robosuite.pipeline.offline.diagnostic_plots import (
    plot_branch_weight_distribution,
    plot_raw_g_distribution,
    plot_v_pos_neg_scatter,
)
from robosuite.pipeline.offline.utils import (
    OfflineAdvantageGProvider,
    load_offline_data_transitions,
    load_pretrain_transitions,
    populate_replay_buffer,
    precompute_offline_advantage,
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
from robosuite.pipeline.train_dipole_rl import _load_iql_warmup_state
from robosuite.pipeline.utils import (
    checkpoint_path,
    maybe_build_metric_logger,
    maybe_log,
    maybe_log_figure,
    write_resolved_config,
    write_run_info,
)

logger = logging.getLogger(__name__)


def _print_policy_param_summary(core: Any) -> None:
    """Print LoRA adapter vs base trainable parameter counts after policy init."""
    model = core.model
    lora_numel = 0
    base_numel = 0
    frozen_numel = 0
    lora_module_paths: list[str] = []
    for name, param in model.named_parameters():
        n = int(param.numel())
        if not param.requires_grad:
            frozen_numel += n
            continue
        if ".lora_A" in name or ".lora_B" in name:
            lora_numel += n
            if name.endswith(".lora_A"):
                lora_module_paths.append(name[: -len(".lora_A")])
        else:
            base_numel += n

    trainable_numel = lora_numel + base_numel
    total_numel = trainable_numel + frozen_numel
    lora_num_modules = int(getattr(model, "lora_num_modules", len(lora_module_paths)))
    print(
        f"[offline] policy trainable params: total={trainable_numel:,} "
        f"(base={base_numel:,}, lora={lora_numel:,})"
    )
    print(
        f"[offline] policy frozen params: {frozen_numel:,} "
        f"(all params={total_numel:,})"
    )
    print(
        f"[offline] lora modules={lora_num_modules} "
        f"(rank={getattr(core.config, 'lora_rank', '?')}, "
        f"alpha={getattr(core.config, 'lora_alpha', '?')}, "
        f"include_aggregator={getattr(core.config, 'lora_include_aggregator', '?')})"
    )
    if lora_module_paths:
        print("[offline] lora target paths:")
        for path in lora_module_paths:
            print(f"  - {path}")


def _freeze_iql(iql: IQLLearner) -> None:
    """Disable grads on every IQL module so it is never updated offline."""
    for module in (
        iql.chunk_projector,
        iql.action_projector,
        iql.q_ensemble,
        iql.v,
        iql.target_v,
    ):
        for param in module.parameters():
            param.requires_grad_(False)


def _resolve_required_path(raw: Any, *, what: str) -> str:
    if raw is None or str(raw).strip().lower() in ("", "null"):
        raise RuntimeError(f"Offline DIPOLE requires {what} to be set.")
    path = to_absolute_path(str(raw))
    if not Path(path).exists():
        raise FileNotFoundError(f"{what} not found: {path}")
    return path


@hydra.main(version_base="1.2", config_path="../config", config_name="offline")
def main(cfg: DictConfig) -> None:
    maybe_set_seed(getattr(cfg, "seed", None))
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    init_checkpoint, init_payload = load_init_checkpoint_payload(cfg)
    if init_checkpoint is None:
        raise RuntimeError(
            "Offline DIPOLE requires runtime.init_checkpoint (the pretrained flow "
            "policy) to be set."
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
    camera_aliases = {
        str(key): str(value)
        for key, value in dict(getattr(cfg.algorithm.flow, "camera_aliases", {}) or {}).items()
    }

    # ------------------------------------------------------------------ #
    # Run directory: outputs/dipole_offline/<task>/<timestamp>/          #
    # ------------------------------------------------------------------ #
    output_root = Path(to_absolute_path(str(cfg.logging.output_root)))
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_subfix = str(OmegaConf.select(cfg, "offline.run_subfix", default="") or "").strip()
    dir_name = f"{timestamp}_{run_subfix}" if run_subfix else timestamp
    run_name = f"{task_name}__offline__{dir_name}"
    run_dir = output_root / task_name / dir_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    print(f"[offline] run={run_name}")
    print(f"[offline] run_dir={run_dir}")
    print(f"[offline] init_checkpoint={init_checkpoint}")

    img_height = int(cfg.env.img_height)
    img_width = int(cfg.env.img_width)
    reward_mode = str(cfg.algorithm.q_learning.config.reward_mode)

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

        # -------------------------------------------------------------- #
        # Build the DIPOLE agent and load the pretrained flow policy.    #
        # -------------------------------------------------------------- #
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
        print(f"[offline] loaded pretrained flow policy from {init_checkpoint}")
        _print_policy_param_summary(agent.core)

        # -------------------------------------------------------------- #
        # Frozen shared encoder + nnPU discriminator + frozen IQL.       #
        # -------------------------------------------------------------- #
        nnpu_ckpt = _resolve_required_path(
            cfg.algorithm.discriminator.checkpoint,
            what="algorithm.discriminator.checkpoint (NNPU_CKPT)",
        )
        cfg.algorithm.discriminator.checkpoint = nnpu_ckpt
        rl_device = str(cfg.algorithm.q_learning.config.device)
        camera_to_view = {
            str(k): str(v)
            for k, v in dict(getattr(cfg.algorithm.discriminator, "camera_to_view", {}) or {}).items()
        }
        encoder_override = getattr(cfg.algorithm.discriminator, "encoder_ckpt", None)
        encoder_ckpt = (
            None
            if encoder_override is None or str(encoder_override).strip().lower() in ("", "null")
            else to_absolute_path(str(encoder_override))
        )
        shared_encoder = SharedDynamicsEncoder(
            nnpu_ckpt_path=nnpu_ckpt,
            encoder_ckpt=encoder_ckpt,
            device=rl_device,
            camera_to_view=camera_to_view,
        )
        shared_encoder.bind_policy_cameras(list(agent.camera_names))
        discriminator = FrozenNNPUDiscriminator(
            nnpu_ckpt_path=nnpu_ckpt,
            task_name=task_name,
            device=rl_device,
            encoder=shared_encoder,
        )
        print(
            f"[offline] encoder ckpt={nnpu_ckpt} device={rl_device} "
            f"state_dim={shared_encoder.state_feature_dim} chunk_dim={shared_encoder.chunk_feature_dim}"
        )

        policy_action_dim = int(agent.flow_config.action_dim)
        iql_cfg = IQLConfig(**OmegaConf.to_container(cfg.algorithm.q_learning.config, resolve=True))
        iql_learner = IQLLearner(
            iql_cfg,
            state_feature_dim=int(shared_encoder.state_feature_dim),
            chunk_feature_dim=int(shared_encoder.chunk_feature_dim),
            action_dim=policy_action_dim,
            n_tokens=int(shared_encoder.inner_encoder.num_patches),
            proprio_dim=int(shared_encoder.inner_encoder.proprio_emb_dim),
        )
        warmup_ckpt = _resolve_required_path(
            cfg.algorithm.q_learning.warmup_ckpt,
            what="algorithm.q_learning.warmup_ckpt (IQL warmup state)",
        )
        _load_iql_warmup_state(
            iql_learner,
            warmup_ckpt,
            expected_state_feature_dim=int(shared_encoder.state_feature_dim),
            expected_chunk_feature_dim=int(shared_encoder.chunk_feature_dim),
            expected_action_dim=policy_action_dim,
        )
        _freeze_iql(iql_learner)
        print(f"[offline] loaded + froze IQL critics from {warmup_ckpt}")

        # -------------------------------------------------------------- #
        # Load + filter offline data into the replay buffer.             #
        # -------------------------------------------------------------- #
        def hdf5_loader(path, demo_names=None):
            return load_hdf5_demos_into_flow_transitions(
                path,
                policy_camera_names=policy_camera_names,
                camera_aliases=camera_aliases,
                img_height=img_height,
                img_width=img_width,
                proprio_keys=tuple(cfg.env.proprio_keys or []),
                renderer=str(cfg.env.renderer),
                control_freq=int(cfg.env.control_freq),
                demo_names=demo_names,
                state_extractor=extractor,
                reward_mode=reward_mode,
            )

        max_pretrain = OmegaConf.select(cfg, "offline.max_pretrain_trajectories", default=None)
        max_pretrain = None if max_pretrain is None else int(max_pretrain)
        pretrain_transitions, episode_base = load_pretrain_transitions(
            data_root=str(cfg.offline.data_root),
            task_name=task_name,
            pretrain_dir=str(cfg.offline.pretrain_dir),
            hdf5_loader=hdf5_loader,
            max_num_trajectories=max_pretrain,
            episode_index_base=0,
        )
        offline_transitions, episode_base, filter_stats = load_offline_data_transitions(
            data_root=str(cfg.offline.data_root),
            task_name=task_name,
            offline_data_dir=str(cfg.offline.offline_data_dir),
            action_horizon=int(iql_cfg.action_horizon),
            camera_names=policy_camera_names,
            image_size=img_height,
            episode_index_base=episode_base,
        )
    finally:
        env.close()

    if not pretrain_transitions and not offline_transitions:
        raise RuntimeError("Offline DIPOLE loaded zero transitions; check data paths.")
    # Fully mixed single buffer (uniform sampling). pretrain rows carry
    # is_intervention=True (forced w_pos=1); offline_data rows are advantage-weighted.
    all_transitions = list(pretrain_transitions) + list(offline_transitions)
    n_valid = populate_replay_buffer(agent.online_buffer, all_transitions)
    print(
        f"[offline] replay_buffer (mixed): {len(agent.online_buffer)} transitions, "
        f"{n_valid} valid windows (pretrain={len(pretrain_transitions)} as w_pos=1, "
        f"offline_data={len(offline_transitions)} advantage-weighted); "
        f"demo_buffer empty ({len(agent.demo_buffer)})"
    )

    if not agent.has_normalizers():
        agent.fit_normalizers_from_transitions(pretrain_transitions)
        print("[offline] fitted flow normalizers from pretrain transitions.")
    else:
        print("[offline] reusing flow normalizers from the pretrained checkpoint.")

    batch_size = int(cfg.algorithm.trainer.batch_size)
    if agent.online_buffer.num_valid_sequences() < batch_size:
        raise RuntimeError(
            f"replay_buffer has only {agent.online_buffer.num_valid_sequences()} valid "
            f"windows (< batch_size={batch_size}); add more data or lower batch_size."
        )

    # Precompute TD advantage (+ failure score) for every valid window once.
    # The provider scores the whole batch; pretrain (is_intervention) rows have
    # their weight overridden to w_pos=1 afterwards by the policy, so their
    # advantage value is computed but unused.
    advantage_raw, failure_raw, start_to_row = precompute_offline_advantage(
        base_buffer=agent.online_buffer,
        iql_learner=iql_learner,
        encoder=shared_encoder,
        discriminator=discriminator,
        iql_cfg=iql_cfg,
        device=rl_device,
        encode_batch_size=int(OmegaConf.select(cfg, "offline.preencode_batch_size", default=64)),
    )
    provider = OfflineAdvantageGProvider(
        iql_learner=iql_learner,
        discriminator=discriminator,
        encoder=shared_encoder,
        alpha=float(cfg.algorithm.advantage_g_provider.alpha),
        beta=float(cfg.algorithm.advantage_g_provider.beta),
        advantage_raw=advantage_raw,
        failure_raw=failure_raw,
        start_to_row=start_to_row,
    )
    provider.bind_policy_cameras(list(agent.camera_names))
    agent.attach_iql_learner(iql_learner)
    agent.attach_discriminator(discriminator)
    agent.attach_g_provider(provider)
    print(
        f"[offline] attached OfflineAdvantageGProvider "
        f"(alpha={cfg.algorithm.advantage_g_provider.alpha}, beta={cfg.algorithm.advantage_g_provider.beta})"
    )

    # ------------------------------------------------------------------ #
    # Logging + run metadata.                                            #
    # ------------------------------------------------------------------ #
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
            "initialized_checkpoint": str(init_checkpoint),
            "nnpu_checkpoint": nnpu_ckpt,
            "iql_warmup_checkpoint": warmup_ckpt,
            "g_mode": "advantage_offline_td",
            "algorithm_type": "dipole_offline",
            "filter_stats": filter_stats,
            "replay_valid_windows": int(n_valid),
            "num_pretrain_transitions": int(len(pretrain_transitions)),
            "num_offline_transitions": int(len(offline_transitions)),
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
    # Offline training loop (policy-only; IQL frozen).                   #
    # ------------------------------------------------------------------ #
    num_steps = int(cfg.offline.num_train_steps)
    log_interval = max(1, int(cfg.offline.log_interval))
    plot_log_interval_raw = OmegaConf.select(cfg, "offline.plot_log_interval", default=None)
    plot_log_interval = max(1, int(log_interval if plot_log_interval_raw is None else plot_log_interval_raw))
    checkpoint_interval = max(1, int(cfg.offline.checkpoint_interval))

    sample_kwargs = {
        "action_mean": agent.core.act_mean,
        "action_std": agent.core.act_std,
        "proprio_mean": agent.core.prop_mean,
        "proprio_std": agent.core.prop_std,
        "device": agent.core.device,
        "augment": True,
    }
    print(f"[offline] training for {num_steps} steps (batch_size={batch_size})")
    try:
        for step in range(num_steps):
            # Uniform draw from the single mixed buffer. No buffer_sources is set,
            # so the policy takes the legacy path: full-batch advantage G, then
            # is_intervention (pretrain) rows are overridden to w_pos=1.
            batch = agent.online_buffer.sample(batch_size, **sample_kwargs)
            collect_diag = (step % plot_log_interval == 0) or (step == num_steps - 1)
            metrics = agent.core.update(batch=batch, collect_diagnostics=collect_diag)
            diag = metrics.pop("_diag", None)
            if step % log_interval == 0 or step == num_steps - 1:
                maybe_log(
                    metric_logger,
                    {f"train/{key}": float(value) for key, value in metrics.items()},
                    step=step,
                )
                print(
                    f"[offline][step {step:6d}] loss={metrics.get('actor_loss', 0.0):.4f} "
                    f"w_pos={metrics.get('w_pos_mean', 0.0):.3f} "
                    f"G_mean={metrics.get('G_mean', 0.0):+.3f} "
                    f"raw_mean={metrics.get('raw_nnpu_score_mean', 0.0):+.3f}"
                )
            if diag is not None:
                fig_g = plot_raw_g_distribution(diag["g_provider_raw"])
                maybe_log_figure(metric_logger, "train/raw_g_hist", fig_g, step)
                fig_w_pos = plot_branch_weight_distribution(diag["w_pos"], label="w_pos")
                maybe_log_figure(metric_logger, "train/w_pos_hist", fig_w_pos, step)
                fig_w_neg = plot_branch_weight_distribution(diag["w_neg"], label="w_neg")
                maybe_log_figure(metric_logger, "train/w_neg_hist", fig_w_neg, step)
                fig_v = plot_v_pos_neg_scatter(diag["v_pos_mse"], diag["v_neg_mse"])
                maybe_log_figure(metric_logger, "train/v_pos_neg_mse_scatter", fig_v, step)
            if step > 0 and step % checkpoint_interval == 0:
                step_path = _save(f"step_{step:08d}", step)
                _save("latest", step)
                print(f"[offline][ckpt] step={step} -> {step_path.name} (+latest)")
    finally:
        final_step = max(0, num_steps - 1)
        final_path = _save("latest", final_step)
        _save(f"step_{final_step:08d}", final_step)
        if metric_logger is not None:
            metric_logger.close()
        print(f"[offline] done. final checkpoint -> {final_path}")


if __name__ == "__main__":
    main()
