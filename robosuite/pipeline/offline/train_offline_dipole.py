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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from robosuite.pipeline.algorithms.discriminator.encoder import SharedDynamicsEncoder
from robosuite.pipeline.algorithms.discriminator.nnpu import FrozenNNPUDiscriminator
from robosuite.pipeline.algorithms.q_learning.common import IQLConfig
from robosuite.pipeline.algorithms.q_learning.iql import IQLLearner
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
from robosuite.pipeline.offline.utils.hard_label_providers import (
    NaiveNegativeGProvider,
    NegAllGProvider,
    precompute_neg_all_membership,
)
from robosuite.pipeline.offline.utils.setup import (
    build_agent_env,
    finalize_normalizers,
    make_hdf5_loader,
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


@dataclass
class OfflinePipeline:
    """Everything the offline DIPOLE run needs after setup.

    Built once by :func:`build_offline_pipeline` and consumed both by the
    training entry (:func:`main`) and by dev tools (e.g. ``dev/vis_batch.py``)
    that want the fully-populated replay buffer + frozen critics + precomputed
    TD advantage without re-implementing the setup.
    """

    agent: Any
    # ``provider`` is OfflineAdvantageGProvider in normal mode, NaiveNegativeGProvider
    # in naive mode; the RL-stack fields at the bottom are None in naive mode.
    provider: Any
    policy_camera_names: list[str]
    init_checkpoint: str
    filter_stats: dict[str, Any]
    num_pretrain_transitions: int
    num_offline_transitions: int
    n_valid: int
    batch_size: int
    sample_kwargs: dict[str, Any]
    static_cache: Any | None = None
    # RL-stack fields: populated in "normal" mode, left None in "naive" mode.
    iql_learner: IQLLearner | None = None
    discriminator: FrozenNNPUDiscriminator | None = None
    shared_encoder: SharedDynamicsEncoder | None = None
    iql_cfg: IQLConfig | None = None
    advantage_raw: torch.Tensor | None = None
    failure_raw: torch.Tensor | None = None
    start_to_row: dict[int, int] | None = None
    nnpu_ckpt: str | None = None
    warmup_ckpt: str | None = None


def build_offline_pipeline(cfg: DictConfig) -> OfflinePipeline:
    """Construct the full offline DIPOLE pipeline (no run_dir / training loop).

    Builds the env (headless, proprio only), the DIPOLE agent with the loaded
    pretrained flow policy, the frozen shared encoder + nnPU discriminator +
    frozen IQL critics, loads and filters the pretrain + offline data into the
    mixed replay buffer, fits normalizers, precomputes the TD advantage for
    every valid window, and attaches the :class:`OfflineAdvantageGProvider`.

    This is the exact setup the training entry used to perform inline; it is
    factored out so read-only tooling can reuse it verbatim.
    """
    mode = str(OmegaConf.select(cfg, "offline.mode", default="normal")).strip().lower()
    if mode not in ("normal", "naive", "neg_all"):
        raise ValueError(f"offline.mode must be 'normal', 'naive' or 'neg_all', got {mode!r}.")
    print(f"[offline] mode={mode}")

    ctx = build_agent_env(cfg, log_tag="offline")
    agent = ctx.agent
    task_name = ctx.task_name
    policy_camera_names = ctx.policy_camera_names
    img_height = ctx.img_height
    init_checkpoint = ctx.init_checkpoint
    try:
        # -------------------------------------------------------------- #
        # Frozen shared encoder + nnPU discriminator + frozen IQL.       #
        # (normal mode only; naive mode is a pure hard-label split.)     #
        # -------------------------------------------------------------- #
        nnpu_ckpt = None
        warmup_ckpt = None
        iql_cfg = None
        shared_encoder = None
        discriminator = None
        iql_learner = None
        if mode == "normal":
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
        else:
            print("[offline][naive] skipping encoder/discriminator/IQL construction (hard-label split).")

        # -------------------------------------------------------------- #
        # Load + filter offline data into the replay buffer.             #
        # -------------------------------------------------------------- #
        hdf5_loader = make_hdf5_loader(ctx, cfg)

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
        if mode == "normal":
            offline_transitions, episode_base, filter_stats = load_offline_data_transitions(
                data_root=str(cfg.offline.data_root),
                task_name=task_name,
                offline_data_dir=str(cfg.offline.offline_data_dir),
                action_horizon=int(iql_cfg.action_horizon),
                camera_names=policy_camera_names,
                image_size=img_height,
                episode_index_base=episode_base,
            )
        else:
            # naive / neg_all share the same hard-label data split:
            # success_rollout is marked is_intervention=True (-> positive branch),
            # fail_rollout is marked is_intervention=False. The negative-branch
            # weighting then differs by mode (naive: constant-negative G provider;
            # neg_all: per-frame membership provider + branch_weight_mode=neg_all).
            action_horizon = int(agent.flow_config.action_horizon)
            success_transitions, episode_base, success_stats = load_offline_data_transitions(
                data_root=str(cfg.offline.data_root),
                task_name=task_name,
                offline_data_dir=str(cfg.offline.offline_data_dir),
                action_horizon=action_horizon,
                camera_names=policy_camera_names,
                image_size=img_height,
                episode_index_base=episode_base,
                keep_kinds={"success"},
                mark_intervention=True,
            )
            fail_transitions, episode_base, fail_stats = load_offline_data_transitions(
                data_root=str(cfg.offline.data_root),
                task_name=task_name,
                offline_data_dir=str(cfg.offline.offline_data_dir),
                action_horizon=action_horizon,
                camera_names=policy_camera_names,
                image_size=img_height,
                episode_index_base=episode_base,
                keep_kinds={"fail"},
                mark_intervention=False,
            )
            offline_transitions = list(success_transitions) + list(fail_transitions)
            filter_stats = {
                key: int(success_stats.get(key, 0)) + int(fail_stats.get(key, 0))
                for key in set(success_stats) | set(fail_stats)
            }
            filter_stats["naive_success_transitions"] = int(len(success_transitions))
            filter_stats["naive_fail_transitions"] = int(len(fail_transitions))
    finally:
        ctx.env.close()

    if not pretrain_transitions and not offline_transitions:
        raise RuntimeError("Offline DIPOLE loaded zero transitions; check data paths.")
    # Fully mixed single buffer (uniform sampling). pretrain rows carry
    # is_intervention=True (forced w_pos=1); offline_data rows are advantage-weighted.
    all_transitions = list(pretrain_transitions) + list(offline_transitions)
    n_valid = populate_replay_buffer(agent.online_buffer, all_transitions)
    offline_role = "advantage-weighted" if mode == "normal" else "hard-labeled (success->pos, fail->neg)"
    print(
        f"[offline] replay_buffer (mixed): {len(agent.online_buffer)} transitions, "
        f"{n_valid} valid windows (pretrain={len(pretrain_transitions)} as w_pos=1, "
        f"offline_data={len(offline_transitions)} {offline_role}); "
        f"demo_buffer empty ({len(agent.demo_buffer)})"
    )

    batch_size = finalize_normalizers(
        agent, cfg, pretrain_transitions,
        log_tag="offline", norm_desc="pretrain transitions",
    )

    advantage_raw = None
    failure_raw = None
    start_to_row = None
    if mode == "normal":
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
    elif mode == "naive":
        # naive: fail frames (is_intervention=False) are the only provider-scored
        # rows; a constant negative G sends them to the negative branch (w_neg=1).
        provider = NaiveNegativeGProvider()
        agent.attach_g_provider(provider)
        print(
            "[offline][naive] attached NaiveNegativeGProvider "
            "(fail -> w_neg=1; success/expert forced w_pos=1)."
        )
    else:
        # neg_all: decoupled hard labels via the policy's branch_weight_mode.
        # w_pos = is_intervention (expert + success), w_neg = offline_data
        # membership (success + fail -> 1, expert -> 0). The negative branch
        # trains on ALL offline_data; the positive branch on success only.
        agent.core.config.branch_weight_mode = "neg_all"
        neg_raw, neg_start_to_row = precompute_neg_all_membership(agent.online_buffer)
        provider = NegAllGProvider(neg_raw, neg_start_to_row)
        agent.attach_g_provider(provider)
        print(
            f"[offline][neg_all] set branch_weight_mode=neg_all; attached "
            f"NegAllGProvider (neg windows={int(neg_raw.sum().item())}/{int(neg_raw.numel())}, "
            "expert->w_neg=0)."
        )

    sample_kwargs = {
        "action_mean": agent.core.act_mean,
        "action_std": agent.core.act_std,
        "proprio_mean": agent.core.prop_mean,
        "proprio_std": agent.core.prop_std,
        "device": agent.core.device,
        "augment": True,
    }
    static_cache = agent.online_buffer.build_static_cache(pin_memory=True)
    cache_mib = float(static_cache.estimated_bytes) / (1024.0 * 1024.0)
    print(
        f"[offline] static_cache rows={len(static_cache)} estimated={cache_mib:.1f} MiB "
        f"device=cpu pin_memory={static_cache.pin_memory}"
    )
    return OfflinePipeline(
        agent=agent,
        provider=provider,
        iql_learner=iql_learner,
        discriminator=discriminator,
        shared_encoder=shared_encoder,
        iql_cfg=iql_cfg,
        advantage_raw=advantage_raw,
        failure_raw=failure_raw,
        start_to_row=start_to_row,
        policy_camera_names=list(policy_camera_names),
        init_checkpoint=str(init_checkpoint),
        nnpu_ckpt=nnpu_ckpt,
        warmup_ckpt=warmup_ckpt,
        filter_stats=filter_stats,
        num_pretrain_transitions=int(len(pretrain_transitions)),
        num_offline_transitions=int(len(offline_transitions)),
        n_valid=int(n_valid),
        batch_size=batch_size,
        sample_kwargs=sample_kwargs,
        static_cache=static_cache,
    )


@hydra.main(version_base="1.2", config_path="../config", config_name="offline")
def main(cfg: DictConfig) -> None:
    pipe = build_offline_pipeline(cfg)
    agent = pipe.agent
    task_name = str(cfg.env.environment)
    policy_camera_names = pipe.policy_camera_names

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

    # ------------------------------------------------------------------ #
    # Logging + run metadata.                                            #
    # ------------------------------------------------------------------ #
    mode = str(OmegaConf.select(cfg, "offline.mode", default="normal")).strip().lower()
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
            "initialized_checkpoint": str(pipe.init_checkpoint),
            "nnpu_checkpoint": pipe.nnpu_ckpt,
            "iql_warmup_checkpoint": pipe.warmup_ckpt,
            "offline_mode": mode,
            "g_mode": {
                "normal": "advantage_offline_td",
                "naive": "naive_hard_split",
                "neg_all": "neg_all_hard_split",
            }[mode],
            "algorithm_type": {
                "normal": "dipole_offline",
                "naive": "dipole_naive",
                "neg_all": "dipole_neg_all",
            }[mode],
            "filter_stats": pipe.filter_stats,
            "replay_valid_windows": int(pipe.n_valid),
            "num_pretrain_transitions": int(pipe.num_pretrain_transitions),
            "num_offline_transitions": int(pipe.num_offline_transitions),
            "static_cache_enabled": pipe.static_cache is not None,
            "static_cache_pin_memory": bool(getattr(pipe.static_cache, "pin_memory", False)),
            "static_cache_estimated_bytes": int(getattr(pipe.static_cache, "estimated_bytes", 0)),
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

    batch_size = pipe.batch_size
    sample_kwargs = pipe.sample_kwargs
    static_cache = pipe.static_cache
    sampler_name = "static_cache" if static_cache is not None else "replay_buffer"
    print(f"[offline] training for {num_steps} steps (batch_size={batch_size}, sampler={sampler_name})")
    try:
        for step in range(num_steps):
            # Uniform draw from the single mixed buffer. No buffer_sources is set,
            # so the policy takes the legacy path: full-batch advantage G, then
            # is_intervention (pretrain) rows are overridden to w_pos=1.
            if static_cache is not None:
                batch = static_cache.sample(batch_size, **sample_kwargs)
            else:
                batch = agent.online_buffer.sample(batch_size, **sample_kwargs)
            collect_diag = (step % plot_log_interval == 0) or (step == num_steps - 1)

            # update
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
