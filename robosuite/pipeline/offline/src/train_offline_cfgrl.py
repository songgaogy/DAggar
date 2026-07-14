"""Offline CFGRL training with independent positive and negative datasets.

The positive policy is trained exactly like offline DAgger: human intervention
actions, expert pretrain demonstrations, and optionally pure on-policy success
rollouts. The negative policy is trained from the full Offline DIPOLE Phase-B
dataset using either uniform weights (``all``) or the routed V/GAE-derived
negative weights used by Offline DIPOLE (``neg-weighted``).
"""

from __future__ import annotations

import datetime
import shutil
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from robosuite.pipeline.algorithms.dipole.common import DipoleBatch
from robosuite.pipeline.algorithms.dipole.replay_buffer import DipoleReplayBuffer
from robosuite.pipeline.algorithms.discriminator.encoder import SharedDynamicsEncoder
from robosuite.pipeline.algorithms.discriminator.nnpu import FrozenNNPUDiscriminator
from robosuite.pipeline.algorithms.q_learning.common import IQLConfig
from robosuite.pipeline.algorithms.q_learning.iql import IQLLearner
from robosuite.pipeline.offline.src.train_dagger import (
    _next_episode_index,
    _resolve_episodes_path,
    _resolve_required_path,
    _weighted_mean,
)
from robosuite.pipeline.offline.src.train_offline_dipole import (
    _as_bool,
    _freeze_iql,
    _warmup_transitions_path,
)
from robosuite.pipeline.offline.utils import (
    OfflineAdvantageGProvider,
    build_agent_env,
    build_branch_weight_policy,
    build_iql_finetune_buffer,
    build_offline_transitions,
    build_online_success_transitions,
    finalize_normalizers,
    finetune_iql,
    load_pretrain_transitions,
    make_hdf5_loader,
    populate_replay_buffer,
    precompute_offline_advantage,
    save_finetuned_iql,
)
from robosuite.pipeline.offline.utils.episode_dataset import ROUTE_POS_ONLY
from robosuite.pipeline.train_dipole_rl import _load_iql_warmup_state
from robosuite.pipeline.utils import (
    checkpoint_path,
    maybe_build_metric_logger,
    maybe_log,
    write_resolved_config,
    write_run_info,
)


_CFG_MODES = frozenset({"all", "neg-weighted"})


def _normalize_cfg_mode(raw: Any) -> str:
    mode = str(raw).strip().lower()
    if mode not in _CFG_MODES:
        expected = ", ".join(sorted(_CFG_MODES))
        raise ValueError(f"offline.cfg_mode must be one of {{{expected}}}, got {raw!r}.")
    return mode


def _require_cuda_device(device: str) -> None:
    if not str(device).startswith("cuda"):
        raise RuntimeError(f"Offline CFGRL requires a CUDA device, got {device!r}.")
    if not torch.cuda.is_available():
        raise RuntimeError("Offline CFGRL requires CUDA, but torch.cuda.is_available() is false.")


def _compute_cfgrl_neg_weights(
    core: Any,
    batch: DipoleBatch,
    *,
    mode: str,
    branch_policy: Any = None,
    g_provider: Any = None,
    want_metrics: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return negative-policy sample weights for the selected CFGRL mode."""
    mode = _normalize_cfg_mode(mode)
    if mode == "all":
        weights = torch.ones(batch.batch_size, dtype=torch.float32, device=core.device)
        metrics = {"w_neg_mean": 1.0, "w_neg_std": 0.0} if want_metrics else {}
        return weights, metrics
    if branch_policy is None or g_provider is None:
        raise RuntimeError("neg-weighted mode requires a branch policy and G provider.")
    _, weights, metrics = branch_policy(
        batch,
        g_provider=g_provider,
        sigmoid_fn=core._g_weights_from_raw,
        device=core.device,
        want_metrics=want_metrics,
    )
    return weights, metrics


def _branch_only_update(
    core: Any,
    batch: DipoleBatch,
    *,
    branch: str,
    weights: torch.Tensor,
    want_metrics: bool = True,
) -> dict[str, float]:
    """Update exactly one of the two independent flow-policy branches."""
    if branch not in ("pos", "neg"):
        raise ValueError(f"branch must be 'pos' or 'neg', got {branch!r}.")

    batch = batch.to(core.device)
    weights = weights.to(device=core.device, dtype=torch.float32).reshape(-1)
    if weights.numel() != batch.batch_size:
        raise ValueError(
            f"weights has {weights.numel()} rows, expected batch_size={batch.batch_size}."
        )

    model = core.model_pos if branch == "pos" else core.model_neg
    optimizer = core.optimizer_pos if branch == "pos" else core.optimizer_neg
    scaler = core.scaler_pos if branch == "pos" else core.scaler_neg

    batch_size = batch.batch_size
    noise = torch.randn_like(batch.action_sequences)
    timesteps = torch.rand(batch_size, device=core.device)
    x_t = (
        (1.0 - timesteps).view(-1, 1, 1) * noise
        + timesteps.view(-1, 1, 1) * batch.action_sequences
    )
    v_target = batch.action_sequences - noise
    language = [core.language_instruction] * batch_size

    model.train(True)
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(enabled=(core.device.type == "cuda"), device_type=core.device.type):
        v_pred = model(
            x_t=x_t.transpose(1, 2),
            t=timesteps,
            images=batch.image_obs,
            proprio=batch.proprio,
            language=language,
        ).transpose(1, 2)
        flow_per_row = torch.mean((v_pred - v_target) ** 2, dim=(1, 2))
        x1 = x_t + (1.0 - timesteps).view(-1, 1, 1) * v_pred
        endpoint_per_row = torch.mean((x1 - batch.action_sequences) ** 2, dim=(1, 2))
        if batch.action_sequences.shape[1] > 1:
            smooth_per_row = torch.mean((x1[:, 1:] - x1[:, :-1]) ** 2, dim=(1, 2))
        else:
            smooth_per_row = torch.zeros(
                batch_size,
                device=core.device,
                dtype=flow_per_row.dtype,
            )
        flow = _weighted_mean(flow_per_row, weights)
        endpoint = _weighted_mean(endpoint_per_row, weights)
        smooth = _weighted_mean(smooth_per_row, weights)
        loss = (
            flow
            + float(core.config.lambda_endpoint) * endpoint
            + float(core.config.lambda_smooth) * smooth
        )

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=float(core.config.grad_clip_norm),
    )
    scaler.step(optimizer)
    scaler.update()

    if not want_metrics:
        return {}
    inactive = "neg" if branch == "pos" else "pos"
    return {
        "actor_loss": float(loss.detach().cpu().item()),
        f"loss_{branch}": float(loss.detach().cpu().item()),
        f"loss_{inactive}": 0.0,
        f"flow_loss_{branch}": float(flow.detach().cpu().item()),
        f"endpoint_loss_{branch}": float(endpoint.detach().cpu().item()),
        f"smooth_loss_{branch}": float(smooth.detach().cpu().item()),
        f"grad_norm_{branch}": float(grad_norm.detach().cpu().item()),
        f"w_{branch}_mean": float(weights.mean().detach().cpu().item()),
        f"w_{branch}_std": float(
            weights.std().detach().cpu().item() if batch_size > 1 else 0.0
        ),
        "frac_intervention": float(
            batch.is_intervention.float().mean().detach().cpu().item()
        ),
    }


def _build_weighted_neg_stack(
    cfg: DictConfig,
    *,
    agent: Any,
    task_name: str,
    camera_names: list[str],
) -> tuple[
    SharedDynamicsEncoder,
    FrozenNNPUDiscriminator,
    IQLLearner,
    IQLConfig,
    str,
    str | None,
    str | None,
    bool,
]:
    """Build the discriminator and IQL stack required by neg-weighted mode."""
    rl_device = str(cfg.algorithm.q_learning.config.device)
    _require_cuda_device(rl_device)
    nnpu_ckpt = _resolve_required_path(
        cfg.algorithm.discriminator.checkpoint,
        what="algorithm.discriminator.checkpoint (NNPU_CKPT)",
    )
    encoder_override = getattr(cfg.algorithm.discriminator, "encoder_ckpt", None)
    encoder_ckpt = (
        None
        if encoder_override is None or str(encoder_override).strip().lower() in ("", "null")
        else to_absolute_path(str(encoder_override))
    )
    camera_to_view = {
        str(key): str(value)
        for key, value in dict(
            getattr(cfg.algorithm.discriminator, "camera_to_view", {}) or {}
        ).items()
    }
    encoder = SharedDynamicsEncoder(
        nnpu_ckpt_path=nnpu_ckpt,
        encoder_ckpt=encoder_ckpt,
        device=rl_device,
        camera_to_view=camera_to_view,
    )
    encoder.bind_policy_cameras(camera_names)
    discriminator = FrozenNNPUDiscriminator(
        nnpu_ckpt_path=nnpu_ckpt,
        task_name=task_name,
        device=rl_device,
        encoder=encoder,
    )

    iql_cfg = IQLConfig(**OmegaConf.to_container(cfg.algorithm.q_learning.config, resolve=True))
    skip_rl = _as_bool(OmegaConf.select(cfg, "offline.skip_rl", default=False))
    warmup_ckpt: str | None = None
    iql_finetuned_override: str | None = None
    if skip_rl:
        iql_finetuned_override = _resolve_required_path(
            OmegaConf.select(cfg, "offline.iql_finetuned_path", default=None),
            what="offline.iql_finetuned_path (finetuned IQL state)",
        )
        initial_iql_ckpt = iql_finetuned_override
    else:
        warmup_ckpt = _resolve_required_path(
            cfg.algorithm.q_learning.warmup_ckpt,
            what="algorithm.q_learning.warmup_ckpt (pretrained IQL state)",
        )
        initial_iql_ckpt = warmup_ckpt

    payload = torch.load(initial_iql_ckpt, map_location="cpu", weights_only=False)
    warmup_meta = payload.get("encoder_meta", {}) if isinstance(payload, dict) else {}
    warmup_cfg = payload.get("cfg", {}) if isinstance(payload, dict) else {}
    for field in (
        "output_reward_coef",
        "disc_reward_coef",
        "expectile_tau",
        "ensemble_lcb_beta",
        "ensemble_bootstrap_prob",
        "discount",
    ):
        source = warmup_cfg if field in warmup_cfg else (
            warmup_meta if field in warmup_meta else None
        )
        if source is None:
            continue
        old = getattr(iql_cfg, field)
        new = type(old)(source[field])
        if old != new:
            print(f"[cfgrl][iql] aligning {field}: {old} -> {new} (from checkpoint)")
        setattr(iql_cfg, field, new)

    action_horizon = int(iql_cfg.action_horizon)
    if int(agent.flow_config.action_horizon) != action_horizon:
        raise RuntimeError(
            "action_horizon mismatch: "
            f"policy={agent.flow_config.action_horizon} vs IQL={action_horizon}."
        )
    action_dim = int(agent.flow_config.action_dim)
    learner = IQLLearner(
        iql_cfg,
        state_feature_dim=int(encoder.state_feature_dim),
        chunk_feature_dim=int(encoder.chunk_feature_dim),
        action_dim=action_dim,
        n_tokens=int(encoder.inner_encoder.num_patches),
        proprio_dim=int(encoder.inner_encoder.proprio_emb_dim),
    )
    _load_iql_warmup_state(
        learner,
        initial_iql_ckpt,
        expected_state_feature_dim=int(encoder.state_feature_dim),
        expected_chunk_feature_dim=int(encoder.chunk_feature_dim),
        expected_action_dim=action_dim,
    )
    print(f"[cfgrl][iql] loaded critics from {initial_iql_ckpt}")
    return (
        encoder,
        discriminator,
        learner,
        iql_cfg,
        nnpu_ckpt,
        warmup_ckpt,
        iql_finetuned_override,
        skip_rl,
    )


@hydra.main(version_base="1.2", config_path="../../config", config_name="train_offline_cfgrl")
def main(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision("high")
    mode = _normalize_cfg_mode(OmegaConf.select(cfg, "offline.cfg_mode", default="all"))

    flow_device = str(OmegaConf.select(cfg, "algorithm.flow.device", default="cuda:0"))
    inference_device = str(
        OmegaConf.select(cfg, "algorithm.flow.inference_device", default=flow_device)
    )
    _require_cuda_device(flow_device)
    _require_cuda_device(inference_device)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ctx = build_agent_env(cfg, log_tag="cfgrl")
    agent = ctx.agent
    task_name = ctx.task_name
    camera_names = list(agent.camera_names)
    action_horizon = int(agent.flow_config.action_horizon)

    episodes_path = _resolve_episodes_path(cfg, task_name)
    episodes_payload = torch.load(episodes_path, map_location="cpu", weights_only=False)
    payload_cameras = [str(name) for name in episodes_payload.get("camera_names", [])]
    if payload_cameras and set(payload_cameras) != set(camera_names):
        raise RuntimeError(
            f"camera mismatch: episodes {payload_cameras} vs policy {camera_names}. "
            "Collect data with the same init_checkpoint."
        )
    streams = build_offline_transitions(
        episodes_payload,
        action_horizon=action_horizon,
        reward_success=float(OmegaConf.select(cfg, "offline.reward_success", default=0.0)),
        reward_fail=float(OmegaConf.select(cfg, "offline.reward_fail", default=-1.0)),
        include_policy_action_neg=True,
    )
    if not streams.policy_bc:
        raise RuntimeError("No policy sections survived filtering for CFGRL negative data.")
    print(f"[cfgrl] cfg_mode={mode} episodes={episodes_path}")
    print(f"[cfgrl] streams: {streams.stats}")

    use_online_success = bool(
        OmegaConf.select(cfg, "offline.use_online_success", default=True)
    )
    next_episode_index = _next_episode_index(
        streams.policy_bc,
        streams.human_pos,
        streams.neg,
    )
    online_success_pos: list[Any] = []
    online_success_stats: dict[str, Any] = {}
    online_success_replaced_policy_bc = 0
    policy_bc_for_neg = list(streams.policy_bc)
    if use_online_success:
        online_success_pos, next_episode_index, online_success_stats = (
            build_online_success_transitions(
                episodes_payload,
                action_horizon=action_horizon,
                episode_index_base=next_episode_index,
                route=ROUTE_POS_ONLY,
            )
        )
        source_indices = {
            int(index)
            for index in online_success_stats.get(
                "pure_success_source_episode_indices",
                [],
            )
        }
        if source_indices:
            policy_bc_for_neg = [
                transition
                for transition in streams.policy_bc
                if int((transition.info or {}).get("source_episode_index", -1))
                not in source_indices
            ]
            online_success_replaced_policy_bc = len(streams.policy_bc) - len(
                policy_bc_for_neg
            )
        print(
            f"[cfgrl] online_success={online_success_stats}; "
            f"replaced_policy_bc={online_success_replaced_policy_bc}"
        )
    else:
        print("[cfgrl] online_success disabled")

    pretrain_data_path = _resolve_required_path(
        OmegaConf.select(cfg, "offline.pretrain_data_path", default=None),
        what="offline.pretrain_data_path",
    )
    hdf5_loader = make_hdf5_loader(ctx, cfg)
    pretrain_pos, _ = load_pretrain_transitions(
        data_root=str(OmegaConf.select(cfg, "offline.data_root", default="data")),
        task_name=task_name,
        pretrain_dir=pretrain_data_path,
        hdf5_loader=hdf5_loader,
        max_num_trajectories=OmegaConf.select(
            cfg,
            "offline.max_pretrain_trajectories",
            default=None,
        ),
        episode_index_base=next_episode_index,
    )
    print(f"[cfgrl] pretrain_data={pretrain_data_path} transitions={len(pretrain_pos)}")

    try:
        ctx.env.close()
    except Exception:
        pass

    positive_transitions = (
        list(streams.human_pos) + list(online_success_pos) + list(pretrain_pos)
    )
    if not positive_transitions:
        raise RuntimeError(
            "No positive transitions found: human_pos + online_success + pretrain is empty."
        )
    negative_transitions = (
        list(policy_bc_for_neg)
        + list(streams.human_pos)
        + list(streams.neg)
        + list(online_success_pos)
        + list(pretrain_pos)
    )
    if not negative_transitions:
        raise RuntimeError("No negative-policy transitions found for CFGRL.")

    run_root = Path(
        to_absolute_path(
            str(
                OmegaConf.select(
                    cfg,
                    "offline.run_root",
                    default="./outputs/dipole-rl-offline",
                )
            )
        )
    )
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    postfix = str(OmegaConf.select(cfg, "offline.run_subfix", default="") or "").strip()
    dir_name = f"{task_name}_{timestamp}_{postfix}" if postfix else f"{task_name}_{timestamp}"
    run_dir = run_root / dir_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    run_name = f"{task_name}__offline_cfgrl_{mode}__{timestamp}"
    print(f"[cfgrl] run_dir={run_dir}")

    metric_logger = maybe_build_metric_logger(cfg, run_name=run_name, run_dir=run_dir)
    write_resolved_config(cfg, run_dir)

    encoder: SharedDynamicsEncoder | None = None
    discriminator: FrozenNNPUDiscriminator | None = None
    iql_learner: IQLLearner | None = None
    iql_cfg: IQLConfig | None = None
    nnpu_ckpt: str | None = None
    warmup_ckpt: str | None = None
    iql_ckpt_path: Path | None = None
    iql_buffer_stats: dict[str, int] = {}
    skip_rl = False
    if mode == "neg-weighted":
        (
            encoder,
            discriminator,
            iql_learner,
            iql_cfg,
            nnpu_ckpt,
            warmup_ckpt,
            iql_finetuned_override,
            skip_rl,
        ) = _build_weighted_neg_stack(
            cfg,
            agent=agent,
            task_name=task_name,
            camera_names=camera_names,
        )
        if skip_rl:
            source_iql = Path(str(iql_finetuned_override))
            iql_ckpt_path = run_dir / "checkpoints" / "iql_state_finetuned.pt"
            shutil.copy2(source_iql, iql_ckpt_path)
            print(f"[cfgrl][iql] skip_rl=true; copied {source_iql} -> {iql_ckpt_path}")
        else:
            iql_buffer, iql_buffer_stats = build_iql_finetune_buffer(
                streams.policy_bc,
                camera_names=camera_names,
                image_size=ctx.img_height,
                action_horizon=action_horizon,
                warmup_transitions_path=_warmup_transitions_path(cfg, task_name),
            )
            finetune_iql(
                iql_learner,
                iql_buffer,
                iql_cfg,
                encoder=encoder,
                discriminator=discriminator,
                num_steps=int(
                    OmegaConf.select(cfg, "offline.iql_finetune.num_steps", default=20000)
                ),
                batch_size=int(
                    OmegaConf.select(cfg, "offline.iql_finetune.batch_size", default=256)
                ),
                value_only_steps=int(
                    OmegaConf.select(
                        cfg,
                        "offline.iql_finetune.value_only_steps",
                        default=0,
                    )
                ),
                preencode=bool(
                    OmegaConf.select(
                        cfg,
                        "offline.iql_finetune.preencode_cache",
                        default=True,
                    )
                ),
                device=str(cfg.algorithm.q_learning.config.device),
                encode_batch_size=int(
                    OmegaConf.select(cfg, "offline.preencode_batch_size", default=64)
                ),
                metric_logger=metric_logger,
                log_interval=int(OmegaConf.select(cfg, "offline.log_interval", default=200)),
            )
            iql_ckpt_path = save_finetuned_iql(
                iql_learner,
                run_dir / "checkpoints" / "iql_state_finetuned.pt",
                encoder=encoder,
                discriminator=discriminator,
                iql_cfg=iql_cfg,
                nnpu_ckpt=nnpu_ckpt,
            )
            print(f"[cfgrl][iql] finetuned -> {iql_ckpt_path}")
        _freeze_iql(iql_learner)

    pos_valid = populate_replay_buffer(agent.online_buffer, positive_transitions)
    batch_size = finalize_normalizers(
        agent,
        cfg,
        positive_transitions,
        log_tag="cfgrl",
        norm_desc="human intervention + online success + expert pretrain",
    )
    pos_cache = agent.online_buffer.build_static_cache(pin_memory=True)
    print(
        f"[cfgrl] pos buffer transitions={len(agent.online_buffer)} "
        f"valid_windows={pos_valid}"
    )

    neg_buffer = DipoleReplayBuffer(
        agent.online_buffer.config,
        name="cfgrl_negative_buffer",
        camera_names=list(agent.online_buffer.camera_names),
        action_horizon=int(agent.online_buffer.action_horizon),
        image_size=int(agent.online_buffer.image_size),
        augmentation_config=agent.online_buffer.augmentation_config,
    )
    neg_valid = populate_replay_buffer(neg_buffer, negative_transitions)
    if neg_valid < batch_size:
        raise RuntimeError(
            f"negative replay buffer has only {neg_valid} valid windows "
            f"(< batch_size={batch_size})."
        )

    branch_policy = None
    g_provider = None
    advantage_estimator: str | None = None
    advantage_gae_lambda: float | None = None
    if mode == "neg-weighted":
        assert iql_learner is not None
        assert iql_cfg is not None
        assert encoder is not None
        assert discriminator is not None
        advantage_estimator = str(
            OmegaConf.select(cfg, "offline.advantage.estimator", default="gae")
        )
        advantage_gae_lambda = float(
            OmegaConf.select(cfg, "offline.advantage.gae_lambda", default=0.6)
        )
        advantage_raw, failure_raw, start_to_row = precompute_offline_advantage(
            base_buffer=neg_buffer,
            iql_learner=iql_learner,
            encoder=encoder,
            discriminator=discriminator,
            iql_cfg=iql_cfg,
            device=str(cfg.algorithm.q_learning.config.device),
            encode_batch_size=int(
                OmegaConf.select(cfg, "offline.preencode_batch_size", default=64)
            ),
            estimator=advantage_estimator,
            gae_lambda=advantage_gae_lambda,
        )
        g_provider = OfflineAdvantageGProvider(
            iql_learner=iql_learner,
            discriminator=discriminator,
            encoder=encoder,
            alpha=float(cfg.algorithm.advantage_g_provider.alpha),
            beta=float(cfg.algorithm.advantage_g_provider.beta),
            advantage_raw=advantage_raw,
            failure_raw=failure_raw,
            start_to_row=start_to_row,
        )
        g_provider.bind_policy_cameras(camera_names)
        agent.core.config.beta = float(
            OmegaConf.select(
                cfg,
                "offline.branch_weight.beta",
                default=agent.core.config.beta,
            )
        )
        agent.core.config.k = float(
            OmegaConf.select(
                cfg,
                "offline.branch_weight.k",
                default=agent.core.config.k,
            )
        )
        branch_policy = build_branch_weight_policy(
            OmegaConf.select(
                cfg,
                "offline.branch_weight",
                default=OmegaConf.create({"type": "routed_sigmoid"}),
            )
        )
        print(
            f"[cfgrl] neg-weighted estimator={advantage_estimator} "
            f"lambda={advantage_gae_lambda} beta={agent.core.config.beta} "
            f"k={agent.core.config.k}"
        )

    neg_cache = neg_buffer.build_static_cache(pin_memory=True)
    if mode == "neg-weighted" and set(neg_cache.start_indices.tolist()) != set(
        start_to_row.keys()
    ):
        raise RuntimeError(
            "negative static-cache indices differ from precomputed advantage indices."
        )
    print(
        f"[cfgrl] neg buffer transitions={len(neg_buffer)} "
        f"valid_windows={neg_valid}"
    )

    write_run_info(
        run_dir,
        {
            "run_name": run_name,
            "run_dir": str(run_dir),
            "started_at": timestamp,
            "task_name": task_name,
            "cfg_mode": mode,
            "algorithm_type": "offline_cfgrl_independent_branches",
            "initialized_checkpoint": str(ctx.init_checkpoint),
            "episodes_path": episodes_path,
            "pretrain_data_path": pretrain_data_path,
            "online_success_enabled": use_online_success,
            "online_success_stats": online_success_stats,
            "online_success_replaced_policy_bc_transitions": online_success_replaced_policy_bc,
            "positive_transitions": len(positive_transitions),
            "negative_transitions": len(negative_transitions),
            "positive_valid_windows": int(pos_valid),
            "negative_valid_windows": int(neg_valid),
            "policy_batch_size_per_branch": int(batch_size),
            "nnpu_checkpoint": nnpu_ckpt,
            "iql_warmup_checkpoint": warmup_ckpt,
            "iql_finetuned_checkpoint": None if iql_ckpt_path is None else str(iql_ckpt_path),
            "skip_rl": bool(skip_rl),
            "advantage_estimator": advantage_estimator,
            "advantage_gae_lambda": advantage_gae_lambda,
            "iql_buffer_stats": iql_buffer_stats,
            "stream_stats": streams.stats,
        },
    )

    num_steps = int(OmegaConf.select(cfg, "offline.num_train_steps", default=15000))
    log_interval = max(1, int(OmegaConf.select(cfg, "offline.log_interval", default=200)))
    checkpoint_interval = max(
        1,
        int(OmegaConf.select(cfg, "offline.checkpoint_interval", default=5000)),
    )
    sample_kwargs = {
        "action_mean": agent.core.act_mean,
        "action_std": agent.core.act_std,
        "proprio_mean": agent.core.prop_mean,
        "proprio_std": agent.core.prop_std,
        "device": agent.core.device,
        "augment": True,
    }

    def _save(tag: str, step: int) -> Path:
        payload = agent.build_checkpoint_payload(
            include_buffers=False,
            extra={"global_step": int(step), "run_name": run_name, "cfg_mode": mode},
        )
        path = checkpoint_path(run_dir, tag)
        agent.write_checkpoint_payload(path, payload)
        return path

    print(
        f"[cfgrl] training {num_steps} steps; each step uses "
        f"pos_batch={batch_size} and neg_batch={batch_size}"
    )
    if num_steps <= 0:
        raise ValueError(f"offline.num_train_steps must be positive, got {num_steps}.")
    last_completed_step = -1
    try:
        for step in range(num_steps):
            # Sample in a fixed order on the main thread. This propagates CUDA
            # errors immediately and keeps seeded RNG consumption reproducible.
            pos_batch = pos_cache.sample(batch_size, **sample_kwargs)
            neg_batch = neg_cache.sample(batch_size, **sample_kwargs)
            is_log = (step % log_interval == 0) or (step == num_steps - 1)

            pos_weights = torch.ones(
                pos_batch.batch_size,
                dtype=torch.float32,
                device=agent.core.device,
            )
            pos_metrics = _branch_only_update(
                agent.core,
                pos_batch,
                branch="pos",
                weights=pos_weights,
                want_metrics=is_log,
            )
            neg_weights, weight_metrics = _compute_cfgrl_neg_weights(
                agent.core,
                neg_batch,
                mode=mode,
                branch_policy=branch_policy,
                g_provider=g_provider,
                want_metrics=is_log,
            )
            neg_metrics = _branch_only_update(
                agent.core,
                neg_batch,
                branch="neg",
                weights=neg_weights,
                want_metrics=is_log,
            )
            last_completed_step = step

            if is_log:
                metrics = {
                    "loss_pos": pos_metrics["loss_pos"],
                    "loss_neg": neg_metrics["loss_neg"],
                    "actor_loss": pos_metrics["loss_pos"] + neg_metrics["loss_neg"],
                    "flow_loss_pos": pos_metrics["flow_loss_pos"],
                    "flow_loss_neg": neg_metrics["flow_loss_neg"],
                    "endpoint_loss_pos": pos_metrics["endpoint_loss_pos"],
                    "endpoint_loss_neg": neg_metrics["endpoint_loss_neg"],
                    "smooth_loss_pos": pos_metrics["smooth_loss_pos"],
                    "smooth_loss_neg": neg_metrics["smooth_loss_neg"],
                    "grad_norm_pos": pos_metrics["grad_norm_pos"],
                    "grad_norm_neg": neg_metrics["grad_norm_neg"],
                    "w_pos_mean": pos_metrics["w_pos_mean"],
                    "w_neg_mean": neg_metrics["w_neg_mean"],
                    "frac_intervention_pos": pos_metrics["frac_intervention"],
                    "frac_intervention_neg": neg_metrics["frac_intervention"],
                    **weight_metrics,
                }
                maybe_log(
                    metric_logger,
                    {f"train/{key}": float(value) for key, value in metrics.items()},
                    step=step,
                )
                print(
                    f"[cfgrl][step {step:6d}] pos={metrics['loss_pos']:.4f} "
                    f"neg={metrics['loss_neg']:.4f} "
                    f"w_neg={metrics['w_neg_mean']:.3f} mode={mode}"
                )
            if step > 0 and step % checkpoint_interval == 0:
                step_path = _save(f"step_{step:08d}", step)
                _save("latest", step)
                print(f"[cfgrl][ckpt] step={step} -> {step_path.name} (+latest)")
    except Exception:
        print(
            f"[cfgrl] training failed after completed_step={last_completed_step}; "
            "no final checkpoint was written."
        )
        raise
    else:
        final_path = _save("latest", last_completed_step)
        _save(f"step_{last_completed_step:08d}", last_completed_step)
        print(f"[cfgrl] done. final checkpoint -> {final_path}")
    finally:
        if metric_logger is not None:
            metric_logger.close()


if __name__ == "__main__":
    main()
