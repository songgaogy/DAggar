from __future__ import annotations

import datetime
import logging
import queue
import threading
import warnings
from pathlib import Path
from typing import Any, Callable

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from robosuite.pipeline.algorithms.discriminator.encoder import SharedDynamicsEncoder
from robosuite.pipeline.algorithms.discriminator.nnpu import FrozenNNPUDiscriminator
from robosuite.pipeline.algorithms.vast.common import VASTConfig
from robosuite.pipeline.algorithms.vast.vast import VASTLearner
from robosuite.pipeline.offline.src.diagnostic_plots import (
    plot_branch_weight_distribution,
    plot_raw_g_distribution,
    plot_v_pos_neg_scatter,
)
from robosuite.pipeline.offline.utils import (
    OfflineAdvantageGProvider,
    build_agent_env,
    build_branch_weight_policy,
    build_vast_finetune_buffer,
    build_offline_transitions,
    build_online_success_transitions,
    finalize_normalizers,
    finetune_vast,
    load_pretrain_transitions,
    make_hdf5_loader,
    populate_replay_buffer,
    precompute_offline_advantage,
    save_finetuned_vast,
    validate_vast_checkpoint_payload,
)
from robosuite.pipeline.offline.utils.episode_dataset import ROUTE_POS_ONLY
from robosuite.pipeline.train_dipole_rl import _load_vast_warmup_state
from robosuite.pipeline.utils import (
    checkpoint_path,
    maybe_build_metric_logger,
    maybe_log,
    maybe_log_figure,
    write_resolved_config,
    write_run_info,
)

logger = logging.getLogger(__name__)


class _BatchPrefetcher:
    """Background double-buffer so step N+1's batch is prepared while step N trains."""

    def __init__(
        self,
        sample_fn: Callable[..., Any],
        *,
        batch_size: int,
        sample_kwargs: dict[str, Any],
        device: torch.device,
        depth: int = 2,
    ) -> None:
        self._sample_fn = sample_fn
        self._batch_size = int(batch_size)
        self._sample_kwargs = sample_kwargs
        self._device = device
        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize=max(1, int(depth)))
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="dipole_offline_prefetch", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            batch = self._sample_fn(self._batch_size, **self._sample_kwargs)
            while not self._stop.is_set():
                try:
                    self._queue.put(batch, timeout=0.5)
                    break
                except queue.Full:
                    continue

    def next(self) -> Any:
        return self._queue.get()

    def close(self) -> None:
        self._stop.set()
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
        self._thread.join(timeout=2.0)


def _freeze_vast(vast: VASTLearner) -> None:
    """Disable grads on all value-stitching modules for Phase B."""
    modules = [vast.v, vast.target_v]
    if getattr(vast, "g", None) is not None:
        modules.append(vast.g)
    for module in modules:
        for param in module.parameters():
            param.requires_grad_(False)


def _resolve_required_path(raw: Any, *, what: str) -> str:
    if raw is None or str(raw).strip().lower() in ("", "null"):
        raise RuntimeError(f"Offline DIPOLE requires {what} to be set.")
    path = to_absolute_path(str(raw))
    if not Path(path).exists():
        raise FileNotFoundError(f"{what} not found: {path}")
    return path


def _resolve_episodes_path(cfg: DictConfig, task_name: str) -> str:
    raw = OmegaConf.select(cfg, "offline.episodes_path", default=None)
    if raw is not None and str(raw).strip().lower() not in ("", "null"):
        return _resolve_required_path(raw, what="offline.episodes_path")
    data_root = str(cfg.offline.data_root)
    default = Path(data_root) / task_name / "offline_data" / "offline_episodes.pt"
    if not default.exists():
        raise FileNotFoundError(
            f"offline_episodes.pt not found at {default}; set offline.episodes_path "
            "or run offline/scripts/collect_data.sh first."
        )
    return str(default)


def _warmup_transitions_path(cfg: DictConfig, task_name: str) -> str:
    data_root = str(cfg.offline.data_root)
    sub = str(OmegaConf.select(cfg, "offline.vast_warmup_transitions_dir", default="offline_data-vast"))
    base = Path(data_root) / task_name
    canonical = base / sub / "vast_offline_transitions.pt"
    if canonical.exists():
        return str(canonical)
    legacy_candidates = (
        base / sub / "iql_offline_transitions.pt",
        base / "offline_data-iql" / "iql_offline_transitions.pt",
        base / "offline_data" / "iql_offline_transitions.pt",
    )
    for legacy in legacy_candidates:
        if legacy.exists():
            warnings.warn(
                f"Loading deprecated warmup buffer {legacy}; use {canonical}.",
                FutureWarning,
                stacklevel=2,
            )
            return str(legacy)
    return str(canonical)


def _next_episode_index(*transition_groups: list[Any]) -> int:
    hi = -1
    for transitions in transition_groups:
        for transition in transitions:
            info = transition.info or {}
            hi = max(hi, int(info.get("episode_index", -1)))
    return hi + 1


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    return text in ("1", "true", "yes", "y", "on")


def _resolve_vast_reward_semantics(
    vast_cfg: VASTConfig,
    checkpoint_cfg: dict[str, Any],
    *,
    skip_rl: bool,
    relabel_disc_reward: bool,
) -> dict[str, Any]:
    """Prepare strict checkpoint loading and record an explicit Phase-A relabel."""
    requested_disc_coef = float(vast_cfg.disc_reward_coef)
    checkpoint_disc_coef = float(
        checkpoint_cfg.get("disc_reward_coef", requested_disc_coef)
    )
    disc_mismatch = requested_disc_coef != checkpoint_disc_coef
    if disc_mismatch and skip_rl:
        raise ValueError(
            "offline.skip_rl=true requires disc_reward_coef to match the loaded "
            f"VAST checkpoint; checkpoint={checkpoint_disc_coef}, "
            f"runtime={requested_disc_coef}. Phase A reward relabeling cannot be skipped."
        )
    if disc_mismatch and not relabel_disc_reward:
        raise ValueError(
            "VAST checkpoint disc_reward_coef mismatch; checkpoint="
            f"{checkpoint_disc_coef}, runtime={requested_disc_coef}. Set "
            "offline.vast_finetune.relabel_disc_reward=true to use the checkpoint "
            "only as Phase A initialization."
        )

    # Only the explicitly approved discriminator-reward mismatch is temporarily
    # aligned for strict loading. Every other semantic field remains subject to
    # the existing checkpoint validator.
    if disc_mismatch:
        print(
            "[offline][vast] aligning disc_reward_coef: "
            f"{requested_disc_coef} -> {checkpoint_disc_coef} (for checkpoint load)"
        )
        vast_cfg.disc_reward_coef = type(vast_cfg.disc_reward_coef)(
            checkpoint_disc_coef
        )

    relabel_enabled = bool(relabel_disc_reward and not skip_rl)
    return {
        "enabled": relabel_enabled,
        "changed": bool(relabel_enabled and disc_mismatch),
        "checkpoint_disc_reward_coef": checkpoint_disc_coef,
        "requested_disc_reward_coef": requested_disc_coef,
        "effective_disc_reward_coef": (
            requested_disc_coef if relabel_enabled else checkpoint_disc_coef
        ),
    }


def _apply_vast_disc_reward_relabel(
    vast_cfg: VASTConfig,
    provenance: dict[str, Any],
) -> None:
    """Restore the requested discriminator reward after strict checkpoint loading."""
    if not bool(provenance["enabled"]):
        return
    restored = type(vast_cfg.disc_reward_coef)(
        provenance["requested_disc_reward_coef"]
    )
    if vast_cfg.disc_reward_coef != restored:
        print(
            "[offline][vast] relabeling Phase-A discriminator reward: "
            f"{vast_cfg.disc_reward_coef} -> {restored}"
        )
    vast_cfg.disc_reward_coef = restored


def _resolve_advantage_config(cfg: DictConfig) -> tuple[str, float]:
    estimator = str(
        OmegaConf.select(cfg, "offline.advantage.estimator", default="gae")
    ).lower()
    if estimator not in {"gae", "td1"}:
        raise ValueError(
            "offline.advantage.estimator must be 'gae' or 'td1', "
            f"got {estimator!r}."
        )
    gae_lambda = float(
        OmegaConf.select(cfg, "offline.advantage.gae_lambda", default=0.6)
    )
    return estimator, gae_lambda


def _precompute_phase_b_advantage(
    *,
    base_buffer: Any,
    vast_learner: VASTLearner,
    encoder: SharedDynamicsEncoder,
    discriminator: FrozenNNPUDiscriminator,
    vast_cfg: VASTConfig,
    device: str,
    encode_batch_size: int,
    estimator: str,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[int, int]]:
    """Dispatch Phase B to VAST-value TD1/GAE, never stitched advantage."""
    return precompute_offline_advantage(
        base_buffer=base_buffer,
        vast_learner=vast_learner,
        encoder=encoder,
        discriminator=discriminator,
        vast_cfg=vast_cfg,
        device=device,
        encode_batch_size=encode_batch_size,
        estimator=estimator,
        gae_lambda=gae_lambda,
    )


@hydra.main(version_base="1.2", config_path="../../config", config_name="train_offline_dipole")
def main(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision("high")
    adv_estimator, adv_gae_lambda = _resolve_advantage_config(cfg)

    # ------------------------------------------------------------------ #
    # Shared setup: env (headless) + DIPOLE agent + frozen encoder/disc. #
    # ------------------------------------------------------------------ #
    ctx = build_agent_env(cfg, log_tag="offline")
    agent = ctx.agent
    task_name = ctx.task_name
    img_height = ctx.img_height
    camera_names = list(agent.camera_names)

    rl_device = str(cfg.algorithm.vast.config.device)
    if not rl_device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError(
            f"Offline VAST requires CUDA tensor execution; requested device={rl_device!r}, "
            f"torch.cuda.is_available()={torch.cuda.is_available()}."
        )
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
        str(k): str(v)
        for k, v in dict(getattr(cfg.algorithm.discriminator, "camera_to_view", {}) or {}).items()
    }
    shared_encoder = SharedDynamicsEncoder(
        nnpu_ckpt_path=nnpu_ckpt,
        encoder_ckpt=encoder_ckpt,
        device=rl_device,
        camera_to_view=camera_to_view,
    )
    shared_encoder.bind_policy_cameras(camera_names)
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

    # VAST config + reward-coefficient alignment with the loaded checkpoint so the
    # value scale stays consistent through finetuning or skipped-rl reuse.
    policy_action_dim = int(agent.flow_config.action_dim)
    vast_cfg_dict = OmegaConf.to_container(cfg.algorithm.vast.config, resolve=True)
    if not isinstance(vast_cfg_dict, dict):
        raise TypeError("algorithm.vast.config must resolve to a mapping.")
    vast_cfg = VASTConfig(**vast_cfg_dict)
    skip_rl = _as_bool(OmegaConf.select(cfg, "offline.skip_rl", default=False))
    warmup_ckpt: str | None = None
    vast_finetuned_override: str | None = None
    if skip_rl:
        vast_finetuned_override = _resolve_required_path(
            OmegaConf.select(cfg, "offline.vast_finetuned_path", default=None),
            what="offline.vast_finetuned_path (finetuned VAST state)",
        )
        initial_vast_ckpt = vast_finetuned_override
    else:
        warmup_ckpt = _resolve_required_path(
            cfg.algorithm.vast.warmup_ckpt,
            what="algorithm.vast.warmup_ckpt (pretrained VAST state)",
        )
        initial_vast_ckpt = warmup_ckpt
    warmup_payload = torch.load(initial_vast_ckpt, map_location="cpu", weights_only=False)
    warmup_cfg = warmup_payload.get("cfg", {}) if isinstance(warmup_payload, dict) else {}
    relabel_disc_reward = _as_bool(
        OmegaConf.select(
            cfg,
            "offline.vast_finetune.relabel_disc_reward",
            default=False,
        )
    )
    reward_relabel_provenance = _resolve_vast_reward_semantics(
        vast_cfg,
        warmup_cfg if isinstance(warmup_cfg, dict) else {},
        skip_rl=skip_rl,
        relabel_disc_reward=relabel_disc_reward,
    )
    validate_vast_checkpoint_payload(
        warmup_payload,
        vast_cfg,
        require_finetuned=bool(skip_rl),
    )

    H = int(vast_cfg.action_horizon)
    if int(agent.flow_config.action_horizon) != H:
        raise RuntimeError(
            f"action_horizon mismatch: policy={agent.flow_config.action_horizon} vs VAST={H}."
        )

    vast_learner = VASTLearner(
        vast_cfg,
        state_feature_dim=int(shared_encoder.state_feature_dim),
        chunk_feature_dim=int(shared_encoder.chunk_feature_dim),
        action_dim=policy_action_dim,
        n_tokens=int(shared_encoder.inner_encoder.num_patches),
        proprio_dim=int(shared_encoder.inner_encoder.proprio_emb_dim),
    )
    _load_vast_warmup_state(
        vast_learner,
        initial_vast_ckpt,
        expected_state_feature_dim=int(shared_encoder.state_feature_dim),
        expected_chunk_feature_dim=int(shared_encoder.chunk_feature_dim),
        expected_action_dim=policy_action_dim,
    )
    _apply_vast_disc_reward_relabel(vast_cfg, reward_relabel_provenance)
    print(f"[offline][vast] loaded VAST critics from {initial_vast_ckpt}")

    # ------------------------------------------------------------------ #
    # Load + split collected episodes into the three routed streams.     #
    # ------------------------------------------------------------------ #
    episodes_path = _resolve_episodes_path(cfg, task_name)
    payload = torch.load(episodes_path, map_location="cpu", weights_only=False)
    payload_cameras = [str(c) for c in payload.get("camera_names", [])]
    if payload_cameras and set(payload_cameras) != set(camera_names):
        raise RuntimeError(
            f"camera mismatch: episodes {payload_cameras} vs policy {camera_names}. "
            "Collect data with the same init_checkpoint."
        )
    streams = build_offline_transitions(
        payload,
        action_horizon=H,
        reward_success=float(OmegaConf.select(cfg, "offline.reward_success", default=0.0)),
        reward_fail=float(OmegaConf.select(cfg, "offline.reward_fail", default=-1.0)),
        include_policy_action_neg=bool(
            OmegaConf.select(cfg, "offline.include_policy_action_neg", default=True)
        ),
    )
    if not streams.policy_bc:
        raise RuntimeError(
            "No policy sections survived filtering; cannot finetune VAST or precompute advantage."
        )
    print(f"[offline] episodes={episodes_path}")
    print(f"[offline] streams: {streams.stats}")

    use_online_success = bool(OmegaConf.select(cfg, "offline.use_online_success", default=False))
    next_ep_base = _next_episode_index(streams.policy_bc, streams.human_pos, streams.neg)
    online_success_pos: list[Any] = []
    online_success_stats: dict[str, Any] = {}
    online_success_replaced_policy_bc = 0
    policy_bc_for_policy = list(streams.policy_bc)
    if use_online_success:
        online_success_pos, next_ep_base, online_success_stats = build_online_success_transitions(
            payload,
            action_horizon=H,
            episode_index_base=next_ep_base,
            route=ROUTE_POS_ONLY,
        )
        source_episode_indices = {
            int(idx)
            for idx in online_success_stats.get("pure_success_source_episode_indices", [])
        }
        if source_episode_indices:
            policy_bc_for_policy = [
                transition
                for transition in streams.policy_bc
                if int((transition.info or {}).get("source_episode_index", -1))
                not in source_episode_indices
            ]
            online_success_replaced_policy_bc = len(streams.policy_bc) - len(policy_bc_for_policy)
        print(
            f"[offline] online_success: {online_success_stats}; "
            f"replaced_policy_bc={online_success_replaced_policy_bc}"
        )
    else:
        print("[offline] online_success: disabled (offline.use_online_success=false)")

    pretrain_pos = []
    pretrain_data_path = None
    raw_pretrain = OmegaConf.select(cfg, "offline.pretrain_data_path", default=None)
    if raw_pretrain is not None and str(raw_pretrain).strip().lower() not in ("", "null"):
        pretrain_data_path = _resolve_required_path(raw_pretrain, what="offline.pretrain_data_path")
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
            episode_index_base=next_ep_base,
        )
        print(f"[offline] pretrain_data={pretrain_data_path} transitions={len(pretrain_pos)}")

    try:
        ctx.env.close()
    except Exception:  # pragma: no cover - env cleanup is best-effort
        pass

    # ------------------------------------------------------------------ #
    # Run directory + logging.                                           #
    # ------------------------------------------------------------------ #
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    explicit_run_dir = OmegaConf.select(cfg, "offline.run_dir", default=None)
    if explicit_run_dir is None or str(explicit_run_dir).strip().lower() in {
        "",
        "none",
        "null",
    }:
        run_root = Path(
            to_absolute_path(
                str(
                    OmegaConf.select(
                        cfg,
                        "offline.run_root",
                        default="./outputs/dipole-rl-offline_vast",
                    )
                )
            )
        ).resolve()
        postfix = str(
            OmegaConf.select(cfg, "offline.run_subfix", default="") or ""
        ).strip()
        dir_name = (
            f"{task_name}_{timestamp}_{postfix}"
            if postfix
            else f"{task_name}_{timestamp}"
        )
        pipeline_run_dir = run_root / dir_name
        run_dir = pipeline_run_dir / "dipole"
        if pipeline_run_dir.exists():
            raise FileExistsError(
                f"Offline DIPOLE pipeline directory already exists: {pipeline_run_dir}"
            )
    else:
        run_dir = Path(to_absolute_path(str(explicit_run_dir))).resolve()
        pipeline_run_dir = run_dir.parent
    if run_dir.exists():
        raise FileExistsError(f"Offline DIPOLE stage directory already exists: {run_dir}")
    (run_dir / "checkpoints").mkdir(parents=True)
    run_name = f"{task_name}__offline_dipole__{timestamp}"
    print(f"[offline] pipeline_run_dir={pipeline_run_dir}")
    print(f"[offline] stage_run_dir={run_dir}")

    metric_logger = maybe_build_metric_logger(cfg, run_name=run_name, run_dir=run_dir)
    write_resolved_config(cfg, run_dir)

    # ------------------------------------------------------------------ #
    # Phase A: VAST finetune (unfrozen).                                  #
    # ------------------------------------------------------------------ #
    warmup_transitions_path = _warmup_transitions_path(cfg, task_name)
    vast_buffer_stats: dict[str, int] = {}
    if skip_rl:
        vast_ckpt_path = run_dir / "checkpoints" / "vast_state_finetuned.pt"
        save_finetuned_vast(
            vast_learner,
            vast_ckpt_path,
            encoder=shared_encoder,
            discriminator=discriminator,
            vast_cfg=vast_cfg,
            nnpu_ckpt=nnpu_ckpt,
            extra_meta={
                "migrated_from_checkpoint": str(vast_finetuned_override),
                "discriminator_checkpoint": nnpu_ckpt,
                "disc_reward_relabel": reward_relabel_provenance,
            },
        )
        normalized_schema = 8 if vast_cfg.vast_v_mode == "indep_ensemble" else 7
        print(
            "[offline][vast] skip_rl=true; normalized finetuned VAST checkpoint "
            f"to schema v{normalized_schema} at {vast_ckpt_path}"
        )
    else:
        vast_buffer, vast_buffer_stats = build_vast_finetune_buffer(
            streams.policy_bc,
            camera_names=camera_names,
            image_size=img_height,
            action_horizon=H,
            warmup_transitions_path=warmup_transitions_path,
            relabel_disc_reward=bool(reward_relabel_provenance["enabled"]),
        )
        finetune_vast(
            vast_learner,
            vast_buffer,
            vast_cfg,
            encoder=shared_encoder,
            discriminator=discriminator,
            num_steps=int(OmegaConf.select(cfg, "offline.vast_finetune.num_steps", default=20000)),
            batch_size=int(OmegaConf.select(cfg, "offline.vast_finetune.batch_size", default=256)),
            preencode=bool(OmegaConf.select(cfg, "offline.vast_finetune.preencode_cache", default=True)),
            device=rl_device,
            encode_batch_size=int(OmegaConf.select(cfg, "offline.preencode_batch_size", default=64)),
            metric_logger=metric_logger,
            log_interval=int(cfg.offline.log_interval),
        )
        vast_ckpt_path = save_finetuned_vast(
            vast_learner,
            run_dir / "checkpoints" / "vast_state_finetuned.pt",
            encoder=shared_encoder,
            discriminator=discriminator,
            vast_cfg=vast_cfg,
            nnpu_ckpt=nnpu_ckpt,
            extra_meta={
                "discriminator_checkpoint": nnpu_ckpt,
                "disc_reward_relabel": reward_relabel_provenance,
            },
        )
        print(f"[offline][vast] finetuned VAST -> {vast_ckpt_path}")

    # Phase A.5: freeze the finetuned critics for the policy update.
    _freeze_vast(vast_learner)

    # ------------------------------------------------------------------ #
    # Phase B: weighted-BC policy update (frozen finetuned VAST).         #
    # ------------------------------------------------------------------ #
    all_transitions = (
        list(policy_bc_for_policy)
        + list(streams.human_pos)
        + list(streams.neg)
        + list(online_success_pos)
        + list(pretrain_pos)
    )
    n_valid = populate_replay_buffer(agent.online_buffer, all_transitions)
    print(
        f"[offline] policy-BC buffer: {len(agent.online_buffer)} transitions, {n_valid} valid windows "
        f"(policy_bc={len(policy_bc_for_policy)}, human_pos={len(streams.human_pos)}, "
        f"neg={len(streams.neg)}, online_success_pos={len(online_success_pos)}, "
        f"pretrain_pos={len(pretrain_pos)})"
    )
    batch_size = finalize_normalizers(
        agent,
        cfg,
        list(policy_bc_for_policy) + list(online_success_pos) + list(pretrain_pos),
        log_tag="offline",
        norm_desc="policy sections + online-success + pretrain positive demos",
    )

    advantage_raw, failure_raw, start_to_row = _precompute_phase_b_advantage(
        base_buffer=agent.online_buffer,
        vast_learner=vast_learner,
        encoder=shared_encoder,
        discriminator=discriminator,
        vast_cfg=vast_cfg,
        device=rl_device,
        encode_batch_size=int(OmegaConf.select(cfg, "offline.preencode_batch_size", default=64)),
        estimator=adv_estimator,
        gae_lambda=adv_gae_lambda,
    )
    gamma_h = float(vast_cfg.discount) ** H
    advantage_mean = float(advantage_raw.mean().item())
    advantage_std = (
        float(advantage_raw.std().item()) if advantage_raw.numel() > 1 else 0.0
    )
    failure_mean = float(failure_raw.mean().item())
    print(
        f"[offline] advantage estimator={adv_estimator} gamma^H={gamma_h:.6f}"
        + (f" lambda={adv_gae_lambda}" if adv_estimator == "gae" else "")
        + f" mean={advantage_mean:+.4f} std={advantage_std:.4f}"
    )
    maybe_log(
        metric_logger,
        {
            "offline_advantage/is_gae": float(adv_estimator == "gae"),
            "offline_advantage/mean": advantage_mean,
            "offline_advantage/std": advantage_std,
            "offline_advantage/failure_mean": failure_mean,
            "offline_advantage/gamma_h": gamma_h,
            "offline_advantage/gae_lambda": (
                adv_gae_lambda if adv_estimator == "gae" else 0.0
            ),
        },
        step=0,
    )
    provider = OfflineAdvantageGProvider(
        vast_learner=vast_learner,
        discriminator=discriminator,
        encoder=shared_encoder,
        alpha=float(cfg.algorithm.advantage_g_provider.alpha),
        beta=float(cfg.algorithm.advantage_g_provider.beta),
        advantage_raw=advantage_raw,
        failure_raw=failure_raw,
        start_to_row=start_to_row,
    )
    provider.bind_policy_cameras(camera_names)
    agent.attach_vast_learner(vast_learner)
    agent.attach_discriminator(discriminator)
    agent.attach_g_provider(provider)

    # Pluggable branch-weight policy: w_pos = sigmoid(beta * (G + k)).
    agent.core.config.beta = float(OmegaConf.select(cfg, "offline.branch_weight.beta", default=agent.core.config.beta))
    agent.core.config.k = float(OmegaConf.select(cfg, "offline.branch_weight.k", default=agent.core.config.k))
    branch_policy = build_branch_weight_policy(
        OmegaConf.select(cfg, "offline.branch_weight", default=OmegaConf.create({"type": "routed_sigmoid"}))
    )
    agent.core.set_branch_weight_policy(branch_policy)
    print(
        f"[offline] branch_weight={type(branch_policy).__name__} beta={agent.core.config.beta} "
        f"k={agent.core.config.k}; provider alpha={cfg.algorithm.advantage_g_provider.alpha} "
        f"beta={cfg.algorithm.advantage_g_provider.beta}"
    )

    static_cache = agent.online_buffer.build_static_cache(pin_memory=True)
    if set(static_cache.start_indices.tolist()) != set(start_to_row.keys()):
        raise RuntimeError(
            "static_cache start indices != precomputed advantage indices; the buffer "
            "must stay static between precompute and cache build."
        )
    cache_mib = float(static_cache.estimated_bytes) / (1024.0 * 1024.0)
    print(f"[offline] static_cache rows={len(static_cache)} estimated={cache_mib:.1f} MiB")

    write_run_info(
        run_dir,
        {
            "run_name": run_name,
            "run_dir": str(run_dir),
            "pipeline_run_dir": str(pipeline_run_dir),
            "stage_run_dir": str(run_dir),
            "started_at": timestamp,
            "task_name": task_name,
            "policy_camera_names": camera_names,
            "initialized_checkpoint": str(ctx.init_checkpoint),
            "nnpu_checkpoint": nnpu_ckpt,
            "discriminator_checkpoint": nnpu_ckpt,
            "discriminator_threshold": float(discriminator.threshold),
            "disc_reward_coef": float(vast_cfg.disc_reward_coef),
            "disc_reward_relabel": reward_relabel_provenance,
            "vast_warmup_checkpoint": warmup_ckpt,
            "vast_finetuned_checkpoint": str(vast_ckpt_path),
            "episodes_path": episodes_path,
            "pretrain_data_path": pretrain_data_path,
            "pretrain_transitions": len(pretrain_pos),
            "online_success_enabled": use_online_success,
            "online_success_transitions": len(online_success_pos),
            "online_success_replaced_policy_bc_transitions": online_success_replaced_policy_bc,
            "online_success_stats": online_success_stats,
            "skip_rl": bool(skip_rl),
            "warmup_transitions_path": warmup_transitions_path,
            "algorithm": "vast_value_stitching_adaptation",
            "g_mode": f"advantage_offline_{adv_estimator}",
            "advantage_estimator": adv_estimator,
            "advantage_gae_lambda": (
                adv_gae_lambda if adv_estimator == "gae" else None
            ),
            "advantage_gamma_h": gamma_h,
            "advantage_mean": advantage_mean,
            "advantage_std": advantage_std,
            "failure_mean": failure_mean,
            "vast_v_mode": str(vast_cfg.vast_v_mode),
            "vast_max_k": int(vast_cfg.vast_max_k),
            "vast_macro_horizon": int(H),
            "vast_sampling_seed": int(vast_cfg.vast_sampling_seed),
            "vast_comp_coef": float(vast_cfg.vast_comp_coef),
            "stream_stats": streams.stats,
            "vast_buffer_stats": vast_buffer_stats,
            "replay_valid_windows": int(n_valid),
        },
    )

    # ------------------------------------------------------------------ #
    # Policy training loop.                                              #
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

    def _save(tag: str, step: int) -> Path:
        payload = agent.build_checkpoint_payload(
            include_buffers=False,
            extra={"global_step": int(step), "run_name": run_name},
        )
        path = checkpoint_path(run_dir, tag)
        agent.write_checkpoint_payload(path, payload)
        return path

    prefetcher = _BatchPrefetcher(
        static_cache.sample,
        batch_size=batch_size,
        sample_kwargs=sample_kwargs,
        device=agent.core.device,
    )
    print(f"[offline] policy training for {num_steps} steps (batch_size={batch_size}, prefetch=on)")
    try:
        for step in range(num_steps):
            batch = prefetcher.next()
            is_log = (step % log_interval == 0) or (step == num_steps - 1)
            collect_diag = (step % plot_log_interval == 0) or (step == num_steps - 1)
            metrics = agent.core.update(
                batch=batch, want_metrics=is_log, collect_diagnostics=collect_diag
            )
            diag = metrics.pop("_diag", None)
            if is_log:
                maybe_log(
                    metric_logger,
                    {f"train/{key}": float(value) for key, value in metrics.items()},
                    step=step,
                )
                print(
                    f"[offline][step {step:6d}] loss={metrics.get('actor_loss', 0.0):.4f} "
                    f"w_pos={metrics.get('w_pos_mean', 0.0):.3f} "
                    f"w_neg={metrics.get('w_neg_mean', 0.0):.3f} "
                    f"G_mean={metrics.get('G_mean', 0.0):+.3f} "
                    f"frac_adv={metrics.get('frac_advantage', 0.0):.2f}"
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
        prefetcher.close()
        final_step = max(0, num_steps - 1)
        final_path = _save("latest", final_step)
        _save(f"step_{final_step:08d}", final_step)
        if metric_logger is not None:
            metric_logger.close()
        print(f"[offline] done. final checkpoint -> {final_path}")


if __name__ == "__main__":
    main()
