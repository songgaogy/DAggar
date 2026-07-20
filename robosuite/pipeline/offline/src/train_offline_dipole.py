"""CUDA-only discriminator-weighted Offline DIPOLE training."""

from __future__ import annotations

import datetime
import logging
import queue
import threading
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from robosuite.pipeline.algorithms.discriminator.encoder import SharedDynamicsEncoder
from robosuite.pipeline.algorithms.discriminator.nnpu import FrozenNNPUDiscriminator
from robosuite.pipeline.algorithms.dipole.replay_buffer import DipoleReplayBuffer
from robosuite.pipeline.offline.src.diagnostic_plots import (
    plot_branch_weight_distribution,
    plot_g_distribution,
    plot_v_pos_neg_scatter,
)
from robosuite.pipeline.offline.utils import (
    OfflineDiscriminatorGProvider,
    POLICY_TRAINING_COUPLED,
    POLICY_TRAINING_FILTERED_BC,
    build_agent_env,
    build_branch_weight_policy,
    build_offline_transitions,
    build_online_success_transitions,
    branch_only_update,
    branch_seed,
    directory_input_provenance,
    file_provenance,
    git_provenance,
    load_pretrain_transitions,
    make_hdf5_loader,
    normalize_policy_training_mode,
    parameter_distance_metrics,
    populate_replay_buffer,
    precompute_discriminator_scores,
    sample_static_cache,
    trainable_parameter_snapshot,
)
from robosuite.pipeline.offline.utils.episode_dataset import (
    ROUTE_DISC_WEIGHTED,
    ROUTE_NEG_ONLY,
    ROUTE_POS_ONLY,
)
from robosuite.pipeline.utils import (
    checkpoint_path,
    maybe_build_metric_logger,
    maybe_log,
    maybe_log_figure,
    write_resolved_config,
    write_run_info,
)


logger = logging.getLogger(__name__)


def _require_pretrained_normalizers(agent: Any) -> None:
    if not agent.has_normalizers():
        raise RuntimeError(
            "The positive-policy matrix requires normalizers from the shared "
            "initial flow checkpoint; condition-specific fitting is forbidden."
        )
    arrays = {
        "act_mean": agent.core.act_mean,
        "act_std": agent.core.act_std,
        "prop_mean": agent.core.prop_mean,
        "prop_std": agent.core.prop_std,
    }
    for name, value in arrays.items():
        array = np.asarray(value)
        if not np.isfinite(array).all():
            raise RuntimeError(f"Initial checkpoint normalizer {name} is not finite.")
    if bool((np.asarray(agent.core.act_std) <= 0).any()) or bool(
        (np.asarray(agent.core.prop_std) <= 0).any()
    ):
        raise RuntimeError("Initial checkpoint normalizer standard deviations must be positive.")


def _new_policy_buffer(agent: Any, *, name: str) -> DipoleReplayBuffer:
    return DipoleReplayBuffer(
        agent.online_buffer.config,
        name=name,
        camera_names=list(agent.online_buffer.camera_names),
        action_horizon=int(agent.online_buffer.action_horizon),
        image_size=int(agent.online_buffer.image_size),
        augmentation_config=agent.online_buffer.augmentation_config,
    )


def _route_counts(routes: list[str]) -> dict[str, int]:
    counts = Counter(str(route) for route in routes)
    return {
        route: int(counts.get(route, 0))
        for route in (ROUTE_DISC_WEIGHTED, ROUTE_POS_ONLY, ROUTE_NEG_ONLY)
    }


def _prefixed_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}/{key}": float(value) for key, value in metrics.items()}


def _validate_expected_positive_counts(
    cfg: DictConfig,
    *,
    human: int,
    pretrain: int,
    pure_success: int,
) -> None:
    actual = {
        "expected_human_transitions": int(human),
        "expected_pretrain_transitions": int(pretrain),
        "expected_pure_success_transitions": int(pure_success),
        "expected_total_positive_transitions": int(
            human + pretrain + pure_success
        ),
    }
    for key, value in actual.items():
        expected = OmegaConf.select(
            cfg,
            f"offline.positive_training.{key}",
            default=None,
        )
        if expected is not None and int(expected) != value:
            raise RuntimeError(
                f"Positive-data count mismatch for {key}: "
                f"expected={int(expected)}, actual={value}."
            )


class _BatchPrefetcher:
    """Prepare the next static-cache batch while the current batch trains."""

    def __init__(
        self,
        sample_fn: Callable[..., Any],
        *,
        batch_size: int,
        sample_kwargs: dict[str, Any],
        depth: int = 2,
    ) -> None:
        self._sample_fn = sample_fn
        self._batch_size = int(batch_size)
        self._sample_kwargs = sample_kwargs
        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize=max(1, int(depth)))
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="dipole_offline_prefetch",
            daemon=True,
        )
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
    default = (
        Path(str(cfg.offline.data_root))
        / task_name
        / "offline_data"
        / "offline_episodes.pt"
    )
    if not default.exists():
        raise FileNotFoundError(
            f"offline_episodes.pt not found at {default}; set offline.episodes_path "
            "or run offline/scripts/collect_data.sh first."
        )
    return str(default)


def _next_episode_index(*transition_groups: list[Any]) -> int:
    highest = -1
    for transitions in transition_groups:
        for transition in transitions:
            highest = max(
                highest,
                int((transition.info or {}).get("episode_index", -1)),
            )
    return highest + 1


def _require_shared_cuda_device(*, score_device: str, policy_device: torch.device) -> None:
    requested = torch.device(score_device)
    if requested.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "Offline DIPOLE requires CUDA tensor execution; "
            f"requested device={requested}, cuda_available={torch.cuda.is_available()}."
        )
    if policy_device.type != "cuda" or policy_device != requested:
        raise RuntimeError(
            "Policy and discriminator score cache must share one CUDA device: "
            f"policy={policy_device}, discriminator={requested}."
        )


@hydra.main(
    version_base="1.2",
    config_path="../../config",
    config_name="train_offline_dipole",
)
def main(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision("high")
    repo_root = Path(to_absolute_path(".")).resolve()
    source_provenance = git_provenance(repo_root)
    training_mode = normalize_policy_training_mode(
        OmegaConf.select(
            cfg,
            "offline.positive_training.mode",
            default=POLICY_TRAINING_COUPLED,
        )
    )
    use_online_success = bool(
        OmegaConf.select(
            cfg,
            "offline.positive_training.include_pure_success",
            default=OmegaConf.select(cfg, "offline.use_online_success", default=False),
        )
    )

    ctx = build_agent_env(cfg, log_tag="offline")
    agent = ctx.agent
    task_name = ctx.task_name
    camera_names = list(agent.camera_names)
    action_horizon = int(agent.flow_config.action_horizon)
    score_device = str(cfg.algorithm.discriminator.learner_device)
    _require_shared_cuda_device(
        score_device=score_device,
        policy_device=agent.core.device,
    )
    _require_pretrained_normalizers(agent)

    nnpu_ckpt = _resolve_required_path(
        cfg.algorithm.discriminator.checkpoint,
        what="algorithm.discriminator.checkpoint (finetuned nnPU checkpoint)",
    )
    encoder_override = getattr(cfg.algorithm.discriminator, "encoder_ckpt", None)
    encoder_ckpt = (
        None
        if encoder_override is None
        or str(encoder_override).strip().lower() in ("", "null")
        else to_absolute_path(str(encoder_override))
    )
    camera_to_view = {
        str(key): str(value)
        for key, value in dict(
            getattr(cfg.algorithm.discriminator, "camera_to_view", {}) or {}
        ).items()
    }
    shared_encoder = SharedDynamicsEncoder(
        nnpu_ckpt_path=nnpu_ckpt,
        encoder_ckpt=encoder_ckpt,
        device=score_device,
        camera_to_view=camera_to_view,
    )
    shared_encoder.bind_policy_cameras(camera_names)
    discriminator = FrozenNNPUDiscriminator(
        nnpu_ckpt_path=nnpu_ckpt,
        task_name=task_name,
        device=score_device,
        encoder=shared_encoder,
    )
    print(
        f"[offline] discriminator={nnpu_ckpt} device={score_device} "
        f"threshold={discriminator.threshold:+.6f}"
    )

    episodes_path = _resolve_episodes_path(cfg, task_name)
    payload = torch.load(episodes_path, map_location="cpu", weights_only=False)
    payload_cameras = [str(camera) for camera in payload.get("camera_names", [])]
    if payload_cameras and set(payload_cameras) != set(camera_names):
        raise RuntimeError(
            f"camera mismatch: episodes {payload_cameras} vs policy {camera_names}. "
            "Collect data with the same init_checkpoint."
        )
    streams = build_offline_transitions(
        payload,
        action_horizon=action_horizon,
        reward_success=float(
            OmegaConf.select(cfg, "offline.reward_success", default=0.0)
        ),
        reward_fail=float(OmegaConf.select(cfg, "offline.reward_fail", default=-1.0)),
        include_policy_action_neg=bool(
            OmegaConf.select(cfg, "offline.include_policy_action_neg", default=True)
        ),
    )
    if not streams.policy_bc:
        raise RuntimeError(
            "No policy sections survived filtering; discriminator-weighted policy "
            "training requires at least one valid policy window."
        )
    print(f"[offline] episodes={episodes_path}")
    print(f"[offline] streams: {streams.stats}")

    next_episode = _next_episode_index(
        streams.policy_bc,
        streams.human_pos,
        streams.neg,
    )
    online_success_pos: list[Any] = []
    online_success_stats: dict[str, Any] = {}
    online_success_replaced_policy_bc = 0
    policy_bc_for_policy = list(streams.policy_bc)
    if use_online_success:
        online_success_pos, next_episode, online_success_stats = (
            build_online_success_transitions(
                payload,
                action_horizon=action_horizon,
                episode_index_base=next_episode,
                route=ROUTE_POS_ONLY,
            )
        )
        source_episode_indices = {
            int(index)
            for index in online_success_stats.get(
                "pure_success_source_episode_indices", []
            )
        }
        if source_episode_indices:
            policy_bc_for_policy = [
                transition
                for transition in streams.policy_bc
                if int((transition.info or {}).get("source_episode_index", -1))
                not in source_episode_indices
            ]
            online_success_replaced_policy_bc = (
                len(streams.policy_bc) - len(policy_bc_for_policy)
            )
            remaining_source_indices = {
                int((transition.info or {}).get("source_episode_index", -1))
                for transition in policy_bc_for_policy
            }
            overlap = source_episode_indices & remaining_source_indices
            if overlap:
                raise RuntimeError(
                    "Pure-success source episodes remain in the soft policy pool: "
                    f"{sorted(overlap)}"
                )
        print(
            f"[offline] online_success: {online_success_stats}; "
            f"replaced_policy_bc={online_success_replaced_policy_bc}"
        )
    else:
        print("[offline] online_success: disabled")

    if use_online_success and not online_success_pos:
        raise RuntimeError(
            "include_pure_success=true but no pure-success transitions were found."
        )

    pretrain_pos: list[Any] = []
    pretrain_data_path: str | None = None
    raw_pretrain = OmegaConf.select(cfg, "offline.pretrain_data_path", default=None)
    if raw_pretrain is not None and str(raw_pretrain).strip().lower() not in (
        "",
        "null",
    ):
        pretrain_data_path = _resolve_required_path(
            raw_pretrain,
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
            episode_index_base=next_episode,
        )
        print(
            f"[offline] pretrain_data={pretrain_data_path} "
            f"transitions={len(pretrain_pos)}"
        )

    _validate_expected_positive_counts(
        cfg,
        human=len(streams.human_pos),
        pretrain=len(pretrain_pos),
        pure_success=len(online_success_pos),
    )

    try:
        ctx.env.close()
    except Exception:
        pass

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
                        default="./outputs/dipole-rl-offline_disc",
                    )
                )
            )
        ).resolve()
        suffix = str(
            OmegaConf.select(cfg, "offline.run_subfix", default="") or ""
        ).strip()
        directory_name = (
            f"{task_name}_{timestamp}_{suffix}"
            if suffix
            else f"{task_name}_{timestamp}"
        )
        pipeline_run_dir = run_root / directory_name
        run_dir = pipeline_run_dir
        if pipeline_run_dir.exists():
            raise FileExistsError(
                f"Offline DIPOLE pipeline directory already exists: {pipeline_run_dir}"
            )
    else:
        # Explicit run_dir is the pipeline root (sibling of discriminator/).
        run_dir = Path(to_absolute_path(str(explicit_run_dir))).resolve()
        pipeline_run_dir = run_dir
    checkpoint_dir = run_dir / "checkpoints"
    if checkpoint_dir.exists():
        raise FileExistsError(
            f"Offline DIPOLE checkpoints directory already exists: {checkpoint_dir}"
        )
    checkpoint_dir.mkdir(parents=True)
    run_name = f"{task_name}__offline_dipole_disc__{timestamp}"
    print(f"[offline] pipeline_run_dir={pipeline_run_dir}")
    print(f"[offline] stage_run_dir={run_dir}")

    metric_logger = maybe_build_metric_logger(
        cfg,
        run_name=run_name,
        run_dir=run_dir,
    )
    write_resolved_config(cfg, run_dir)

    all_transitions = (
        list(policy_bc_for_policy)
        + list(streams.human_pos)
        + list(streams.neg)
        + list(online_success_pos)
        + list(pretrain_pos)
    )
    valid_windows = populate_replay_buffer(agent.online_buffer, all_transitions)
    print(
        f"[offline] policy buffer: {len(agent.online_buffer)} transitions, "
        f"{valid_windows} valid windows "
        f"(disc_weighted={len(policy_bc_for_policy)}, "
        f"human_pos={len(streams.human_pos)}, neg={len(streams.neg)}, "
        f"online_success_pos={len(online_success_pos)}, "
        f"pretrain_pos={len(pretrain_pos)})"
    )
    batch_size = int(cfg.algorithm.trainer.batch_size)
    if valid_windows < batch_size:
        raise RuntimeError(
            f"Policy buffer has only {valid_windows} valid windows "
            f"(< batch_size={batch_size})."
        )

    static_cache = agent.online_buffer.build_static_cache(pin_memory=True)
    cache_mib = float(static_cache.estimated_bytes) / (1024.0 * 1024.0)
    print(
        f"[offline] static_cache rows={len(static_cache)} "
        f"estimated={cache_mib:.1f} MiB"
    )
    positive_transitions = (
        list(streams.human_pos) + list(online_success_pos) + list(pretrain_pos)
    )
    positive_static_cache = None
    positive_valid_windows: int | None = None
    if training_mode == POLICY_TRAINING_FILTERED_BC:
        positive_buffer = _new_policy_buffer(
            agent,
            name="offline_filtered_positive_buffer",
        )
        positive_valid_windows = populate_replay_buffer(
            positive_buffer,
            positive_transitions,
        )
        if positive_valid_windows < batch_size:
            raise RuntimeError(
                "Filtered positive buffer has only "
                f"{positive_valid_windows} valid windows "
                f"(< batch_size={batch_size})."
            )
        positive_static_cache = positive_buffer.build_static_cache(pin_memory=True)
        if set(positive_static_cache.routes) != {ROUTE_POS_ONLY}:
            raise RuntimeError(
                "Filtered positive buffer contains a non-pos_only route."
            )
        print(
            f"[offline] filtered positive buffer: transitions={len(positive_transitions)} "
            f"valid_windows={positive_valid_windows}"
        )
    elif training_mode != POLICY_TRAINING_COUPLED:
        positive_static_cache = static_cache
    raw_scores, g_values, start_to_row = (
        precompute_discriminator_scores(
            static_cache=static_cache,
            encoder=shared_encoder,
            discriminator=discriminator,
            device=score_device,
            batch_size=int(
                OmegaConf.select(cfg, "offline.preencode_batch_size", default=64)
            ),
        )
    )
    provider = OfflineDiscriminatorGProvider(
        raw_scores=raw_scores,
        g_values=g_values,
        start_to_row=start_to_row,
        threshold=float(discriminator.threshold),
    )
    agent.attach_discriminator(discriminator)
    agent.attach_g_provider(provider)

    branch_beta = float(
        OmegaConf.select(
            cfg,
            "offline.branch_weight.beta",
            default=agent.core.config.beta,
        )
    )
    agent.core.config.beta = branch_beta
    agent.core.config.k = 0.0
    branch_policy = build_branch_weight_policy(
        OmegaConf.select(
            cfg,
            "offline.branch_weight",
            default=OmegaConf.create({"type": "routed_sigmoid"}),
        )
    )
    agent.core.set_branch_weight_policy(branch_policy)
    print(
        f"[offline] branch_weight={type(branch_policy).__name__} beta={branch_beta} "
        "formula=w_pos=sigmoid(beta*(threshold-raw_score))"
    )

    raw_score_mean = float(raw_scores.mean().item())
    raw_score_std = float(raw_scores.std().item()) if raw_scores.numel() > 1 else 0.0
    g_mean = float(g_values.mean().item())
    g_std = (
        float(g_values.std().item())
        if g_values.numel() > 1
        else 0.0
    )
    maybe_log(
        metric_logger,
        {
            "offline_discriminator/raw_score_mean": raw_score_mean,
            "offline_discriminator/raw_score_std": raw_score_std,
            "offline_discriminator/threshold": float(discriminator.threshold),
            "offline_discriminator/G_mean": g_mean,
            "offline_discriminator/G_std": g_std,
            "offline_discriminator/beta": branch_beta,
        },
        step=0,
    )
    input_provenance = {
        "git": source_provenance,
        "initial_policy": file_provenance(ctx.init_checkpoint),
        "finetuned_discriminator": file_provenance(nnpu_ckpt),
        "offline_episodes": file_provenance(episodes_path),
        "pretrain_data": (
            None
            if pretrain_data_path is None
            else directory_input_provenance(pretrain_data_path)
        ),
    }
    run_info = {
            "run_name": run_name,
            "run_dir": str(run_dir),
            "pipeline_run_dir": str(pipeline_run_dir),
            "stage_run_dir": str(run_dir),
            "started_at": timestamp,
            "task_name": task_name,
            "policy_camera_names": camera_names,
            "initialized_checkpoint": str(ctx.init_checkpoint),
            "finetuned_discriminator_checkpoint": nnpu_ckpt,
            "episodes_path": episodes_path,
            "pretrain_data_path": pretrain_data_path,
            "pretrain_transitions": len(pretrain_pos),
            "online_success_enabled": use_online_success,
            "online_success_transitions": len(online_success_pos),
            "online_success_replaced_policy_bc_transitions": (
                online_success_replaced_policy_bc
            ),
            "online_success_stats": online_success_stats,
            "online_success_replaced_policy_bc": int(
                online_success_replaced_policy_bc
            ),
            "algorithm": "discriminator_weighted_offline_dipole",
            "positive_training_mode": training_mode,
            "positive_transitions": len(positive_transitions),
            "positive_valid_windows": positive_valid_windows,
            "policy_buffer_route_counts": _route_counts(static_cache.routes),
            "positive_buffer_route_counts": (
                None
                if positive_static_cache is None
                else _route_counts(positive_static_cache.routes)
            ),
            "rng_streams": {
                "base_seed": int(cfg.seed),
                "derivation": (
                    "branch_seed(base_seed, step, branch in {pos,neg}, "
                    "phase in {sample,update})"
                ),
            },
            "score_semantics": "raw_score=-head_logit; higher=failure_like",
            "threshold": float(discriminator.threshold),
            "g_formula": "threshold-raw_score",
            "branch_beta": branch_beta,
            "raw_score_mean": raw_score_mean,
            "raw_score_std": raw_score_std,
            "g_mean": g_mean,
            "g_std": g_std,
            "stream_stats": streams.stats,
            "replay_valid_windows": int(valid_windows),
            "score_cache_rows": len(start_to_row),
            "provenance": input_provenance,
            "completed_steps": 0,
            "output_checkpoint": None,
        }
    write_run_info(run_dir, run_info)

    num_steps = int(cfg.offline.num_train_steps)
    if num_steps <= 0:
        raise ValueError("offline.num_train_steps must be positive.")
    log_interval = max(1, int(cfg.offline.log_interval))
    plot_interval_raw = OmegaConf.select(
        cfg,
        "offline.plot_log_interval",
        default=None,
    )
    plot_interval = max(
        1,
        int(log_interval if plot_interval_raw is None else plot_interval_raw),
    )
    checkpoint_interval = max(1, int(cfg.offline.checkpoint_interval))
    sample_kwargs = {
        "action_mean": agent.core.act_mean,
        "action_std": agent.core.act_std,
        "proprio_mean": agent.core.prop_mean,
        "proprio_std": agent.core.prop_std,
        "device": agent.core.device,
        "augment": True,
    }

    def save_checkpoint(tag: str, completed_steps: int) -> Path:
        checkpoint = agent.build_checkpoint_payload(
            include_buffers=False,
            extra={
                "global_step": int(completed_steps),
                "completed_steps": int(completed_steps),
                "run_name": run_name,
                "positive_training_mode": training_mode,
            },
        )
        path = checkpoint_path(run_dir, tag)
        agent.write_checkpoint_payload(path, checkpoint)
        return path

    parameter_snapshots = trainable_parameter_snapshot(agent.core)
    prefetcher = None
    if training_mode == POLICY_TRAINING_COUPLED:
        prefetcher = _BatchPrefetcher(
            static_cache.sample,
            batch_size=batch_size,
            sample_kwargs=sample_kwargs,
        )
    else:
        assert positive_static_cache is not None
        pos_numpy_rng = np.random.default_rng(
            branch_seed(int(cfg.seed), step=0, branch="pos", phase="sample")
        )
        neg_numpy_rng = np.random.default_rng(
            branch_seed(int(cfg.seed), step=0, branch="neg", phase="sample")
        )
    print(
        f"[offline] policy training for {num_steps} steps "
        f"(mode={training_mode}, batch_size={batch_size}, "
        f"prefetch={'on' if prefetcher is not None else 'off'})"
    )
    completed_steps = 0
    final_path: Path | None = None
    try:
        for step in range(num_steps):
            is_log = step % log_interval == 0 or step == num_steps - 1
            collect_diagnostics = (
                step % plot_interval == 0 or step == num_steps - 1
            )
            if training_mode == POLICY_TRAINING_COUPLED:
                assert prefetcher is not None
                batch = prefetcher.next()
                metrics = agent.core.update(
                    batch=batch,
                    want_metrics=is_log,
                    collect_diagnostics=collect_diagnostics,
                )
                diagnostics = metrics.pop("_diag", None)
            else:
                pos_batch = sample_static_cache(
                    positive_static_cache,
                    batch_size,
                    sample_kwargs=sample_kwargs,
                    numpy_rng=pos_numpy_rng,
                    torch_seed=branch_seed(
                        int(cfg.seed), step=step, branch="pos", phase="sample"
                    ),
                )
                neg_batch = sample_static_cache(
                    static_cache,
                    batch_size,
                    sample_kwargs=sample_kwargs,
                    numpy_rng=neg_numpy_rng,
                    torch_seed=branch_seed(
                        int(cfg.seed), step=step, branch="neg", phase="sample"
                    ),
                )
                want_branch_metrics = bool(is_log or collect_diagnostics)
                if training_mode == POLICY_TRAINING_FILTERED_BC:
                    pos_weights = torch.ones(
                        pos_batch.batch_size,
                        device=agent.core.device,
                        dtype=torch.float32,
                    )
                    pos_weight_metrics = {
                        "frac_pos_only": 1.0,
                        "frac_neg_only": 0.0,
                        "frac_disc_weighted": 0.0,
                        "effective_pos_mass/pos_only": 1.0,
                    }
                else:
                    pos_weights, _, pos_weight_metrics = branch_policy(
                        pos_batch,
                        g_provider=provider,
                        sigmoid_fn=agent.core._g_weights_from_g,  # noqa: SLF001
                        device=agent.core.device,
                        want_metrics=want_branch_metrics,
                    )
                _, neg_weights, neg_weight_metrics = branch_policy(
                    neg_batch,
                    g_provider=provider,
                    sigmoid_fn=agent.core._g_weights_from_g,  # noqa: SLF001
                    device=agent.core.device,
                    want_metrics=want_branch_metrics,
                )
                pos_metrics = branch_only_update(
                    agent.core,
                    pos_batch,
                    branch="pos",
                    weights=pos_weights,
                    torch_seed=branch_seed(
                        int(cfg.seed), step=step, branch="pos", phase="update"
                    ),
                    want_metrics=want_branch_metrics,
                )
                neg_metrics = branch_only_update(
                    agent.core,
                    neg_batch,
                    branch="neg",
                    weights=neg_weights,
                    torch_seed=branch_seed(
                        int(cfg.seed), step=step, branch="neg", phase="update"
                    ),
                    want_metrics=want_branch_metrics,
                )
                pos_v_mse = pos_metrics.pop("v_pos_mse", None)
                neg_v_mse = neg_metrics.pop("v_neg_mse", None)
                metrics = {
                    **pos_metrics,
                    **neg_metrics,
                    **_prefixed_metrics("positive_weights", pos_weight_metrics),
                    **_prefixed_metrics("negative_weights", neg_weight_metrics),
                }
                if want_branch_metrics:
                    metrics["actor_loss"] = (
                        metrics["loss_pos"] + metrics["loss_neg"]
                    )
                    metrics["G_mean"] = float(
                        pos_weight_metrics.get(
                            "G_mean",
                            neg_weight_metrics.get("G_mean", 0.0),
                        )
                    )
                diagnostics = None
                if collect_diagnostics:
                    diagnostics = {
                        "g_values": (
                            agent.core._provider_g_for_batch(pos_batch)  # noqa: SLF001
                            .detach()
                            .cpu()
                            .numpy()
                            if training_mode != POLICY_TRAINING_FILTERED_BC
                            else np.zeros(pos_batch.batch_size, dtype=np.float32)
                        ),
                        "w_pos": pos_weights.detach().cpu().numpy(),
                        "w_neg": neg_weights.detach().cpu().numpy(),
                        "v_pos_mse": pos_v_mse.detach().cpu().numpy(),
                        "v_neg_mse": neg_v_mse.detach().cpu().numpy(),
                        "independent_batches": True,
                    }
            completed_steps = step + 1
            if is_log:
                metrics.update(
                    parameter_distance_metrics(agent.core, parameter_snapshots)
                )
                maybe_log(
                    metric_logger,
                    {f"train/{key}": float(value) for key, value in metrics.items()},
                    step=step,
                )
                raw_score_mean = metrics.get(
                    "raw_score_mean",
                    metrics.get(
                        "positive_weights/raw_score_mean",
                        metrics.get("negative_weights/raw_score_mean", 0.0),
                    ),
                )
                frac_disc_weighted = metrics.get(
                    "frac_disc_weighted",
                    metrics.get("positive_weights/frac_disc_weighted", 0.0),
                )
                print(
                    f"[offline][step {step:6d}] "
                    f"loss={metrics.get('actor_loss', 0.0):.4f} "
                    f"w_pos={metrics.get('w_pos_mean', 0.0):.3f} "
                    f"w_neg={metrics.get('w_neg_mean', 0.0):.3f} "
                    f"G_mean={metrics.get('G_mean', 0.0):+.3f} "
                    f"raw_score={raw_score_mean:+.3f} "
                    f"frac_disc={frac_disc_weighted:.2f}"
                )
            if diagnostics is not None:
                maybe_log_figure(
                    metric_logger,
                    "train/G_hist",
                    plot_g_distribution(diagnostics["g_values"]),
                    step,
                )
                maybe_log_figure(
                    metric_logger,
                    "train/w_pos_hist",
                    plot_branch_weight_distribution(
                        diagnostics["w_pos"], label="w_pos"
                    ),
                    step,
                )
                maybe_log_figure(
                    metric_logger,
                    "train/w_neg_hist",
                    plot_branch_weight_distribution(
                        diagnostics["w_neg"], label="w_neg"
                    ),
                    step,
                )
                if not diagnostics.get("independent_batches", False):
                    maybe_log_figure(
                        metric_logger,
                        "train/v_pos_neg_mse_scatter",
                        plot_v_pos_neg_scatter(
                            diagnostics["v_pos_mse"],
                            diagnostics["v_neg_mse"],
                        ),
                        step,
                    )
            if completed_steps % checkpoint_interval == 0:
                step_path = save_checkpoint(
                    f"step_{completed_steps:08d}", completed_steps
                )
                save_checkpoint("latest", completed_steps)
                print(
                    f"[offline][ckpt] completed_steps={completed_steps} -> "
                    f"{step_path.name} (+latest)"
                )
    except Exception:
        print(
            f"[offline] training failed after completed_steps={completed_steps}; "
            "no final checkpoint was written."
        )
        raise
    else:
        final_path = save_checkpoint("latest", completed_steps)
        save_checkpoint(f"step_{completed_steps:08d}", completed_steps)
        run_info["completed_steps"] = int(completed_steps)
        run_info["output_checkpoint"] = file_provenance(final_path)
        write_run_info(run_dir, run_info)
        print(f"[offline] done. final checkpoint -> {final_path}")
    finally:
        if prefetcher is not None:
            prefetcher.close()
        if metric_logger is not None:
            metric_logger.close()


if __name__ == "__main__":
    main()
