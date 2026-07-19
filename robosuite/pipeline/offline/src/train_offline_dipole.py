"""CUDA-only discriminator-weighted Offline DIPOLE training."""

from __future__ import annotations

import datetime
import logging
import queue
import threading
from pathlib import Path
from typing import Any, Callable

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from robosuite.pipeline.algorithms.discriminator.encoder import SharedDynamicsEncoder
from robosuite.pipeline.algorithms.discriminator.nnpu import FrozenNNPUDiscriminator
from robosuite.pipeline.offline.src.diagnostic_plots import (
    plot_branch_weight_distribution,
    plot_g_distribution,
    plot_v_pos_neg_scatter,
)
from robosuite.pipeline.offline.utils import (
    OfflineDiscriminatorGProvider,
    build_agent_env,
    build_branch_weight_policy,
    build_offline_transitions,
    build_online_success_transitions,
    finalize_normalizers,
    load_pretrain_transitions,
    make_hdf5_loader,
    populate_replay_buffer,
    precompute_discriminator_scores,
)
from robosuite.pipeline.offline.utils.episode_dataset import ROUTE_POS_ONLY
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

    use_online_success = bool(
        OmegaConf.select(cfg, "offline.use_online_success", default=False)
    )
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
        print(
            f"[offline] online_success: {online_success_stats}; "
            f"replaced_policy_bc={online_success_replaced_policy_bc}"
        )
    else:
        print("[offline] online_success: disabled")

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
    batch_size = finalize_normalizers(
        agent,
        cfg,
        list(policy_bc_for_policy) + list(online_success_pos) + list(pretrain_pos),
        log_tag="offline",
        norm_desc="policy sections + online-success + pretrain positive demos",
    )

    static_cache = agent.online_buffer.build_static_cache(pin_memory=True)
    cache_mib = float(static_cache.estimated_bytes) / (1024.0 * 1024.0)
    print(
        f"[offline] static_cache rows={len(static_cache)} "
        f"estimated={cache_mib:.1f} MiB"
    )
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
            "algorithm": "discriminator_weighted_offline_dipole",
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
        },
    )

    num_steps = int(cfg.offline.num_train_steps)
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

    def save_checkpoint(tag: str, step: int) -> Path:
        checkpoint = agent.build_checkpoint_payload(
            include_buffers=False,
            extra={"global_step": int(step), "run_name": run_name},
        )
        path = checkpoint_path(run_dir, tag)
        agent.write_checkpoint_payload(path, checkpoint)
        return path

    prefetcher = _BatchPrefetcher(
        static_cache.sample,
        batch_size=batch_size,
        sample_kwargs=sample_kwargs,
    )
    print(
        f"[offline] policy training for {num_steps} steps "
        f"(batch_size={batch_size}, prefetch=on)"
    )
    try:
        for step in range(num_steps):
            batch = prefetcher.next()
            is_log = step % log_interval == 0 or step == num_steps - 1
            collect_diagnostics = (
                step % plot_interval == 0 or step == num_steps - 1
            )
            metrics = agent.core.update(
                batch=batch,
                want_metrics=is_log,
                collect_diagnostics=collect_diagnostics,
            )
            diagnostics = metrics.pop("_diag", None)
            if is_log:
                maybe_log(
                    metric_logger,
                    {f"train/{key}": float(value) for key, value in metrics.items()},
                    step=step,
                )
                print(
                    f"[offline][step {step:6d}] "
                    f"loss={metrics.get('actor_loss', 0.0):.4f} "
                    f"w_pos={metrics.get('w_pos_mean', 0.0):.3f} "
                    f"w_neg={metrics.get('w_neg_mean', 0.0):.3f} "
                    f"G_mean={metrics.get('G_mean', 0.0):+.3f} "
                    f"raw_score={metrics.get('raw_score_mean', 0.0):+.3f} "
                    f"frac_disc={metrics.get('frac_disc_weighted', 0.0):.2f}"
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
                maybe_log_figure(
                    metric_logger,
                    "train/v_pos_neg_mse_scatter",
                    plot_v_pos_neg_scatter(
                        diagnostics["v_pos_mse"],
                        diagnostics["v_neg_mse"],
                    ),
                    step,
                )
            if step > 0 and step % checkpoint_interval == 0:
                step_path = save_checkpoint(f"step_{step:08d}", step)
                save_checkpoint("latest", step)
                print(
                    f"[offline][ckpt] step={step} -> "
                    f"{step_path.name} (+latest)"
                )
    finally:
        prefetcher.close()
        final_step = max(0, num_steps - 1)
        final_path = save_checkpoint("latest", final_step)
        save_checkpoint(f"step_{final_step:08d}", final_step)
        if metric_logger is not None:
            metric_logger.close()
        print(f"[offline] done. final checkpoint -> {final_path}")


if __name__ == "__main__":
    main()
