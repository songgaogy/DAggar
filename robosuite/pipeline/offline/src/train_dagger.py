"""Offline DAgger positive-only policy training.

This entry trains a standard DIPOLE checkpoint with only positive targets:
human intervention actions from ``offline_episodes.pt`` plus expert pretrain
HDF5 demos. The negative policy is kept in the checkpoint for compatibility with
``eval_offline_dipole.py`` but is never forwarded, backpropagated, or optimized.
"""

from __future__ import annotations

import datetime
import logging
import queue
import threading
from pathlib import Path
from typing import Any, Callable

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from robosuite.pipeline.algorithms.dipole.common import DipoleBatch
from robosuite.pipeline.offline.utils import (
    build_agent_env,
    build_offline_transitions,
    build_online_success_transitions,
    finalize_normalizers,
    load_pretrain_transitions,
    make_hdf5_loader,
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


class _BatchPrefetcher:
    """Background batch prefetcher for the static offline replay cache."""

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
        self._thread = threading.Thread(target=self._run, name="dagger_offline_prefetch", daemon=True)
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
        raise RuntimeError(f"Offline DAgger requires {what} to be set.")
    path = to_absolute_path(str(raw))
    if not Path(path).exists():
        raise FileNotFoundError(f"{what} not found: {path}")
    return path


def _resolve_episodes_path(cfg: DictConfig, task_name: str) -> str:
    raw = OmegaConf.select(cfg, "offline.episodes_path", default=None)
    if raw is not None and str(raw).strip().lower() not in ("", "null"):
        return _resolve_required_path(raw, what="offline.episodes_path")
    data_root = str(OmegaConf.select(cfg, "offline.data_root", default="data"))
    default = Path(data_root) / task_name / "offline_data" / "offline_episodes.pt"
    if not default.exists():
        raise FileNotFoundError(
            f"offline_episodes.pt not found at {default}; set offline.episodes_path "
            "or run offline/scripts/collect_data.sh first."
        )
    return str(default)


def _next_episode_index(*transition_groups: list[Any]) -> int:
    hi = -1
    for transitions in transition_groups:
        for transition in transitions:
            info = transition.info or {}
            hi = max(hi, int(info.get("episode_index", -1)))
    return hi + 1


def _require_cuda_device(device: str) -> None:
    if not str(device).startswith("cuda"):
        raise RuntimeError(f"Offline DAgger requires a CUDA device, got {device!r}.")
    if not torch.cuda.is_available():
        raise RuntimeError("Offline DAgger requires CUDA, but torch.cuda.is_available() is false.")


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.reshape(-1)
    values = values.reshape(-1)
    weight_sum = torch.sum(weights).clamp_min(1e-6)
    return torch.sum(values * weights) / weight_sum


def _positive_only_update(
    core: Any,
    batch: DipoleBatch,
    *,
    want_metrics: bool = True,
) -> dict[str, float]:
    """One policy-gradient step on ``model_pos`` only."""
    batch = batch.to(core.device)
    want_metrics = bool(want_metrics)

    B = batch.batch_size
    noise = torch.randn_like(batch.action_sequences)
    timesteps = torch.rand(B, device=core.device)
    x_t = (
        (1.0 - timesteps).view(-1, 1, 1) * noise
        + timesteps.view(-1, 1, 1) * batch.action_sequences
    )
    v_target = batch.action_sequences - noise
    language = [core.language_instruction] * B
    x_t_swapped = x_t.transpose(1, 2)
    weights = torch.ones(B, dtype=torch.float32, device=core.device)

    core.model_pos.train(True)
    core.optimizer_pos.zero_grad(set_to_none=True)
    with torch.amp.autocast(enabled=(core.device.type == "cuda"), device_type=core.device.type):
        v_pred = core.model_pos(
            x_t=x_t_swapped,
            t=timesteps,
            images=batch.image_obs,
            proprio=batch.proprio,
            language=language,
        ).transpose(1, 2)
        fp = torch.mean((v_pred - v_target) ** 2, dim=(1, 2))
        x1 = x_t + (1.0 - timesteps).view(-1, 1, 1) * v_pred
        ep = torch.mean((x1 - batch.action_sequences) ** 2, dim=(1, 2))
        if batch.action_sequences.shape[1] > 1:
            sm = torch.mean((x1[:, 1:] - x1[:, :-1]) ** 2, dim=(1, 2))
        else:
            sm = torch.zeros(B, device=core.device, dtype=fp.dtype)
        flow = _weighted_mean(fp, weights)
        endpoint = _weighted_mean(ep, weights)
        smooth = _weighted_mean(sm, weights)
        loss = (
            flow
            + float(core.config.lambda_endpoint) * endpoint
            + float(core.config.lambda_smooth) * smooth
        )

    core.scaler_pos.scale(loss).backward()
    core.scaler_pos.unscale_(core.optimizer_pos)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        core.model_pos.parameters(),
        max_norm=float(core.config.grad_clip_norm),
    )
    core.scaler_pos.step(core.optimizer_pos)
    core.scaler_pos.update()

    if not want_metrics:
        return {}
    return {
        "actor_loss": float(loss.detach().cpu().item()),
        "loss_pos": float(loss.detach().cpu().item()),
        "loss_neg": 0.0,
        "flow_loss_pos": float(flow.detach().cpu().item()),
        "endpoint_loss_pos": float(endpoint.detach().cpu().item()),
        "smooth_loss_pos": float(smooth.detach().cpu().item()),
        "grad_norm_pos": float(grad_norm.detach().cpu().item()),
        "w_pos_mean": 1.0,
        "w_neg_mean": 0.0,
        "frac_intervention": float(batch.is_intervention.float().mean().detach().cpu().item()),
    }


@hydra.main(version_base="1.2", config_path="../../config", config_name="train_offline_dipole")
def main(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision("high")

    device = str(OmegaConf.select(cfg, "algorithm.flow.device", default="cuda:0"))
    inference_device = str(OmegaConf.select(cfg, "algorithm.flow.inference_device", default=device))
    _require_cuda_device(device)
    _require_cuda_device(inference_device)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ctx = build_agent_env(cfg, log_tag="dagger")
    agent = ctx.agent
    task_name = ctx.task_name
    camera_names = list(agent.camera_names)
    H = int(agent.flow_config.action_horizon)

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
        include_policy_action_neg=False,
    )
    print(f"[dagger] episodes={episodes_path}")
    print(f"[dagger] streams: {streams.stats}")

    # Optional: add entire pure on-policy success rollouts (no human intervention
    # anywhere) as positive BC demonstrations. Episode indices are threaded so the
    # online-success episodes never collide with human_pos / pretrain windows.
    use_online_success = bool(OmegaConf.select(cfg, "offline.use_online_success", default=False))
    next_ep_base = _next_episode_index(streams.policy_bc, streams.human_pos, streams.neg)
    online_success_pos: list[Any] = []
    online_success_stats: dict[str, Any] = {}
    if use_online_success:
        online_success_pos, next_ep_base, online_success_stats = build_online_success_transitions(
            payload,
            action_horizon=H,
            episode_index_base=next_ep_base,
        )
        print(f"[dagger] online_success: {online_success_stats}")
    else:
        print("[dagger] online_success: disabled (offline.use_online_success=false)")

    raw_pretrain = OmegaConf.select(cfg, "offline.pretrain_data_path", default=None)
    pretrain_data_path = _resolve_required_path(raw_pretrain, what="offline.pretrain_data_path")
    hdf5_loader = make_hdf5_loader(ctx, cfg)
    pretrain_pos, _ = load_pretrain_transitions(
        data_root=str(OmegaConf.select(cfg, "offline.data_root", default="data")),
        task_name=task_name,
        pretrain_dir=pretrain_data_path,
        hdf5_loader=hdf5_loader,
        max_num_trajectories=OmegaConf.select(cfg, "offline.max_pretrain_trajectories", default=None),
        episode_index_base=next_ep_base,
    )
    print(f"[dagger] pretrain_data={pretrain_data_path} transitions={len(pretrain_pos)}")

    try:
        ctx.env.close()
    except Exception:
        pass

    positive_transitions = list(streams.human_pos) + list(online_success_pos) + list(pretrain_pos)
    if not positive_transitions:
        raise RuntimeError(
            "No positive transitions found: human_pos + online_success + pretrain_pos is empty."
        )

    n_valid = populate_replay_buffer(agent.online_buffer, positive_transitions)
    print(
        f"[dagger] policy-BC buffer: {len(agent.online_buffer)} transitions, {n_valid} valid windows "
        f"(human_pos={len(streams.human_pos)}, online_success={len(online_success_pos)}, "
        f"pretrain_pos={len(pretrain_pos)})"
    )
    batch_size = finalize_normalizers(
        agent,
        cfg,
        positive_transitions,
        log_tag="dagger",
        norm_desc="human intervention + pretrain positive demos",
    )

    run_root = Path(
        to_absolute_path(str(OmegaConf.select(cfg, "offline.run_root", default="./outputs/dipole-dagger-offline")))
    )
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    postfix = str(OmegaConf.select(cfg, "offline.run_subfix", default="dagger") or "").strip()
    dir_name = f"{task_name}_{timestamp}_{postfix}" if postfix else f"{task_name}_{timestamp}"
    run_dir = run_root / dir_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    run_name = f"{task_name}__offline_dagger__{timestamp}"
    print(f"[dagger] run_dir={run_dir}")

    metric_logger = maybe_build_metric_logger(cfg, run_name=run_name, run_dir=run_dir)
    write_resolved_config(cfg, run_dir)
    write_run_info(
        run_dir,
        {
            "run_name": run_name,
            "run_dir": str(run_dir),
            "started_at": timestamp,
            "task_name": task_name,
            "policy_camera_names": camera_names,
            "initialized_checkpoint": str(ctx.init_checkpoint),
            "episodes_path": episodes_path,
            "pretrain_data_path": pretrain_data_path,
            "human_pos_transitions": len(streams.human_pos),
            "online_success_enabled": use_online_success,
            "online_success_transitions": len(online_success_pos),
            "online_success_stats": online_success_stats,
            "pretrain_transitions": len(pretrain_pos),
            "positive_transitions": len(positive_transitions),
            "algorithm_type": "offline_dagger_positive_only",
            "negative_policy": "kept_from_initial_checkpoint_not_updated",
            "stream_stats": streams.stats,
            "replay_valid_windows": int(n_valid),
        },
    )

    static_cache = agent.online_buffer.build_static_cache(pin_memory=True)
    cache_mib = float(static_cache.estimated_bytes) / (1024.0 * 1024.0)
    print(f"[dagger] static_cache rows={len(static_cache)} estimated={cache_mib:.1f} MiB")

    num_steps = int(OmegaConf.select(cfg, "offline.num_train_steps", default=15000))
    log_interval = max(1, int(OmegaConf.select(cfg, "offline.log_interval", default=200)))
    checkpoint_interval = max(1, int(OmegaConf.select(cfg, "offline.checkpoint_interval", default=5000)))
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
    )
    print(f"[dagger] positive-only policy training for {num_steps} steps (batch_size={batch_size})")
    try:
        for step in range(num_steps):
            batch = prefetcher.next()
            is_log = (step % log_interval == 0) or (step == num_steps - 1)
            metrics = _positive_only_update(agent.core, batch, want_metrics=is_log)
            if is_log:
                maybe_log(
                    metric_logger,
                    {f"train/{key}": float(value) for key, value in metrics.items()},
                    step=step,
                )
                print(
                    f"[dagger][step {step:6d}] loss={metrics.get('actor_loss', 0.0):.4f} "
                    f"pos={metrics.get('loss_pos', 0.0):.4f} neg=0.0000 "
                    f"frac_int={metrics.get('frac_intervention', 0.0):.2f}"
                )
            if step > 0 and step % checkpoint_interval == 0:
                step_path = _save(f"step_{step:08d}", step)
                _save("latest", step)
                print(f"[dagger][ckpt] step={step} -> {step_path.name} (+latest)")
    finally:
        prefetcher.close()
        final_step = max(0, num_steps - 1)
        final_path = _save("latest", final_step)
        _save(f"step_{final_step:08d}", final_step)
        if metric_logger is not None:
            metric_logger.close()
        print(f"[dagger] done. final checkpoint -> {final_path}")


if __name__ == "__main__":
    main()
