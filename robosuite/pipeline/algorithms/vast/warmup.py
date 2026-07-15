"""Offline VAST warmup entry point.

Loads HDF5 expert (and optional success/failure) demos for a single task,
encodes them with the SharedDynamicsEncoder, then runs the joint G/V loop
(no Q head). Dumps `vast_state.pt` for the online phase to pick up via
`algorithm.vast.warmup_ckpt`.

Run as a Hydra module (CLI overrides land on `train_dipole_rl.yaml`):
    python -m robosuite.pipeline.algorithms.vast.warmup \
        env.environment=PickPlaceBread \
        runtime.init_checkpoint=checkpoints/.../flow.pt \
        algorithm.vast.warmup_joint_steps=30000 \
        algorithm.vast.config.device=cuda:1 \
        +warmup.output_path=outputs/dipole_rl-vast/baseline/PickPlaceBread/vast_state.pt

`+warmup.output_path` is required (use `+` to add the key — it lives only
in the warmup namespace).

Per-split HDF5 caps: ``warmup.num_trajectories.{expert,success_rollout,fail_rollout}``
(null = all demos in that folder).
"""

from __future__ import annotations

import builtins
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from robosuite.pipeline.algorithms.discriminator.encoder import SharedDynamicsEncoder
from robosuite.pipeline.algorithms.discriminator.nnpu import FrozenNNPUDiscriminator
from robosuite.pipeline.algorithms.flow_dagger.common import (
    FlowAugmentationConfig,
    ReplayBufferConfig,
)
from robosuite.pipeline.algorithms.flow_dagger.replay_buffer import FlowDaggerReplayBuffer
from robosuite.pipeline.algorithms.vast.common import VASTConfig
from robosuite.pipeline.algorithms.vast.vast import VASTLearner
from robosuite.pipeline.algorithms.vast.replay import VASTReplayBuffer
from robosuite.pipeline.envs import build_robosuite_env
from robosuite.pipeline.envs.robosuite import RobosuiteRuntimeConfig
from robosuite.pipeline.train_dipole import (
    bind_flow_proprio_extractor,
    build_flow_runtime_cfg,
    load_hdf5_demos_into_flow_transitions,
    load_init_checkpoint_payload,
    resolve_flow_task_metadata,
)
from robosuite.pipeline.utils import resolve_task_demo_paths
from robosuite.pipeline.utils.io import list_hdf5_demo_names, load_transition_shard
from robosuite.pipeline.utils.train_utils import (
    TensorBoardMetricLogger,
    resolve_camera_names,
    resolve_demo_task_name,
    resolve_requested_device,
    set_seed,
)


print = partial(builtins.print, flush=True)

# Robosuite task dirs use these folder names (see flow_multi/generate_rollout_data.py).
DEFAULT_WARMUP_DEMO_SPLITS: tuple[str, ...] = ("expert", "success_rollout", "fail_rollout")


def _build_warmup_tensorboard(
    cfg: DictConfig,
    *,
    output_path: Path,
) -> TensorBoardMetricLogger | None:
    raw_dir = OmegaConf.select(cfg, "warmup.tensorboard_dir", default=None)
    log_dir = (
        Path(str(raw_dir))
        if raw_dir is not None
        else output_path.parent / "tensorboard"
    )
    if not log_dir.is_absolute():
        log_dir = Path(to_absolute_path(str(log_dir)))
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("[warmup][tensorboard] tensorboard is not installed; skipping.")
        return None
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = TensorBoardMetricLogger(SummaryWriter(log_dir=str(log_dir)), log_dir)
    resolved_cfg = OmegaConf.to_yaml(cfg, resolve=True)
    logger.log_text("run/config_resolved", f"```\n{resolved_cfg}\n```", step=0)
    logger.log_text("run/vast_output_path", str(output_path), step=0)
    logger.log_text(
        "run/vast_algorithm", "vast_value_stitching_adaptation", step=0
    )
    print(f"[warmup][tensorboard] log_dir={log_dir}")
    return logger


@torch.no_grad()
def _vast_batch_debug_metrics(batch: Any, *, prefix: str) -> dict[str, float]:
    rewards = batch.rewards.detach()
    dones = batch.dones.detach()
    actions = batch.action_chunk.detach()
    return {
        f"{prefix}/batch_reward_mean": float(rewards.mean().item()),
        f"{prefix}/batch_reward_std": float(rewards.std(unbiased=False).item()),
        f"{prefix}/batch_reward_min": float(rewards.min().item()),
        f"{prefix}/batch_reward_max": float(rewards.max().item()),
        f"{prefix}/batch_done_ratio": float(dones.mean().item()),
        f"{prefix}/batch_action_abs_mean": float(actions.abs().mean().item()),
        f"{prefix}/batch_action_l2_mean": float(
            actions.flatten(start_dim=1).norm(dim=1).mean().item()
        ),
        f"{prefix}/batch_online_ratio": float(batch.is_online.detach().mean().item()),
        f"{prefix}/batch_intervention_ratio": float(
            batch.is_intervention.detach().mean().item()
        ),
    }


def _prefixed_metrics(metrics: dict[str, float], *, prefix: str) -> dict[str, float]:
    return {f"{prefix}/{key}": float(value) for key, value in metrics.items()}


def _resolve_warmup_num_load_workers(cfg: DictConfig) -> int:
    raw = OmegaConf.select(cfg, "warmup.num_load_workers", default="auto")
    if raw is None or str(raw).strip().lower() == "auto":
        cpu_count = os.cpu_count() or 1
        affinity_count = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else cpu_count
        return max(1, min(affinity_count, 16))
    return max(0, int(raw))


def _resolve_warmup_demo_chunk_size(cfg: DictConfig) -> int:
    return max(1, int(OmegaConf.select(cfg, "warmup.load_worker_demo_chunk_size", default=2)))


def _resolve_warmup_demo_splits(cfg: DictConfig) -> tuple[str, ...]:
    """HDF5 subdirs under ``data/<task_name>/`` to load for offline VAST warmup."""
    raw = OmegaConf.select(cfg, "warmup.demo_splits", default=None)
    if raw is None:
        return DEFAULT_WARMUP_DEMO_SPLITS
    splits = tuple(str(name) for name in list(raw))
    if not splits:
        raise ValueError("warmup.demo_splits must be a non-empty list of directory names.")
    return splits


def _normalize_trajectory_cap(value: Any) -> int | None:
    if value is None:
        return None
    cap = int(value)
    if cap < 0:
        raise ValueError(f"Trajectory cap must be >= 0 or null, got {cap}.")
    return cap


def _resolve_split_max_trajectories(cfg: DictConfig, split: str) -> int | None:
    """Per-split HDF5 demo cap from ``warmup.num_trajectories.<split>``.

    Falls back to legacy ``data.num_trajectories`` for the expert split only.
    """
    limits = OmegaConf.select(cfg, "warmup.num_trajectories", default=None)
    if limits is not None:
        if split in limits:
            return _normalize_trajectory_cap(limits[split])
        base_split = str(split).removesuffix("-labeled")
        if base_split != str(split) and base_split in limits:
            return _normalize_trajectory_cap(limits[base_split])
        return None

    if split == "expert":
        legacy = OmegaConf.select(cfg, "data.num_trajectories", default=None)
        return _normalize_trajectory_cap(legacy)
    return None


def _summarize_gt_fail_labels(transitions: list) -> dict[str, int | bool]:
    gt_fail_frames = 0
    annotated_frames = 0
    episodes_with_gt_fail: set[int] = set()
    episodes_seen: set[int] = set()
    has_gt_fail_key = False
    for idx, trans in enumerate(transitions):
        info = trans.info or {}
        if "gt_fail" not in info:
            continue
        has_gt_fail_key = True
        episode_index = int(info.get("episode_index", idx))
        episodes_seen.add(episode_index)
        if str(info.get("gt_fail_source", "")) != "missing_default_false":
            annotated_frames += 1
        if bool(info.get("gt_fail", False)):
            gt_fail_frames += 1
            episodes_with_gt_fail.add(episode_index)
    return {
        "gt_fail_available": bool(has_gt_fail_key),
        "gt_fail_annotated_frames": int(annotated_frames),
        "gt_fail_positive_frames": int(gt_fail_frames),
        "gt_fail_positive_episodes": int(len(episodes_with_gt_fail)),
        "gt_fail_total_episodes": int(len(episodes_seen)),
    }


def _annotate_episode_metadata(
    transitions: list,
    *,
    buffer_role: str,
    demo_source: str,
    episode_index_base: int,
) -> int:
    """Inject `episode_index` / `episode_step` / `buffer_role` into transition.info
    so the chunk-window valid-start cache can bound chunks to a single demo.

    Returns the next episode_index_base after this batch.
    """
    if not transitions:
        return episode_index_base
    current_index = episode_index_base
    step_in_episode = 0
    for trans in transitions:
        info = dict(trans.info or {})
        info["episode_index"] = int(current_index)
        info["episode_step"] = int(step_in_episode)
        info["episode_namespace"] = demo_source
        info["buffer_role"] = buffer_role
        trans.info = info
        if bool(trans.done):
            current_index += 1
            step_in_episode = 0
        else:
            step_in_episode += 1
    # If the last transition wasn't marked done (defensive), still bump the counter.
    if not bool(transitions[-1].done):
        current_index += 1
    return current_index


def _tag_transitions_with_hdf5_path(
    transitions: list[Any], hdf5_path: Path
) -> None:
    resolved = str(hdf5_path.resolve())
    for trans in transitions:
        info = dict(trans.info or {})
        info["source_hdf5_path"] = resolved
        trans.info = info


def _freeze_post_success_tail(transitions: list[Any]) -> int:
    """Collapse each demo's post-success drift into one frozen absorbing anchor.

    Within every demo (delimited by ``Transition.done``), find the first frame
    whose ``info["success"]`` is True (``t_s``) and overwrite every *later* frame
    with a frozen copy of that ``t_s`` frame: ``obs`` (images + proprio),
    ``next_obs`` and ``action`` all become the ``t_s`` values,
    ``info["success"]`` is pinned True, and ``reward`` is set to ``0.0`` so
    every frozen tail chunk matches the absorbing terminal target. ``done`` is
    left untouched (only the demo's last frame keeps ``done=True``).

    Rationale: the recorded success rollouts keep running the live policy after
    success, so the post-success tail is real *drift* (moving state, non-zero
    actions). Fitting VAST on those many distinct meaningless states is the
    "task burden" we want to drop, while the terminal-value anchor (V≈0 at
    success) is what stabilizes offline VAST TD. Freezing the tail to a single
    ``(s_{t_s}, a_{t_s})`` keeps every frozen frame a valid chunk start (so the
    anchor *sampling density* is preserved) but makes all those chunks encode to
    one identical latent — a clean, dense, unbiased absorbing anchor.

    No-op for demos with no success frame (e.g. ``fail_rollout`` truncations).
    The anchor ``obs`` / ``action`` are shared *by reference* across the tail
    (read-only downstream) to avoid copying hundreds of image frames per demo.
    Returns the number of frozen frames.
    """
    if not transitions:
        return 0
    frozen = 0
    n = len(transitions)
    demo_start = 0
    for idx in range(n):
        if not (bool(transitions[idx].done) or idx == n - 1):
            continue
        demo = transitions[demo_start : idx + 1]
        first_success = next(
            (j for j, t in enumerate(demo) if bool((t.info or {}).get("success", False))),
            None,
        )
        if first_success is not None:
            anchor = demo[first_success]
            anchor_obs = anchor.obs
            anchor_action = anchor.action
            for tail in demo[first_success + 1 :]:
                tail.obs = anchor_obs
                tail.next_obs = anchor_obs
                tail.action = anchor_action
                tail.reward = 0.0
                info = dict(tail.info or {})
                info["success"] = True
                info["frozen_post_success"] = True
                tail.info = info
                frozen += 1
        demo_start = idx + 1
    return frozen


def _select_split_demo_jobs(
    demo_paths: list[Path],
    *,
    max_num_trajectories: int | None,
) -> list[tuple[Path, list[str] | None]]:
    jobs: list[tuple[Path, list[str] | None]] = []
    remaining = max_num_trajectories
    for path in demo_paths:
        if remaining is not None and remaining <= 0:
            break
        suffix = path.suffix.lower()
        if suffix == ".pt":
            if remaining is not None:
                raise ValueError(
                    "Trajectory-limited demo loading only supports HDF5 expert files. "
                    f"Remove max_num_trajectories or convert {path} to HDF5 input."
                )
            jobs.append((path, None))
            continue
        if suffix not in {".hdf5", ".h5"}:
            raise ValueError(f"Unsupported demo file type: {path}. Expected .pt, .hdf5, or .h5.")
        demo_names = list_hdf5_demo_names(path)
        if remaining is not None:
            demo_names = demo_names[:remaining]
        if not demo_names:
            continue
        jobs.append((path, demo_names))
        if remaining is not None:
            remaining -= len(demo_names)
    return jobs


def _chunk_demo_names(demo_names: list[str], chunk_size: int) -> list[list[str]]:
    return [
        demo_names[start : start + int(chunk_size)]
        for start in range(0, len(demo_names), int(chunk_size))
    ]


def _load_hdf5_worker(
    *,
    path: str,
    demo_names: list[str],
    runtime_cfg: RobosuiteRuntimeConfig,
    policy_camera_names: list[str],
    camera_aliases: dict[str, str],
    img_height: int,
    img_width: int,
    proprio_keys: tuple[str, ...],
    renderer: str,
    control_freq: int,
    flow_env_metadata: dict[str, Any] | None,
    reward_mode: str = "-1/0",
    prefer_hdf5_success_labels: bool = False,
    bulk_read_hdf5_images: bool = False,
) -> list[Any]:
    env = build_robosuite_env(runtime_cfg)
    try:
        env.reset()
        extractor = bind_flow_proprio_extractor(env, flow_env_metadata)
        return load_hdf5_demos_into_flow_transitions(
            path,
            policy_camera_names=policy_camera_names,
            camera_aliases=camera_aliases,
            img_height=int(img_height),
            img_width=int(img_width),
            proprio_keys=tuple(proprio_keys),
            renderer=str(renderer),
            control_freq=int(control_freq),
            demo_names=list(demo_names),
            state_extractor=extractor,
            reward_mode=str(reward_mode),
            prefer_hdf5_success_labels=bool(prefer_hdf5_success_labels),
            bulk_read_hdf5_images=bool(bulk_read_hdf5_images),
        )
    finally:
        env.close()


def _load_split_into_buffer(
    *,
    buffer: FlowDaggerReplayBuffer,
    task_name: str,
    data_root: str,
    split: str,
    hdf5_loader,
    runtime_cfg: RobosuiteRuntimeConfig,
    policy_camera_names: list[str],
    camera_aliases: dict[str, str],
    img_height: int,
    img_width: int,
    proprio_keys: tuple[str, ...],
    renderer: str,
    episode_index_base: int,
    max_num_trajectories: int | None,
    control_freq: int,
    num_load_workers: int,
    load_worker_start_method: str,
    load_worker_demo_chunk_size: int,
    flow_env_metadata: dict[str, Any] | None,
    reward_mode: str = "-1/0",
    prefer_hdf5_success_labels: bool = False,
    bulk_read_hdf5_images: bool = False,
) -> tuple[int, int]:
    """Load one split's HDF5 demos into `buffer`. Returns (n_transitions_added,
    new_episode_index_base)."""
    demo_paths = resolve_task_demo_paths(task_name=task_name, data_root=data_root, split=split)
    if not demo_paths:
        print(f"[warmup] split='{split}' resolved to zero demo paths under {data_root}/{task_name}/{split} — skipping.")
        return 0, episode_index_base

    jobs = _select_split_demo_jobs(demo_paths, max_num_trajectories=max_num_trajectories)
    if not jobs:
        print(f"[warmup] split='{split}' selected zero demos — skipping.")
        return 0, episode_index_base

    hdf5_jobs = [
        (path, demo_names)
        for path, demo_names in jobs
        if path.suffix.lower() in {".hdf5", ".h5"} and demo_names
    ]
    hdf5_chunks: list[tuple[Path, int, list[str]]] = []
    total_hdf5_demos = 0
    for path, demo_names in hdf5_jobs:
        total_hdf5_demos += len(demo_names or [])
        for chunk_index, chunk_demo_names in enumerate(
            _chunk_demo_names(list(demo_names or []), load_worker_demo_chunk_size)
        ):
            hdf5_chunks.append((path, chunk_index, chunk_demo_names))
    effective_workers = min(max(0, int(num_load_workers)), len(hdf5_chunks))
    print(
        f"[warmup] split='{split}' loading {len(jobs)} files "
        f"({len(hdf5_jobs)} hdf5, {total_hdf5_demos} demos, "
        f"{len(hdf5_chunks)} chunks, workers={effective_workers})"
    )

    loaded_chunks: dict[tuple[Path, int], list[Any]] = {}
    if effective_workers > 1:
        context = mp.get_context(str(load_worker_start_method))
        with ProcessPoolExecutor(max_workers=effective_workers, mp_context=context) as executor:
            futures = {
                executor.submit(
                    _load_hdf5_worker,
                    path=str(path),
                    demo_names=list(chunk_demo_names),
                    runtime_cfg=runtime_cfg,
                    policy_camera_names=list(policy_camera_names),
                    camera_aliases=dict(camera_aliases),
                    img_height=int(img_height),
                    img_width=int(img_width),
                    proprio_keys=tuple(proprio_keys),
                    renderer=str(renderer),
                    control_freq=int(control_freq),
                    flow_env_metadata=None if flow_env_metadata is None else dict(flow_env_metadata),
                    reward_mode=str(reward_mode),
                    prefer_hdf5_success_labels=bool(prefer_hdf5_success_labels),
                    bulk_read_hdf5_images=bool(bulk_read_hdf5_images),
                ): (path, chunk_index, len(chunk_demo_names))
                for path, chunk_index, chunk_demo_names in hdf5_chunks
            }
            with tqdm(
                total=total_hdf5_demos,
                desc=f"[warmup] load {split}",
                unit="demo",
            ) as progress:
                for future in as_completed(futures):
                    path, chunk_index, num_demos = futures[future]
                    loaded_chunks[(path, chunk_index)] = future.result()
                    progress.update(int(num_demos))

    total_transitions = 0
    next_base = episode_index_base
    serial_load_progress = None
    if effective_workers <= 1 and total_hdf5_demos > 0:
        serial_load_progress = tqdm(
            total=total_hdf5_demos,
            desc=f"[warmup] load {split}",
            unit="demo",
        )
    for raw_path, demo_names in tqdm(
        jobs,
        desc=f"[warmup] finalize {split}",
        unit="file",
    ):
        raw_path = Path(raw_path)
        if raw_path.suffix.lower() == ".pt":
            path_transitions = load_transition_shard(raw_path)
        elif effective_workers > 1:
            path_transitions = []
            for chunk_index, _ in enumerate(
                _chunk_demo_names(list(demo_names or []), load_worker_demo_chunk_size)
            ):
                path_transitions.extend(loaded_chunks[(raw_path, chunk_index)])
        else:
            path_transitions = hdf5_loader(raw_path, demo_names=list(demo_names or []))
            if serial_load_progress is not None and raw_path.suffix.lower() in {".hdf5", ".h5"}:
                serial_load_progress.update(len(demo_names or []))
        if not path_transitions:
            continue
        _tag_transitions_with_hdf5_path(path_transitions, raw_path)
        next_base = _annotate_episode_metadata(
            path_transitions,
            buffer_role="offline",
            demo_source=f"{task_name}/{split}",
            episode_index_base=next_base,
        )
        for trans in path_transitions:
            buffer.add(trans)
        total_transitions += len(path_transitions)
    if serial_load_progress is not None:
        serial_load_progress.close()

    if total_transitions == 0:
        return 0, episode_index_base
    print(
        f"[warmup] split='{split}' loaded {total_transitions} transitions "
        f"({next_base - episode_index_base} episodes)"
    )
    return total_transitions, next_base


def _resolve_policy_action_dim(init_payload: dict[str, Any] | None, env) -> int:
    if init_payload is not None:
        act_mean = init_payload.get("act_mean", None)
        if act_mean is not None:
            arr = np.asarray(act_mean)
            return int(arr.shape[-1])
    action_low, _ = env.action_spec
    return int(np.asarray(action_low).shape[-1])


@hydra.main(version_base="1.2", config_path="../../config", config_name="train_dipole_rl")
def main(cfg: DictConfig) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "VAST/VAST warmup requires CUDA; torch.cuda.is_available() is False."
        )
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    # Seed-controlled reproducibility: identical SEED + same machine/GPU yields
    # the same demo-load order (deterministic buffer assembly) and the same
    # np/torch sampling stream during the warmup loops. cudnn.benchmark / tf32
    # stay enabled for speed, so this is run-to-run reproducible, not bit-exact
    # across hardware.
    seed = int(OmegaConf.select(cfg, "seed", default=42))
    set_seed(seed)
    print(f"[warmup] seed={seed}")

    warmup_cfg = OmegaConf.select(cfg, "warmup", default=None)
    if warmup_cfg is None or warmup_cfg.get("output_path", None) is None:
        raise ValueError(
            "warmup requires +warmup.output_path=<path>; pass it via Hydra CLI override."
        )
    output_path = Path(to_absolute_path(str(warmup_cfg.output_path)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tb_logger = _build_warmup_tensorboard(cfg, output_path=output_path)

    vast_cfg_block = cfg.algorithm.vast
    configured_joint_steps = OmegaConf.select(
        cfg, "algorithm.vast.warmup_joint_steps", default=None
    )
    if configured_joint_steps is None:
        raise ValueError(
            "algorithm.vast.warmup_joint_steps must be set; the legacy "
            "warmup_value_steps/warmup_full_steps parameters were removed."
        )
    total_steps = int(configured_joint_steps)
    if total_steps < 1:
        raise ValueError("algorithm.vast.warmup_joint_steps must be >= 1.")
    batch_size = int(getattr(cfg.algorithm.trainer, "batch_size", 64)) if "trainer" in cfg.algorithm else 64
    batch_size = int(getattr(warmup_cfg, "batch_size", batch_size))

    init_checkpoint, init_payload = load_init_checkpoint_payload(cfg)
    if init_checkpoint is None:
        raise FileNotFoundError(
            "VAST warmup requires runtime.init_checkpoint to be set (used to resolve "
            "policy action_dim and camera layout)."
        )
    print(f"[warmup] init_checkpoint={init_checkpoint}")

    task_name = str(cfg.env.environment)
    task_data_name = resolve_demo_task_name(cfg)
    data_root = to_absolute_path(str(cfg.data.demo_root))

    # Resolve camera layout the way train_dipole.py does: prefer init_checkpoint.
    requested_camera_names = resolve_camera_names(cfg)
    if bool(getattr(cfg.runtime, "use_init_checkpoint_camera_names", True)) and init_payload is not None:
        policy_camera_names = [str(name) for name in init_payload.get("camera_names", [])]
        if len(policy_camera_names) == 0:
            policy_camera_names = list(requested_camera_names)
    else:
        policy_camera_names = list(requested_camera_names)
    camera_aliases: dict[str, str] = {
        str(key): str(value)
        for key, value in dict(OmegaConf.select(cfg, "algorithm.dipole.camera_aliases", default={}) or {}).items()
    }

    # Build env so we can extract proprio from HDF5 states using the same
    # extractor train_dipole.py uses (the dynamics encoder was trained against
    # that exact proprio format).
    flow_env_metadata = resolve_flow_task_metadata(init_payload, task_name)
    # Offline warmup reads images from HDF5 and proprio from flattened states;
    # no sim.render() — skip EGL/offscreen context creation.
    main_runtime_cfg = build_flow_runtime_cfg(
        cfg,
        env_metadata=flow_env_metadata,
        camera_names=policy_camera_names,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        renderer=str(cfg.env.renderer),
    )
    print(f"[warmup] building env env={task_name} cameras={policy_camera_names}")
    discriminator: FrozenNNPUDiscriminator | None = None

    env = build_robosuite_env(main_runtime_cfg)
    try:
        env.reset()
        extractor = bind_flow_proprio_extractor(env, flow_env_metadata)
        policy_action_dim = _resolve_policy_action_dim(init_payload, env)
        print(f"[warmup] policy_action_dim={policy_action_dim}")

        # Build encoder and bind to policy cameras.
        nnpu_ckpt = to_absolute_path(str(cfg.algorithm.discriminator.checkpoint))
        encoder_ckpt_raw = OmegaConf.select(
            cfg, "algorithm.discriminator.encoder_ckpt", default=None
        )
        encoder_ckpt = (
            to_absolute_path(str(encoder_ckpt_raw)) if encoder_ckpt_raw else None
        )
        requested_device = str(vast_cfg_block.config.device)
        default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        device = resolve_requested_device(requested_device, fallback=default_device)
        if str(requested_device).startswith("cuda") and device == "cpu":
            raise RuntimeError(
                f"Requested device '{requested_device}' but torch.cuda.is_available() is False. "
                "Run `nvidia-smi` and fix any driver/library mismatch (reboot after a driver update)."
            )
        camera_to_view: dict[str, str] = {
            str(k): str(v)
            for k, v in dict(OmegaConf.select(cfg, "algorithm.discriminator.camera_to_view", default={}) or {}).items()
        }
        print(f"[warmup] building SharedDynamicsEncoder from {nnpu_ckpt} on {device}")
        encoder = SharedDynamicsEncoder(
            nnpu_ckpt_path=nnpu_ckpt,
            encoder_ckpt=encoder_ckpt,
            device=device,
            camera_to_view=camera_to_view,
        )
        encoder.bind_policy_cameras(policy_camera_names)
        discriminator = FrozenNNPUDiscriminator(
            nnpu_ckpt_path=nnpu_ckpt,
            task_name=str(cfg.algorithm.discriminator.task_name),
            device=device,
            encoder=encoder,
        )

        vast_cfg_dict = OmegaConf.to_container(vast_cfg_block.config, resolve=True)
        vast_cfg_dict["device"] = device
        vast_cfg = VASTConfig(**vast_cfg_dict)

        capacity = int(getattr(cfg.runtime, "demo_buffer_capacity", 200_000))
        buffer = FlowDaggerReplayBuffer(
            config=ReplayBufferConfig(capacity=capacity, batch_size=batch_size),
            name="vast_warmup_offline",
            camera_names=policy_camera_names,
            action_horizon=int(vast_cfg.action_horizon),
            image_size=int(cfg.env.img_height),
            augmentation_config=FlowAugmentationConfig(),
        )

        reward_mode = str(vast_cfg.reward_mode)
        if reward_mode not in {"0/1", "-1/0"}:
            raise ValueError(f"Invalid reward_mode={reward_mode!r}; expected '0/1' or '-1/0'.")
        print(f"[warmup] reward_mode={reward_mode}")
        if reward_mode == "0/1" and float(vast_cfg.disc_reward_coef) != 0.0:
            raise ValueError("disc_reward_coef must be 0.0 when reward_mode is 0/1")
        prefer_hdf5_success_labels = bool(
            OmegaConf.select(cfg, "warmup.prefer_hdf5_success_labels", default=True)
        )
        bulk_read_hdf5_images = bool(
            OmegaConf.select(cfg, "warmup.bulk_read_hdf5_images", default=True)
        )
        print(
            f"[warmup] hdf5 fast path: prefer_success_labels={prefer_hdf5_success_labels} "
            f"bulk_read_images={bulk_read_hdf5_images}"
        )

        def hdf5_loader(path, demo_names=None):
            return load_hdf5_demos_into_flow_transitions(
                path,
                policy_camera_names=policy_camera_names,
                camera_aliases=camera_aliases,
                img_height=int(cfg.env.img_height),
                img_width=int(cfg.env.img_width),
                proprio_keys=tuple(cfg.env.proprio_keys or []),
                renderer=str(cfg.env.renderer),
                control_freq=int(cfg.env.control_freq),
                demo_names=demo_names,
                state_extractor=extractor,
                reward_mode=reward_mode,
                prefer_hdf5_success_labels=prefer_hdf5_success_labels,
                bulk_read_hdf5_images=bulk_read_hdf5_images,
            )

        demo_splits = _resolve_warmup_demo_splits(cfg)
        split_caps = {split: _resolve_split_max_trajectories(cfg, split) for split in demo_splits}
        num_load_workers = _resolve_warmup_num_load_workers(cfg)
        load_worker_demo_chunk_size = _resolve_warmup_demo_chunk_size(cfg)
        load_worker_start_method = str(
            OmegaConf.select(cfg, "warmup.load_worker_start_method", default="spawn")
        )
        if load_worker_start_method not in mp.get_all_start_methods():
            raise ValueError(
                f"warmup.load_worker_start_method={load_worker_start_method!r} is not available. "
                f"Available: {mp.get_all_start_methods()}"
            )
        print(
            f"[warmup] demo_splits={demo_splits} caps={split_caps} "
            f"task={task_data_name} root={data_root} "
            f"num_load_workers={num_load_workers} "
            f"load_worker_demo_chunk_size={load_worker_demo_chunk_size} "
            f"start_method={load_worker_start_method}"
        )

        freeze_post_success = bool(
            OmegaConf.select(cfg, "warmup.freeze_post_success", default=True)
        )
        print(f"[warmup] freeze_post_success={freeze_post_success}")

        print("[warmup] start loading splits... it may takes a few minutes...")
        episode_index_base = 0
        total_loaded = 0
        for split in demo_splits:
            n_loaded, episode_index_base = _load_split_into_buffer(
                buffer=buffer,
                task_name=task_data_name,
                data_root=data_root,
                split=split,
                hdf5_loader=hdf5_loader,
                runtime_cfg=main_runtime_cfg,
                policy_camera_names=policy_camera_names,
                camera_aliases=camera_aliases,
                img_height=int(cfg.env.img_height),
                img_width=int(cfg.env.img_width),
                proprio_keys=tuple(cfg.env.proprio_keys or []),
                renderer=str(cfg.env.renderer),
                episode_index_base=episode_index_base,
                max_num_trajectories=split_caps[split],
                control_freq=int(cfg.env.control_freq),
                num_load_workers=num_load_workers,
                load_worker_start_method=load_worker_start_method,
                load_worker_demo_chunk_size=load_worker_demo_chunk_size,
                flow_env_metadata=flow_env_metadata,
                reward_mode=reward_mode,
                prefer_hdf5_success_labels=prefer_hdf5_success_labels,
                bulk_read_hdf5_images=bulk_read_hdf5_images,
            )
            total_loaded += n_loaded
        if total_loaded == 0:
            raise RuntimeError(
                f"VAST warmup loaded zero transitions for task={task_data_name} under {data_root}. "
                "Check data.demo_root and that at least the 'expert' split exists."
            )
        print(f"[warmup] buffer ready: {len(buffer)} transitions, valid_starts={buffer.num_valid_sequences()}")

    finally:
        env.close()

    # Optionally persist the assembled offline transitions (raw images + proprio +
    # actions + rewards + dones + metadata) so they can be reloaded verbatim into a
    # replay buffer later (e.g. offline DIPOLE under pipeline/offline). Reuses
    # FlowDaggerReplayBuffer.save() (torch.save of the full storage state_dict).
    save_data = bool(OmegaConf.select(cfg, "warmup.num_trajectories.save_data", default=False))
    if save_data:
        save_dir = str(OmegaConf.select(cfg, "warmup.num_trajectories.save_dir", default="offline_data"))
        offline_dir = Path(data_root) / task_data_name / save_dir
        offline_path = offline_dir / "vast_offline_transitions.pt"
        buffer.save(offline_path)
        meta = {
            "task": task_data_name,
            "task_env": task_name,
            "seed": int(seed),
            "demo_splits": list(demo_splits),
            "split_caps": {str(k): split_caps[k] for k in demo_splits},
            "n_transitions": int(len(buffer)),
            "num_valid_sequences": int(buffer.num_valid_sequences()),
            "action_horizon": int(vast_cfg.action_horizon),
            "camera_names": list(policy_camera_names),
            "image_size": int(cfg.env.img_height),
            "img_height": int(cfg.env.img_height),
            "img_width": int(cfg.env.img_width),
            "camera_aliases": dict(camera_aliases),
            "renderer": str(cfg.env.renderer),
            "control_freq": int(cfg.env.control_freq),
            "reward_mode": reward_mode,
            **_summarize_gt_fail_labels(buffer._storage),  # noqa: SLF001
            "source": "vast.warmup",
        }
        meta_path = offline_path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2))
        print(
            f"[warmup] saved offline transitions -> {offline_path} "
            f"({len(buffer)} transitions, valid_starts={buffer.num_valid_sequences()})"
        )

    # Collapse each success demo's post-success drift into one frozen absorbing
    # (s, a) anchor for the VAST critics. Applied AFTER the offline-data save so
    # the persisted `offline_data` keeps the raw drift frames (offline DIPOLE's
    # policy BC must not over-imitate a single repeated success-moment action);
    # only the in-memory buffer the VAST warmup consumes is frozen. done/episode
    # structure is unchanged, so valid-start windows (and thus anchor sampling
    # density) are preserved while every post-success chunk now encodes to one
    # identical latent. No-op for fail/no-success demos.
    if freeze_post_success:
        with buffer._lock:  # noqa: SLF001 — intentional in-place storage edit
            n_frozen = _freeze_post_success_tail(buffer._storage)  # noqa: SLF001
        print(
            f"[warmup] freeze_post_success: collapsed {n_frozen} post-success "
            f"frames into absorbing anchors (offline_data save kept raw)"
        )

    # Build VAST learner and replay sampler. The Token/Group projectors need the
    # encoder's patch-token layout (chunk feature = n_tokens x chunk_token_dim;
    # state visual block = n_tokens x state_visual_token_dim; proprio_dim is the
    # trailing block on the state feature).
    n_tokens = int(encoder.inner_encoder.num_patches)
    proprio_dim = int(encoder.inner_encoder.proprio_emb_dim)
    vast = VASTLearner(
        cfg=vast_cfg,
        state_feature_dim=encoder.state_feature_dim,
        chunk_feature_dim=encoder.chunk_feature_dim,
        action_dim=policy_action_dim,
        n_tokens=n_tokens,
        proprio_dim=proprio_dim,
    )
    replay = VASTReplayBuffer(base_buffer=buffer, cfg=vast_cfg)
    if not replay.ready(batch_size):
        raise RuntimeError(
            f"VAST warmup buffer has too few valid chunks ({buffer.num_valid_sequences()}) "
            f"to fill batch_size={batch_size}."
        )
    if tb_logger is not None:
        tb_logger.log(
            {
                "run/seed": float(seed),
                "run/batch_size": float(batch_size),
                "run/joint_steps": float(total_steps),
                "data/transitions": float(len(buffer)),
                "data/valid_starts": float(buffer.num_valid_sequences()),
                "data/freeze_post_success": float(freeze_post_success),
                "vast/discount": float(vast_cfg.discount),
                "vast/expectile_tau": float(vast_cfg.expectile_tau),
                "vast/v_ensemble_size": float(vast.ensemble_size),
                "vast/v_lr": float(vast_cfg.v_lr),
                "vast/target_polyak": float(vast_cfg.target_polyak),
                "vast/grad_clip_norm": float(vast_cfg.grad_clip_norm),
                "reward/output_reward_coef": float(vast_cfg.output_reward_coef),
                "reward/disc_reward_coef": float(vast_cfg.disc_reward_coef),
                "reward/nnpu_threshold": float(
                    discriminator.threshold if discriminator is not None else float("nan")
                ),
            },
            step=0,
        )
        tb_logger.log_text(
            "run/warmup_paths",
            json.dumps(
                {
                    "output_path": str(output_path),
                    "init_checkpoint": str(init_checkpoint),
                    "nnpu_checkpoint": str(nnpu_ckpt),
                    "task_env": task_name,
                    "task_data": task_data_name,
                    "demo_splits": list(demo_splits),
                    "split_caps": {str(k): split_caps[k] for k in demo_splits},
                    "device": str(device),
                    "preencode_cache": bool(
                        OmegaConf.select(cfg, "warmup.preencode_cache", default=True)
                    ),
                },
                indent=2,
                sort_keys=True,
            ),
            step=0,
        )

    print(
        f"[warmup] disc_reward: frozen nnPU scores on chunk features "
        f"(coef={float(vast_cfg.disc_reward_coef)} "
        f"output_reward_coef={float(vast_cfg.output_reward_coef)} "
        f"threshold={getattr(discriminator, 'threshold', None)} source=checkpoint)"
    )

    # Optional pre-encoded replay cache: encode every valid chunk once up front
    # and sample from cached tensors during the loops, instead of re-running the
    # frozen encoder every step. Results-neutral (frozen encoder + static offline
    # rewards + identical RNG-driven sampling); a pure warmup speedup.
    train_replay: Any = replay
    preencode_cache = bool(OmegaConf.select(cfg, "warmup.preencode_cache", default=True))
    if preencode_cache and total_steps > 0:
        preencode_batch_size = max(
            1,
            int(OmegaConf.select(cfg, "warmup.preencode_batch_size", default=batch_size)),
        )
        cache_device = str(OmegaConf.select(cfg, "warmup.preencode_cache_device", default="cpu"))
        if cache_device.lower() in {"learner", "device"}:
            cache_device = str(device)
        print(
            f"[warmup] preencoding VAST replay cache: valid_starts={buffer.num_valid_sequences()} "
            f"encode_batch={preencode_batch_size} cache_device={cache_device}"
        )
        train_replay = replay.preencode_step_cache(
            encoder=encoder,
            discriminator=discriminator,
            device=device,
            encode_batch_size=preencode_batch_size,
            cache_device=cache_device,
            progress_desc="[warmup] preencode VAST",
        )
        print(f"[warmup] preencoded cache ready: {len(train_replay)} chunks")

    # Action normalization is owned by the frozen dynamics encoder. There is no
    # There is one joint G+V loop; no legacy value/full phases remain.
    loop_name = "VAST joint G+V"
    print(f"[warmup] starting {loop_name} loop for {total_steps} steps (batch={batch_size})")
    for step in range(total_steps):
        batch = train_replay.sample_step_batch(
            batch_size,
            encoder=encoder,
            discriminator=discriminator,
            device=device,
        )
        metrics = vast.update(batch)
        if tb_logger is not None:
            tb_logger.log(
                {
                    **_prefixed_metrics(metrics, prefix="train/vast"),
                    **_vast_batch_debug_metrics(batch, prefix="train/vast"),
                },
                step=step,
            )
        if step % max(1, total_steps // 20) == 0 or step == total_steps - 1:
            print(
                f"[warmup][joint]  step={step:6d} "
                f"v_loss={metrics['v_loss']:.4f} v_mean={metrics['v_mean']:+.3f} "
                f"target_mean={metrics['target_mean']:+.3f} "
                f"td={metrics['td_error_abs_mean']:.4f}"
            )
            if tb_logger is not None:
                tb_logger.flush()

    payload = {
        "vast_state": vast.state_dict(),
        "cfg": asdict(vast_cfg),
        "encoder_meta": {
            "nnpu_checkpoint": nnpu_ckpt,
            "state_feature_dim": int(encoder.state_feature_dim),
            "chunk_feature_dim": int(encoder.chunk_feature_dim),
            "view_names": list(encoder.view_names),
            "policy_camera_names": list(policy_camera_names),
            "image_size": int(cfg.env.img_height),
            "img_height": int(cfg.env.img_height),
            "img_width": int(cfg.env.img_width),
            "camera_aliases": dict(camera_aliases),
            "renderer": str(cfg.env.renderer),
            "control_freq": int(cfg.env.control_freq),
            "task": task_data_name,
            "task_env": task_name,
            "nnpu_task": str(discriminator.task_name),
            "policy_action_dim": int(policy_action_dim),
            "threshold": (
                float(discriminator.threshold) if discriminator is not None else float("nan")
            ),
            "threshold_source": "checkpoint",
            "disc_reward_coef": float(vast_cfg.disc_reward_coef),
            "output_reward_coef": float(vast_cfg.output_reward_coef),
            "disc_reward_source": "FrozenNNPUDiscriminator(-sigmoid(failure_score - threshold))",
            # V-side Token/Group dim-reduction projector layout.
            "n_tokens": int(n_tokens),
            "proprio_dim": int(proprio_dim),
            "state_proj_dim": int(vast_cfg.state_proj_dim),
            "proprio_proj_dim": int(vast_cfg.proprio_proj_dim),
            "proj_activation": str(vast_cfg.proj_activation),
            "v_ensemble_size": int(vast.ensemble_size),
            "expectile_tau": float(vast_cfg.expectile_tau),
            "ensemble_method": (
                "independent_v_mean" if vast_cfg.vast_v_mode == "indep_ensemble" else None
            ),
            "algorithm": "vast_value_stitching_adaptation",
            "vast_v_mode": str(vast_cfg.vast_v_mode),
            "vast_max_k": int(vast_cfg.vast_max_k),
            "vast_comp_coef": float(vast_cfg.vast_comp_coef),
            "vast_sampling_seed": int(vast_cfg.vast_sampling_seed),
            "action_horizon": int(vast_cfg.action_horizon),
        },
        "schema_version": 8 if vast_cfg.vast_v_mode == "indep_ensemble" else 7,
        "algorithm": "vast_value_stitching_adaptation",
    }
    torch.save(payload, output_path)
    print(f"[warmup] wrote VAST state to {output_path}")
    if tb_logger is not None:
        tb_logger.flush()
        tb_logger.close()



if __name__ == "__main__":  # pragma: no cover
    main()
