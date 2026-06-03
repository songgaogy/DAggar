"""Offline Q/V warmup entry point for IQL.

Loads HDF5 expert (and optional success/failure) demos for a single task,
encodes them with the SharedFrozenEncoder, then runs `warmup_value_only`
followed by full IQL `update` steps. Dumps `iql_state.pt` for the online
phase to pick up via `algorithm.q_learning.warmup_ckpt`.

Run as a Hydra module (CLI overrides land on `train_dipole_rl.yaml`):
    python -m robosuite.pipeline.algorithms.q_learning.warmup \
        env.environment=PickPlaceBread \
        runtime.init_checkpoint=checkpoints/.../flow.pt \
        algorithm.q_learning.warmup_value_steps=20000 \
        algorithm.q_learning.warmup_full_steps=5000 \
        algorithm.q_learning.config.device=cuda:1 \
        +warmup.output_path=outputs/DIPOLE_RL/iql_qv_cache/PickPlaceBread/iql_state.pt

`+warmup.output_path` is required (use `+` to add the key — it lives only
in the warmup namespace).

Per-split HDF5 caps: ``warmup.num_trajectories.{expert,success_rollout,fail_rollout}``
(null = all demos in that folder).
"""

from __future__ import annotations

import builtins
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

from robosuite.pipeline.algorithms.discriminator.encoder import SharedFrozenEncoder
from robosuite.pipeline.algorithms.discriminator.lpb_v2_scorer import (
    LPBV2OfflineScorer,
    annotate_transitions_lpb_by_demo,
)
from robosuite.pipeline.algorithms.flow_dagger.common import (
    FlowAugmentationConfig,
    ReplayBufferConfig,
)
from robosuite.pipeline.algorithms.flow_dagger.replay_buffer import FlowDaggerReplayBuffer
from robosuite.pipeline.algorithms.q_learning.common import IQLConfig
from robosuite.pipeline.algorithms.q_learning.iql import IQLLearner
from robosuite.pipeline.algorithms.q_learning.replay import IQLReplayBuffer
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
    resolve_camera_names,
    resolve_demo_task_name,
    resolve_requested_device,
)


print = partial(builtins.print, flush=True)

# Robosuite task dirs use these folder names (see flow_multi/generate_rollout_data.py).
DEFAULT_WARMUP_DEMO_SPLITS: tuple[str, ...] = ("expert", "success_rollout", "fail_rollout")


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
    """HDF5 subdirs under ``data/<task_name>/`` to load for offline IQL warmup."""
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
        return None

    if split == "expert":
        legacy = OmegaConf.select(cfg, "data.num_trajectories", default=None)
        return _normalize_trajectory_cap(legacy)
    return None


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
    lpb_scorer: LPBV2OfflineScorer | None,
    control_freq: int,
    num_load_workers: int,
    load_worker_start_method: str,
    load_worker_demo_chunk_size: int,
    flow_env_metadata: dict[str, Any] | None,
    reward_mode: str = "-1/0",
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
        if lpb_scorer is not None:
            n_demos = annotate_transitions_lpb_by_demo(
                path_transitions,
                lpb_scorer,
                fps=int(control_freq),
            )
            print(
                f"[warmup][lpb] scored {n_demos} demos in {Path(raw_path).name} "
                f"tau={lpb_scorer.tau:.6f}"
            )
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
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    warmup_cfg = OmegaConf.select(cfg, "warmup", default=None)
    if warmup_cfg is None or warmup_cfg.get("output_path", None) is None:
        raise ValueError(
            "warmup requires +warmup.output_path=<path>; pass it via Hydra CLI override."
        )
    output_path = Path(to_absolute_path(str(warmup_cfg.output_path)))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    q_cfg_block = cfg.algorithm.q_learning
    value_steps = int(getattr(q_cfg_block, "warmup_value_steps", 20000))
    full_steps = int(getattr(q_cfg_block, "warmup_full_steps", 5000))
    batch_size = int(getattr(cfg.algorithm.trainer, "batch_size", 64)) if "trainer" in cfg.algorithm else 64
    batch_size = int(getattr(warmup_cfg, "batch_size", batch_size))

    init_checkpoint, init_payload = load_init_checkpoint_payload(cfg)
    if init_checkpoint is None:
        raise FileNotFoundError(
            "IQL warmup requires runtime.init_checkpoint to be set (used to resolve "
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
    # extractor train_dipole.py uses (the LPB encoder was trained against
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
    lpb_scorer: LPBV2OfflineScorer | None = None
    disc_cfg_dict: dict[str, Any] = {}

    env = build_robosuite_env(main_runtime_cfg)
    try:
        env.reset()
        extractor = bind_flow_proprio_extractor(env, flow_env_metadata)
        policy_action_dim = _resolve_policy_action_dim(init_payload, env)
        print(f"[warmup] policy_action_dim={policy_action_dim}")

        # Build encoder and bind to policy cameras.
        bce_ckpt = to_absolute_path(str(cfg.algorithm.discriminator.warm_start_ckpt))
        requested_device = str(q_cfg_block.config.device)
        default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        device = resolve_requested_device(requested_device, fallback=default_device)
        if str(requested_device).startswith("cuda") and device == "cpu":
            raise RuntimeError(
                f"Requested device '{requested_device}' but torch.cuda.is_available() is False. "
                "Run `nvidia-smi` and fix any driver/library mismatch (reboot after a driver update)."
            )
        camera_to_view: dict[str, str] = {
            str(k): str(v)
            for k, v in dict(OmegaConf.select(cfg, "algorithm.dipole.lpb_detector.camera_to_view", default={}) or {}).items()
        }
        print(f"[warmup] building SharedFrozenEncoder from {bce_ckpt} on {device}")
        encoder = SharedFrozenEncoder(
            bce_ckpt_path=bce_ckpt,
            device=device,
            camera_to_view=camera_to_view,
        )
        encoder.bind_policy_cameras(policy_camera_names)

        iql_cfg_dict = OmegaConf.to_container(q_cfg_block.config, resolve=True)
        iql_cfg_dict["device"] = device
        iql_cfg_dict["resnet_pretrained_path"] = to_absolute_path(
            str(iql_cfg_dict["resnet_pretrained_path"])
        )
        iql_cfg = IQLConfig(**iql_cfg_dict)

        capacity = int(getattr(cfg.runtime, "demo_buffer_capacity", 200_000))
        buffer = FlowDaggerReplayBuffer(
            config=ReplayBufferConfig(capacity=capacity, batch_size=batch_size),
            name="iql_warmup_offline",
            camera_names=policy_camera_names,
            action_horizon=int(iql_cfg.action_horizon),
            image_size=int(cfg.env.img_height),
            augmentation_config=FlowAugmentationConfig(),
        )

        reward_mode = str(iql_cfg.reward_mode)
        if reward_mode not in {"0/1", "-1/0"}:
            raise ValueError(f"Invalid reward_mode={reward_mode!r}; expected '0/1' or '-1/0'.")
        print(f"[warmup] reward_mode={reward_mode}")
        if reward_mode == "0/1" and float(iql_cfg.disc_reward_coef) != 0.0:
            raise ValueError("disc_reward_coef must be 0.0 when reward_mode is 0/1")

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

        disc_cfg_dict = OmegaConf.to_container(
            cfg.algorithm.discriminator.config, resolve=True
        )
        if "warm_start_ckpt" in disc_cfg_dict and disc_cfg_dict["warm_start_ckpt"]:
            disc_cfg_dict["warm_start_ckpt"] = to_absolute_path(
                str(disc_cfg_dict["warm_start_ckpt"])
            )
        if float(iql_cfg.disc_reward_coef) != 0.0:
            meta_json = disc_cfg_dict.get("meta_json_path")
            lpb_scorer = LPBV2OfflineScorer(
                bce_ckpt_path=bce_ckpt,
                task_name=str(task_data_name),
                device=str(device),
                batch_size=32,
                meta_json_path=(
                    str(Path(to_absolute_path(str(meta_json))).resolve())
                    if meta_json
                    else None
                ),
            )

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
                lpb_scorer=lpb_scorer,
                control_freq=int(cfg.env.control_freq),
                num_load_workers=num_load_workers,
                load_worker_start_method=load_worker_start_method,
                load_worker_demo_chunk_size=load_worker_demo_chunk_size,
                flow_env_metadata=flow_env_metadata,
                reward_mode=reward_mode,
            )
            total_loaded += n_loaded
        if total_loaded == 0:
            raise RuntimeError(
                f"IQL warmup loaded zero transitions for task={task_data_name} under {data_root}. "
                "Check data.demo_root and that at least the 'expert' split exists."
            )
        print(f"[warmup] buffer ready: {len(buffer)} transitions, valid_starts={buffer.num_valid_sequences()}")

    finally:
        env.close()

    # Build IQL learner and replay sampler. LPB encoder remains discriminator-only.
    proprio_dim = int(np.asarray(buffer._storage[0].obs["state"]).shape[-1])  # noqa: SLF001
    iql = IQLLearner(
        cfg=iql_cfg,
        camera_names=policy_camera_names,
        proprio_dim=proprio_dim,
        action_dim=policy_action_dim,
    )
    replay = IQLReplayBuffer(base_buffer=buffer, cfg=iql_cfg)
    if not replay.ready(batch_size):
        raise RuntimeError(
            f"IQL warmup buffer has too few valid chunks ({buffer.num_valid_sequences()}) "
            f"to fill batch_size={batch_size}."
        )

    lpb_tau = float(lpb_scorer.tau) if lpb_scorer is not None else None
    lpb_tau_source = str(lpb_scorer.tau_source) if lpb_scorer is not None else None
    print(
        f"[warmup] disc_reward: LPB benchmark scores on transition.info "
        f"(coef={float(iql_cfg.disc_reward_coef)} "
        f"output_reward_coef={float(iql_cfg.output_reward_coef)} "
        f"tau={lpb_tau} source={lpb_tau_source})"
    )

    # Warmup loops (r_disc from pre-annotated LPB fields; no OnlineBCEDiscriminator).
    print(f"[warmup] starting value-only loop for {value_steps} steps (batch={batch_size})")
    for step in range(value_steps):
        batch = replay.sample_step_batch(
            batch_size,
            encoder=encoder,
            discriminator=None,
            device=device,
        )
        metrics = iql.warmup_value_only(batch)
        if step % max(1, value_steps // 20) == 0 or step == value_steps - 1:
            print(
                f"[warmup][value]  step={step:6d} "
                f"v_loss={metrics['v_loss']:.4f} v_mean={metrics['v_mean']:+.3f} "
                f"target_mean={metrics['target_mean']:+.3f}"
            )

    print(f"[warmup] starting full IQL update loop for {full_steps} steps")
    for step in range(full_steps):
        batch = replay.sample_step_batch(
            batch_size,
            encoder=encoder,
            discriminator=None,
            device=device,
        )
        metrics = iql.update(batch)
        if step % max(1, full_steps // 20) == 0 or step == full_steps - 1:
            print(
                f"[warmup][full]   step={step:6d} "
                f"q_loss={metrics['q_loss']:.4f} v_loss={metrics['v_loss']:.4f} "
                f"q1={metrics['q1_mean']:+.3f} v={metrics['v_mean']:+.3f} "
                f"td={metrics['td_error_abs_mean']:.4f}"
            )

    payload = {
        "iql_state": iql.state_dict(),
        "cfg": asdict(iql_cfg),
        "encoder_meta": {
            "bce_ckpt": bce_ckpt,
            "view_names": list(encoder.view_names),
            "policy_camera_names": list(policy_camera_names),
            "proprio_dim": int(proprio_dim),
            "resnet_pretrained_path": str(iql.resnet_pretrained_path),
            "task": task_data_name,
            "task_env": task_name,
            "policy_action_dim": int(policy_action_dim),
            "disc_warm_start_ckpt": str(bce_ckpt),
            "meta_json_path": str(disc_cfg_dict.get("meta_json_path", "")),
            "bce_youden_threshold": float(lpb_tau) if lpb_tau is not None else float("nan"),
            "bce_threshold_source": str(lpb_tau_source or ""),
            "disc_reward_coef": float(iql_cfg.disc_reward_coef),
            "output_reward_coef": float(iql_cfg.output_reward_coef),
            "disc_reward_source": "LPBV2OfflineScorer(-sigmoid(failure_score - tau))",
        },
        "schema_version": 2,
    }
    torch.save(payload, output_path)
    print(f"[warmup] wrote IQL state to {output_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
