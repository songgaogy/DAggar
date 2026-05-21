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
"""

from __future__ import annotations

import builtins
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from robosuite.pipeline.algorithms.discriminator.encoder import SharedFrozenEncoder
from robosuite.pipeline.algorithms.discriminator.online_bce import (
    DiscriminatorConfig,
    OnlineBCEDiscriminator,
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
from robosuite.pipeline.train_dipole import (
    bind_flow_proprio_extractor,
    build_flow_runtime_cfg,
    load_hdf5_demos_into_flow_transitions,
    load_init_checkpoint_payload,
    resolve_flow_task_metadata,
)
from robosuite.pipeline.utils import resolve_task_demo_paths
from robosuite.pipeline.utils.io import load_demo_paths
from robosuite.pipeline.utils.train_utils import (
    resolve_camera_names,
    resolve_demo_task_name,
)


print = partial(builtins.print, flush=True)


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


def _load_split_into_buffer(
    *,
    buffer: FlowDaggerReplayBuffer,
    task_name: str,
    data_root: str,
    split: str,
    hdf5_loader,
    cache_dir: Path,
    episode_index_base: int,
    max_num_trajectories: int | None,
) -> tuple[int, int]:
    """Load one split's HDF5 demos into `buffer`. Returns (n_transitions_added,
    new_episode_index_base)."""
    demo_paths = resolve_task_demo_paths(task_name=task_name, data_root=data_root, split=split)
    if not demo_paths:
        print(f"[warmup] split='{split}' resolved to zero demo paths under {data_root}/{task_name}/{split} — skipping.")
        return 0, episode_index_base
    transitions = load_demo_paths(
        demo_paths,
        cache_dir=cache_dir,
        mirror_cache_dir=None,
        hdf5_loader=hdf5_loader,
        max_num_trajectories=max_num_trajectories,
        cache_key=f"iql_warmup_{split}",
    )
    next_base = _annotate_episode_metadata(
        transitions,
        buffer_role="offline",
        demo_source=f"{task_name}/{split}",
        episode_index_base=episode_index_base,
    )
    for trans in transitions:
        buffer.add(trans)
    print(f"[warmup] split='{split}' loaded {len(transitions)} transitions ({next_base - episode_index_base} episodes)")
    return len(transitions), next_base


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
    max_num_trajectories = OmegaConf.select(cfg, "data.num_trajectories", default=None)
    if max_num_trajectories is not None:
        max_num_trajectories = int(max_num_trajectories)
        if max_num_trajectories <= 0:
            max_num_trajectories = None

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
    main_runtime_cfg = build_flow_runtime_cfg(
        cfg,
        env_metadata=flow_env_metadata,
        camera_names=policy_camera_names,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        renderer=str(cfg.env.renderer),
    )
    print(f"[warmup] building env env={task_name} cameras={policy_camera_names}")
    env = build_robosuite_env(main_runtime_cfg)
    try:
        env.reset()
        extractor = bind_flow_proprio_extractor(env, flow_env_metadata)
        policy_action_dim = _resolve_policy_action_dim(init_payload, env)
        print(f"[warmup] policy_action_dim={policy_action_dim}")

        # Build encoder and bind to policy cameras.
        bce_ckpt = to_absolute_path(str(cfg.algorithm.discriminator.warm_start_ckpt))
        device = str(q_cfg_block.config.device)
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

        # Build buffer with action_horizon matching IQL config.
        iql_cfg = IQLConfig(**OmegaConf.to_container(q_cfg_block.config, resolve=True))
        capacity = int(getattr(cfg.runtime, "demo_buffer_capacity", 200_000))
        buffer = FlowDaggerReplayBuffer(
            config=ReplayBufferConfig(capacity=capacity, batch_size=batch_size),
            name="iql_warmup_offline",
            camera_names=policy_camera_names,
            action_horizon=int(iql_cfg.action_horizon),
            image_size=int(cfg.env.img_height),
            augmentation_config=FlowAugmentationConfig(),
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
            )

        cache_dir = output_path.parent / "_demo_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        episode_index_base = 0
        total_loaded = 0
        for split in ("expert", "success", "fail"):
            n_loaded, episode_index_base = _load_split_into_buffer(
                buffer=buffer,
                task_name=task_data_name,
                data_root=data_root,
                split=split,
                hdf5_loader=hdf5_loader,
                cache_dir=cache_dir,
                episode_index_base=episode_index_base,
                max_num_trajectories=max_num_trajectories if split == "expert" else None,
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

    # Build IQL learner and replay sampler.
    iql = IQLLearner(cfg=iql_cfg, context_dim=encoder.context_dim, action_dim=policy_action_dim)
    replay = IQLReplayBuffer(base_buffer=buffer, cfg=iql_cfg)
    if not replay.ready(batch_size):
        raise RuntimeError(
            f"IQL warmup buffer has too few valid chunks ({buffer.num_valid_sequences()}) "
            f"to fill batch_size={batch_size}."
        )

    # Build a FROZEN OnlineBCEDiscriminator so the warmup reward composition
    # (r_total = r_env + disc_reward_coef * r_disc) matches the online phase.
    # The head is warm-started from `algorithm.discriminator.warm_start_ckpt`
    # via DiscriminatorConfig.warm_start_ckpt, kept in eval mode, with all
    # parameters frozen. We NEVER call `disc.update()` here — only
    # `intrinsic_reward(...)` under no_grad inside the replay sampler.
    disc_cfg_dict = OmegaConf.to_container(
        cfg.algorithm.discriminator.config, resolve=True
    )
    if "warm_start_ckpt" in disc_cfg_dict and disc_cfg_dict["warm_start_ckpt"]:
        disc_cfg_dict["warm_start_ckpt"] = to_absolute_path(
            str(disc_cfg_dict["warm_start_ckpt"])
        )
    disc_cfg = DiscriminatorConfig(**disc_cfg_dict)
    if not (str(disc_cfg.device) == str(iql_cfg.device) == str(encoder.device)):
        raise RuntimeError(
            "IQL warmup device mismatch: "
            f"disc_cfg.device={disc_cfg.device} iql_cfg.device={iql_cfg.device} "
            f"encoder.device={encoder.device}"
        )
    discriminator = OnlineBCEDiscriminator(
        cfg=disc_cfg,
        encoder=encoder,
        context_dim=int(encoder.context_dim),
        action_dim=int(policy_action_dim),
        action_horizon=int(iql_cfg.action_horizon),
    )
    for p in discriminator.head.parameters():
        p.requires_grad_(False)
    discriminator.head.eval()
    print(
        f"[warmup] frozen disc ready ({type(discriminator).__name__}); "
        f"warm_start={disc_cfg.warm_start_ckpt} "
        f"disc_reward_coef={float(iql_cfg.disc_reward_coef)}"
    )

    # Warmup loops.
    print(f"[warmup] starting value-only loop for {value_steps} steps (batch={batch_size})")
    for step in range(value_steps):
        batch = replay.sample_step_batch(
            batch_size,
            encoder=encoder,
            discriminator=discriminator,
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
            discriminator=discriminator,
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
            "context_dim": int(encoder.context_dim),
            "view_names": list(encoder.view_names),
            "policy_camera_names": list(policy_camera_names),
            "task": task_data_name,
            "task_env": task_name,
            "policy_action_dim": int(policy_action_dim),
            "disc_warm_start_ckpt": str(disc_cfg.warm_start_ckpt or ""),
            "disc_reward_coef": float(iql_cfg.disc_reward_coef),
        },
        "schema_version": 1,
    }
    torch.save(payload, output_path)
    print(f"[warmup] wrote IQL state to {output_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
