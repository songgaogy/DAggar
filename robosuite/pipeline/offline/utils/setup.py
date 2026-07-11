"""Shared env + agent construction for the offline DIPOLE entries.

Both the main offline run (``train_offline_dipole.build_offline_pipeline``) and
the success-only SFT control (``legacy.train_success_only``) build the exact same
thing before they diverge on data selection / critics:

1. seed + CUDA matmul flags,
2. a headless robosuite env (proprio only, no rendering) + proprio extractor,
3. the DIPOLE agent with the pretrained flow policy loaded and its per-policy
   (positive / negative) trainable-vs-frozen parameter summary printed,
4. after each caller populates the replay buffer: flow normalizers (fit once) +
   a batch-size sanity check.

Those steps used to be copy-pasted between the two entries; they now live here so
the entries only contain what actually differs. The env is returned **open** —
the caller may need the bound extractor to load HDF5 demos — and the caller owns
closing it (typically in a ``finally``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from robosuite.pipeline.common.types import Transition
from robosuite.pipeline.envs import build_robosuite_env
from robosuite.pipeline.factory import build_algorithm
from robosuite.pipeline.train_dipole import (
    bind_flow_proprio_extractor,
    build_flow_runtime_cfg,
    load_hdf5_demos_into_flow_transitions,
    load_init_checkpoint_payload,
    maybe_set_seed,
    resolve_camera_names,
    resolve_flow_task_metadata,
)


def print_policy_param_summary(core: Any, *, log_tag: str = "offline") -> None:
    """Print per-policy trainable-vs-frozen parameter counts after policy init.

    DIPOLE now trains two independent full-tune flow policies (``model_pos`` /
    ``model_neg``), each under the base freeze regime (frozen CLIP + ResNet
    stem/layer1/layer2, trainable layer3/layer4 + heads). ``trainable`` should be
    identical for the two policies and equal to a stock ``build_flow_policy`` model.
    """

    def _counts(model: Any) -> tuple[int, int]:
        trainable = 0
        frozen = 0
        for _, param in model.named_parameters():
            n = int(param.numel())
            if param.requires_grad:
                trainable += n
            else:
                frozen += n
        return trainable, frozen

    for branch, model in (("pos", core.model_pos), ("neg", core.model_neg)):
        trainable, frozen = _counts(model)
        total = trainable + frozen
        print(
            f"[{log_tag}] policy[{branch}] trainable={trainable:,} "
            f"frozen={frozen:,} (all params={total:,})"
        )


@dataclass
class AgentEnvContext:
    """Result of :func:`build_agent_env` — the env is returned still open."""

    agent: Any
    env: Any
    extractor: Any
    policy_camera_names: list[str]
    camera_aliases: dict[str, str]
    img_height: int
    img_width: int
    task_name: str
    init_checkpoint: str
    init_payload: dict[str, Any] | None
    flow_env_metadata: dict[str, Any] | None


def build_agent_env(cfg: DictConfig, *, log_tag: str = "offline") -> AgentEnvContext:
    """Build the headless env + DIPOLE agent with the pretrained flow policy.

    Performs the seed/CUDA setup, constructs a proprio-only robosuite env and its
    extractor, builds the DIPOLE agent from ``cfg.algorithm`` (folding in the
    pretrained checkpoint's model/prompt/action-horizon metadata), loads the flow
    policy weights, and prints the LoRA parameter summary.

    The returned env is **open** so the caller can reuse the bound extractor to
    load HDF5 demos; the caller must close it.
    """
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
            "policy the LoRA adapters attach to) to be set."
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

    print(f"[{log_tag}] init_checkpoint={init_checkpoint}")

    img_height = int(cfg.env.img_height)
    img_width = int(cfg.env.img_width)

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
    print(f"[{log_tag}] loaded pretrained flow policy from {init_checkpoint}")
    print_policy_param_summary(agent.core, log_tag=log_tag)

    return AgentEnvContext(
        agent=agent,
        env=env,
        extractor=extractor,
        policy_camera_names=list(policy_camera_names),
        camera_aliases=camera_aliases,
        img_height=img_height,
        img_width=img_width,
        task_name=task_name,
        init_checkpoint=str(init_checkpoint),
        init_payload=init_payload,
        flow_env_metadata=flow_env_metadata,
    )


def make_hdf5_loader(ctx: AgentEnvContext, cfg: DictConfig) -> Callable[..., list[Transition]]:
    """Build the ``(path, demo_names=None) -> list[Transition]`` HDF5 loader.

    Binds the env-derived camera / image / proprio settings and the extractor
    from ``ctx`` so callers can load expert demos without re-threading them.
    """
    reward_mode = str(cfg.algorithm.q_learning.config.reward_mode)

    def hdf5_loader(path, demo_names=None):
        return load_hdf5_demos_into_flow_transitions(
            path,
            policy_camera_names=ctx.policy_camera_names,
            camera_aliases=ctx.camera_aliases,
            img_height=ctx.img_height,
            img_width=ctx.img_width,
            proprio_keys=tuple(cfg.env.proprio_keys or []),
            renderer=str(cfg.env.renderer),
            control_freq=int(cfg.env.control_freq),
            demo_names=demo_names,
            state_extractor=ctx.extractor,
            reward_mode=reward_mode,
        )

    return hdf5_loader


def finalize_normalizers(
    agent: Any,
    cfg: DictConfig,
    norm_transitions: list[Transition],
    *,
    log_tag: str = "offline",
    norm_desc: str = "pretrain transitions",
) -> int:
    """Fit flow normalizers (if absent) and assert the buffer has enough windows.

    Returns the configured ``batch_size`` so the caller can reuse it.
    """
    if not agent.has_normalizers():
        agent.fit_normalizers_from_transitions(norm_transitions)
        print(f"[{log_tag}] fitted flow normalizers from {norm_desc}.")
    else:
        print(f"[{log_tag}] reusing flow normalizers from the pretrained checkpoint.")

    batch_size = int(cfg.algorithm.trainer.batch_size)
    if agent.online_buffer.num_valid_sequences() < batch_size:
        raise RuntimeError(
            f"replay_buffer has only {agent.online_buffer.num_valid_sequences()} valid "
            f"windows (< batch_size={batch_size}); add more data or lower batch_size."
        )
    return batch_size


__all__ = [
    "AgentEnvContext",
    "build_agent_env",
    "finalize_normalizers",
    "make_hdf5_loader",
    "print_policy_param_summary",
]
