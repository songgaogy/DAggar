"""
Detect whether DIPOLE pos/neg branches have diverged or collapsed.

Combines:
1. Static LoRA-adapter metrics (no env required).
2. Functional probes on observations (optional):
   - ``task_scene_cond`` scale and pos/neg condition divergence;
   - instantaneous velocity-field divergence at several ODE timesteps;
   - planned action-chunk divergence across guidance omega values.

Observation sources for the functional probe:
- env resets (``--env-name``), or
- random frames from expert HDF5 demos (``--probe-data``).

Usage::

    # Static only (embedding weights):
    python -m robosuite.pipeline.offline.utils.probe_dipole \
        --checkpoint outputs/dipole_offline/.../checkpoints/step_00014999.pt

    # action probe on expert HDF5 frames:
    python -m robosuite.pipeline.offline.utils.probe_dipole \
        --checkpoint outputs/dipole_offline/.../checkpoints/step_00014999.pt \
        --probe-data ./data/PickPlaceCereal/expert
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

from robosuite.pipeline.algorithms.dipole.models.flow import DipoleFlowPolicy
from robosuite.pipeline.algorithms.flow_dagger.models.flow import _center_crop_resize
from robosuite.pipeline.envs import build_robosuite_env
from robosuite.pipeline.eval_dipole import (
    _build_dipole_policy,
    _load_json,
    _load_resolved_config,
    _reset_env,
    _resolve_eval_device,
    _resolve_init_checkpoint,
    _resolve_run_dir,
)
from robosuite.pipeline.train_dipole import (
    _center_crop_resize_image,
    _reset_flow_env_for_demo,
    _resolve_hdf5_demo_group,
    bind_flow_proprio_extractor,
    build_flow_runtime_cfg,
    convert_env_camera_observation,
    load_init_checkpoint_payload,
    normalize_policy_observation,
    resolve_flow_task_metadata,
)
from robosuite.pipeline.utils import EnvRandomReducer

Hdf5FrameRef = tuple[Path, str, int]


@dataclass(frozen=True)
class ProbeContext:
    policy: DipoleFlowPolicy
    policy_camera_names: list[str]
    camera_aliases: dict[str, str]
    img_height: int
    img_width: int


def _infer_env_name_from_probe_data(probe_data: str) -> str | None:
    path = Path(probe_data)
    parts = path.parts
    if "expert" in parts:
        expert_idx = parts.index("expert")
        if expert_idx >= 1:
            return str(parts[expert_idx - 1])
    return None


def _resolve_env_task_names(
    *,
    env_name: str | None,
    task_name: str | None,
    probe_data: str | None,
    run_info: dict[str, Any] | None,
    checkpoint_payload: dict[str, Any],
) -> tuple[str | None, str | None]:
    resolved_task = task_name or env_name
    if resolved_task is None and run_info is not None:
        resolved_task = run_info.get("task_name") or run_info.get("env_name")
    if resolved_task is None:
        resolved_task = checkpoint_payload.get("task_name")
    if env_name is None and run_info is not None:
        env_name = run_info.get("env_name")
    if env_name is None and probe_data is not None:
        env_name = _infer_env_name_from_probe_data(probe_data)
    if env_name is None:
        env_name = resolved_task
    return (
        str(env_name) if env_name is not None else None,
        str(resolved_task) if resolved_task is not None else None,
    )


def _resolve_hdf5_paths(probe_data: str) -> list[Path]:
    resolved = Path(to_absolute_path(probe_data))
    if resolved.is_file():
        return [resolved]
    if resolved.is_dir():
        files = sorted(resolved.rglob("*.hdf5")) + sorted(resolved.rglob("*.h5"))
        if not files:
            raise FileNotFoundError(f"No .hdf5/.h5 files found under {resolved}")
        return files
    matches = sorted(Path(path) for path in glob.glob(str(resolved), recursive=True))
    files = [path for path in matches if path.is_file() and path.suffix.lower() in {".hdf5", ".h5"}]
    if not files:
        raise FileNotFoundError(f"No HDF5 files matched probe-data pattern: {probe_data}")
    return files


def _index_hdf5_frames(hdf5_paths: list[Path]) -> list[Hdf5FrameRef]:
    frame_index: list[Hdf5FrameRef] = []
    for path in hdf5_paths:
        with h5py.File(path, "r") as file_handle:
            demos_group = _resolve_hdf5_demo_group(file_handle)
            for demo_name in demos_group.keys():
                demo_group = demos_group[demo_name]
                num_steps = int(len(demo_group["actions"]))
                for step_idx in range(num_steps):
                    frame_index.append((path, str(demo_name), step_idx))
    if not frame_index:
        raise RuntimeError("Indexed zero frames from probe-data HDF5 files.")
    return frame_index


def _load_hdf5_frame_obs(
    frame_ref: Hdf5FrameRef,
    *,
    env,
    state_extractor,
    policy_camera_names: list[str],
    camera_aliases: dict[str, str],
    img_height: int,
    img_width: int,
) -> dict[str, Any]:
    path, demo_name, step_idx = frame_ref
    with h5py.File(path, "r") as file_handle:
        demo_group = _resolve_hdf5_demo_group(file_handle)[demo_name]
        states = np.asarray(demo_group["states"])
        obs_group = demo_group["observations"]
        model_xml = demo_group.attrs.get("model_file", None)
        if isinstance(model_xml, bytes):
            model_xml = model_xml.decode("utf-8")
        model_xml = str(model_xml) if model_xml else None

        required_hdf5_camera_names = {
            camera_aliases.get(camera_name, camera_name) for camera_name in policy_camera_names
        }
        for camera_name in required_hdf5_camera_names:
            if camera_name not in obs_group:
                raise KeyError(f"Missing camera '{camera_name}' in {path}:{demo_name}")

        raw_images = {
            camera_name: _center_crop_resize_image(
                np.asarray(obs_group[camera_name]["images"][step_idx], dtype=np.uint8),
                img_height=img_height,
                img_width=img_width,
            )
            for camera_name in required_hdf5_camera_names
        }

    _reset_flow_env_for_demo(env, state_extractor, model_xml)
    obs_state = state_extractor.extract(states[step_idx]).astype(np.float32)
    return normalize_policy_observation(
        {**raw_images, "state": obs_state},
        policy_camera_names=policy_camera_names,
        camera_aliases=camera_aliases,
    )


def _build_probe_context(
    *,
    checkpoint_path: Path,
    env_name: str,
    task_name: str,
    init_checkpoint: Path,
    device: str | None,
    use_camera_obs: bool,
) -> tuple[ProbeContext, Any, Any]:
    init_cfg = OmegaConf.create({"runtime": {"init_checkpoint": str(init_checkpoint)}})
    _, init_payload = load_init_checkpoint_payload(init_cfg)
    env_metadata = resolve_flow_task_metadata(init_payload, task_name)
    if env_metadata is None:
        raise ValueError(f"Could not find env metadata for task '{task_name}'.")

    run_dir = _resolve_run_dir(checkpoint_path)
    resolved_cfg = _load_resolved_config(run_dir)
    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if resolved_cfg is None:
        image_size = int(checkpoint_payload["flow_config"].get("image_size", 128))
        resolved_cfg = OmegaConf.create(
            {
                "env": {
                    "environment": env_name,
                    "renderer": "mjviewer",
                    "img_height": image_size,
                    "img_width": image_size,
                    "proprio_keys": [],
                    "control_freq": 20,
                    "horizon": None,
                }
            }
        )
    resolved_cfg.env.environment = str(env_name)

    policy_camera_names = [str(name) for name in checkpoint_payload["camera_names"]]
    camera_aliases = {
        str(key): str(value)
        for key, value in dict(
            checkpoint_payload.get("flow_config", {}).get("camera_aliases", {}) or {}
        ).items()
    }
    runtime_cfg = build_flow_runtime_cfg(
        resolved_cfg,
        env_metadata=env_metadata,
        camera_names=policy_camera_names,
        has_renderer=False,
        has_offscreen_renderer=bool(use_camera_obs),
        use_camera_obs=bool(use_camera_obs),
        renderer=str(resolved_cfg.env.renderer),
    )
    env = build_robosuite_env(runtime_cfg)
    proprio_extractor = bind_flow_proprio_extractor(env, env_metadata)
    eval_device = _resolve_eval_device(checkpoint_payload, device)
    policy = _build_dipole_policy(
        checkpoint_payload,
        task_name=task_name,
        device=eval_device,
        omega=1.0,
    )
    probe_context = ProbeContext(
        policy=policy,
        policy_camera_names=policy_camera_names,
        camera_aliases=camera_aliases,
        img_height=int(resolved_cfg.env.img_height),
        img_width=int(resolved_cfg.env.img_width),
    )
    return probe_context, env, proprio_extractor


@dataclass(frozen=True)
class EmbeddingMetrics:
    num_lora_layers: int
    lora_B_norm_mean: float
    lora_delta_norm_mean: float
    lora_delta_norm_max: float


@dataclass(frozen=True)
class ConditionMetrics:
    num_obs: int
    cond_dim: int
    task_scene_cond_norm_mean: float
    task_scene_cond_norm_std: float
    polarized_cond_l2_mean: float
    polarized_cond_l2_relative_mean: float
    polarized_cond_cos_mean: float
    context_tokens_norm_mean: float


@dataclass(frozen=True)
class FunctionalMetrics:
    num_obs: int
    condition: ConditionMetrics
    velocity_l2_mean: float
    velocity_l2_std: float
    velocity_relative_l2_mean: float
    velocity_cosine_mean: float
    action_l2_mean_omega0_vs_1: float
    action_relative_l2_mean_omega0_vs_1: float
    action_l2_mean_omega0_vs_2: float
    omega_sensitivity_mean: float


def _load_lora_summary(checkpoint_path: Path) -> EmbeddingMetrics:
    """Aggregate dual-LoRA branch-divergence stats from a checkpoint state dict.

    Each wrapped condition Linear stores ``<path>.pos_lora_A/pos_lora_B`` and
    ``<path>.neg_lora_A/neg_lora_B``. The per-layer delta reported here is the
    Frobenius norm of the positive/negative branch difference
    ``(neg_B @ neg_A) - (pos_B @ pos_A)`` -- what actually drives CFG guidance.
    The ``scaling`` factor lives on the live module, not the state dict, so it is
    omitted here.
    """
    payload: Any = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Checkpoint at {checkpoint_path} is not a dict payload.")
    core = payload.get("core")
    if not isinstance(core, dict):
        raise KeyError(f"Checkpoint {checkpoint_path} has no 'core' field.")
    model_state = core.get("model")
    if not isinstance(model_state, dict):
        raise KeyError(f"Checkpoint {checkpoint_path} is missing 'core.model'.")

    def _delta(prefix: str) -> torch.Tensor | None:
        b = model_state.get(prefix + "_B")
        a = model_state.get(prefix + "_A")
        if not isinstance(b, torch.Tensor) or not isinstance(a, torch.Tensor):
            return None
        return b.detach().float().cpu() @ a.detach().float().cpu()

    neg_b_norms: list[float] = []
    diff_norms: list[float] = []
    for key, value in model_state.items():
        if not key.endswith(".neg_lora_B") or not isinstance(value, torch.Tensor):
            continue
        path = key[: -len(".neg_lora_B")]
        neg_delta = _delta(path + ".neg_lora")
        pos_delta = _delta(path + ".pos_lora")
        if neg_delta is None:
            raise KeyError(f"State dict has '{key}' but is missing its neg_lora_A pair.")
        if pos_delta is None:
            pos_delta = torch.zeros_like(neg_delta)
        neg_b_norms.append(float(torch.linalg.matrix_norm(value.detach().float().cpu()).item()))
        diff_norms.append(float(torch.linalg.matrix_norm(neg_delta - pos_delta).item()))
    if not neg_b_norms:
        raise KeyError("State dict has no '.neg_lora_B' adapter keys (not a dual-LoRA DIPOLE checkpoint).")
    return EmbeddingMetrics(
        num_lora_layers=len(neg_b_norms),
        lora_B_norm_mean=float(sum(neg_b_norms) / len(neg_b_norms)),
        lora_delta_norm_mean=float(sum(diff_norms) / len(diff_norms)),
        lora_delta_norm_max=float(max(diff_norms)),
    )


def _verdict(
    embedding: EmbeddingMetrics,
    functional: FunctionalMetrics | None,
    *,
    delta_good: float,
    delta_warn: float,
    action_rel_warn: float,
) -> str:
    if embedding.lora_delta_norm_max < delta_warn:
        return "COLLAPSED: LoRA adapters never grew — negative branch is still ~identical to base."

    if functional is not None:
        if (
            embedding.lora_delta_norm_mean < delta_good
            and functional.action_relative_l2_mean_omega0_vs_1 < action_rel_warn
        ):
            return (
                "COLLAPSED: LoRA delta tiny and omega=0/1 action chunks are "
                "almost identical — CFG guidance will have little effect."
            )
        if functional.action_relative_l2_mean_omega0_vs_1 < action_rel_warn:
            return (
                "WEAK: LoRA adapters grew but planned actions barely change across omega; "
                "sweeping omega is unlikely to help much."
            )
        if (
            embedding.lora_delta_norm_mean < delta_good
            and functional.velocity_relative_l2_mean < action_rel_warn
        ):
            return (
                "WEAK: branches differ little in velocity field; omega may only weakly steer policy."
            )

    if embedding.lora_delta_norm_mean < delta_good:
        return (
            "WEAK: LoRA delta is small; check functional metrics if env was used."
        )
    return "HEALTHY: LoRA adapters are clearly active; branches are separated."


@torch.inference_mode()
def _obs_to_tensors(
    policy: DipoleFlowPolicy,
    obs: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    images = []
    for camera_name in policy.camera_names:
        image = np.asarray(obs[camera_name], dtype=np.uint8)
        image = _center_crop_resize(image, int(policy.config.image_size))
        images.append(np.transpose(image.astype(np.float32) / 255.0, (2, 0, 1)))
    image_tensor = torch.from_numpy(np.stack(images, axis=0)).unsqueeze(0).to(policy.inference_device)
    image_tensor = (image_tensor - policy._inference_image_mean) / policy._inference_image_std

    proprio = np.asarray(obs["state"], dtype=np.float32)
    if policy.prop_mean is not None and policy.prop_std is not None:
        proprio = (proprio - policy.prop_mean) / (policy.prop_std + 1e-6)
    proprio_tensor = torch.from_numpy(proprio).unsqueeze(0).to(policy.inference_device)
    return image_tensor, proprio_tensor


@torch.inference_mode()
def _probe_condition_metrics(
    policy: DipoleFlowPolicy,
    *,
    image_tensor: torch.Tensor,
    proprio_tensor: torch.Tensor,
    context: dict[str, torch.Tensor] | None = None,
) -> dict[str, float]:
    """Probe encoded condition vectors fed into the flow UNet."""
    model = policy.inference_model
    if context is None:
        context = model.encode_multimodal_context(
            images=image_tensor,
            proprio=proprio_tensor,
            language=[policy.language_instruction],
        )
    # Both branches diverge from base at the aggregator; use each adapter's cond.
    cond_pos = model.branch_task_scene_cond(context, branch="pos")
    cond_neg = model.branch_task_scene_cond(context, branch="neg")
    context_tokens = context["context_tokens"]

    cond_norm = torch.linalg.vector_norm(cond_pos, dim=1)
    polarized_delta = cond_pos - cond_neg
    polarized_l2 = torch.linalg.vector_norm(polarized_delta, dim=1)
    polarized_rel = polarized_l2 / (cond_norm + 1e-8)
    polarized_cos = (cond_pos * cond_neg).sum(dim=1) / (
        torch.linalg.vector_norm(cond_pos, dim=1)
        * torch.linalg.vector_norm(cond_neg, dim=1)
        + 1e-8
    )
    token_norm = torch.linalg.vector_norm(context_tokens, dim=2).mean(dim=1)

    return {
        "cond_dim": float(cond_pos.shape[1]),
        "task_scene_cond_norm": float(cond_norm.mean().item()),
        "polarized_cond_l2": float(polarized_l2.mean().item()),
        "polarized_cond_l2_relative": float(polarized_rel.mean().item()),
        "polarized_cond_cos": float(polarized_cos.mean().item()),
        "context_tokens_norm_mean": float(token_norm.mean().item()),
    }


@torch.inference_mode()
def _probe_velocity_divergence(
    policy: DipoleFlowPolicy,
    *,
    image_tensor: torch.Tensor,
    proprio_tensor: torch.Tensor,
    ode_timesteps: list[float],
    context: dict[str, torch.Tensor] | None = None,
) -> tuple[float, float, float]:
    """Return (l2, relative_l2, cosine) averaged over requested ODE timesteps."""
    model = policy.inference_model
    batch_size = int(proprio_tensor.shape[0])
    action_horizon = int(policy.config.action_horizon)
    action_dim = int(policy.config.action_dim)
    x = torch.zeros(
        batch_size,
        action_dim,
        action_horizon,
        device=proprio_tensor.device,
        dtype=proprio_tensor.dtype,
    )
    if context is None:
        context = model.encode_multimodal_context(
            images=image_tensor,
            proprio=proprio_tensor,
            language=[policy.language_instruction],
        )
    pos_context = dict(context)
    pos_context["task_scene_cond"] = model.branch_task_scene_cond(context, branch="pos")
    neg_context = dict(context)
    neg_context["task_scene_cond"] = model.branch_task_scene_cond(context, branch="neg")

    l2_vals: list[float] = []
    rel_vals: list[float] = []
    cos_vals: list[float] = []
    for frac in ode_timesteps:
        t = torch.full(
            (batch_size,),
            float(frac),
            device=proprio_tensor.device,
            dtype=proprio_tensor.dtype,
        )
        v_pos = model.forward_from_context(x_t=x, t=t, context=pos_context, negative=False)
        v_neg = model.forward_from_context(x_t=x, t=t, context=neg_context, negative=True)
        flat_pos = v_pos.reshape(batch_size, -1)
        flat_neg = v_neg.reshape(batch_size, -1)
        flat_diff = flat_pos - flat_neg
        l2 = torch.linalg.vector_norm(flat_diff, dim=1)
        pos_norm = torch.linalg.vector_norm(flat_pos, dim=1)
        rel = l2 / (pos_norm + 1e-8)
        cos = (flat_pos * flat_neg).sum(dim=1) / (
            torch.linalg.vector_norm(flat_pos, dim=1)
            * torch.linalg.vector_norm(flat_neg, dim=1)
            + 1e-8
        )
        l2_vals.append(float(l2.mean().item()))
        rel_vals.append(float(rel.mean().item()))
        cos_vals.append(float(cos.mean().item()))
    n = float(len(ode_timesteps))
    return sum(l2_vals) / n, sum(rel_vals) / n, sum(cos_vals) / n


def _plan_chunk_with_omega(
    policy: DipoleFlowPolicy,
    obs: dict[str, Any],
    *,
    omega: float,
    deterministic: bool,
) -> np.ndarray:
    old_omega = float(policy.config.guidance_omega)
    policy.config.guidance_omega = float(omega)
    try:
        return policy.plan_action_chunk(obs, deterministic=deterministic)
    finally:
        policy.config.guidance_omega = old_omega


def _action_pair_metrics(chunk_a: np.ndarray, chunk_b: np.ndarray) -> tuple[float, float]:
    flat_a = np.asarray(chunk_a, dtype=np.float32).reshape(-1)
    flat_b = np.asarray(chunk_b, dtype=np.float32).reshape(-1)
    l2 = float(np.linalg.norm(flat_a - flat_b))
    rel = l2 / (float(np.linalg.norm(flat_a)) + 1e-8)
    return l2, rel


def _collect_env_observations(
    *,
    checkpoint_path: Path,
    env_name: str,
    task_name: str,
    init_checkpoint: Path,
    num_obs: int,
    seed: int | None,
    device: str | None,
) -> tuple[list[dict[str, Any]], DipoleFlowPolicy]:
    probe_context, env, proprio_extractor = _build_probe_context(
        checkpoint_path=checkpoint_path,
        env_name=env_name,
        task_name=task_name,
        init_checkpoint=init_checkpoint,
        device=device,
        use_camera_obs=True,
    )

    observations: list[dict[str, Any]] = []
    reducer = EnvRandomReducer(seed)
    try:
        for episode_idx in range(int(num_obs)):
            episode_seed = reducer.prepare_episode(env, int(episode_idx))
            raw_obs, _ = _reset_env(env)
            if episode_seed is not None:
                EnvRandomReducer.seed_global(int(episode_seed))
            observations.append(
                convert_env_camera_observation(
                    raw_obs,
                    env=env,
                    extractor=proprio_extractor,
                    policy_camera_names=probe_context.policy_camera_names,
                    camera_aliases=probe_context.camera_aliases,
                    img_height=probe_context.img_height,
                    img_width=probe_context.img_width,
                )
            )
    finally:
        try:
            env.close()
        finally:
            proprio_extractor.close()
    return observations, probe_context.policy


def _collect_hdf5_observations(
    *,
    checkpoint_path: Path,
    env_name: str,
    task_name: str,
    init_checkpoint: Path,
    probe_data: str,
    num_obs: int,
    seed: int | None,
    device: str | None,
) -> tuple[list[dict[str, Any]], DipoleFlowPolicy, list[Hdf5FrameRef]]:
    hdf5_paths = _resolve_hdf5_paths(probe_data)
    frame_index = _index_hdf5_frames(hdf5_paths)
    rng = random.Random(seed)
    sample_count = min(int(num_obs), len(frame_index))
    sampled_frames = rng.sample(frame_index, k=sample_count)

    probe_context, env, proprio_extractor = _build_probe_context(
        checkpoint_path=checkpoint_path,
        env_name=env_name,
        task_name=task_name,
        init_checkpoint=init_checkpoint,
        device=device,
        use_camera_obs=False,
    )

    observations: list[dict[str, Any]] = []
    try:
        for frame_ref in sampled_frames:
            observations.append(
                _load_hdf5_frame_obs(
                    frame_ref,
                    env=env,
                    state_extractor=proprio_extractor,
                    policy_camera_names=probe_context.policy_camera_names,
                    camera_aliases=probe_context.camera_aliases,
                    img_height=probe_context.img_height,
                    img_width=probe_context.img_width,
                )
            )
    finally:
        try:
            env.close()
        finally:
            proprio_extractor.close()
    return observations, probe_context.policy, sampled_frames


def _functional_metrics(
    policy: DipoleFlowPolicy,
    observations: list[dict[str, Any]],
    *,
    deterministic: bool,
    ode_timesteps: list[float],
) -> FunctionalMetrics:
    vel_l2: list[float] = []
    vel_rel: list[float] = []
    vel_cos: list[float] = []
    action_l2_01: list[float] = []
    action_rel_01: list[float] = []
    action_l2_02: list[float] = []
    omega_sensitivity: list[float] = []
    cond_norms: list[float] = []
    polarized_l2: list[float] = []
    polarized_l2_rel: list[float] = []
    polarized_cos: list[float] = []
    token_norms: list[float] = []
    cond_dim = 0

    model = policy.inference_model
    for obs in observations:
        image_tensor, proprio_tensor = _obs_to_tensors(policy, obs)
        with torch.inference_mode():
            context = model.encode_multimodal_context(
                images=image_tensor,
                proprio=proprio_tensor,
                language=[policy.language_instruction],
            )
        cond_probe = _probe_condition_metrics(
            policy,
            image_tensor=image_tensor,
            proprio_tensor=proprio_tensor,
            context=context,
        )
        cond_dim = int(cond_probe["cond_dim"])
        cond_norms.append(cond_probe["task_scene_cond_norm"])
        polarized_l2.append(cond_probe["polarized_cond_l2"])
        polarized_l2_rel.append(cond_probe["polarized_cond_l2_relative"])
        polarized_cos.append(cond_probe["polarized_cond_cos"])
        token_norms.append(cond_probe["context_tokens_norm_mean"])

        l2, rel, cos = _probe_velocity_divergence(
            policy,
            image_tensor=image_tensor,
            proprio_tensor=proprio_tensor,
            ode_timesteps=ode_timesteps,
            context=context,
        )
        vel_l2.append(l2)
        vel_rel.append(rel)
        vel_cos.append(cos)

        chunk_0 = _plan_chunk_with_omega(policy, obs, omega=0.0, deterministic=deterministic)
        chunk_1 = _plan_chunk_with_omega(policy, obs, omega=1.0, deterministic=deterministic)
        chunk_2 = _plan_chunk_with_omega(policy, obs, omega=2.0, deterministic=deterministic)
        l2_01, rel_01 = _action_pair_metrics(chunk_0, chunk_1)
        l2_02, _ = _action_pair_metrics(chunk_0, chunk_2)
        action_l2_01.append(l2_01)
        action_rel_01.append(rel_01)
        action_l2_02.append(l2_02)
        omega_sensitivity.append(rel_01)

    def _mean_std(values: list[float]) -> tuple[float, float]:
        arr = np.asarray(values, dtype=np.float64)
        return float(arr.mean()), float(arr.std())

    vel_l2_mean, vel_l2_std = _mean_std(vel_l2)
    vel_rel_mean, _ = _mean_std(vel_rel)
    vel_cos_mean, _ = _mean_std(vel_cos)
    act_l2_01_mean, _ = _mean_std(action_l2_01)
    act_rel_01_mean, _ = _mean_std(action_rel_01)
    act_l2_02_mean, _ = _mean_std(action_l2_02)
    omega_sens_mean, _ = _mean_std(omega_sensitivity)
    cond_norm_mean, cond_norm_std = _mean_std(cond_norms)
    polarized_l2_mean, _ = _mean_std(polarized_l2)
    polarized_l2_rel_mean, _ = _mean_std(polarized_l2_rel)
    polarized_cos_mean, _ = _mean_std(polarized_cos)
    token_norm_mean, _ = _mean_std(token_norms)

    condition = ConditionMetrics(
        num_obs=len(observations),
        cond_dim=cond_dim,
        task_scene_cond_norm_mean=cond_norm_mean,
        task_scene_cond_norm_std=cond_norm_std,
        polarized_cond_l2_mean=polarized_l2_mean,
        polarized_cond_l2_relative_mean=polarized_l2_rel_mean,
        polarized_cond_cos_mean=polarized_cos_mean,
        context_tokens_norm_mean=token_norm_mean,
    )

    return FunctionalMetrics(
        num_obs=len(observations),
        condition=condition,
        velocity_l2_mean=vel_l2_mean,
        velocity_l2_std=vel_l2_std,
        velocity_relative_l2_mean=vel_rel_mean,
        velocity_cosine_mean=vel_cos_mean,
        action_l2_mean_omega0_vs_1=act_l2_01_mean,
        action_relative_l2_mean_omega0_vs_1=act_rel_01_mean,
        action_l2_mean_omega0_vs_2=act_l2_02_mean,
        omega_sensitivity_mean=omega_sens_mean,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="DIPOLE checkpoint (.pt).")
    parser.add_argument("--env-name", default=None, help="Robosuite env for env-reset functional probes.")
    parser.add_argument("--task-name", default=None, help="Task name for language instruction / metadata.")
    parser.add_argument(
        "--probe-data",
        default=None,
        help=(
            "Expert HDF5 dataset for functional probes. Accepts a file, directory "
            "(recursively scans *.hdf5/*.h5), or glob. Example: ./data/PickPlaceCereal/expert"
        ),
    )
    parser.add_argument("--init-checkpoint", default=None, help="Base flow checkpoint for env metadata.")
    parser.add_argument("--num-obs", type=int, default=64, help="Observations sampled for functional probes.")
    parser.add_argument("--seed", type=int, default=10086, help="RNG seed for frame / env sampling.")
    parser.add_argument("--device", default=None, help="Eval device override, e.g. cuda:0.")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic ODE noise for action probes (default: deterministic / zero-noise).",
    )
    parser.add_argument(
        "--ode-timesteps",
        default="0.0,0.5,1.0",
        help="Comma-separated ODE time fractions for velocity probes.",
    )
    parser.add_argument("--delta-good", type=float, default=0.1, help="Mean LoRA delta norm above => well diverged.")
    parser.add_argument("--delta-warn", type=float, default=1e-4, help="Max LoRA delta norm below => never trained.")
    parser.add_argument(
        "--action-rel-warn",
        type=float,
        default=0.02,
        help="Relative action delta (omega 0 vs 1) below => functional collapse.",
    )
    parser.add_argument("--json", dest="emit_json", action="store_true", help="Print JSON summary.")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to write JSON report.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit code 2 when verdict starts with COLLAPSED.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    checkpoint_path = Path(to_absolute_path(str(args.checkpoint))).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    embedding = _load_lora_summary(checkpoint_path)
    functional: FunctionalMetrics | None = None
    probe_source: str | None = None
    probe_data_summary: dict[str, Any] | None = None

    run_dir = _resolve_run_dir(checkpoint_path)
    run_info = _load_json(run_dir / "run_info.json") if run_dir is not None else None
    resolved_cfg = _load_resolved_config(run_dir)
    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    init_checkpoint = _resolve_init_checkpoint(
        requested=args.init_checkpoint,
        run_info=run_info,
        resolved_cfg=resolved_cfg,
    )

    env_name, task_name = _resolve_env_task_names(
        env_name=args.env_name,
        task_name=args.task_name,
        probe_data=args.probe_data,
        run_info=run_info,
        checkpoint_payload=checkpoint_payload,
    )

    run_functional = args.probe_data is not None or env_name is not None
    if run_functional:
        if init_checkpoint is None:
            raise FileNotFoundError(
                "Functional probe requires a base flow checkpoint. "
                "Pass --init-checkpoint or evaluate from a run dir with run_info.json."
            )
        if task_name is None:
            raise ValueError(
                "Could not resolve task name. Pass --task-name or --env-name, "
                "or run from a checkpoint with run_info.json."
            )
        if env_name is None:
            env_name = task_name

        ode_timesteps = [float(x.strip()) for x in str(args.ode_timesteps).split(",") if x.strip()]
        if args.probe_data is not None:
            observations, policy, sampled_frames = _collect_hdf5_observations(
                checkpoint_path=checkpoint_path,
                env_name=str(env_name),
                task_name=str(task_name),
                init_checkpoint=init_checkpoint,
                probe_data=str(args.probe_data),
                num_obs=int(args.num_obs),
                seed=args.seed,
                device=args.device,
            )
            probe_source = "hdf5"
            probe_data_summary = {
                "probe_data": str(to_absolute_path(str(args.probe_data))),
                "num_hdf5_files": len(_resolve_hdf5_paths(str(args.probe_data))),
                "num_sampled_frames": len(sampled_frames),
                "sampled_frames": [
                    {"path": str(path), "demo": demo_name, "step": step_idx}
                    for path, demo_name, step_idx in sampled_frames
                ],
            }
        else:
            observations, policy = _collect_env_observations(
                checkpoint_path=checkpoint_path,
                env_name=str(env_name),
                task_name=str(task_name),
                init_checkpoint=init_checkpoint,
                num_obs=int(args.num_obs),
                seed=args.seed,
                device=args.device,
            )
            probe_source = "env_reset"

        functional = _functional_metrics(
            policy,
            observations,
            deterministic=not bool(args.stochastic),
            ode_timesteps=ode_timesteps,
        )

    verdict = _verdict(
        embedding,
        functional,
        delta_good=float(args.delta_good),
        delta_warn=float(args.delta_warn),
        action_rel_warn=float(args.action_rel_warn),
    )

    summary: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "lora": asdict(embedding),
        "functional": None if functional is None else asdict(functional),
        "probe_source": probe_source,
        "probe_data": probe_data_summary,
        "env_name": env_name,
        "task_name": task_name,
        "thresholds": {
            "delta_good": float(args.delta_good),
            "delta_warn": float(args.delta_warn),
            "action_rel_warn": float(args.action_rel_warn),
        },
        "verdict": verdict,
    }

    print(f"[detect_dipole] checkpoint = {checkpoint_path}")
    print("[detect_dipole] --- LoRA adapters (static) ---")
    print(f"[detect_dipole] num_lora_layers     = {embedding.num_lora_layers}")
    print(f"[detect_dipole] mean ||lora_B||      = {embedding.lora_B_norm_mean:.4f}")
    print(f"[detect_dipole] mean ||lora_B@lora_A|| = {embedding.lora_delta_norm_mean:.4f}")
    print(f"[detect_dipole] max  ||lora_B@lora_A|| = {embedding.lora_delta_norm_max:.4f}")

    if functional is not None:
        source_label = "hdf5 frames" if probe_source == "hdf5" else "env reset obs"
        cond = functional.condition
        print(f"[detect_dipole] --- condition ({source_label}) ---")
        if probe_data_summary is not None:
            print(
                f"[detect_dipole] probe_data      = {probe_data_summary['probe_data']} "
                f"({probe_data_summary['num_hdf5_files']} files)"
            )
        print(f"[detect_dipole] num_obs         = {functional.num_obs}")
        print(f"[detect_dipole] cond_dim        = {cond.cond_dim}")
        print(
            f"[detect_dipole] ||task_scene_cond|| = {cond.task_scene_cond_norm_mean:.4f} "
            f"(std={cond.task_scene_cond_norm_std:.4f})"
        )
        print(
            f"[detect_dipole] ||cond_pos - cond_neg|| = {cond.polarized_cond_l2_mean:.4f} "
            f"(rel={cond.polarized_cond_l2_relative_mean:.4f}, cos={cond.polarized_cond_cos_mean:+.4f})"
        )
        print(f"[detect_dipole] ||context_tokens||_mean = {cond.context_tokens_norm_mean:.4f}")
        print(f"[detect_dipole] --- velocity / action ({source_label}) ---")
        print(
            f"[detect_dipole] velocity ||v+ - v-|| = {functional.velocity_l2_mean:.4f} "
            f"(rel={functional.velocity_relative_l2_mean:.4f}, cos={functional.velocity_cosine_mean:+.4f})"
        )
        print(
            f"[detect_dipole] action Δ(ω=0 vs 1)  = {functional.action_l2_mean_omega0_vs_1:.4f} "
            f"(rel={functional.action_relative_l2_mean_omega0_vs_1:.4f})"
        )
        print(
            f"[detect_dipole] action Δ(ω=0 vs 2)  = {functional.action_l2_mean_omega0_vs_2:.4f}"
        )
        print(f"[detect_dipole] omega sensitivity   = {functional.omega_sensitivity_mean:.4f}")
    else:
        print(
            "[detect_dipole] functional probe skipped "
            "(pass --probe-data or --env-name for full check)."
        )

    print(f"[detect_dipole] {verdict}")

    if args.emit_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[detect_dipole] wrote report -> {args.output}")

    if args.strict and verdict.startswith("COLLAPSED"):
        return 2
    if not math.isfinite(embedding.lora_delta_norm_mean):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
