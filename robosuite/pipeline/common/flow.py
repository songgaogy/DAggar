"""Shared flow-policy environment, checkpoint, and demonstration helpers.

These helpers intentionally contain no training-loop orchestration so batch
collection, offline training, warmup, and evaluation can use them without
importing a legacy entrypoint.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from robosuite.pipeline.common.types import Transition
from robosuite.pipeline.common.environment import (
    build_runtime_config_from_env_info,
    make_checkpoint_directory,
    sparse_success_reward,
)
from robosuite.pipeline.utils import (
    checkpoint_path,
    now_readable,
    resolve_camera_names,
    resolve_checkpoint_reference,
    resolve_requested_device,
    set_seed,
)
from robosuite.pipeline.utils.train_utils import reset_robosuite_env
from robosuite.policy.flow_multi_update.utils.env_util import (
    RobosuiteProprioExtractor,
    camera_obs_key,
)


def resolve_base_policy_directory(output_root: Path) -> Path:
    return output_root / "base_policy"


def maybe_set_seed(seed_value: Any) -> None:
    if seed_value is None:
        print("[INFO] Running without a fixed random seed.")
        return
    normalized = str(seed_value).strip().lower()
    if normalized in {"", "none", "null"}:
        print("[INFO] Running without a fixed random seed.")
        return
    set_seed(int(seed_value))


def serialize_seed(seed_value: Any) -> int | None:
    if seed_value is None:
        return None
    normalized = str(seed_value).strip().lower()
    if normalized in {"", "none", "null"}:
        return None
    return int(seed_value)


def format_base_policy_trajectory_tag(max_num_trajectories: int | None) -> str:
    if max_num_trajectories is None:
        return "all"
    return f"{int(max_num_trajectories):05d}"


def resolve_base_policy_checkpoint_path(
    output_root: Path,
    *,
    env_name: str,
    demo_source_name: str,
    max_num_trajectories: int | None,
    pretrain_steps: int,
) -> Path:
    trajectory_tag = format_base_policy_trajectory_tag(max_num_trajectories)
    checkpoint_name = (
        f"{env_name}__{demo_source_name}__traj_{trajectory_tag}__pretrain_{int(pretrain_steps):08d}.pt"
    )
    return resolve_base_policy_directory(output_root) / checkpoint_name


def resolve_run_directory(cfg: DictConfig) -> tuple[str, Path]:
    output_root = Path(to_absolute_path(str(cfg.logging.output_root)))
    explicit_run_name = cfg.logging.run_name
    if explicit_run_name is not None:
        run_name = str(explicit_run_name)
        return run_name, make_checkpoint_directory(output_root, run_name)
    run_name = f"dipole_{cfg.env.environment}_{now_readable()}"
    return run_name, make_checkpoint_directory(output_root, run_name)


def find_latest_resumable_run(output_root: Path, env_name: str) -> Path | None:
    env_prefix = f"dipole_{env_name}_"
    candidates = []
    for latest_checkpoint in output_root.glob("*/checkpoints/latest.pt"):
        run_dir = latest_checkpoint.parent.parent
        if run_dir.name.startswith(env_prefix):
            candidates.append((latest_checkpoint.stat().st_mtime, run_dir))
    if not candidates:
        for latest_checkpoint in output_root.glob("*/checkpoints/latest.pt"):
            candidates.append((latest_checkpoint.stat().st_mtime, latest_checkpoint.parent.parent))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def resume_checkpoint_candidates(cfg: DictConfig, output_root: Path) -> list[Path]:
    candidates: list[Path] = []
    explicit_checkpoint = resolve_checkpoint_reference(cfg.runtime.checkpoint)
    if explicit_checkpoint is not None:
        candidates.append(explicit_checkpoint)
    elif bool(cfg.runtime.resume):
        resumable_run = find_latest_resumable_run(output_root, str(cfg.env.environment))
        if resumable_run is not None:
            latest = checkpoint_path(resumable_run, "latest")
            if latest.exists():
                candidates.append(latest)
            candidates.extend(sorted((resumable_run / "checkpoints").glob("step_*.pt"), reverse=True))
            candidates.extend(sorted((resumable_run / "checkpoints").glob("pretrain_*.pt"), reverse=True))

    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen or not candidate.exists():
            continue
        seen.add(resolved)
        deduped.append(candidate)
    return deduped


def resolve_algorithm_devices(algorithm_cfg: dict[str, Any]) -> tuple[str, str]:
    flow_cfg = algorithm_cfg.setdefault("flow", {})
    learner_requested = flow_cfg.get("device", "cuda:0")
    inference_requested = flow_cfg.get("inference_device", None)
    normalized_learner_request = "cuda:0" if learner_requested is None else str(learner_requested).strip().lower()

    default_learner = "cuda:0"
    learner_device = resolve_requested_device(learner_requested, fallback=default_learner)

    if inference_requested is None or str(inference_requested).lower() == "auto":
        if normalized_learner_request in {"cuda", "cuda:0"} and torch.cuda.device_count() >= 2:
            learner_device = "cuda:1"
            inference_device = "cuda:0"
        else:
            inference_device = learner_device
    else:
        inference_fallback = learner_device
        inference_device = resolve_requested_device(inference_requested, fallback=inference_fallback)

    flow_cfg["device"] = learner_device
    flow_cfg["inference_device"] = inference_device
    return learner_device, inference_device


def format_publish_line(metrics: dict[str, float]) -> str:
    return (
        f"[publish] policy #{int(metrics.get('learner_publish_count', 0.0))} synced "
        f"at learner_update={int(metrics.get('learner_last_published_update', 0.0))}"
    )


def load_init_checkpoint_payload(cfg: DictConfig) -> tuple[Path | None, dict[str, Any] | None]:
    init_checkpoint_cfg = getattr(cfg.runtime, "init_checkpoint", None)
    init_checkpoint = resolve_checkpoint_reference(init_checkpoint_cfg)
    if init_checkpoint is None or not init_checkpoint.exists():
        return None, None
    payload = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
    return init_checkpoint, payload


def resolve_flow_task_metadata(init_payload: dict[str, Any] | None, task_name: str) -> dict[str, Any] | None:
    if not init_payload:
        return None
    task_metadata_map = init_payload.get("task_metadata_map", None)
    if isinstance(task_metadata_map, dict):
        if task_name in task_metadata_map:
            return dict(task_metadata_map[task_name])
        if len(task_metadata_map) == 1:
            return dict(next(iter(task_metadata_map.values())))
    env_metadata = init_payload.get("env_metadata", None)
    if env_metadata is not None:
        return dict(env_metadata)
    return None


def merge_checkpoint_model_config(
    flow_cfg: dict[str, Any],
    init_payload: dict[str, Any] | None,
) -> None:
    """Load checkpoint architecture while preserving run-local static assets."""
    if not init_payload or "model_cfg" not in init_payload:
        return
    configured_model = copy.deepcopy(flow_cfg.get("model", {}))
    checkpoint_model = copy.deepcopy(init_payload["model_cfg"])
    for section, key in (
        ("image_encoder", "pretrained_path"),
        ("language_encoder", "pretrained_name"),
    ):
        configured_section = configured_model.get(section, {})
        if key in configured_section:
            checkpoint_model.setdefault(section, {})[key] = configured_section[key]
    flow_cfg["model"] = checkpoint_model


def bind_flow_proprio_extractor(env, env_metadata: dict[str, Any] | None = None) -> RobosuiteProprioExtractor:
    _ = env_metadata
    extractor = RobosuiteProprioExtractor.__new__(RobosuiteProprioExtractor)
    extractor.env = env
    extractor.sim = env.sim
    extractor._build_robot_joint_indices()
    return extractor


def normalize_policy_observation(
    obs: dict[str, Any],
    *,
    policy_camera_names: list[str],
    camera_aliases: dict[str, str],
) -> dict[str, Any]:
    normalized_obs = {"state": np.asarray(obs["state"], dtype=np.float32)}
    for camera_name in policy_camera_names:
        if camera_name in obs:
            normalized_obs[camera_name] = np.asarray(obs[camera_name], dtype=np.uint8)
            continue
        alias_source = camera_aliases.get(camera_name)
        if alias_source is None or alias_source not in obs:
            raise KeyError(
                f"Unable to resolve observation camera '{camera_name}'. "
                f"Available keys: {sorted(obs.keys())}, aliases: {camera_aliases}"
            )
        normalized_obs[camera_name] = np.asarray(obs[alias_source], dtype=np.uint8)
    return normalized_obs


def _center_crop_resize_image(image: np.ndarray, img_height: int, img_width: int) -> np.ndarray:
    height, width = image.shape[:2]
    crop_size = min(height, width)
    y0 = (height - crop_size) // 2
    x0 = (width - crop_size) // 2
    crop = image[y0 : y0 + crop_size, x0 : x0 + crop_size]
    if crop.shape[0] == img_height and crop.shape[1] == img_width:
        return np.asarray(crop, dtype=np.uint8)
    ys = np.linspace(0, crop_size - 1, img_height).astype(np.int32)
    xs = np.linspace(0, crop_size - 1, img_width).astype(np.int32)
    return np.asarray(crop[ys][:, xs], dtype=np.uint8)


def convert_env_camera_observation(
    raw_obs: dict[str, Any],
    *,
    env,
    extractor: RobosuiteProprioExtractor,
    policy_camera_names: list[str],
    camera_aliases: dict[str, str],
    img_height: int,
    img_width: int,
) -> dict[str, Any]:
    converted_obs: dict[str, Any] = {}
    for camera_name in policy_camera_names:
        source_camera = camera_aliases.get(camera_name, camera_name)
        source_key = camera_obs_key(source_camera)
        if source_key not in raw_obs:
            raise KeyError(
                f"Unable to resolve env camera observation for '{camera_name}'. "
                f"Expected key '{source_key}' in env obs keys {sorted(raw_obs.keys())}."
            )
        converted_obs[camera_name] = _center_crop_resize_image(
            np.asarray(raw_obs[source_key], dtype=np.uint8),
            img_height=img_height,
            img_width=img_width,
        )
    converted_obs["state"] = extractor.extract(env.sim.get_state().flatten()).astype(np.float32)
    return converted_obs


def reset_flow_policy_observation(
    env,
    *,
    preserve_mjviewer: bool,
    extractor: RobosuiteProprioExtractor,
    policy_camera_names: list[str],
    camera_aliases: dict[str, str],
    img_height: int,
    img_width: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_obs, info = reset_robosuite_env(env, preserve_mjviewer=preserve_mjviewer)
    return (
        convert_env_camera_observation(
            raw_obs,
            env=env,
            extractor=extractor,
            policy_camera_names=policy_camera_names,
            camera_aliases=camera_aliases,
            img_height=img_height,
            img_width=img_width,
        ),
        info,
    )


def build_flow_runtime_cfg(
    cfg: DictConfig,
    *,
    env_metadata: dict[str, Any] | None,
    camera_names: list[str],
    has_renderer: bool,
    has_offscreen_renderer: bool,
    use_camera_obs: bool = False,
    renderer: str | None = None,
):
    if env_metadata is None:
        raise ValueError("flow-dagger requires env metadata to build a flow-aligned runtime config.")

    runtime_cfg = build_runtime_config_from_env_info(
        env_metadata,
        camera_names=camera_names,
        img_height=int(cfg.env.img_height),
        img_width=int(cfg.env.img_width),
        proprio_keys=tuple(cfg.env.proprio_keys or []),
        has_renderer=has_renderer,
        renderer=str(cfg.env.renderer) if renderer is None else str(renderer),
        reward_shaping=False,
        control_freq=int(cfg.env.control_freq),
    )
    if cfg.env.horizon is not None:
        runtime_cfg.horizon = int(cfg.env.horizon)
    runtime_cfg.use_camera_obs = bool(use_camera_obs)
    if use_camera_obs:
        runtime_cfg.has_offscreen_renderer = True
    else:
        runtime_cfg.has_offscreen_renderer = bool(has_offscreen_renderer)
    return runtime_cfg


def _resolve_hdf5_demo_group(file_handle: h5py.File | h5py.Group) -> h5py.Group:
    if "demos" in file_handle:
        return file_handle["demos"]
    if "data" in file_handle:
        return file_handle["data"]
    raise KeyError("Expected the demo file to contain either a 'demos' group or a 'data' group.")


def _reset_flow_env_for_demo(
    env,
    state_extractor: RobosuiteProprioExtractor,
    model_xml: str | None,
) -> None:
    if model_xml:
        xml = env.edit_model_xml(model_xml)
        env.reset_from_xml_string(xml)
        env.done = False
        env.timestep = 0
        env.cur_time = 0.0
    else:
        env.reset()
    state_extractor.env = env
    state_extractor.sim = env.sim
    state_extractor._build_robot_joint_indices()


def _sparse_env_step_reward(
    env,
    state: np.ndarray,
    action: np.ndarray,
    reward_mode: str = "-1/0",
) -> tuple[float, bool]:
    """Replay one stored transition and return sparse -1/0 outcome reward from env success."""
    env.sim.set_state_from_flattened(np.asarray(state))
    env.sim.forward()
    env.done = False
    step_output = env.step(np.asarray(action, dtype=np.float32))
    if len(step_output) == 5:
        _, _, _, _, info = step_output
    else:
        _, _, _, info = step_output
    reward, success = sparse_success_reward(
        env, info if isinstance(info, dict) else None, reward_mode=reward_mode
    )
    return float(reward), bool(success)


def _sparse_reward_from_success(success: bool, reward_mode: str) -> float:
    if reward_mode == "0/1":
        return 1.0 if success else 0.0
    if reward_mode == "-1/0":
        return 0.0 if success else -1.0
    raise ValueError(f"Invalid reward mode: {reward_mode}")


def _post_action_success_from_hdf5_labels(
    is_success: np.ndarray,
    *,
    demo_success_attr: bool,
    path: str | Path,
    demo_name: str,
    expected_len: int,
) -> np.ndarray:
    labels = np.asarray(is_success, dtype=np.bool_).reshape(-1)
    if int(labels.shape[0]) != int(expected_len):
        raise ValueError(
            f"{path}:{demo_name} is_success length {labels.shape[0]} != actions length {expected_len}"
        )
    if np.any(labels[:-1] & ~labels[1:]):
        raise ValueError(f"{path}:{demo_name} is_success must be monotonic false->true.")

    post = np.zeros_like(labels, dtype=np.bool_)
    if len(labels) > 1:
        post[:-1] = labels[1:]
    post[-1] = bool(labels[-1] or (demo_success_attr and not labels[:-1].any()))
    return post


def load_hdf5_demos_into_flow_transitions(
    path: str | Path,
    *,
    policy_camera_names: list[str],
    camera_aliases: dict[str, str],
    img_height: int,
    img_width: int,
    proprio_keys: tuple[str, ...],
    renderer: str,
    control_freq: int,
    demo_names: list[str] | None,
    state_extractor: RobosuiteProprioExtractor,
    reward_mode: str = "-1/0",
    prefer_hdf5_success_labels: bool = False,
    bulk_read_hdf5_images: bool = False,
) -> list:
    _ = proprio_keys
    _ = renderer
    _ = control_freq
    env = state_extractor.env
    transitions: list[Transition] = []
    with h5py.File(Path(path), "r") as file_handle:
        demos_group = _resolve_hdf5_demo_group(file_handle)
        available_demo_names = {str(name) for name in demos_group.keys()}
        if demo_names is None:
            selected_demo_names = sorted(available_demo_names)
        else:
            selected_demo_names = [str(name) for name in demo_names if str(name) in available_demo_names]
        for demo_name in selected_demo_names:
            demo_group = demos_group[demo_name]
            states = np.asarray(demo_group["states"])
            actions = np.asarray(demo_group["actions"], dtype=np.float32)
            if len(states) == 0 or len(actions) == 0:
                continue
            demo_success_attr = bool(demo_group.attrs.get("successful", False))
            model_xml = demo_group.attrs.get("model_file", None)
            if isinstance(model_xml, bytes):
                model_xml = model_xml.decode("utf-8")
            model_xml = str(model_xml) if model_xml else None
            hdf5_step_success: np.ndarray | None = None
            if bool(prefer_hdf5_success_labels) and "is_success" in demo_group:
                hdf5_step_success = _post_action_success_from_hdf5_labels(
                    np.asarray(demo_group["is_success"][:], dtype=np.bool_),
                    demo_success_attr=demo_success_attr,
                    path=path,
                    demo_name=str(demo_name),
                    expected_len=len(actions),
                )
            if hdf5_step_success is None:
                _reset_flow_env_for_demo(env, state_extractor, model_xml)
            if "intervention_labels" in demo_group:
                intervention_labels = np.asarray(demo_group["intervention_labels"], dtype=np.bool_)
            else:
                intervention_labels = np.zeros(len(actions), dtype=np.bool_)
            annotations = demo_group.get("annotations", None)
            if annotations is not None and "failure_frame_mask" in annotations:
                gt_fail_labels = np.asarray(annotations["failure_frame_mask"], dtype=np.bool_)
                if int(gt_fail_labels.shape[0]) != int(len(actions)):
                    raise ValueError(
                        f"{path}:{demo_name} annotations/failure_frame_mask length "
                        f"{gt_fail_labels.shape[0]} != actions length {len(actions)}"
                    )
                gt_fail_source = "annotations/failure_frame_mask"
            else:
                gt_fail_labels = np.zeros(len(actions), dtype=np.bool_)
                gt_fail_source = "missing_default_false"
            if annotations is not None and "failure_segment_index" in annotations:
                failure_segment_index = np.asarray(annotations["failure_segment_index"], dtype=np.int32)
                if int(failure_segment_index.shape[0]) != int(len(actions)):
                    raise ValueError(
                        f"{path}:{demo_name} annotations/failure_segment_index length "
                        f"{failure_segment_index.shape[0]} != actions length {len(actions)}"
                    )
            else:
                failure_segment_index = np.full(len(actions), -1, dtype=np.int32)
            obs_group = demo_group["observations"]
            required_hdf5_camera_names = {
                camera_aliases.get(camera_name, camera_name) for camera_name in policy_camera_names
            }
            for camera_name in required_hdf5_camera_names:
                if camera_name not in obs_group:
                    raise KeyError(f"Missing camera '{camera_name}' in {path}:{demo_name}")
            image_arrays = None
            if bool(bulk_read_hdf5_images):
                image_arrays = {
                    camera_name: np.asarray(obs_group[camera_name]["images"][:], dtype=np.uint8)
                    for camera_name in required_hdf5_camera_names
                }

            for step_idx in range(len(actions)):
                next_idx = min(step_idx + 1, len(actions) - 1)
                raw_obs_images = {
                    camera_name: _center_crop_resize_image(
                        image_arrays[camera_name][step_idx]
                        if image_arrays is not None
                        else np.asarray(obs_group[camera_name]["images"][step_idx], dtype=np.uint8),
                        img_height=img_height,
                        img_width=img_width,
                    )
                    for camera_name in required_hdf5_camera_names
                }
                raw_next_obs_images = {
                    camera_name: _center_crop_resize_image(
                        image_arrays[camera_name][next_idx]
                        if image_arrays is not None
                        else np.asarray(obs_group[camera_name]["images"][next_idx], dtype=np.uint8),
                        img_height=img_height,
                        img_width=img_width,
                    )
                    for camera_name in required_hdf5_camera_names
                }
                obs_state = state_extractor.extract(states[step_idx]).astype(np.float32)
                if step_idx + 1 < len(states):
                    next_state = state_extractor.extract(states[step_idx + 1]).astype(np.float32)
                else:
                    next_state = obs_state.copy()
                obs_images = normalize_policy_observation(
                    {**raw_obs_images, "state": obs_state},
                    policy_camera_names=policy_camera_names,
                    camera_aliases=camera_aliases,
                )
                next_obs_images = normalize_policy_observation(
                    {**raw_next_obs_images, "state": next_state},
                    policy_camera_names=policy_camera_names,
                    camera_aliases=camera_aliases,
                )
                is_last_step = step_idx == len(actions) - 1
                if hdf5_step_success is not None:
                    step_success = bool(hdf5_step_success[step_idx])
                    reward = _sparse_reward_from_success(step_success, reward_mode)
                else:
                    reward, step_success = _sparse_env_step_reward(
                        env, states[step_idx], actions[step_idx], reward_mode=reward_mode
                    )
                transitions.append(
                    Transition(
                        obs=obs_images,
                        action=np.asarray(actions[step_idx], dtype=np.float32),
                        reward=float(reward),
                        next_obs=next_obs_images,
                        done=bool(is_last_step),
                        grasp_penalty=None,
                        is_intervention=bool(intervention_labels[step_idx]),
                        info={
                            "success": bool(step_success),
                            "gt_fail": bool(gt_fail_labels[step_idx]),
                            "gt_fail_source": gt_fail_source,
                            "failure_segment_index": int(failure_segment_index[step_idx]),
                            "demo_success_attr": bool(demo_success_attr),
                            "demo_name": str(demo_name),
                            "reward_convention": "sparse_success_-1_0",
                        },
                        reward_source="env_success",
                        demo_source="offline_demo",
                    )
                )
    return transitions


__all__ = [
    "bind_flow_proprio_extractor",
    "build_flow_runtime_cfg",
    "convert_env_camera_observation",
    "find_latest_resumable_run",
    "format_base_policy_trajectory_tag",
    "format_publish_line",
    "load_hdf5_demos_into_flow_transitions",
    "load_init_checkpoint_payload",
    "maybe_set_seed",
    "normalize_policy_observation",
    "reset_flow_policy_observation",
    "resolve_algorithm_devices",
    "resolve_base_policy_checkpoint_path",
    "resolve_base_policy_directory",
    "resolve_camera_names",
    "resolve_flow_task_metadata",
    "resolve_run_directory",
    "resume_checkpoint_candidates",
    "serialize_seed",
]
