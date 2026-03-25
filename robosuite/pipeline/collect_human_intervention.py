"""
Collect human intervention demos with pluggable policy and discriminator runtimes.
"""

from __future__ import annotations

import datetime
import json
import os
import shutil
import time
from glob import glob
from typing import Any, Optional

import h5py
import hydra
import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.controllers.composite.composite_controller import WholeBody
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper

from robosuite.pipeline.base import OnlineDiscriminatorDecision, load_checkpoint_camera_names
from robosuite.pipeline.factory import build_online_discriminator, build_policy


def now_readable(ts: Optional[datetime.datetime] = None) -> str:
    if ts is None:
        ts = datetime.datetime.now()
    return ts.strftime("%Y-%m-%d_%H-%M-%S")


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def check_intervention(device, input_ac_dict) -> bool:
    if input_ac_dict is None:
        return True

    threshold = 0.1
    total_mag = 0.0
    for key, value in input_ac_dict.items():
        if ("delta" in key) and isinstance(value, np.ndarray):
            total_mag += np.linalg.norm(value)
    return total_mag > threshold


def list_ep_dirs(tmp_root: str) -> list[str]:
    if not os.path.isdir(tmp_root):
        return []
    ep_dirs = [os.path.join(tmp_root, d) for d in os.listdir(tmp_root) if d.startswith("ep_")]
    ep_dirs = [d for d in ep_dirs if os.path.isdir(d)]
    ep_dirs.sort(key=lambda path: os.path.getmtime(path))
    return ep_dirs


def get_current_ep_dir(env_wrapped, tmp_root: str) -> Optional[str]:
    ep_dir = getattr(env_wrapped, "ep_directory", None)
    if isinstance(ep_dir, str) and len(ep_dir) > 0:
        return ep_dir
    ep_dirs = list_ep_dirs(tmp_root)
    return ep_dirs[-1] if len(ep_dirs) > 0 else None


def read_single_demo_from_ep(ep_dir: str):
    state_paths = os.path.join(ep_dir, "state_*.npz")
    states = []
    actions = []
    success = False
    env_name = None

    for state_file in sorted(glob(state_paths)):
        payload = np.load(state_file, allow_pickle=True)
        env_name = str(payload["env"])
        states.extend(payload["states"])
        for action_info in payload["action_infos"]:
            actions.append(action_info["actions"])
        success = success or bool(payload["successful"])

    if len(states) == 0:
        return None

    del states[-1]

    return {
        "env_name": env_name,
        "states": np.asarray(states),
        "actions": np.asarray(actions),
        "success": bool(success),
        "ep_dir": ep_dir,
    }


def _serialize_attr_value(value: Any):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def append_demo_to_hdf5(
    hdf5_path: str,
    demo_id: int,
    demo_payload: dict,
    step_labels: list[str],
    env_info: str,
    camera_names: list[str],
    images_dict: dict[str, np.ndarray],
    trace_arrays: Optional[dict[str, np.ndarray]] = None,
    pipeline_metadata: Optional[dict[str, Any]] = None,
) -> None:
    file_handle = h5py.File(hdf5_path, "a")

    if "demos" not in file_handle:
        demos_group = file_handle.create_group("demos")
        file_handle.attrs["created_at"] = now_readable()
        file_handle.attrs["repository_version"] = suite.__version__
        file_handle.attrs["env"] = demo_payload.get("env_name", "")
        file_handle.attrs["env_info"] = env_info
        file_handle.attrs["camera_names"] = json.dumps(list(camera_names))
    else:
        demos_group = file_handle["demos"]

    demo_group = demos_group.create_group(f"demo_{demo_id:06d}")
    xml_path = os.path.join(demo_payload["ep_dir"], "model.xml")
    if os.path.isfile(xml_path):
        with open(xml_path, "r", encoding="utf-8") as file_obj:
            demo_group.attrs["model_file"] = file_obj.read()

    states = demo_payload["states"]
    actions = demo_payload["actions"]
    total_steps = min(states.shape[0], actions.shape[0], len(step_labels))
    for camera_name in camera_names:
        images = images_dict.get(camera_name, None)
        if isinstance(images, np.ndarray) and images.ndim == 4:
            total_steps = min(total_steps, images.shape[0])

    resolved_labels = list(step_labels[:total_steps])
    cooldown_like_labels = {"COOLDOWN", "WAITING_HUMAN", "WAITING_POLICY"}
    for idx in range(len(resolved_labels)):
        if resolved_labels[idx] not in cooldown_like_labels:
            continue
        resolves_to = "END"
        for next_idx in range(idx + 1, len(resolved_labels)):
            if resolved_labels[next_idx] not in cooldown_like_labels:
                resolves_to = resolved_labels[next_idx]
                break
        if resolves_to != "POLICY":
            resolved_labels[idx] = "INTERVENING"

    keep_indices = [idx for idx in range(total_steps) if resolved_labels[idx] not in cooldown_like_labels]
    states_filtered = states[keep_indices]
    actions_filtered = actions[keep_indices]
    labels_binary = np.asarray(
        [1 if resolved_labels[idx] == "INTERVENING" else 0 for idx in keep_indices],
        dtype=np.uint8,
    )

    segments = []
    in_segment = False
    start_idx = 0
    for idx, label in enumerate(labels_binary.tolist()):
        if label == 1 and not in_segment:
            start_idx = idx
            in_segment = True
        elif label == 0 and in_segment:
            segments.append([start_idx, idx - 1])
            in_segment = False
    if in_segment:
        segments.append([start_idx, len(labels_binary) - 1])

    demo_group.attrs["length"] = int(len(keep_indices))
    demo_group.attrs["successful"] = bool(demo_payload.get("success", False))
    demo_group.attrs["intervention_segments"] = json.dumps(segments)
    demo_group.create_dataset("states", data=states_filtered)
    demo_group.create_dataset("actions", data=actions_filtered)
    demo_group.create_dataset("intervention_labels", data=labels_binary)

    observations_group = demo_group.create_group("observations")
    for camera_name in camera_names:
        camera_group = observations_group.create_group(camera_name)
        images = images_dict.get(camera_name, None)
        if images is None or not isinstance(images, np.ndarray) or images.ndim != 4 or images.shape[0] == 0:
            camera_group.create_dataset("images", data=np.zeros((0,), dtype=np.uint8))
            continue
        images_filtered = images[keep_indices].astype(np.uint8)
        camera_group.create_dataset(
            "images",
            data=images_filtered,
            dtype=np.uint8,
            compression="gzip",
            compression_opts=4,
            chunks=True,
        )

    if trace_arrays is not None or pipeline_metadata is not None:
        pipeline_group = demo_group.create_group("pipeline")
        if pipeline_metadata is not None:
            for key, value in pipeline_metadata.items():
                pipeline_group.attrs[str(key)] = _serialize_attr_value(value)
        if trace_arrays is not None:
            discriminator_group = pipeline_group.create_group("discriminator")
            for key, values in trace_arrays.items():
                values_np = np.asarray(values)
                if values_np.shape[0] < total_steps:
                    continue
                discriminator_group.create_dataset(key, data=values_np[:total_steps][keep_indices])

    file_handle.close()


def prompt_user_action() -> str:
    while True:
        answer = input("Action for this demo: [s]ave / [d]elete / [q]uit / [f]inish ? ").strip().lower()
        if answer in ("s", "d", "q", "f"):
            return answer
        print("Invalid input. Please enter s, d, f or q.")


def build_env(env_config: dict, cfg, camera_names: list[str]):
    env = suite.make(
        **env_config,
        has_renderer=True,
        renderer=cfg.renderer,
        has_offscreen_renderer=True,
        render_camera=camera_names[0],
        camera_names=camera_names,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )
    return VisualizationWrapper(env)


def build_device(env, cfg):
    if cfg.device == "keyboard":
        from robosuite.devices import Keyboard

        return Keyboard(env=env, pos_sensitivity=cfg.pos_sensitivity, rot_sensitivity=cfg.rot_sensitivity)
    if cfg.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        return SpaceMouse(
            env=env,
            vendor_id=0x256F,
            product_id=0xC635,
            pos_sensitivity=cfg.pos_sensitivity,
            rot_sensitivity=cfg.rot_sensitivity,
        )
    if cfg.device == "dualsense":
        from robosuite.devices import DualSense

        return DualSense(
            env=env,
            pos_sensitivity=cfg.pos_sensitivity,
            rot_sensitivity=cfg.rot_sensitivity,
            reverse_xy=cfg.reverse_xy,
        )
    if cfg.device == "mjgui":
        assert cfg.renderer == "mjviewer", "Mocap is only supported with the mjviewer renderer"
        from robosuite.devices.mjgui import MJGUI

        return MJGUI(env=env)
    raise ValueError(f"Invalid device choice: {cfg.device}")


def resolve_camera_names(cfg: DictConfig) -> list[str]:
    requested = [str(name) for name in list(getattr(cfg, "camera", []) or [])]
    required = []
    if str(cfg.policy.type) == "flow_multi":
        required.extend(load_checkpoint_camera_names(str(cfg.policy.ckpt)))
    if bool(cfg.discriminator.enabled):
        required.extend(load_checkpoint_camera_names(str(cfg.discriminator.policy.ckpt)))

    merged = []
    for camera_name in required + requested:
        if camera_name and camera_name not in merged:
            merged.append(camera_name)
    if len(merged) == 0:
        merged = ["agentview"]
    return merged


def _empty_trace() -> dict[str, list]:
    return {
        "score": [],
        "threshold": [],
        "prediction": [],
        "raw_step_score": [],
        "available_steps": [],
        "evaluated": [],
    }


def _append_trace(trace: dict[str, list], decision: OnlineDiscriminatorDecision) -> None:
    trace["score"].append(float(decision.score))
    trace["threshold"].append(float(decision.threshold))
    trace["prediction"].append(int(decision.prediction))
    trace["raw_step_score"].append(float(decision.raw_step_score))
    trace["available_steps"].append(int(decision.available_steps))
    trace["evaluated"].append(int(bool(decision.evaluated)))


def _build_idle_action(env) -> np.ndarray:
    return np.zeros_like(np.asarray(env.action_spec[0]), dtype=np.float32)


def collect_intervention_trajectory(
    env,
    device,
    policy,
    discriminator,
    task_name: str,
    arm: str,
    max_fr: Optional[float],
    goal_update_mode: str,
    camera_names: list[str],
    img_height: int,
    img_width: int,
    cool_down: float,
    alert_cooldown_steps: int,
):
    del arm, cool_down
    env.render()
    task_completion_hold_count = -1
    device.start_control()

    for robot in env.robots:
        robot.print_action_info_dict()

    all_prev_gripper_actions = [
        {
            f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
            for robot_arm in robot.arms
            if robot.gripper[robot_arm].dof > 0
        }
        for robot in env.robots
    ]

    images = {camera_name: [] for camera_name in camera_names}
    step_labels: list[str] = []
    discriminator_trace = _empty_trace()
    state_mode = "POLICY"
    prev_state_mode = "POLICY"
    last_alert_step = -int(alert_cooldown_steps)
    alert_count = 0
    recovery_count = 0
    waiting_for_human = False
    human_intervened_after_alert = False
    idle_action = _build_idle_action(env)

    obs = env.unwrapped._get_observations()
    initial_state = env.sim.get_state().flatten().copy()
    initial_images = {}
    for camera_name in camera_names:
        frame = env.sim.render(height=img_height, width=img_width, camera_name=camera_name)
        initial_images[camera_name] = frame
        images[camera_name].append(frame)

    policy.reset(obs, initial_images, env.unwrapped)
    discriminator.reset(task_name=task_name, initial_state=initial_state, initial_images=initial_images)
    print("\n[INFO] Loop started. Default: Policy. Intervene with Device.")

    while True:
        start = time.time()
        active_robot = env.robots[device.active_robot]
        input_ac_dict = device.input2action(goal_update_mode=goal_update_mode)
        if input_ac_dict is None:
            break

        is_intervening_now = check_intervention(device, input_ac_dict)
        if waiting_for_human:
            if is_intervening_now:
                state_mode = "INTERVENING"
                human_intervened_after_alert = True
            else:
                state_mode = "WAITING_POLICY" if human_intervened_after_alert else "WAITING_HUMAN"
        else:
            state_mode = "INTERVENING" if is_intervening_now else "POLICY"

        if state_mode == "INTERVENING" and prev_state_mode != "INTERVENING":
            policy.notify_intervention()

        if state_mode != prev_state_mode:
            if state_mode in {"INTERVENING", "POLICY", "WAITING_HUMAN", "WAITING_POLICY"}:
                print(f"[State] {state_mode}")
            prev_state_mode = state_mode

        step_labels.append(state_mode)

        if state_mode == "INTERVENING":
            from copy import deepcopy

            action_dict = deepcopy(input_ac_dict)
            for arm_name in active_robot.arms:
                if isinstance(active_robot.composite_controller, WholeBody):
                    controller_input_type = active_robot.composite_controller.joint_action_policy.input_type
                else:
                    controller_input_type = active_robot.part_controllers[arm_name].input_type

                if controller_input_type == "delta":
                    action_dict[arm_name] = input_ac_dict[f"{arm_name}_delta"]
                elif controller_input_type == "absolute":
                    action_dict[arm_name] = input_ac_dict[f"{arm_name}_abs"]
                else:
                    raise ValueError(f"Unsupported controller input type: {controller_input_type}")

            env_action_list = [
                robot.create_action_vector(all_prev_gripper_actions[idx])
                for idx, robot in enumerate(env.robots)
            ]
            env_action_list[device.active_robot] = active_robot.create_action_vector(action_dict)
            env_action = np.concatenate(env_action_list)

            for gripper_action in all_prev_gripper_actions[device.active_robot]:
                all_prev_gripper_actions[device.active_robot][gripper_action] = action_dict[gripper_action]
        elif state_mode in {"WAITING_HUMAN", "WAITING_POLICY"}:
            env_action = idle_action.copy()
        else:
            env_action = policy.get_action()

        obs, reward, done, info = env.step(env_action)
        del reward, info

        step_images = {}
        for camera_name in camera_names:
            frame = env.sim.render(height=img_height, width=img_width, camera_name=camera_name)
            step_images[camera_name] = frame
            images[camera_name].append(frame)

        next_state = env.sim.get_state().flatten().copy()
        policy.update_history(obs, step_images, env.unwrapped)
        decision = discriminator.record_step(
            action=env_action,
            next_state=next_state,
            next_images=step_images,
        )
        _append_trace(discriminator_trace, decision)

        step_index = len(step_labels) - 1
        if (
            state_mode == "POLICY"
            and decision.prediction == 1
            and decision.available_steps > 0
            and (step_index - last_alert_step) >= alert_cooldown_steps
        ):
            alert_count += 1
            last_alert_step = step_index
            waiting_for_human = True
            human_intervened_after_alert = False
            print(
                "[Discriminator] "
                f"name={decision.metadata.get('detector_name', discriminator.name)} "
                f"score={decision.score:.4f} threshold={decision.threshold:.4f} "
                f"pred={decision.prediction} prefix_steps={decision.available_steps} "
                "-> waiting for human intervention"
            )
            policy.notify_intervention()
        elif (
            state_mode == "WAITING_POLICY"
            and human_intervened_after_alert
            and decision.available_steps > 0
            and decision.prediction == 0
        ):
            waiting_for_human = False
            human_intervened_after_alert = False
            recovery_count += 1
            print(
                "[Discriminator] "
                f"name={decision.metadata.get('detector_name', discriminator.name)} "
                f"score={decision.score:.4f} threshold={decision.threshold:.4f} "
                "-> back to in-domain, resume policy rollout"
            )
            policy.notify_intervention()

        env.render()

        if task_completion_hold_count == 0:
            break

        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1
            else:
                task_completion_hold_count = 10
        else:
            task_completion_hold_count = -1

        if done:
            break

        if max_fr is not None:
            elapsed = time.time() - start
            sleep_time = 1.0 / max_fr - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    images_np = {}
    for camera_name in camera_names:
        if len(images[camera_name]) == 0:
            images_np[camera_name] = np.zeros((0,), dtype=np.uint8)
        else:
            images_np[camera_name] = np.stack(images[camera_name], axis=0).astype(np.uint8)

    trace_arrays = {
        key: np.asarray(values)
        for key, values in discriminator_trace.items()
    }
    pipeline_metadata = {
        "policy_name": str(cfg_safe_name(policy)),
        "discriminator_name": str(discriminator.name),
        "task_name": str(task_name),
        "alert_count": int(alert_count),
        "recovery_count": int(recovery_count),
        "camera_names": list(camera_names),
    }
    return images_np, step_labels, trace_arrays, pipeline_metadata


def cfg_safe_name(runtime) -> str:
    return getattr(runtime, "__class__", type(runtime)).__name__


@hydra.main(version_base="1.2", config_path="./config", config_name="collect_human_intervention")
def main(cfg: DictConfig):
    camera_names = resolve_camera_names(cfg)
    intervention_dir = to_absolute_path(str(cfg.intervention.directory))
    safe_mkdir(intervention_dir)
    base_time = now_readable()
    tmp_root = os.path.join("/tmp", f"robosuite_pipeline_intervention_{base_time}")
    safe_mkdir(tmp_root)
    tmp_hdf5_path = os.path.join(intervention_dir, f"intervention_{base_time}_0.hdf5")
    print(f"Output HDF5: {tmp_hdf5_path}")
    print(f"Runtime cameras: {camera_names}")

    controller_config = load_composite_controller_config(
        controller=cfg.intervention.controller,
        robot=cfg.intervention.robots[0],
    )
    if controller_config["type"] == "WHOLE_BODY_MINK_IK":
        from robosuite.examples.third_party_controller.mink_controller import WholeBodyMinkIK  # noqa: F401
    if controller_config["type"] == "WHOLE_BODY_IK":
        assert len(cfg.intervention.robots) == 1, "Whole Body IK only supports one robot"

    env_config = {
        "env_name": cfg.intervention.environment,
        "robots": list(cfg.intervention.robots),
        "controller_configs": controller_config,
    }
    if "TwoArm" in cfg.intervention.environment:
        env_config["env_configuration"] = cfg.intervention.config
    env_info = json.dumps(env_config)

    class DummyArgs:
        pass

    args = DummyArgs()
    for key, value in cfg.intervention.items():
        setattr(args, key, value)

    env = build_env(env_config, args, camera_names=camera_names)
    device = build_device(env, args)
    policy = build_policy(cfg.policy, env_name=str(cfg.intervention.environment), env=env.unwrapped)
    discriminator = build_online_discriminator(cfg.discriminator)
    env_wrapped = DataCollectionWrapper(env, tmp_root)
    saved_count = 0
    env_wrapped.reset()

    try:
        while True:
            images_dict, step_labels, trace_arrays, pipeline_metadata = collect_intervention_trajectory(
                env=env_wrapped,
                device=device,
                policy=policy,
                discriminator=discriminator,
                task_name=str(getattr(cfg.policy, "task_name", None) or cfg.intervention.environment),
                arm=cfg.intervention.arm,
                max_fr=cfg.intervention.max_fr,
                goal_update_mode=cfg.intervention.goal_update_mode,
                camera_names=camera_names,
                img_height=int(cfg.intervention.img_height),
                img_width=int(cfg.intervention.img_width),
                cool_down=float(cfg.intervention.cool_down_second),
                alert_cooldown_steps=int(cfg.discriminator.monitor.alert_cooldown_steps),
            )

            ep_dir = get_current_ep_dir(env_wrapped, tmp_root)
            env_wrapped.reset()
            if ep_dir is None or not os.path.isdir(ep_dir):
                print("Empty demo. Discarded.")
                continue

            demo_payload = read_single_demo_from_ep(ep_dir)
            if demo_payload is None:
                shutil.rmtree(ep_dir, ignore_errors=True)
                print("Empty demo. Discarded.")
                continue

            action = prompt_user_action()
            if action == "s":
                saved_count += 1
                append_demo_to_hdf5(
                    hdf5_path=tmp_hdf5_path,
                    demo_id=saved_count,
                    demo_payload=demo_payload,
                    step_labels=step_labels,
                    env_info=env_info,
                    camera_names=camera_names,
                    images_dict=images_dict,
                    trace_arrays=trace_arrays,
                    pipeline_metadata=pipeline_metadata,
                )
                print(f"Saved demo_{saved_count:06d}.")
                shutil.rmtree(ep_dir, ignore_errors=True)
                continue

            if action == "d":
                shutil.rmtree(ep_dir, ignore_errors=True)
                print("Deleted current demo.")
                continue

            if action == "q":
                shutil.rmtree(ep_dir, ignore_errors=True)
                print("Quit requested.")
                break

            if action == "f":
                saved_count += 1
                append_demo_to_hdf5(
                    hdf5_path=tmp_hdf5_path,
                    demo_id=saved_count,
                    demo_payload=demo_payload,
                    step_labels=step_labels,
                    env_info=env_info,
                    camera_names=camera_names,
                    images_dict=images_dict,
                    trace_arrays=trace_arrays,
                    pipeline_metadata=pipeline_metadata,
                )
                print(f"Saved demo_{saved_count:06d}.")
                shutil.rmtree(ep_dir, ignore_errors=True)
                print("Save and quit")
                break
    finally:
        try:
            if hasattr(device, "stop_control"):
                device.stop_control()
        except Exception:
            pass
        try:
            policy.close()
        except Exception:
            pass
        try:
            discriminator.close()
        except Exception:
            pass
        try:
            env.close()
        except Exception:
            pass

    final_hdf5_path = os.path.join(intervention_dir, f"intervention_{base_time}_{saved_count}.hdf5")
    print(f"\nFinal episodes: {saved_count}")
    try:
        if os.path.isfile(tmp_hdf5_path):
            os.replace(tmp_hdf5_path, final_hdf5_path)
            print(f"Final HDF5: {final_hdf5_path}")
        else:
            print("No HDF5 file created.")
    except Exception as exc:
        print(f"Failed to rename HDF5: {exc}")
        print(f"Kept temporary file: {tmp_hdf5_path}")


if __name__ == "__main__":
    main()
