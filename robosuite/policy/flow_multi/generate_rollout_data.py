import datetime
import json
import os

import h5py
import hydra
import numpy as np
import robosuite as suite
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from robosuite.policy.flow_multi.eval_flow import (
    preprocess_observation_images,
    resolve_checkpoint_task_name,
    resolve_language_instruction,
    sample_action_sequence,
)
from robosuite.policy.flow_multi.model import build_flow_policy
from robosuite.policy.flow_multi.utils.env_util import RobosuiteProprioExtractor


DEFAULT_TASK_OUTPUT_ROOTS = {
    "PickPlaceBread": "PickPlaceBread",
    "PickPlaceCereal": "PickPlaceCereal",
    "PickPlaceMilk": "PickPlaceMilk",
    "PickPlaceCan": "PandaPickPlaceCan",
    "Stack": "PandaStack",
    "Lift": "PandaLift",
}


def now_readable(ts: datetime.datetime | None = None) -> str:
    if ts is None:
        ts = datetime.datetime.now()
    return ts.strftime("%Y-%m-%d_%H-%M-%S")


def normalize_keep_mode(keep_mode: str) -> str:
    keep_mode = str(keep_mode).strip().lower()
    if keep_mode not in {"all", "success", "fail"}:
        raise ValueError(f"Unsupported keep_mode: {keep_mode}")
    return keep_mode


def resolve_num_trajs(cfg_generate: DictConfig) -> int:
    num_trajs = getattr(cfg_generate, "num_trajs", None)
    if num_trajs is None:
        num_trajs = cfg_generate.episodes
    num_trajs = int(num_trajs)
    if num_trajs <= 0:
        raise ValueError("generate.num_trajs must be >= 1")
    return num_trajs


def should_keep_episode(keep_mode: str, success: bool) -> bool:
    if keep_mode == "all":
        return True
    if keep_mode == "success":
        return bool(success)
    return not bool(success)


def discover_all_cameras(env) -> list[str]:
    return sorted(str(name) for name in env.sim.model.camera_names)


def resolve_output_dir(task_name: str, keep_mode: str, output_dir: str | None) -> str:
    if output_dir is not None and str(output_dir).strip().lower() not in {"", "none", "null"}:
        return to_absolute_path(str(output_dir))

    task_root = DEFAULT_TASK_OUTPUT_ROOTS.get(task_name, task_name)
    rollout_dir_name = {
        "all": "rollout",
        "success": "success_rollout",
        "fail": "fail_rollout",
    }[keep_mode]
    return to_absolute_path(os.path.join("data", task_root, rollout_dir_name))


def append_demo_to_hdf5(
    hdf5_path: str,
    demo_id: int,
    env_name: str,
    env_info: str,
    all_camera_names: list[str],
    states: list[np.ndarray],
    actions: list[np.ndarray],
    images_dict: dict[str, list[np.ndarray]],
    render_height: int,
    render_width: int,
    success: bool,
    xml_str: str,
):
    with h5py.File(hdf5_path, "a") as file_handle:
        if "demos" not in file_handle:
            demos_grp = file_handle.create_group("demos")
            file_handle.attrs["created_at"] = now_readable()
            file_handle.attrs["repository_version"] = suite.__version__
            file_handle.attrs["env"] = env_name
            file_handle.attrs["env_info"] = env_info
            file_handle.attrs["camera_names"] = json.dumps(list(all_camera_names))
        else:
            demos_grp = file_handle["demos"]

        demo_grp = demos_grp.create_group(f"demo_{demo_id:06d}")
        demo_grp.attrs["length"] = int(len(actions))
        demo_grp.attrs["successful"] = bool(success)
        if len(xml_str) > 0:
            demo_grp.attrs["model_file"] = xml_str

        demo_grp.create_dataset("states", data=np.asarray(states))
        demo_grp.create_dataset("actions", data=np.asarray(actions))

        obs_grp = demo_grp.create_group("observations")
        empty_images = np.zeros((0, render_height, render_width, 3), dtype=np.uint8)
        for camera_name in all_camera_names:
            cam_grp = obs_grp.create_group(camera_name)
            cam_images = images_dict.get(camera_name, [])
            cam_np = empty_images if len(cam_images) == 0 else np.stack(cam_images, axis=0).astype(np.uint8)
            cam_grp.create_dataset(
                "images",
                data=cam_np,
                dtype=np.uint8,
                compression="gzip",
                compression_opts=4,
                chunks=True,
            )


def save_rollout_summary(
    attempt_idx: int,
    saved_count: int,
    keep_mode: str,
    success: bool,
    step_count: int,
):
    print(
        f"attempt={attempt_idx} saved={saved_count} keep_mode={keep_mode} "
        f"success={int(success)} steps={step_count}"
    )


@hydra.main(version_base="1.2", config_path="./config", config_name="generate_rollout_data")
def main(cfg: DictConfig):
    keep_mode = normalize_keep_mode(cfg.generate.keep_mode)
    num_trajs = resolve_num_trajs(cfg.generate)
    device = torch.device(cfg.generate.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(to_absolute_path(cfg.generate.ckpt), map_location="cpu", weights_only=False)
    task_name = resolve_checkpoint_task_name(getattr(cfg.generate, "task_name", None), checkpoint)
    task_prompt_map = checkpoint.get("task_prompt_map", {})
    language_instruction = resolve_language_instruction(task_prompt_map, task_name)

    train_camera_names = list(checkpoint["camera_names"])
    act_mean = checkpoint["act_mean"]
    act_std = checkpoint["act_std"]
    prop_mean = checkpoint["prop_mean"]
    prop_std = checkpoint["prop_std"]
    model_cfg = checkpoint["model_cfg"]
    task_metadata_map = checkpoint.get("task_metadata_map")
    if task_metadata_map is None:
        env_metadata = dict(checkpoint["env_metadata"])
    else:
        if task_name not in task_metadata_map:
            raise KeyError(f"Task '{task_name}' not found in checkpoint task_metadata_map")
        env_metadata = dict(task_metadata_map[task_name])

    action_dim = int(np.asarray(act_mean).shape[-1])
    proprio_dim = int(np.asarray(prop_mean).shape[-1])
    action_horizon = int(np.asarray(act_mean).shape[0])

    model = build_flow_policy(
        model_cfg,
        proprio_dim=proprio_dim,
        action_dim=action_dim,
        camera_names=train_camera_names,
    ).to(device)
    state_dict = checkpoint.get("ema_model", checkpoint["model"])
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    extractor = RobosuiteProprioExtractor(
        env_kwargs=env_metadata,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=train_camera_names,
        reward_shaping=False,
    )
    env = extractor.env
    all_camera_names = discover_all_cameras(env)

    output_dir = resolve_output_dir(
        task_name=task_name,
        keep_mode=keep_mode,
        output_dir=getattr(cfg.generate, "output_dir", None),
    )
    os.makedirs(output_dir, exist_ok=True)
    base_time = now_readable()
    output_path = os.path.join(
        output_dir,
        f"flow_multi_rollout_{task_name}_{keep_mode}_{base_time}_{num_trajs}.hdf5",
    )

    env_info = json.dumps(env_metadata)
    render_height = int(cfg.generate.render_height)
    render_width = int(cfg.generate.render_width)

    saved_count = 0
    attempt_idx = 0
    while saved_count < num_trajs:
        attempt_idx += 1
        obs = env.reset()
        done = False
        success = False
        step_count = 0
        try:
            xml_str = env.sim.model.get_xml()
        except Exception:
            xml_str = ""

        ep_states = []
        ep_actions = []
        ep_images = {camera_name: [] for camera_name in all_camera_names}

        while step_count < int(cfg.generate.max_steps) and not done:
            images = preprocess_observation_images(
                obs=obs,
                camera_names=train_camera_names,
                image_size=int(cfg.data.image_size),
                device=device,
            )
            proprio = extractor.extract(env.sim.get_state().flatten()).astype(np.float32)
            if prop_mean is not None:
                proprio = (proprio - prop_mean) / prop_std
            proprio_tensor = torch.from_numpy(proprio).unsqueeze(0).to(device)

            action_seq = sample_action_sequence(
                model=model,
                images=images,
                proprio=proprio_tensor,
                language=[language_instruction],
                action_horizon=action_horizon,
                n_steps=int(cfg.generate.n_ode_steps),
            )[0].cpu().numpy()
            if act_mean is not None:
                action_seq = action_seq * act_std + act_mean

            execute_steps = min(int(cfg.generate.action_horizon), action_horizon)
            for action in action_seq[:execute_steps]:
                current_state = env.sim.get_state().flatten().copy()
                ep_states.append(current_state)
                ep_actions.append(np.asarray(action, dtype=np.float32).copy())

                for camera_name in all_camera_names:
                    frame = env.sim.render(
                        height=render_height,
                        width=render_width,
                        camera_name=camera_name,
                    )
                    ep_images[camera_name].append(np.asarray(frame, dtype=np.uint8))

                obs, _, done, info = env.step(action)
                step_count += 1
                success = (
                    success
                    or bool(info.get("success", False))
                    or bool(info.get("is_success", False))
                    or bool(env._check_success())
                )
                if success or done or step_count >= int(cfg.generate.max_steps):
                    break

        save_rollout_summary(
            attempt_idx=attempt_idx,
            saved_count=saved_count,
            keep_mode=keep_mode,
            success=success,
            step_count=step_count,
        )

        if should_keep_episode(keep_mode=keep_mode, success=success):
            saved_count += 1
            append_demo_to_hdf5(
                hdf5_path=output_path,
                demo_id=saved_count,
                env_name=str(env_metadata["env_name"]),
                env_info=env_info,
                all_camera_names=all_camera_names,
                states=ep_states,
                actions=ep_actions,
                images_dict=ep_images,
                render_height=render_height,
                render_width=render_width,
                success=success,
                xml_str=xml_str,
            )
            print(f"saved demo_{saved_count:06d} to {output_path}")
        else:
            print("skipped save because episode did not match keep_mode")

    extractor.close()
    print(f"finished saved_episodes={saved_count} attempts={attempt_idx} output={output_path}")


if __name__ == "__main__":
    main()
