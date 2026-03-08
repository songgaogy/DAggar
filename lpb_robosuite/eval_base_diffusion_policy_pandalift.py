import argparse
import collections
import glob
import json
import os
import pathlib
import random
import re
from typing import Dict, List, Optional, Tuple

import dill
import h5py
import hydra
import imageio.v2 as imageio
import numpy as np
import torch
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace


def _resolve_checkpoint_path(path_or_dir: str, file_suffixes=(".ckpt", ".pth")) -> str:
    path_or_dir = os.path.expanduser(path_or_dir)
    if os.path.isfile(path_or_dir):
        return path_or_dir
    if os.path.isdir(path_or_dir):
        candidates = []
        for suffix in file_suffixes:
            candidates.extend(glob.glob(os.path.join(path_or_dir, f"*{suffix}")))
        if len(candidates) == 0:
            raise FileNotFoundError(f"No checkpoint files with suffixes {file_suffixes} under: {path_or_dir}")

        def _score(p):
            name = os.path.basename(p)
            nums = re.findall(r"\d+", name)
            epoch = int(nums[-1]) if len(nums) > 0 else -1
            return (epoch, os.path.getmtime(p))

        return max(candidates, key=_score)
    raise FileNotFoundError(f"Checkpoint path does not exist: {path_or_dir}")


def _find_hdf5_files(dataset_path: str) -> List[str]:
    dataset_path = os.path.expanduser(dataset_path)
    if os.path.isfile(dataset_path):
        return [dataset_path]
    if os.path.isdir(dataset_path):
        return sorted(glob.glob(os.path.join(dataset_path, "**", "*.hdf5"), recursive=True))
    matches = sorted(glob.glob(dataset_path))
    return [m for m in matches if os.path.isfile(m)]


def _load_env_info(dataset_path: str) -> Optional[dict]:
    files = _find_hdf5_files(dataset_path)
    for fp in files:
        try:
            with h5py.File(fp, "r") as f:
                candidates = []
                if "env_info" in f.attrs:
                    candidates.append(f.attrs["env_info"])
                if "data" in f and "env_info" in f["data"].attrs:
                    candidates.append(f["data"].attrs["env_info"])
                for c in candidates:
                    if isinstance(c, bytes):
                        c = c.decode("utf-8")
                    if isinstance(c, str):
                        return json.loads(c)
                    if isinstance(c, dict):
                        return c
        except Exception:
            continue
    return None


def _camera_name_from_obs_key(obs_key: str) -> str:
    if obs_key.endswith("_image"):
        return obs_key[: -len("_image")]
    if obs_key.endswith("_rgb"):
        return obs_key[: -len("_rgb")]
    return obs_key


def _joint_dims(joint_type: int) -> Tuple[int, int]:
    # Mujoco joint types: free=0, ball=1, slide=2, hinge=3
    if joint_type == 0:
        return 7, 6
    if joint_type == 1:
        return 4, 3
    if joint_type in (2, 3):
        return 1, 1
    return 1, 1


def _build_robot_joint_indices(env):
    model = env.sim.model
    joint_names = list(getattr(model, "joint_names", []))
    if len(joint_names) == 0:
        joint_names = [model.joint_id2name(i) for i in range(int(model.njnt))]

    qpos_inds = []
    qvel_inds = []
    for name in joint_names:
        if (name is None) or (not str(name).startswith("robot0_")):
            continue
        jid = model.joint_name2id(name)
        n_qpos, n_dof = _joint_dims(int(model.jnt_type[jid]))
        qpos_adr = int(model.jnt_qposadr[jid])
        qvel_adr = int(model.jnt_dofadr[jid])
        qpos_inds.extend(range(qpos_adr, qpos_adr + n_qpos))
        qvel_inds.extend(range(qvel_adr, qvel_adr + n_dof))
    return np.array(qpos_inds, dtype=np.int64), np.array(qvel_inds, dtype=np.int64)


def _resize_image_nearest(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = image.shape[:2]
    if (h == target_h) and (w == target_w):
        return image
    ys = np.linspace(0, h - 1, target_h).astype(np.int64)
    xs = np.linspace(0, w - 1, target_w).astype(np.int64)
    return image[ys][:, xs]


def _find_image_in_obs(obs: Dict, key: str) -> np.ndarray:
    candidates = [key]
    if key.endswith("_image"):
        base = key[: -len("_image")]
        candidates.extend([base, base + "_rgb"])
    elif key.endswith("_rgb"):
        base = key[: -len("_rgb")]
        candidates.extend([base, base + "_image"])
    else:
        candidates.extend([key + "_image", key + "_rgb"])

    for k in candidates:
        if k in obs:
            img = np.asarray(obs[k])
            if img.ndim == 3 and img.shape[0] == 3 and img.shape[-1] != 3:
                img = np.moveaxis(img, 0, -1)
            if img.dtype != np.uint8:
                if np.issubdtype(img.dtype, np.floating):
                    img = np.clip(img, 0.0, 1.0) * 255.0
                img = img.astype(np.uint8)
            return img
    raise KeyError(f"Cannot find image key `{key}` in obs keys {list(obs.keys())}")


def _match_shape_1d(value: np.ndarray, expected_dim: int) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32).reshape(-1)
    if value.shape[0] == expected_dim:
        return value
    if value.shape[0] > expected_dim:
        return value[:expected_dim]
    out = np.zeros((expected_dim,), dtype=np.float32)
    out[: value.shape[0]] = value
    return out


def _extract_model_obs(raw_obs: Dict, env, shape_meta_obs: Dict, qpos_inds, qvel_inds) -> Dict[str, np.ndarray]:
    out = {}
    for key, attr in shape_meta_obs.items():
        obs_type = attr.get("type", "low_dim")
        expected_shape = tuple(attr["shape"])
        if obs_type == "rgb":
            c, h, w = expected_shape
            image = _find_image_in_obs(raw_obs, key)
            image = _resize_image_nearest(image, h, w).astype(np.float32) / 255.0
            out[key] = np.moveaxis(image, -1, 0)
            continue

        if key in raw_obs:
            value = np.asarray(raw_obs[key], dtype=np.float32)
        else:
            qpos = np.asarray(env.sim.data.qpos)
            qvel = np.asarray(env.sim.data.qvel)
            if key == "robot0_joint_qpos":
                value = qpos[qpos_inds].astype(np.float32)
            elif key in ("robot0_jointvel_qpos", "robot0_joint_vel_qpos"):
                value = qvel[qvel_inds].astype(np.float32)
            else:
                raise KeyError(f"Missing lowdim key `{key}` in env obs and no fallback rule.")
        if len(expected_shape) != 1:
            raise ValueError(f"Only 1D lowdim obs are supported, key={key}, shape={expected_shape}")
        out[key] = _match_shape_1d(value, expected_shape[0])
    return out


def _match_action_dim(action: np.ndarray, env_action_dim: int) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape[0] == env_action_dim:
        return action
    if action.shape[0] > env_action_dim:
        return action[:env_action_dim]
    out = np.zeros((env_action_dim,), dtype=np.float32)
    out[: action.shape[0]] = action
    return out


def _is_success(env, info: Dict) -> bool:
    if isinstance(info, dict):
        for key in ("success", "is_success", "task_success"):
            if key in info:
                try:
                    return bool(info[key])
                except Exception:
                    pass
    if hasattr(env, "_check_success"):
        try:
            return bool(env._check_success())
        except Exception:
            pass
    return False


def _create_env(env_name: str, robots: str, camera_name: str, image_h: int, image_w: int, env_info: Optional[dict]):
    import robosuite as suite

    env_kwargs = {}
    if isinstance(env_info, dict):
        env_kwargs = dict(env_info)
    # Follow eval_flow.py behavior: use default controller unless explicitly set by script.
    env_kwargs.pop("controller_configs", None)
    env_kwargs.pop("controller_config", None)

    # Ensure env identity and rendering setup for evaluation.
    env_kwargs["env_name"] = env_name
    env_kwargs["robots"] = robots
    env_kwargs["has_renderer"] = False
    env_kwargs["has_offscreen_renderer"] = True
    env_kwargs["use_camera_obs"] = True
    env_kwargs["camera_names"] = [camera_name]
    env_kwargs["camera_heights"] = image_h
    env_kwargs["camera_widths"] = image_w
    env_kwargs["ignore_done"] = True
    env_kwargs["reward_shaping"] = False

    try:
        env = suite.make(**env_kwargs)
    except Exception as e:
        print(f"[EvalBase] env creation with env_info kwargs failed ({e}), retrying with minimal kwargs.")
        minimal_kwargs = dict(
            env_name=env_name,
            robots=robots,
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=[camera_name],
            camera_heights=image_h,
            camera_widths=image_w,
            ignore_done=True,
            reward_shaping=False,
        )
        env = suite.make(**minimal_kwargs)

    if hasattr(env, "hard_reset"):
        env.hard_reset = False
    return env


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy_checkpoint",
        type=str,
        default="data/outputs/2026.03.06/13.42.42_train_diffusion_unet_hybrid_pandalift_image/checkpoints",
    )
    parser.add_argument("--dataset_path", type=str, default="../data/PandaLift/expert")
    parser.add_argument("--output_dir", type=str, default="data/release_pandalift_base_policy")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--n_video", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--seed_start", type=int, default=100000)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--camera_name", type=str, default="agentview")
    parser.add_argument("--render_obs_key", type=str, default="agentview_image")
    parser.add_argument("--env_name", type=str, default="Lift")
    parser.add_argument("--robots", type=str, default="Panda")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    media_dir = os.path.join(output_dir, "media")
    pathlib.Path(media_dir).mkdir(parents=True, exist_ok=True)

    ckpt_path = _resolve_checkpoint_path(args.policy_checkpoint, file_suffixes=(".ckpt", ".pth"))
    print(f"Using policy checkpoint: {ckpt_path}")

    with open(ckpt_path, "rb") as f:
        payload = torch.load(f, pickle_module=dill)
    cfg = payload["cfg"]

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    device = torch.device(args.device)
    policy.to(device)
    policy.eval()

    normalizer_path = os.path.join(os.path.dirname(os.path.dirname(ckpt_path)), "normalizer.pth")
    policy.normalizer.load_state_dict(torch.load(normalizer_path, map_location=device))
    policy.normalizer.to(device)

    shape_meta = cfg.shape_meta
    shape_meta_obs = shape_meta["obs"]
    n_obs_steps = int(cfg.n_obs_steps)
    n_action_steps = int(cfg.policy.n_action_steps)

    if args.max_steps is not None:
        max_steps = int(args.max_steps)
    else:
        max_steps = 400

    if args.render_obs_key not in shape_meta_obs:
        args.render_obs_key = list(shape_meta_obs.keys())[0]
    render_c, render_h, render_w = tuple(shape_meta_obs[args.render_obs_key]["shape"])
    assert render_c == 3

    env_info = _load_env_info(args.dataset_path)
    if env_info is not None:
        print("Loaded env_info from dataset.")
    else:
        print("env_info not found in dataset. Using explicit env args.")

    episode_logs = []
    success_count = 0

    for ep in range(args.n_episodes):
        seed = args.seed_start + ep
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        env = _create_env(
            env_name=args.env_name,
            robots=args.robots,
            camera_name=args.camera_name,
            image_h=render_h,
            image_w=render_w,
            env_info=env_info,
        )
        if hasattr(env, "seed"):
            try:
                env.seed(seed)
            except Exception:
                pass

        qpos_inds, qvel_inds = _build_robot_joint_indices(env)
        raw_obs = env.reset()
        policy.reset()

        history = collections.deque(maxlen=n_obs_steps)
        first_obs = _extract_model_obs(raw_obs, env, shape_meta_obs, qpos_inds, qvel_inds)
        for _ in range(n_obs_steps):
            history.append(first_obs)

        done = False
        success = False
        step_count = 0
        total_reward = 0.0
        max_reward = -np.inf
        frames = []
        record_video = ep < args.n_video

        while (not done) and (not success) and (step_count < max_steps):
            np_obs = {k: np.stack([h[k] for h in history], axis=0).astype(np.float32) for k in history[0].keys()}
            obs_dict = dict_apply(np_obs, lambda x: torch.from_numpy(np.expand_dims(x, axis=0)).to(device=device))

            with torch.no_grad():
                action_dict = policy.predict_action(obs_dict)
            action_seq = action_dict["action"].detach().cpu().numpy().squeeze(0)

            for i in range(min(action_seq.shape[0], n_action_steps)):
                env_action = _match_action_dim(action_seq[i], int(env.action_dim))
                raw_obs, reward, done_flag, info = env.step(env_action)
                done = bool(np.all(done_flag)) if isinstance(done_flag, (list, tuple, np.ndarray)) else bool(done_flag)
                success = _is_success(env, info)

                total_reward += float(reward)
                max_reward = max(max_reward, float(reward))
                step_count += 1

                if record_video:
                    frame = _find_image_in_obs(raw_obs, args.render_obs_key)
                    frame = np.rot90(frame, 2)  # rotate 180 degrees as requested
                    frames.append(frame)

                model_obs = _extract_model_obs(raw_obs, env, shape_meta_obs, qpos_inds, qvel_inds)
                history.append(model_obs)

                if done or success or (step_count >= max_steps):
                    break

        if max_reward == -np.inf:
            max_reward = 0.0

        if record_video and len(frames) > 0:
            video_path = os.path.join(media_dir, f"episode_{ep:03d}_seed_{seed}_succ_{int(success)}.mp4")
            imageio.mimsave(video_path, frames, fps=args.fps)
        else:
            video_path = None

        episode_logs.append(
            {
                "episode": ep,
                "seed": seed,
                "success": int(success),
                "steps": int(step_count),
                "return": float(total_reward),
                "max_reward": float(max_reward),
                "video_path": video_path,
            }
        )
        success_count += int(success)
        print(f"[EvalBase] ep={ep:03d} seed={seed} success={int(success)} return={total_reward:.4f} steps={step_count}")
        env.close()

    success_rate = success_count / float(args.n_episodes)
    mean_return = float(np.mean([x["return"] for x in episode_logs])) if len(episode_logs) > 0 else 0.0
    mean_max_reward = float(np.mean([x["max_reward"] for x in episode_logs])) if len(episode_logs) > 0 else 0.0
    mean_steps = float(np.mean([x["steps"] for x in episode_logs])) if len(episode_logs) > 0 else 0.0

    summary = {
        "policy_checkpoint": ckpt_path,
        "dataset_path": os.path.abspath(os.path.expanduser(args.dataset_path)),
        "n_episodes": int(args.n_episodes),
        "n_video": int(args.n_video),
        "seed_start": int(args.seed_start),
        "success_rate": float(success_rate),
        "success_count": int(success_count),
        "mean_return": mean_return,
        "mean_max_reward": mean_max_reward,
        "mean_steps": mean_steps,
        "episodes": episode_logs,
    }

    summary_path = os.path.join(output_dir, "base_policy_eval_results.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"[EvalBase] Saved results to {summary_path}")
    print(f"[EvalBase] success_rate={success_rate:.4f} ({success_count}/{args.n_episodes})")


if __name__ == "__main__":
    main()
