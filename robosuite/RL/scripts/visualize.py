import os
import argparse
import imageio
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from gymnasium import spaces
import gymnasium as gym

LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env_obs_dim, env_action_dim, env_action_space):
        super().__init__()
        self.input_dim = env_obs_dim

        self.fc1 = nn.Linear(self.input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, env_action_dim)
        self.fc_logstd = nn.Linear(256, env_action_dim)
        self.ln1 = nn.LayerNorm(256)
        self.ln2 = nn.LayerNorm(256)

        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env_action_space.high - env_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env_action_space.high + env_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.gelu(self.ln1(self.fc1(x)))
        x = F.gelu(self.ln2(self.fc2(x)))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_eval_action(self, x):
        mean, _ = self.forward(x)
        action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action


class DataCollectionEnv(gym.Env):
    def __init__(self, env_id, img_size=224, render=False, max_episode_steps=300):
        self.max_episode_steps = max_episode_steps
        self.step_count = 0
        self.image_size = img_size

        robot = "Panda"
        task = "Lift"
        if "Nut" in env_id:
            task = "NutAssembly"

        self.env = suite.make(
            env_name=task,
            robots=robot,
            controller_configs=load_composite_controller_config(controller=None, robot=robot),
            has_renderer=render,
            has_offscreen_renderer=True,
            renderer="mjviewer",
            use_camera_obs=True,
            use_object_obs=True,
            camera_names=["agentview", "robot0_eye_in_hand"],
            camera_heights=img_size,
            camera_widths=img_size,
            reward_shaping=True,
            control_freq=20,
            horizon=self.max_episode_steps,
        )

        low, high = self.env.action_spec
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        dummy_obs = self.env.reset()
        state_vec = self._process_state_for_actor(dummy_obs)
        self.obs_dim = state_vec.shape[0]
        self.action_dim = self.action_space.shape[0]

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.step_count = 0
        obs_dict = self.env.reset()
        return obs_dict

    def step(self, action):
        obs_dict, reward, done, info = self.env.step(action)
        self.step_count += 1

        success = False
        try:
            if self.env._check_success():
                success = True
        except Exception:
            pass
        info["success"] = success

        terminated = False
        truncated = self.step_count >= self.max_episode_steps

        return obs_dict, reward, terminated, truncated, info

    def _process_state_for_actor(self, obs_dict):
        values = []
        robot_keys = [
            "robot0_joint_pos_cos", "robot0_joint_pos_sin", "robot0_joint_vel",
            "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "robot0_gripper_qvel"
        ]
        object_keys = ["cube_pos", "cube_quat", "gripper_to_cube_pos"]

        for k in robot_keys:
            if k in obs_dict:
                values.append(obs_dict[k])
            else:
                if k == "robot0_joint_pos_cos" and "robot0_joint_pos" in obs_dict:
                    values.append(np.cos(obs_dict["robot0_joint_pos"]))
                elif k == "robot0_joint_pos_sin" and "robot0_joint_pos" in obs_dict:
                    values.append(np.sin(obs_dict["robot0_joint_pos"]))

        for k in object_keys:
            if k in obs_dict:
                values.append(obs_dict[k])

        if len(values) == 0:
            for _, v in obs_dict.items():
                arr = np.array(v)
                if arr.ndim == 1:
                    values.append(arr)

        return np.concatenate(values).astype(np.float32)

    def get_images(self, obs_dict):
        agentview = obs_dict["agentview_image"]
        eye_in_hand = obs_dict["robot0_eye_in_hand_image"]

        agentview = np.flipud(agentview)
        eye_in_hand = np.flipud(eye_in_hand)

        agentview = agentview.astype(np.uint8)
        eye_in_hand = eye_in_hand.astype(np.uint8)
        return agentview, eye_in_hand


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def overlay_text(img_bgr: np.ndarray, lines: List[str]) -> np.ndarray:
    out = img_bgr.copy()
    
    x, y = 5, 15  
    scale = 0.35  
    dy = 12       
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    thick = 1

    for i, line in enumerate(lines):
        yy = y + i * dy
        cv2.putText(out, line, (x, yy), font, scale, (0, 0, 0), thick + 1, cv2.LINE_AA)
        cv2.putText(out, line, (x, yy), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return out


def rollout_and_save_videos(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = DataCollectionEnv(
        env_id="PandaLift",
        img_size=args.img_size,
        render=args.render_onscreen,
        max_episode_steps=args.max_episode_steps,
    )

    actor = Actor(env.obs_dim, env.action_dim, env.action_space).to(device)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    state_dict = torch.load(args.checkpoint, map_location=device)
    actor.load_state_dict(state_dict, strict=True)
    actor.eval()

    out_success = os.path.join(args.dir, "success")
    out_fail = os.path.join(args.dir, "fail")
    ensure_dir(out_success)
    ensure_dir(out_fail)

    saved = 0
    attempts = 0
    while saved < args.num_episodes:
        attempts += 1
        obs_dict = env.reset(seed=attempts * 100)
        terminated = False
        truncated = False

        frames: List[np.ndarray] = []
        ep_return = 0.0
        ep_success = False
        t = 0

        while not (terminated or truncated):
            state_vec = env._process_state_for_actor(obs_dict)
            state_tensor = torch.tensor(state_vec, device=device).unsqueeze(0)

            with torch.no_grad():
                action = actor.get_eval_action(state_tensor).cpu().numpy()[0]

            next_obs_dict, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            ep_success = bool(info.get("success", False))

            img_agent, img_wrist = env.get_images(next_obs_dict)

            img_agent_bgr = cv2.cvtColor(img_agent, cv2.COLOR_RGB2BGR)
            img_wrist_bgr = cv2.cvtColor(img_wrist, cv2.COLOR_RGB2BGR)

            line1 = f"t={t:04d} r={float(reward):+.4f} R={ep_return:+.4f}"
            line2 = f"success={int(ep_success)}"
            img_agent_bgr = overlay_text(img_agent_bgr, [line1, line2])

            frame = np.concatenate([img_agent_bgr, img_wrist_bgr], axis=1)
            frames.append(frame)

            obs_dict = next_obs_dict
            t += 1

        subdir = out_success if ep_success else out_fail
        fname = f"ep_{saved:06d}_seed_{attempts*100}_len_{t}_R_{ep_return:.3f}.mp4"
        out_path = os.path.join(subdir, fname)

        rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        
        try:
            imageio.mimsave(out_path, rgb_frames, fps=args.fps, format='FFMPEG', codec='libx264', quality=9)
        except Exception as e:
            print(f"Error saving video: {e}")
            imageio.mimsave(out_path, rgb_frames, fps=args.fps)

        print(f"[OK] saved={out_path} success={ep_success} len={t} return={ep_return:.3f}")
        saved += 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dir", type=str, required=True)
    p.add_argument("--num_episodes", type=int, default=20)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--max_episode_steps", type=int, default=200)
    p.add_argument("--render_onscreen", action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    ensure_dir(args.dir)
    rollout_and_save_videos(args)


if __name__ == "__main__":
    main()
