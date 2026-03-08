import collections
import os
import pathlib
from typing import Dict, List, Optional, Tuple

import imageio.v2 as imageio
import numpy as np
import torch
import tqdm
import wandb
import robosuite as suite

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


class SequentialRobosuitePandaLiftImageRunner(BaseImageRunner):
    """
    LPB evaluation runner for robosuite PandaLift.
    """

    def __init__(
        self,
        output_dir,
        shape_meta: dict,
        dataset_path=None,  # kept for interface compatibility
        n_train=0,
        n_train_vis=0,
        train_start_idx=0,
        n_test=50,
        n_test_vis=6,
        test_start_seed=10000,
        max_steps=400,
        n_obs_steps=2,
        n_action_steps=8,
        render_obs_key="agentview_image",
        camera_name=None,
        fps=20,
        crf=22,  # unused, kept for compatibility with existing configs
        past_action=False,
        abs_action=False,  # should be False for PandaLift setup here
        env_name="Lift",
        robots="Panda",
        reward_shaping=False,
        control_freq=20,
        camera_heights=None,
        camera_widths=None,
        use_camera_obs=True,
        ignore_done=True,
        controller_configs=None,
        env_kwargs=None,
        tqdm_interval_sec=5.0,
        n_envs=None,  # unused, sequential runner
    ):
        super().__init__(output_dir)
        _ = (dataset_path, crf, n_envs)

        self.output_dir = output_dir
        self.media_dir = pathlib.Path(output_dir).joinpath("media")
        self.media_dir.mkdir(parents=True, exist_ok=True)

        self.shape_meta = shape_meta
        self.obs_shape_meta = shape_meta["obs"]
        self.rgb_keys = []
        self.lowdim_keys = []
        for key, attr in self.obs_shape_meta.items():
            obs_type = attr.get("type", "low_dim")
            if obs_type == "rgb":
                self.rgb_keys.append(key)
            else:
                self.lowdim_keys.append(key)
        if len(self.rgb_keys) == 0:
            raise ValueError("PandaLift image runner requires at least one rgb observation key.")

        self.render_obs_key = render_obs_key if render_obs_key in self.obs_shape_meta else self.rgb_keys[0]
        if camera_name is None:
            camera_name = self._camera_name_from_obs_key(self.render_obs_key)
        self.camera_name = camera_name

        # Use the configured image shape if size is not explicitly provided.
        render_shape = tuple(self.obs_shape_meta[self.render_obs_key]["shape"])
        if len(render_shape) != 3:
            raise ValueError(
                f"Expected rgb shape of form [C,H,W] for `{self.render_obs_key}`, got {render_shape}."
            )
        c, h, w = render_shape
        if c != 3:
            raise ValueError(f"Expected 3-channel image for `{self.render_obs_key}`, got C={c}.")
        if camera_heights is None:
            camera_heights = h
        if camera_widths is None:
            camera_widths = w

        self.n_train = int(n_train)
        self.n_train_vis = int(n_train_vis)
        self.train_start_idx = int(train_start_idx)
        self.n_test = int(n_test)
        self.n_test_vis = int(n_test_vis)
        self.test_start_seed = int(test_start_seed)
        self.max_steps = int(max_steps)
        self.n_obs_steps = int(n_obs_steps)
        self.n_action_steps = int(n_action_steps)
        self.fps = int(fps)
        self.past_action = bool(past_action)
        self.abs_action = bool(abs_action)
        self.env_name = env_name
        self.robots = robots
        self.reward_shaping = bool(reward_shaping)
        self.control_freq = int(control_freq)
        self.camera_heights = camera_heights
        self.camera_widths = camera_widths
        self.use_camera_obs = bool(use_camera_obs)
        self.ignore_done = bool(ignore_done)
        self.controller_configs = controller_configs
        self.env_kwargs = dict(env_kwargs) if isinstance(env_kwargs, dict) else {}
        self.tqdm_interval_sec = float(tqdm_interval_sec)

        self.env_configs = []
        for i in range(self.n_train):
            seed = self.train_start_idx + i
            self.env_configs.append(
                {
                    "prefix": "train/",
                    "seed": seed,
                    "enable_render": i < self.n_train_vis,
                }
            )
        for i in range(self.n_test):
            seed = self.test_start_seed + i
            self.env_configs.append(
                {
                    "prefix": "test/",
                    "seed": seed,
                    "enable_render": i < self.n_test_vis,
                }
            )

        self._robot_qpos_inds = None
        self._robot_qvel_inds = None
        self._warned_shape_mismatch = set()
        self._warned_action_shape_mismatch = set()

    @staticmethod
    def _camera_name_from_obs_key(obs_key: str) -> str:
        if obs_key.endswith("_image"):
            return obs_key[: -len("_image")]
        if obs_key.endswith("_rgb"):
            return obs_key[: -len("_rgb")]
        return obs_key

    @staticmethod
    def _joint_dims(joint_type: int) -> Tuple[int, int]:
        # Mujoco joint types: free=0, ball=1, slide=2, hinge=3
        if joint_type == 0:
            return 7, 6
        if joint_type == 1:
            return 4, 3
        if joint_type in (2, 3):
            return 1, 1
        return 1, 1

    def _make_env(self):
        kwargs = dict(
            env_name=self.env_name,
            robots=self.robots,
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=self.use_camera_obs,
            camera_names=[self.camera_name],
            camera_heights=self.camera_heights,
            camera_widths=self.camera_widths,
            reward_shaping=self.reward_shaping,
            ignore_done=self.ignore_done,
            control_freq=self.control_freq,
        )
        if self.controller_configs is not None:
            kwargs["controller_configs"] = self.controller_configs
        kwargs.update(self.env_kwargs)
        env = suite.make(**kwargs)
        if hasattr(env, "hard_reset"):
            env.hard_reset = False
        self._init_robot_joint_indices(env)
        return env

    def _init_robot_joint_indices(self, env):
        model = env.sim.model
        joint_names = list(getattr(model, "joint_names", []))
        if len(joint_names) == 0:
            joint_names = [model.joint_id2name(i) for i in range(int(model.njnt))]

        qpos_inds = []
        qvel_inds = []
        for joint_name in joint_names:
            if (joint_name is None) or (not str(joint_name).startswith("robot0_")):
                continue
            joint_id = model.joint_name2id(joint_name)
            joint_type = int(model.jnt_type[joint_id])
            qpos_adr = int(model.jnt_qposadr[joint_id])
            qvel_adr = int(model.jnt_dofadr[joint_id])
            n_qpos, n_dof = self._joint_dims(joint_type)
            qpos_inds.extend(range(qpos_adr, qpos_adr + n_qpos))
            qvel_inds.extend(range(qvel_adr, qvel_adr + n_dof))

        self._robot_qpos_inds = np.array(qpos_inds, dtype=np.int64)
        self._robot_qvel_inds = np.array(qvel_inds, dtype=np.int64)

    def _set_env_seed(self, env, seed: int):
        np.random.seed(seed)
        if hasattr(env, "seed"):
            try:
                env.seed(seed)
            except Exception:
                pass

    @staticmethod
    def _resize_image_nearest(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        in_h, in_w = image.shape[:2]
        if (in_h == target_h) and (in_w == target_w):
            return image
        ys = np.linspace(0, in_h - 1, target_h).astype(np.int64)
        xs = np.linspace(0, in_w - 1, target_w).astype(np.int64)
        return image[ys][:, xs]

    def _find_image_in_obs(self, obs: Dict, key: str) -> np.ndarray:
        candidates: List[str] = [key]
        base = key
        if key.endswith("_image"):
            base = key[: -len("_image")]
            candidates.extend([base, base + "_rgb"])
        elif key.endswith("_rgb"):
            base = key[: -len("_rgb")]
            candidates.extend([base, base + "_image"])
        else:
            candidates.extend([key + "_image", key + "_rgb"])

        for candidate in candidates:
            if candidate in obs:
                image = np.asarray(obs[candidate])
                if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
                    image = np.moveaxis(image, 0, -1)
                if image.dtype != np.uint8:
                    if np.issubdtype(image.dtype, np.floating):
                        image = np.clip(image, 0.0, 1.0) * 255.0
                    image = image.astype(np.uint8)
                return image

        raise KeyError(
            f"Could not find image obs for key `{key}` (tried {candidates}). "
            f"Available keys: {list(obs.keys())}"
        )

    def _match_lowdim_shape(self, value: np.ndarray, expected_dim: int, key: str) -> np.ndarray:
        value = np.asarray(value, dtype=np.float32).reshape(-1)
        if value.shape[0] == expected_dim:
            return value
        warn_key = (key, value.shape[0], expected_dim)
        if warn_key not in self._warned_shape_mismatch:
            print(
                f"[SequentialRobosuitePandaLiftImageRunner] "
                f"Lowdim key `{key}` dim mismatch: got {value.shape[0]}, expected {expected_dim}. "
                "Applying trim/pad to match."
            )
            self._warned_shape_mismatch.add(warn_key)
        if value.shape[0] > expected_dim:
            return value[:expected_dim]
        out = np.zeros((expected_dim,), dtype=np.float32)
        out[: value.shape[0]] = value
        return out

    def _extract_lowdim_from_env(self, env, key: str) -> np.ndarray:
        if self._robot_qpos_inds is None or self._robot_qvel_inds is None:
            self._init_robot_joint_indices(env)
        qpos = np.asarray(env.sim.data.qpos)
        qvel = np.asarray(env.sim.data.qvel)
        if key in ("robot0_jointvel_qpos", "robot0_joint_vel_qpos"):
            return qvel[self._robot_qvel_inds].astype(np.float32)
        if key == "robot0_joint_qpos":
            return qpos[self._robot_qpos_inds].astype(np.float32)
        raise KeyError(f"Unsupported lowdim key fallback: `{key}`")

    def _obs_to_model_obs(self, env, obs: Dict) -> Dict[str, np.ndarray]:
        out = {}
        for key in self.rgb_keys:
            image = self._find_image_in_obs(obs, key)
            c, h, w = tuple(self.obs_shape_meta[key]["shape"])
            image = self._resize_image_nearest(image, h, w)
            image = image.astype(np.float32) / 255.0
            out[key] = np.moveaxis(image, -1, 0)

        for key in self.lowdim_keys:
            if key in obs:
                value = np.asarray(obs[key], dtype=np.float32)
            else:
                value = self._extract_lowdim_from_env(env, key)
            expected_shape = tuple(self.obs_shape_meta[key]["shape"])
            if len(expected_shape) != 1:
                raise ValueError(
                    f"Only 1-D lowdim observations are supported in this runner. "
                    f"Got key `{key}` with shape {expected_shape}."
                )
            out[key] = self._match_lowdim_shape(value, expected_shape[0], key)
        return out

    def _match_env_action_dim(self, action: np.ndarray, env_action_dim: int) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        policy_action_dim = int(action.shape[0])
        if policy_action_dim == env_action_dim:
            return action

        warn_key = (policy_action_dim, env_action_dim)
        if warn_key not in self._warned_action_shape_mismatch:
            print(
                f"[SequentialRobosuitePandaLiftImageRunner] "
                f"Action dim mismatch: policy={policy_action_dim}, env={env_action_dim}. "
                "Applying trim/pad for execution."
            )
            self._warned_action_shape_mismatch.add(warn_key)

        if policy_action_dim > env_action_dim:
            return action[:env_action_dim]

        out = np.zeros((env_action_dim,), dtype=np.float32)
        out[:policy_action_dim] = action
        return out

    @staticmethod
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

    def _save_video(self, frames: List[np.ndarray], prefix: str, seed: int) -> Optional[str]:
        if len(frames) == 0:
            return None
        filename = self.media_dir.joinpath(f"{prefix.strip('/')}_seed_{seed}.mp4")
        imageio.mimsave(str(filename), frames, fps=self.fps)
        return str(filename)

    def run(self, policy: BaseImagePolicy):
        device = policy.device
        all_video_paths = [None] * len(self.env_configs)
        all_scores = [0.0] * len(self.env_configs)
        all_success = [0.0] * len(self.env_configs)

        for i, env_cfg in enumerate(self.env_configs):
            prefix = env_cfg["prefix"]
            seed = env_cfg["seed"]
            enable_render = env_cfg["enable_render"]

            env = self._make_env()
            self._set_env_seed(env, seed)
            raw_obs = env.reset()

            policy.reset()
            history = collections.deque(maxlen=self.n_obs_steps)
            current_model_obs = self._obs_to_model_obs(env, raw_obs)
            for _ in range(self.n_obs_steps):
                history.append(current_model_obs)

            episode_rewards = []
            frames = []
            done = False
            success = False
            past_action = None
            step_count = 0

            pbar = tqdm.tqdm(
                total=self.max_steps,
                desc=f"Eval PandaLift {i+1}/{len(self.env_configs)}",
                leave=False,
                mininterval=self.tqdm_interval_sec,
            )

            while (not done) and (not success) and (step_count < self.max_steps):
                np_obs = {
                    key: np.stack([h[key] for h in history], axis=0).astype(np.float32)
                    for key in history[0].keys()
                }
                if self.past_action and (past_action is not None):
                    np_obs["past_action"] = past_action[-(self.n_obs_steps - 1) :].astype(np.float32)

                obs_dict = dict_apply(
                    np_obs, lambda x: torch.from_numpy(np.expand_dims(x, axis=0)).to(device=device)
                )

                with torch.no_grad():
                    action_dict = policy.predict_action_dyn_guided(obs_dict)
                action = action_dict["action"].detach().cpu().numpy().squeeze(0)
                if not np.all(np.isfinite(action)):
                    raise RuntimeError("Nan or Inf action detected during PandaLift evaluation.")

                for j in range(min(action.shape[0], self.n_action_steps)):
                    env_action = self._match_env_action_dim(action[j], int(env.action_dim))
                    raw_obs, reward, done_flag, info = env.step(env_action)
                    if isinstance(done_flag, (list, tuple, np.ndarray)):
                        done = bool(np.all(done_flag))
                    else:
                        done = bool(done_flag)
                    success = self._is_success(env, info)
                    episode_rewards.append(float(reward))

                    if enable_render:
                        frame = self._find_image_in_obs(raw_obs, self.render_obs_key)
                        frames.append(frame)

                    current_model_obs = self._obs_to_model_obs(env, raw_obs)
                    history.append(current_model_obs)
                    past_action = action

                    step_count += 1
                    pbar.update(1)

                    if done or success or (step_count >= self.max_steps):
                        break

            pbar.close()
            max_reward = float(np.max(episode_rewards)) if len(episode_rewards) > 0 else 0.0
            all_scores[i] = max_reward
            all_success[i] = float(success)
            all_video_paths[i] = self._save_video(frames, prefix, seed) if enable_render else None

            env.close()
            del env

        log_data = {}
        grouped_scores = collections.defaultdict(list)
        grouped_success = collections.defaultdict(list)
        for i, env_cfg in enumerate(self.env_configs):
            prefix = env_cfg["prefix"]
            seed = env_cfg["seed"]
            grouped_scores[prefix].append(all_scores[i])
            grouped_success[prefix].append(all_success[i])
            log_data[f"{prefix}sim_max_reward_{seed}"] = all_scores[i]
            log_data[f"{prefix}sim_success_{seed}"] = all_success[i]
            video_path = all_video_paths[i]
            if video_path is not None and os.path.exists(video_path):
                log_data[f"{prefix}sim_video_{seed}"] = wandb.Video(video_path)

        for prefix, values in grouped_scores.items():
            log_data[f"{prefix}mean_score"] = float(np.mean(values)) if len(values) > 0 else 0.0
        for prefix, values in grouped_success.items():
            log_data[f"{prefix}success_rate"] = float(np.mean(values)) if len(values) > 0 else 0.0

        return log_data
