import os

import hydra
import imageio
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from tqdm import tqdm

from robosuite.policy.flow_multi.model import build_flow_policy
from robosuite.policy.flow_multi.utils.env_util import RobosuiteProprioExtractor, camera_obs_key


def resolve_language_instruction(task_prompt_map: dict, task_name: str) -> str:
    prompt_value = task_prompt_map.get(task_name, task_name)
    if isinstance(prompt_value, (list, tuple)):
        if len(prompt_value) == 0:
            raise ValueError(f"Task '{task_name}' has an empty prompt list")
        return str(prompt_value[0])
    return str(prompt_value)


def center_crop_resize(img: np.ndarray, out_size: int):
    height, width = img.shape[:2]
    crop_size = min(height, width)
    y0 = (height - crop_size) // 2
    x0 = (width - crop_size) // 2
    crop = img[y0 : y0 + crop_size, x0 : x0 + crop_size]
    if crop_size == out_size:
        return crop
    ys = np.linspace(0, crop_size - 1, out_size).astype(np.int32)
    xs = np.linspace(0, crop_size - 1, out_size).astype(np.int32)
    return crop[ys][:, xs]


def preprocess_observation_images(obs: dict, camera_names: list[str], image_size: int, device: torch.device):
    images = []
    for camera_name in camera_names:
        key = camera_obs_key(camera_name)
        image = center_crop_resize(obs[key], image_size).astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        images.append(image)
    images = torch.from_numpy(np.stack(images, axis=0)).unsqueeze(0).to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=images.dtype).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=images.dtype).view(1, 1, 3, 1, 1)
    return (images - mean) / std


@torch.no_grad()
def sample_action_sequence(model, images, proprio, language, action_horizon: int, n_steps: int):
    batch_size = proprio.shape[0]
    x = torch.randn(batch_size, model.action_dim, action_horizon, device=proprio.device)
    dt = 1.0 / float(n_steps)
    for step in range(n_steps):
        t = torch.full((batch_size,), float(step) / float(n_steps), device=proprio.device)
        v = model(x_t=x, t=t, images=images, proprio=proprio, language=language)
        x = x + dt * v
    return x.transpose(1, 2)


def resolve_checkpoint_task_name(task_name: str | None, checkpoint: dict) -> str:
    if task_name:
        return str(task_name)
    task_metadata_map = checkpoint.get("task_metadata_map")
    if isinstance(task_metadata_map, dict) and len(task_metadata_map) == 1:
        return next(iter(task_metadata_map.keys()))
    raise ValueError(
        "A task name must be specified for multitask checkpoints."
    )


def resolve_eval_task_names(cfg: DictConfig, checkpoint: dict) -> list[str]:
    task_name_cfg = getattr(cfg.eval, "task_name", None)
    if task_name_cfg is None:
        return [resolve_checkpoint_task_name(None, checkpoint)]
    if isinstance(task_name_cfg, str):
        task_names = [task_name_cfg]
    else:
        task_names = [str(task_name) for task_name in task_name_cfg]
    if len(task_names) == 0:
        raise ValueError("eval.task_name must contain at least one task name")
    return task_names


@hydra.main(version_base="1.2", config_path="./config", config_name="eval_flow")
def main(cfg: DictConfig):
    device = torch.device(cfg.eval.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(to_absolute_path(cfg.eval.ckpt), map_location="cpu", weights_only=False)

    camera_names = list(checkpoint["camera_names"])
    act_mean = checkpoint["act_mean"]
    act_std = checkpoint["act_std"]
    prop_mean = checkpoint["prop_mean"]
    prop_std = checkpoint["prop_std"]
    eval_task_names = resolve_eval_task_names(cfg, checkpoint)
    task_prompt_map = checkpoint.get("task_prompt_map", {})

    model_cfg = checkpoint["model_cfg"]
    action_dim = int(np.asarray(act_mean).shape[-1])
    proprio_dim = int(np.asarray(prop_mean).shape[-1])
    action_horizon = int(np.asarray(act_mean).shape[0])

    model = build_flow_policy(
        model_cfg,
        proprio_dim=proprio_dim,
        action_dim=action_dim,
        camera_names=camera_names,
    ).to(device)
    state_dict = checkpoint.get("ema_model", checkpoint["model"])
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    task_metadata_map = checkpoint.get("task_metadata_map")
    if task_metadata_map is not None:
        missing_tasks = [task_name for task_name in eval_task_names if task_name not in task_metadata_map]
        if len(missing_tasks) > 0:
            raise KeyError(f"Tasks {missing_tasks} not found in checkpoint task_metadata_map")

    env_cache = {}
    rng = np.random.default_rng()

    video_dir = to_absolute_path(cfg.eval.video_dir) if cfg.eval.video_dir else None
    if video_dir is not None:
        os.makedirs(video_dir, exist_ok=True)

    def get_task_context(task_name: str):
        if task_name not in env_cache:
            if task_metadata_map is None:
                env_metadata = checkpoint["env_metadata"]
            else:
                env_metadata = task_metadata_map[task_name]
            extractor = RobosuiteProprioExtractor(
                env_kwargs=env_metadata,
                has_renderer=False,
                has_offscreen_renderer=True,
                use_camera_obs=True,
                camera_names=camera_names,
                reward_shaping=False,
            )
            env_cache[task_name] = {
                "extractor": extractor,
                "env": extractor.env,
                "language_instruction": resolve_language_instruction(task_prompt_map, task_name),
            }
        return env_cache[task_name]

    def rollout(task_name: str, record_video: bool = False):
        task_context = get_task_context(task_name)
        extractor = task_context["extractor"]
        env = task_context["env"]
        language_instruction = task_context["language_instruction"]
        obs = env.reset()
        total_reward = 0.0
        success = False
        frames = []
        step_count = 0

        while step_count < int(cfg.eval.max_steps):
            images = preprocess_observation_images(obs, camera_names, int(cfg.data.image_size), device)
            proprio = extractor.extract(env.sim.get_state().flatten()).astype(np.float32)
            if prop_mean is not None:
                proprio = (proprio - prop_mean) / prop_std
            proprio = torch.from_numpy(proprio).unsqueeze(0).to(device)

            action_seq = sample_action_sequence(
                model=model,
                images=images,
                proprio=proprio,
                language=[language_instruction],
                action_horizon=action_horizon,
                n_steps=int(cfg.eval.n_ode_steps),
            )[0].cpu().numpy()
            if act_mean is not None:
                action_seq = action_seq * act_std + act_mean

            execute_steps = min(int(cfg.eval.action_horizon), action_horizon)
            for action in action_seq[:execute_steps]:
                if record_video:
                    frames.append(np.flipud(obs[camera_obs_key(cfg.eval.video_camera)]))
                obs, reward, done, info = env.step(action)
                total_reward += float(reward)
                step_count += 1
                success = (
                    success
                    or bool(info.get("success", False))
                    or bool(info.get("is_success", False))
                    or bool(env._check_success())
                )
                if success or done or step_count >= int(cfg.eval.max_steps):
                    break

            if success or done:
                break

        return total_reward, step_count, success, frames

    # evaluate each task
    for task_name in list(eval_task_names):
        for episode_idx in range(int(cfg.eval.episodes)):
            total_reward, step_count, success, frames = rollout(task_name=task_name, record_video=video_dir is not None)
            print(
                f"task={task_name} episode={episode_idx} return={total_reward:.4f} "
                f"steps={step_count} success={int(success)}"
            )
            if video_dir is not None and len(frames) > 0:
                video_path = os.path.join(video_dir, f"{task_name}_episode_{episode_idx}_return_{total_reward:.2f}.mp4")
                imageio.mimsave(video_path, frames, fps=int(cfg.eval.video_fps))
                print(f"saved video: {video_path}")

    if bool(cfg.eval.succ_rate):
        num_eval_episodes = int(cfg.eval.succ_rate_episodes)
        success_count = 0
        task_eval_counts = {task_name: 0 for task_name in eval_task_names}
        task_success_counts = {task_name: 0 for task_name in eval_task_names}
        for _ in tqdm(range(num_eval_episodes), desc="Success-rate eval"):
            task_name = str(rng.choice(eval_task_names))
            _, _, success, _ = rollout(task_name=task_name, record_video=False)
            success_count += int(success)
            task_eval_counts[task_name] += 1
            task_success_counts[task_name] += int(success)
        print(
            f"checkpoint={cfg.eval.ckpt} tasks={eval_task_names} "
            f"success_rate={success_count / float(num_eval_episodes):.4f} "
            f"({success_count}/{num_eval_episodes})"
        )
        for task_name in eval_task_names:
            num_task_episodes = task_eval_counts[task_name]
            if num_task_episodes == 0:
                continue
            print(
                f"task={task_name} success_rate="
                f"{task_success_counts[task_name] / float(num_task_episodes):.4f} "
                f"({task_success_counts[task_name]}/{num_task_episodes})"
            )

    for task_context in env_cache.values():
        task_context["extractor"].close()


if __name__ == "__main__":
    main()
