import argparse
import json
import os
import pathlib
import sys
from collections import deque
from tqdm import tqdm

import dill
import h5py
import hydra
import numpy as np
import torch
import robomimic.utils.file_utils as FileUtils

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.robomimic_replay_image_dataset import undo_transform_action
from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
from diffusion_policy.env_runner.robomimic_image_runner import (
    _configure_worker_render_gpu,
    _parse_render_gpu_ids,
    create_env,
)
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.workspace.base_workspace import BaseWorkspace

sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate robomimic-format rollout datasets from all diffusion checkpoints."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--checkpoint-dir",
        help="Directory containing diffusion policy checkpoints.",
    )
    group.add_argument(
        "--checkpoint-path",
        help="Single diffusion policy checkpoint to export.",
    )
    parser.add_argument(
        "--expert-dataset",
        required=True,
        help="Expert dataset used for env metadata and dtype/template matching.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save generated rollout HDF5 files.",
    )
    parser.add_argument("--device", default="cuda:0", help="Torch device.")
    parser.add_argument("--n-test", type=int, default=50, help="Rollouts per checkpoint.")
    parser.add_argument(
        "--test-start-seed",
        type=int,
        default=100000,
        help="First evaluation seed for rollout generation.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override max episode steps. Defaults to checkpoint config.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing rollout files.",
    )
    return parser.parse_args()


def stack_last_n_obs(history, n_steps):
    assert len(history) > 0
    items = list(history)
    result = dict()
    for key in items[-1].keys():
        stacked = np.zeros((n_steps,) + items[-1][key].shape, dtype=items[-1][key].dtype)
        valid_items = [item[key] for item in items[-min(n_steps, len(items)) :]]
        stacked[-len(valid_items) :] = np.asarray(valid_items)
        if len(valid_items) < n_steps:
            stacked[: n_steps - len(valid_items)] = valid_items[0]
        result[key] = stacked
    return result


def prepare_policy_obs(obs_history, device):
    stacked_obs = stack_last_n_obs(obs_history, obs_history.maxlen)
    return dict_apply(
        stacked_obs,
        lambda x: torch.from_numpy(np.expand_dims(x, axis=0)).to(device=device),
    )


def load_policy(checkpoint_path, device):
    with open(checkpoint_path, "rb") as f:
        payload = torch.load(f, pickle_module=dill, map_location="cpu")

    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir="rollout_export")
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    use_ema = bool(cfg.training.use_ema)
    policy = workspace.ema_model if use_ema else workspace.model
    policy.to(device)
    policy.eval()

    normalizer_path = pathlib.Path(checkpoint_path).parent.parent / "normalizer.pth"
    if normalizer_path.exists():
        policy.normalizer.load_state_dict(torch.load(normalizer_path, map_location=device))
        policy.normalizer.to(device)

    return payload, policy


def get_source_template(dataset_path):
    with h5py.File(dataset_path, "r") as f:
        data_group = f["data"]
        first_demo = data_group[sorted(data_group.keys())[0]]
        template = {
            "env_args": data_group.attrs["env_args"],
            "field_dtypes": {
                "actions": first_demo["actions"].dtype,
                "abs_actions": first_demo["abs_actions"].dtype,
                "states": first_demo["states"].dtype,
                "rewards": first_demo["rewards"].dtype,
                "dones": first_demo["dones"].dtype,
            },
            "obs_dtypes": {
                key: first_demo["obs"][key].dtype
                for key in first_demo["obs"].keys()
            },
            "obs_keys": list(first_demo["obs"].keys()),
        }
    return template


def to_dataset_obs(raw_obs, obs_keys, obs_dtypes):
    result = {}
    for key in obs_keys:
        value = np.asarray(raw_obs[key])
        if key.endswith("_image"):
            if value.ndim != 3:
                raise RuntimeError(f"Unexpected image shape for {key}: {value.shape}")
            if value.shape[0] in (1, 3):
                value = np.moveaxis(value, 0, -1)
            if value.dtype != np.uint8:
                scale = 255.0 if np.max(value) <= 1.0 + 1e-6 else 1.0
                value = np.clip(value * scale, 0, 255).astype(np.uint8)
        result[key] = value.astype(obs_dtypes[key], copy=False)
    return result


def save_rollout_hdf5(output_path, episodes, template):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = int(sum(len(episode["actions"]) for episode in episodes))

    with h5py.File(output_path, "w") as f:
        data_group = f.create_group("data")
        data_group.attrs["env_args"] = template["env_args"]
        data_group.attrs["total"] = total

        for episode_idx, episode in enumerate(episodes):
            demo_group = data_group.create_group(f"demo_{episode_idx}")
            num_samples = len(episode["actions"])
            demo_group.attrs["num_samples"] = num_samples

            demo_group.create_dataset(
                "actions",
                data=np.asarray(episode["actions"], dtype=template["field_dtypes"]["actions"]),
            )
            demo_group.create_dataset(
                "abs_actions",
                data=np.asarray(episode["abs_actions"], dtype=template["field_dtypes"]["abs_actions"]),
            )
            demo_group.create_dataset(
                "states",
                data=np.asarray(episode["states"], dtype=template["field_dtypes"]["states"]),
            )
            demo_group.create_dataset(
                "rewards",
                data=np.asarray(episode["rewards"], dtype=template["field_dtypes"]["rewards"]),
            )
            demo_group.create_dataset(
                "dones",
                data=np.asarray(episode["dones"], dtype=template["field_dtypes"]["dones"]),
            )

            obs_group = demo_group.create_group("obs")
            for key in template["obs_keys"]:
                obs_group.create_dataset(key, data=np.asarray(episode["obs"][key], dtype=template["obs_dtypes"][key]))


def rollout_checkpoint(
    checkpoint_path,
    expert_dataset,
    output_path,
    device,
    n_test,
    test_start_seed,
    max_steps_override,
):
    payload, policy = load_policy(checkpoint_path, device=device)
    cfg = payload["cfg"]
    template = get_source_template(expert_dataset)
    shape_meta = cfg.shape_meta
    env_meta = FileUtils.get_env_metadata_from_dataset(expert_dataset)
    env_meta["env_kwargs"]["use_object_obs"] = False
    abs_action = bool(cfg.task.abs_action)
    if abs_action:
        env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False

    render_gpu_ids = _parse_render_gpu_ids()
    render_gpu_id = render_gpu_ids[0]
    _configure_worker_render_gpu(render_gpu_id)

    robomimic_env = create_env(
        env_meta=env_meta,
        shape_meta=shape_meta,
        enable_render=True,
        render_gpu_device_id=render_gpu_id,
    )
    robomimic_env.env.hard_reset = False
    wrapped_env = RobomimicImageWrapper(
        env=robomimic_env,
        shape_meta=shape_meta,
        render_obs_key=cfg.task.env_runner.render_obs_key,
    )

    rotation_transformer = RotationTransformer("axis_angle", "rotation_6d")
    max_steps = max_steps_override or int(cfg.task.env_runner.max_steps)
    n_obs_steps = int(cfg.n_obs_steps)

    episodes = []
    summary = {
        "checkpoint_path": str(checkpoint_path),
        "output_path": str(output_path),
        "episodes": [],
    }

    try:
        for rollout_idx in tqdm(range(n_test), desc=f"start collecting {n_test} rollout"):
            seed = test_start_seed + rollout_idx
            wrapped_env.seed(seed)
            current_obs = wrapped_env.reset()
            current_raw_obs = wrapped_env.get_raw_observation()
            current_state = wrapped_env.env.get_state()["states"]

            obs_history = deque([current_obs], maxlen=n_obs_steps)
            policy.reset()

            episode = {
                "actions": [],
                "abs_actions": [],
                "states": [],
                "rewards": [],
                "dones": [],
                "obs": {key: [] for key in template["obs_keys"]},
            }

            success = False
            step_count = 0
            while step_count < max_steps:
                obs_dict = prepare_policy_obs(obs_history, device=device)
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)
                action_seq = dict_apply(action_dict, lambda x: x.detach().cpu().numpy().squeeze(0))["action"]
                if not np.all(np.isfinite(action_seq)):
                    raise RuntimeError(f"Non-finite action predicted at seed {seed}.")

                abs_action_seq = action_seq
                if abs_action:
                    env_action_seq = undo_transform_action(action_seq, rotation_transformer)
                else:
                    env_action_seq = action_seq

                for action_idx in range(env_action_seq.shape[0]):
                    if step_count >= max_steps:
                        break

                    episode["states"].append(np.asarray(current_state))
                    current_dataset_obs = to_dataset_obs(
                        current_raw_obs,
                        obs_keys=template["obs_keys"],
                        obs_dtypes=template["obs_dtypes"],
                    )
                    for key, value in current_dataset_obs.items():
                        episode["obs"][key].append(value)

                    env_action = np.asarray(env_action_seq[action_idx])
                    abs_env_action = np.asarray(
                        env_action if abs_action else abs_action_seq[action_idx]
                    )
                    episode["actions"].append(env_action.copy())
                    episode["abs_actions"].append(abs_env_action.copy())

                    next_obs, reward, done, _ = wrapped_env.step(env_action)
                    success = bool(wrapped_env.get_success_label())
                    step_count += 1
                    truncated = step_count >= max_steps
                    done_flag = bool(done or success or truncated)

                    episode["rewards"].append(float(reward))
                    episode["dones"].append(int(done_flag))

                    current_obs = next_obs
                    current_raw_obs = wrapped_env.get_raw_observation()
                    current_state = wrapped_env.env.get_state()["states"]
                    obs_history.append(current_obs)

                    if done_flag:
                        break

                if success or step_count >= max_steps:
                    break

            for key in episode["obs"]:
                if len(episode["obs"][key]) == 0:
                    raise RuntimeError(f"Episode with seed {seed} produced zero samples.")
                episode["obs"][key] = np.stack(episode["obs"][key], axis=0)
            episode["actions"] = np.asarray(episode["actions"])
            episode["abs_actions"] = np.asarray(episode["abs_actions"])
            episode["states"] = np.asarray(episode["states"])
            episode["rewards"] = np.asarray(episode["rewards"])
            episode["dones"] = np.asarray(episode["dones"])

            episodes.append(episode)
            summary["episodes"].append(
                {
                    "seed": seed,
                    "num_samples": int(len(episode["actions"])),
                    "success": bool(success),
                    "max_reward": float(np.max(episode["rewards"])) if len(episode["rewards"]) > 0 else 0.0,
                }
            )
    finally:
        wrapped_env.close()

    save_rollout_hdf5(output_path=output_path, episodes=episodes, template=template)

    success_rate = float(np.mean([episode["success"] for episode in summary["episodes"]])) if summary["episodes"] else 0.0
    summary["success_rate"] = success_rate
    summary["num_episodes"] = len(summary["episodes"])
    summary["total_samples"] = int(sum(episode["num_samples"] for episode in summary["episodes"]))
    summary["actions_match_abs_actions"] = bool(abs_action)

    summary_path = output_path.with_suffix(".json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(
        f"[rollout] checkpoint={checkpoint_path} output={output_path} "
        f"episodes={summary['num_episodes']} success_rate={success_rate:.3f}"
    )


def main():
    args = parse_args()
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.checkpoint_path is not None:
        checkpoint_paths = [pathlib.Path(args.checkpoint_path)]
    else:
        checkpoint_dir = pathlib.Path(args.checkpoint_dir)
        checkpoint_paths = sorted(checkpoint_dir.glob("*.ckpt"), key=lambda path: int(path.stem))
        if len(checkpoint_paths) == 0:
            raise FileNotFoundError(f"No checkpoints found under {checkpoint_dir}")

    device = torch.device(args.device)
    for checkpoint_path in checkpoint_paths:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        output_path = output_dir / f"{checkpoint_path.stem}.hdf5"
        if output_path.exists() and not args.overwrite:
            print(f"[skip] existing rollout file: {output_path}")
            continue

        rollout_checkpoint(
            checkpoint_path=checkpoint_path,
            expert_dataset=args.expert_dataset,
            output_path=output_path,
            device=device,
            n_test=args.n_test,
            test_start_seed=args.test_start_seed,
            max_steps_override=args.max_steps,
        )


if __name__ == "__main__":
    main()
