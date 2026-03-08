import sys
import os
import pathlib
import glob
import re

import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import torch
import dill
import wandb
import json
from diffusion_policy.workspace.base_workspace import BaseWorkspace

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)


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
            # Prefer explicit epoch/model number in filename, then mtime.
            epoch = int(nums[-1]) if len(nums) > 0 else -1
            return (epoch, os.path.getmtime(p))

        best = max(candidates, key=_score)
        print(f"Resolved checkpoint directory {path_or_dir} -> {best}")
        return best

    raise FileNotFoundError(f"Checkpoint path does not exist: {path_or_dir}")


@hydra.main(version_base=None, config_path="dyn_model/conf/planner", config_name="eval_transport")
def main(cfg: DictConfig):
    output_dir = cfg.output_dir

    if os.path.exists(output_dir):
        confirm = input(f"Output path {output_dir} already exists! Overwrite? (y/N): ")
        if confirm.lower() != 'y':
            sys.exit(1)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    config_save_path = os.path.join(output_dir, 'eval_config.yaml')
    OmegaConf.save(config=cfg, f=config_save_path)
    print(f"Configuration saved to {config_save_path}")

    policy_checkpoint_path = _resolve_checkpoint_path(
        cfg.policy_checkpoint, file_suffixes=(".ckpt", ".pth")
    )
    dynamics_checkpoint_path = _resolve_checkpoint_path(
        cfg.dynamics_model_checkpoint, file_suffixes=(".pth", ".ckpt")
    )

    # Load policy_checkpoint
    with open(policy_checkpoint_path, 'rb') as f:
        payload = torch.load(f, pickle_module=dill)
    
    # Update configuration based on payload
    cfg_task_env_runner = payload['cfg']
    with open_dict(cfg_task_env_runner):
        cfg_task_env_runner.n_action_steps = cfg.n_action_steps
    with open_dict(cfg_task_env_runner.policy):
        cfg_task_env_runner.policy.n_action_steps = cfg.n_action_steps

    # Fill env runner fields for tasks trained with NoopImageRunner (e.g. PandaLift),
    # while keeping compatibility with existing robomimic/libero configs.
    env_runner_cfg = cfg_task_env_runner.task.env_runner
    with open_dict(env_runner_cfg):
        env_runner_cfg.n_action_steps = cfg.n_action_steps
        env_runner_cfg.n_test = cfg.n_test
        env_runner_cfg.n_test_vis = cfg.get('n_test_vis', cfg.n_test)
        env_runner_cfg.n_train = 0
        env_runner_cfg.n_train_vis = 0
        env_runner_cfg.test_start_seed = cfg.test_start_seed
        env_runner_cfg.n_obs_steps = cfg_task_env_runner.n_obs_steps

        if 'libero' in cfg.policy_checkpoint:
            env_runner_cfg.dataset_path = cfg.dataset_path

        if 'shape_meta' not in env_runner_cfg:
            env_runner_cfg.shape_meta = cfg_task_env_runner.shape_meta

        dataset_path = cfg.get('dataset_path', None)
        if dataset_path is None and 'dataset_path' in cfg_task_env_runner.task:
            dataset_path = cfg_task_env_runner.task.dataset_path
        if dataset_path is None and 'dataset' in cfg_task_env_runner.task and 'dataset_path' in cfg_task_env_runner.task.dataset:
            dataset_path = cfg_task_env_runner.task.dataset.dataset_path
        if dataset_path is not None and 'dataset_path' not in env_runner_cfg:
            env_runner_cfg.dataset_path = dataset_path

        if 'max_steps' in cfg:
            env_runner_cfg.max_steps = cfg.max_steps
        elif 'max_steps' not in env_runner_cfg:
            env_runner_cfg.max_steps = 400

        if 'render_obs_key' in cfg:
            env_runner_cfg.render_obs_key = cfg.render_obs_key
        if 'camera_name' in cfg:
            env_runner_cfg.camera_name = cfg.camera_name
        if 'env_name' in cfg:
            env_runner_cfg.env_name = cfg.env_name
        if 'robots' in cfg:
            env_runner_cfg.robots = cfg.robots
        if 'fps' in cfg:
            env_runner_cfg.fps = cfg.fps
        if 'reward_shaping' in cfg:
            env_runner_cfg.reward_shaping = cfg.reward_shaping

    # Initialize workspace
    cls = hydra.utils.get_class(cfg_task_env_runner._target_)
    workspace = cls(cfg_task_env_runner, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # Get policy from workspace
    policy = workspace.model
    if cfg_task_env_runner.training.use_ema:
        policy = workspace.ema_model

    device = torch.device(cfg.device)
    policy.to(device)
    policy.eval()
    
    normalizer_dir = os.path.dirname(os.path.dirname(policy_checkpoint_path))
    normalizer_path = os.path.join(normalizer_dir, 'normalizer.pth')
    policy.normalizer.load_state_dict(torch.load(normalizer_path))
    policy.normalizer.to(device)

    policy.initialize_planner(
        planner_target=cfg.planner_target,
        demo_dataset_config=payload['cfg'].task.dataset,
        dynamics_model_ckpt=dynamics_checkpoint_path,
        action_step=cfg_task_env_runner.n_action_steps,
        output_dir=cfg.output_dir,
        guidance_start_timestep=cfg.guidance_start_timestep,
        guidance_scale=cfg.guidance_scale,
        threshold=cfg.threshold,
        demo_dataset_path=cfg.get('demo_dataset_path', None)
    )

    # Run evaluation - use env_runner_target from the planner config
    cfg_task_env_runner.task.env_runner._target_ = cfg.env_runner_target

    # Check if it's a libero task by examining the dataset target
    dataset_target = payload['cfg'].task.dataset._target_
    if 'libero' in dataset_target:
        env_runner = hydra.utils.instantiate(
            cfg_task_env_runner.task.env_runner,
            output_dir=output_dir,
            task_dir=cfg_task_env_runner.task.env_runner.dataset_path
        )
    else:
        env_runner = hydra.utils.instantiate(
            cfg_task_env_runner.task.env_runner,
            output_dir=output_dir
        )

    runner_log = env_runner.run(policy)
    
    # Save evaluation results separately
    results = {}
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            results[key] = value._path
        else:
            results[key] = value
    
    results_path = os.path.join(output_dir, 'eval_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, sort_keys=True)

    print(f"Evaluation results saved to {results_path}")


if __name__ == '__main__':
    main()
