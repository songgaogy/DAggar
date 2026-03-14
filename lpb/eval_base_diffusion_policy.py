import argparse
import json
import os
import pathlib
import sys

import dill
import hydra
import torch
import wandb

from diffusion_policy.workspace.base_workspace import BaseWorkspace

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate base diffusion policy checkpoint and report success rate."
    )
    parser.add_argument("--policy-checkpoint", required=True, help="Path to policy .ckpt file")
    parser.add_argument("--output-dir", required=True, help="Directory to save eval outputs")
    parser.add_argument("--device", default="cuda:0", help="Torch device, e.g. cuda:0 or cpu")
    parser.add_argument("--n-test", type=int, default=50, help="Number of test episodes")
    parser.add_argument("--test-start-seed", type=int, default=100000, help="First test seed")
    parser.add_argument("--n-test-vis", type=int, default=0, help="Number of test videos to save")
    parser.add_argument("--n-train", type=int, default=0, help="Number of train inits to eval")
    parser.add_argument("--n-train-vis", type=int, default=0, help="Number of train videos to save")
    parser.add_argument("--n-envs", type=int, default=None, help="Number of vector env workers")
    parser.add_argument(
        "--env-runner-target",
        default=None,
        help="Override env runner target, e.g. diffusion_policy.env_runner.robomimic_image_sequential_runner.SequentialRobomimicImageRunner",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Optional dataset path override (for tasks that need explicit path at eval)",
    )
    parser.add_argument(
        "--success-reward-threshold",
        type=float,
        default=0.5,
        help="Episode success is max_reward > threshold",
    )
    parser.add_argument(
        "--use-ema",
        action="store_true",
        help="Force EMA policy for evaluation (if available)",
    )
    parser.add_argument(
        "--no-ema",
        action="store_true",
        help="Force non-EMA policy for evaluation",
    )
    return parser.parse_args()


def _convert_runner_log(runner_log):
    out = {}
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            out[key] = value._path
        else:
            out[key] = value
    return out


def _get_test_max_rewards(runner_log):
    values = []
    for key, value in runner_log.items():
        if key.startswith("test/sim_max_reward_"):
            values.append(float(value))
    return values


def main():
    args = parse_args()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.policy_checkpoint, "rb") as f:
        payload = torch.load(f, pickle_module=dill)

    cfg = payload["cfg"]
    cfg.task.env_runner.n_test = args.n_test
    cfg.task.env_runner.n_test_vis = args.n_test_vis
    cfg.task.env_runner.n_train = args.n_train
    cfg.task.env_runner.n_train_vis = args.n_train_vis
    cfg.task.env_runner.test_start_seed = args.test_start_seed

    if args.n_envs is not None:
        cfg.task.env_runner.n_envs = args.n_envs
    if args.env_runner_target is not None:
        cfg.task.env_runner._target_ = args.env_runner_target
    if args.dataset_path is not None:
        cfg.task.dataset_path = args.dataset_path
        cfg.task.env_runner.dataset_path = args.dataset_path
        if "dataset" in cfg.task and "dataset_path" in cfg.task.dataset:
            cfg.task.dataset.dataset_path = args.dataset_path

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=str(output_dir))
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    use_ema = cfg.training.use_ema
    if args.use_ema:
        use_ema = True
    if args.no_ema:
        use_ema = False
    policy = workspace.ema_model if use_ema else workspace.model

    device = torch.device(args.device)
    policy.to(device)
    policy.eval()

    normalizer_path = pathlib.Path(args.policy_checkpoint).parent.parent / "normalizer.pth"
    if normalizer_path.exists():
        policy.normalizer.load_state_dict(torch.load(normalizer_path, map_location=device))
        policy.normalizer.to(device)
    else:
        print(f"[WARN] normalizer not found at {normalizer_path}, using checkpoint-loaded normalizer.")

    dataset_target = payload["cfg"].task.dataset._target_
    if "libero" in dataset_target:
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=str(output_dir),
            task_dir=cfg.task.env_runner.dataset_path,
        )
    else:
        env_runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=str(output_dir))
    runner_log = env_runner.run(policy)

    test_max_rewards = _get_test_max_rewards(runner_log)
    if len(test_max_rewards) == 0:
        raise RuntimeError("No test episodes were found in runner log.")

    success_flags = [1.0 if r > args.success_reward_threshold else 0.0 for r in test_max_rewards]
    success_rate = float(sum(success_flags) / len(success_flags))
    test_mean_score = float(runner_log.get("test/mean_score", sum(test_max_rewards) / len(test_max_rewards)))

    summary = {
        "policy_checkpoint": args.policy_checkpoint,
        "device": args.device,
        "n_test": len(test_max_rewards),
        "test_start_seed": args.test_start_seed,
        "success_reward_threshold": args.success_reward_threshold,
        "test_mean_score": test_mean_score,
        "success_rate": success_rate,
        "used_ema_policy": bool(use_ema),
    }

    converted_log = _convert_runner_log(runner_log)
    results = {
        "summary": summary,
        "runner_log": converted_log,
    }
    results_path = output_dir / "eval_base_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    print("===== Base Diffusion Policy Evaluation =====")
    print(f"checkpoint: {args.policy_checkpoint}")
    print(f"n_test: {len(test_max_rewards)}")
    print(f"test_mean_score: {test_mean_score:.4f}")
    print(f"success_rate(threshold>{args.success_reward_threshold}): {success_rate:.4f}")
    print(f"saved: {results_path}")


if __name__ == "__main__":
    main()
