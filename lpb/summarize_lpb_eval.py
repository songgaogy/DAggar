#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize LPB eval_results.json and optionally compare with base policy results."
    )
    parser.add_argument("--eval-results", required=True, help="Path to LPB eval_results.json")
    parser.add_argument(
        "--success-reward-threshold",
        type=float,
        default=0.5,
        help="Episode success is max_reward > threshold",
    )
    parser.add_argument(
        "--base-results",
        default=None,
        help="Optional path to eval_base_results.json for seed-wise comparison",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output summary path (default: same dir as eval results, file eval_lpb_summary.json)",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def _extract_runner_log(payload: Dict) -> Dict:
    if "runner_log" in payload and isinstance(payload["runner_log"], dict):
        return payload["runner_log"]
    return payload


def _extract_seed_rewards(runner_log: Dict) -> Dict[int, float]:
    rewards = {}
    for key, value in runner_log.items():
        if key.startswith("test/sim_max_reward_"):
            seed = int(key.split("_")[-1])
            rewards[seed] = float(value)
    if len(rewards) == 0:
        raise RuntimeError("No test/sim_max_reward_* entries found in runner log.")
    return rewards


def _summarize_rewards(
    rewards: Dict[int, float], success_reward_threshold: float
) -> Tuple[float, float, Dict[int, int]]:
    seeds = sorted(rewards.keys())
    success = {
        seed: 1 if rewards[seed] > success_reward_threshold else 0 for seed in seeds
    }
    success_rate = sum(success.values()) / len(success)
    mean_score = sum(rewards.values()) / len(rewards)
    return success_rate, mean_score, success


def _compare_seed_outcomes(
    base_success: Dict[int, int], lpb_success: Dict[int, int]
) -> Dict[str, object]:
    overlap = sorted(set(base_success.keys()) & set(lpb_success.keys()))
    improved = [seed for seed in overlap if base_success[seed] == 0 and lpb_success[seed] == 1]
    regressed = [seed for seed in overlap if base_success[seed] == 1 and lpb_success[seed] == 0]
    both_fail = [seed for seed in overlap if base_success[seed] == 0 and lpb_success[seed] == 0]
    both_success = [seed for seed in overlap if base_success[seed] == 1 and lpb_success[seed] == 1]
    return {
        "overlap_n": len(overlap),
        "improved_n": len(improved),
        "regressed_n": len(regressed),
        "both_fail_n": len(both_fail),
        "both_success_n": len(both_success),
        "improved_seeds": improved,
        "regressed_seeds": regressed,
        "both_fail_seeds": both_fail,
    }


def main() -> None:
    args = parse_args()

    eval_results_path = Path(args.eval_results)
    if not eval_results_path.exists():
        raise FileNotFoundError(f"eval results not found: {eval_results_path}")

    payload = _load_json(eval_results_path)
    runner_log = _extract_runner_log(payload)
    lpb_rewards = _extract_seed_rewards(runner_log)
    lpb_success_rate, lpb_mean_score, lpb_success = _summarize_rewards(
        lpb_rewards, args.success_reward_threshold
    )

    summary = {
        "source_eval_results": str(eval_results_path),
        "success_reward_threshold": args.success_reward_threshold,
        "n_test": len(lpb_rewards),
        "test_mean_score": lpb_mean_score,
        "success_rate": lpb_success_rate,
        "seed_rewards": {str(k): lpb_rewards[k] for k in sorted(lpb_rewards)},
    }

    if args.base_results:
        base_path = Path(args.base_results)
        if not base_path.exists():
            raise FileNotFoundError(f"base results not found: {base_path}")
        base_payload = _load_json(base_path)
        base_runner_log = _extract_runner_log(base_payload)
        base_rewards = _extract_seed_rewards(base_runner_log)
        base_success_rate, base_mean_score, base_success = _summarize_rewards(
            base_rewards, args.success_reward_threshold
        )
        seed_comparison = _compare_seed_outcomes(base_success, lpb_success)
        summary["base_comparison"] = {
            "source_base_results": str(base_path),
            "base_n_test": len(base_rewards),
            "base_test_mean_score": base_mean_score,
            "base_success_rate": base_success_rate,
            "lpb_minus_base_success_rate": lpb_success_rate - base_success_rate,
            "seed_transition": seed_comparison,
        }

    output_path = (
        Path(args.output)
        if args.output
        else eval_results_path.parent / "eval_lpb_summary.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print("===== LPB Evaluation Summary =====")
    print(f"n_test: {summary['n_test']}")
    print(f"test_mean_score: {summary['test_mean_score']:.4f}")
    print(f"success_rate(threshold>{args.success_reward_threshold}): {summary['success_rate']:.4f}")
    if "base_comparison" in summary:
        delta = summary["base_comparison"]["lpb_minus_base_success_rate"]
        print(f"lpb_minus_base_success_rate: {delta:+.4f}")
        print(
            "regressed_n: "
            f"{summary['base_comparison']['seed_transition']['regressed_n']}, "
            "both_fail_n: "
            f"{summary['base_comparison']['seed_transition']['both_fail_n']}"
        )
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
