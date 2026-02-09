#!/usr/bin/env python3
import argparse
import numpy as np

def try_import_robosuite():
    try:
        import robosuite as suite
        return suite
    except Exception as e:
        raise RuntimeError(f"Failed to import robosuite: {e}")

def infer_success(info: dict):
    if not isinstance(info, dict):
        return None
    for k in ["success", "is_success", "task_success", "episode_success"]:
        if k in info:
            v = info[k]
            if isinstance(v, (bool, np.bool_)):
                return bool(v)
            if isinstance(v, (int, float, np.integer, np.floating)):
                return bool(v)
    return None

def find_inner_env(env):
    visited = set()
    cur = env
    for _ in range(20):
        if id(cur) in visited:
            break
        visited.add(id(cur))
        for attr in ["env", "unwrapped", "_env", "wrapped_env"]:
            if hasattr(cur, attr):
                nxt = getattr(cur, attr)
                if nxt is not None and nxt is not cur:
                    cur = nxt
                    break
        else:
            break
    return cur

def staged_rewards_if_any(env):
    base = find_inner_env(env)
    for obj in [env, base]:
        if hasattr(obj, "staged_rewards") and callable(getattr(obj, "staged_rewards")):
            try:
                vals = obj.staged_rewards()
                if isinstance(vals, (list, tuple, np.ndarray)):
                    return np.array(vals, dtype=np.float32)
            except Exception:
                return None
    return None

def make_env(env_name, robots, camera, dense, seed, render, offscreen):
    suite = try_import_robosuite()
    env = suite.make(
        env_name=env_name,
        robots=robots,
        has_renderer=render,
        has_offscreen_renderer=offscreen,
        use_camera_obs=True,
        camera_names=camera,
        camera_heights=84,
        camera_widths=84,
        reward_shaping=bool(dense),
        control_freq=20,
        horizon=200,
        ignore_done=False,
    )
    try:
        env.seed(seed)
    except Exception:
        pass
    return env

def random_action(env, rng):
    try:
        low, high = env.action_spec
        low = np.asarray(low, dtype=np.float32)
        high = np.asarray(high, dtype=np.float32)
        return rng.uniform(low=low, high=high).astype(np.float32)
    except Exception:
        if hasattr(env, "action_space"):
            return env.action_space.sample()
        raise RuntimeError("Cannot sample random action: env has no action_spec/action_space")

def obs_from_env(env):
    obs = env.reset()
    return obs

def summarize_rewards(rewards):
    rewards = np.asarray(rewards, dtype=np.float32)
    nonzero = np.mean(np.abs(rewards) > 1e-12) if rewards.size else 0.0
    stats = {
        "count": int(rewards.size),
        "min": float(np.min(rewards)) if rewards.size else 0.0,
        "mean": float(np.mean(rewards)) if rewards.size else 0.0,
        "max": float(np.max(rewards)) if rewards.size else 0.0,
        "nonzero_fraction": float(nonzero),
    }
    uniq = np.unique(np.round(rewards, 6))
    if uniq.size > 20:
        sampled = np.random.choice(uniq, size=20, replace=False)
        sampled.sort()
        stats["unique_values_sample"] = sampled.tolist()
        stats["unique_values_count_rounded"] = int(uniq.size)
    else:
        stats["unique_values_sample"] = uniq.tolist()
        stats["unique_values_count_rounded"] = int(uniq.size)
    return stats

def run_once(args, dense_flag):
    env = make_env(
        env_name=args.env,
        robots=args.robots,
        camera=args.camera,
        dense=dense_flag,
        seed=args.seed,
        render=args.render,
        offscreen=not args.render,
    )
    rng = np.random.default_rng(args.seed)

    rewards = []
    successes = []
    stage_vals = []

    obs = obs_from_env(env)
    for t in range(args.steps):
        a = random_action(env, rng)
        obs, r, done, info = env.step(a)
        r = float(r)
        rewards.append(r)

        succ = infer_success(info)
        if succ is not None:
            successes.append(int(succ))

        sr = staged_rewards_if_any(env)
        if sr is not None:
            stage_vals.append(sr)

        if (t % args.print_every) == 0 or done:
            msg = f"[t={t:04d}] r={r:.6f} done={bool(done)}"
            if succ is not None:
                msg += f" success={bool(succ)}"
            if sr is not None:
                comps = ",".join([f"s{i}={float(v):.6f}" for i, v in enumerate(sr.tolist())])
                msg += f" comps=({comps})"
            print(msg, flush=True)

        if done and args.reset_on_done:
            obs = obs_from_env(env)

        if done:
            if args.reset_on_done:
                obs = obs_from_env(env)
                continue
            break


    env.close()

    stats = summarize_rewards(rewards)
    out = {
        "dense": bool(dense_flag),
        "reward_stats": stats,
    }
    if len(successes) > 0:
        out["success_rate_over_logged_steps"] = float(np.mean(successes))
        out["success_count"] = int(np.sum(successes))
        out["success_logged_steps"] = int(len(successes))

    if len(stage_vals) > 0:
        stage_arr = np.stack(stage_vals, axis=0)
        out["staged_rewards_mean"] = np.mean(stage_arr, axis=0).astype(np.float32).tolist()
        out["staged_rewards_max"] = np.max(stage_arr, axis=0).astype(np.float32).tolist()
        out["staged_rewards_nonzero_fraction"] = (
            np.mean(np.abs(stage_arr) > 1e-12, axis=0).astype(np.float32).tolist()
        )
        out["staged_rewards_count"] = int(stage_arr.shape[0])
        out["staged_rewards_dims"] = int(stage_arr.shape[1])

    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Lift")
    parser.add_argument("--robots", type=str, default="Panda")
    parser.add_argument("--camera", type=str, default="agentview")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", type=str, choices=["dense", "sparse", "both"], default="both")
    parser.add_argument("--render", action="store_true", help="Enable on-screen rendering")
    parser.add_argument("--reset-on-done", action="store_true")
    args = parser.parse_args()

    if args.mode in ["dense", "both"]:
        print("\n=== Running DENSE reward (reward_shaping=True) ===", flush=True)
        out_dense = run_once(args, dense_flag=True)
        print("\nDENSE summary:", flush=True)
        for k, v in out_dense.items():
            print(f"  {k}: {v}", flush=True)

    if args.mode in ["sparse", "both"]:
        print("\n=== Running SPARSE reward (reward_shaping=False) ===", flush=True)
        out_sparse = run_once(args, dense_flag=False)
        print("\nSPARSE summary:", flush=True)
        for k, v in out_sparse.items():
            print(f"  {k}: {v}", flush=True)

if __name__ == "__main__":
    main()
