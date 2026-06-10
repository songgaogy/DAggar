"""Debug: verify success_rollout demos re-fire success on replay.

Mirrors the done/reward logic of `load_hdf5_demos_into_flow_transitions`
(set stored state -> env.step(stored action) -> sparse_success_reward) but
skips image IO for speed. Confirms that in 0/1 mode every successful demo
yields a terminal step with done=True and reward=1.0.

Run:
    python -m robosuite.pipeline.scripts.utils.debug_success_done \
        --split-dir data/PickPlaceCereal/success_rollout --max-demos 20
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import h5py
import numpy as np

from robosuite.pipeline.envs.robosuite import sparse_success_reward
from robosuite.policy.flow_multi.utils.env_util import RobosuiteProprioExtractor, parse_env_info


def replay_demo_success(env, states: np.ndarray, actions: np.ndarray, reward_mode: str):
    """Return (first_success_idx | None, reward_at_terminal, n_steps_run)."""
    for step_idx in range(len(actions)):
        env.sim.set_state_from_flattened(np.asarray(states[step_idx]))
        env.sim.forward()
        env.done = False
        step_output = env.step(np.asarray(actions[step_idx], dtype=np.float32))
        info = step_output[-1] if isinstance(step_output[-1], dict) else None
        reward, success = sparse_success_reward(env, info, reward_mode=reward_mode)
        if success:
            return step_idx, float(reward), step_idx + 1
    return None, 0.0, len(actions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", default="data/PickPlaceCereal/success_rollout")
    parser.add_argument("--reward-mode", default="0/1")
    parser.add_argument("--max-demos", type=int, default=20, help="per hdf5 file; <=0 = all")
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    hdf5_paths = sorted(glob.glob(str(split_dir / "*.hdf5"))) + sorted(glob.glob(str(split_dir / "*.h5")))
    if not hdf5_paths:
        raise FileNotFoundError(f"No HDF5 under {split_dir}")

    total = 0
    ok = 0
    bad: list[str] = []
    for path in hdf5_paths:
        with h5py.File(path, "r") as f:
            env_info = parse_env_info(f.attrs["env_info"])
        extractor = RobosuiteProprioExtractor(
            env_info, has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False
        )
        env = extractor.env
        try:
            with h5py.File(path, "r") as f:
                g = f["demos"] if "demos" in f else f["data"]
                keys = sorted(g.keys())
                if args.max_demos > 0:
                    keys = keys[: args.max_demos]
                for key in keys:
                    demo = g[key]
                    states = np.asarray(demo["states"])
                    actions = np.asarray(demo["actions"], dtype=np.float32)
                    succ_attr = bool(demo.attrs.get("successful", False))
                    model_xml = demo.attrs.get("model_file", None)
                    if isinstance(model_xml, bytes):
                        model_xml = model_xml.decode("utf-8")
                    if model_xml:
                        env.reset_from_xml_string(env.edit_model_xml(str(model_xml)))
                        env.done = False
                        env.timestep = 0
                        extractor.env = env
                        extractor.sim = env.sim
                        extractor._build_robot_joint_indices()
                    else:
                        env.reset()
                    idx, reward, n_run = replay_demo_success(env, states, actions, args.reward_mode)
                    total += 1
                    fired = idx is not None and reward == 1.0
                    if fired:
                        ok += 1
                    else:
                        bad.append(f"{Path(path).name}::{key}")
                    print(
                        f"{Path(path).name}::{key} succ_attr={int(succ_attr)} "
                        f"n_actions={len(actions)} first_success_idx={idx} "
                        f"reward@term={reward} -> {'OK' if fired else 'NO-DONE'}"
                    )
        finally:
            env.close()

    print("\n==== SUMMARY ====")
    print(f"reward_mode={args.reward_mode} demos_checked={total} done+reward1={ok} no_done={len(bad)}")
    if bad:
        print("NO-DONE demos:")
        for b in bad:
            print("  ", b)


if __name__ == "__main__":
    main()
