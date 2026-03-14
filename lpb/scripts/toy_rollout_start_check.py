#!/usr/bin/env python3
"""
Toy rollout-start checker for LPB transport config.

This script isolates env-runner initialization from full training so rollout
startup issues (mujoco / robosuite / rendering backends) are easier to debug.
"""

import argparse
import os
import pathlib
import sys
import traceback

import hydra
from omegaconf import OmegaConf

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
os.chdir(ROOT_DIR)

from diffusion_policy.common.mujoco_py_compat import install_mujoco_py_stub


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="image_transport_diffusion_policy_cnn.yaml",
        help="Path to LPB transport config yaml (relative to cwd).",
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Path to transport hdf5 dataset.",
    )
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--n-train", type=int, default=1)
    parser.add_argument("--n-test", type=int, default=1)
    parser.add_argument("--output-dir", default="/tmp/lpb_toy_rollout")
    return parser.parse_args()


def main():
    args = parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
    os.environ.setdefault("LPB_MUJOCO_PY_STUB", "1")
    os.environ.setdefault("LPB_RENDER_GPU_IDS", os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    os.environ.setdefault("LPB_VECTOR_ENV_MODE", "auto")
    os.environ.setdefault("LPB_VECTOR_ENV_CONTEXT", "spawn")
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

    print("[Step 1] Install mujoco_py compatibility stub")
    install_mujoco_py_stub()
    import mujoco_py  # noqa: WPS433

    print(f"  - mujoco_py.__file__: {getattr(mujoco_py, '__file__', None)}")
    print(f"  - mujoco_py exception class: {mujoco_py.builder.MujocoException.__name__}")

    print("[Step 2] Load config and patch dataset/env-runner parameters")
    cfg = OmegaConf.load(args.config)
    cfg.task.dataset_path = args.dataset_path
    cfg.task.env_runner.dataset_path = args.dataset_path
    cfg.task.env_runner.n_envs = args.n_envs
    cfg.task.env_runner.n_train = args.n_train
    cfg.task.env_runner.n_test = args.n_test

    print("[Step 3] Instantiate env runner (rollout startup path)")
    print(f"  - LPB_RENDER_GPU_IDS: {os.environ.get('LPB_RENDER_GPU_IDS')}")
    print(f"  - LPB_VECTOR_ENV_MODE: {os.environ.get('LPB_VECTOR_ENV_MODE')}")
    print(f"  - LPB_VECTOR_ENV_CONTEXT: {os.environ.get('LPB_VECTOR_ENV_CONTEXT')}")
    try:
        runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=args.output_dir)
        print(f"  - runner type: {type(runner).__name__}")
        if hasattr(runner, "env_render_gpu_ids"):
            print(f"  - env_render_gpu_ids: {runner.env_render_gpu_ids}")
        print("RESULT: SUCCESS (rollout startup path initialized)")
    except Exception as exc:
        print(f"  - runner init exception: {type(exc).__name__}: {exc}")
        print("RESULT: FAILED (see traceback below)")
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
