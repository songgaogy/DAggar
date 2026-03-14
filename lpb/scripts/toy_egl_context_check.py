#!/usr/bin/env python3
"""
Minimal EGL context checker per GPU id.
"""

import argparse
import os
import traceback
import multiprocessing as mp


def _worker(device_id: int):
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(device_id)
    try:
        from robosuite.renderers.context.egl_context import EGLGLContext

        ctx = EGLGLContext(max_width=64, max_height=64, device_id=device_id)
        ctx.make_current()
        ctx.free()
        print(f"[OK] pid={os.getpid()} gpu={device_id} created EGL context")
        return 0
    except Exception as exc:
        print(f"[FAIL] pid={os.getpid()} gpu={device_id} -> {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default="0,1", help="Comma separated GPU ids")
    parser.add_argument(
        "--mode",
        default="sequential",
        choices=["sequential", "parallel"],
        help="Run checks one-by-one or concurrently.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    devices = [int(x.strip()) for x in args.devices.split(",") if x.strip()]
    if not devices:
        raise ValueError("No devices provided")

    print(f"MUJOCO_GL={os.environ.get('MUJOCO_GL', 'egl')} mode={args.mode} devices={devices}")
    fails = 0
    if args.mode == "sequential":
        ctx = mp.get_context("spawn")
        for d in devices:
            p = ctx.Process(target=_worker, args=(d,))
            p.start()
            p.join()
            fails += (p.exitcode != 0)
    else:
        ctx = mp.get_context("spawn")
        procs = [ctx.Process(target=_worker, args=(d,)) for d in devices]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
            fails += (p.exitcode != 0)

    if fails:
        print(f"RESULT: {fails}/{len(devices)} failed")
        return 1
    print("RESULT: all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
