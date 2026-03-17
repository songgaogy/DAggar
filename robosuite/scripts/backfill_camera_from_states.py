"""
Backfill missing camera observations in a robosuite demonstration HDF5 by replaying
recorded MuJoCo simulator states and rendering a new camera view.

This works for datasets collected from full simulator states (e.g. via
DataCollectionWrapper / collect_human_demonstrations.py). It does not work if the
dataset only stores derived robot observations such as joint positions or end-effector
poses, because object poses and other simulator internals are then lost.

Examples:
    python robosuite/scripts/backfill_camera_from_states.py \
        --input /path/to/demo.hdf5 \
        --camera eye_in_hand

    python robosuite/scripts/backfill_camera_from_states.py \
        --input /path/to/data_dir \
        --all-cameras \
        --height 256 \
        --width 256
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import glob
import json
import os
import shutil

import h5py
import numpy as np

import robosuite as suite


def list_hdf5_files(input_path: str) -> list[str]:
    if os.path.isfile(input_path):
        if not input_path.endswith(".hdf5"):
            raise ValueError(f"Expected a .hdf5 file, got: {input_path}")
        return [input_path]

    if not os.path.isdir(input_path):
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    files = sorted(glob.glob(os.path.join(input_path, "*.hdf5")))
    if len(files) == 0:
        raise FileNotFoundError(f"No .hdf5 files found under: {input_path}")
    return files


def get_demo_root_and_attrs(h5_file: h5py.File):
    if "demos" in h5_file:
        return h5_file["demos"], h5_file.attrs
    if "data" in h5_file:
        return h5_file["data"], h5_file["data"].attrs
    raise KeyError("Could not find a demo root group. Expected either 'demos' or 'data'.")


def decode_if_bytes(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.decode("utf-8")
    return value


def infer_hw_from_demo(demo_group: h5py.Group) -> tuple[int, int] | None:
    if "observations" not in demo_group:
        return None

    obs_group = demo_group["observations"]
    for cam_name in obs_group.keys():
        cam_group = obs_group[cam_name]
        if "images" not in cam_group:
            continue
        images = cam_group["images"]
        if images.ndim == 4 and images.shape[0] > 0:
            return int(images.shape[1]), int(images.shape[2])
    return None


def load_env_kwargs(attrs) -> dict:
    if "env_info" not in attrs:
        raise KeyError("Dataset is missing 'env_info', so the environment cannot be rebuilt.")

    env_info = decode_if_bytes(attrs["env_info"])
    if isinstance(env_info, str):
        env_kwargs = json.loads(env_info)
    else:
        env_kwargs = dict(env_info)
    return env_kwargs


def ensure_camera_names_attr(attrs, camera_names: list[str]) -> None:
    if "camera_names" not in attrs:
        attrs["camera_names"] = json.dumps(list(camera_names))
        return

    raw = decode_if_bytes(attrs["camera_names"])
    try:
        names = json.loads(raw) if isinstance(raw, str) else list(raw)
    except Exception:
        names = [str(raw)]

    changed = False
    for camera_name in camera_names:
        if camera_name not in names:
            names.append(camera_name)
            changed = True

    if changed:
        attrs["camera_names"] = json.dumps(names)


def reset_env_from_demo_xml(env, model_xml: str) -> None:
    env.reset()
    xml = env.edit_model_xml(model_xml)
    env.reset_from_xml_string(xml)
    env.sim.reset()
    env.sim.forward()


def get_available_cameras(env, model_xml: str) -> list[str]:
    reset_env_from_demo_xml(env, model_xml)
    camera_names = [str(name) for name in env.sim.model.camera_names]
    return sorted(camera_names)


def render_demo_cameras(
    env,
    model_xml: str,
    states: np.ndarray,
    camera_names: list[str],
    height: int,
    width: int,
) -> dict[str, np.ndarray]:
    reset_env_from_demo_xml(env, model_xml)
    available_cameras = {str(name) for name in env.sim.model.camera_names}

    missing_cameras = [cam for cam in camera_names if cam not in available_cameras]
    if missing_cameras:
        raise ValueError(
            f"Requested cameras not found in this model: {missing_cameras}. "
            f"Available cameras: {sorted(available_cameras)}"
        )

    frames = {cam: [] for cam in camera_names}
    for state in states:
        env.sim.set_state_from_flattened(state)
        env.sim.forward()
        for camera_name in camera_names:
            frame = env.sim.render(height=height, width=width, camera_name=camera_name)
            frames[camera_name].append(np.asarray(frame, dtype=np.uint8))

    outputs = {}
    for camera_name in camera_names:
        if len(frames[camera_name]) == 0:
            outputs[camera_name] = np.zeros((0, height, width, 3), dtype=np.uint8)
        else:
            outputs[camera_name] = np.stack(frames[camera_name], axis=0)
    return outputs


def process_file(file_path: str, args) -> None:
    print(f"\nProcessing: {file_path}")

    with h5py.File(file_path, "r") as h5_file:
        demo_root, attrs = get_demo_root_and_attrs(h5_file)
        env_kwargs = load_env_kwargs(attrs)
        demo_keys = sorted(list(demo_root.keys()))
        if len(demo_keys) == 0:
            print("  No demos found, skipping.")
            return

        if args.height is not None and args.width is not None:
            height = int(args.height)
            width = int(args.width)
        else:
            hw = infer_hw_from_demo(demo_root[demo_keys[0]])
            if hw is None:
                raise ValueError(
                    "Could not infer image size from existing observations. "
                    "Please pass both --height and --width."
                )
            height, width = hw

    env = suite.make(
        **env_kwargs,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera=None,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

    try:
        with h5py.File(file_path, "a") as h5_file:
            demo_root, attrs = get_demo_root_and_attrs(h5_file)
            demo_keys = sorted(list(demo_root.keys()))
            total = len(demo_keys)

            first_demo_group = demo_root[demo_keys[0]]
            if "model_file" not in first_demo_group.attrs:
                raise KeyError("First demo is missing model_file, so camera discovery cannot proceed.")

            first_model_xml = decode_if_bytes(first_demo_group.attrs["model_file"])
            discovered_cameras = get_available_cameras(env, first_model_xml)

            if args.all_cameras:
                requested_cameras = discovered_cameras
            else:
                requested_cameras = list(args.camera)
                unknown_cameras = [cam for cam in requested_cameras if cam not in discovered_cameras]
                if unknown_cameras:
                    raise ValueError(
                        f"Requested cameras not found: {unknown_cameras}. "
                        f"Available cameras: {discovered_cameras}"
                    )

            ensure_camera_names_attr(attrs, requested_cameras)
            print(f"  Target cameras: {requested_cameras}")

            for idx, demo_key in enumerate(demo_keys, start=1):
                demo_group = demo_root[demo_key]
                obs_group = demo_group.require_group("observations")

                if "states" not in demo_group:
                    print(f"  [{idx}/{total}] {demo_key}: missing states, skipping.")
                    continue

                if "model_file" not in demo_group.attrs:
                    print(f"  [{idx}/{total}] {demo_key}: missing model_file, skipping.")
                    continue

                model_xml = decode_if_bytes(demo_group.attrs["model_file"])
                states = demo_group["states"][()]
                demo_available_cameras = get_available_cameras(env, model_xml)
                if args.all_cameras:
                    target_cameras = demo_available_cameras
                else:
                    target_cameras = [cam for cam in requested_cameras if cam in demo_available_cameras]

                pending_cameras = []
                for camera_name in target_cameras:
                    has_existing = camera_name in obs_group and "images" in obs_group[camera_name]
                    if has_existing and not args.overwrite:
                        continue
                    pending_cameras.append(camera_name)

                if len(pending_cameras) == 0:
                    print(f"  [{idx}/{total}] {demo_key}: all target cameras already exist, skipping.")
                    continue

                print(
                    f"  [{idx}/{total}] {demo_key}: rendering {states.shape[0]} frames for cameras {pending_cameras}..."
                )
                rendered = render_demo_cameras(
                    env=env,
                    model_xml=model_xml,
                    states=states,
                    camera_names=pending_cameras,
                    height=height,
                    width=width,
                )

                for camera_name in pending_cameras:
                    if camera_name in obs_group:
                        del obs_group[camera_name]
                    cam_group = obs_group.create_group(camera_name)
                    cam_group.create_dataset(
                        "images",
                        data=rendered[camera_name],
                        dtype=np.uint8,
                        compression="gzip",
                        compression_opts=4,
                        chunks=True,
                    )
                    cam_group.attrs["generated_from_states"] = True
                    print(
                        f"    wrote {camera_name}: shape {rendered[camera_name].shape}"
                    )

                ensure_camera_names_attr(attrs, target_cameras)
    finally:
        env.close()


def maybe_copy_single_input(input_path: str, output_path: str | None) -> str:
    if output_path is None:
        return input_path

    if os.path.isdir(input_path):
        raise ValueError("--output is only supported when --input points to a single .hdf5 file.")

    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    shutil.copy2(input_path, output_path)
    return output_path


def copy_inputs_to_output_dir(file_paths: list[str], output_dir: str) -> list[str]:
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    target_paths = []
    for file_path in file_paths:
        target_path = os.path.join(output_dir, os.path.basename(file_path))
        shutil.copy2(file_path, target_path)
        target_paths.append(target_path)
    return target_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to a single .hdf5 file or a directory containing .hdf5 files.",
    )
    parser.add_argument(
        "--camera",
        nargs="*",
        default=None,
        help="One or more camera names to render, e.g. eye_in_hand robotview. "
        "If omitted, use --all-cameras.",
    )
    parser.add_argument(
        "--all-cameras",
        action="store_true",
        help="Recover every camera exposed by the robosuite model for each demo.",
    )
    parser.add_argument("--height", type=int, default=None, help="Rendered image height. Defaults to existing dataset size.")
    parser.add_argument("--width", type=int, default=None, help="Rendered image width. Defaults to existing dataset size.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the target camera group if it already exists.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output .hdf5 path. Only valid when --input is a single file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory. When set, each input .hdf5 is copied there first "
        "and the recovered file keeps the same basename.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of workers. Parallelism is applied across hdf5 files.",
    )
    parser.add_argument(
        "--worker-type",
        type=str,
        default="process",
        choices=["thread", "process"],
        help="Parallel worker type. 'process' is usually faster / safer for MuJoCo rendering.",
    )
    args = parser.parse_args()

    if (args.height is None) ^ (args.width is None):
        raise ValueError("Please provide both --height and --width together.")
    if not args.all_cameras and not args.camera:
        raise ValueError("Please provide --camera ... or enable --all-cameras.")
    if args.all_cameras and args.camera:
        raise ValueError("Use either --camera ... or --all-cameras, not both.")
    if args.output is not None and args.output_dir is not None:
        raise ValueError("Use either --output or --output-dir, not both.")
    if args.num_workers <= 0:
        raise ValueError("--num-workers must be >= 1.")

    input_path = os.path.abspath(args.input)
    file_paths = list_hdf5_files(input_path)

    if len(file_paths) == 1 and args.output is not None:
        target_file = maybe_copy_single_input(file_paths[0], args.output)
        process_file(target_file, args)
        print(f"\nFinished. Updated file: {target_file}")
        return

    if args.output is not None and len(file_paths) > 1:
        raise ValueError("--output cannot be used when --input resolves to multiple files.")

    if args.output_dir is not None:
        file_paths = copy_inputs_to_output_dir(file_paths, args.output_dir)

    num_workers = min(int(args.num_workers), len(file_paths))
    if num_workers == 1:
        for file_path in file_paths:
            process_file(file_path, args)
    else:
        executor_cls = ProcessPoolExecutor if args.worker_type == "process" else ThreadPoolExecutor
        print(
            f"Running recovery with {num_workers} {args.worker_type} workers across {len(file_paths)} files."
        )
        future_to_path = {}
        executor_kwargs = {"max_workers": num_workers}
        if args.worker_type == "thread":
            executor_kwargs["thread_name_prefix"] = "backfill"

        with executor_cls(**executor_kwargs) as executor:
            for file_path in file_paths:
                future = executor.submit(process_file, file_path, args)
                future_to_path[future] = file_path

            for future in as_completed(future_to_path):
                file_path = future_to_path[future]
                try:
                    future.result()
                except Exception as exc:
                    raise RuntimeError(f"Failed while processing {file_path}: {exc}") from exc

    print("\nFinished.")


if __name__ == "__main__":
    main()
