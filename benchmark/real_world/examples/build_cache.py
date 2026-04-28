"""Build raw array cache for the real-world Agilex benchmark.

The cache keeps raw benchmark inputs in a source-agnostic layout:
    <cache_root>/<task>/<sha1>/

Each trajectory directory stores images, proprio, actions, labels, and enough
metadata to reconstruct an AgilexBenchmarkTrajectory without reopening HDF5.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from benchmark.real_world import FailureBenchmark


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fail-root", required=True)
    parser.add_argument("--success-root", required=True)
    parser.add_argument("--cache-root", default="data/agilex_train_cache")
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--max-fail-per-task", type=int, default=None)
    parser.add_argument("--max-success-per-task", type=int, default=None)
    parser.add_argument("--cameras", nargs="*", default=None,
                        help="Camera streams to cache. Omit to cache all available cameras.")
    parser.add_argument("--image-size", type=int, default=224,
                        help="Center-crop and resize cached images to this square size.")
    parser.add_argument("--proprio-field", type=str, default="qpos")
    parser.add_argument("--proprio-start", type=int, default=7)
    parser.add_argument("--proprio-stop", type=int, default=14)
    parser.add_argument("--action-start", type=int, default=7)
    parser.add_argument("--action-stop", type=int, default=14)
    return parser.parse_args()


def _cache_key(traj, cameras: Sequence[str]) -> str:
    payload = "|".join(
        [
            str(traj.task_name),
            str(traj.video_id),
            str(getattr(traj, "file_path", "")),
            str(getattr(traj, "episode_path", "")),
            ",".join(cameras),
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _ensure_local_cache_root(cache_root: Path) -> None:
    repo_data = (Path.cwd() / "data").resolve()
    resolved = cache_root.resolve()
    if repo_data not in (resolved, *resolved.parents):
        raise ValueError(
            f"cache_root must be under {repo_data}; got {resolved}"
        )
    for parent in [cache_root, *cache_root.parents]:
        if parent.exists() and parent.is_symlink():
            raise ValueError(
                f"cache_root path must not contain symlinks; found {parent}"
            )
        if parent.resolve() == repo_data.parent:
            break


def _empty_images(num_frames: int) -> np.ndarray:
    return np.zeros((int(num_frames), 0, 3, 0, 0), dtype=np.uint8)


def _center_crop_resize(img: np.ndarray, out_size: int) -> np.ndarray:
    h, w = int(img.shape[0]), int(img.shape[1])
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    crop = img[y0 : y0 + side, x0 : x0 + side, :]
    if side == int(out_size):
        return crop
    ys = np.linspace(0, side - 1, int(out_size)).astype(np.int32)
    xs = np.linspace(0, side - 1, int(out_size)).astype(np.int32)
    return crop[ys][:, xs]


def _resize_images_thwc(images: np.ndarray, out_size: int) -> np.ndarray:
    arr = np.asarray(images, dtype=np.uint8)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"images must be (T, H, W, 3), got {arr.shape}")
    out = np.empty((arr.shape[0], int(out_size), int(out_size), 3), dtype=np.uint8)
    for i in range(int(arr.shape[0])):
        out[i] = _center_crop_resize(arr[i], int(out_size))
    return out


def _stack_images_chw(
    images_by_cam: dict[str, np.ndarray],
    cameras: Sequence[str],
    image_size: int,
) -> np.ndarray:
    if not cameras:
        first_len = 0
        if images_by_cam:
            first_len = int(next(iter(images_by_cam.values())).shape[0])
        return _empty_images(first_len)
    arrays = []
    for cam in cameras:
        arr = np.asarray(images_by_cam[cam], dtype=np.uint8)
        if arr.ndim != 4 or arr.shape[-1] != 3:
            raise ValueError(f"camera {cam!r} has invalid image shape {arr.shape}")
        arr = _resize_images_thwc(arr, int(image_size))
        arrays.append(np.transpose(arr, (0, 3, 1, 2)))
    return np.stack(arrays, axis=1)


def _write_one(
    cache_root: Path,
    traj,
    cameras_arg: Sequence[str] | None,
    image_size: int,
) -> str:
    cameras = (
        list(cameras_arg)
        if cameras_arg is not None
        else list(traj.available_cameras)
    )
    images_by_cam = traj.load_images(cameras=cameras) if cameras else {}
    images_chw = _stack_images_chw(images_by_cam, cameras, int(image_size))
    proprio = np.asarray(traj.load_states(), dtype=np.float32)
    actions = np.asarray(traj.load_actions(), dtype=np.float32)

    if traj.is_failure:
        failure_mask = np.asarray(traj.load_failure_mask(), dtype=np.uint8).reshape(-1)
        seg_idx = np.asarray(traj.load_failure_segment_index(), dtype=np.int32).reshape(-1)
    else:
        failure_mask = np.zeros((int(traj.num_frames),), dtype=np.uint8)
        seg_idx = np.full((int(traj.num_frames),), -1, dtype=np.int32)

    lengths = [
        int(traj.num_frames),
        int(proprio.shape[0]),
        int(actions.shape[0]),
        int(failure_mask.shape[0]),
        int(seg_idx.shape[0]),
    ]
    if images_chw.ndim >= 1:
        lengths.append(int(images_chw.shape[0]))
    T = int(min(lengths))
    if T <= 0:
        raise ValueError(f"empty trajectory after alignment: {traj.describe()}")

    task_dir = cache_root / str(traj.task_name)
    task_dir.mkdir(parents=True, exist_ok=True)
    out_path = task_dir / _cache_key(traj, cameras)
    out_path.mkdir(parents=True, exist_ok=True)

    np.save(str(out_path / "images_chw.npy"), images_chw[:T], allow_pickle=False)
    np.save(str(out_path / "proprio.npy"), proprio[:T], allow_pickle=False)
    np.save(str(out_path / "actions.npy"), actions[:T], allow_pickle=False)
    np.save(str(out_path / "failure_mask.npy"), failure_mask[:T], allow_pickle=False)
    np.save(
        str(out_path / "failure_segment_index.npy"),
        seg_idx[:T],
        allow_pickle=False,
    )
    metadata = {
        "cache_format": "agilex_raw_array_v1",
        "task_name": str(traj.task_name),
        "video_id": str(traj.video_id),
        "is_failure": bool(traj.is_failure),
        "num_frames": int(T),
        "camera_names": list(cameras),
        "image_size": int(image_size),
        "file_path": str(getattr(traj, "file_path", "")),
        "episode_path": str(getattr(traj, "episode_path", "")),
        "source_hdf5_path": str(getattr(traj, "source_hdf5_path", "")),
        "source_demo_key": str(getattr(traj, "source_demo_key", "")),
        "failure_segments": list(traj.failure_segments),
    }
    with open(out_path / "metadata.json", "w") as fp:
        json.dump(metadata, fp, indent=2)
    return str(out_path)


def main() -> None:
    args = _parse_args()
    bench = FailureBenchmark(
        fail_labeled_root=args.fail_root,
        success_root=args.success_root,
        tasks=args.tasks,
        max_fail_per_task=args.max_fail_per_task,
        max_success_per_task=args.max_success_per_task,
        proprio_field=args.proprio_field,
        proprio_slice=slice(int(args.proprio_start), int(args.proprio_stop)),
        action_slice=slice(int(args.action_start), int(args.action_stop)),
    )
    trajs = bench.trajectories()
    cache_root = Path(args.cache_root).expanduser()
    _ensure_local_cache_root(cache_root)
    cache_root = cache_root.resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    n_fail = sum(1 for t in trajs if t.is_failure)
    print(
        f"[real_world][cache] discovered {len(trajs)} trajectories "
        f"(failure={n_fail}, success={len(trajs) - n_fail})"
    )
    written = 0
    skipped = 0
    cameras = None if args.cameras is None else list(args.cameras)
    for traj in trajs:
        try:
            out_path = _write_one(cache_root, traj, cameras, int(args.image_size))
            written += 1
            print(f"[real_world][cache] wrote {out_path}")
        except Exception as exc:
            skipped += 1
            print(f"[real_world][cache] skip {traj.describe()}: {exc}")
    print(f"[real_world][cache] done written={written} skipped={skipped} root={cache_root}")


if __name__ == "__main__":
    main()
