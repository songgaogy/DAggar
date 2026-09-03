"""Build the CUDA-only frozen-DINOv3 cache used by RPT pretraining."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import h5py
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from robosuite.discriminator.dyn_disc.data.rpt_cache_dataset import (
    CACHE_VERSION,
    MANIFEST_NAME,
    load_cache_manifest,
    manifest_fingerprint,
)
from robosuite.discriminator.dyn_disc.models.dinov3_encoder import DINOv3Encoder
from robosuite.discriminator.utils.robosuite_benchmark import canonical_task_name


DEFAULT_INPUTS = (
    "data/NutAssemblyRound/success_rollout",
    "data/NutAssemblySquare/success_rollout",
    "data/PickPlaceBread/success_rollout",
    "data/PickPlaceCan/success_rollout",
    "data/PickPlaceCereal/success_rollout",
    "data/PickPlaceMilk/success_rollout",
    "data/Stack/success_rollout",
)
DEFAULT_CACHE_DIR = "checkpoints/dyn_disc/ablations/RPT/cache/dinov3_mean_v1"
DEFAULT_MODEL_PATH = "data/pretrained/dinov3-vitb16-pretrain-lvd1689m_80M"
VIEW_NAMES = ("agentview", "robot0_eye_in_hand")
PROPRIO_MAP = {
    "NutAssemblyRound": (1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27, 28, 29, 30),
    "NutAssemblySquare": (1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27, 28, 29, 30),
    "Stack": (1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27, 28, 29, 30),
    "PickPlaceBread": (1, 2, 3, 4, 5, 6, 7, 38, 39, 40, 41, 42, 43, 44),
    "PickPlaceCan": (1, 2, 3, 4, 5, 6, 7, 38, 39, 40, 41, 42, 43, 44),
    "PickPlaceCereal": (1, 2, 3, 4, 5, 6, 7, 38, 39, 40, 41, 42, 43, 44),
    "PickPlaceMilk": (1, 2, 3, 4, 5, 6, 7, 38, 39, 40, 41, 42, 43, 44),
}


class _HDF5FrameBatchDataset(Dataset):
    """Load aligned frame batches with one lazy HDF5 handle per worker."""

    def __init__(
        self,
        plan: Mapping[str, Any],
        *,
        image_batch_size: int,
        proprio_indices: Sequence[int],
        io_chunk_frames: int,
    ) -> None:
        self.path = str(plan["path"])
        self.demo_keys = tuple(str(key) for key in plan["demo_keys"])
        self.proprio_indices = np.asarray(proprio_indices, dtype=np.int64)
        self._handle: Optional[h5py.File] = None

        episodes = []
        native_chunks = []
        running = 0
        with h5py.File(self.path, "r") as handle:
            for demo_key in self.demo_keys:
                demo = handle["demos"][demo_key]
                observations = demo["observations"]
                length = min(
                    int(demo["states"].shape[0]),
                    int(demo["actions"].shape[0]),
                    *(int(observations[view]["images"].shape[0]) for view in VIEW_NAMES),
                )
                episodes.append((demo_key, length, running))
                running += length
                for view in VIEW_NAMES:
                    chunks = observations[view]["images"].chunks
                    if chunks:
                        native_chunks.append(int(chunks[0]))

        native_chunk_sizes = set(native_chunks)
        if int(io_chunk_frames) > 0:
            alignment = int(io_chunk_frames)
            incompatible = sorted(size for size in native_chunk_sizes if alignment % size)
            if incompatible:
                raise ValueError(
                    f"io_chunk_frames={alignment} is not a multiple of native HDF5 "
                    f"time chunks {incompatible}"
                )
        else:
            if len(native_chunk_sizes) > 1:
                raise ValueError(
                    "Camera datasets use different HDF5 time chunks; set "
                    "--io-chunk-frames to a common multiple"
                )
            alignment = next(iter(native_chunk_sizes), 1)
        max_frames = int(image_batch_size) // len(VIEW_NAMES)
        if max_frames < alignment:
            raise ValueError(
                f"image batch size {image_batch_size} is too small for two views and "
                f"the {alignment}-frame HDF5 chunk; use at least {alignment * len(VIEW_NAMES)}"
            )
        self.frames_per_batch = max(alignment, (max_frames // alignment) * alignment)
        self.total_frames = running
        self.episode_ends = [start + length for _, length, start in episodes]
        self.batch_specs = []
        for demo_key, length, output_start in episodes:
            for start in range(0, length, self.frames_per_batch):
                stop = min(length, start + self.frames_per_batch)
                self.batch_specs.append((demo_key, start, stop, output_start + start))

    def __len__(self) -> int:
        return len(self.batch_specs)

    def _file(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.path, "r", rdcc_nbytes=32 * 1024**2)
        return self._handle

    def __getitem__(self, index: int) -> Dict[str, Any]:
        demo_key, start, stop, output_start = self.batch_specs[index]
        demo = self._file()["demos"][demo_key]
        states = np.asarray(demo["states"][start:stop], dtype=np.float32)
        if int(self.proprio_indices.max()) >= states.shape[1]:
            raise ValueError(
                f"Proprio indices exceed state_dim={states.shape[1]} for {demo_key}"
            )
        actions = np.asarray(demo["actions"][start:stop], dtype=np.float32)
        if actions.ndim != 2:
            raise ValueError(f"Actions must be rank two for {demo_key}, got {actions.shape}")
        if actions.shape[1] < 7:
            actions = np.pad(actions, ((0, 0), (0, 7 - actions.shape[1])))
        observations = demo["observations"]
        images = np.stack(
            [np.asarray(observations[view]["images"][start:stop]) for view in VIEW_NAMES],
            axis=1,
        )
        return {
            "images": torch.from_numpy(np.ascontiguousarray(images)),
            "proprio": torch.from_numpy(states[:, self.proprio_indices].copy()),
            "actions": torch.from_numpy(actions[:, :7].copy()),
            "output_start": int(output_start),
        }

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        state["_handle"] = None
        return state

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is not None:
            handle.close()


def _expand_inputs(inputs: Sequence[str]) -> list[Path]:
    files: set[Path] = set()
    for value in inputs:
        path = Path(value).expanduser()
        if path.is_dir():
            files.update(candidate.resolve() for candidate in path.glob("*.hdf5"))
        elif path.is_file() and path.suffix == ".hdf5":
            files.add(path.resolve())
        else:
            for candidate in glob.glob(value):
                candidate_path = Path(candidate)
                if candidate_path.is_file() and candidate_path.suffix == ".hdf5":
                    files.add(candidate_path.resolve())
    if not files:
        raise FileNotFoundError(f"No HDF5 inputs found under: {list(inputs)}")
    return sorted(files)


def _task_name(path: Path, proprio_map: Mapping[str, Sequence[int]]) -> str:
    tasks = tuple(sorted(proprio_map, key=len, reverse=True))
    for part in reversed(path.parts):
        for text in (part, Path(part).stem):
            if text in proprio_map:
                return text
            aliased = canonical_task_name(text)
            if aliased in proprio_map:
                return aliased
            if text.startswith("Panda") and text[len("Panda") :] in proprio_map:
                return text[len("Panda") :]
            matches = [task for task in tasks if task in text]
            if matches:
                return matches[0]
    raise ValueError(f"Cannot infer task name from {path}; expected one of {sorted(proprio_map)}")


def _valid_demo_keys(handle: h5py.File, max_count: int) -> list[str]:
    if "demos" not in handle:
        return []
    keys = []
    for key in sorted(handle["demos"].keys()):
        demo = handle["demos"][key]
        if "states" not in demo or "actions" not in demo or "observations" not in demo:
            continue
        if any(
            view not in demo["observations"] or "images" not in demo["observations"][view]
            for view in VIEW_NAMES
        ):
            continue
        lengths = [int(demo["states"].shape[0]), int(demo["actions"].shape[0])]
        lengths.extend(int(demo["observations"][view]["images"].shape[0]) for view in VIEW_NAMES)
        if min(lengths) <= 0:
            continue
        keys.append(key)
        if len(keys) >= max_count:
            break
    return keys


def _source_plans(
    files: Sequence[Path], proprio_map: Mapping[str, Sequence[int]], max_per_task: int
) -> list[Dict[str, Any]]:
    counts = {task: 0 for task in proprio_map}
    plans = []
    for path in files:
        task = _task_name(path, proprio_map)
        remaining = max_per_task - counts[task]
        if remaining <= 0:
            continue
        with h5py.File(path, "r") as handle:
            demo_keys = _valid_demo_keys(handle, remaining)
        if not demo_keys:
            continue
        counts[task] += len(demo_keys)
        stat = path.stat()
        source_id = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:16]
        plans.append(
            {
                "path": str(path),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "task": task,
                "demo_keys": demo_keys,
                "shard": f"shards/{task}_{source_id}.pt",
            }
        )
    missing = [task for task, count in counts.items() if count == 0]
    if missing:
        raise RuntimeError(f"No valid success demos found for tasks: {missing}")
    return plans


def _build_shard(
    plan: Mapping[str, Any],
    cache_dir: Path,
    encoder: DINOv3Encoder,
    device: torch.device,
    batch_size: int,
    proprio_map: Mapping[str, Sequence[int]],
    num_workers: int,
    prefetch_factor: int,
    io_chunk_frames: int,
) -> Dict[str, Any]:
    indices = np.asarray(proprio_map[str(plan["task"])], dtype=np.int64)
    dataset = _HDF5FrameBatchDataset(
        plan,
        image_batch_size=batch_size,
        proprio_indices=indices,
        io_chunk_frames=io_chunk_frames,
    )
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=int(num_workers),
        pin_memory=True,
        prefetch_factor=int(prefetch_factor) if int(num_workers) > 0 else None,
        multiprocessing_context="spawn" if int(num_workers) > 0 else None,
        in_order=False,
    )
    visual = torch.empty(
        dataset.total_frames, len(VIEW_NAMES), encoder.latent_dim,
        dtype=torch.float32, pin_memory=True,
    )
    proprio = torch.empty(dataset.total_frames, 14, dtype=torch.float32)
    actions = torch.empty(dataset.total_frames, 7, dtype=torch.float32)
    torch.cuda.synchronize(device)
    started = torch.cuda.Event(enable_timing=True)
    finished = torch.cuda.Event(enable_timing=True)
    started.record()
    rank = os.environ.get("RANK", "0")
    encoded_frames = 0
    progress = tqdm(
        loader,
        desc=f"[rpt][cache][rank {rank}] {plan['task']}",
        total=len(dataset),
        unit="batch",
        dynamic_ncols=True,
        position=int(rank),
        leave=True,
    )
    for batch in progress:
        if int(num_workers) > 0 and not batch["images"].is_pinned():
            raise RuntimeError("HDF5 worker output is not pinned; asynchronous H2D is unavailable")
        output_start = int(batch["output_start"])
        count = int(batch["images"].shape[0])
        output_stop = output_start + count
        images = batch["images"].to(device, non_blocking=True)
        images = images.permute(0, 1, 4, 2, 3)
        images = images.reshape(count * len(VIEW_NAMES), *images.shape[2:])
        latents = encoder.encode_images(images).view(count, len(VIEW_NAMES), -1)
        visual[output_start:output_stop].copy_(latents, non_blocking=True)
        proprio[output_start:output_stop].copy_(batch["proprio"])
        actions[output_start:output_stop].copy_(batch["actions"])
        encoded_frames += count
        progress.set_postfix(frames=f"{encoded_frames}/{dataset.total_frames}", refresh=False)
    finished.record()
    torch.cuda.synchronize(device)
    elapsed_s = max(started.elapsed_time(finished) / 1000.0, 1e-9)
    tqdm.write(
        f"[rpt][cache][rank {rank}] "
        f"{plan['task']} frames={dataset.total_frames} images/s="
        f"{dataset.total_frames * len(VIEW_NAMES) / elapsed_s:.1f} "
        f"workers={num_workers} prefetch={prefetch_factor} "
        f"frames/forward<={dataset.frames_per_batch}"
    )
    shard = {
        "visual_latents": visual,
        "proprio": proprio,
        "actions": actions,
        "episode_ends": torch.tensor(dataset.episode_ends, dtype=torch.int64),
        "source": dict(plan),
        "proprio_min": proprio.amin(dim=0),
        "proprio_max": proprio.amax(dim=0),
    }
    output_path = cache_dir / str(plan["shard"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(f".rank{os.environ.get('RANK', '0')}.tmp")
    torch.save(shard, temporary)
    os.replace(temporary, output_path)
    return {
        "path": str(plan["shard"]),
        "frames": dataset.total_frames,
        "episodes": len(dataset.episode_ends),
    }


def _manifest_header(
    plans: Sequence[Mapping[str, Any]],
    encoder: DINOv3Encoder,
    proprio_map: Mapping[str, Sequence[int]],
    max_per_task: int,
    image_batch_size: int,
    io_chunk_frames: int,
) -> Dict[str, Any]:
    return {
        "cache_version": CACHE_VERSION,
        "sources": [dict(plan) for plan in plans],
        "encoder": encoder.manifest_config(),
        "view_names": list(VIEW_NAMES),
        "latent_dim": 768,
        "latent_dtype": "float32",
        "proprio_dim": 14,
        "action_dim": 7,
        "inference_batching": {
            "version": 2,
            "view_batching": "frame_major_combined",
            "image_batch_size": int(image_batch_size),
            "io_chunk_frames": int(io_chunk_frames),
        },
        "proprio_map": {task: list(indices) for task, indices in sorted(proprio_map.items())},
        "max_trajectories_per_task": int(max_per_task),
    }


def _validate_existing(cache_dir: Path, expected: Mapping[str, Any]) -> bool:
    manifest_path = cache_dir / MANIFEST_NAME
    if not manifest_path.exists():
        return False
    manifest = load_cache_manifest(cache_dir)
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Existing RPT cache manifest does not match current {key}; use a new cache directory")
    for shard in manifest.get("shards", []):
        if not (cache_dir / shard["path"]).is_file():
            raise FileNotFoundError(f"Existing RPT cache is missing shard {shard['path']}")
    return True


def build_rpt_cache(
    inputs: Sequence[str] = DEFAULT_INPUTS,
    cache_dir: str = DEFAULT_CACHE_DIR,
    model_path: str = DEFAULT_MODEL_PATH,
    batch_size: int = 800,
    max_trajectories_per_task: int = 100,
    num_workers: int = 2,
    prefetch_factor: int = 2,
    io_chunk_frames: int = 0,
    encoder: Optional[DINOv3Encoder] = None,
    proprio_map: Mapping[str, Sequence[int]] = PROPRIO_MAP,
) -> Optional[Dict[str, Any]]:
    """Build assigned shards; rank zero returns the completed manifest."""
    if not torch.cuda.is_available():
        raise RuntimeError("RPT cache generation requires CUDA")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if int(max_trajectories_per_task) <= 0:
        raise ValueError("max_trajectories_per_task must be positive")
    if int(num_workers) < 0 or int(prefetch_factor) <= 0 or int(io_chunk_frames) < 0:
        raise ValueError("num_workers/io_chunk_frames must be non-negative and prefetch_factor positive")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if encoder is None:
        encoder = DINOv3Encoder(model_path=model_path, view_names=VIEW_NAMES).to(device)
    else:
        encoder = encoder.to(device)
    encoder.eval()

    files = _expand_inputs(inputs)
    plans = _source_plans(files, proprio_map, int(max_trajectories_per_task))
    output_dir = Path(cache_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    header = _manifest_header(
        plans,
        encoder,
        proprio_map,
        max_trajectories_per_task,
        int(batch_size),
        int(io_chunk_frames),
    )
    if _validate_existing(output_dir, header):
        return load_cache_manifest(output_dir) if rank == 0 else None

    for plan in plans[rank::world_size]:
        _build_shard(
            plan, output_dir, encoder, device, int(batch_size), proprio_map,
            int(num_workers), int(prefetch_factor), int(io_chunk_frames),
        )
    if world_size > 1:
        dist.barrier()
    if rank != 0:
        return None

    shards = []
    global_min = None
    global_max = None
    total_frames = 0
    total_episodes = 0
    for plan in plans:
        path = output_dir / str(plan["shard"])
        if not path.is_file():
            raise FileNotFoundError(f"Rank output shard missing: {path}")
        try:
            shard = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        except TypeError:
            shard = torch.load(path, map_location="cpu")
        frames = int(shard["visual_latents"].shape[0])
        episodes = int(shard["episode_ends"].numel())
        shards.append({"path": str(plan["shard"]), "frames": frames, "episodes": episodes})
        total_frames += frames
        total_episodes += episodes
        shard_min, shard_max = shard["proprio_min"], shard["proprio_max"]
        global_min = shard_min if global_min is None else torch.minimum(global_min, shard_min)
        global_max = shard_max if global_max is None else torch.maximum(global_max, shard_max)

    manifest = dict(header)
    manifest.update(
        {
            "shards": shards,
            "total_frames": total_frames,
            "total_episodes": total_episodes,
            "statistics": {"proprio_min": global_min.tolist(), "proprio_max": global_max.tolist()},
        }
    )
    manifest["fingerprint"] = manifest_fingerprint(manifest)
    temporary = output_dir / f".{MANIFEST_NAME}.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, output_dir / MANIFEST_NAME)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", default=list(DEFAULT_INPUTS))
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--batch-size", type=int, default=800,
        help="Maximum total camera images per DINO forward.",
    )
    parser.add_argument("--max-trajectories-per-task", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--io-chunk-frames", type=int, default=0,
        help="Frame quantum; 0 uses the native time chunk, otherwise use a common multiple.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_rpt_cache(
        inputs=args.input,
        cache_dir=args.cache_dir,
        model_path=args.model_path,
        batch_size=args.batch_size,
        max_trajectories_per_task=args.max_trajectories_per_task,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        io_chunk_frames=args.io_chunk_frames,
    )
    if manifest is not None:
        print(
            f"RPT cache ready: {args.cache_dir} "
            f"({manifest['total_episodes']} episodes, {manifest['total_frames']} frames)"
        )


if __name__ == "__main__":
    main()
