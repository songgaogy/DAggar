from __future__ import annotations

import datetime
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def resolve_path(reference: str | Path | None) -> Path | None:
    if reference is None or str(reference).strip().lower() in {"", "none", "null"}:
        return None
    return Path(reference).expanduser().resolve()


def create_run_directory(
    output_root: str | Path, task_name: str, run_name: str | None = None
) -> tuple[str, Path]:
    if run_name is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{task_name}_{timestamp}"
    run_dir = Path(output_root).expanduser().resolve() / task_name / run_name
    if run_dir.exists():
        raise FileExistsError(f"DSRL run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    return run_name, run_dir


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # A checkpoint loaded with map_location="cuda" also moves RNG byte tensors.
    # PyTorch generator APIs require their serialized states on the host.
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def file_identity(path: str | Path, *, hash_bytes: int = 1 << 20) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    stat = source.stat()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        remaining = int(hash_bytes)
        while remaining > 0:
            chunk = handle.read(min(remaining, 1 << 20))
            if not chunk:
                break
            digest.update(chunk)
            remaining -= len(chunk)
    return {
        "path": os.fspath(source),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "prefix_sha256": digest.hexdigest(),
    }
