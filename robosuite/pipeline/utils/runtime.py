from __future__ import annotations

import datetime
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def resolve_path(reference: str | Path | None) -> Path | None:
    if reference is None or str(reference).strip().lower() in {"", "none", "null"}:
        return None
    path = Path(reference).expanduser().resolve()
    if not path.is_dir():
        return path
    candidate = path / "latest.pt" if path.name == "checkpoints" else path / "checkpoints" / "latest.pt"
    if not candidate.exists():
        raise FileNotFoundError(f"No checkpoints/latest.pt found under {path}.")
    return candidate


def create_run_directory(
    output_root: str | Path,
    task_name: str,
    run_name: str | None = None,
) -> tuple[str, Path]:
    if run_name is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"awr_{timestamp}"
    run_dir = Path(output_root).expanduser().resolve() / str(task_name) / str(run_name)
    if run_dir.exists():
        raise FileExistsError(f"AWR run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    return str(run_name), run_dir


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(destination)
