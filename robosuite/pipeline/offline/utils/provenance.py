"""Reproducibility metadata for offline policy and discriminator runs."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable

from robosuite.pipeline.offline.discriminator.contracts import sha256_file


def git_provenance(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()

    def _git(*args: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain=v1", "--untracked-files=normal")
    diff = subprocess.run(
        ["git", "-C", str(root), "diff", "--binary", "HEAD"],
        check=True,
        capture_output=True,
    ).stdout
    untracked_paths = [
        root / line
        for line in _git("ls-files", "--others", "--exclude-standard").splitlines()
        if line
    ]
    return {
        "commit": commit,
        "dirty": bool(status),
        "status": status.splitlines(),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "untracked_files": [
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for path in untracked_paths
            if path.is_file()
        ],
    }


def file_provenance(path_value: str | Path) -> dict[str, Any]:
    path = Path(path_value).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Provenance input is not a file: {path}")
    return {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def file_set_provenance(paths: Iterable[str | Path]) -> dict[str, Any]:
    entries = [file_provenance(path) for path in sorted(map(Path, paths))]
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return {
        "files": entries,
        "aggregate_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def directory_input_provenance(path_value: str | Path) -> dict[str, Any]:
    """Hash policy-training manifests and HDF5 files, excluding rendered videos."""
    root = Path(path_value).resolve()
    if root.is_file():
        return file_set_provenance([root])
    if not root.is_dir():
        raise FileNotFoundError(f"Provenance input does not exist: {root}")
    candidates = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".h5", ".hdf5", ".json"}
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No HDF5 or manifest inputs found for provenance under {root}."
        )
    result = file_set_provenance(candidates)
    result["root"] = str(root)
    return result


def manifest_shard_provenance(
    manifest_path: str | Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Hash exactly the latent shards referenced by a discriminator manifest."""
    resolved_manifest = Path(manifest_path).resolve()
    root = resolved_manifest.parent
    shard_paths: list[Path] = []
    for split_entries in dict(manifest.get("splits", {})).values():
        for entry in list(split_entries or []):
            shard_paths.append((root / str(entry["path"])).resolve())
    result = file_set_provenance(shard_paths)
    result["manifest"] = file_provenance(resolved_manifest)
    return result


__all__ = [
    "directory_input_provenance",
    "file_provenance",
    "file_set_provenance",
    "git_provenance",
    "manifest_shard_provenance",
]
