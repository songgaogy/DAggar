"""Filesystem and state contract for batch-online DIPOLE runs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping

from tqdm import tqdm

ROUND_STAGES = ("collection", "disc", "vast", "policy")
_STAGE_STATUSES = {"pending", "running", "completed", "failed"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class RunLayout:
    """Canonical paths rooted at one batch-online run directory."""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).resolve())

    @property
    def config_path(self) -> Path:
        return self.root / "config_resolved.yaml"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def state_path(self) -> Path:
        return self.root / "state.json"

    @property
    def inputs_dir(self) -> Path:
        return self.root / "inputs"

    @property
    def checkpoints_dir(self) -> Path:
        return self.inputs_dir / "checkpoints"

    @property
    def data_dir(self) -> Path:
        return self.inputs_dir / "data"

    @property
    def cache_dir(self) -> Path:
        return self.root / "cache"

    @property
    def rounds_dir(self) -> Path:
        return self.root / "rounds"

    def round_dir(self, round_index: int) -> Path:
        _validate_round_index(round_index)
        return self.rounds_dir / f"{round_index:03d}"

    def round_data_dir(self, round_index: int) -> Path:
        return self.round_dir(round_index) / "data"

    def eval_vis_dir(self, round_index: int) -> Path:
        """Return the auxiliary evaluation and visualization root for a round."""

        return self.round_dir(round_index) / "eval_vis"

    def disc_eval_vis_dir(self, round_index: int) -> Path:
        """Return the discriminator evaluation and visualization output root."""

        return self.eval_vis_dir(round_index) / "disc"

    def vast_eval_vis_dir(self, round_index: int) -> Path:
        """Return the VAST visualization output root."""

        return self.eval_vis_dir(round_index) / "vast"

    def stage_dir(self, round_index: int, stage: str) -> Path:
        _validate_training_stage(stage)
        return self.round_dir(round_index) / stage

    def attempt_dir(self, round_index: int, stage: str, attempt: int) -> Path:
        if attempt < 1:
            raise ValueError("attempt must be at least 1")
        return self.stage_dir(round_index, stage) / "attempts" / f"{attempt:03d}"

    def create_base_directories(self) -> None:
        for path in (
            self.checkpoints_dir,
            self.data_dir,
            self.cache_dir,
            self.rounds_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def create_round_directories(self, round_index: int) -> None:
        self.round_data_dir(round_index).mkdir(parents=True, exist_ok=True)
        for stage in ROUND_STAGES[1:]:
            self.stage_dir(round_index, stage).mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class InputSnapshot:
    """One immutable input copied below ``run_root/inputs``."""

    name: str
    source: Path
    destination: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", Path(self.source))
        if not self.name or not self.name.strip():
            raise ValueError("snapshot name must not be empty")
        relative = PurePosixPath(self.destination)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("snapshot destination must be a safe relative path")
        if not relative.parts or relative.parts[0] not in {"checkpoints", "data"}:
            raise ValueError("snapshot destination must be under checkpoints/ or data/")


@dataclass(frozen=True)
class InputReference:
    """One immutable input retained at its external source path."""

    name: str
    source: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", Path(self.source))
        if not self.name or not self.name.strip():
            raise ValueError("reference name must not be empty")


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace a JSON file and fsync its containing directory."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def load_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def write_text_atomic(path: Path, text: str) -> None:
    """Atomically replace a UTF-8 text file and fsync its containing directory."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def sha256_file(
    path: Path,
    *,
    chunk_size: int = 1024 * 1024,
    progress: Callable[[int], None] | None = None,
) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
            if progress is not None:
                progress(len(chunk))
    return digest.hexdigest()


def create_run(layout: RunLayout, *, task_name: str) -> dict[str, Any]:
    """Create an empty run with round 000 ready for collection."""

    if not task_name or not task_name.strip():
        raise ValueError("task_name must not be empty")
    if layout.state_path.exists() or layout.manifest_path.exists():
        raise FileExistsError(f"Run already initialized: {layout.root}")
    layout.create_base_directories()
    layout.create_round_directories(0)
    now = _utc_now()
    state = {
        "schema_version": 1,
        "task_name": task_name,
        "created_at": now,
        "updated_at": now,
        "active_round": 0,
        "active_stage": "collection",
        "rounds": {"000": _new_round_state(0)},
    }
    manifest = {
        "schema_version": 1,
        "task_name": task_name,
        "created_at": now,
        "inputs": [],
        "artifacts": [],
    }
    write_json_atomic(layout.manifest_path, manifest)
    write_json_atomic(layout.state_path, state)
    return state


def rollback_run_for_retrain(
    layout: RunLayout,
    *,
    round_index: int,
    stage: str,
) -> None:
    """Roll a run back in place to the strict prefix before ``round_index/stage``."""

    _validate_round_index(round_index)
    _validate_training_stage(stage)
    _validate_retrain_target(layout, round_index=round_index)
    _rollback_retrain_run(layout, round_index=round_index, stage=stage)


def start_stage(
    layout: RunLayout,
    round_index: int,
    stage: str,
    *,
    inputs: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Start the only currently legal stage and allocate its attempt directory."""

    state = load_json(layout.state_path)
    stage_state = _require_active_stage(state, round_index, stage)
    if stage_state["status"] not in {"pending", "failed"}:
        raise RuntimeError(f"Cannot start {stage}: status is {stage_state['status']}")
    attempt = len(stage_state["attempts"]) + 1
    now = _utc_now()
    attempt_record: dict[str, Any] = {
        "attempt": attempt,
        "status": "running",
        "started_at": now,
        "process_id": os.getpid(),
        "hostname": socket.gethostname(),
    }
    if metadata:
        attempt_record["metadata"] = dict(metadata)
    stage_state["status"] = "running"
    if inputs is not None:
        stage_state["inputs"] = dict(inputs)
    stage_state["attempts"].append(attempt_record)
    stage_state["started_at"] = now
    state["updated_at"] = now
    if stage != "collection":
        layout.attempt_dir(round_index, stage, attempt).mkdir(parents=True, exist_ok=False)
    write_json_atomic(layout.state_path, state)
    return attempt_record


def recover_interrupted_stage(layout: RunLayout) -> bool:
    """Mark a stale running attempt failed so a launcher can restart it."""
    state = load_json(layout.state_path)
    stage = state.get("active_stage")
    if stage not in ROUND_STAGES:
        return False
    round_index = _state_round_index(state)
    stage_state = state["rounds"][f"{round_index:03d}"]["stages"][stage]
    if stage_state.get("status") != "running":
        return False
    attempt = stage_state["attempts"][-1]
    process_id = attempt.get("process_id")
    hostname = attempt.get("hostname")
    if (
        hostname == socket.gethostname()
        and isinstance(process_id, int)
        and _process_is_alive(process_id)
    ):
        raise RuntimeError(
            f"Round {round_index:03d} stage {stage!r} is already running in "
            f"process {process_id}."
        )
    now = _utc_now()
    error = "Interrupted before stage state was committed."
    attempt.update({"status": "failed", "failed_at": now, "error": error})
    stage_state.update({"status": "failed", "failed_at": now, "error": error})
    state["updated_at"] = now
    write_json_atomic(layout.state_path, state)
    return True


def complete_stage(
    layout: RunLayout,
    round_index: int,
    stage: str,
    *,
    outputs: Mapping[str, Any] | None = None,
    parent_checkpoint: str | None = None,
    sampling_weights: Mapping[str, Any] | None = None,
    effective_losses: Mapping[str, Any] | None = None,
    cache_keys: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Publish stage metadata and advance the state machine."""

    state = load_json(layout.state_path)
    stage_state = _require_active_stage(state, round_index, stage)
    if stage_state["status"] != "running":
        raise RuntimeError(f"Cannot complete {stage}: status is {stage_state['status']}")
    now = _utc_now()
    attempt = stage_state["attempts"][-1]
    attempt.update({"status": "completed", "completed_at": now})
    stage_state.update(
        {
            "status": "completed",
            "completed_at": now,
            "outputs": dict(outputs or {}),
            "parent_checkpoint": parent_checkpoint,
            "sampling_weights": dict(sampling_weights or {}),
            "effective_losses": dict(effective_losses or {}),
            "cache_keys": dict(cache_keys or {}),
        }
    )
    stage_state.pop("failed_at", None)
    stage_state.pop("error", None)
    next_stage = _next_stage(stage)
    state["active_stage"] = next_stage
    state["updated_at"] = now
    write_json_atomic(layout.state_path, state)
    return stage_state


def fail_stage(
    layout: RunLayout,
    round_index: int,
    stage: str,
    *,
    error: str,
) -> dict[str, Any]:
    """Record a failed attempt while keeping the same stage retryable."""

    state = load_json(layout.state_path)
    stage_state = _require_active_stage(state, round_index, stage)
    if stage_state["status"] != "running":
        raise RuntimeError(f"Cannot fail {stage}: status is {stage_state['status']}")
    now = _utc_now()
    attempt = stage_state["attempts"][-1]
    attempt.update({"status": "failed", "failed_at": now, "error": error})
    stage_state.update({"status": "failed", "failed_at": now, "error": error})
    state["updated_at"] = now
    write_json_atomic(layout.state_path, state)
    return stage_state


def open_next_round(layout: RunLayout) -> dict[str, Any]:
    """Create the next round after the current round's policy is complete."""

    state = load_json(layout.state_path)
    current_index = _state_round_index(state)
    current = state["rounds"][f"{current_index:03d}"]
    if current["stages"]["policy"]["status"] != "completed":
        raise RuntimeError("Cannot open a new round before policy completion")
    if state["active_stage"] is not None:
        raise RuntimeError("Completed round has an unexpected active stage")
    next_index = current_index + 1
    layout.create_round_directories(next_index)
    state["rounds"][f"{next_index:03d}"] = _new_round_state(next_index)
    state["active_round"] = next_index
    state["active_stage"] = "collection"
    state["updated_at"] = _utc_now()
    write_json_atomic(layout.state_path, state)
    return state["rounds"][f"{next_index:03d}"]


def snapshot_inputs(
    layout: RunLayout,
    snapshots: Iterable[InputSnapshot],
    *,
    external_inputs: Iterable[InputReference] = (),
    prefer_reflink: bool = True,
) -> list[dict[str, Any]]:
    """Prepare copied and external inputs and record verified hashes in the manifest."""

    manifest = load_json(layout.manifest_path)
    if manifest.get("inputs"):
        raise RuntimeError("Run inputs have already been snapshotted")
    resolved_snapshots: list[tuple[InputSnapshot, Path, list[Path]]] = []
    resolved_external: list[tuple[InputReference, Path, list[Path]]] = []
    records: list[dict[str, Any]] = []
    destinations: set[str] = set()
    names: set[str] = set()
    created_destinations: list[Path] = []
    total_bytes = 0

    for snapshot in snapshots:
        if snapshot.name in names:
            raise ValueError(f"Duplicate input name: {snapshot.name}")
        names.add(snapshot.name)
        source = snapshot.source.resolve(strict=True)
        source_files = _input_files(source)
        destination = layout.inputs_dir / snapshot.destination
        destination_relative = destination.relative_to(layout.root).as_posix()
        if destination_relative in destinations or destination.exists():
            raise FileExistsError(f"Duplicate snapshot destination: {destination_relative}")
        destinations.add(destination_relative)
        resolved_snapshots.append((snapshot, source, source_files))
        total_bytes += 3 * sum(path.stat().st_size for path in source_files)

    for reference in external_inputs:
        if reference.name in names:
            raise ValueError(f"Duplicate input name: {reference.name}")
        names.add(reference.name)
        source = reference.source.resolve(strict=True)
        source_files = _input_files(source)
        resolved_external.append((reference, source, source_files))
        total_bytes += sum(path.stat().st_size for path in source_files)

    progress_bar = tqdm(
        total=total_bytes,
        desc="Preparing run inputs",
        unit="B",
        unit_scale=True,
        dynamic_ncols=True,
    )
    try:
        for snapshot, source, source_files in resolved_snapshots:
            destination = layout.inputs_dir / snapshot.destination
            destination_relative = destination.relative_to(layout.root).as_posix()
            created_destinations.append(destination)
            files = _copy_and_hash(
                source,
                destination,
                source_files=source_files,
                prefer_reflink=prefer_reflink,
                progress=progress_bar.update,
            )
            records.append(
                {
                    "name": snapshot.name,
                    "source": str(source),
                    "snapshot_path": destination_relative,
                    "storage_mode": "snapshot",
                    "kind": "directory" if source.is_dir() else "file",
                    "sha256": _tree_digest(files),
                    "size_bytes": sum(int(item["size_bytes"]) for item in files),
                    "files": files,
                }
            )
        for reference, source, source_files in resolved_external:
            files = _hash_input(source, source_files, progress=progress_bar.update)
            records.append(
                {
                    "name": reference.name,
                    "source": str(source),
                    "snapshot_path": None,
                    "storage_mode": "external",
                    "kind": "directory" if source.is_dir() else "file",
                    "sha256": _tree_digest(files),
                    "size_bytes": sum(int(item["size_bytes"]) for item in files),
                    "files": files,
                }
            )
    except BaseException:
        for destination in reversed(created_destinations):
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink(missing_ok=True)
        raise
    finally:
        progress_bar.close()
    manifest["inputs"] = records
    manifest["updated_at"] = _utc_now()
    write_json_atomic(layout.manifest_path, manifest)
    return records


def record_artifact(
    layout: RunLayout,
    path: Path,
    *,
    round_index: int,
    stage: str,
) -> dict[str, Any]:
    """Record a published artifact, replacing the same stage/path on retry."""
    _validate_round_index(round_index)
    _validate_stage(stage)
    resolved = Path(path).resolve(strict=True)
    relative = resolved.relative_to(layout.root).as_posix()
    manifest = load_json(layout.manifest_path)
    artifacts = list(manifest.get("artifacts", []))
    record = {
        "path": relative,
        "round": int(round_index),
        "stage": stage,
        "size_bytes": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
        "published_at": _utc_now(),
    }
    matching_indices = [
        index
        for index, item in enumerate(artifacts)
        if str(item.get("path")) == relative
    ]
    if len(matching_indices) > 1:
        raise ValueError(f"Manifest contains duplicate artifact paths: {relative}")
    if matching_indices:
        previous = artifacts[matching_indices[0]]
        if int(previous.get("round", -1)) != round_index or previous.get("stage") != stage:
            raise RuntimeError(
                f"Artifact path {relative} is already owned by another stage."
            )
        artifacts[matching_indices[0]] = record
    else:
        artifacts.append(record)
    manifest["artifacts"] = artifacts
    manifest["updated_at"] = _utc_now()
    write_json_atomic(layout.manifest_path, manifest)
    return record


def _copy_and_hash(
    source: Path,
    destination: Path,
    *,
    source_files: list[Path] | None = None,
    prefer_reflink: bool,
    progress: Callable[[int], None] | None = None,
) -> list[dict[str, Any]]:
    source_files = _input_files(source) if source_files is None else source_files
    if source.is_dir():
        destination.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    for source_file in source_files:
        relative = Path(source_file.name) if source.is_file() else source_file.relative_to(source)
        destination_file = destination if source.is_file() else destination / relative
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        mode = _copy_file(
            source_file,
            destination_file,
            prefer_reflink=prefer_reflink,
            progress=progress,
        )
        source_hash = sha256_file(source_file, progress=progress)
        destination_hash = sha256_file(destination_file, progress=progress)
        if source_hash != destination_hash:
            raise OSError(f"SHA-256 mismatch after copying {source_file}")
        records.append(
            {
                "path": relative.as_posix(),
                "size_bytes": destination_file.stat().st_size,
                "sha256": destination_hash,
                "copy_mode": mode,
            }
        )
    return records


def _input_files(source: Path) -> list[Path]:
    if not source.is_file() and not source.is_dir():
        raise ValueError(f"Input must be a regular file or directory: {source}")
    if source.is_file():
        return [source]
    return sorted(path for path in source.rglob("*") if path.is_file())


def _hash_input(
    source: Path,
    source_files: list[Path],
    *,
    progress: Callable[[int], None] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source_file in source_files:
        relative = Path(source_file.name) if source.is_file() else source_file.relative_to(source)
        records.append(
            {
                "path": relative.as_posix(),
                "size_bytes": source_file.stat().st_size,
                "sha256": sha256_file(source_file, progress=progress),
            }
        )
    return records


def _copy_file(
    source: Path,
    destination: Path,
    *,
    prefer_reflink: bool,
    progress: Callable[[int], None] | None = None,
    chunk_size: int = 1024 * 1024,
) -> str:
    size_bytes = source.stat().st_size
    if prefer_reflink:
        result = subprocess.run(
            ["cp", "--reflink=always", "--preserve=mode,timestamps", str(source), str(destination)],
            capture_output=True,
            check=False,
            text=True,
        )
        if result.returncode == 0:
            if progress is not None:
                progress(size_bytes)
            return "reflink"
        destination.unlink(missing_ok=True)
    with source.open("rb") as source_handle, destination.open("wb") as destination_handle:
        for chunk in iter(lambda: source_handle.read(chunk_size), b""):
            destination_handle.write(chunk)
            if progress is not None:
                progress(len(chunk))
    shutil.copystat(source, destination)
    return "copy"


def _tree_digest(files: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in files:
        digest.update(str(record["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_retrain_target(
    layout: RunLayout,
    *,
    round_index: int,
) -> None:
    for path in (layout.config_path, layout.manifest_path, layout.state_path):
        if not path.is_file():
            raise FileNotFoundError(f"Run metadata does not exist: {path}")
    state = load_json(layout.state_path)
    load_json(layout.manifest_path)
    rounds = state.get("rounds")
    if not isinstance(rounds, dict) or f"{round_index:03d}" not in rounds:
        raise ValueError(f"Round {round_index:03d} does not exist in {layout.root}")

    for round_key, round_state in rounds.items():
        stages = round_state.get("stages") if isinstance(round_state, dict) else None
        if not isinstance(stages, dict):
            raise ValueError(f"Malformed stage state for round {round_key}")
        if round_state.get("index") != int(round_key):
            raise ValueError(f"Round {round_key} has a mismatched index")
        for stage_name in ROUND_STAGES:
            stage_state = stages.get(stage_name)
            if not isinstance(stage_state, dict):
                raise ValueError(f"Missing round {round_key} stage {stage_name}")
            if stage_state.get("status") == "running":
                raise RuntimeError(
                    f"Cannot roll back a run with a running stage: round {round_key} "
                    f"stage {stage_name}"
                )


def _rollback_retrain_run(
    layout: RunLayout,
    *,
    round_index: int,
    stage: str,
) -> None:
    stage_index = ROUND_STAGES.index(stage)
    state = load_json(layout.state_path)
    manifest = load_json(layout.manifest_path)
    rounds = state["rounds"]
    retained_artifacts = [
        artifact
        for artifact in manifest.get("artifacts", [])
        if int(artifact.get("round", -1)) < round_index
        or (
            int(artifact.get("round", -1)) == round_index
            and artifact.get("stage") in ROUND_STAGES[:stage_index]
        )
    ]
    for round_key in list(rounds):
        if int(round_key) > round_index:
            del rounds[round_key]
            shutil.rmtree(layout.rounds_dir / round_key, ignore_errors=True)
    for round_dir in layout.rounds_dir.iterdir():
        if (
            round_dir.is_dir()
            and round_dir.name.isdigit()
            and int(round_dir.name) > round_index
        ):
            shutil.rmtree(round_dir)

    target_round = rounds[f"{round_index:03d}"]
    pending = _new_round_state(round_index)["stages"]
    for stage_name in ROUND_STAGES[stage_index:]:
        target_round["stages"][stage_name] = pending[stage_name]
        if stage_name != "collection":
            shutil.rmtree(layout.stage_dir(round_index, stage_name), ignore_errors=True)
            layout.stage_dir(round_index, stage_name).mkdir(parents=True)

    eval_vis = layout.eval_vis_dir(round_index)
    if stage_index <= ROUND_STAGES.index("disc"):
        shutil.rmtree(eval_vis, ignore_errors=True)
    elif stage_index == ROUND_STAGES.index("vast"):
        shutil.rmtree(layout.vast_eval_vis_dir(round_index), ignore_errors=True)
        if eval_vis.is_dir() and not any(eval_vis.iterdir()):
            eval_vis.rmdir()

    cache_start = round_index if stage == "disc" else round_index + 1
    feature_cache = layout.cache_dir / "discriminator_features"
    for cache_path in feature_cache.glob("round_*.pt"):
        match = re.fullmatch(r"round_(\d+)\.pt", cache_path.name)
        if match is not None and int(match.group(1)) >= cache_start:
            cache_path.unlink()

    now = _utc_now()
    state.update(
        {
            "updated_at": now,
            "active_round": round_index,
            "active_stage": stage,
        }
    )
    write_json_atomic(layout.state_path, state)

    manifest["artifacts"] = retained_artifacts
    manifest["updated_at"] = now
    write_json_atomic(layout.manifest_path, manifest)


def _new_round_state(round_index: int) -> dict[str, Any]:
    return {
        "index": round_index,
        "stages": {
            stage: {
                "status": "pending",
                "attempts": [],
                "inputs": {},
                "outputs": {},
                "parent_checkpoint": None,
                "sampling_weights": {},
                "effective_losses": {},
                "cache_keys": {},
            }
            for stage in ROUND_STAGES
        },
    }


def _require_active_stage(
    state: dict[str, Any], round_index: int, stage: str
) -> dict[str, Any]:
    _validate_stage(stage)
    if _state_round_index(state) != round_index or state.get("active_stage") != stage:
        raise RuntimeError(
            f"Expected round {state.get('active_round')} stage {state.get('active_stage')}, "
            f"got round {round_index} stage {stage}"
        )
    round_key = f"{round_index:03d}"
    try:
        stage_state = state["rounds"][round_key]["stages"][stage]
    except (KeyError, TypeError) as exc:
        raise ValueError("Malformed run state") from exc
    if stage_state.get("status") not in _STAGE_STATUSES:
        raise ValueError("Malformed stage status")
    return stage_state


def _state_round_index(state: Mapping[str, Any]) -> int:
    round_index = state.get("active_round")
    if not isinstance(round_index, int) or isinstance(round_index, bool) or round_index < 0:
        raise ValueError("Malformed active_round")
    return round_index


def _next_stage(stage: str) -> str | None:
    index = ROUND_STAGES.index(stage)
    return ROUND_STAGES[index + 1] if index + 1 < len(ROUND_STAGES) else None


def _validate_stage(stage: str) -> None:
    if stage not in ROUND_STAGES:
        raise ValueError(f"Unknown stage: {stage}")


def _validate_training_stage(stage: str) -> None:
    if stage not in ROUND_STAGES[1:]:
        raise ValueError(f"Expected a training stage, got: {stage}")


def _validate_round_index(round_index: int) -> None:
    if not isinstance(round_index, int) or isinstance(round_index, bool) or round_index < 0:
        raise ValueError("round_index must be a non-negative integer")


def _process_is_alive(process_id: int) -> bool:
    if process_id <= 0:
        return False
    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
