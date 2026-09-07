from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch


class LegacyTransition:
    """Compatibility target for pickled pipeline Transition objects."""

    def __init__(self, **values: Any) -> None:
        self.__dict__.update(values)


class _LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module == "robosuite.pipeline.common.types" and name == "Transition":
            return LegacyTransition
        return super().find_class(module, name)


_PICKLE_MODULE = SimpleNamespace(
    __name__="dsrl_legacy_pickle",
    Unpickler=_LegacyUnpickler,
    Pickler=pickle.Pickler,
    load=pickle.load,
    loads=pickle.loads,
    dump=pickle.dump,
    dumps=pickle.dumps,
)


@dataclass(frozen=True)
class LegacyBuffer:
    path: Path
    metadata: Mapping[str, Any]
    payload: Mapping[str, Any]
    storage: Sequence[Any]


def _read_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Legacy metadata does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Legacy metadata must be a JSON object: {path}")
    return value


def _validate_metadata(
    metadata: Mapping[str, Any],
    payload: Mapping[str, Any],
    storage: Sequence[Any],
    *,
    expected_task: str | None,
    expected_horizon: int | None,
    expected_cameras: Sequence[str] | None,
) -> None:
    declared_count = metadata.get("n_transitions")
    if declared_count is not None and int(declared_count) != len(storage):
        raise ValueError(
            f"Legacy transition count mismatch: metadata={declared_count}, payload={len(storage)}."
        )
    if expected_task is not None and metadata.get("task") != expected_task:
        raise ValueError(
            f"Legacy task mismatch: expected '{expected_task}', got '{metadata.get('task')}'."
        )
    payload_horizon = payload.get("action_horizon", metadata.get("action_horizon"))
    if expected_horizon is not None and int(payload_horizon) != int(expected_horizon):
        raise ValueError(
            f"Legacy action horizon mismatch: expected {expected_horizon}, got {payload_horizon}."
        )
    payload_cameras = tuple(payload.get("camera_names", metadata.get("camera_names", ())))
    if expected_cameras is not None and payload_cameras != tuple(expected_cameras):
        raise ValueError(
            f"Legacy cameras mismatch: expected {tuple(expected_cameras)}, got {payload_cameras}."
        )


def load_legacy_buffer(
    path: str | Path,
    *,
    metadata_path: str | Path | None = None,
    expected_task: str | None = None,
    expected_horizon: int | None = None,
    expected_cameras: Sequence[str] | None = None,
) -> LegacyBuffer:
    """Load the legacy pickle once while preserving its storage list by reference."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Legacy replay does not exist: {source}")
    meta_source = (
        Path(metadata_path).expanduser().resolve()
        if metadata_path is not None
        else source.with_suffix(".meta.json")
    )
    metadata = _read_metadata(meta_source)
    payload = torch.load(
        source,
        map_location="cpu",
        weights_only=False,
        pickle_module=_PICKLE_MODULE,
    )
    if not isinstance(payload, Mapping):
        raise TypeError(f"Legacy replay payload must be a mapping, got {type(payload)!r}.")
    storage = payload.get("storage")
    if not isinstance(storage, Sequence):
        raise TypeError("Legacy replay payload is missing sequence field 'storage'.")
    _validate_metadata(
        metadata,
        payload,
        storage,
        expected_task=expected_task,
        expected_horizon=expected_horizon,
        expected_cameras=expected_cameras,
    )
    return LegacyBuffer(source, metadata, payload, storage)

