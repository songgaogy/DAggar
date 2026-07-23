from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import torch

from .agent import AWRAgent
from .trainer import AWRTrainer


QV_FORMAT = "robosuite-awr-qv"
QV_VERSION = 1


def build_qv_metadata(
    values: Mapping[str, Any] | None = None, **extra: Any
) -> dict[str, Any]:
    metadata = {} if values is None else dict(values)
    metadata.update(extra)
    return copy.deepcopy(metadata)


def cache_metadata_matches(
    actual: Mapping[str, Any], expected: Mapping[str, Any]
) -> bool:
    return dict(actual) == dict(expected)


def save_qv_cache(
    path: str | Path,
    *,
    agent: AWRAgent,
    trainer: AWRTrainer,
    metadata: Mapping[str, Any],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    torch.save(
        {
            "format": QV_FORMAT,
            "version": QV_VERSION,
            "metadata": copy.deepcopy(dict(metadata)),
            "qv_core": agent.core.qv_state_dict(),
            "trainer_state": trainer.state_dict(),
        },
        temporary,
    )
    temporary.replace(target)


def load_qv_cache(
    path: str | Path,
    *,
    agent: AWRAgent,
    trainer: AWRTrainer | None = None,
    expected_metadata: Mapping[str, Any] | None = None,
    load_optimizers: bool = True,
) -> dict[str, Any]:
    payload = torch.load(
        Path(path), map_location=agent.awr_config.device, weights_only=False
    )
    if (
        payload.get("format") != QV_FORMAT
        or int(payload.get("version", -1)) != QV_VERSION
    ):
        raise ValueError("Unsupported AWR Q/V cache format.")
    metadata = dict(payload.get("metadata", {}))
    if expected_metadata is not None and not cache_metadata_matches(
        metadata, expected_metadata
    ):
        raise ValueError("AWR Q/V cache metadata does not match.")
    agent.core.load_qv_state_dict(
        payload["qv_core"], load_optimizers=load_optimizers
    )
    if trainer is not None:
        trainer.load_state_dict(payload.get("trainer_state"))
    return metadata


__all__ = [
    "QV_FORMAT",
    "QV_VERSION",
    "build_qv_metadata",
    "cache_metadata_matches",
    "load_qv_cache",
    "save_qv_cache",
]
