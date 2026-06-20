"""DIPOLE / IQL algorithm exports (lazy to avoid eager dipole → benchmark imports)."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "DipoleAgent",
    "DipoleBatch",
    "DipoleConfig",
    "DipoleTrainer",
    "FlowAugmentationConfig",
    "NNPUGProvider",
    "ReplayBufferConfig",
    "TrainerConfig",
    "Transition",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "DipoleAgent": (".dipole", "DipoleAgent"),
    "DipoleBatch": (".dipole", "DipoleBatch"),
    "DipoleConfig": (".dipole", "DipoleConfig"),
    "DipoleTrainer": (".dipole", "DipoleTrainer"),
    "FlowAugmentationConfig": (".dipole", "FlowAugmentationConfig"),
    "NNPUGProvider": (".dipole", "NNPUGProvider"),
    "ReplayBufferConfig": (".dipole", "ReplayBufferConfig"),
    "TrainerConfig": (".dipole", "TrainerConfig"),
    "Transition": (".dipole", "Transition"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_name, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_name, __name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
