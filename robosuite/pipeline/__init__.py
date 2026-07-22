"""Robosuite DIPOLE / VAST pipeline (lazy public API)."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "DipoleAgent",
    "DipoleBatch",
    "DipoleConfig",
    "FlowAugmentationConfig",
    "ReplayBufferConfig",
    "TrainerConfig",
    "Transition",
    "RobosuiteInterventionRuntime",
    "RobosuiteObservationAdapter",
    "RobosuiteRuntimeConfig",
    "build_algorithm",
    "build_robosuite_env",
    "load_demo_paths",
    "load_hdf5_demos_into_transitions",
    "load_transition_shard",
    "save_transition_shard",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "DipoleAgent": (".algorithms", "DipoleAgent"),
    "DipoleBatch": (".algorithms", "DipoleBatch"),
    "DipoleConfig": (".algorithms", "DipoleConfig"),
    "FlowAugmentationConfig": (".algorithms", "FlowAugmentationConfig"),
    "ReplayBufferConfig": (".algorithms", "ReplayBufferConfig"),
    "TrainerConfig": (".algorithms", "TrainerConfig"),
    "Transition": (".algorithms", "Transition"),
    "RobosuiteInterventionRuntime": (".common.environment", "RobosuiteInterventionRuntime"),
    "RobosuiteObservationAdapter": (".common.environment", "RobosuiteObservationAdapter"),
    "RobosuiteRuntimeConfig": (".common.environment", "RobosuiteRuntimeConfig"),
    "build_algorithm": (".factory", "build_algorithm"),
    "build_robosuite_env": (".common.environment", "build_robosuite_env"),
    "load_demo_paths": (".utils", "load_demo_paths"),
    "load_hdf5_demos_into_transitions": (
        ".common.environment",
        "load_hdf5_demos_into_transitions",
    ),
    "load_transition_shard": (".utils", "load_transition_shard"),
    "save_transition_shard": (".utils", "save_transition_shard"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_name, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_name, __name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
