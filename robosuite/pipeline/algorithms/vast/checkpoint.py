"""Checkpoint names and read-only compatibility for the VAST pipeline."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import torch

VAST_ALGORITHM = "vast_value_stitching_adaptation"
VAST_SCHEMA_VERSION = 8
SINGLE_VAST_SCHEMA_VERSION = 7
LEGACY_VAST_SCHEMA_VERSION = 6


def normalize_vast_payload(payload: Any, *, source: str | Path | None = None) -> dict[str, Any]:
    """Return a canonical VAST payload, accepting schema-v6 read-only aliases."""
    if not isinstance(payload, dict):
        raise ValueError("VAST checkpoint payload must be a dictionary.")
    schema = int(payload.get("schema_version", -1))
    normalized = dict(payload)
    if schema in {SINGLE_VAST_SCHEMA_VERSION, VAST_SCHEMA_VERSION}:
        if "vast_state" not in normalized:
            raise ValueError(f"Schema-v{schema} VAST checkpoint is missing 'vast_state'.")
        algorithm = normalized.get("algorithm")
        if algorithm is None:
            algorithm = normalized.get("encoder_meta", {}).get("algorithm")
        if algorithm != VAST_ALGORITHM:
            raise ValueError(
                f"Schema-v{schema} VAST checkpoint has invalid algorithm="
                f"{algorithm!r}; expected {VAST_ALGORITHM!r}."
            )
        mode = str(
            normalized.get("vast_state", {}).get(
                "vast_v_mode", normalized.get("cfg", {}).get("vast_v_mode", "")
            )
        )
        if mode == "ensemble_lcb":
            raise ValueError(
                "Legacy ensemble_lcb checkpoints are incompatible with "
                "indep_ensemble; re-run init_vast.sh."
            )
        if schema == SINGLE_VAST_SCHEMA_VERSION and mode not in {"", "single_vast"}:
            raise ValueError(
                "Schema-v7 ensemble checkpoints are unsupported; re-run "
                "init_vast.sh with vast_v_mode=indep_ensemble."
            )
        if schema == VAST_SCHEMA_VERSION and mode != "indep_ensemble":
            raise ValueError(
                "Schema-v8 VAST checkpoints require vast_v_mode='indep_ensemble', "
                f"got {mode!r}."
            )
        normalized["algorithm"] = VAST_ALGORITHM
        return normalized
    if schema == LEGACY_VAST_SCHEMA_VERSION:
        if "iql_state" not in normalized:
            raise ValueError("Deprecated schema-v6 checkpoint is missing 'iql_state'.")
        warnings.warn(
            "Loading deprecated schema-v6 VAST checkpoint key 'iql_state'"
            f"{f' from {source}' if source is not None else ''}; re-save as schema-v7.",
            FutureWarning,
            stacklevel=2,
        )
        normalized["vast_state"] = normalized["iql_state"]
        normalized["algorithm"] = VAST_ALGORITHM
        return normalized
    if schema == 5:
        raise ValueError(
            "Legacy schema-v5 checkpoints cannot initialize VAST; re-run init_vast.sh."
        )
    raise ValueError(
        f"Unsupported VAST checkpoint schema_version={schema}; expected "
        f"{VAST_SCHEMA_VERSION}, single-vast {SINGLE_VAST_SCHEMA_VERSION}, or "
        f"deprecated {LEGACY_VAST_SCHEMA_VERSION}."
    )


def load_vast_payload(path: str | Path) -> dict[str, Any]:
    """Load and normalize a VAST checkpoint without changing it on disk."""
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"VAST checkpoint does not exist: {resolved}")
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    return normalize_vast_payload(payload, source=resolved)


__all__ = [
    "LEGACY_VAST_SCHEMA_VERSION",
    "SINGLE_VAST_SCHEMA_VERSION",
    "VAST_ALGORITHM",
    "VAST_SCHEMA_VERSION",
    "load_vast_payload",
    "normalize_vast_payload",
]
