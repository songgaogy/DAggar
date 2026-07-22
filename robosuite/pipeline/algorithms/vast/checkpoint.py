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


def _validate_vast_weights_state(learner: Any, state: dict[str, Any]) -> None:
    """Apply the learner's strict checkpoint contract without optimizer checks."""
    if "q_ensemble" in state or "q1" in state:
        raise ValueError(
            "load_vast_weights: checkpoint contains an unsupported Q head "
            "(q_ensemble/q1). Re-run the VAST warmup."
        )
    schema = int(state.get("learner_schema_version", -1))
    checkpoint_mode = str(state.get("vast_v_mode", ""))
    if checkpoint_mode == "ensemble_lcb":
        raise ValueError(
            "load_vast_weights: legacy ensemble_lcb checkpoints are incompatible "
            "with indep_ensemble; re-run VAST warmup."
        )
    if schema not in {6, 7, 8}:
        raise ValueError(
            f"load_vast_weights: unsupported learner schema {schema}; expected schema 6, 7, or 8."
        )
    for key in ("state_feature_dim", "v", "target_v", "g"):
        if key not in state:
            raise ValueError(f"load_vast_weights: checkpoint is missing required key {key!r}.")
    if schema in {7, 8}:
        algorithm = str(state.get("algorithm", ""))
        if algorithm != VAST_ALGORITHM:
            raise ValueError(
                f"load_vast_weights: schema-{schema} checkpoint has algorithm={algorithm!r}; "
                f"expected {VAST_ALGORITHM!r}."
            )
    elif str(state.get("method", "")) != "vast_value_stitching":
        raise ValueError("load_vast_weights: schema-6 checkpoint is not a VAST value-stitching state.")
    if schema in {6, 7} and checkpoint_mode not in {"", "single_vast"}:
        raise ValueError(
            "load_vast_weights: pre-schema-8 ensemble checkpoints are unsupported."
        )
    if schema == 8:
        if checkpoint_mode != "indep_ensemble":
            raise ValueError(
                "load_vast_weights: schema-8 requires vast_v_mode='indep_ensemble', "
                f"got {checkpoint_mode!r}."
            )
        if state.get("ensemble_method") != "independent_v_mean":
            raise ValueError(
                "load_vast_weights: schema-8 checkpoint has invalid "
                f"ensemble_method={state.get('ensemble_method')!r}."
            )

    for key, runtime_val in (
        ("state_proj_dim", learner.state_proj_dim),
        ("proprio_proj_dim", learner.proprio_proj_dim),
        ("n_tokens", learner.n_tokens),
        ("proprio_dim", learner.proprio_dim),
        ("v_ensemble_size", learner.ensemble_size),
        ("state_feature_dim", learner.state_feature_dim),
    ):
        if int(state.get(key, -1)) != int(runtime_val):
            raise ValueError(
                f"load_vast_weights: {key} mismatch "
                f"(ckpt={state.get(key)!r}, runtime={runtime_val!r})."
            )
    for key, runtime_val in (
        ("vast_v_mode", learner.vast_v_mode),
        ("vast_max_k", int(learner.cfg.vast_max_k)),
        ("vast_sampling_seed", int(learner.cfg.vast_sampling_seed)),
        ("action_horizon", int(learner.cfg.action_horizon)),
    ):
        checkpoint_val = state.get(key)
        if checkpoint_val != runtime_val:
            raise ValueError(
                f"load_vast_weights: {key} mismatch "
                f"(ckpt={checkpoint_val!r}, runtime={runtime_val!r})."
            )
    saved_cfg = state.get("cfg", {})
    for key in (
        "discount",
        "expectile_tau",
        "vast_comp_coef",
        "output_reward_coef",
        "disc_reward_coef",
    ):
        runtime_val = float(getattr(learner.cfg, key))
        if float(saved_cfg.get(key, float("nan"))) != runtime_val:
            raise ValueError(
                f"load_vast_weights: config {key} mismatch "
                f"(ckpt={saved_cfg.get(key)!r}, runtime={runtime_val!r})."
            )


def load_vast_weights(
    learner: Any,
    checkpoint: str | Path,
    *,
    expected_state_feature_dim: int,
    expected_chunk_feature_dim: int,
    expected_action_dim: int,
) -> dict[str, Any]:
    """Load V/target-V/G weights while intentionally resetting optimizers."""
    payload = load_vast_payload(checkpoint)
    metadata = payload.get("encoder_meta", {})
    for key, expected in (
        ("state_feature_dim", expected_state_feature_dim),
        ("chunk_feature_dim", expected_chunk_feature_dim),
        ("policy_action_dim", expected_action_dim),
    ):
        if int(metadata.get(key, -1)) != int(expected):
            raise ValueError(
                f"VAST checkpoint {key}={metadata.get(key)!r} != expected {int(expected)}."
            )

    state = payload["vast_state"]
    _validate_vast_weights_state(learner, state)
    learner.v.load_state_dict(state["v"])
    learner.target_v.load_state_dict(state["target_v"])
    learner.g.load_state_dict(state["g"])
    return payload


__all__ = [
    "LEGACY_VAST_SCHEMA_VERSION",
    "SINGLE_VAST_SCHEMA_VERSION",
    "VAST_ALGORITHM",
    "VAST_SCHEMA_VERSION",
    "load_vast_payload",
    "load_vast_weights",
    "normalize_vast_payload",
]
