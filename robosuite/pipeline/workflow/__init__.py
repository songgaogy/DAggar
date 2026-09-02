"""Run contract for the batch-online DIPOLE pipeline."""

from .state import (
    ROUND_STAGES,
    InputReference,
    InputSnapshot,
    RunLayout,
    complete_stage,
    create_run,
    fail_stage,
    load_json,
    open_next_round,
    record_artifact,
    recover_interrupted_stage,
    rollback_run_for_retrain,
    sha256_file,
    snapshot_inputs,
    start_stage,
    write_json_atomic,
)

__all__ = [
    "ROUND_STAGES",
    "InputReference",
    "InputSnapshot",
    "RunLayout",
    "complete_stage",
    "create_run",
    "fail_stage",
    "load_json",
    "open_next_round",
    "record_artifact",
    "recover_interrupted_stage",
    "rollback_run_for_retrain",
    "sha256_file",
    "snapshot_inputs",
    "start_stage",
    "write_json_atomic",
]
