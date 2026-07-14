"""CPU-free unit tests for offline VAST sampling and checkpoint validation."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from robosuite.pipeline.algorithms.vast.checkpoint import load_vast_payload
from robosuite.pipeline.algorithms.vast.common import VASTConfig
from robosuite.pipeline.algorithms.vast.utils.vis_vast import load_vast_payload as load_vis_payload
from robosuite.pipeline.offline.utils.advantage import sample_vast_macro_horizons
from robosuite.pipeline.offline.utils.vast_finetune import validate_vast_checkpoint_payload


def _cfg() -> VASTConfig:
    return VASTConfig(
        vast_v_mode="single_vast",
        action_horizon=2,
        vast_max_k=4,
        vast_sampling_seed=17,
        disc_reward_coef=0.0,
    )


def test_vast_macro_sampling_is_reproducible_and_falls_back_at_tail() -> None:
    starts = [0, 2, 4, 6, 20, 22]
    episodes = [0, 0, 0, 0, 1, 1]
    kwargs = dict(
        valid_starts=starts,
        episode_indices=episodes,
        horizon=2,
        max_k=4,
        seed=123,
        terminal_starts={4},
    )
    first = sample_vast_macro_horizons(**kwargs)
    second = sample_vast_macro_horizons(**kwargs)
    assert first == second

    sampled_k, future_indices, fallback = first
    assert sampled_k[0] in {2, 3}
    assert sampled_k[1] == 2
    assert fallback[2:] == [True, True, False, True]
    assert future_indices[2] == starts[2] + 2
    assert all(k <= 4 for k in sampled_k)


def test_vast_checkpoint_validation_accepts_matching_finetuned_schema_v7() -> None:
    cfg = _cfg()
    payload = {
        "schema_version": 7,
        "vast_state": {},
        "algorithm": "vast_value_stitching_adaptation",
        "cfg": asdict(cfg),
        "encoder_meta": {"algorithm": "vast_value_stitching_adaptation", "finetuned_offline": True},
    }
    validate_vast_checkpoint_payload(payload, cfg, require_finetuned=True)


def test_vast_checkpoint_validation_reads_deprecated_schema_v6() -> None:
    cfg = _cfg()
    payload = {
        "schema_version": 6,
        "iql_state": {"legacy": True},
        "cfg": asdict(cfg),
        "encoder_meta": {"method": "vast_value_stitching", "finetuned_offline": True},
    }
    with pytest.warns(FutureWarning, match="iql_state"):
        validate_vast_checkpoint_payload(payload, cfg, require_finetuned=True)
    assert payload["vast_state"] is payload["iql_state"]


def test_schema6_file_load_chain_is_normalized(tmp_path: Path) -> None:
    cfg = _cfg()
    checkpoint = tmp_path / "iql_state.pt"
    torch.save(
        {
            "schema_version": 6,
            "iql_state": {"learner_schema_version": 6},
            "cfg": asdict(cfg),
            "encoder_meta": {"method": "vast_value_stitching"},
        },
        checkpoint,
    )
    with pytest.warns(FutureWarning, match="iql_state"):
        payload = load_vast_payload(checkpoint)
    assert payload["vast_state"] is payload["iql_state"]
    with pytest.warns(FutureWarning, match="iql_state"):
        vis_payload = load_vis_payload(checkpoint)
    assert vis_payload["vast_state"] is vis_payload["iql_state"]


def test_vast_checkpoint_validation_rejects_legacy_and_config_mismatch() -> None:
    cfg = _cfg()
    legacy = {"schema_version": 5, "cfg": asdict(cfg), "encoder_meta": {}}
    with pytest.raises(ValueError, match="schema-v5"):
        validate_vast_checkpoint_payload(legacy, cfg, require_finetuned=False)

    mismatch = {
        "schema_version": 7,
        "vast_state": {},
        "algorithm": "vast_value_stitching_adaptation",
        "cfg": {**asdict(cfg), "vast_max_k": cfg.vast_max_k + 1},
        "encoder_meta": {"algorithm": "vast_value_stitching_adaptation", "finetuned_offline": True},
    }
    with pytest.raises(ValueError, match="vast_max_k"):
        validate_vast_checkpoint_payload(mismatch, cfg, require_finetuned=True)
