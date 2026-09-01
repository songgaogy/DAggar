"""CPU-free unit tests for offline VAST sampling and checkpoint validation."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from robosuite.pipeline.algorithms.flow_dagger.common import ReplayBufferConfig
from robosuite.pipeline.algorithms.flow_dagger.replay_buffer import FlowDaggerReplayBuffer
from robosuite.pipeline.algorithms.vast.checkpoint import load_vast_payload
from robosuite.pipeline.algorithms.vast.common import VASTConfig
from robosuite.pipeline.algorithms.vast.replay import VASTReplayBuffer
from robosuite.pipeline.common.types import Transition
from robosuite.pipeline.modules.visualization.vast.renderer import load_vast_payload as load_vis_payload
from robosuite.pipeline.modules.training.dipole import runner as train_offline
from robosuite.pipeline.modules.training.dipole.advantage import sample_vast_macro_horizons
from robosuite.pipeline.modules.training.dipole.vast_finetune import (
    build_vast_finetune_buffer,
    validate_vast_checkpoint_payload,
)
from robosuite.pipeline.config.adapters import offline_stage_config
from robosuite.pipeline.workflow.runner import load_default_config


def _cfg() -> VASTConfig:
    return VASTConfig(
        vast_v_mode="single_vast",
        action_horizon=2,
        vast_max_k=4,
        vast_sampling_seed=17,
        disc_reward_coef=0.0,
    )


def test_offline_config_uses_confirmed_td1_baseline() -> None:
    cfg = offline_stage_config(
        load_default_config(task="PickPlaceCereal"),
        stage="all",
        episodes_paths=["round-000.pt"],
    )
    estimator, gae_lambda = train_offline._resolve_advantage_config(cfg)
    assert estimator == "td1"
    assert gae_lambda == pytest.approx(0.6)
    assert cfg.offline.vast_finetune.relabel_disc_reward is True
    assert cfg.offline.branch_weight.eta == pytest.approx(0.5)
    assert train_offline.build_branch_weight_policy(cfg.offline.branch_weight).eta == pytest.approx(0.5)
    assert OmegaConf.select(cfg, "offline.use_online_success") is None


def test_phase_a_explicitly_relabels_discriminator_reward_after_strict_load() -> None:
    cfg = _cfg()
    cfg.disc_reward_coef = 0.2
    provenance = train_offline._resolve_vast_reward_semantics(
        cfg,
        {**asdict(cfg), "disc_reward_coef": 0.0},
        skip_rl=False,
        relabel_disc_reward=True,
    )

    assert cfg.disc_reward_coef == pytest.approx(0.0)
    assert provenance == {
        "enabled": True,
        "changed": True,
        "checkpoint_disc_reward_coef": 0.0,
        "requested_disc_reward_coef": 0.2,
        "effective_disc_reward_coef": 0.2,
    }
    train_offline._apply_vast_disc_reward_relabel(cfg, provenance)
    assert cfg.disc_reward_coef == pytest.approx(0.2)


def test_vast_finetune_freezes_only_loaded_warmup_copy(tmp_path: Path) -> None:
    source = FlowDaggerReplayBuffer(
        config=ReplayBufferConfig(capacity=16, batch_size=1),
        name="warmup-source",
        camera_names=["agentview"],
        action_horizon=2,
        image_size=4,
    )
    for step in range(4):
        obs = {
            "state": np.full(3, step, dtype=np.float32),
            "agentview": np.full((4, 4, 3), step, dtype=np.uint8),
        }
        next_obs = {
            "state": np.full(3, step + 1, dtype=np.float32),
            "agentview": np.full((4, 4, 3), step + 1, dtype=np.uint8),
        }
        source.add(
            Transition(
                obs=obs,
                action=np.full(2, step, dtype=np.float32),
                reward=0.0 if step >= 1 else -1.0,
                next_obs=next_obs,
                done=step == 3,
                info={
                    "episode_index": 0,
                    "episode_step": step,
                    "success": step >= 1,
                    "nnpu_disc_intrinsic": -0.25,
                },
            )
        )
    warmup_path = tmp_path / "vast_offline_transitions.pt"
    source.save(warmup_path)

    finetune, stats = build_vast_finetune_buffer(
        [],
        camera_names=["agentview"],
        image_size=4,
        action_horizon=2,
        warmup_transitions_path=warmup_path,
        relabel_disc_reward=True,
        freeze_warmup_post_success=True,
        capacity=16,
    )

    assert stats["frozen_warmup_post_success_transitions"] == 2
    assert stats["discarded_cached_disc_rewards"] == 4
    assert np.all(np.asarray(finetune._storage[2].obs["state"]) == 1)  # noqa: SLF001
    assert np.all(np.asarray(finetune._storage[3].obs["state"]) == 1)  # noqa: SLF001
    assert "nnpu_disc_intrinsic" not in (finetune._storage[2].info or {})  # noqa: SLF001

    reloaded = FlowDaggerReplayBuffer(
        config=ReplayBufferConfig(capacity=16, batch_size=1),
        name="warmup-reloaded",
        camera_names=["agentview"],
        action_horizon=2,
        image_size=4,
    )
    reloaded.load(warmup_path)
    assert np.all(np.asarray(source._storage[2].obs["state"]) == 2)  # noqa: SLF001
    assert np.all(np.asarray(reloaded._storage[2].obs["state"]) == 2)  # noqa: SLF001
    assert np.all(np.asarray(reloaded._storage[3].obs["state"]) == 3)  # noqa: SLF001

    unfrozen, unfrozen_stats = build_vast_finetune_buffer(
        [],
        camera_names=["agentview"],
        image_size=4,
        action_horizon=2,
        warmup_transitions_path=warmup_path,
        freeze_warmup_post_success=False,
        capacity=16,
    )
    assert unfrozen_stats["frozen_warmup_post_success_transitions"] == 0
    assert np.all(np.asarray(unfrozen._storage[2].obs["state"]) == 2)  # noqa: SLF001
    assert np.all(np.asarray(unfrozen._storage[3].obs["state"]) == 3)  # noqa: SLF001


@pytest.mark.parametrize("length", [4, 11])
def test_vast_policy_success_tail_anchors_every_chunk_phase(length: int) -> None:
    horizon = 4
    policy: list[Transition] = []
    for step in range(length):
        policy.append(
            Transition(
                obs={
                    "state": np.asarray([step], dtype=np.float32),
                    "agentview": np.full((4, 4, 3), step, dtype=np.uint8),
                },
                action=np.asarray([step], dtype=np.float32),
                reward=0.0 if step == length - 1 else -1.0,
                next_obs={
                    "state": np.asarray([step + 1], dtype=np.float32),
                    "agentview": np.full((4, 4, 3), step + 1, dtype=np.uint8),
                },
                done=step == length - 1,
                info={
                    "episode_index": 0,
                    "episode_step": step,
                    "success": step == length - 1,
                },
            )
        )
    source_final_info = dict(policy[-1].info or {})

    buffer, stats = build_vast_finetune_buffer(
        policy,
        camera_names=["agentview"],
        image_size=4,
        action_horizon=horizon,
        warmup_transitions_path=None,
        capacity=64,
    )

    assert len(policy) == length
    assert policy[-1].done is True
    assert policy[-1].info == source_final_info
    assert not any(
        bool((transition.info or {}).get("synthetic_vast_success_tail", False))
        for transition in policy
    )
    assert len(buffer) == length + horizon - 1
    assert stats["policy_source_transitions"] == length
    assert stats["policy_vast_transitions"] == length + horizon - 1
    assert stats["synthetic_success_tail_transitions"] == horizon - 1
    assert stats["padded_success_sections"] == 1
    assert stats["absorbing_reward_mask_transitions"] == horizon - 1
    assert [index for index, item in enumerate(buffer._storage) if item.done] == [  # noqa: SLF001
        length + horizon - 2
    ]
    valid_starts = buffer._get_valid_start_indices_locked()  # noqa: SLF001
    assert valid_starts == list(range(length))

    cfg = VASTConfig(action_horizon=horizon, vast_max_k=1, disc_reward_coef=0.0)
    replay = VASTReplayBuffer(base_buffer=buffer, cfg=cfg)
    terminal_starts = valid_starts[-horizon:]
    assert {start % horizon for start in terminal_starts} == set(range(horizon))
    assert all(replay._raw_chunk_done_locked(start) for start in terminal_starts)  # noqa: SLF001
    assert not any(
        replay._raw_chunk_done_locked(start)  # noqa: SLF001
        for start in valid_starts[:-horizon]
    )


@pytest.mark.parametrize(
    "end_reason",
    ["manual_reset", "env_done", "max_steps", "worker_exception"],
)
def test_vast_policy_non_success_ending_adds_absorbing_tail(
    end_reason: str,
) -> None:
    horizon = 4
    policy = [
        Transition(
            obs={
                "state": np.asarray([step], dtype=np.float32),
                "agentview": np.full((4, 4, 3), step, dtype=np.uint8),
            },
            action=np.asarray([step], dtype=np.float32),
            reward=-1.0,
            next_obs={
                "state": np.asarray([step + 1], dtype=np.float32),
                "agentview": np.full((4, 4, 3), step + 1, dtype=np.uint8),
            },
            done=step == 5,
            info={
                "episode_index": 0,
                "episode_step": step,
                "success": False,
                "policy_section_end_reason": end_reason,
            },
        )
        for step in range(6)
    ]

    buffer, stats = build_vast_finetune_buffer(
        policy,
        camera_names=["agentview"],
        image_size=4,
        action_horizon=horizon,
        warmup_transitions_path=None,
        capacity=32,
    )

    assert len(buffer) == len(policy) + horizon
    assert stats["synthetic_success_tail_transitions"] == 0
    assert stats["synthetic_failure_tail_transitions"] == horizon
    assert stats["padded_failure_sections"] == 1
    assert stats["padded_failure_sections_by_reason"] == {end_reason: 1}
    assert stats["synthetic_failure_disc_reward_transitions"] == horizon
    assert buffer._get_valid_start_indices_locked() == list(range(7))  # noqa: SLF001
    replay = VASTReplayBuffer(
        base_buffer=buffer,
        cfg=VASTConfig(action_horizon=horizon, vast_max_k=1, disc_reward_coef=0.0),
    )
    assert all(
        replay._raw_chunk_done_locked(start) is False  # noqa: SLF001
        for start in buffer._get_valid_start_indices_locked()  # noqa: SLF001
    )


def test_vast_intervention_failure_adds_continuing_absorbing_boundary() -> None:
    horizon = 4
    length = 6
    policy = [
        Transition(
            obs={
                "state": np.asarray([step], dtype=np.float32),
                "agentview": np.full((4, 4, 3), step, dtype=np.uint8),
            },
            action=np.asarray([step], dtype=np.float32),
            reward=-1.0,
            next_obs={
                "state": np.asarray([step + 1], dtype=np.float32),
                "agentview": np.full((4, 4, 3), step + 1, dtype=np.uint8),
            },
            done=step == length - 1,
            info={
                "episode_index": 0,
                "episode_step": step,
                "success": False,
                "policy_section_end_reason": "ended_human_intervention",
                "nnpu_disc_intrinsic": -0.25,
                "nnpu_failure_score": 0.5,
                "nnpu_threshold": 0.0,
            },
        )
        for step in range(length)
    ]
    source_final_info = dict(policy[-1].info or {})

    buffer, stats = build_vast_finetune_buffer(
        policy,
        camera_names=["agentview"],
        image_size=4,
        action_horizon=horizon,
        warmup_transitions_path=None,
        reward_failure=-1.0,
        capacity=32,
    )

    assert len(policy) == length
    assert policy[-1].done is True
    assert policy[-1].info == source_final_info
    assert len(buffer) == length + horizon
    assert stats["synthetic_failure_tail_transitions"] == horizon
    assert stats["padded_failure_sections"] == 1
    assert stats["failure_absorbing_boundary_windows"] == 1
    assert stats["synthetic_failure_disc_reward_transitions"] == horizon
    assert stats["padded_failure_sections_by_reason"] == {
        "ended_human_intervention": 1
    }
    valid_starts = buffer._get_valid_start_indices_locked()  # noqa: SLF001
    assert valid_starts == list(range(length + 1))
    assert [index for index, item in enumerate(buffer._storage) if item.done] == [  # noqa: SLF001
        length + horizon - 1
    ]

    synthetic = buffer._storage[length:]  # noqa: SLF001
    assert len(synthetic) == horizon
    assert all(item.reward == -1.0 for item in synthetic)
    assert all((item.info or {})["success"] is False for item in synthetic)
    assert all((item.info or {})["absorbing_failure"] is True for item in synthetic)
    assert all(
        (item.info or {})["synthetic_vast_failure_tail"] is True
        for item in synthetic
    )
    assert all(
        "nnpu_disc_intrinsic" not in (item.info or {})
        and "nnpu_failure_score" not in (item.info or {})
        and "nnpu_threshold" not in (item.info or {})
        for item in synthetic
    )
    assert all(np.asarray(item.obs["state"]).item() == length for item in synthetic)
    assert all(np.asarray(item.next_obs["state"]).item() == length for item in synthetic)

    replay = VASTReplayBuffer(
        base_buffer=buffer,
        cfg=VASTConfig(action_horizon=horizon, vast_max_k=1, disc_reward_coef=0.0),
    )
    assert all(
        replay._raw_chunk_done_locked(start) is False  # noqa: SLF001
        for start in valid_starts
    )


def test_phase_a_without_relabel_rejects_reward_mismatch() -> None:
    cfg = _cfg()
    cfg.disc_reward_coef = 0.2
    with pytest.raises(ValueError, match="relabel_disc_reward=true"):
        train_offline._resolve_vast_reward_semantics(
            cfg,
            {**asdict(cfg), "disc_reward_coef": 0.0},
            skip_rl=False,
            relabel_disc_reward=False,
        )


def test_skip_rl_rejects_discriminator_reward_mismatch() -> None:
    cfg = _cfg()
    cfg.disc_reward_coef = 0.2
    with pytest.raises(ValueError, match="skip_rl.*cannot be skipped"):
        train_offline._resolve_vast_reward_semantics(
            cfg,
            {**asdict(cfg), "disc_reward_coef": 0.0},
            skip_rl=True,
            relabel_disc_reward=True,
        )


@pytest.mark.parametrize("estimator", ["gae", "td1"])
def test_phase_b_dispatches_only_vast_value_advantage(
    monkeypatch: pytest.MonkeyPatch,
    estimator: str,
) -> None:
    captured: dict[str, object] = {}
    expected = (object(), object(), {7: 0})

    def _fake_precompute(**kwargs: object) -> tuple[object, object, dict[int, int]]:
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(train_offline, "precompute_offline_advantage", _fake_precompute)
    result = train_offline._precompute_phase_b_advantage(
        base_buffer=object(),
        vast_learner=object(),
        encoder=object(),
        discriminator=object(),
        vast_cfg=object(),
        device="cuda:0",
        encode_batch_size=32,
        estimator=estimator,
        gae_lambda=0.6,
    )

    assert result is expected
    assert captured["estimator"] == estimator
    assert captured["gae_lambda"] == pytest.approx(0.6)
    assert captured["device"] == "cuda:0"


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


def test_vast_checkpoint_validation_accepts_independent_ensemble_schema_v8() -> None:
    cfg = VASTConfig(
        vast_v_mode="indep_ensemble",
        v_ensemble_size=3,
        action_horizon=2,
        vast_max_k=4,
        vast_sampling_seed=17,
        disc_reward_coef=0.0,
    )
    payload = {
        "schema_version": 8,
        "vast_state": {
            "learner_schema_version": 8,
            "vast_v_mode": "indep_ensemble",
            "ensemble_method": "independent_v_mean",
        },
        "algorithm": "vast_value_stitching_adaptation",
        "cfg": asdict(cfg),
        "encoder_meta": {
            "algorithm": "vast_value_stitching_adaptation",
            "vast_v_mode": "indep_ensemble",
            "ensemble_method": "independent_v_mean",
            "finetuned_offline": True,
        },
    }
    validate_vast_checkpoint_payload(payload, cfg, require_finetuned=True)


def test_vast_checkpoint_validation_rejects_legacy_ensemble_lcb() -> None:
    cfg = _cfg()
    payload = {
        "schema_version": 7,
        "vast_state": {"learner_schema_version": 7, "vast_v_mode": "ensemble_lcb"},
        "algorithm": "vast_value_stitching_adaptation",
        "cfg": {**asdict(cfg), "vast_v_mode": "ensemble_lcb"},
        "encoder_meta": {"algorithm": "vast_value_stitching_adaptation"},
    }
    with pytest.raises(ValueError, match="Legacy ensemble_lcb"):
        validate_vast_checkpoint_payload(payload, cfg, require_finetuned=False)


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
