"""Non-tensor tests for VAST weights-only checkpoint loading."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from robosuite.pipeline.algorithms.vast import checkpoint as checkpoint_module


class _StateRecorder:
    def __init__(self) -> None:
        self.loaded = None

    def load_state_dict(self, state) -> None:
        self.loaded = state


def _learner():
    cfg = SimpleNamespace(
        vast_max_k=4,
        vast_sampling_seed=17,
        action_horizon=2,
        discount=0.99,
        expectile_tau=0.7,
        vast_comp_coef=1.0,
        output_reward_coef=1.0,
        disc_reward_coef=0.0,
    )
    return SimpleNamespace(
        cfg=cfg,
        state_proj_dim=256,
        proprio_proj_dim=64,
        n_tokens=196,
        proprio_dim=64,
        ensemble_size=3,
        state_feature_dim=384,
        vast_v_mode="indep_ensemble",
        v=_StateRecorder(),
        target_v=_StateRecorder(),
        g=_StateRecorder(),
        v_optims=[object()],
        g_optim=object(),
    )


def _payload():
    cfg = {
        "discount": 0.99,
        "expectile_tau": 0.7,
        "vast_comp_coef": 1.0,
        "output_reward_coef": 1.0,
        "disc_reward_coef": 0.0,
    }
    return {
        "encoder_meta": {
            "state_feature_dim": 384,
            "chunk_feature_dim": 512,
            "policy_action_dim": 7,
        },
        "vast_state": {
            "learner_schema_version": 8,
            "algorithm": "vast_value_stitching_adaptation",
            "vast_v_mode": "indep_ensemble",
            "ensemble_method": "independent_v_mean",
            "state_feature_dim": 384,
            "state_proj_dim": 256,
            "proprio_proj_dim": 64,
            "n_tokens": 196,
            "proprio_dim": 64,
            "v_ensemble_size": 3,
            "vast_max_k": 4,
            "vast_sampling_seed": 17,
            "action_horizon": 2,
            "cfg": cfg,
            "v": {"v": 1},
            "target_v": {"target_v": 2},
            "g": {"g": 3},
            "v_optims": [{"must_not_load": True}],
            "g_optim": {"must_not_load": True},
        },
    }


def test_load_vast_weights_loads_only_network_states(monkeypatch: pytest.MonkeyPatch) -> None:
    learner = _learner()
    payload = _payload()
    monkeypatch.setattr(checkpoint_module, "load_vast_payload", lambda _: payload)

    result = checkpoint_module.load_vast_weights(
        learner,
        "unused.pt",
        expected_state_feature_dim=384,
        expected_chunk_feature_dim=512,
        expected_action_dim=7,
    )

    assert result is payload
    assert learner.v.loaded == {"v": 1}
    assert learner.target_v.loaded == {"target_v": 2}
    assert learner.g.loaded == {"g": 3}


def test_load_vast_weights_checks_encoder_metadata_before_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    learner = _learner()
    payload = _payload()
    payload["encoder_meta"]["chunk_feature_dim"] = 999
    monkeypatch.setattr(checkpoint_module, "load_vast_payload", lambda _: payload)

    with pytest.raises(ValueError, match="chunk_feature_dim"):
        checkpoint_module.load_vast_weights(
            learner,
            "unused.pt",
            expected_state_feature_dim=384,
            expected_chunk_feature_dim=512,
            expected_action_dim=7,
        )

    assert learner.v.loaded is None
    assert learner.target_v.loaded is None
    assert learner.g.loaded is None
