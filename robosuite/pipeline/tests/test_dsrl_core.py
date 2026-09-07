from __future__ import annotations

import copy

import pytest
import torch

from robosuite.pipeline.src.dsrl import DSRLAgent, DSRLBatch, DSRLConfig, DSRLTrainer, NetworkConfig


def _config(**overrides) -> DSRLConfig:
    values = {
        "network": NetworkConfig(
            visual_dim=12,
            proprio_dim=3,
            action_horizon=2,
            action_dim=2,
            hidden_dims=(16, 16, 16),
        ),
        "learner_device": "cuda:0",
        "batch_size": 4,
        "utd_steps": 2,
    }
    values.update(overrides)
    return DSRLConfig(**values)


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.fail("CUDA is required by the DSRL test suite.")


def _batch(config: DSRLConfig, batch_size: int | None = None) -> DSRLBatch:
    size = batch_size or config.batch_size
    device = torch.device(config.learner_device)
    network = config.network
    return DSRLBatch(
        visual_features=torch.randn(size, 3, network.visual_dim // 3, device=device),
        proprio=torch.randn(size, network.proprio_dim, device=device),
        latents=torch.empty(size, network.action_horizon, network.action_dim, device=device).uniform_(
            -network.latent_limit, network.latent_limit
        ),
        rewards=torch.randint(0, 2, (size,), device=device).float(),
        dones=torch.randint(0, 2, (size,), device=device).float(),
        next_visual_features=torch.randn(size, 3, network.visual_dim // 3, device=device),
        next_proprio=torch.randn(size, network.proprio_dim, device=device),
    )


def test_widowx_defaults_and_cpu_rejection() -> None:
    config = DSRLConfig()
    assert config.network.hidden_dims == (1024, 1024, 1024)
    assert config.network.latent_limit == 2.0
    assert config.gamma == 0.97
    assert config.utd_steps == 30
    assert config.target_entropy == 0.0
    with pytest.raises(ValueError, match="explicit CUDA"):
        DSRLAgent(_config(learner_device="cpu"))


def test_network_shapes_direct_state_and_latent_bounds() -> None:
    _require_cuda()
    config = _config()
    agent = DSRLAgent(config)
    batch = _batch(config)
    state = agent._state(batch.visual_features, batch.proprio)
    half_state = agent._state(batch.visual_features.half(), batch.proprio.half())
    latent, log_prob = agent.actor.sample(state)
    q1, q2 = agent.qa(state, batch.latents)
    assert state.shape == (config.batch_size, config.network.visual_dim + config.network.proprio_dim)
    assert half_state.dtype == torch.float32
    assert latent.shape == (config.batch_size, config.network.chunk_dim)
    assert log_prob.shape == (config.batch_size, 1)
    assert torch.all(latent.abs() <= config.network.latent_limit)
    assert q1.shape == q2.shape == (config.batch_size, 1)
    assert all(not parameter.requires_grad for parameter in agent.target_qa.parameters())
    assert agent.act(batch.visual_features, batch.proprio).shape == batch.latents.shape
    assert not hasattr(agent, "bottleneck")
    assert not hasattr(agent, "qw")
    assert not hasattr(agent, "target_actor")


def test_sac_update_changes_actor_q_alpha_and_polyak_target() -> None:
    _require_cuda()
    config = _config(initial_alpha=0.5)
    agent = DSRLAgent(config)
    batch = _batch(config)
    actor_before = copy.deepcopy(agent.actor.state_dict())
    q_before = copy.deepcopy(agent.qa.state_dict())
    target_before = copy.deepcopy(agent.target_qa.state_dict())
    log_alpha_before = agent.log_alpha.detach().clone()
    metrics = agent.update(batch)

    assert any(not torch.equal(value, agent.actor.state_dict()[name]) for name, value in actor_before.items())
    assert any(not torch.equal(value, agent.qa.state_dict()[name]) for name, value in q_before.items())
    assert any(
        not torch.equal(value, agent.target_qa.state_dict()[name]) for name, value in target_before.items()
    )
    assert not torch.equal(log_alpha_before, agent.log_alpha.detach())
    assert (agent.qa_updates, agent.actor_updates, agent.alpha_updates) == (1, 1, 1)
    assert {"qa_loss", "actor_loss", "alpha_loss", "target_q_mean"} <= metrics.keys()


def test_batch_rejects_non_binary_reward() -> None:
    _require_cuda()
    config = _config()
    batch = _batch(config)
    batch.rewards[0] = -1.0
    with pytest.raises(ValueError, match="rewards must contain only zero or one"):
        batch.validate(
            action_horizon=config.network.action_horizon,
            action_dim=config.network.action_dim,
            device=torch.device(config.learner_device),
        )


def test_trainer_utd_count_and_checkpoint_round_trip() -> None:
    _require_cuda()
    config = _config(utd_steps=3)
    calls: list[int] = []

    def provider(batch_size: int) -> DSRLBatch:
        calls.append(batch_size)
        return _batch(config, batch_size)

    agent = DSRLAgent(config)
    trainer = DSRLTrainer(agent, provider)
    metrics = trainer.update_cycle()
    assert calls == [config.batch_size] * 3
    assert (agent.qa_updates, agent.actor_updates, agent.alpha_updates) == (3, 3, 3)
    assert metrics["qa_updates_this_cycle"] == 3.0
    assert "qw_updates_this_cycle" not in metrics

    state = trainer.state_dict()
    serialized = str(state.keys()) + str(state["agent"].keys())
    assert "qw" not in serialized
    assert "bottleneck" not in serialized
    assert "target_actor" not in serialized
    restored = DSRLTrainer(DSRLAgent(config), provider)
    restored.load_state_dict(state)
    assert restored.cycles == 1
    assert restored.agent.qa_updates == 3
    for name, value in agent.actor.state_dict().items():
        assert torch.equal(value, restored.agent.actor.state_dict()[name])
