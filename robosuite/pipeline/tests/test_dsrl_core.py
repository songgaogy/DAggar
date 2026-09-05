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
            state_dim=8,
            action_horizon=2,
            action_dim=2,
            hidden_dims=(16, 16, 16),
        ),
        "learner_device": "cuda:0",
        "batch_size": 4,
        "utd_steps": 2,
        "qw_steps": 1,
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
        dino_features=torch.randn(size, 3, network.visual_dim // 3, device=device),
        proprio=torch.randn(size, network.proprio_dim, device=device),
        flow_context=torch.randn(size, 5, device=device),
        actions=torch.randn(size, network.action_horizon, network.action_dim, device=device),
        rewards=torch.randn(size, device=device),
        dones=torch.randint(0, 2, (size,), device=device).float(),
        next_dino_features=torch.randn(size, 3, network.visual_dim // 3, device=device),
        next_proprio=torch.randn(size, network.proprio_dim, device=device),
        next_flow_context=torch.randn(size, 5, device=device),
    )


def _decoder(_context: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
    return torch.tanh(latent)


def test_cpu_device_is_rejected_without_tensor_work() -> None:
    with pytest.raises(ValueError, match="explicit CUDA"):
        DSRLAgent(_config(learner_device="cpu"), _decoder)


def test_network_shapes_and_latent_bounds() -> None:
    _require_cuda()
    config = _config()
    agent = DSRLAgent(config, _decoder)
    batch = _batch(config)
    state = agent.bottleneck(batch.dino_features, batch.proprio)
    half_state = agent.bottleneck(batch.dino_features.half(), batch.proprio.half())
    latent, log_prob = agent.actor.sample(state)
    q1, q2 = agent.qa(state, batch.actions)
    assert state.shape == (config.batch_size, config.network.state_dim)
    assert half_state.dtype == torch.float32
    assert latent.shape == (config.batch_size, config.network.chunk_dim)
    assert log_prob.shape == (config.batch_size, 1)
    assert torch.all(latent.abs() <= config.network.latent_limit)
    assert q1.shape == q2.shape == (config.batch_size, 1)
    assert all(not parameter.requires_grad for parameter in agent.target_qa.parameters())
    decoded = agent.act(batch.dino_features, batch.proprio, batch.flow_context)
    assert decoded.shape == batch.actions.shape


def test_gradient_boundaries_for_main_and_qw_updates() -> None:
    _require_cuda()
    config = _config()
    agent = DSRLAgent(config, _decoder)
    batch = _batch(config)
    bottleneck_before = copy.deepcopy(agent.bottleneck.state_dict())
    agent.update_qa_actor(batch)
    assert any(
        not torch.equal(value, agent.bottleneck.state_dict()[name]) for name, value in bottleneck_before.items()
    )
    assert all(parameter.grad is None for parameter in agent.qw.parameters())
    bottleneck_before = copy.deepcopy(agent.bottleneck.state_dict())
    actor_before = copy.deepcopy(agent.actor.state_dict())
    agent.update_qw(batch)
    assert any(
        not torch.equal(value, agent.bottleneck.state_dict()[name]) for name, value in bottleneck_before.items()
    )
    assert all(torch.equal(value, agent.actor.state_dict()[name]) for name, value in actor_before.items())


def test_trainer_order_counts_and_checkpoint_round_trip() -> None:
    _require_cuda()
    config = _config(utd_steps=3, qw_steps=2)
    calls: list[int] = []

    def provider(batch_size: int) -> DSRLBatch:
        calls.append(batch_size)
        return _batch(config, batch_size)

    agent = DSRLAgent(config, _decoder)
    trainer = DSRLTrainer(agent, provider)
    order: list[str] = []
    original_main = agent.update_qa_actor
    original_qw = agent.update_qw

    def tracked_main(batch: DSRLBatch) -> dict[str, float]:
        order.append("main")
        return original_main(batch)

    def tracked_qw(batch: DSRLBatch) -> dict[str, float]:
        order.append("qw")
        return original_qw(batch)

    agent.update_qa_actor = tracked_main
    agent.update_qw = tracked_qw
    metrics = trainer.update_cycle()
    assert calls == [config.batch_size] * 5
    assert order == ["main", "main", "main", "qw", "qw"]
    assert (agent.qa_updates, agent.actor_updates, agent.alpha_updates, agent.qw_updates) == (3, 3, 3, 2)
    assert metrics["qa_updates_this_cycle"] == 3.0
    assert metrics["qw_updates_this_cycle"] == 2.0

    restored = DSRLTrainer(DSRLAgent(config, _decoder), provider)
    restored.load_state_dict(trainer.state_dict())
    assert restored.cycles == 1
    assert restored.agent.qa_updates == 3
    assert restored.agent.qw_updates == 2
    for name, value in agent.actor.state_dict().items():
        assert torch.equal(value, restored.agent.actor.state_dict()[name])
