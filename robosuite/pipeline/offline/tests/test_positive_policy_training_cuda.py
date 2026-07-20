"""CUDA contracts for controlled positive-policy training modes."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from robosuite.pipeline.algorithms.dipole.agent import DipoleAgent
from robosuite.pipeline.algorithms.dipole.common import DipoleBatch
from robosuite.pipeline.algorithms.dipole.common import DipoleConfig, TrainerConfig
from robosuite.pipeline.common.types import EncoderConfig
from robosuite.pipeline.offline.utils.episode_dataset import (
    ROUTE_POS_ONLY,
    build_online_success_transitions,
)
from robosuite.pipeline.offline.utils.policy_training import (
    branch_only_update,
    branch_seed,
    sample_static_cache,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Positive-policy branch tests require CUDA.",
)


class _TinyFlow(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(value, device="cuda"))

    def forward(
        self,
        *,
        x_t: torch.Tensor,
        t: torch.Tensor,
        images: torch.Tensor,
        proprio: torch.Tensor,
        language: list[str],
    ) -> torch.Tensor:
        _ = t, images, proprio, language
        return x_t * self.scale + self.scale


def _core() -> SimpleNamespace:
    model_pos = _TinyFlow(0.1)
    model_neg = _TinyFlow(-0.1)
    return SimpleNamespace(
        device=torch.device("cuda:0"),
        model_pos=model_pos,
        model_neg=model_neg,
        optimizer_pos=torch.optim.SGD(model_pos.parameters(), lr=0.01),
        optimizer_neg=torch.optim.SGD(model_neg.parameters(), lr=0.01),
        scaler_pos=torch.amp.GradScaler(device="cuda"),
        scaler_neg=torch.amp.GradScaler(device="cuda"),
        language_instruction="task",
        config=SimpleNamespace(
            lambda_endpoint=0.0,
            lambda_smooth=0.0,
            grad_clip_norm=10.0,
        ),
    )


def _batch(batch_size: int = 4) -> DipoleBatch:
    device = torch.device("cuda:0")
    return DipoleBatch(
        image_obs=torch.zeros(batch_size, 1, 3, 2, 2, device=device),
        image_obs_raw=torch.zeros(batch_size, 1, 3, 2, 2, device=device),
        proprio=torch.zeros(batch_size, 2, device=device),
        proprio_raw=torch.zeros(batch_size, 2, device=device),
        action_sequences=torch.randn(batch_size, 2, 2, device=device),
        action_sequences_raw=torch.zeros(batch_size, 2, 2, device=device),
        is_intervention=torch.zeros(batch_size, device=device, dtype=torch.bool),
        metadata={},
    )


def test_branch_only_update_changes_only_selected_policy() -> None:
    core = _core()
    batch = _batch()
    pos_before = core.model_pos.scale.detach().clone()
    neg_before = core.model_neg.scale.detach().clone()

    metrics = branch_only_update(
        core,
        batch,
        branch="pos",
        weights=torch.ones(batch.batch_size, device="cuda"),
        torch_seed=branch_seed(42, step=0, branch="pos", phase="update"),
    )

    assert not torch.equal(core.model_pos.scale.detach(), pos_before)
    torch.testing.assert_close(core.model_neg.scale.detach(), neg_before)
    assert metrics["w_pos_mean"] == pytest.approx(1.0)
    assert metrics["frac_w_pos_saturated_high"] == pytest.approx(1.0)


class _CudaCache:
    def sample(
        self,
        batch_size: int,
        *,
        device: str,
        rng: np.random.Generator,
        **_: object,
    ) -> DipoleBatch:
        rows = rng.integers(0, 1000, size=batch_size)
        row_tensor = torch.as_tensor(rows, device=device, dtype=torch.float32)
        random_values = torch.rand(batch_size, device=device)
        values = row_tensor + random_values
        return DipoleBatch(
            image_obs=values[:, None, None, None, None],
            image_obs_raw=values[:, None, None, None, None],
            proprio=values[:, None],
            proprio_raw=values[:, None],
            action_sequences=values[:, None, None],
            action_sequences_raw=values[:, None, None],
            is_intervention=torch.zeros(
                batch_size, device=device, dtype=torch.bool
            ),
            metadata={"rows": rows.tolist()},
        )


def test_sampling_rng_is_independent_of_global_and_other_branch_streams() -> None:
    kwargs = {"device": "cuda:0", "augment": True}
    pos_seed = branch_seed(42, step=7, branch="pos", phase="sample")
    expected = sample_static_cache(
        _CudaCache(),
        8,
        sample_kwargs=kwargs,
        numpy_rng=np.random.default_rng(pos_seed),
        torch_seed=pos_seed,
    )

    torch.rand(1024, device="cuda")
    sample_static_cache(
        _CudaCache(),
        8,
        sample_kwargs=kwargs,
        numpy_rng=np.random.default_rng(
            branch_seed(42, step=7, branch="neg", phase="sample")
        ),
        torch_seed=branch_seed(42, step=7, branch="neg", phase="sample"),
    )
    actual = sample_static_cache(
        _CudaCache(),
        8,
        sample_kwargs=kwargs,
        numpy_rng=np.random.default_rng(pos_seed),
        torch_seed=pos_seed,
    )

    assert actual.metadata == expected.metadata
    torch.testing.assert_close(actual.image_obs, expected.image_obs)


def test_branch_seed_streams_are_disjoint() -> None:
    seeds = {
        branch_seed(42, step=step, branch=branch, phase=phase)
        for step in range(3)
        for branch in ("pos", "neg")
        for phase in ("sample", "update")
    }
    assert len(seeds) == 12


def test_pure_success_is_materialized_as_hard_positive_route() -> None:
    length = 4
    policy_action = np.arange(length * 2, dtype=np.float32).reshape(length, 2)
    payload = {
        "camera_names": ["agentview"],
        "episodes": [
            {
                "obs": {
                    "agentview": np.zeros((length, 2, 2, 3), dtype=np.uint8),
                    "state": np.zeros((length, 3), dtype=np.float32),
                },
                "next_obs": {
                    "agentview": np.zeros((length, 2, 2, 3), dtype=np.uint8),
                    "state": np.zeros((length, 3), dtype=np.float32),
                },
                "executed_action": policy_action.copy(),
                "policy_action": policy_action,
                "is_intervention": np.zeros(length, dtype=np.bool_),
                "success": np.asarray([False, False, False, True]),
                "done": np.asarray([False, False, False, True]),
                "terminal_reason": "success",
            }
        ],
    }

    transitions, _, stats = build_online_success_transitions(
        payload,
        action_horizon=2,
        route=ROUTE_POS_ONLY,
    )

    assert len(transitions) == length
    assert stats["pure_success_source_episode_indices"] == [0]
    assert all(item.info["route"] == ROUTE_POS_ONLY for item in transitions)


def test_checkpoint_payload_round_trip_preserves_mode_and_cuda_weights(
    tmp_path,
) -> None:
    agent = object.__new__(DipoleAgent)
    agent.encoder_config = EncoderConfig()
    agent.flow_config = DipoleConfig(
        action_dim=2,
        proprio_dim=3,
        device="cuda:0",
    )
    agent.trainer_config = TrainerConfig(batch_size=4)
    agent.model_cfg = {"test": True}
    agent.camera_names = ["agentview"]
    agent.task_name = "Task"
    agent.language_instruction = "task"
    cuda_weight = torch.arange(4, device="cuda", dtype=torch.float32)
    agent.core = SimpleNamespace(
        state_dict=lambda: {
            "core_pos": {"model": {"weight": cuda_weight}},
            "core_neg": {"model": {"weight": -cuda_weight}},
        }
    )

    payload = agent.build_checkpoint_payload(
        include_buffers=False,
        extra={
            "completed_steps": 5000,
            "positive_training_mode": "filtered_bc",
        },
    )
    path = tmp_path / "step_00005000.pt"
    agent.write_checkpoint_payload(path, payload)
    restored = torch.load(path, map_location="cuda:0", weights_only=False)

    assert restored["extra"]["completed_steps"] == 5000
    assert restored["extra"]["positive_training_mode"] == "filtered_bc"
    assert restored["core"]["core_pos"]["model"]["weight"].is_cuda
    torch.testing.assert_close(
        restored["core"]["core_pos"]["model"]["weight"],
        cuda_weight,
    )
