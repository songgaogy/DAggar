from __future__ import annotations

from pathlib import Path

from robosuite.pipeline.algorithms.dipole.agent import DipoleAgent
from robosuite.pipeline.algorithms.dipole.common import DipoleConfig, TrainerConfig
from robosuite.pipeline.common.types import EncoderConfig


class _Core:
    def __init__(self) -> None:
        self.loaded = None
        self.instruction = None

    def set_language_instruction(self, value: str) -> None:
        self.instruction = value

    def load_dual_model_state(self, value: dict) -> None:
        self.loaded = value

    def state_dict(self) -> dict:
        return {"core_pos": {"model": {}}, "core_neg": {"model": {}}}


def _agent() -> DipoleAgent:
    agent = object.__new__(DipoleAgent)
    agent.model_cfg = {"original": True}
    agent.camera_names = ["agentview"]
    agent.task_name = "PickPlaceCereal"
    agent.language_instruction = "old"
    agent.encoder_config = EncoderConfig()
    agent.flow_config = DipoleConfig(action_dim=2, proprio_dim=4)
    agent.trainer_config = TrainerConfig(batch_size=2)
    agent.core = _Core()
    agent.reset_policy_state = lambda: None
    return agent


def test_dipole_checkpoint_loader_uses_weights_only_core(monkeypatch) -> None:
    core_state = {"core_pos": {"model": {}}, "core_neg": {"model": {}}}
    payload = {
        "core": core_state,
        "model_cfg": {"new": True},
        "camera_names": ["robot0_eye_in_hand"],
        "task_name": "PickPlaceCereal",
        "language_instruction": "pick cereal",
    }
    monkeypatch.setattr(
        "robosuite.pipeline.algorithms.dipole.agent.torch.load",
        lambda *args, **kwargs: payload,
    )
    agent = _agent()

    loaded = agent.load_policy_checkpoint(Path("policy.pt"))

    assert loaded is payload
    assert agent.core.loaded is core_state
    assert agent.core.instruction == "pick cereal"
    assert agent.camera_names == ["robot0_eye_in_hand"]


def test_base_checkpoint_loader_dispatches_to_flow_loader(monkeypatch) -> None:
    payload = {"ema_model": {}}
    monkeypatch.setattr(
        "robosuite.pipeline.algorithms.dipole.agent.torch.load",
        lambda *args, **kwargs: payload,
    )
    agent = _agent()
    calls = []
    agent.load_flow_policy_checkpoint = lambda path, task_name=None: calls.append(
        (path, task_name)
    ) or payload

    loaded = agent.load_policy_checkpoint("base.pt", task_name="PickPlaceCereal")

    assert loaded is payload
    assert calls == [(Path("base.pt"), "PickPlaceCereal")]


def test_policy_checkpoint_preserves_parent_environment_contract() -> None:
    agent = _agent()
    parent = {
        "task_metadata_map": {"PickPlaceCereal": {"env_name": "PickPlaceCereal"}},
        "task_prompt_map": {"PickPlaceCereal": "pick cereal"},
    }

    payload = agent.build_checkpoint_payload(parent_payload=parent)

    assert payload["task_metadata_map"] == parent["task_metadata_map"]
    assert payload["task_prompt_map"] == parent["task_prompt_map"]
    assert payload["task_metadata_map"] is not parent["task_metadata_map"]
