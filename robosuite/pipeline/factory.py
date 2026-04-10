from __future__ import annotations

from typing import Any, Callable

from .algorithms.flow_dagger import FlowDaggerAgent
from .algorithms.hg_dagger import HGDaggerAgent
from .algorithms.hil_serl import HILSERLAgent
from .base import (
    DummyPolicyRuntime,
    FlowMultiPolicyRuntime,
    LPBDiceOnlineDiscriminator,
    NullOnlineDiscriminator,
    TPUDOnlineDiscriminator,
)


PolicyBuilder = Callable[[Any, str, Any], object]
DiscriminatorBuilder = Callable[[Any], object]
AlgorithmBuilder = Callable[..., object]

_POLICY_BUILDERS: dict[str, PolicyBuilder] = {}
_DISCRIMINATOR_BUILDERS: dict[str, DiscriminatorBuilder] = {}
_ALGORITHM_BUILDERS: dict[str, AlgorithmBuilder] = {}


def register_policy(name: str):
    def decorator(builder: PolicyBuilder):
        _POLICY_BUILDERS[str(name)] = builder
        return builder

    return decorator


def register_discriminator(name: str):
    def decorator(builder: DiscriminatorBuilder):
        _DISCRIMINATOR_BUILDERS[str(name)] = builder
        return builder

    return decorator


def register_algorithm(name: str):
    def decorator(builder: AlgorithmBuilder):
        _ALGORITHM_BUILDERS[str(name)] = builder
        return builder

    return decorator


@register_policy("dummy")
def _build_dummy_policy(cfg: Any, env_name: str, env) -> DummyPolicyRuntime:
    return DummyPolicyRuntime(action_dim=env.action_spec[0].shape[0])


@register_policy("flow_multi")
def _build_flow_multi_policy(cfg: Any, env_name: str, env) -> FlowMultiPolicyRuntime:
    return FlowMultiPolicyRuntime(cfg=cfg, env_name=env_name, env=env)


@register_discriminator("none")
def _build_null_discriminator(cfg: Any) -> NullOnlineDiscriminator:
    return NullOnlineDiscriminator()


@register_discriminator("lpb_dice")
def _build_lpb_dice_discriminator(cfg: Any) -> LPBDiceOnlineDiscriminator:
    return LPBDiceOnlineDiscriminator(cfg=cfg)


@register_discriminator("bce")
def _build_bce_discriminator(cfg: Any) -> TPUDOnlineDiscriminator:
    return TPUDOnlineDiscriminator(cfg=cfg)


@register_algorithm("hil-serl")
def _build_hil_serl_algorithm(
    cfg: Any,
    observation_space=None,
    action_space=None,
    observation_example=None,
    sample_action=None,
    action_low=None,
    action_high=None,
    device: str | None = None,
) -> HILSERLAgent:
    return HILSERLAgent.from_config(
        cfg=cfg,
        observation_space=observation_space,
        action_space=action_space,
        observation_example=observation_example,
        sample_action=sample_action,
        action_low=action_low,
        action_high=action_high,
        device=device,
    )


@register_algorithm("hg-dagger")
def _build_hg_dagger_algorithm(
    cfg: Any,
    observation_space=None,
    action_space=None,
    observation_example=None,
    sample_action=None,
    action_low=None,
    action_high=None,
    device: str | None = None,
) -> HGDaggerAgent:
    return HGDaggerAgent.from_config(
        cfg=cfg,
        observation_space=observation_space,
        action_space=action_space,
        observation_example=observation_example,
        sample_action=sample_action,
        action_low=action_low,
        action_high=action_high,
        device=device,
    )


@register_algorithm("flow-dagger")
def _build_flow_dagger_algorithm(
    cfg: Any,
    observation_space=None,
    action_space=None,
    observation_example=None,
    sample_action=None,
    action_low=None,
    action_high=None,
    device: str | None = None,
) -> FlowDaggerAgent:
    return FlowDaggerAgent.from_config(
        cfg=cfg,
        observation_space=observation_space,
        action_space=action_space,
        observation_example=observation_example,
        sample_action=sample_action,
        action_low=action_low,
        action_high=action_high,
        device=device,
    )


def build_policy(cfg: Any, env_name: str, env):
    policy_type = str(cfg.type)
    if policy_type not in _POLICY_BUILDERS:
        raise KeyError(f"Unsupported policy type: {policy_type}. Available: {sorted(_POLICY_BUILDERS.keys())}")
    return _POLICY_BUILDERS[policy_type](cfg, env_name, env)


def build_online_discriminator(cfg: Any):
    enabled = bool(getattr(cfg, "enabled", True))
    disc_type = "none" if not enabled else str(cfg.type)
    if disc_type not in _DISCRIMINATOR_BUILDERS:
        raise KeyError(
            f"Unsupported discriminator type: {disc_type}. "
            f"Available: {sorted(_DISCRIMINATOR_BUILDERS.keys())}"
        )
    return _DISCRIMINATOR_BUILDERS[disc_type](cfg)


def build_algorithm(
    cfg: Any,
    observation_space=None,
    action_space=None,
    observation_example=None,
    sample_action=None,
    action_low=None,
    action_high=None,
    device: str | None = None,
):
    algorithm_type = str(getattr(cfg, "type", None) or (cfg.get("type") if isinstance(cfg, dict) else ""))
    if algorithm_type not in _ALGORITHM_BUILDERS:
        raise KeyError(
            f"Unsupported algorithm type: {algorithm_type}. Available: {sorted(_ALGORITHM_BUILDERS.keys())}"
        )
    return _ALGORITHM_BUILDERS[algorithm_type](
        cfg,
        observation_space=observation_space,
        action_space=action_space,
        observation_example=observation_example,
        sample_action=sample_action,
        action_low=action_low,
        action_high=action_high,
        device=device,
    )
