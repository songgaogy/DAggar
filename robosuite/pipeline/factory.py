from __future__ import annotations

from typing import Any, Callable

from .algorithms.dipole import DipoleAgent


AlgorithmBuilder = Callable[..., object]

_ALGORITHM_BUILDERS: dict[str, AlgorithmBuilder] = {}


def register_algorithm(name: str):
    def decorator(builder: AlgorithmBuilder):
        _ALGORITHM_BUILDERS[str(name)] = builder
        return builder

    return decorator


@register_algorithm("dipole")
@register_algorithm("dipole_rl")
def _build_dipole_algorithm(
    cfg: Any,
    observation_space=None,
    action_space=None,
    observation_example=None,
    sample_action=None,
    action_low=None,
    action_high=None,
    device: str | None = None,
) -> DipoleAgent:
    """Builder for both `dipole` and `dipole_rl` algorithm types.

    The same `DipoleAgent` is used in both modes — the RL additions
    (VAST learner, frozen nnPU discriminator, AdvantageGProvider) are owned by the
    trainer in `train_dipole_rl.py`, not the agent. `agent.from_config`
    reads `algorithm.dipole.g_mode` to decide whether to expect an
    AdvantageGProvider attachment later.
    """
    return DipoleAgent.from_config(
        cfg=cfg,
        observation_space=observation_space,
        action_space=action_space,
        observation_example=observation_example,
        sample_action=sample_action,
        action_low=action_low,
        action_high=action_high,
        device=device,
    )


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
