"""Pure configuration and provenance helpers for discriminator finetuning."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from omegaconf import OmegaConf

from .objectives import LossTermConfig
from .pools import LatentTrajectory


def mapping_config(node: Any, *, name: str) -> dict[str, Any]:
    value = OmegaConf.to_container(node, resolve=True)
    if not isinstance(value, dict):
        raise TypeError(f"{name} must resolve to a mapping.")
    return value


def configured_loss_terms(
    objective_config: Mapping[str, Any],
) -> tuple[LossTermConfig, ...]:
    raw_terms = objective_config.get("terms")
    if not isinstance(raw_terms, Mapping) or not raw_terms:
        raise ValueError("objective.terms must be a non-empty mapping.")
    terms: list[LossTermConfig] = []
    for name, raw in raw_terms.items():
        if not isinstance(raw, Mapping):
            raise TypeError(f"objective term {name!r} must be a mapping.")
        terms.append(LossTermConfig(name=str(name), **dict(raw)))
    active = tuple(term for term in terms if term.active)
    if not active:
        raise ValueError("At least one enabled loss term must have positive weight.")
    active_types = [term.type for term in active]
    if len(active_types) != len(set(active_types)):
        raise ValueError("Only one active term of each loss type is supported.")
    steps = objective_config.get("steps_per_epoch")
    if steps is None and "nnpu" not in active_types:
        raise ValueError(
            "objective.steps_per_epoch must be set when nnPU replay is inactive."
        )
    if steps is not None and (
        isinstance(steps, bool) or not isinstance(steps, int) or steps < 1
    ):
        raise ValueError("objective.steps_per_epoch must be null or a positive integer.")
    return tuple(terms)


def active_training_pool_names(
    loss_terms: Sequence[LossTermConfig],
) -> list[str]:
    return sorted(
        {
            pool
            for term in loss_terms
            if term.active
            for pool in (
                ("pretrain_positive", "pretrain_unlabeled")
                if term.type == "nnpu"
                else ("offline_positive", "offline_gt_negative")
            )
        }
    )


def trajectory_stats(trajectories: Sequence[LatentTrajectory]) -> dict[str, int]:
    dimensions = {int(item.features.shape[1]) for item in trajectories}
    if len(dimensions) > 1:
        raise ValueError(f"Latent dimensions do not match: {sorted(dimensions)}.")
    return {
        "trajectories": int(len(trajectories)),
        "frames": int(sum(int(item.features.shape[0]) for item in trajectories)),
        "latent_dim": 0 if not dimensions else int(next(iter(dimensions))),
    }


def resolved_sampler_config(
    loss_terms: Sequence[LossTermConfig],
    *,
    seed: int,
    device: str,
) -> dict[str, Any]:
    """Record the independent replacement sampler implied by each loss term."""
    pool_names = {
        "nnpu": ("pretrain_positive", "pretrain_unlabeled"),
        "supervised_bce": ("offline_positive", "offline_gt_negative"),
    }
    return {
        "strategy": "independent_uniform_with_replacement",
        "seed": int(seed),
        "device": str(device),
        "terms": {
            term.name: {
                "enabled": bool(term.enabled),
                "active": bool(term.active),
                "positive_pool": pool_names[term.type][0],
                "negative_or_unlabeled_pool": pool_names[term.type][1],
                "batch_size": int(term.batch_size),
                "positive_batch_size": int(term.positive_batch_size),
                "negative_or_unlabeled_batch_size": int(term.negative_batch_size),
            }
            for term in loss_terms
        },
    }


def reserved_unlabeled_provenance(
    trajectories: Sequence[LatentTrajectory],
    *,
    selected_frame_keys: Sequence[Sequence[int]],
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """Describe disjoint reserved-U shards without CPU tensor operations."""
    selected = {(int(key[0]), int(key[1])) for key in selected_frame_keys}
    frames = 0
    removed = 0
    shards: list[dict[str, Any]] = []
    for trajectory in trajectories:
        metadata = trajectory.metadata
        episode = int(metadata["source_episode_index"])
        lo = int(metadata["frame_start"])
        hi = int(metadata["frame_end"])
        run_start: int | None = None
        local_index = 0
        for frame in range(lo, hi):
            keep = (episode, frame) not in selected
            frames += int(keep)
            removed += int(not keep)
            if keep and run_start is None:
                run_start = frame
            if not keep and run_start is not None:
                shards.append(
                    {
                        "id": f"{trajectory.identifier}-reserved-{local_index:03d}",
                        "parent_id": trajectory.identifier,
                        "source_episode_index": episode,
                        "frame_start": run_start,
                        "frame_end": frame,
                        "frames": frame - run_start,
                    }
                )
                local_index += 1
                run_start = None
        if run_start is not None:
            shards.append(
                {
                    "id": f"{trajectory.identifier}-reserved-{local_index:03d}",
                    "parent_id": trajectory.identifier,
                    "source_episode_index": episode,
                    "frame_start": run_start,
                    "frame_end": hi,
                    "frames": hi - run_start,
                }
            )
    return (
        {
            "trajectories": int(len(shards)),
            "frames": int(frames),
            "excluded_gt_negative_frames": int(removed),
        },
        shards,
    )


__all__ = [
    "active_training_pool_names",
    "configured_loss_terms",
    "mapping_config",
    "reserved_unlabeled_provenance",
    "resolved_sampler_config",
    "trajectory_stats",
]
