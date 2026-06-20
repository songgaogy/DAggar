from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class EnvRandomReducer:
    """Apply deterministic per-episode seeds to robosuite rollout envs."""

    base_seed: int | None

    @property
    def enabled(self) -> bool:
        return self.base_seed is not None

    def seed_for_episode(self, episode_index: int) -> int | None:
        if self.base_seed is None:
            return None
        return int(self.base_seed) + int(episode_index)

    def prepare_episode(self, env: Any, episode_index: int, *, seed_global: bool = True) -> int | None:
        episode_seed = self.seed_for_episode(episode_index)
        if episode_seed is None:
            return None
        if seed_global:
            self.seed_global(episode_seed)
        self.seed_env(env, episode_seed)
        return int(episode_seed)

    @staticmethod
    def seed_global(seed: int) -> None:
        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    @staticmethod
    def seed_env(env: Any, seed: int) -> None:
        rng = np.random.default_rng(int(seed))
        for candidate in _iter_env_chain(env):
            # Only record the integer seed on plain `seed` attributes. robosuite's MujocoEnv stores
            # `seed` as a construction-time int that reset() no longer reads, so this is informational
            # only -- the actual layout control comes from the `rng` injection below. Crucially, do
            # NOT overwrite a callable `seed()` method (gym-style wrappers expose one): assigning an
            # int there would clobber the method and break later `env.seed()` calls.
            if hasattr(candidate, "seed") and not callable(getattr(candidate, "seed", None)):
                try:
                    candidate.seed = int(seed)
                except Exception:
                    pass
            if hasattr(candidate, "rng"):
                try:
                    candidate.rng = rng
                except Exception:
                    pass
            _seed_known_sampler_graphs(candidate, rng)


def _iter_env_chain(env: Any):
    seen: set[int] = set()
    stack = [env]
    while stack:
        candidate = stack.pop()
        if candidate is None:
            continue
        obj_id = id(candidate)
        if obj_id in seen:
            continue
        seen.add(obj_id)
        yield candidate
        for attr in ("unwrapped", "env"):
            try:
                child = getattr(candidate, attr)
            except Exception:
                continue
            if child is not candidate:
                stack.append(child)


def _seed_known_sampler_graphs(env: Any, rng: np.random.Generator) -> None:
    for attr in (
        "placement_initializer",
        "placement_sampler",
        "object_placement_sampler",
        "arena",
        "model",
    ):
        if hasattr(env, attr):
            try:
                _seed_rng_graph(getattr(env, attr), rng, seen=set(), depth=0)
            except Exception:
                continue


def _seed_rng_graph(obj: Any, rng: np.random.Generator, *, seen: set[int], depth: int) -> None:
    if obj is None or depth > 8:
        return
    obj_id = id(obj)
    if obj_id in seen:
        return
    seen.add(obj_id)

    if hasattr(obj, "rng"):
        try:
            obj.rng = rng
        except Exception:
            pass

    if isinstance(obj, dict):
        children = obj.values()
    elif isinstance(obj, (list, tuple, set)):
        children = obj
    else:
        children = []
        for attr in ("samplers", "mujoco_objects", "objects", "fixtures"):
            if hasattr(obj, attr):
                try:
                    children.append(getattr(obj, attr))
                except Exception:
                    pass

    for child in children:
        _seed_rng_graph(child, rng, seen=seen, depth=depth + 1)


def seed_range(base_seed: int, count: int) -> range:
    """Inclusive-of-start, exclusive-of-end seed range produced by ``base_seed + episode_index``."""
    return range(int(base_seed), int(base_seed) + int(count))


def ranges_overlap(a_base: int, a_count: int, b_base: int, b_count: int) -> bool:
    """True iff the two ``[base, base+count)`` integer seed ranges intersect."""
    if a_count <= 0 or b_count <= 0:
        return False
    a_lo, a_hi = int(a_base), int(a_base) + int(a_count)  # [a_lo, a_hi)
    b_lo, b_hi = int(b_base), int(b_base) + int(b_count)  # [b_lo, b_hi)
    return a_lo < b_hi and b_lo < a_hi


def assert_disjoint_seed_ranges(
    train_base: int,
    train_count: int,
    eval_base: int,
    eval_count: int,
    *,
    context: str = "",
) -> None:
    """Hard-assert that eval episode seeds never reuse a training episode seed.

    With deterministic layouts ``layout_seed = base_seed + episode_index`` the eval set must live in
    a seed band disjoint from training, otherwise the policy is benchmarked on the exact initial
    configurations it was trained on. Raises ``ValueError`` on any overlap.
    """
    if not ranges_overlap(train_base, train_count, eval_base, eval_count):
        return
    overlap = sorted(set(seed_range(train_base, train_count)) & set(seed_range(eval_base, eval_count)))
    preview = overlap[:10] + (["..."] if len(overlap) > 10 else [])
    suffix = f" [{context}]" if context else ""
    raise ValueError(
        "Eval layout seeds overlap training layout seeds"
        f"{suffix}: train=[{train_base}, {train_base + train_count}) "
        f"eval=[{eval_base}, {eval_base + eval_count}) "
        f"overlap_count={len(overlap)} overlap={preview}. "
        "Pick an eval --seed in a band disjoint from training (e.g. a large offset)."
    )
