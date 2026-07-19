"""Replay-buffer helpers for offline DIPOLE training."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from robosuite.pipeline.algorithms.flow_dagger.replay_buffer import FlowDaggerReplayBuffer
from robosuite.pipeline.common.types import Transition
from robosuite.pipeline.offline.utils.episode_dataset import ROUTE_POS_ONLY

logger = logging.getLogger(__name__)

# HDF5 loader signature: (path, demo_names=None) -> list[Transition]
Hdf5Loader = Callable[..., list[Transition]]


def _split_into_episodes(transitions: list[Transition]) -> list[list[Transition]]:
    """Split a flat transition list into per-episode lists using ``done``.

    Episodes are delimited by ``Transition.done``; a trailing run without a
    terminal ``done`` is still emitted as its own episode.
    """
    episodes: list[list[Transition]] = []
    current: list[Transition] = []
    for t in transitions:
        current.append(t)
        if bool(t.done):
            episodes.append(current)
            current = []
    if current:
        episodes.append(current)
    return episodes


def _trajectory_kind(namespace: str) -> str:
    """Classify an offline trajectory by its namespace suffix."""
    ns = str(namespace).lower()
    if "success_rollout" in ns:
        return "success"
    if "fail_rollout" in ns:
        return "fail"
    if "expert" in ns:
        return "expert"
    return "unknown"


def _materialize_run(
    run: list[Transition],
    *,
    episode_index: int,
    namespace: str,
    mark_intervention: bool | None = None,
    route: str | None = None,
) -> list[Transition]:
    """Re-stamp a contiguous run as a standalone episode.

    Rewrites ``episode_index`` / ``episode_step`` / ``buffer_role`` and forces
    ``done`` only on the final frame so valid chunk windows stay inside the run.
    When ``mark_intervention`` is given it overrides ``is_intervention`` on every
    frame (pretrain rows set it True so the policy forces ``w_pos=1`` on them).
    """
    out: list[Transition] = []
    n = len(run)
    for step, src in enumerate(run):
        info = dict(src.info or {})
        info["episode_index"] = int(episode_index)
        info["episode_step"] = int(step)
        info["episode_namespace"] = str(namespace)
        info["buffer_role"] = "offline"
        if route is not None:
            info["route"] = str(route)
        is_intervention = (
            bool(src.is_intervention) if mark_intervention is None else bool(mark_intervention)
        )
        out.append(
            Transition(
                obs=src.obs,
                action=src.action,
                reward=src.reward,
                next_obs=src.next_obs,
                done=bool(step == n - 1),
                grasp_penalty=src.grasp_penalty,
                is_intervention=is_intervention,
                info=info,
                reward_source=src.reward_source or "precomputed",
                demo_source=src.demo_source or namespace,
            )
        )
    return out


def load_pretrain_transitions(
    *,
    data_root: str | Path,
    task_name: str,
    pretrain_dir: str,
    hdf5_loader: Hdf5Loader,
    max_num_trajectories: int | None = None,
    episode_index_base: int = 0,
) -> tuple[list[Transition], int]:
    """Load all frames from the expert pretrain HDF5 demos.

    Returns ``(transitions, next_episode_index_base)``. ``hdf5_loader`` is the
    same closure ``train_dipole``/warmup build around
    :func:`load_hdf5_demos_into_flow_transitions` (it owns the env + proprio
    extractor).
    """
    raw_pretrain = Path(str(pretrain_dir).strip())
    if raw_pretrain.exists():
        demo_dir = raw_pretrain
    else:
        demo_dir = Path(data_root) / str(task_name) / str(pretrain_dir).strip().lstrip("/")
    if not demo_dir.exists():
        raise FileNotFoundError(
            f"pretrain_data dir not found: {demo_dir} "
            "(check offline.pretrain_data_path or offline.data_root / task_name / pretrain_dir)."
        )
    if demo_dir.is_file() and demo_dir.suffix.lower() in {".hdf5", ".h5"}:
        demo_paths = [demo_dir]
    else:
        demo_paths = [
            path
            for path in sorted(demo_dir.iterdir())
            if path.is_file() and path.suffix.lower() in {".hdf5", ".h5"}
        ]
    if not demo_paths:
        raise FileNotFoundError(f"pretrain_data dir has no .hdf5/.h5 files: {demo_dir}")

    raw: list[Transition] = []
    remaining = max_num_trajectories
    for path in demo_paths:
        if remaining is not None and remaining <= 0:
            break
        loaded = hdf5_loader(path, demo_names=None)
        raw.extend(loaded)

    # Re-annotate as contiguous episodes with a unique episode_index range.
    episodes = _split_into_episodes(raw)
    if remaining is not None:
        episodes = episodes[: int(remaining)]
    out: list[Transition] = []
    next_index = int(episode_index_base)
    for episode in episodes:
        out.extend(
            _materialize_run(
                episode,
                episode_index=next_index,
                namespace=f"{task_name}/pretrain_expert",
                # Pretrain rows update only the positive branch (w_pos=1): the
                # routed weight policy forces this on pos_only rows.
                mark_intervention=True,
                route=ROUTE_POS_ONLY,
            )
        )
        next_index += 1
    logger.info(
        "[offline] pretrain_data: %d files -> %d transitions, %d episodes",
        len(demo_paths),
        len(out),
        next_index - int(episode_index_base),
    )
    return out, next_index


def populate_replay_buffer(
    buffer: FlowDaggerReplayBuffer,
    transitions: list[Transition],
) -> int:
    """Add ``transitions`` to ``buffer`` (typically ``agent.online_buffer``).

    Returns the buffer's valid-sequence count after insertion.
    """
    for transition in transitions:
        buffer.add(transition)
    return int(buffer.num_valid_sequences())


__all__ = [
    "load_pretrain_transitions",
    "populate_replay_buffer",
]
