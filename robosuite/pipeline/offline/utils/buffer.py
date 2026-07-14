"""Offline replay-buffer assembly for offline DIPOLE training.

Loads two disk sources into one replay buffer (`agent.online_buffer`), fully
mixed and sampled uniformly (the demo buffer stays empty). Pretrain frames are
marked ``is_intervention=True`` so the policy forces ``w_pos=1`` on them
(positive-branch-only update); offline_data frames keep ``is_intervention=False``
and are advantage-weighted. See README step-4:

1. ``pretrain_data`` — clean expert HDF5 demos under
   ``<data_root>/<task>/<pretrain_dir>``. Every frame is kept.
2. ``offline_data`` — the assembled transitions exported by the VAST warmup
   (``<data_root>/<task>/<offline_data_dir>/vast_offline_transitions.pt``,
   reloaded verbatim with :meth:`FlowDaggerReplayBuffer.load`). These are the
   ``success_rollout`` / ``fail_rollout`` / ``expert`` splits the warmup saw.
   We filter them per README step-4:
     - success trajectories: keep only the pre-success part (``info["success"]``
       is False);
     - fail trajectories: keep only the no-failure part (``info["gt_fail"]`` is
       False);
     - expert trajectories: skipped here (already supplied by ``pretrain_data``,
       so we do not double-count them).

Filtering can leave temporal gaps inside a trajectory, so each filtered demo is
split into maximal contiguous runs (by the original per-frame ``episode_step``)
and every run is re-annotated as its own episode with a fresh
``episode_index`` / ``episode_step`` / ``done`` boundary. This keeps the
chunk-window valid-start cache (``_get_valid_start_indices_locked``) bounded to
temporally adjacent frames.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from robosuite.pipeline.algorithms.flow_dagger.common import (
    FlowAugmentationConfig,
    ReplayBufferConfig,
)
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


def _orig_episode_step(transition: Transition) -> int:
    info = transition.info or {}
    return int(info.get("episode_step", -1))


def _contiguous_runs(episode: list[Transition]) -> list[list[Transition]]:
    """Split a (possibly filtered) episode into maximal temporally-adjacent runs.

    Adjacency is decided by the original ``info["episode_step"]`` incrementing by
    exactly 1. When that field is unavailable (``-1``) we treat the whole episode
    as a single run (the warmup always stamps it, so this is just defensive).
    """
    if not episode:
        return []
    runs: list[list[Transition]] = []
    current: list[Transition] = [episode[0]]
    prev_step = _orig_episode_step(episode[0])
    for t in episode[1:]:
        step = _orig_episode_step(t)
        if prev_step >= 0 and step == prev_step + 1:
            current.append(t)
        else:
            runs.append(current)
            current = [t]
        prev_step = step
    runs.append(current)
    return runs


def _trajectory_kind(namespace: str) -> str:
    """Classify a trajectory by its ``info["episode_namespace"]`` suffix."""
    ns = str(namespace).lower()
    if "success_rollout" in ns:
        return "success"
    if "fail_rollout" in ns:
        return "fail"
    if "expert" in ns:
        return "expert"
    return "unknown"


def _keep_frame(transition: Transition, kind: str) -> bool:
    """README step-4 frame filter for offline_data."""
    info = transition.info or {}
    if kind == "success":
        return not bool(info.get("success", False))
    if kind == "fail":
        return not bool(info.get("gt_fail", False))
    # expert is skipped at the trajectory level; unknown keeps all frames.
    return True


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


def load_offline_data_transitions(
    *,
    data_root: str | Path,
    task_name: str,
    offline_data_dir: str,
    action_horizon: int,
    camera_names: list[str],
    image_size: int,
    episode_index_base: int = 0,
    keep_kinds: set[str] | None = None,
    mark_intervention: bool | None = None,
) -> tuple[list[Transition], int, dict[str, int]]:
    """Load + filter the warmup-exported offline transitions.

    Returns ``(transitions, next_episode_index_base, stats)``. ``stats`` counts
    kept/dropped frames per trajectory kind for logging/verification.

    ``keep_kinds`` optionally restricts which trajectory kinds are kept (``None``
    keeps the default success + fail; e.g. ``{"success"}`` drops fail entirely for
    a success-only run). ``mark_intervention`` is forwarded to
    :func:`_materialize_run` to override ``is_intervention`` on every kept frame
    (``True`` forces the policy's ``w_pos=1`` positive-branch-only update).
    """
    offline_dir = Path(data_root) / str(task_name) / str(offline_data_dir)
    offline_path = offline_dir / "vast_offline_transitions.pt"
    if not offline_path.exists():
        raise FileNotFoundError(
            f"offline_data not found: {offline_path}. Run init_vast.sh with "
            "warmup.num_trajectories.save_data=true first."
        )

    # Reload verbatim via FlowDaggerReplayBuffer.load (matches how it was saved).
    loader_buffer = FlowDaggerReplayBuffer(
        config=ReplayBufferConfig(capacity=10_000_000, batch_size=1),
        name="offline_data_loader",
        camera_names=list(camera_names),
        action_horizon=int(action_horizon),
        image_size=int(image_size),
        augmentation_config=FlowAugmentationConfig(),
    )
    loader_buffer.load(offline_path)
    storage = list(loader_buffer._storage)  # noqa: SLF001 — read-only access to loaded transitions

    episodes = _split_into_episodes(storage)
    stats: dict[str, int] = {
        "success_kept": 0,
        "success_dropped": 0,
        "fail_kept": 0,
        "fail_dropped": 0,
        "expert_skipped": 0,
        "unknown_kept": 0,
        "kind_skipped": 0,
    }

    out: list[Transition] = []
    next_index = int(episode_index_base)
    for episode in episodes:
        if not episode:
            continue
        namespace = str((episode[0].info or {}).get("episode_namespace", ""))
        kind = _trajectory_kind(namespace)
        if kind == "expert":
            stats["expert_skipped"] += len(episode)
            continue
        if keep_kinds is not None and kind not in keep_kinds:
            # e.g. success-only run drops fail (and unknown) trajectories entirely.
            stats["kind_skipped"] += len(episode)
            continue
        kept = [t for t in episode if _keep_frame(t, kind)]
        dropped = len(episode) - len(kept)
        if kind == "success":
            stats["success_kept"] += len(kept)
            stats["success_dropped"] += dropped
        elif kind == "fail":
            stats["fail_kept"] += len(kept)
            stats["fail_dropped"] += dropped
        else:
            stats["unknown_kept"] += len(kept)
        if not kept:
            continue
        # A filtered episode may have temporal gaps -> split into adjacent runs.
        for run in _contiguous_runs(kept):
            out.extend(
                _materialize_run(
                    run,
                    episode_index=next_index,
                    namespace=namespace,
                    mark_intervention=mark_intervention,
                )
            )
            next_index += 1

    logger.info(
        "[offline] offline_data: %d transitions kept "
        "(success_kept=%d/dropped=%d, fail_kept=%d/dropped=%d, "
        "expert_skipped=%d, unknown_kept=%d, kind_skipped=%d) -> %d episodes",
        len(out),
        stats["success_kept"],
        stats["success_dropped"],
        stats["fail_kept"],
        stats["fail_dropped"],
        stats["expert_skipped"],
        stats["unknown_kept"],
        stats["kind_skipped"],
        next_index - int(episode_index_base),
    )
    return out, next_index, stats


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
    "load_offline_data_transitions",
    "populate_replay_buffer",
]
