"""Convert collected ``offline_episodes.pt`` into routed DIPOLE transitions.

Source: ``data/<task>/offline_data/offline_episodes.pt`` produced by
``offline/utils/collect_data.py``. The payload is a flat dict
``{**metadata, "episodes": [episode_dict, ...]}`` where every ``episode_dict``
holds parallel per-step arrays of length ``T``::

    obs / next_obs : {"state": (T, D), <camera>: (T, H, W, 3) uint8}
    executed_action / policy_action / human_action : (T, A)
    is_intervention : (T,) bool     reward / success / done : (T,)
    episode_step : arange(T)        terminal_reason : str

This module implements PROMPT.md ``[tbd] Offline DIPOLE`` §Data usage: split each
episode on ``is_intervention`` into contiguous **policy** / **human** sections and
emit three streams of :class:`~robosuite.pipeline.common.types.Transition`, each
frame tagged with ``info["route"] in {"advantage", "pos_only", "neg_only"}``:

- **policy_bc** (kept policy sections): ``action = executed_action``, per-frame
  reward by section outcome, ``route="advantage"``. Fed to BOTH the VAST buffer and
  the policy-BC buffer.
- **human_pos** (every human section): ``action = executed_action`` (== human),
  ``route="pos_only"`` (forces ``w_pos=1, w_neg=0``). Policy-BC only.
- **neg** (same human frames, opt-in default on): ``action = policy_action``,
  ``route="neg_only"`` (forces ``w_pos=0, w_neg=1``). Policy-BC only.

Chunk-boundary handling (confirmed with the user):
- Each section becomes its own ``episode_index`` with contiguous ``episode_step``
  so ``FlowDaggerReplayBuffer._is_valid_sequence_start_locked`` never lets a chunk
  window cross a section boundary.
- Human / neg sections shorter than ``H`` are **padded to ``H``** (repeat the last
  frame) — they are BC-only, so a fabricated tail never reaches VAST.
- Policy sections shorter than ``H`` are **dropped** (avoid fabricating ``s'`` for
  VAST).

Reward / done semantics (config-overridable ``reward_success`` / ``reward_fail``):
- **success** policy section (episode ended in ``terminal_reason=="success"``):
  every frame gets ``reward_success`` (default 0) and carries per-frame
  ``info["success"]``, so ``VASTReplayBuffer._build_step_batch`` takes the absorbing
  branch at the success frame and keeps bootstrapping elsewhere.
- any other kept policy section (ended in human-intervention / ``manual_reset``):
  every frame gets ``reward_fail`` (default -1), no ``info["success"]`` key, and
  ``done=True`` on the section's last frame → the VAST else-branch treats the
  section boundary as a truncation-terminal (no bootstrap past ``-1``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from robosuite.pipeline.common.types import Transition

logger = logging.getLogger(__name__)

# Per-frame routing tag read by RoutedSigmoidBranchWeightPolicy (branch_weights.py)
# and threaded through DipoleReplayBuffer / DipoleOfflineStaticCache metadata.
ROUTE_ADVANTAGE = "advantage"
ROUTE_POS_ONLY = "pos_only"
ROUTE_NEG_ONLY = "neg_only"

# terminal_reason values that make the *last* policy section a keepable one.
_KEEP_LAST_POLICY_REASONS = frozenset({"success", "manual_reset"})


@dataclass
class Section:
    kind: str          # "policy" | "human"
    lo: int            # inclusive start frame
    hi: int            # exclusive end frame


@dataclass
class OfflineStreams:
    policy_bc: list[Transition] = field(default_factory=list)
    human_pos: list[Transition] = field(default_factory=list)
    neg: list[Transition] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


def _camera_names_from_payload(payload: dict[str, Any], episodes: list[dict[str, Any]]) -> list[str]:
    """Prefer the metadata camera list; else infer from an episode's obs keys."""
    names = payload.get("camera_names")
    if names:
        return [str(n) for n in names]
    if episodes:
        return [str(k) for k in episodes[0]["obs"].keys() if str(k) != "state"]
    return []


def _split_sections(is_intervention: np.ndarray) -> list[Section]:
    """Maximal contiguous runs of equal ``is_intervention``; alternating kinds."""
    n = int(is_intervention.shape[0])
    sections: list[Section] = []
    if n == 0:
        return sections
    lo = 0
    cur = bool(is_intervention[0])
    for i in range(1, n):
        flag = bool(is_intervention[i])
        if flag != cur:
            sections.append(Section("human" if cur else "policy", lo, i))
            lo = i
            cur = flag
    sections.append(Section("human" if cur else "policy", lo, n))
    return sections


def _keep_policy_section(
    sections: list[Section], idx: int, terminal_reason: str
) -> tuple[bool, str]:
    """Return ``(keep, reason_tag)`` for the policy section at ``sections[idx]``.

    Keep if it is immediately followed by a human section (ended in
    human-intervention) or, when it is the last section, if the episode ended in
    ``success`` / ``manual_reset``. ``reason_tag`` is used for both the outcome
    reward and the drop-stats histogram.
    """
    if idx + 1 < len(sections):
        # Next section is human by construction -> policy handed over to a human.
        return True, "ended_human_intervention"
    # Last section.
    reason = str(terminal_reason)
    if reason in _KEEP_LAST_POLICY_REASONS:
        return True, reason
    return False, f"dropped_{reason or 'unknown'}"


def _frame_obs(obs_arrays: dict[str, Any], i: int, camera_names: list[str]) -> dict[str, np.ndarray]:
    frame: dict[str, np.ndarray] = {
        "state": np.asarray(obs_arrays["state"][i], dtype=np.float32).copy()
    }
    for cam in camera_names:
        frame[cam] = np.asarray(obs_arrays[cam][i], dtype=np.uint8).copy()
    return frame


def _materialize_section(
    *,
    episode: dict[str, Any],
    section: Section,
    camera_names: list[str],
    action_key: str,
    route: str,
    reward_value: float,
    episode_index: int,
    action_horizon: int,
    pad_to_h: bool,
    mark_success: bool,
    source_episode_index: int | None = None,
) -> tuple[list[Transition], bool]:
    """Turn one section into a standalone-episode list of transitions.

    Returns ``(transitions, was_padded)``. Assigns contiguous ``episode_step``
    (0..n-1) and forces ``done`` only on the last frame so chunk windows stay
    inside the section. When ``pad_to_h`` and the section is shorter than ``H``,
    the last frame's obs/next_obs/action are repeated up to ``H`` (BC-only use).
    """
    lo, hi = section.lo, section.hi
    frame_indices = list(range(lo, hi))
    length = len(frame_indices)
    if length == 0:
        return [], False

    was_padded = False
    if length < action_horizon:
        if not pad_to_h:
            return [], False  # policy sections shorter than H are dropped
        frame_indices = frame_indices + [frame_indices[-1]] * (action_horizon - length)
        was_padded = True

    obs_arrays = episode["obs"]
    next_obs_arrays = episode["next_obs"]
    actions = np.asarray(episode[action_key], dtype=np.float32)
    success_arr = np.asarray(episode["success"], dtype=np.bool_)
    grasp = episode.get("grasp_penalty")

    n = len(frame_indices)
    out: list[Transition] = []
    for step, src_i in enumerate(frame_indices):
        info: dict[str, Any] = {
            "episode_index": int(episode_index),
            "episode_step": int(step),
            "route": str(route),
            "buffer_role": "offline",
            "episode_namespace": f"offline_{route}",
            "source_frame_index": int(src_i),
        }
        if source_episode_index is not None:
            info["source_episode_index"] = int(source_episode_index)
        if mark_success:
            info["success"] = bool(success_arr[src_i])
        grasp_penalty = None
        if grasp is not None:
            gp = float(np.asarray(grasp)[src_i])
            grasp_penalty = None if np.isnan(gp) else gp
        out.append(
            Transition(
                obs=_frame_obs(obs_arrays, src_i, camera_names),
                action=np.asarray(actions[src_i], dtype=np.float32).copy(),
                reward=float(reward_value),
                next_obs=_frame_obs(next_obs_arrays, src_i, camera_names),
                done=bool(step == n - 1),
                grasp_penalty=grasp_penalty,
                is_intervention=(route == ROUTE_POS_ONLY),
                info=info,
                reward_source="offline_section",
                demo_source=f"offline_{route}",
            )
        )
    return out, was_padded


def build_offline_transitions(
    payload: dict[str, Any],
    *,
    action_horizon: int,
    reward_success: float = 0.0,
    reward_fail: float = -1.0,
    include_policy_action_neg: bool = True,
) -> OfflineStreams:
    """Split collected episodes on ``is_intervention`` and route into 3 streams.

    See the module docstring for the routing / reward / padding rules.
    """
    episodes = list(payload.get("episodes", []))
    camera_names = _camera_names_from_payload(payload, episodes)
    H = int(action_horizon)

    streams = OfflineStreams()
    ep_counter = 0
    stats: dict[str, Any] = {
        "num_episodes": len(episodes),
        "policy_sections_total": 0,
        "policy_sections_kept": 0,
        "policy_sections_dropped": 0,
        "policy_sections_dropped_short": 0,
        "human_sections_total": 0,
        "policy_bc_transitions": 0,
        "human_pos_transitions": 0,
        "neg_transitions": 0,
        "padded_human_sections": 0,
        "drop_reasons": {},
        "keep_reasons": {},
    }

    for source_episode_index, episode in enumerate(episodes):
        is_intervention = np.asarray(episode["is_intervention"], dtype=np.bool_)
        terminal_reason = str(episode.get("terminal_reason", ""))
        sections = _split_sections(is_intervention)

        for idx, section in enumerate(sections):
            if section.kind == "policy":
                stats["policy_sections_total"] += 1
                keep, reason_tag = _keep_policy_section(sections, idx, terminal_reason)
                if not keep:
                    stats["policy_sections_dropped"] += 1
                    stats["drop_reasons"][reason_tag] = stats["drop_reasons"].get(reason_tag, 0) + 1
                    continue
                is_success = reason_tag == "success"
                reward_value = reward_success if is_success else reward_fail
                transitions, _ = _materialize_section(
                    episode=episode,
                    section=section,
                    camera_names=camera_names,
                    action_key="executed_action",
                    route=ROUTE_ADVANTAGE,
                    reward_value=reward_value,
                    episode_index=ep_counter,
                    action_horizon=H,
                    pad_to_h=False,
                    mark_success=is_success,
                    source_episode_index=source_episode_index,
                )
                if not transitions:
                    # Kept by policy but too short for a single H window -> dropped.
                    stats["policy_sections_dropped_short"] += 1
                    stats["drop_reasons"]["short_lt_H"] = stats["drop_reasons"].get("short_lt_H", 0) + 1
                    continue
                stats["policy_sections_kept"] += 1
                stats["keep_reasons"][reason_tag] = stats["keep_reasons"].get(reason_tag, 0) + 1
                streams.policy_bc.extend(transitions)
                stats["policy_bc_transitions"] += len(transitions)
                ep_counter += 1
            else:  # human section
                stats["human_sections_total"] += 1
                pos_tr, padded = _materialize_section(
                    episode=episode,
                    section=section,
                    camera_names=camera_names,
                    action_key="executed_action",
                    route=ROUTE_POS_ONLY,
                    reward_value=0.0,
                    episode_index=ep_counter,
                    action_horizon=H,
                    pad_to_h=True,
                    mark_success=False,
                    source_episode_index=source_episode_index,
                )
                if pos_tr:
                    streams.human_pos.extend(pos_tr)
                    stats["human_pos_transitions"] += len(pos_tr)
                    if padded:
                        stats["padded_human_sections"] += 1
                    ep_counter += 1
                if include_policy_action_neg:
                    neg_tr, _ = _materialize_section(
                        episode=episode,
                        section=section,
                        camera_names=camera_names,
                        action_key="policy_action",
                        route=ROUTE_NEG_ONLY,
                        reward_value=0.0,
                        episode_index=ep_counter,
                        action_horizon=H,
                        pad_to_h=True,
                        mark_success=False,
                        source_episode_index=source_episode_index,
                    )
                    if neg_tr:
                        streams.neg.extend(neg_tr)
                        stats["neg_transitions"] += len(neg_tr)
                        ep_counter += 1

    streams.stats = stats
    logger.info(
        "[offline] episode split: %d episodes -> policy_bc=%d (kept %d/%d sections, "
        "dropped %d + %d short), human_pos=%d, neg=%d (padded human sections=%d)",
        stats["num_episodes"],
        stats["policy_bc_transitions"],
        stats["policy_sections_kept"],
        stats["policy_sections_total"],
        stats["policy_sections_dropped"],
        stats["policy_sections_dropped_short"],
        stats["human_pos_transitions"],
        stats["neg_transitions"],
        stats["padded_human_sections"],
    )
    return streams


def build_online_success_transitions(
    payload: dict[str, Any],
    *,
    action_horizon: int,
    episode_index_base: int = 0,
    route: str = ROUTE_ADVANTAGE,
) -> tuple[list[Transition], int, dict[str, Any]]:
    """Extract entire policy-success episodes that had **zero** human intervention.

    Offline DAgger's ``USE_ONLINE_SUCCESS`` positive stream: an episode qualifies
    only when ``terminal_reason == "success"`` **and** ``is_intervention`` is False
    on every frame (a pure on-policy success rollout — no SpaceMouse corrections
    anywhere). Each qualifying episode becomes one standalone-episode list of
    positive BC transitions (``action = executed_action`` which, with no
    intervention, equals ``policy_action``; ``route = advantage`` by default),
    so its chunk windows never cross into other data. Callers can pass
    ``route=pos_only`` when the same pure-success rollouts should force
    ``w_pos=1, w_neg=0`` in routed DIPOLE policy training.

    Episodes shorter than ``action_horizon`` are dropped (cannot form a window;
    no padding — these are full rollouts, not stubs). Returns
    ``(transitions, next_episode_index_base, stats)``.
    """
    episodes = list(payload.get("episodes", []))
    camera_names = _camera_names_from_payload(payload, episodes)
    H = int(action_horizon)

    out: list[Transition] = []
    next_index = int(episode_index_base)
    stats: dict[str, Any] = {
        "success_episodes_total": 0,
        "pure_success_episodes_kept": 0,
        "dropped_has_intervention": 0,
        "dropped_short_lt_H": 0,
        "online_success_transitions": 0,
        "pure_success_source_episode_indices": [],
    }
    for source_episode_index, episode in enumerate(episodes):
        if str(episode.get("terminal_reason", "")) != "success":
            continue
        stats["success_episodes_total"] += 1
        is_intervention = np.asarray(episode["is_intervention"], dtype=np.bool_)
        if bool(is_intervention.any()):
            stats["dropped_has_intervention"] += 1
            continue
        section = Section("policy", 0, int(is_intervention.shape[0]))
        transitions, _ = _materialize_section(
            episode=episode,
            section=section,
            camera_names=camera_names,
            action_key="executed_action",
            route=route,
            reward_value=0.0,
            episode_index=next_index,
            action_horizon=H,
            pad_to_h=False,
            mark_success=False,
            source_episode_index=source_episode_index,
        )
        if not transitions:
            stats["dropped_short_lt_H"] += 1
            continue
        out.extend(transitions)
        stats["online_success_transitions"] += len(transitions)
        stats["pure_success_episodes_kept"] += 1
        stats["pure_success_source_episode_indices"].append(int(source_episode_index))
        next_index += 1

    logger.info(
        "[offline] online-success: %d/%d success episodes are pure on-policy "
        "(no intervention) -> %d transitions (dropped %d intervened, %d short)",
        stats["pure_success_episodes_kept"],
        stats["success_episodes_total"],
        stats["online_success_transitions"],
        stats["dropped_has_intervention"],
        stats["dropped_short_lt_H"],
    )
    return out, next_index, stats


__all__ = [
    "OfflineStreams",
    "Section",
    "build_offline_transitions",
    "build_online_success_transitions",
    "ROUTE_ADVANTAGE",
    "ROUTE_POS_ONLY",
    "ROUTE_NEG_ONLY",
]
