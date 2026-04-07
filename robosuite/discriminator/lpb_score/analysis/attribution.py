from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from robosuite.discriminator.utils.types import TrajectoryDetectionResult
from robosuite.discriminator.utils.visualization import map_step_values_to_frames


TERM_ORDER: tuple[str, ...] = (
    "state_error",
    "action_error",
    "next_state_error",
)

TERM_LABELS: dict[str, str] = {
    "state_error": "State Error",
    "action_error": "Action Error",
    "next_state_error": "Next-State Error",
}

TERM_SHORT_LABELS: dict[str, str] = {
    "state_error": "ST",
    "action_error": "AC",
    "next_state_error": "NS",
}

TERM_COLORS: dict[str, str] = {
    "state_error": "#1f77b4",
    "action_error": "#ff7f0e",
    "next_state_error": "#2ca02c",
}


def ordered_term_keys(values: Mapping[str, np.ndarray] | None) -> list[str]:
    """Keep plots and summaries in a stable term order."""
    if not values:
        return []
    ordered = [key for key in TERM_ORDER if key in values]
    for key in values.keys():
        if key not in ordered:
            ordered.append(str(key))
    return ordered


def map_term_values_to_frames(
    values: Mapping[str, np.ndarray] | None,
    *,
    num_frames: int,
) -> dict[str, np.ndarray]:
    """Broadcast step-wise term scores to frame-aligned curves."""
    if not values:
        return {}
    out: dict[str, np.ndarray] = {}
    for key in ordered_term_keys(values):
        arr = np.asarray(values[key], dtype=np.float32)
        if arr.size == 0:
            continue
        out[key] = map_step_values_to_frames(
            arr,
            num_frames=int(num_frames),
            tail_fill=float(arr[-1]),
        ).astype(np.float32)
    return out


def summarize_trajectory_term_attribution(
    result: TrajectoryDetectionResult,
) -> dict[str, object]:
    """Summarize which term dominates a single detected trajectory."""
    aggregate_shares = result.metadata.get("aggregate_contribution_shares", {}) or {}
    dominant_terms = [str(term) for term in result.metadata.get("dominant_aggregate_terms", []) or []]
    keys = ordered_term_keys(aggregate_shares)
    dominant_counts = {key: 0 for key in keys}
    for term in dominant_terms:
        if term in dominant_counts:
            dominant_counts[term] += 1

    mean_aggregate_share = {
        key: float(np.mean(np.asarray(aggregate_shares[key], dtype=np.float32)))
        for key in keys
    }

    labels = None if result.labels is None else np.asarray(result.labels, dtype=np.int64).reshape(-1)
    mean_fail_region_share: dict[str, float] = {}
    if labels is not None and labels.size > 0 and int(np.sum(labels == 1)) > 0:
        fail_mask = labels == 1
        for key in keys:
            arr = np.asarray(aggregate_shares[key], dtype=np.float32).reshape(-1)
            if arr.shape[0] == labels.shape[0]:
                mean_fail_region_share[key] = float(np.mean(arr[fail_mask]))

    first_crossing_index = result.metadata.get("first_crossing_index", None)
    if first_crossing_index is not None:
        first_crossing_index = int(first_crossing_index)

    return {
        "mean_aggregate_share": mean_aggregate_share,
        "mean_fail_region_share": mean_fail_region_share,
        "dominant_term_counts": dominant_counts,
        "dominant_term_ratio": {
            key: float(count / len(dominant_terms)) if dominant_terms else 0.0
            for key, count in dominant_counts.items()
        },
        "first_crossing_step": int(first_crossing_index + 1) if first_crossing_index is not None else None,
        "first_crossing_dominant_term": result.metadata.get("first_crossing_dominant_term", None),
        "first_crossing_term_shares": {
            key: float(value)
            for key, value in (result.metadata.get("first_crossing_term_shares", {}) or {}).items()
        },
    }


def summarize_result_set_term_attribution(
    results: Sequence[TrajectoryDetectionResult],
    *,
    positive_region_only: bool,
) -> dict[str, object]:
    """Summarize dominant terms over a dataset split."""
    term_keys = ordered_term_keys(
        {
            key: value
            for result in results
            for key, value in (result.metadata.get("aggregate_contribution_shares", {}) or {}).items()
        }
    )
    dominant_counts = {key: 0 for key in term_keys}
    first_crossing_counts = {key: 0 for key in term_keys}
    mean_share_values: dict[str, list[float]] = {key: [] for key in term_keys}

    num_prefixes = 0
    num_first_crossings = 0
    for result in results:
        shares = result.metadata.get("aggregate_contribution_shares", {}) or {}
        dominant_terms = [str(term) for term in result.metadata.get("dominant_aggregate_terms", []) or []]
        if not shares or not dominant_terms:
            continue

        mask = np.ones((len(dominant_terms),), dtype=bool)
        labels = None if result.labels is None else np.asarray(result.labels, dtype=np.int64).reshape(-1)
        if positive_region_only:
            if labels is None or labels.shape[0] != mask.shape[0]:
                continue
            mask = labels == 1
            if not bool(mask.any()):
                continue

        num_prefixes += int(np.sum(mask))
        for idx, term in enumerate(dominant_terms):
            if bool(mask[idx]) and term in dominant_counts:
                dominant_counts[term] += 1

        for key in term_keys:
            arr = np.asarray(shares.get(key, []), dtype=np.float32).reshape(-1)
            if arr.shape[0] == mask.shape[0] and bool(mask.any()):
                mean_share_values[key].append(float(np.mean(arr[mask])))

        crossing_idx = result.metadata.get("first_crossing_index", None)
        crossing_term = result.metadata.get("first_crossing_dominant_term", None)
        if crossing_idx is not None and crossing_term in first_crossing_counts:
            crossing_idx = int(crossing_idx)
            if 0 <= crossing_idx < mask.shape[0] and bool(mask[crossing_idx]):
                first_crossing_counts[str(crossing_term)] += 1
                num_first_crossings += 1

    return {
        "num_trajectories": int(len(results)),
        "num_prefixes": int(num_prefixes),
        "dominant_term_counts": dominant_counts,
        "dominant_term_ratio": {
            key: (float(dominant_counts[key]) / float(num_prefixes)) if num_prefixes > 0 else 0.0
            for key in term_keys
        },
        "mean_aggregate_share": {
            key: float(np.mean(values)) if values else 0.0
            for key, values in mean_share_values.items()
        },
        "first_crossing_dominant_counts": first_crossing_counts,
        "first_crossing_dominant_ratio": {
            key: (float(first_crossing_counts[key]) / float(num_first_crossings))
            if num_first_crossings > 0
            else 0.0
            for key in term_keys
        },
    }
