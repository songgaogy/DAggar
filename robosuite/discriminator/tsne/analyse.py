from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import h5py
import hydra
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from hydra.utils import to_absolute_path
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from omegaconf import DictConfig
from sklearn.manifold import TSNE

matplotlib.use("Agg")

from robosuite.discriminator.dyn_bce.task_registry import (
    normalize_task_name,
    ordered_task_names,
    resolve_checkpoint_task_name,
)
from robosuite.discriminator.lpb_score.app.pipeline import build_flow_encoder
from robosuite.discriminator.lpb_score.core.dataset import (
    DemoRef,
    load_latent_trajectories,
    prepare_cached_trajectories,
)


@dataclass(frozen=True)
class SuboptimalDemoRef:
    task_name: str
    file_path: str
    demo_key: str
    sub_start: int
    sub_stop: int


@dataclass(frozen=True)
class TrajectorySequence:
    latents: np.ndarray
    failure_labels: np.ndarray
    task_name: str
    task_index: int
    source_name: str
    split: str
    file_path: str
    demo_key: str
    rollout_id: str
    ood_start: int | None = None
    ood_stop: int | None = None


EXPERT_COLOR = "#B9D6F2"
SUBOPTIMAL_COLOR = "#2563EB"
OOD_COLOR = "#D94841"
BACKGROUND_FACE_COLOR = "#FAFBFD"
GRID_COLOR = "#DCE3EC"
ANNOTATION_TEXT = "OOD segment fades red -> blue from sub_start to sub_stop."


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"Unsupported JSON type: {type(value)!r}")


def _list_hdf5_demos(data_dir: str) -> list[tuple[str, str]]:
    refs: list[tuple[str, str]] = []
    if not os.path.isdir(data_dir):
        return refs
    for file_path in sorted(glob.glob(os.path.join(data_dir, "*.hdf5"))):
        with h5py.File(file_path, "r") as file_handle:
            if "demos" not in file_handle:
                continue
            for demo_key in sorted(file_handle["demos"].keys()):
                refs.append((file_path, demo_key))
    return refs


def _sample_items(items: list[Any], max_items: int, seed: int) -> list[Any]:
    if int(max_items) <= 0 or len(items) <= int(max_items):
        return list(items)
    rng = np.random.default_rng(int(seed))
    indices = np.arange(len(items))
    rng.shuffle(indices)
    chosen = np.sort(indices[: int(max_items)])
    return [items[int(idx)] for idx in chosen.tolist()]


def _build_rollout_id(task_name: str, source_name: str, file_path: str, demo_key: str) -> str:
    stem = os.path.basename(file_path).replace(".hdf5", "")
    return f"{task_name}|{source_name}|{stem}|{demo_key}"


def _blend_hex(start_hex: str, end_hex: str, weight: float) -> str:
    start_rgb = np.asarray(mcolors.to_rgb(start_hex), dtype=np.float32)
    end_rgb = np.asarray(mcolors.to_rgb(end_hex), dtype=np.float32)
    alpha = float(np.clip(weight, 0.0, 1.0))
    blended = (1.0 - alpha) * start_rgb + alpha * end_rgb
    return str(mcolors.to_hex(blended))


def _sample_suboptimal_point_styles(
    sample_steps: np.ndarray,
    *,
    ood_start: int,
    ood_stop: int,
) -> tuple[list[str], list[str], np.ndarray]:
    groups = ["suboptimal_normal"] * int(sample_steps.shape[0])
    colors = [SUBOPTIMAL_COLOR] * int(sample_steps.shape[0])
    progress = np.full((int(sample_steps.shape[0]),), np.nan, dtype=np.float32)

    ood_mask = (sample_steps >= int(ood_start)) & (sample_steps < int(ood_stop))
    ood_indices = np.flatnonzero(ood_mask)
    if ood_indices.size == 0:
        return groups, colors, progress

    denom = max(1, int(ood_indices.size - 1))
    for local_rank, frame_index in enumerate(ood_indices.tolist()):
        blend_weight = float(local_rank) / float(denom)
        groups[int(frame_index)] = "suboptimal_ood"
        colors[int(frame_index)] = _blend_hex(OOD_COLOR, SUBOPTIMAL_COLOR, blend_weight)
        progress[int(frame_index)] = float(blend_weight)
    return groups, colors, progress


def _fallback_cache_roots(cache_root: str) -> list[str]:
    fallback_roots: list[str] = []
    legacy_cache_root = to_absolute_path("./data/.lpb_new_cache")
    if os.path.isdir(legacy_cache_root) and os.path.abspath(cache_root) != os.path.abspath(legacy_cache_root):
        fallback_roots.append(legacy_cache_root)
    return fallback_roots


def _list_standard_demo_refs(cfg: DictConfig) -> tuple[list[DemoRef], dict[str, dict[str, dict[str, int]]], list[str]]:
    task_names = ordered_task_names(list(cfg.data.tasks))
    source_names = [str(name) for name in list(cfg.data.source_data_types)]
    root_dir = to_absolute_path(str(cfg.data.root_dir))
    max_rollouts = int(cfg.data.max_rollouts_per_task_source)
    sample_seed = int(cfg.data.sample_seed)

    refs: list[DemoRef] = []
    summary: dict[str, dict[str, dict[str, int]]] = {}
    for task_offset, task_name in enumerate(task_names):
        summary[task_name] = {}
        for source_offset, source_name in enumerate(source_names):
            data_dir = os.path.join(root_dir, task_name, source_name)
            raw_refs = [
                DemoRef(
                    task_name=task_name,
                    data_type=source_name,
                    split="analysis",
                    file_path=file_path,
                    demo_key=demo_key,
                )
                for file_path, demo_key in _list_hdf5_demos(data_dir)
            ]
            sampled_refs = _sample_items(
                raw_refs,
                max_items=max_rollouts,
                seed=sample_seed + task_offset * 131 + source_offset * 17,
            )
            refs.extend(sampled_refs)
            summary[task_name][source_name] = {
                "available": int(len(raw_refs)),
                "selected": int(len(sampled_refs)),
            }
    return refs, summary, task_names


def _labels_from_source_name(source_name: str, length: int) -> np.ndarray:
    labels = np.zeros((int(length),), dtype=np.int64)
    if str(source_name) == "fail_rollout":
        labels.fill(1)
    return labels


def _encode_standard_sequences(
    *,
    refs: list[DemoRef],
    encoder,
    task_to_index: dict[str, int],
    cfg: DictConfig,
) -> tuple[list[TrajectorySequence], dict[str, Any]]:
    if not refs:
        return [], {"num_requested": 0, "num_encoded": 0, "num_sequences": 0}

    cache_root = to_absolute_path(str(cfg.data.cache_dir))
    cached_refs = prepare_cached_trajectories(
        refs=refs,
        encoder=encoder,
        cache_root=cache_root,
        task_to_index=task_to_index,
        encode_demo_batch_size=int(cfg.data.encode_demo_batch_size),
        fallback_cache_roots=_fallback_cache_roots(cache_root),
        build_missing_cache=bool(cfg.data.build_missing_cache),
    )
    trajectories = load_latent_trajectories(cached_refs)

    sequences: list[TrajectorySequence] = []
    for trajectory in trajectories:
        length = min(int(trajectory.latents.shape[0]), int(trajectory.actions.shape[0]))
        if length <= 0:
            continue
        source_name = str(trajectory.data_type)
        sequences.append(
            TrajectorySequence(
                latents=np.asarray(trajectory.latents[:length], dtype=np.float32),
                failure_labels=_labels_from_source_name(source_name=source_name, length=length),
                task_name=str(trajectory.task_name),
                task_index=int(trajectory.task_index),
                source_name=source_name,
                split=str(trajectory.split),
                file_path=str(trajectory.file_path),
                demo_key=str(trajectory.demo_key),
                rollout_id=_build_rollout_id(
                    task_name=str(trajectory.task_name),
                    source_name=source_name,
                    file_path=str(trajectory.file_path),
                    demo_key=str(trajectory.demo_key),
                ),
                ood_start=None,
                ood_stop=None,
            )
        )

    return (
        sequences,
        {
            "num_requested": int(len(refs)),
            "num_cached_refs": int(len(cached_refs)),
            "num_sequences": int(len(sequences)),
        },
    )


def _list_suboptimal_demo_refs(cfg: DictConfig) -> tuple[list[SuboptimalDemoRef], dict[str, dict[str, int]]]:
    if not bool(cfg.suboptimal.enabled):
        return [], {}

    root_dir = to_absolute_path(str(cfg.suboptimal.root_dir))
    task_names = ordered_task_names(list(cfg.suboptimal.tasks))
    max_rollouts = int(cfg.suboptimal.max_rollouts_per_task)
    sample_seed = int(cfg.suboptimal.sample_seed)

    refs: list[SuboptimalDemoRef] = []
    summary: dict[str, dict[str, int]] = {}
    for task_offset, task_name in enumerate(task_names):
        checkpoint_name = resolve_checkpoint_task_name(task_name)
        sub_dir = os.path.join(root_dir, f"{checkpoint_name}_allview")
        if not os.path.isdir(sub_dir):
            raise FileNotFoundError(f"Missing suboptimal directory for {task_name}: {sub_dir}")

        task_refs: list[SuboptimalDemoRef] = []
        for file_path in sorted(glob.glob(os.path.join(sub_dir, "*.hdf5"))):
            with h5py.File(file_path, "r") as file_handle:
                if "demos" not in file_handle:
                    continue
                for demo_key in sorted(file_handle["demos"].keys()):
                    demo = file_handle["demos"][demo_key]
                    task_refs.append(
                        SuboptimalDemoRef(
                            task_name=task_name,
                            file_path=file_path,
                            demo_key=demo_key,
                            sub_start=int(np.asarray(demo["sub_start"])[()]),
                            sub_stop=int(np.asarray(demo["sub_stop"])[()]),
                        )
                    )
        sampled_refs = _sample_items(
            task_refs,
            max_items=max_rollouts,
            seed=sample_seed + task_offset * 97,
        )
        refs.extend(sampled_refs)
        summary[task_name] = {
            "available": int(len(task_refs)),
            "selected": int(len(sampled_refs)),
        }
    return refs, summary


def _encode_suboptimal_sequences(
    *,
    refs: list[SuboptimalDemoRef],
    encoder,
    task_to_index: dict[str, int],
    cfg: DictConfig,
) -> tuple[list[TrajectorySequence], dict[str, Any]]:
    if not refs:
        return [], {"num_requested": 0, "num_sequences": 0, "num_dropped": 0}

    cache_root = to_absolute_path(str(cfg.data.cache_dir))
    sequences: list[TrajectorySequence] = []
    dropped: list[dict[str, Any]] = []
    for idx, ref in enumerate(refs):
        encoded = encoder.load_or_encode_demo(
            cache_root=cache_root,
            task_name=ref.task_name,
            file_path=ref.file_path,
            demo_key=ref.demo_key,
        )
        length = min(int(encoded.latents.shape[0]), int(encoded.actions.shape[0]))
        if length <= 0:
            dropped.append(
                {
                    "task_name": ref.task_name,
                    "file_path": ref.file_path,
                    "demo_key": ref.demo_key,
                    "reason": "empty_sequence",
                }
            )
            continue

        start = int(np.clip(ref.sub_start, 0, length))
        stop = int(np.clip(ref.sub_stop, 0, length))
        if stop <= start:
            dropped.append(
                {
                    "task_name": ref.task_name,
                    "file_path": ref.file_path,
                    "demo_key": ref.demo_key,
                    "reason": "empty_positive_interval_after_clip",
                    "sub_start": int(ref.sub_start),
                    "sub_stop": int(ref.sub_stop),
                    "length": int(length),
                }
            )
            continue

        failure_labels = np.zeros((length,), dtype=np.int64)
        failure_labels[start:stop] = 1
        sequences.append(
            TrajectorySequence(
                latents=np.asarray(encoded.latents[:length], dtype=np.float32),
                failure_labels=failure_labels,
                task_name=str(ref.task_name),
                task_index=int(task_to_index[ref.task_name]),
                source_name="suboptimal",
                split="analysis",
                file_path=str(ref.file_path),
                demo_key=str(ref.demo_key),
                rollout_id=_build_rollout_id(
                    task_name=str(ref.task_name),
                    source_name="suboptimal",
                    file_path=str(ref.file_path),
                    demo_key=str(ref.demo_key),
                ),
                ood_start=int(start),
                ood_stop=int(stop),
            )
        )
        if (idx + 1) % 20 == 0 or (idx + 1) == len(refs):
            print(f"[tsne] encoded_suboptimal {idx + 1}/{len(refs)}")

    return (
        sequences,
        {
            "num_requested": int(len(refs)),
            "num_sequences": int(len(sequences)),
            "num_dropped": int(len(dropped)),
            "dropped": dropped,
        },
    )


def _sampled_length(sequence: TrajectorySequence, timestep_stride: int) -> int:
    return int(np.arange(0, int(sequence.latents.shape[0]), max(1, int(timestep_stride))).shape[0])


def _limit_sequences_by_total_points(
    sequences: list[TrajectorySequence],
    *,
    timestep_stride: int,
    max_total_points: int,
    seed: int,
) -> tuple[list[TrajectorySequence], dict[str, int]]:
    point_counts = [_sampled_length(sequence, timestep_stride) for sequence in sequences]
    input_points = int(sum(point_counts))
    if int(max_total_points) <= 0 or input_points <= int(max_total_points):
        return (
            list(sequences),
            {
                "num_input_sequences": int(len(sequences)),
                "num_output_sequences": int(len(sequences)),
                "input_points": input_points,
                "output_points": input_points,
                "max_total_points": int(max_total_points),
            },
        )

    protected_indices: list[int] = []
    protected_pairs: list[tuple[str, str]] = []
    for task_name in ordered_task_names([sequence.task_name for sequence in sequences]):
        protected_pairs.append((str(task_name), "expert"))
        protected_pairs.append((str(task_name), "suboptimal"))

    fallback_sources = ("success_rollout", "fail_rollout")
    for task_name, source_name in protected_pairs:
        protected_idx = next(
            (
                idx
                for idx, sequence in enumerate(sequences)
                if sequence.task_name == task_name and sequence.source_name == source_name
            ),
            None,
        )
        if protected_idx is not None and protected_idx not in protected_indices:
            protected_indices.append(int(protected_idx))
            continue

        fallback_idx = next(
            (
                idx
                for idx, sequence in enumerate(sequences)
                if sequence.task_name == task_name and sequence.source_name in fallback_sources
            ),
            None,
        )
        if fallback_idx is not None and fallback_idx not in protected_indices:
            protected_indices.append(int(fallback_idx))

    selected_indices = list(protected_indices)
    selected_points = int(sum(point_counts[idx] for idx in selected_indices))

    rng = np.random.default_rng(int(seed))
    remaining_indices = [idx for idx in range(len(sequences)) if idx not in set(protected_indices)]
    rng.shuffle(remaining_indices)
    for idx in remaining_indices:
        next_count = int(point_counts[idx])
        if selected_indices and selected_points + next_count > int(max_total_points):
            continue
        selected_indices.append(int(idx))
        selected_points += next_count

    if not selected_indices and sequences:
        selected_indices.append(0)
        selected_points = int(point_counts[0])

    selected_indices = sorted(set(selected_indices))
    limited_sequences = [sequences[idx] for idx in selected_indices]
    return (
        limited_sequences,
        {
            "num_input_sequences": int(len(sequences)),
            "num_output_sequences": int(len(limited_sequences)),
            "input_points": input_points,
            "output_points": int(selected_points),
            "max_total_points": int(max_total_points),
        },
    )


def _build_point_dataframe(
    sequences: list[TrajectorySequence],
    *,
    timestep_stride: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    features: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    stride = max(1, int(timestep_stride))

    for rollout_index, sequence in enumerate(sequences):
        sample_steps = np.arange(0, int(sequence.latents.shape[0]), stride, dtype=np.int64)
        if sample_steps.size == 0:
            continue
        features.append(np.asarray(sequence.latents[sample_steps], dtype=np.float32))
        sampled_labels = np.asarray(sequence.failure_labels[sample_steps], dtype=np.int64)
        if sequence.source_name == "expert":
            display_groups = ["expert"] * int(sample_steps.shape[0])
            display_colors = [EXPERT_COLOR] * int(sample_steps.shape[0])
            ood_progress = np.full((int(sample_steps.shape[0]),), np.nan, dtype=np.float32)
        elif sequence.source_name == "suboptimal" and sequence.ood_start is not None and sequence.ood_stop is not None:
            display_groups, display_colors, ood_progress = _sample_suboptimal_point_styles(
                sample_steps,
                ood_start=int(sequence.ood_start),
                ood_stop=int(sequence.ood_stop),
            )
        else:
            display_groups = [str(sequence.source_name)] * int(sample_steps.shape[0])
            display_colors = ["#7A8798"] * int(sample_steps.shape[0])
            ood_progress = np.full((int(sample_steps.shape[0]),), np.nan, dtype=np.float32)

        for local_idx, timestep in enumerate(sample_steps.tolist()):
            failure_label = int(sampled_labels[local_idx])
            records.append(
                {
                    "rollout_index": int(rollout_index),
                    "rollout_id": str(sequence.rollout_id),
                    "task_name": str(sequence.task_name),
                    "task_index": int(sequence.task_index),
                    "timestep": int(timestep),
                    "source_name": str(sequence.source_name),
                    "split": str(sequence.split),
                    "failure_label": failure_label,
                    "success_label": int(1 - failure_label),
                    "outcome_name": "failure" if failure_label == 1 else "success",
                    "file_path": str(sequence.file_path),
                    "demo_key": str(sequence.demo_key),
                    "rollout_length": int(sequence.latents.shape[0]),
                    "display_group": str(display_groups[local_idx]),
                    "display_color": str(display_colors[local_idx]),
                    "ood_progress": float(ood_progress[local_idx])
                    if not np.isnan(ood_progress[local_idx])
                    else np.nan,
                    "ood_start": int(sequence.ood_start) if sequence.ood_start is not None else None,
                    "ood_stop": int(sequence.ood_stop) if sequence.ood_stop is not None else None,
                }
            )

    if not features:
        raise RuntimeError("No latent points remain after sampling.")

    feature_matrix = np.concatenate(features, axis=0).astype(np.float32, copy=False)
    frame = pd.DataFrame.from_records(records)
    return feature_matrix, frame


def _resolve_learning_rate(value: Any) -> Any:
    if isinstance(value, str):
        text = str(value).strip()
        if text.lower() == "auto":
            return "auto"
        return float(text)
    return float(value)


def _run_tsne(features: np.ndarray, cfg: DictConfig) -> tuple[np.ndarray, dict[str, Any]]:
    num_points = int(features.shape[0])
    if num_points < 3:
        raise RuntimeError(f"t-SNE requires at least 3 points, got {num_points}.")

    perplexity_requested = float(cfg.tsne.perplexity)
    perplexity_used = min(perplexity_requested, float(num_points - 1))
    if perplexity_used <= 0:
        perplexity_used = 1.0

    learning_rate = _resolve_learning_rate(cfg.tsne.learning_rate)
    n_jobs = int(cfg.tsne.n_jobs)
    tsne = TSNE(
        n_components=2,
        perplexity=float(perplexity_used),
        early_exaggeration=float(cfg.tsne.early_exaggeration),
        learning_rate=learning_rate,
        max_iter=int(cfg.tsne.max_iter),
        init=str(cfg.tsne.init),
        metric=str(cfg.tsne.metric),
        random_state=int(cfg.seed),
        method=str(cfg.tsne.method),
        angle=float(cfg.tsne.angle),
        n_jobs=None if n_jobs == 0 else int(n_jobs),
        verbose=int(cfg.tsne.verbose),
    )
    embedding = tsne.fit_transform(features).astype(np.float32, copy=False)
    return (
        embedding,
        {
            "perplexity_requested": float(perplexity_requested),
            "perplexity_used": float(perplexity_used),
            "learning_rate": learning_rate,
            "max_iter": int(cfg.tsne.max_iter),
            "init": str(cfg.tsne.init),
            "metric": str(cfg.tsne.metric),
            "method": str(cfg.tsne.method),
            "angle": float(cfg.tsne.angle),
            "n_jobs": None if n_jobs == 0 else int(n_jobs),
        },
    )


def _save_figure(fig: plt.Figure, base_path: str, dpi: int) -> dict[str, str]:
    png_path = f"{base_path}.png"
    pdf_path = f"{base_path}.pdf"
    fig.savefig(png_path, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"png": png_path, "pdf": pdf_path}


def _configure_plot_style() -> None:
    sns.set_theme(
        style="white",
        context="paper",
        font_scale=1.55,
        rc={
            "axes.facecolor": BACKGROUND_FACE_COLOR,
            "figure.facecolor": "white",
            "axes.edgecolor": "#CBD5E1",
            "axes.labelcolor": "#1F2937",
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "grid.color": GRID_COLOR,
            "grid.alpha": 0.55,
            "grid.linewidth": 0.8,
        },
    )


def _style_embedding_axes(ax: plt.Axes) -> None:
    ax.grid(True)
    ax.set_facecolor(BACKGROUND_FACE_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")


def _scatter_expert_background(ax: plt.Axes, frame: pd.DataFrame, cfg: DictConfig) -> None:
    expert_frame = frame[frame["source_name"] == "expert"]
    if expert_frame.empty:
        return
    ax.scatter(
        expert_frame["tsne_x"].to_numpy(dtype=np.float32),
        expert_frame["tsne_y"].to_numpy(dtype=np.float32),
        s=float(cfg.analysis.expert_point_size),
        alpha=float(cfg.analysis.expert_alpha),
        color=EXPERT_COLOR,
        linewidths=0.0,
        rasterized=True,
        zorder=1,
    )


def _plot_suboptimal_rollouts(ax: plt.Axes, frame: pd.DataFrame, cfg: DictConfig) -> None:
    suboptimal_frame = frame[frame["source_name"] == "suboptimal"].copy()
    if suboptimal_frame.empty:
        return

    for _, rollout_frame in suboptimal_frame.groupby("rollout_id", sort=False):
        rollout_frame = rollout_frame.sort_values("timestep")
        points = rollout_frame[["tsne_x", "tsne_y"]].to_numpy(dtype=np.float32)
        colors = rollout_frame["display_color"].astype(str).tolist()
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=float(cfg.analysis.suboptimal_point_size),
            c=colors,
            alpha=float(cfg.analysis.suboptimal_alpha),
            linewidths=0.0,
            rasterized=True,
            zorder=3,
        )

        ood_frame = rollout_frame[rollout_frame["display_group"] == "suboptimal_ood"]
        if not ood_frame.empty:
            onset = ood_frame.iloc[0]
            ax.scatter(
                [float(onset["tsne_x"])],
                [float(onset["tsne_y"])],
                s=float(cfg.analysis.ood_onset_marker_size),
                color=OOD_COLOR,
                edgecolors="white",
                linewidths=0.8,
                zorder=5,
            )


def _failure_pattern_legend(ax: plt.Axes, cfg: DictConfig) -> None:
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=6.5,
            markerfacecolor=EXPERT_COLOR,
            markeredgecolor="none",
            alpha=float(cfg.analysis.expert_alpha),
            label="Expert",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=6.5,
            markerfacecolor=SUBOPTIMAL_COLOR,
            markeredgecolor="none",
            label="Suboptimal normal",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=6.5,
            markerfacecolor=OOD_COLOR,
            markeredgecolor="none",
            label="OOD onset / transition",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        fontsize=float(cfg.analysis.failure_legend_fontsize),
        title="Pattern",
        title_fontsize=float(cfg.analysis.failure_legend_title_fontsize),
    )


def _plot_by_task(frame: pd.DataFrame, cfg: DictConfig, run_dir: str) -> dict[str, str]:
    task_names = ordered_task_names(frame["task_name"].astype(str).unique().tolist())
    palette = dict(zip(task_names, sns.color_palette("tab10", n_colors=len(task_names))))
    fig, ax = plt.subplots(figsize=(9.2, 7.6))
    sns.scatterplot(
        data=frame,
        x="tsne_x",
        y="tsne_y",
        hue="task_name",
        hue_order=task_names,
        palette=palette,
        s=float(cfg.analysis.point_size),
        alpha=float(cfg.analysis.point_alpha),
        linewidth=0.0,
        ax=ax,
    )
    _style_embedding_axes(ax)
    ax.set_title("Policy Latent t-SNE by Task")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(
        title="Task",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=True,
        fontsize=float(cfg.analysis.task_legend_fontsize),
        title_fontsize=float(cfg.analysis.task_legend_title_fontsize),
    )
    return _save_figure(fig, os.path.join(run_dir, "plot_b_task"), dpi=int(cfg.analysis.plot_dpi))


def _plot_failure_pattern(frame: pd.DataFrame, cfg: DictConfig, run_dir: str) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(9.5, 7.7))
    _scatter_expert_background(ax, frame, cfg)
    _plot_suboptimal_rollouts(ax, frame, cfg)
    _style_embedding_axes(ax)
    _failure_pattern_legend(ax, cfg)
    ax.text(
        0.02,
        0.02,
        ANNOTATION_TEXT,
        transform=ax.transAxes,
        fontsize=float(cfg.analysis.annotation_fontsize),
        color="#475569",
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#E2E8F0", "alpha": 0.92},
    )
    ax.set_title("Policy Latent Failure Pattern")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    return _save_figure(fig, os.path.join(run_dir, "failure_pattern"), dpi=int(cfg.analysis.plot_dpi))


def _plot_per_task_failure_patterns(
    frame: pd.DataFrame,
    cfg: DictConfig,
    run_dir: str,
) -> dict[str, dict[str, str]]:
    per_task_dir = os.path.join(run_dir, "per_task")
    os.makedirs(per_task_dir, exist_ok=True)

    outputs: dict[str, dict[str, str]] = {}
    task_names = ordered_task_names(frame["task_name"].astype(str).unique().tolist())
    for task_name in task_names:
        task_frame = frame[frame["task_name"] == task_name].copy()
        if task_frame.empty:
            continue

        task_output_dir = os.path.join(per_task_dir, task_name)
        os.makedirs(task_output_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8.8, 7.2))
        _scatter_expert_background(ax, task_frame, cfg)
        _plot_suboptimal_rollouts(ax, task_frame, cfg)
        _style_embedding_axes(ax)
        _failure_pattern_legend(ax, cfg)
        ax.text(
            0.02,
            0.02,
            ANNOTATION_TEXT,
            transform=ax.transAxes,
            fontsize=float(cfg.analysis.annotation_fontsize),
            color="#475569",
            ha="left",
            va="bottom",
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#E2E8F0", "alpha": 0.92},
        )
        ax.set_title(f"Policy Latent Failure Pattern | {task_name}")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        outputs[task_name] = _save_figure(
            fig,
            os.path.join(task_output_dir, "failure_pattern"),
            dpi=int(cfg.analysis.plot_dpi),
        )
    return outputs


def _summary_counts(frame: pd.DataFrame) -> dict[str, Any]:
    counts = (
        frame.groupby(["task_name", "source_name", "outcome_name"], dropna=False)
        .size()
        .reset_index(name="num_points")
    )
    display_counts = (
        frame.groupby(["task_name", "display_group"], dropna=False)
        .size()
        .reset_index(name="num_points")
    )
    rollouts = (
        frame.groupby(["task_name", "source_name"], dropna=False)["rollout_id"]
        .nunique()
        .reset_index(name="num_rollouts")
    )
    return {
        "points_by_task_source_outcome": counts.to_dict(orient="records"),
        "points_by_task_display_group": display_counts.to_dict(orient="records"),
        "rollouts_by_task_source": rollouts.to_dict(orient="records"),
    }


def run_analyse(cfg: DictConfig) -> None:
    np.random.seed(int(cfg.seed))
    torch.manual_seed(int(cfg.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.seed))

    _configure_plot_style()
    encoder = build_flow_encoder(cfg)
    try:
        standard_refs, standard_ref_summary, task_names = _list_standard_demo_refs(cfg)
        task_to_index = {
            task_name: idx
            for idx, task_name in enumerate(task_names)
        }
        standard_sequences, standard_encode_summary = _encode_standard_sequences(
            refs=standard_refs,
            encoder=encoder,
            task_to_index=task_to_index,
            cfg=cfg,
        )

        suboptimal_refs, suboptimal_ref_summary = _list_suboptimal_demo_refs(cfg)
        suboptimal_sequences, suboptimal_encode_summary = _encode_suboptimal_sequences(
            refs=suboptimal_refs,
            encoder=encoder,
            task_to_index=task_to_index,
            cfg=cfg,
        )

        sequences = list(standard_sequences) + list(suboptimal_sequences)
        if not sequences:
            raise RuntimeError("No trajectories were selected for t-SNE analysis.")

        sequences, point_limit_summary = _limit_sequences_by_total_points(
            sequences,
            timestep_stride=int(cfg.data.timestep_stride),
            max_total_points=int(cfg.analysis.max_total_points),
            seed=int(cfg.seed),
        )
        features, frame = _build_point_dataframe(
            sequences,
            timestep_stride=int(cfg.data.timestep_stride),
        )
        embedding, tsne_summary = _run_tsne(features, cfg)
        frame["tsne_x"] = embedding[:, 0]
        frame["tsne_y"] = embedding[:, 1]

        expert_task_names = set(frame.loc[frame["source_name"] == "expert", "task_name"].astype(str).tolist())
        suboptimal_task_names = set(frame.loc[frame["source_name"] == "suboptimal", "task_name"].astype(str).tolist())
        missing_expert_tasks = ordered_task_names(list(set(task_names) - expert_task_names))
        missing_suboptimal_tasks = ordered_task_names(list(set(task_names) - suboptimal_task_names))
        if missing_expert_tasks:
            raise RuntimeError(
                "Failure-pattern plots require expert points for every task. "
                f"Missing expert data for: {missing_expert_tasks}"
            )
        if missing_suboptimal_tasks:
            raise RuntimeError(
                "Failure-pattern plots require suboptimal points for every task. "
                f"Missing suboptimal data for: {missing_suboptimal_tasks}"
            )

        save_dir = to_absolute_path(str(cfg.save_dir))
        os.makedirs(save_dir, exist_ok=True)
        run_dir = os.path.join(save_dir, f"run_{_now_tag()}")
        os.makedirs(run_dir, exist_ok=True)

        csv_path = os.path.join(run_dir, "tsne_points.csv")
        frame.to_csv(csv_path, index=False)

        npz_path = os.path.join(run_dir, "tsne_features.npz")
        np.savez_compressed(
            npz_path,
            latents=features.astype(np.float32),
            tsne_xy=embedding.astype(np.float32),
            rollout_index=frame["rollout_index"].to_numpy(dtype=np.int64),
            task_index=frame["task_index"].to_numpy(dtype=np.int64),
            timestep=frame["timestep"].to_numpy(dtype=np.int64),
            failure_label=frame["failure_label"].to_numpy(dtype=np.int64),
            success_label=frame["success_label"].to_numpy(dtype=np.int64),
            task_name=frame["task_name"].astype(str).to_numpy(),
            source_name=frame["source_name"].astype(str).to_numpy(),
            split=frame["split"].astype(str).to_numpy(),
            rollout_id=frame["rollout_id"].astype(str).to_numpy(),
            file_path=frame["file_path"].astype(str).to_numpy(),
            demo_key=frame["demo_key"].astype(str).to_numpy(),
            display_group=frame["display_group"].astype(str).to_numpy(),
            display_color=frame["display_color"].astype(str).to_numpy(),
            ood_progress=frame["ood_progress"].to_numpy(dtype=np.float32),
        )

        plot_b_paths = _plot_by_task(frame, cfg, run_dir)
        failure_pattern_paths = _plot_failure_pattern(frame, cfg, run_dir)
        per_task_paths = _plot_per_task_failure_patterns(frame, cfg, run_dir)

        summary = {
            "timestamp": _now_tag(),
            "policy_ckpt": to_absolute_path(str(cfg.policy.ckpt)),
            "save_dir": run_dir,
            "task_names": task_names,
            "feature_dim": int(features.shape[1]),
            "num_sequences": int(len(sequences)),
            "num_points": int(features.shape[0]),
            "data": {
                "root_dir": to_absolute_path(str(cfg.data.root_dir)),
                "cache_dir": to_absolute_path(str(cfg.data.cache_dir)),
                "image_size": int(cfg.data.image_size),
                "source_data_types": [str(name) for name in list(cfg.data.source_data_types)],
                "max_rollouts_per_task_source": int(cfg.data.max_rollouts_per_task_source),
                "timestep_stride": int(cfg.data.timestep_stride),
                "build_missing_cache": bool(cfg.data.build_missing_cache),
            },
            "suboptimal": {
                "enabled": bool(cfg.suboptimal.enabled),
                "root_dir": to_absolute_path(str(cfg.suboptimal.root_dir)),
                "tasks": [normalize_task_name(str(name)) for name in list(cfg.suboptimal.tasks)],
                "max_rollouts_per_task": int(cfg.suboptimal.max_rollouts_per_task),
                "ref_summary": suboptimal_ref_summary,
                "encode_summary": suboptimal_encode_summary,
            },
            "standard": {
                "ref_summary": standard_ref_summary,
                "encode_summary": standard_encode_summary,
            },
            "sampling": point_limit_summary,
            "tsne": tsne_summary,
            "counts": _summary_counts(frame),
            "outputs": {
                "csv_path": csv_path,
                "npz_path": npz_path,
                "plot_b_task": plot_b_paths,
                "failure_pattern": failure_pattern_paths,
                "per_task": per_task_paths,
            },
        }
        summary_path = os.path.join(run_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as file_handle:
            json.dump(summary, file_handle, indent=2, ensure_ascii=False, default=_json_default)

        print(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default))
        print(f"[tsne] Saved analysis summary to: {summary_path}")
    finally:
        encoder.close()


@hydra.main(version_base="1.2", config_path="./config", config_name="analyse")
def main(cfg: DictConfig) -> None:
    run_analyse(cfg)


if __name__ == "__main__":
    main()
