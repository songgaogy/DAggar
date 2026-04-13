from __future__ import annotations

import os

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from omegaconf import DictConfig

matplotlib.use("Agg")

from robosuite.discriminator.dyn_bce.task_registry import ordered_task_names
from robosuite.discriminator.tsne.embedding import (
    _embedding_axis_labels,
    _embedding_display_name,
    _embedding_method,
)
from robosuite.discriminator.tsne.schemas import (
    ANNOTATED_FAILURE_TEXT,
    ANNOTATION_TEXT,
    BACKGROUND_FACE_COLOR,
    EXPERT_COLOR,
    FAILURE_END_COLOR,
    FAILURE_NORMAL_COLOR,
    FAILURE_START_COLOR,
    GRID_COLOR,
    OOD_COLOR,
    SUBOPTIMAL_COLOR,
    SUCCESS_COLOR,
    _blend_hex,
)


def _safe_path_component(name: str) -> str:
    return str(name).replace("/", "_").replace(" ", "_")


def _save_figure(fig: plt.Figure, base_path: str, dpi: int) -> dict[str, str]:
    png_path = f"{base_path}.png"
    pdf_path = f"{base_path}.pdf"
    fig.savefig(png_path, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"png": png_path, "pdf": pdf_path}


def _configure_plot_style() -> None:
    sns.set_theme(
        style="ticks",
        context="paper",
        font_scale=1.42,
        rc={
            "font.family": "DejaVu Serif",
            "axes.facecolor": BACKGROUND_FACE_COLOR,
            "figure.facecolor": "white",
            "axes.edgecolor": "#CBD5E1",
            "axes.labelcolor": "#1F2937",
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "grid.color": GRID_COLOR,
            "grid.alpha": 0.48,
            "grid.linewidth": 0.75,
            "axes.axisbelow": True,
            "legend.edgecolor": "#D7DFEA",
            "legend.framealpha": 0.96,
        },
    )


def _style_embedding_axes(ax: plt.Axes) -> None:
    ax.grid(True)
    ax.set_facecolor(BACKGROUND_FACE_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.tick_params(length=3.0, width=0.8)


def _scatter_expert_background(ax: plt.Axes, frame: pd.DataFrame, cfg: DictConfig) -> None:
    expert_frame = frame[frame["source_name"] == "expert"]
    if expert_frame.empty:
        return
    ax.scatter(
        expert_frame["embedding_x"].to_numpy(dtype=np.float32),
        expert_frame["embedding_y"].to_numpy(dtype=np.float32),
        s=float(cfg.analysis.expert_point_size),
        alpha=float(cfg.analysis.expert_alpha),
        color=EXPERT_COLOR,
        linewidths=0.0,
        rasterized=True,
        zorder=1,
    )


def _scatter_success_background(ax: plt.Axes, frame: pd.DataFrame, cfg: DictConfig) -> None:
    success_frame = frame[frame["source_name"] == "success_rollout"]
    if success_frame.empty:
        return
    ax.scatter(
        success_frame["embedding_x"].to_numpy(dtype=np.float32),
        success_frame["embedding_y"].to_numpy(dtype=np.float32),
        s=float(cfg.analysis.success_point_size),
        alpha=float(cfg.analysis.success_alpha),
        color=SUCCESS_COLOR,
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
        points = rollout_frame[["embedding_x", "embedding_y"]].to_numpy(dtype=np.float32)
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
                [float(onset["embedding_x"])],
                [float(onset["embedding_y"])],
                s=float(cfg.analysis.ood_onset_marker_size),
                color=OOD_COLOR,
                edgecolors="white",
                linewidths=0.8,
                zorder=5,
            )


def _rollout_failure_modes(rollout_frame: pd.DataFrame) -> set[str]:
    modes = {
        str(mode)
        for mode in rollout_frame["failure_mode"].astype(str).tolist()
        if str(mode).strip()
    }
    return modes


def _annotated_failure_colors_for_mode(
    rollout_frame: pd.DataFrame,
    *,
    selected_mode: str | None,
) -> list[str]:
    colors: list[str] = []
    for _, row in rollout_frame.iterrows():
        failure_label = int(row["failure_label"])
        failure_mode = str(row["failure_mode"]).strip()
        progress = float(row["ood_progress"]) if not pd.isna(row["ood_progress"]) else np.nan
        if failure_label == 0:
            colors.append(FAILURE_NORMAL_COLOR)
            continue
        if selected_mode is None or failure_mode == str(selected_mode):
            if np.isnan(progress):
                colors.append(FAILURE_START_COLOR)
            else:
                colors.append(_blend_hex(FAILURE_START_COLOR, FAILURE_END_COLOR, progress))
        else:
            colors.append(FAILURE_NORMAL_COLOR)
    return colors


def _plot_annotated_failure_rollouts(
    ax: plt.Axes,
    frame: pd.DataFrame,
    cfg: DictConfig,
    *,
    selected_mode: str | None = None,
) -> None:
    failure_frame = frame[frame["source_name"] == "fail_rollout"].copy()
    if failure_frame.empty:
        return

    for _, rollout_frame in failure_frame.groupby("rollout_id", sort=False):
        if selected_mode is not None and str(selected_mode) not in _rollout_failure_modes(rollout_frame):
            continue
        rollout_frame = rollout_frame.sort_values("timestep")
        points = rollout_frame[["embedding_x", "embedding_y"]].to_numpy(dtype=np.float32)
        colors = _annotated_failure_colors_for_mode(
            rollout_frame,
            selected_mode=selected_mode,
        )
        if float(cfg.analysis.failure_path_alpha) > 0.0:
            ax.plot(
                points[:, 0],
                points[:, 1],
                color="#D3DCE8",
                linewidth=float(cfg.analysis.failure_path_linewidth),
                alpha=float(cfg.analysis.failure_path_alpha),
                zorder=2,
                rasterized=True,
            )
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=float(cfg.analysis.failure_point_size),
            c=colors,
            alpha=float(cfg.analysis.failure_alpha),
            linewidths=0.0,
            rasterized=True,
            zorder=3,
        )

        segment_frame = rollout_frame[
            (rollout_frame["display_group"] == "fail_rollout_segment")
            & (
                rollout_frame["failure_mode"].astype(str) == str(selected_mode)
                if selected_mode is not None
                else True
            )
        ]
        if not segment_frame.empty:
            onset = segment_frame.iloc[0]
            ax.scatter(
                [float(onset["embedding_x"])],
                [float(onset["embedding_y"])],
                s=float(cfg.analysis.failure_onset_marker_size),
                color=FAILURE_START_COLOR,
                edgecolors="white",
                linewidths=0.85,
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


def _annotated_failure_legend(ax: plt.Axes, cfg: DictConfig) -> None:
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=6.8,
            markerfacecolor=SUCCESS_COLOR,
            markeredgecolor="none",
            alpha=float(cfg.analysis.success_alpha),
            label="Success rollout",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=6.8,
            markerfacecolor=FAILURE_NORMAL_COLOR,
            markeredgecolor="none",
            label="Failure rollout (pre-failure)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=6.8,
            markerfacecolor=FAILURE_START_COLOR,
            markeredgecolor="none",
            label="Failure segment start",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=6.8,
            markerfacecolor=FAILURE_END_COLOR,
            markeredgecolor="none",
            label="Failure segment end",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        fontsize=float(cfg.analysis.failure_legend_fontsize),
        title="Embedding Groups",
        title_fontsize=float(cfg.analysis.failure_legend_title_fontsize),
    )


def _annotated_failure_modes(frame: pd.DataFrame) -> list[str]:
    failure_modes = sorted(
        {
            str(mode)
            for mode in frame.loc[frame["failure_mode"].astype(str) != "", "failure_mode"].astype(str).tolist()
            if str(mode).strip()
        }
    )
    return failure_modes


def _set_embedding_axis_labels(ax: plt.Axes, cfg: DictConfig) -> None:
    x_label, y_label = _embedding_axis_labels(_embedding_method(cfg))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


def _annotated_failure_output_basename(selected_mode: str | None = None) -> str:
    if selected_mode is None:
        return "annotated_failure_embedding"
    return "annotated_failure_embedding_by_mode"


def _plot_annotated_failure_embedding(
    frame: pd.DataFrame,
    cfg: DictConfig,
    run_dir: str,
    *,
    selected_mode: str | None = None,
) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(9.8, 7.8))
    _scatter_success_background(ax, frame, cfg)
    _plot_annotated_failure_rollouts(ax, frame, cfg, selected_mode=selected_mode)
    _style_embedding_axes(ax)
    _annotated_failure_legend(ax, cfg)
    ax.text(
        0.02,
        0.02,
        ANNOTATED_FAILURE_TEXT
        if selected_mode is None
        else f"Failure mode: {selected_mode}. Success rollouts stay in the background.",
        transform=ax.transAxes,
        fontsize=float(cfg.analysis.annotation_fontsize),
        color="#475569",
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#E2E8F0", "alpha": 0.94},
    )
    display_name = _embedding_display_name(_embedding_method(cfg))
    ax.set_title(
        f"Policy Latent {display_name} | Success Rollouts and Annotated Failures"
        if selected_mode is None
        else f"Policy Latent {display_name} | Failure Mode: {selected_mode}"
    )
    _set_embedding_axis_labels(ax, cfg)
    base_name = _annotated_failure_output_basename(selected_mode=selected_mode)
    return _save_figure(fig, os.path.join(run_dir, base_name), dpi=int(cfg.analysis.plot_dpi))


def _plot_annotated_failure_embedding_per_mode(
    frame: pd.DataFrame,
    cfg: DictConfig,
    run_dir: str,
) -> dict[str, dict[str, str]]:
    per_mode_dir = os.path.join(run_dir, "per_mode")
    os.makedirs(per_mode_dir, exist_ok=True)

    outputs: dict[str, dict[str, str]] = {}
    for failure_mode in _annotated_failure_modes(frame):
        mode_dir = os.path.join(per_mode_dir, _safe_path_component(failure_mode))
        os.makedirs(mode_dir, exist_ok=True)
        outputs[failure_mode] = _plot_annotated_failure_embedding(
            frame,
            cfg,
            mode_dir,
            selected_mode=failure_mode,
        )
    return outputs


def _plot_by_task(frame: pd.DataFrame, cfg: DictConfig, run_dir: str) -> dict[str, str]:
    task_names = ordered_task_names(frame["task_name"].astype(str).unique().tolist())
    palette = dict(zip(task_names, sns.color_palette("tab10", n_colors=len(task_names))))
    fig, ax = plt.subplots(figsize=(9.2, 7.6))
    sns.scatterplot(
        data=frame,
        x="embedding_x",
        y="embedding_y",
        hue="task_name",
        hue_order=task_names,
        palette=palette,
        s=float(cfg.analysis.point_size),
        alpha=float(cfg.analysis.point_alpha),
        linewidth=0.0,
        ax=ax,
    )
    _style_embedding_axes(ax)
    ax.set_title(f"Policy Latent {_embedding_display_name(_embedding_method(cfg))} by Task")
    _set_embedding_axis_labels(ax, cfg)
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


def _plot_annotated_failure_embedding_by_task(
    frame: pd.DataFrame,
    cfg: DictConfig,
    run_dir: str,
) -> dict[str, str]:
    task_names = ordered_task_names(frame["task_name"].astype(str).unique().tolist())
    palette = dict(zip(task_names, sns.color_palette("tab10", n_colors=len(task_names))))
    fig, ax = plt.subplots(figsize=(9.4, 7.7))
    sns.scatterplot(
        data=frame,
        x="embedding_x",
        y="embedding_y",
        hue="task_name",
        hue_order=task_names,
        palette=palette,
        s=float(cfg.analysis.failure_point_size),
        alpha=float(cfg.analysis.point_alpha),
        linewidth=0.0,
        ax=ax,
    )
    _style_embedding_axes(ax)
    display_name = _embedding_display_name(_embedding_method(cfg))
    ax.set_title(f"Policy Latent {display_name} | Selected Points Colored by Task")
    _set_embedding_axis_labels(ax, cfg)
    ax.legend(
        title="Task",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=True,
        fontsize=float(cfg.analysis.task_legend_fontsize),
        title_fontsize=float(cfg.analysis.task_legend_title_fontsize),
    )
    return _save_figure(
        fig,
        os.path.join(run_dir, "annotated_failure_embedding_by_task"),
        dpi=int(cfg.analysis.plot_dpi),
    )


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
    ax.set_title(f"Policy Latent Failure Pattern | {_embedding_display_name(_embedding_method(cfg))}")
    _set_embedding_axis_labels(ax, cfg)
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
        ax.set_title(
            f"Policy Latent Failure Pattern | {task_name} | {_embedding_display_name(_embedding_method(cfg))}"
        )
        _set_embedding_axis_labels(ax, cfg)
        outputs[task_name] = _save_figure(
            fig,
            os.path.join(task_output_dir, "failure_pattern"),
            dpi=int(cfg.analysis.plot_dpi),
        )
    return outputs
