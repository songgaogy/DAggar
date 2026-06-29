"""Matplotlib figures for offline DIPOLE training diagnostics."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_raw_g_distribution(
    g_raw: np.ndarray,
    *,
    density: bool = True,
    bins: int = 30,
) -> plt.Figure:
    """Histogram of provider raw G for the current training batch."""
    values = np.asarray(g_raw, dtype=np.float64).reshape(-1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(
        values,
        bins=bins,
        density=density,
        alpha=0.75,
        color="steelblue",
        edgecolor="white",
        label=f"raw G (n={values.size})",
    )
    ax.set_xlabel("Raw G")
    ax.set_ylabel("Density" if density else "Count")
    ax.set_title("Batch raw G distribution")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_branch_weight_distribution(
    weights: np.ndarray,
    *,
    label: str,
    density: bool = True,
    bins: int = 30,
) -> plt.Figure:
    """Histogram of per-sample branch weights (w_pos or w_neg) in [0, 1]."""
    values = np.asarray(weights, dtype=np.float64).reshape(-1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(
        values,
        bins=bins,
        range=(0.0, 1.0),
        density=density,
        alpha=0.75,
        color="forestgreen" if label == "w_pos" else "indianred",
        edgecolor="white",
        label=f"{label} (n={values.size}, mean={values.mean():.3f})",
    )
    ax.set_xlabel(label)
    ax.set_ylabel("Density" if density else "Count")
    ax.set_title(f"Batch {label} distribution")
    ax.set_xlim(0.0, 1.0)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_v_pos_neg_scatter(v_pos_mse: np.ndarray, v_neg_mse: np.ndarray) -> plt.Figure:
    """Scatter of per-sample flow MSE for positive vs negative branches."""
    x = np.asarray(v_pos_mse, dtype=np.float64).reshape(-1)
    y = np.asarray(v_neg_mse, dtype=np.float64).reshape(-1)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(x, y, s=12, alpha=0.6, edgecolors="none")
    lo = float(min(np.min(x), np.min(y))) if x.size > 0 else 0.0
    hi = float(max(np.max(x), np.max(y))) if x.size > 0 else 1.0
    if hi <= lo:
        hi = lo + 1.0
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.0, alpha=0.5, label="y=x")
    ax.set_xlabel("v_pos flow MSE")
    ax.set_ylabel("v_neg flow MSE")
    ax.set_title(f"Per-sample branch flow MSE (n={x.size})")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return fig
