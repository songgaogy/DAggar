"""Integration test: visualize LPB disc reward on one HDF5 demo (failure_score + r_disc).

Requires repo assets (skipped automatically when missing):
  - checkpoints/lpb_v2/bce_ckeckpoints/<TASK>/bce_head.pth + meta.json
  - data/<TASK>/fail_rollout/*.hdf5

Run (writes PNG under pytest tmp_path; use -s to print paths):
  pytest robosuite/pipeline/algorithms/q_learning/tests/test_visualize_lpb_disc_reward.py -s

Optional env:
  LPB_DISC_VIZ_TASK=PickPlaceMilk
  LPB_DISC_VIZ_HDF5=/abs/path/to/demo.hdf5
  LPB_DISC_VIZ_DEMO=demo_000035
  LPB_DISC_VIZ_DEVICE=cuda:0
  LPB_DISC_VIZ_OUTPUT=/abs/path/to/plot.png   # if set, also save here
  LPB_DISC_VIZ_SKIP_ASSERT=1                # skip fail-rollout shape checks (e.g. success_rollout)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from robosuite.pipeline.algorithms.discriminator.lpb_v2_scorer import (
    LPBV2OfflineScorer,
    lpb_disc_intrinsic_from_failure_score,
)
from robosuite.pipeline.utils.io import list_hdf5_demo_names


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _default_assets() -> tuple[Path, Path, Path, str] | None:
    task = os.environ.get("LPB_DISC_VIZ_TASK", "PickPlaceMilk")
    root = _repo_root()
    ckpt = root / "checkpoints/lpb_v2/bce_ckeckpoints" / task / "bce_head.pth"
    meta = ckpt.parent / "meta.json"
    hdf5_env = os.environ.get("LPB_DISC_VIZ_HDF5")
    if hdf5_env:
        hdf5 = Path(hdf5_env).resolve()
    else:
        fail_dir = root / "data" / task / "fail_rollout"
        candidates = sorted(fail_dir.glob("*.hdf5")) + sorted(fail_dir.glob("*.h5"))
        hdf5 = candidates[0] if candidates else Path()
    demo = os.environ.get("LPB_DISC_VIZ_DEMO", "demo_000035")
    if not ckpt.is_file() or not meta.is_file() or not hdf5.is_file():
        return None
    demo_names = list_hdf5_demo_names(hdf5)
    if demo not in demo_names:
        demo = demo_names[0] if demo_names else ""
    if not demo:
        return None
    return ckpt, meta, hdf5, demo


def _resolve_device() -> str:
    requested = os.environ.get("LPB_DISC_VIZ_DEVICE", "cuda:0")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


def gamma_aggregate_sliding(
    per_step: np.ndarray, *, horizon: int, discount: float
) -> np.ndarray:
    """Σ_{i=0}^{H-1} γ^i r_{t+i} with start = min(t, T-H) (matches vis_qv windows)."""
    values = np.asarray(per_step, dtype=np.float64).reshape(-1)
    transition_count = int(values.shape[0])
    if transition_count == 0:
        return np.zeros((0,), dtype=np.float32)
    horizon_i = max(1, int(horizon))
    discount_f = float(discount)
    max_start = max(0, transition_count - horizon_i)
    powers = discount_f ** np.arange(horizon_i, dtype=np.float64)
    out = np.zeros((transition_count,), dtype=np.float64)
    for step in range(transition_count):
        start = min(int(step), max_start)
        chunk = values[start : start + horizon_i]
        out[step] = float((chunk * powers[: chunk.shape[0]]).sum())
    return out.astype(np.float32)


def plot_lpb_disc_reward_timeseries(
    *,
    failure_score: np.ndarray,
    intrinsic: np.ndarray,
    disc_gamma_agg: np.ndarray,
    tau: float,
    title: str,
    output_png: Path,
    action_horizon: int = 8,
) -> None:
    steps = np.arange(int(failure_score.shape[0]), dtype=np.int32)
    pred_failure = failure_score >= float(tau)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(title, fontsize=11)

    ax0 = axes[0]
    ax0.plot(steps, failure_score, color="tab:blue", linewidth=1.4, label="failure_score")
    ax0.axhline(float(tau), color="tab:red", linestyle="--", linewidth=1.1, label=f"tau={tau:.3f}")
    ax0.set_ylabel("failure_score")
    ax0.legend(loc="best", fontsize=8)
    ax0.grid(True, alpha=0.25)

    ax1 = axes[1]
    ax1.plot(steps, intrinsic, color="tab:pink", linewidth=1.4, label="r_disc per frame")
    ax1.axhline(-1.0, color="gray", linestyle=":", linewidth=1)
    ax1.axhline(0.0, color="gray", linestyle=":", linewidth=1)
    ax1.set_ylabel("r_disc")
    ax1.set_ylim(-1.05, 0.05)
    ax1.legend(loc="best", fontsize=8)
    ax1.grid(True, alpha=0.25)

    ax2 = axes[2]
    ax2.plot(
        steps,
        disc_gamma_agg,
        color="tab:purple",
        linewidth=1.4,
        label=f"gamma-agg r_disc (H={int(action_horizon)})",
    )
    ax2.set_ylabel("chunk r_disc")
    ax2.set_xlabel("frame")
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(True, alpha=0.25)

    if bool(pred_failure.any()):
        ymin = float(np.nanmin(intrinsic))
        band = np.where(pred_failure, ymin, np.nan)
        ax1.plot(steps, band, color="tab:red", linewidth=4, alpha=0.35, label="pred failure")

    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def visualize_demo_lpb_disc_reward(
    *,
    bce_ckpt: Path,
    meta_json: Path,
    hdf5_path: Path,
    demo_key: str,
    task_name: str,
    device: str,
    output_png: Path,
    action_horizon: int = 8,
    discount: float = 0.99,
) -> dict[str, float]:
    """Score one demo and write a 3-panel disc-reward figure. Returns summary stats."""
    scorer = LPBV2OfflineScorer(
        bce_ckpt_path=bce_ckpt,
        task_name=task_name,
        device=device,
        batch_size=32,
        meta_json_path=meta_json,
    )
    failure_score = np.asarray(
        scorer.score_hdf5_demo(hdf5_path, demo_key, fps=20), dtype=np.float32
    )
    tau = float(scorer.tau)
    intrinsic = np.asarray(
        lpb_disc_intrinsic_from_failure_score(failure_score, tau), dtype=np.float32
    )
    disc_gamma_agg = gamma_aggregate_sliding(
        intrinsic, horizon=int(action_horizon), discount=float(discount)
    )

    title = f"{task_name} {hdf5_path.name}::{demo_key} (LPB disc reward)"
    plot_lpb_disc_reward_timeseries(
        failure_score=failure_score,
        intrinsic=intrinsic,
        disc_gamma_agg=disc_gamma_agg,
        tau=tau,
        title=title,
        output_png=output_png,
        action_horizon=int(action_horizon),
    )

    corr = float(np.corrcoef(failure_score, intrinsic)[0, 1])
    return {
        "num_frames": float(failure_score.shape[0]),
        "tau": tau,
        "failure_score_early_mean": float(failure_score[:50].mean()),
        "failure_score_late_mean": float(failure_score[-50:].mean()),
        "intrinsic_early_mean": float(intrinsic[:50].mean()),
        "intrinsic_late_mean": float(intrinsic[-50:].mean()),
        "corr_failure_score_intrinsic": corr,
        "pred_failure_frames": float((failure_score >= tau).sum()),
    }


@pytest.fixture
def lpb_disc_viz_assets() -> tuple[Path, Path, Path, str]:
    assets = _default_assets()
    if assets is None:
        pytest.skip(
            "LPB disc viz assets missing. Need bce_head.pth, meta.json, and an HDF5 fail demo."
        )
    return assets


def test_visualize_lpb_disc_reward_on_fail_demo(
    lpb_disc_viz_assets: tuple[Path, Path, Path, str],
    tmp_path: Path,
) -> None:
    """End-to-end: LPB score one trajectory and save disc-reward timeseries PNG."""
    ckpt, meta, hdf5, demo_key = lpb_disc_viz_assets
    task = os.environ.get("LPB_DISC_VIZ_TASK", "PickPlaceMilk")
    out_png = tmp_path / f"lpb_disc_reward_{task}_{demo_key}.png"
    extra_out = os.environ.get("LPB_DISC_VIZ_OUTPUT")
    if extra_out:
        out_png = Path(extra_out).resolve()
        out_png.parent.mkdir(parents=True, exist_ok=True)

    horizon = int(os.environ.get("LPB_DISC_VIZ_HORIZON", "8"))
    discount = float(os.environ.get("LPB_DISC_VIZ_DISCOUNT", "0.99"))

    summary = visualize_demo_lpb_disc_reward(
        bce_ckpt=ckpt,
        meta_json=meta,
        hdf5_path=hdf5,
        demo_key=demo_key,
        task_name=task,
        device=_resolve_device(),
        output_png=out_png,
        action_horizon=horizon,
        discount=discount,
    )

    print(f"[lpb_disc_viz] wrote {out_png}")
    print(f"[lpb_disc_viz] summary={json.dumps(summary, indent=2)}")

    assert out_png.is_file() and out_png.stat().st_size > 10_000
    if os.environ.get("LPB_DISC_VIZ_SKIP_ASSERT", "").strip() in ("1", "true", "yes"):
        return
    assert summary["num_frames"] >= 50.0
    assert summary["corr_failure_score_intrinsic"] < -0.5
    # Fail rollout: later frames should be more failure-like (higher score, more negative r_disc).
    assert summary["failure_score_late_mean"] > summary["failure_score_early_mean"]
    assert summary["intrinsic_late_mean"] < summary["intrinsic_early_mean"]
