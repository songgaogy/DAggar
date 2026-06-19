"""Diagnostic: is P (pre-done success frames) separable from U (whole failure rollouts)?

Uses the EXACT same frozen-encoder feature space the nnPU head trains on
(``feature_source=transformer``, ``transformer_layer=1``) so the read-out is
faithful to training. No nnPU is trained here -- instead we fit a fully
*supervised* logistic-regression probe on P-vs-U (trajectory-disjoint train/test
split). That supervised AUROC is the **separability upper bound**:

  * supervised AUROC ~ 0.5  -> P and U are not separable in this feature space;
    the nnPU collapse (risk == pi_p) is a data/feature problem, not optimization.
  * supervised AUROC high but nnPU collapsed -> features are fine; the collapse
    is an nnPU optimization / pi_p / surrogate problem.

It also reports, for reference, P_whole-vs-U (whole success traj, i.e. the
pre-change positives) and P_predone-vs-P_postdone, and draws g(z) histograms.

Run (GPU):
    MODEL_CKPT=checkpoints/.../checkpoint/model_50.pth \
    /home/dodo/miniconda3/envs/dagger/bin/python \
        -m robosuite.discriminator.dyn_disc.diagnose_pu_separability \
        --task PickPlaceCereal --n-success 50 --n-fail 50
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from robosuite.discriminator.utils.robosuite_benchmark import (
    discover_success_rollouts,
    discover_unlabeled_failures,
)
from robosuite.discriminator.dyn_disc.adapters.pu_bce import PUBCEBenchmarkDiscriminator


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    default_ckpt = os.environ.get(
        "MODEL_CKPT",
        "checkpoints/dyn_disc/dynamics/dinov3_dyn_robosuite-20260619_024518/checkpoint/model_50.pth",
    )
    p.add_argument("--model-ckpt", default=default_ckpt)
    p.add_argument("--data-root", default="data")
    p.add_argument("--task", default="PickPlaceCereal")
    p.add_argument("--success-train-split", default="success_rollout")
    p.add_argument("--fail-train-split", default="fail_rollout")
    p.add_argument("--n-success", type=int, default=50)
    p.add_argument("--n-fail", type=int, default=50)
    p.add_argument("--transformer-layer", type=int, default=1)
    p.add_argument("--feature-source", default="transformer")
    p.add_argument("--device", default="cuda")
    p.add_argument("--test-frac", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


def _stack(feats):
    """list[(T,D) tensor] -> (sum_T, D) float32 numpy."""
    arrs = [f.detach().cpu().numpy().reshape(-1, f.shape[-1]).astype(np.float32)
            for f in feats if f.numel() > 0]
    return np.concatenate(arrs, axis=0) if arrs else np.zeros((0, 0), np.float32)


def _probe_auroc(Xtr, ytr, Xte, yte, seed):
    """Supervised separability upper bound; returns (auroc, g_on_test)."""
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    clf.fit(sc.transform(Xtr), ytr)
    g = clf.decision_function(sc.transform(Xte))  # >0 -> class 1 (positive/success-like)
    return float(roc_auc_score(yte, g)), g


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir or f"checkpoints/pu_diag_{args.task}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[diag] task={args.task} ckpt={args.model_ckpt}", flush=True)
    succ = discover_success_rollouts(
        data_root=args.data_root, tasks=[args.task],
        split=args.success_train_split, max_success_per_task=args.n_success,
    )
    fail = discover_unlabeled_failures(
        data_root=args.data_root, tasks=[args.task],
        split=args.fail_train_split, max_fail_per_task=args.n_fail,
    )
    fail = [t for t in fail if bool(t.is_failure)]
    print(f"[diag] discovered success={len(succ)} fail={len(fail)}", flush=True)
    if not succ or not fail:
        raise SystemExit("need both success and fail trajectories")

    disc = PUBCEBenchmarkDiscriminator(
        model_ckpt=args.model_ckpt,
        unlabeled_fail_trajectories=fail,
        pi_p=0.3,
        device=args.device,
        feature_source=args.feature_source,
        transformer_layer=args.transformer_layer,
        verbose_fit=False,
    )

    # ---- encode, trajectory by trajectory (so we can split disjointly) ----
    print("[diag] encoding success (pre-done + whole) ...", flush=True)
    succ_predone, succ_postdone, succ_whole = [], [], []
    pre_lens, whole_lens = [], []
    for t in succ:
        t_end = disc._success_prefix_frame_end(t)
        f_pre = disc._encode(t, frame_end=t_end)
        f_whole = disc._encode(t)
        succ_predone.append(f_pre)
        succ_whole.append(f_whole)
        n_pre = int(f_pre.shape[0])
        succ_postdone.append(f_whole[n_pre:])
        pre_lens.append(n_pre)
        whole_lens.append(int(f_whole.shape[0]))

    print("[diag] encoding fail (whole) ...", flush=True)
    fail_whole = [disc._encode(t) for t in fail]

    pre_lens = np.array(pre_lens); whole_lens = np.array(whole_lens)
    print(f"[diag] encoded feat-frames/traj: pre-done mean={pre_lens.mean():.1f} "
          f"whole mean={whole_lens.mean():.1f} "
          f"(pre-done is {100*pre_lens.sum()/whole_lens.sum():.1f}% of whole-success frames)",
          flush=True)

    # ---- trajectory-disjoint train/test split ----
    def split_idx(n):
        idx = rng.permutation(n)
        k = max(1, int(round(n * (1 - args.test_frac))))
        return set(idx[:k].tolist()), set(idx[k:].tolist())

    s_tr, s_te = split_idx(len(succ))
    f_tr, f_te = split_idx(len(fail))

    def gather(seq, idxs):
        return _stack([seq[i] for i in sorted(idxs)])

    # Build explicit matrices for the three comparisons.
    P_pre_tr, P_pre_te = gather(succ_predone, s_tr), gather(succ_predone, s_te)
    P_whole_tr, P_whole_te = gather(succ_whole, s_tr), gather(succ_whole, s_te)
    P_post_tr, P_post_te = gather(succ_postdone, s_tr), gather(succ_postdone, s_te)
    U_tr, U_te = gather(fail_whole, f_tr), gather(fail_whole, f_te)

    results = {}

    def comparison(name, A_tr, A_te, B_tr, B_te):
        Xtr = np.concatenate([A_tr, B_tr], 0)
        ytr = np.concatenate([np.ones(len(A_tr)), np.zeros(len(B_tr))])
        Xte = np.concatenate([A_te, B_te], 0)
        yte = np.concatenate([np.ones(len(A_te)), np.zeros(len(B_te))])
        auroc, g = _probe_auroc(Xtr, ytr, Xte, yte, args.seed)
        gA = g[: len(A_te)]
        gB = g[len(A_te):]
        results[name] = dict(auroc=auroc, gA=gA, gB=gB,
                             nA=len(A_te), nB=len(B_te))
        print(f"[diag] {name:28s} supervised test AUROC = {auroc:.4f}  "
              f"(class1 n={len(A_te)}, class0 n={len(B_te)})", flush=True)
        return auroc

    print("\n[diag] === separability upper bounds (supervised logistic probe) ===", flush=True)
    comparison("P_predone_vs_U", P_pre_tr, P_pre_te, U_tr, U_te)      # <- current setup
    comparison("P_whole_vs_U", P_whole_tr, P_whole_te, U_tr, U_te)    # <- pre-change setup (reference)
    comparison("P_predone_vs_P_postdone", P_pre_tr, P_pre_te, P_post_tr, P_post_te)

    # ---- histograms ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for ax, name in zip(axes, ["P_predone_vs_U", "P_whole_vs_U", "P_predone_vs_P_postdone"]):
        r = results[name]
        lo = min(r["gA"].min(), r["gB"].min())
        hi = max(r["gA"].max(), r["gB"].max())
        bins = np.linspace(lo, hi, 60)
        ax.hist(r["gA"], bins=bins, alpha=0.55, density=True, label="class1")
        ax.hist(r["gB"], bins=bins, alpha=0.55, density=True, label="class0")
        ax.set_title(f"{name}\nAUROC={r['auroc']:.3f}")
        ax.set_xlabel("probe g(z) (>0 -> class1)")
        ax.legend()
    fig.suptitle(f"PU separability probe — task={args.task}  layer={args.transformer_layer}")
    fig.tight_layout()
    png = out_dir / f"separability_{args.task}.png"
    fig.savefig(png, dpi=130)
    print(f"\n[diag] saved histogram -> {png}", flush=True)

    print("\n[diag] === interpretation ===", flush=True)
    a_cur = results["P_predone_vs_U"]["auroc"]
    a_ref = results["P_whole_vs_U"]["auroc"]
    print(f"  current (pre-done) P-vs-U AUROC = {a_cur:.3f}", flush=True)
    print(f"  pre-change (whole) P-vs-U AUROC = {a_ref:.3f}", flush=True)
    print(f"  drop from dropping post-done frames = {a_ref - a_cur:+.3f}", flush=True)
    if a_cur < 0.6:
        print("  => P_predone is NOT separable from U: nnPU collapse is a "
              "FEATURE/DATA problem (the head has no signal to fit).", flush=True)
    else:
        print("  => P_predone IS separable (AUROC>=0.6): collapse is more likely "
              "an nnPU optimization / pi_p / surrogate problem.", flush=True)


if __name__ == "__main__":
    main()
