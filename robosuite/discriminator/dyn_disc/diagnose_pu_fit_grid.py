"""Diagnostic #2: which nnPU optimization knob causes the collapse?

The separability probe (diagnose_pu_separability.py) showed P_predone-vs-U is
separable (supervised AUROC ~0.77), so the collapse (risk == pi_p) is an nnPU
*optimization* problem, not a feature problem. This script encodes P (pre-done
success) and U (whole fail) ONCE with the frozen RPT encoder, then fits the REAL
``PUBCEDiscriminator`` head over a grid of (pi_p, surrogate, nn_correction) and
reports, per config:

  * final-epoch risk and neg_risk (collapsed iff risk ~ pi_p and neg_risk ~ 0),
  * the head's own held-out failure-ranking AUROC: rank U above P using the
    failure score -g(z). 0.5 == collapsed/no signal; high == learned.

Run (GPU):
    MODEL_CKPT=.../model_50.pth \
    /home/dodo/miniconda3/envs/dagger/bin/python \
        -m robosuite.discriminator.dyn_disc.diagnose_pu_fit_grid \
        --task PickPlaceCereal --n-success 50 --n-fail 50
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from robosuite.discriminator.utils.robosuite_benchmark import (
    discover_success_rollouts,
    discover_unlabeled_failures,
)
from robosuite.discriminator.dyn_disc.adapters.pu_bce import PUBCEBenchmarkDiscriminator
from robosuite.discriminator.dyn_disc.detectors.pu_bce import PUBCEDiscriminator


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    default_ckpt = os.environ.get(
        "MODEL_CKPT",
        "checkpoints/dyn_disc/ablations/RPT/pretrain/latest/checkpoint/model_50.pth",
    )
    p.add_argument("--model-ckpt", default=default_ckpt)
    p.add_argument("--data-root", default="data")
    p.add_argument("--task", default="PickPlaceCereal")
    p.add_argument("--success-train-split", default="success_rollout")
    p.add_argument("--fail-train-split", default="fail_rollout")
    p.add_argument("--n-success", type=int, default=50)
    p.add_argument("--n-fail", type=int, default=50)
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--test-frac", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _split(n, rng, test_frac):
    idx = rng.permutation(n)
    k = max(1, int(round(n * (1 - test_frac))))
    return sorted(idx[:k].tolist()), sorted(idx[k:].tolist())


def main() -> None:
    args = _parse_args()
    rng = np.random.default_rng(args.seed)

    succ = discover_success_rollouts(
        data_root=args.data_root, tasks=[args.task],
        split=args.success_train_split, max_success_per_task=args.n_success,
    )
    fail = [t for t in discover_unlabeled_failures(
        data_root=args.data_root, tasks=[args.task],
        split=args.fail_train_split, max_fail_per_task=args.n_fail,
    ) if bool(t.is_failure)]
    print(f"[grid] success={len(succ)} fail={len(fail)}", flush=True)

    disc = PUBCEBenchmarkDiscriminator(
        model_ckpt=args.model_ckpt, unlabeled_fail_trajectories=fail, pi_p=0.3,
        device=args.device, verbose_fit=False,
    )

    print("[grid] encoding P (pre-done) + U (whole) ...", flush=True)
    P = [disc._encode(t, frame_end=disc._success_prefix_frame_end(t)) for t in succ]
    U = [disc._encode(t) for t in fail]
    in_dim = int(P[0].shape[-1])

    s_tr, s_te = _split(len(P), rng, args.test_frac)
    f_tr, f_te = _split(len(U), rng, args.test_frac)
    P_tr = [P[i] for i in s_tr]
    U_tr = [U[i] for i in f_tr]
    # held-out frame matrices
    P_te = torch.cat([P[i].reshape(-1, in_dim) for i in s_te], 0)
    U_te = torch.cat([U[i].reshape(-1, in_dim) for i in f_te], 0)

    # dummy per-task calib pool (head.fit requires it; we don't use the threshold)
    calib = {args.task: [P[i] for i in s_tr[: max(1, len(s_tr) // 4)]]}

    def fail_auroc(det: PUBCEDiscriminator) -> float:
        gP = det._logits_np(P_te)
        gU = det._logits_np(U_te)
        # failure score = -g ; U should rank higher (more failure) than P
        y = np.concatenate([np.ones(len(gU)), np.zeros(len(gP))])
        s = np.concatenate([-gU, -gP])
        return float(roc_auc_score(y, s))

    grid = []
    for surrogate in ("sigmoid", "logistic"):
        for pi_p in (0.3, 0.5, 0.7):
            for nn_corr in (True,):
                grid.append((surrogate, pi_p, nn_corr))

    print("\n[grid] surrogate  pi_p  nnPU |  final_risk  final_neg | collapsed? | fail-AUROC", flush=True)
    print("[grid] " + "-" * 78, flush=True)
    for surrogate, pi_p, nn_corr in grid:
        det = PUBCEDiscriminator(in_dim=in_dim, hidden=256, num_layers=2, device=args.device)
        det.fit(
            P_tr, U_tr, calib,
            pi_p=pi_p, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
            delta=10.0, seed=args.seed, loss_surrogate=surrogate,
            nn_correction=nn_corr, beta=0.0, verbose=False,
        )
        hist = det._train_history[-1]
        risk = hist["loss"]; neg = hist["neg_risk"]
        collapsed = abs(risk - pi_p) < 0.02 and abs(neg) < 0.02
        auroc = fail_auroc(det)
        print(f"[grid] {surrogate:9s} {pi_p:.2f}  {str(nn_corr):4s} | "
              f"{risk:9.4f}  {neg:+8.4f} | {'YES' if collapsed else 'no ':9s} | {auroc:.4f}",
              flush=True)

    print("\n[grid] fail-AUROC ~0.5 == collapsed (head learned nothing); "
          ">0.7 == head learned a useful failure ranking.", flush=True)


if __name__ == "__main__":
    main()
