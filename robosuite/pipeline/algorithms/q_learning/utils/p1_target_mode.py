"""P1 diagnostic: compare three value-learning TARGET MODES on the SAME fixed
windows, to localize whether the IQL warmup failure is the action-free V
bootstrap specifically, or TD propagation in general.

Arms (all on the SAME cached features / SAME train+val demo split as P0):
  1. iql_v     — reproduce the production warmup faithfully (reuse IQLLearner,
                 IQLConfig, IQLStepBatch). 20000 warmup_value_only + 10000 update.
  2. sarsa_q   — FQE / behavior evaluation: a 2-net Q ensemble (min over 2)
                 trained with y = r + gamma^H (1-done) Q_target(s', a'_recorded).
  3. mc_return — supervised regression of V (and Q) to G_label (reuse P0 result).

Everything is derived from P0's preencoded_features.pt. NO re-encoding, NO buffer
reload. See the task brief for the reward / done / next_row derivation.

Run (module-style):
    cd <repo-root> && MUJOCO_GL=egl PYTHONPATH=. \
        CUDA_VISIBLE_DEVICES=1 python \
        -m robosuite.pipeline.algorithms.q_learning.utils.p1_target_mode \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from robosuite.pipeline.algorithms.q_learning.common import IQLConfig, IQLStepBatch
from robosuite.pipeline.algorithms.q_learning.iql import IQLLearner
from robosuite.pipeline.algorithms.q_learning.networks import QChunkNetwork, VNetwork

# ---- Fixed hyperparameters (mirror train_dipole_rl.yaml q_learning.config) ----
GAMMA = 0.99
ACTION_HORIZON = 8
HIDDEN_DIMS = (512, 512)
OUTPUT_REWARD_COEF = 0.1  # already baked into G_label
EXPECTILE_TAU = 0.7
Q_ENSEMBLE_SIZE = 10
V_SUBSET_SIZE = 2
Q_LR = V_LR = 3e-4
TARGET_POLYAK = 0.005
GRAD_CLIP = 2.0
WEIGHT_DECAY = 1e-6

REPO_ROOT = Path(__file__).resolve().parents[5]
P0_DIR = REPO_ROOT / "outputs/dipole_rl-debug/p0_mc_probe"
FEAT_PATH = P0_DIR / "preencoded_features.pt"
OUT_DIR = REPO_ROOT / "outputs/dipole_rl-debug/p1_target_mode"


def _arr(x) -> np.ndarray:
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


# --------------------------------------------------------------------------- #
# Reconstruction of n-step reward / done / next_row from cached rows          #
# --------------------------------------------------------------------------- #
def build_step_quantities(blob: dict) -> dict:
    G = _arr(blob["G_label"]).astype(np.float64)
    ep = _arr(blob["episode_index_per_row"]).astype(np.int64)
    st = _arr(blob["episode_step_per_row"]).astype(np.int64)
    succ = _arr(blob["is_success_per_row"]).astype(bool)
    n = len(G)

    key2row = {(int(ep[r]), int(st[r])): r for r in range(n)}
    next_row = np.array(
        [key2row.get((int(ep[r]), int(st[r]) + ACTION_HORIZON), -1) for r in range(n)],
        dtype=np.int64,
    )
    have_next = next_row >= 0

    # n-step chunk reward via telescoping: r = G[s] - gamma^H * G[next_row(s)]
    disc_H = GAMMA**ACTION_HORIZON
    r_chunk = np.full(n, np.nan, dtype=np.float64)
    r_chunk[have_next] = G[have_next] - disc_H * G[next_row[have_next]]

    # first-success step t_s per success episode = min step where G == 0 (frozen
    # absorbing anchor sets reward 0 => return-to-go 0 from the success frame on).
    succ_eps = sorted({int(e) for e in ep[succ]})
    t_s: dict[int, int] = {}
    for e in succ_eps:
        m = ep == e
        zero_steps = st[m][np.isclose(G[m], 0.0)]
        t_s[e] = int(zero_steps.min()) if zero_steps.size else 10**9

    # done per production semantics: success episode AND chunk contains a success
    # frame (step + H - 1 >= t_s). Fail episodes never done.
    done = np.zeros(n, dtype=np.float64)
    for r in range(n):
        if succ[r] and int(st[r]) + ACTION_HORIZON - 1 >= t_s[int(ep[r])]:
            done[r] = 1.0

    return {
        "G": G, "ep": ep, "st": st, "succ": succ,
        "next_row": next_row, "have_next": have_next,
        "r_chunk": r_chunk, "done": done, "t_s": t_s,
        "succ_eps": succ_eps,
    }


def reproduce_p0_split(succ_eps: list[int], fail_eps: list[int], seed: int, val_frac: float):
    """Replicate P0's demo split EXACTLY: one shared RandomState, success split
    first then fail split, sorted episode lists. (mc_return_probe.py L494-503)"""
    rng = np.random.RandomState(seed)

    def _split(eps: list[int]):
        eps = list(eps)
        rng.shuffle(eps)
        n_val = max(1, int(round(len(eps) * val_frac))) if eps else 0
        return set(eps[n_val:]), set(eps[:n_val])

    succ_train, succ_val = _split(sorted(succ_eps))
    fail_train, fail_val = _split(sorted(fail_eps))
    return succ_train, succ_val, fail_train, fail_val


# --------------------------------------------------------------------------- #
# Arm 2: SARSA-Q / FQE (2-net min ensemble, behavior-policy evaluation)       #
# --------------------------------------------------------------------------- #
class SarsaQ:
    def __init__(self, chunk_dim: int, device: str, n_nets: int = 2):
        self.device = device
        self.n_nets = n_nets
        self.q = nn.ModuleList(
            [QChunkNetwork(chunk_feature_dim=chunk_dim, hidden_dims=HIDDEN_DIMS) for _ in range(n_nets)]
        ).to(device)
        self.q_target = nn.ModuleList(
            [QChunkNetwork(chunk_feature_dim=chunk_dim, hidden_dims=HIDDEN_DIMS) for _ in range(n_nets)]
        ).to(device)
        self.q_target.load_state_dict(self.q.state_dict())
        for p in self.q_target.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.AdamW(self.q.parameters(), lr=Q_LR, weight_decay=WEIGHT_DECAY)
        self.disc_H = GAMMA**ACTION_HORIZON

    def _stack(self, mods, feat):
        return torch.stack([m(feat) for m in mods], dim=0)  # (K,B,1)

    def update(self, q_feat, next_q_feat, rewards, dones):
        with torch.no_grad():
            q_next = self._stack(self.q_target, next_q_feat).min(dim=0).values  # (B,1)
            target = rewards + self.disc_H * (1.0 - dones) * q_next
        q_pred = self._stack(self.q, q_feat)  # (K,B,1)
        loss = nn.functional.mse_loss(q_pred, target.expand_as(q_pred), reduction="none").mean(dim=(1, 2)).sum()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), GRAD_CLIP)
        self.opt.step()
        with torch.no_grad():
            tau = TARGET_POLYAK
            for tgt, src in zip(self.q_target.parameters(), self.q.parameters()):
                tgt.data.mul_(1.0 - tau).add_(src.data, alpha=tau)
            td = (q_pred.detach() - target.expand_as(q_pred)).abs().mean()
        return {"q_loss": float(loss.detach().item()), "td_abs": float(td.item()),
                "target_mean": float(target.mean().item()), "q_mean": float(q_pred.detach().mean().item())}

    @torch.no_grad()
    def predict(self, q_feat, batch=8192):
        out = []
        for m in self.q:
            m.eval()
        for s in range(0, q_feat.shape[0], batch):
            qf = q_feat[s:s + batch]
            out.append(self._stack(self.q, qf).min(dim=0).values.squeeze(-1).cpu())
        return torch.cat(out).numpy()


# --------------------------------------------------------------------------- #
# Arm 3: MC supervised regression (reuse P0 architecture)                     #
# --------------------------------------------------------------------------- #
class _MLPReg(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mc_head(make_head, feats, labels, train_idx, *, steps, batch, lr, device, tag):
    head = make_head().to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    head.train()
    n = train_idx.numel()
    for step in range(steps):
        sel = train_idx[torch.randint(0, n, (batch,), device=device)]
        pred = head(feats.index_select(0, sel))
        loss = nn.functional.mse_loss(pred, labels.index_select(0, sel))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), GRAD_CLIP)
        opt.step()
        if step % max(1, steps // 5) == 0 or step == steps - 1:
            print(f"[p1][mc:{tag}] step={step:6d} mse={loss.item():.5f}")
    head.eval()
    return head


@torch.no_grad()
def predict_head(head, feats, batch=8192):
    out = []
    head.eval()
    for s in range(0, feats.shape[0], batch):
        out.append(head(feats[s:s + batch]).cpu())
    return torch.cat(out).numpy()


# --------------------------------------------------------------------------- #
# Eval helpers                                                                #
# --------------------------------------------------------------------------- #
def adjacent_jump(values, ep, st, eps_set):
    jumps = []
    for e in eps_set:
        m = ep == e
        if m.sum() < 2:
            continue
        order = np.argsort(st[m])
        v = values[m][order]
        jumps.extend(np.abs(np.diff(v)).tolist())
    if not jumps:
        return float("nan"), float("nan"), 0
    a = np.asarray(jumps)
    return float(a.mean()), float(a.max()), len(jumps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--iql-warmup-steps", type=int, default=20000)
    ap.add_argument("--iql-full-steps", type=int, default=10000)
    ap.add_argument("--sarsa-steps", type=int, default=25000)
    ap.add_argument("--mc-steps", type=int, default=25000)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="tiny step counts for a wiring smoke test")
    args = ap.parse_args()

    if args.smoke:
        args.iql_warmup_steps = 200
        args.iql_full_steps = 100
        args.sarsa_steps = 200
        args.mc_steps = 200

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------- #
    # Load cached features (NO encode, NO buffer reload)                #
    # ----------------------------------------------------------------- #
    print(f"[p1] loading cached features {FEAT_PATH}")
    blob = torch.load(FEAT_PATH, map_location="cpu", weights_only=False)
    q_feat = blob["q_chunk_feature"].float()  # (R, D_chunk)
    v_feat = blob["v_state_feature"].float()  # (R, D_state)
    q = build_step_quantities(blob)
    G, ep, st, succ = q["G"], q["ep"], q["st"], q["succ"]
    next_row, have_next, r_chunk, done = q["next_row"], q["have_next"], q["r_chunk"], q["done"]
    n = len(G)
    assert q_feat.shape[0] == n and v_feat.shape[0] == n, "feature/row mismatch"

    # ----------------------------------------------------------------- #
    # SANITY: telescoping reward + next_row map                         #
    # ----------------------------------------------------------------- #
    print("[p1] ===== r_chunk / next_row SANITY =====")
    exp_r = -OUTPUT_REWARD_COEF * sum(GAMMA**i for i in range(ACTION_HORIZON))
    print(f"[p1] expected n-step reward for an all-(-0.1)-step chunk: {exp_r:.6f}")
    sanity = {"expected_nonsuccess_r_chunk": exp_r, "examples": []}
    n_match = 0
    n_checked = 0
    # Verify on pre-success/fail chunks (no success frame => done==0 => reward == exp_r)
    nonsucc_chunk = have_next & (done == 0)
    for r in (list(np.where(succ & nonsucc_chunk)[0][:3]) + list(np.where((~succ) & nonsucc_chunk)[0][:3])):
        match = bool(np.isclose(r_chunk[r], exp_r, atol=1e-4))
        n_checked += 1
        n_match += int(match)
        ex = {"row": int(r), "ep": int(ep[r]), "step": int(st[r]), "succ": bool(succ[r]),
              "G": float(G[r]), "G_next": float(G[next_row[r]]), "r_chunk": float(r_chunk[r]),
              "matches_expected": match}
        sanity["examples"].append(ex)
        print(f"[p1]   row={r} ep={int(ep[r])} step={int(st[r])} succ={bool(succ[r])} "
              f"G={G[r]:+.4f} G_next={G[next_row[r]]:+.4f} r_chunk={r_chunk[r]:.6f} match={match}")
    # global check across ALL non-success chunks with valid next
    glob = r_chunk[nonsucc_chunk & have_next]
    glob = glob[~np.isnan(glob)]
    max_dev = float(np.max(np.abs(glob - exp_r))) if glob.size else float("nan")
    print(f"[p1] max |r_chunk - expected| over all non-success-frame chunks: {max_dev:.3e} (n={glob.size})")
    sanity["max_abs_dev_nonsuccess_chunks"] = max_dev
    sanity["n_nonsuccess_chunks_checked"] = int(glob.size)
    if not np.isnan(max_dev) and max_dev > 1e-3:
        print("[p1] FATAL: telescoping identity violated — STOPPING.")
        return
    print("[p1] telescoping identity holds.")

    # ----------------------------------------------------------------- #
    # Missing-next handling + anomaly report                            #
    # ----------------------------------------------------------------- #
    miss = ~have_next
    drop_fail = int((miss & (~succ)).sum())
    presucc_done0_missing = int((miss & succ & (done == 0)).sum())
    miss_succ_done1 = int((miss & succ & (done == 1)).sum())
    anomaly_rows = np.where(miss & succ & (done == 0))[0]
    anomalies = [{"row": int(r), "ep": int(ep[r]), "step": int(st[r]),
                  "G": float(G[r]), "t_s": int(q["t_s"][int(ep[r])])} for r in anomaly_rows]
    print(f"[p1] missing-next: total={int(miss.sum())} | fail-tail(drop)={drop_fail} "
          f"| success done==1(no bootstrap needed)={miss_succ_done1} "
          f"| PRE-success done==0 MISSING(anomaly, drop)={presucc_done0_missing}")
    if anomalies:
        print("[p1] ANOMALY pre-success done==0 windows w/o next_row (treated like fail tails, dropped):")
        for a in anomalies[:12]:
            print(f"       row={a['row']} ep={a['ep']} step={a['step']} G={a['G']:+.4f} t_s={a['t_s']}")

    # Bootstrap training set: rows that need a target.
    #   - done==1 rows: target is r only (no bootstrap) -> keep even if next missing.
    #   - done==0 rows: need next_row -> require have_next, else DROP.
    boot_ok = (done == 1) | ((done == 0) & have_next)
    # Safe next index (point missing to self; masked out by done==1 in target anyway).
    next_safe = np.where(have_next, next_row, np.arange(n))
    n_dropped = int((~boot_ok).sum())
    print(f"[p1] bootstrap training rows: kept={int(boot_ok.sum())} dropped={n_dropped} "
          f"(={drop_fail} fail-tail + {presucc_done0_missing} success-anomaly)")

    # ----------------------------------------------------------------- #
    # Reproduce P0 split (seed 42, success-then-fail shared RNG)         #
    # ----------------------------------------------------------------- #
    succ_eps = q["succ_eps"]
    fail_eps = sorted({int(e) for e in ep[~succ]})
    succ_train, succ_val, fail_train, fail_val = reproduce_p0_split(
        succ_eps, fail_eps, args.seed, args.val_frac)
    val_eps = succ_val | fail_val
    is_val_row = np.array([int(e) in val_eps for e in ep], dtype=bool)
    print(f"[p1] split: succ train/val={len(succ_train)}/{len(succ_val)} "
          f"fail train/val={len(fail_train)}/{len(fail_val)}")
    print(f"[p1] success val demos: {sorted(succ_val)}")
    print(f"[p1] fail val demos:    {sorted(fail_val)}")

    # train rows = non-val rows
    is_train_row = ~is_val_row

    # ----------------------------------------------------------------- #
    # Move tensors to device                                            #
    # ----------------------------------------------------------------- #
    q_feat_d = q_feat.to(device)
    v_feat_d = v_feat.to(device)
    r_chunk_t = torch.from_numpy(np.nan_to_num(r_chunk, nan=0.0).astype(np.float32)).to(device).unsqueeze(1)
    done_t = torch.from_numpy(done.astype(np.float32)).to(device).unsqueeze(1)
    G_t = torch.from_numpy(G.astype(np.float32)).to(device)
    next_safe_t = torch.from_numpy(next_safe).to(device)

    # Training-row indices for the bootstrap arms (train demos AND bootstrap-ok).
    boot_train_idx = torch.from_numpy(np.where(is_train_row & boot_ok)[0]).to(device)
    # MC arm uses all train rows (no bootstrap requirement).
    mc_train_idx = torch.from_numpy(np.where(is_train_row)[0]).to(device)
    print(f"[p1] boot_train_idx={boot_train_idx.numel()}  mc_train_idx={mc_train_idx.numel()}")

    chunk_dim = q_feat.shape[1]
    state_dim = v_feat.shape[1]

    def sample_boot(bs):
        sel = boot_train_idx[torch.randint(0, boot_train_idx.numel(), (bs,), device=device)]
        nxt = next_safe_t.index_select(0, sel)
        return sel, nxt

    # ================================================================= #
    # ARM 1: iql_v  (reuse production IQLLearner)                        #
    # ================================================================= #
    print("[p1] ===== ARM 1: iql_v (production warmup reproduction) =====")
    iql_cfg = IQLConfig(
        action_horizon=ACTION_HORIZON, discount=GAMMA, expectile_tau=EXPECTILE_TAU,
        q_lr=Q_LR, v_lr=V_LR, target_polyak=TARGET_POLYAK, hidden_dims=HIDDEN_DIMS,
        q_ensemble_size=Q_ENSEMBLE_SIZE, v_subset_size=V_SUBSET_SIZE,
        grad_clip_norm=GRAD_CLIP, weight_decay=WEIGHT_DECAY, device=device,
        reward_mode="-1/0", output_reward_coef=OUTPUT_REWARD_COEF, disc_reward_coef=0.0,
    )
    iql = IQLLearner(cfg=iql_cfg, state_feature_dim=state_dim, chunk_feature_dim=chunk_dim, action_dim=7)

    def make_step_batch(sel, nxt):
        bs = sel.numel()
        return IQLStepBatch(
            q_chunk_feature=q_feat_d.index_select(0, sel),
            v_state_feature=v_feat_d.index_select(0, sel),
            next_v_state_feature=v_feat_d.index_select(0, nxt),
            action_chunk=torch.zeros(bs, ACTION_HORIZON, 7, device=device),
            rewards=r_chunk_t.index_select(0, sel),
            dones=done_t.index_select(0, sel),
            is_online=torch.zeros(bs, 1, device=device),
            is_intervention=torch.zeros(bs, 1, device=device),
        )

    iql_curve = []
    for step in range(args.iql_warmup_steps):
        sel, nxt = sample_boot(args.batch)
        m = iql.warmup_value_only(make_step_batch(sel, nxt))
        if step % max(1, args.iql_warmup_steps // 5) == 0 or step == args.iql_warmup_steps - 1:
            iql_curve.append({"phase": "warmup", "step": step, **m})
            print(f"[p1][iql_v:warmup] step={step:6d} v_loss={m['v_loss']:.5f} "
                  f"v_mean={m['v_mean']:+.4f} target_mean={m['target_mean']:+.4f}")
    for step in range(args.iql_full_steps):
        sel, nxt = sample_boot(args.batch)
        m = iql.update(make_step_batch(sel, nxt))
        if step % max(1, args.iql_full_steps // 5) == 0 or step == args.iql_full_steps - 1:
            iql_curve.append({"phase": "full", "step": step, "v_loss": m["v_loss"], "q_loss": m["q_loss"],
                              "v_mean": m["v_mean"], "target_q_mean": m["target_q_mean"],
                              "td_error_abs_mean": m["td_error_abs_mean"]})
            print(f"[p1][iql_v:full] step={step:6d} q_loss={m['q_loss']:.4f} v_loss={m['v_loss']:.5f} "
                  f"v_mean={m['v_mean']:+.4f} target_q_mean={m['target_q_mean']:+.4f} "
                  f"td={m['td_error_abs_mean']:.4f}")

    @torch.no_grad()
    def iql_v_predict(feat, batch=8192):
        out = []
        iql.v.eval()
        for s in range(0, feat.shape[0], batch):
            out.append(iql.v(feat[s:s + batch]).squeeze(-1).cpu())
        return torch.cat(out).numpy()

    @torch.no_grad()
    def iql_q_predict(feat, batch=8192):
        out = []
        for s in range(0, feat.shape[0], batch):
            qf = feat[s:s + batch]
            qs = torch.stack([qn(qf) for qn in iql.q_ensemble], dim=0).mean(dim=0)
            out.append(qs.squeeze(-1).cpu())
        return torch.cat(out).numpy()

    iql_v_val = iql_v_predict(v_feat_d)   # V(s_t) for scoring the behavior trajectory
    iql_q_val = iql_q_predict(q_feat_d)

    # ================================================================= #
    # ARM 2: sarsa_q (FQE)                                               #
    # ================================================================= #
    print("[p1] ===== ARM 2: sarsa_q (FQE behavior evaluation) =====")
    sarsa = SarsaQ(chunk_dim, device, n_nets=2)
    sarsa_curve = []
    for step in range(args.sarsa_steps):
        sel, nxt = sample_boot(args.batch)
        m = sarsa.update(q_feat_d.index_select(0, sel), q_feat_d.index_select(0, nxt),
                         r_chunk_t.index_select(0, sel), done_t.index_select(0, sel))
        if step % max(1, args.sarsa_steps // 5) == 0 or step == args.sarsa_steps - 1:
            sarsa_curve.append({"step": step, **m})
            print(f"[p1][sarsa_q] step={step:6d} q_loss={m['q_loss']:.4f} "
                  f"q_mean={m['q_mean']:+.4f} target_mean={m['target_mean']:+.4f} td={m['td_abs']:.4f}")
    sarsa_q_val = sarsa.predict(q_feat_d)  # Q(s_t, a_t^recorded)

    # ================================================================= #
    # ARM 3: mc_return (supervised regression to G_label)               #
    # ================================================================= #
    print("[p1] ===== ARM 3: mc_return (supervised V/Q regression) =====")
    mc_v_head = train_mc_head(
        lambda: _MLPReg(VNetwork(context_dim=state_dim, hidden_dims=HIDDEN_DIMS)),
        v_feat_d, G_t, mc_train_idx, steps=args.mc_steps, batch=args.batch, lr=V_LR, device=device, tag="V")
    mc_q_head = train_mc_head(
        lambda: _MLPReg(QChunkNetwork(chunk_feature_dim=chunk_dim, hidden_dims=HIDDEN_DIMS)),
        q_feat_d, G_t, mc_train_idx, steps=args.mc_steps, batch=args.batch, lr=Q_LR, device=device, tag="Q")
    mc_v_val = predict_head(mc_v_head, v_feat_d)
    mc_q_val = predict_head(mc_q_head, q_feat_d)

    # ================================================================= #
    # EVALUATION                                                        #
    # ================================================================= #
    # The "value of the behavior trajectory" per arm:
    #   iql_v   -> V(s_t)
    #   sarsa_q -> Q(s_t, a_t^recorded)
    #   mc_return -> V(s_t) (primary) and Q(s_t,a_t) (reported too)
    arm_value = {
        "iql_v": iql_v_val,
        "sarsa_q": sarsa_q_val,
        "mc_return": mc_v_val,
    }
    arm_value_extra = {"iql_v_Q": iql_q_val, "mc_return_Q": mc_q_val}

    succ_val_mask = succ & is_val_row
    fail_val_mask = (~succ) & is_val_row

    def mae_vs_G(vals, mask):
        if mask.sum() == 0:
            return float("nan")
        return float(np.abs(vals[mask] - G[mask]).mean())

    results = {
        "config": {
            "device": device, "seed": args.seed, "val_frac": args.val_frac,
            "iql_warmup_steps": args.iql_warmup_steps, "iql_full_steps": args.iql_full_steps,
            "sarsa_steps": args.sarsa_steps, "mc_steps": args.mc_steps, "batch": args.batch,
            "gamma": GAMMA, "action_horizon": ACTION_HORIZON, "expectile_tau": EXPECTILE_TAU,
            "q_ensemble_size": Q_ENSEMBLE_SIZE, "v_subset_size": V_SUBSET_SIZE,
            "hidden_dims": list(HIDDEN_DIMS), "target_polyak": TARGET_POLYAK,
            "output_reward_coef": OUTPUT_REWARD_COEF,
        },
        "sanity": sanity,
        "missing_next": {
            "total": int(miss.sum()), "fail_tail_dropped": drop_fail,
            "success_done1_no_bootstrap_needed": miss_succ_done1,
            "presuccess_done0_missing_anomaly": presucc_done0_missing,
            "anomaly_rows": anomalies,
            "total_bootstrap_rows_dropped": n_dropped,
            "bootstrap_rows_kept": int(boot_ok.sum()),
        },
        "split": {
            "success_train": sorted(succ_train), "success_val": sorted(succ_val),
            "fail_train": sorted(fail_train), "fail_val": sorted(fail_val),
        },
        "done_count": int(done.sum()),
    }

    # --- success-val MAE vs G_label (the KEY comparison; G_label IS ground truth here) ---
    print("[p1] ===== SUCCESS-VAL MAE vs G_label (ground truth on success) =====")
    succ_mae = {}
    for arm, vals in arm_value.items():
        succ_mae[arm] = mae_vs_G(vals, succ_val_mask)
        print(f"[p1]   {arm:<10} success-val MAE = {succ_mae[arm]:.4f}")
    # extra Q variants
    succ_mae_extra = {k: mae_vs_G(v, succ_val_mask) for k, v in arm_value_extra.items()}
    fail_mae = {arm: mae_vs_G(vals, fail_val_mask) for arm, vals in arm_value.items()}

    # --- mean value on fail-val (MC NOT ground truth here) ---
    print("[p1] ===== FAIL-VAL mean value (MC is truncation-biased, NOT ground truth) =====")
    fail_mean = {}
    for arm, vals in arm_value.items():
        fail_mean[arm] = float(vals[fail_val_mask].mean()) if fail_val_mask.sum() else float("nan")
        print(f"[p1]   {arm:<10} fail-val mean value = {fail_mean[arm]:+.4f}")
    succ_mean = {arm: float(vals[succ_val_mask].mean()) for arm, vals in arm_value.items()}

    # --- adjacent-window jump ---
    print("[p1] ===== ADJACENT-WINDOW value jump (val demos) =====")
    jump = {}
    for arm, vals in arm_value.items():
        sj = adjacent_jump(vals, ep, st, succ_val)
        fj = adjacent_jump(vals, ep, st, fail_val)
        jump[arm] = {"success_val": {"mean": sj[0], "max": sj[1], "n": sj[2]},
                     "fail_val": {"mean": fj[0], "max": fj[1], "n": fj[2]}}
        print(f"[p1]   {arm:<10} succ jump mean/max={sj[0]:.4f}/{sj[1]:.4f}  "
              f"fail jump mean/max={fj[0]:.4f}/{fj[1]:.4f}")

    results["success_val_mae_vs_G"] = succ_mae
    results["success_val_mae_vs_G_extra"] = succ_mae_extra
    results["fail_val_mae_vs_G_NOTE"] = "MC is truncation-biased on fail; MAE-vs-G not ground truth"
    results["fail_val_mae_vs_G"] = fail_mae
    results["fail_val_mean_value"] = fail_mean
    results["success_val_mean_value"] = succ_mean
    results["adjacent_jump"] = jump

    # --- DECISIVE: pre-success-window pessimism table ---
    # For a couple success-val demos, print the LAST few PRE-success windows
    # (done==0, step < t_s): step, G_label, each arm value, and td_target for the
    # bootstrap arms (r + gamma^H (1-done) V_target(s')  /  Q_target(s',a')).
    print("[p1] ===== PRE-SUCCESS-WINDOW PESSIMISM (decisive) =====")
    disc_H = GAMMA**ACTION_HORIZON

    @torch.no_grad()
    def iql_td_target(row):
        nr = next_row[row]
        if nr < 0:
            return float("nan")
        vt = iql.target_v(v_feat_d[nr:nr + 1]).item()
        return float(r_chunk[row] + disc_H * (1.0 - done[row]) * vt)

    @torch.no_grad()
    def sarsa_td_target(row):
        nr = next_row[row]
        if nr < 0:
            return float("nan")
        qf = q_feat_d[nr:nr + 1]
        qt = torch.stack([m(qf) for m in sarsa.q_target], dim=0).min(dim=0).values.item()
        return float(r_chunk[row] + disc_H * (1.0 - done[row]) * qt)

    pessimism = {}
    pessimism_demos = sorted(succ_val)[:2]
    for e in pessimism_demos:
        m = np.where(ep == e)[0]
        steps_e = st[m]
        order = np.argsort(steps_e)
        rows_sorted = m[order]
        ts_e = q["t_s"][e]
        # last few PRE-success windows: done==0 rows, take last 6
        pre_rows = [r for r in rows_sorted if done[r] == 0]
        pre_rows = pre_rows[-6:]
        rows_out = []
        print(f"[p1] success-val demo ep={e} (t_s={ts_e}):")
        print(f"[p1]   {'step':>5}{'G_label':>10}{'iql_v':>10}{'iql_td':>10}"
              f"{'sarsa_q':>10}{'sarsa_td':>10}{'mc_V':>10}")
        for r in pre_rows:
            iql_td = iql_td_target(r)
            sa_td = sarsa_td_target(r)
            row_d = {"step": int(st[r]), "G_label": float(G[r]),
                     "iql_v": float(iql_v_val[r]), "iql_td_target": iql_td,
                     "sarsa_q": float(sarsa_q_val[r]), "sarsa_td_target": sa_td,
                     "mc_return": float(mc_v_val[r])}
            rows_out.append(row_d)
            print(f"[p1]   {int(st[r]):>5}{G[r]:>10.3f}{iql_v_val[r]:>10.3f}{iql_td:>10.3f}"
                  f"{sarsa_q_val[r]:>10.3f}{sa_td:>10.3f}{mc_v_val[r]:>10.3f}")
        pessimism[f"ep{e}"] = {"t_s": int(ts_e), "windows": rows_out}
    results["pre_success_pessimism"] = pessimism

    # ----------------------------------------------------------------- #
    # Per-demo CSV + PNG (2 success + 2 fail val demos)                 #
    # ----------------------------------------------------------------- #
    dump_succ = sorted(succ_val)[:2]
    dump_fail = sorted(fail_val)[:2]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        have_mpl = True
    except Exception as exc:  # pragma: no cover
        print(f"[p1] matplotlib unavailable ({exc}); CSV only")
        have_mpl = False

    for tag, e in [("success", x) for x in dump_succ] + [("fail", x) for x in dump_fail]:
        m = ep == e
        order = np.argsort(st[m])
        steps_e = st[m][order]
        gl = G[m][order]
        iv = arm_value["iql_v"][m][order]
        sq = arm_value["sarsa_q"][m][order]
        mc = arm_value["mc_return"][m][order]
        csv_path = OUT_DIR / f"demo_{tag}_ep{e}.csv"
        with csv_path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["episode_step", "G_label", "iql_v", "sarsa_q", "mc_return"])
            for s, g, a, b2, c in zip(steps_e, gl, iv, sq, mc):
                w.writerow([int(s), f"{g:.6f}", f"{a:.6f}", f"{b2:.6f}", f"{c:.6f}"])
        if have_mpl:
            plt.figure(figsize=(9, 4))
            plt.plot(steps_e, gl, label="G_label (MC truth)", lw=2.5, color="k")
            plt.plot(steps_e, iv, label="iql_v: V(s)", lw=1.5)
            plt.plot(steps_e, sq, label="sarsa_q: Q(s,a)", lw=1.5)
            plt.plot(steps_e, mc, label="mc_return: V(s)", lw=1.5, ls="--")
            plt.xlabel("episode_step")
            plt.ylabel("value / return")
            plt.title(f"P1 target-mode — {tag} val demo ep={e}")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(OUT_DIR / f"demo_{tag}_ep{e}.png", dpi=110)
            plt.close()
    print(f"[p1] dumped CSV/PNG: success={dump_succ} fail={dump_fail} -> {OUT_DIR}")

    results["dump_demos"] = {"success": dump_succ, "fail": dump_fail}
    results["train_curves"] = {"iql_v": iql_curve, "sarsa_q": sarsa_curve}

    out_json = OUT_DIR / "p1_results.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"[p1] wrote results -> {out_json}")
    print("[p1] DONE")


if __name__ == "__main__":
    main()
