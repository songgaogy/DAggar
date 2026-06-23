"""P1.5 diagnostic: TRUE held-out reproduction + data-size sweep.

Splits the leading hypothesis from `OFFLINE_RL_DEBUG_PROGRESS.md` §6:
  (a) data-limited overfit  — held-out MAE FALLS as train demos grow.
  (b) representation ceiling — held-out MAE STUCK at a floor (even the MC oracle),
      more data won't help; the frozen encoder can't separate near-success states
      on an independent recording.

Two things at once:
  1. TRUE held-out reproduction. P0/P1 only ever evaluated an IN-DISTRIBUTION val
     (demos cut from the SAME 50-demo recording). The production symptom
     (success Q-TD MAE ~0.89, success adjacent-V jump ~0.53) was measured on the
     SEPARATE 2026-06-18 recording (`success_rollout-val` / `fail_rollout-val`),
     which neither P0 nor P1 has touched. This script encodes those held-out
     demos and evaluates on them — does the production gap reproduce?
  2. Data-size sweep. Train heads on the first N success + N fail demos from P0's
     cached train features (N in {10,20,30,40,50}; nested prefixes so the curve is
     monotone-in-data), evaluate EVERY point on the SAME true held-out set.

Design (per the agreed plan):
  - "Reuse P0/P1 heads" = retrain the SAME architectures + SAME recipes + seed 42
    DETERMINISTICALLY (P0/P1 never saved head weights). Train-side features are
    NOT re-encoded — they are subset by demo from P0's `preencoded_features.pt`
    (which already covers all 50+50 train demos). Only the held-out recording is
    freshly encoded, ONCE.
  - All three target-mode arms (iql_v / sarsa_q / mc_return), reusing the exact
    P1 implementations (imported, not reimplemented).

Held-out features come from a transitions `.pt` assembled by the production warmup
(0 training steps + save_data) over the `-val` splits — see the module docstring
of `OFFLINE_RL_DEBUG_PROGRESS.md` and the assemble command in the runbook below.

Run (module-style):
    cd <repo-root> && MUJOCO_GL=egl PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
      python -m robosuite.pipeline.algorithms.q_learning.utils.p1_5_heldout_sweep \
      --device cuda:0
    # reuse a prior held-out encoding:
    ... --reuse-heldout-features outputs/dipole_rl-debug/p1_5_heldout/heldout_features.pt
    # wiring smoke (tiny steps):
    ... --smoke
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from robosuite.pipeline.algorithms.discriminator.encoder import SharedDynamicsEncoder
from robosuite.pipeline.algorithms.flow_dagger.common import (
    FlowAugmentationConfig,
    ReplayBufferConfig,
)
from robosuite.pipeline.algorithms.flow_dagger.replay_buffer import FlowDaggerReplayBuffer
from robosuite.pipeline.algorithms.q_learning.common import IQLConfig, IQLStepBatch
from robosuite.pipeline.algorithms.q_learning.iql import IQLLearner
from robosuite.pipeline.algorithms.q_learning.networks import QChunkNetwork, VNetwork
from robosuite.pipeline.algorithms.q_learning.replay import IQLReplayBuffer
from robosuite.pipeline.algorithms.q_learning.warmup import _freeze_post_success_tail

# Reuse P0's label/encode helpers and P1's training arms verbatim (no duplication).
from robosuite.pipeline.algorithms.q_learning.utils.mc_return_probe import (
    _compute_return_to_go,
    _episode_index_of,
    _episode_step_of,
    _is_success_frame,
)
from robosuite.pipeline.algorithms.q_learning.utils.p1_target_mode import (
    SarsaQ,
    _MLPReg,
    adjacent_jump,
    build_step_quantities,
    predict_head,
    train_mc_head,
)

# ---- Fixed hyperparameters (mirror train_dipole_rl.yaml q_learning.config) ----
OUTPUT_REWARD_COEF = 0.1
GAMMA = 0.99
ACTION_HORIZON = 8
HIDDEN_DIMS = (512, 512)
EXPECTILE_TAU = 0.7
Q_ENSEMBLE_SIZE = 10
V_SUBSET_SIZE = 2
Q_LR = V_LR = 3e-4
TARGET_POLYAK = 0.005
GRAD_CLIP = 2.0
WEIGHT_DECAY = 1e-6
CAMERA_NAMES = ["agentview", "robot0_robotview", "robot0_eye_in_hand"]
SWEEP_SIZES = (10, 20, 30, 40, 50)

REPO_ROOT = Path(__file__).resolve().parents[5]
P0_FEAT_PATH = REPO_ROOT / "outputs/dipole_rl-debug/p0_mc_probe/preencoded_features.pt"
HELDOUT_BUFFER = REPO_ROOT / "data/PickPlaceCereal/offline_data_heldout/iql_offline_transitions.pt"
HELDOUT_META = REPO_ROOT / "data/PickPlaceCereal/offline_data_heldout/iql_offline_transitions.meta.json"
NNPU_CKPT = (
    REPO_ROOT
    / "checkpoints/dyn_disc/pu_bce_eval_robosuite/run_20260619_194127_PickPlaceCereal"
    / "checkpoints/pu_bce_head.pth"
)
OUT_DIR = REPO_ROOT / "outputs/dipole_rl-debug/p1_5_heldout"


# --------------------------------------------------------------------------- #
# Held-out encoding (mirrors P0's encode block; needs NO robosuite env)        #
# --------------------------------------------------------------------------- #
def encode_heldout(buffer_path: Path, device: str, encode_batch: int) -> dict:
    """Load the assembled held-out transitions, apply freeze_post_success, encode
    every valid window once with the frozen SharedDynamicsEncoder, and return a
    blob with the SAME schema as P0's preencoded_features.pt (so
    build_step_quantities works unchanged)."""
    meta = json.loads(HELDOUT_META.read_text()) if HELDOUT_META.exists() else {}
    img_height = int(meta.get("image_size", meta.get("img_height", 84)))
    if meta:
        assert int(meta["action_horizon"]) == ACTION_HORIZON, "held-out action_horizon mismatch"
        assert list(meta["camera_names"]) == CAMERA_NAMES, "held-out camera_names mismatch"
        n_cap = max(50000, int(meta.get("n_transitions", 40000)) + 10)
    else:
        n_cap = 60000

    buffer = FlowDaggerReplayBuffer(
        config=ReplayBufferConfig(capacity=n_cap, batch_size=256),
        name="p1_5_heldout",
        camera_names=CAMERA_NAMES,
        action_horizon=ACTION_HORIZON,
        image_size=img_height,
        augmentation_config=FlowAugmentationConfig(),
    )
    print(f"[p1.5] loading held-out buffer {buffer_path}")
    buffer.load(buffer_path)
    print(f"[p1.5] loaded {len(buffer)} transitions, valid_starts={buffer.num_valid_sequences()}")

    with buffer._lock:  # noqa: SLF001
        n_frozen = _freeze_post_success_tail(buffer._storage)  # noqa: SLF001
        buffer._rebuild_valid_start_cache_locked()  # noqa: SLF001
    print(f"[p1.5] freeze_post_success: {n_frozen} frames frozen to absorbing anchor")

    storage = list(buffer._storage)  # noqa: SLF001
    with buffer._lock:  # noqa: SLF001
        valid_starts = list(buffer._get_valid_start_indices_locked())  # noqa: SLF001
    valid_starts_arr = np.asarray(valid_starts, dtype=np.int64)

    g_all = _compute_return_to_go(storage)
    G_label = g_all[valid_starts_arr].astype(np.float32)
    ep_index_per_row = np.asarray([_episode_index_of(storage[s]) for s in valid_starts], dtype=np.int64)
    ep_step_per_row = np.asarray([_episode_step_of(storage[s]) for s in valid_starts], dtype=np.int64)

    demo_is_success: dict[int, bool] = {}
    for t in storage:
        ep = _episode_index_of(t)
        if _is_success_frame(t):
            demo_is_success[ep] = True
        demo_is_success.setdefault(ep, False)
    is_success_per_row = np.asarray(
        [demo_is_success.get(int(ep), False) for ep in ep_index_per_row], dtype=bool
    )
    n_succ = int(is_success_per_row.sum())
    n_fail = int((~is_success_per_row).sum())
    n_succ_demos = len({int(e) for e, s in demo_is_success.items() if s})
    n_fail_demos = len({int(e) for e, s in demo_is_success.items() if not s})
    print(f"[p1.5] held-out rows: success={n_succ} fail={n_fail} "
          f"(success demos={n_succ_demos} fail demos={n_fail_demos})")

    print(f"[p1.5] building encoder from {NNPU_CKPT} on {device}")
    encoder = SharedDynamicsEncoder(
        nnpu_ckpt_path=str(NNPU_CKPT), encoder_ckpt=None, device=device, camera_to_view={},
    )
    encoder.bind_policy_cameras(CAMERA_NAMES)
    iql_cfg = IQLConfig(action_horizon=ACTION_HORIZON, device=device, disc_reward_coef=0.0,
                        output_reward_coef=OUTPUT_REWARD_COEF, discount=GAMMA, reward_mode="-1/0")
    replay = IQLReplayBuffer(base_buffer=buffer, cfg=iql_cfg)
    print(f"[p1.5] preencoding {len(valid_starts)} held-out windows (encode_batch={encode_batch})...")
    cache = replay.preencode_step_cache(
        encoder=encoder, discriminator=None, device=device,
        encode_batch_size=encode_batch, cache_device="cpu", progress_desc="[p1.5] preencode",
    )
    q_feat = cache.q_chunk_feature
    v_feat = cache.v_state_feature
    assert q_feat.shape[0] == len(valid_starts), "held-out cache row mismatch"
    print(f"[p1.5] held-out features: q_chunk={tuple(q_feat.shape)} v_state={tuple(v_feat.shape)}")

    return {
        "q_chunk_feature": q_feat,
        "v_state_feature": v_feat,
        "valid_starts": valid_starts_arr,
        "episode_index_per_row": ep_index_per_row,
        "episode_step_per_row": ep_step_per_row,
        "G_label": G_label,
        "is_success_per_row": is_success_per_row,
        "meta": {"n_frozen": int(n_frozen), "buffer_path": str(buffer_path),
                 "nnpu_ckpt": str(NNPU_CKPT)},
    }


# --------------------------------------------------------------------------- #
# IQL arm (the only arm whose loop isn't a single imported call)              #
# --------------------------------------------------------------------------- #
def train_iql_arm(train_blob_q: dict, q_feat_d, v_feat_d, r_chunk_t, done_t, next_safe_t,
                  boot_train_idx, *, device, batch, warmup_steps, full_steps,
                  state_dim, chunk_dim):
    """Faithful production-warmup reproduction (reuse IQLLearner/IQLConfig/IQLStepBatch).
    boot_train_idx restricts sampling to the selected demos' bootstrap-ok rows."""
    iql_cfg = IQLConfig(
        action_horizon=ACTION_HORIZON, discount=GAMMA, expectile_tau=EXPECTILE_TAU,
        q_lr=Q_LR, v_lr=V_LR, target_polyak=TARGET_POLYAK, hidden_dims=HIDDEN_DIMS,
        q_ensemble_size=Q_ENSEMBLE_SIZE, v_subset_size=V_SUBSET_SIZE,
        grad_clip_norm=GRAD_CLIP, weight_decay=WEIGHT_DECAY, device=device,
        reward_mode="-1/0", output_reward_coef=OUTPUT_REWARD_COEF, disc_reward_coef=0.0,
    )
    iql = IQLLearner(cfg=iql_cfg, state_feature_dim=state_dim, chunk_feature_dim=chunk_dim, action_dim=7)
    n = boot_train_idx.numel()

    def sample(bs):
        sel = boot_train_idx[torch.randint(0, n, (bs,), device=device)]
        nxt = next_safe_t.index_select(0, sel)
        return sel, nxt

    def make_batch(sel, nxt):
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

    for step in range(warmup_steps):
        sel, nxt = sample(batch)
        iql.warmup_value_only(make_batch(sel, nxt))
    for step in range(full_steps):
        sel, nxt = sample(batch)
        iql.update(make_batch(sel, nxt))
    return iql


@torch.no_grad()
def iql_predict_v(iql, feat, batch=8192):
    out = []
    iql.v.eval()
    for s in range(0, feat.shape[0], batch):
        out.append(iql.v(feat[s:s + batch]).squeeze(-1).cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def iql_predict_q(iql, feat, batch=8192):
    out = []
    for s in range(0, feat.shape[0], batch):
        qf = feat[s:s + batch]
        qs = torch.stack([qn(qf) for qn in iql.q_ensemble], dim=0).mean(dim=0)
        out.append(qs.squeeze(-1).cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def iql_qtd_mae(iql, ho, q_feat_d, v_feat_d, mask):
    """Production-style Q-TD MAE on held-out rows in `mask`:
    | Q_ensemble_mean(s,a) - [r + gamma^H (1-done) V_target(s')] |, rows with valid next."""
    G, ep, st = ho["G"], ho["ep"], ho["st"]
    next_row, have_next, r_chunk, done = ho["next_row"], ho["have_next"], ho["r_chunk"], ho["done"]
    disc_H = GAMMA ** ACTION_HORIZON
    rows = np.where(mask & have_next)[0]
    if rows.size == 0:
        return float("nan"), 0
    q_pred = torch.stack([qn(q_feat_d[rows]) for qn in iql.q_ensemble], dim=0).mean(dim=0).squeeze(-1).cpu().numpy()
    nxt = next_row[rows]
    vt = iql.target_v(v_feat_d[nxt]).squeeze(-1).cpu().numpy()
    td = r_chunk[rows] + disc_H * (1.0 - done[rows]) * vt
    return float(np.abs(q_pred - td).mean()), int(rows.size)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--encode-batch", type=int, default=32)
    ap.add_argument("--iql-warmup-steps", type=int, default=20000)
    ap.add_argument("--iql-full-steps", type=int, default=10000)
    ap.add_argument("--sarsa-steps", type=int, default=25000)
    ap.add_argument("--mc-steps", type=int, default=25000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--heldout-buffer", default=str(HELDOUT_BUFFER))
    ap.add_argument("--reuse-heldout-features", default="")
    ap.add_argument("--sweep", default="", help="comma sizes override, e.g. 10,30,50")
    ap.add_argument("--smoke", action="store_true")
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
    sweep_sizes = (tuple(int(x) for x in args.sweep.split(",")) if args.sweep else SWEEP_SIZES)

    # ---------------------------------------------------------------- #
    # 1. Held-out features (encode once, or reuse)                     #
    # ---------------------------------------------------------------- #
    ho_feat_path = OUT_DIR / "heldout_features.pt"
    if args.reuse_heldout_features and Path(args.reuse_heldout_features).exists():
        print(f"[p1.5] reusing held-out features {args.reuse_heldout_features}")
        ho_blob = torch.load(args.reuse_heldout_features, map_location="cpu", weights_only=False)
    elif ho_feat_path.exists() and not args.smoke:
        print(f"[p1.5] reusing held-out features {ho_feat_path}")
        ho_blob = torch.load(ho_feat_path, map_location="cpu", weights_only=False)
    else:
        ho_blob = encode_heldout(Path(args.heldout_buffer), device, args.encode_batch)
        torch.save(ho_blob, ho_feat_path)
        print(f"[p1.5] saved held-out features -> {ho_feat_path}")

    ho_q = ho_blob["q_chunk_feature"].float()
    ho_v = ho_blob["v_state_feature"].float()
    ho = build_step_quantities(ho_blob)  # G, ep, st, succ, next_row, have_next, r_chunk, done
    ho_succ = ho["succ"]
    ho_q_d = ho_q.to(device)
    ho_v_d = ho_v.to(device)
    ho_succ_mask = ho_succ
    ho_fail_mask = ~ho_succ
    print(f"[p1.5] held-out eval rows: success={int(ho_succ_mask.sum())} fail={int(ho_fail_mask.sum())}")

    # ---------------------------------------------------------------- #
    # 2. Train cache (P0) — full 50+50 demos; subset by demo per point #
    # ---------------------------------------------------------------- #
    print(f"[p1.5] loading train cache {P0_FEAT_PATH}")
    tr_blob = torch.load(P0_FEAT_PATH, map_location="cpu", weights_only=False)
    tr_q = tr_blob["q_chunk_feature"].float()
    tr_v = tr_blob["v_state_feature"].float()
    tr = build_step_quantities(tr_blob)
    tr_ep, tr_succ = tr["ep"], tr["succ"]
    tr_next_row, tr_have_next, tr_r_chunk, tr_done = tr["next_row"], tr["have_next"], tr["r_chunk"], tr["done"]
    n_tr = len(tr["G"])
    state_dim = tr_v.shape[1]
    chunk_dim = tr_q.shape[1]
    assert ho_v.shape[1] == state_dim and ho_q.shape[1] == chunk_dim, "train/held-out feature dim mismatch"

    succ_eps = sorted({int(e) for e in tr_ep[tr_succ]})
    fail_eps = sorted({int(e) for e in tr_ep[~tr_succ]})
    print(f"[p1.5] train demos: success={len(succ_eps)} fail={len(fail_eps)}")

    # device tensors (full train cache; we index_select per point)
    tr_q_d = tr_q.to(device)
    tr_v_d = tr_v.to(device)
    tr_r_t = torch.from_numpy(np.nan_to_num(tr_r_chunk, nan=0.0).astype(np.float32)).to(device).unsqueeze(1)
    tr_done_t = torch.from_numpy(tr_done.astype(np.float32)).to(device).unsqueeze(1)
    tr_G_t = torch.from_numpy(tr["G"].astype(np.float32)).to(device)
    tr_next_safe = np.where(tr_have_next, tr_next_row, np.arange(n_tr))
    tr_next_safe_t = torch.from_numpy(tr_next_safe).to(device)
    tr_boot_ok = (tr_done == 1) | ((tr_done == 0) & tr_have_next)

    # ---------------------------------------------------------------- #
    # 3. Sweep                                                         #
    # ---------------------------------------------------------------- #
    def mae_vs_G(vals, mask):
        m = mask
        if int(m.sum()) == 0:
            return float("nan")
        return float(np.abs(vals[m] - ho["G"][m]).mean())

    ho_succ_eps = sorted({int(e) for e in ho["ep"][ho_succ]})
    ho_fail_eps = sorted({int(e) for e in ho["ep"][~ho_succ]})

    sweep_results: list[dict] = []
    last_point_preds: dict[str, np.ndarray] = {}
    for N in sweep_sizes:
        sel_eps = set(succ_eps[:N]) | set(fail_eps[:N])
        in_sel = np.array([int(e) in sel_eps for e in tr_ep], dtype=bool)
        mc_train_idx = torch.from_numpy(np.where(in_sel)[0]).to(device)
        boot_train_idx = torch.from_numpy(np.where(in_sel & tr_boot_ok)[0]).to(device)
        print(f"\n[p1.5] ===== sweep N={N} (succ+fail demos) | "
              f"mc_rows={mc_train_idx.numel()} boot_rows={boot_train_idx.numel()} =====")

        # --- arm: mc_return (oracle upper bound) ---
        mc_v_head = train_mc_head(
            lambda: _MLPReg(VNetwork(context_dim=state_dim, hidden_dims=HIDDEN_DIMS)),
            tr_v_d, tr_G_t, mc_train_idx, steps=args.mc_steps, batch=args.batch, lr=V_LR,
            device=device, tag=f"V|N{N}")
        mc_q_head = train_mc_head(
            lambda: _MLPReg(QChunkNetwork(chunk_feature_dim=chunk_dim, hidden_dims=HIDDEN_DIMS)),
            tr_q_d, tr_G_t, mc_train_idx, steps=args.mc_steps, batch=args.batch, lr=Q_LR,
            device=device, tag=f"Q|N{N}")
        mc_v_ho = predict_head(mc_v_head, ho_v_d)
        mc_q_ho = predict_head(mc_q_head, ho_q_d)

        # --- arm: sarsa_q (FQE) ---
        sarsa = SarsaQ(chunk_dim, device, n_nets=2)
        for step in range(args.sarsa_steps):
            sel = boot_train_idx[torch.randint(0, boot_train_idx.numel(), (args.batch,), device=device)]
            nxt = tr_next_safe_t.index_select(0, sel)
            sarsa.update(tr_q_d.index_select(0, sel), tr_q_d.index_select(0, nxt),
                         tr_r_t.index_select(0, sel), tr_done_t.index_select(0, sel))
        sarsa_q_ho = sarsa.predict(ho_q_d)

        # --- arm: iql_v (production reproduction) ---
        iql = train_iql_arm(tr, tr_q_d, tr_v_d, tr_r_t, tr_done_t, tr_next_safe_t, boot_train_idx,
                            device=device, batch=args.batch, warmup_steps=args.iql_warmup_steps,
                            full_steps=args.iql_full_steps, state_dim=state_dim, chunk_dim=chunk_dim)
        iql_v_ho = iql_predict_v(iql, ho_v_d)
        iql_q_ho = iql_predict_q(iql, ho_q_d)

        # --- metrics on TRUE held-out ---
        arm_value = {"iql_v": iql_v_ho, "sarsa_q": sarsa_q_ho, "mc_return": mc_v_ho}
        succ_mae = {a: mae_vs_G(v, ho_succ_mask) for a, v in arm_value.items()}
        fail_mae = {a: mae_vs_G(v, ho_fail_mask) for a, v in arm_value.items()}
        fail_mean = {a: float(v[ho_fail_mask].mean()) if int(ho_fail_mask.sum()) else float("nan")
                     for a, v in arm_value.items()}
        succ_mean = {a: float(v[ho_succ_mask].mean()) if int(ho_succ_mask.sum()) else float("nan")
                     for a, v in arm_value.items()}
        jump = {}
        for a, v in arm_value.items():
            sj = adjacent_jump(v, ho["ep"], ho["st"], set(ho_succ_eps))
            fj = adjacent_jump(v, ho["ep"], ho["st"], set(ho_fail_eps))
            jump[a] = {"success": {"mean": sj[0], "max": sj[1], "n": sj[2]},
                       "fail": {"mean": fj[0], "max": fj[1], "n": fj[2]}}
        qtd_succ, qtd_succ_n = iql_qtd_mae(iql, ho, ho_q_d, ho_v_d, ho_succ_mask)
        qtd_fail, qtd_fail_n = iql_qtd_mae(iql, ho, ho_q_d, ho_v_d, ho_fail_mask)

        point = {
            "N": N, "n_mc_rows": int(mc_train_idx.numel()), "n_boot_rows": int(boot_train_idx.numel()),
            "heldout_success_mae_vs_G": succ_mae,
            "heldout_fail_mae_vs_G": fail_mae,
            "heldout_success_mean_value": succ_mean,
            "heldout_fail_mean_value": fail_mean,
            "heldout_adjacent_jump": jump,
            "iql_v_heldout_qtd_mae": {"success": qtd_succ, "success_n": qtd_succ_n,
                                       "fail": qtd_fail, "fail_n": qtd_fail_n},
        }
        sweep_results.append(point)
        print(f"[p1.5] N={N} HELD-OUT success MAE-vs-G: "
              f"iql_v={succ_mae['iql_v']:.4f} sarsa_q={succ_mae['sarsa_q']:.4f} "
              f"mc_return(oracle)={succ_mae['mc_return']:.4f}")
        print(f"[p1.5] N={N} HELD-OUT success V-jump (iql_v) mean/max="
              f"{jump['iql_v']['success']['mean']:.4f}/{jump['iql_v']['success']['max']:.4f} | "
              f"iql_v Q-TD MAE success={qtd_succ:.4f} fail={qtd_fail:.4f}")

        if N == sweep_sizes[-1]:
            last_point_preds = {"iql_v": iql_v_ho, "sarsa_q": sarsa_q_ho,
                                "mc_return": mc_v_ho, "mc_return_Q": mc_q_ho, "iql_v_Q": iql_q_ho}

    # ---------------------------------------------------------------- #
    # 4. Report: production-symptom comparison + (a)/(b) verdict        #
    # ---------------------------------------------------------------- #
    prod_ref = {"success_qtd_mae": 0.89, "success_v_jump_mean": 0.53, "success_v_jump_max": "5-6"}
    floor = {a: min(p["heldout_success_mae_vs_G"][a] for p in sweep_results)
             for a in ("iql_v", "sarsa_q", "mc_return")}
    first = {a: sweep_results[0]["heldout_success_mae_vs_G"][a]
             for a in ("iql_v", "sarsa_q", "mc_return")}
    last = {a: sweep_results[-1]["heldout_success_mae_vs_G"][a]
            for a in ("iql_v", "sarsa_q", "mc_return")}
    drop = {a: first[a] - last[a] for a in first}  # >0 => MAE fell with more data

    print("\n[p1.5] ===== DATA-SIZE TREND (held-out success MAE-vs-G) =====")
    print(f"[p1.5] {'N':>5}{'iql_v':>10}{'sarsa_q':>10}{'mc(orac)':>10}")
    for p in sweep_results:
        m = p["heldout_success_mae_vs_G"]
        print(f"[p1.5] {p['N']:>5}{m['iql_v']:>10.4f}{m['sarsa_q']:>10.4f}{m['mc_return']:>10.4f}")
    print(f"[p1.5] MAE drop {sweep_sizes[0]}->{sweep_sizes[-1]} demos: "
          f"iql_v={drop['iql_v']:+.4f} sarsa_q={drop['sarsa_q']:+.4f} mc_oracle={drop['mc_return']:+.4f}")
    print(f"[p1.5] production ref: success Q-TD MAE~{prod_ref['success_qtd_mae']} "
          f"V-jump mean~{prod_ref['success_v_jump_mean']}")

    summary = {
        "config": {
            "device": device, "seed": args.seed, "batch": args.batch,
            "iql_warmup_steps": args.iql_warmup_steps, "iql_full_steps": args.iql_full_steps,
            "sarsa_steps": args.sarsa_steps, "mc_steps": args.mc_steps,
            "sweep_sizes": list(sweep_sizes), "gamma": GAMMA, "action_horizon": ACTION_HORIZON,
            "output_reward_coef": OUTPUT_REWARD_COEF, "hidden_dims": list(HIDDEN_DIMS),
            "demo_selection": "sorted-episode prefix (nested 10<20<...<50)",
        },
        "heldout": {
            "buffer": str(args.heldout_buffer),
            "n_success_rows": int(ho_succ_mask.sum()), "n_fail_rows": int(ho_fail_mask.sum()),
            "n_success_demos": len(ho_succ_eps), "n_fail_demos": len(ho_fail_eps),
        },
        "production_reference": prod_ref,
        "sweep": sweep_results,
        "trend": {"first_point": first, "last_point": last, "floor": floor, "mae_drop": drop},
    }
    (OUT_DIR / "p1_5_results.json").write_text(json.dumps(summary, indent=2))
    print(f"[p1.5] wrote results -> {OUT_DIR / 'p1_5_results.json'}")

    # ---------------------------------------------------------------- #
    # 5. Plots: sweep curve + per-demo dumps at the full-data point     #
    # ---------------------------------------------------------------- #
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        have_mpl = True
    except Exception as exc:  # pragma: no cover
        print(f"[p1.5] matplotlib unavailable ({exc}); skipping plots")
        have_mpl = False

    if have_mpl:
        xs = [p["N"] for p in sweep_results]
        plt.figure(figsize=(8, 5))
        for a, lab in [("mc_return", "mc_return (oracle)"), ("iql_v", "iql_v"), ("sarsa_q", "sarsa_q")]:
            ys = [p["heldout_success_mae_vs_G"][a] for p in sweep_results]
            plt.plot(xs, ys, marker="o", label=lab)
        plt.axhline(prod_ref["success_qtd_mae"], ls=":", color="gray",
                    label=f"prod Q-TD MAE ~{prod_ref['success_qtd_mae']}")
        plt.xlabel("train demos per class (N)")
        plt.ylabel("HELD-OUT success MAE vs G")
        plt.title("P1.5 data-size sweep — does held-out error fall with data?\n"
                  "(falling => data-limited/overfit; flat floor => representation ceiling)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "sweep_curve.png", dpi=120)
        plt.close()
        print(f"[p1.5] wrote sweep curve -> {OUT_DIR / 'sweep_curve.png'}")

    # per-demo CSV/PNG at the full-data point (2 success + 2 fail held-out demos)
    if last_point_preds:
        dump_succ = ho_succ_eps[:2]
        dump_fail = ho_fail_eps[:2]
        for tag, e in [("success", x) for x in dump_succ] + [("fail", x) for x in dump_fail]:
            m = ho["ep"] == e
            order = np.argsort(ho["st"][m])
            steps_e = ho["st"][m][order]
            gl = ho["G"][m][order]
            iv = last_point_preds["iql_v"][m][order]
            sq = last_point_preds["sarsa_q"][m][order]
            mc = last_point_preds["mc_return"][m][order]
            with (OUT_DIR / f"demo_{tag}_ep{e}.csv").open("w", newline="") as fh:
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
                plt.title(f"P1.5 held-out (full-data N={sweep_sizes[-1]}) — {tag} demo ep={e}")
                plt.legend()
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(OUT_DIR / f"demo_{tag}_ep{e}.png", dpi=110)
                plt.close()
        print(f"[p1.5] dumped held-out demo CSV/PNG: success={dump_succ} fail={dump_fail}")

    print("[p1.5] DONE")


if __name__ == "__main__":
    main()
