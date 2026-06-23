"""P0 diagnostic: Monte-Carlo return-to-go regression sanity check for IQL warmup.

Isolates whether the offline IQL warmup's poor Q/V on SUCCESS rollouts is caused
by *bootstrapping* (the IQL/TD objective) or by the *representation / head
capacity*. It does so by REMOVING bootstrapping entirely: it regresses the SAME
Q/V head architectures (on the SAME frozen encoder features) directly to
Monte-Carlo discounted return-to-go labels via supervised MSE.

Interpretation:
  - MC regression fits SUCCESS curves smoothly with low MAE  => representation +
    head capacity are sufficient => the warmup problem is bootstrapping.
  - MC regression ALSO fails on SUCCESS (high MAE and/or large adjacent-window V
    jumps) => the problem is feature preprocessing / action-chunk encoding /
    head optimization, and later bootstrap experiments are moot.

This script is ADDITIVE and self-contained: it reuses already-public functions
(`_freeze_post_success_tail`, `IQLReplayBuffer.preencode_step_cache`,
`QChunkNetwork`, `VNetwork`, `SharedDynamicsEncoder`). It does not modify any
core file.

Run (module-style):
    cd <repo-root> && MUJOCO_GL=egl PYTHONPATH=. \
        CUDA_VISIBLE_DEVICES=1 python \
        -m robosuite.pipeline.algorithms.q_learning.utils.mc_return_probe \
        --device cuda:0            # cuda:0 == physical GPU 1 after CVD renumber
    # smoke first:
    ... --max-demos 6 --train-steps 2000
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
from robosuite.pipeline.algorithms.q_learning.common import IQLConfig
from robosuite.pipeline.algorithms.q_learning.networks import QChunkNetwork, VNetwork
from robosuite.pipeline.algorithms.q_learning.replay import IQLReplayBuffer
from robosuite.pipeline.algorithms.q_learning.warmup import _freeze_post_success_tail

# ---- Fixed hyperparameters (mirror train_dipole_rl.yaml) ----
OUTPUT_REWARD_COEF = 0.1
GAMMA = 0.99
ACTION_HORIZON = 8
HIDDEN_DIMS = (512, 512)
GRAD_CLIP = 2.0
CAMERA_NAMES = ["agentview", "robot0_robotview", "robot0_eye_in_hand"]

REPO_ROOT = Path(__file__).resolve().parents[5]
BUFFER_PATH = REPO_ROOT / "data/PickPlaceCereal/offline_data/iql_offline_transitions.pt"
META_PATH = REPO_ROOT / "data/PickPlaceCereal/offline_data/iql_offline_transitions.meta.json"
NNPU_CKPT = (
    REPO_ROOT
    / "checkpoints/dyn_disc/pu_bce_eval_robosuite/run_20260619_194127_PickPlaceCereal"
    / "checkpoints/pu_bce_head.pth"
)
OUT_DIR = REPO_ROOT / "outputs/dipole_rl-debug/p0_mc_probe"


def _episode_index_of(t: Any) -> int:
    return int((t.info or {}).get("episode_index", -1))


def _episode_step_of(t: Any) -> int:
    return int((t.info or {}).get("episode_step", -1))


def _is_success_frame(t: Any) -> bool:
    return bool((t.info or {}).get("success", False))


def _truncate_to_max_demos(storage: list[Any], max_success: int, max_fail: int) -> list[Any]:
    """Keep only the first `max_success` success demos and `max_fail` fail demos.

    A demo is delimited by `episode_index`. "success" = demo contains any
    success frame. Preserves original storage order (so episode_index stays
    monotonic enough for valid-start logic, which only checks contiguity within
    an episode, not global monotonicity)."""
    # Group contiguous transitions by episode_index.
    demos: list[list[Any]] = []
    cur: list[Any] = []
    cur_ep = None
    for t in storage:
        ep = _episode_index_of(t)
        if cur_ep is None or ep == cur_ep:
            cur.append(t)
            cur_ep = ep
        else:
            demos.append(cur)
            cur = [t]
            cur_ep = ep
    if cur:
        demos.append(cur)

    kept: list[Any] = []
    n_succ = n_fail = 0
    for demo in demos:
        is_succ = any(_is_success_frame(t) for t in demo)
        if is_succ and n_succ < max_success:
            kept.extend(demo)
            n_succ += 1
        elif (not is_succ) and n_fail < max_fail:
            kept.extend(demo)
            n_fail += 1
    print(f"[p0] truncated to {n_succ} success + {n_fail} fail demos ({len(kept)} transitions)")
    return kept


def _compute_return_to_go(storage: list[Any]) -> np.ndarray:
    """Per-transition discounted return-to-go, episode-aware.

    r_k = OUTPUT_REWARD_COEF * raw_reward (item.reward). Within each episode
    (delimited by episode_index), g[i] = r[i] + GAMMA * g[i+1]; the episode's
    last frame uses g = r[last]. Never crosses an episode boundary.
    Returns g as a (N,) float array aligned to storage indices.
    """
    n = len(storage)
    g = np.zeros(n, dtype=np.float64)
    # Walk backward; reset accumulator at episode boundaries.
    next_g = 0.0
    next_ep = None
    for i in range(n - 1, -1, -1):
        t = storage[i]
        ep = _episode_index_of(t)
        raw = float(t.reward) if t.reward is not None else 0.0
        r = OUTPUT_REWARD_COEF * raw
        if next_ep is None or ep != next_ep:
            # i is the LAST frame of its episode (walking backward, the frame
            # after it belongs to a different episode or doesn't exist).
            g[i] = r
        else:
            g[i] = r + GAMMA * next_g
        next_g = g[i]
        next_ep = ep
    return g


class MLPReg(nn.Module):
    """Thin wrapper so Q/V heads share one training/eval path."""

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _train_head(
    head: nn.Module,
    feats: torch.Tensor,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    *,
    steps: int,
    batch: int,
    lr: float,
    wd: float,
    device: str,
    tag: str,
) -> None:
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    head.train()
    n_train = train_idx.numel()
    log_every = min(1000, max(1, steps // 10))
    loss_curve: list[tuple[int, float]] = []
    final_loss = float("nan")
    for step in range(steps):
        sel = train_idx[torch.randint(0, n_train, (batch,), device=device)]
        x = feats.index_select(0, sel)
        y = labels.index_select(0, sel)
        pred = head(x)
        loss = nn.functional.mse_loss(pred, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), GRAD_CLIP)
        opt.step()
        final_loss = loss.item()
        if step % log_every == 0 or step == steps - 1:
            loss_curve.append((step, final_loss))
            print(f"[p0][{tag}] step={step:6d} mse={final_loss:.5f}")
    head.eval()
    return {"final_loss": final_loss, "loss_curve": loss_curve}


def _overfit_sanity(
    head_factory,
    feats: torch.Tensor,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    *,
    batch: int,
    steps: int,
    lr: float,
    device: str,
    tag: str,
) -> dict:
    """Train a FRESH head on a SINGLE fixed batch for many steps.

    Proves the head+optimizer wiring can drive MSE near 0 when overfitting is
    trivially possible. If this fails, the bug is in the head/feature/optimizer
    wiring, not in data or capacity.
    """
    head = head_factory().to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    n_train = train_idx.numel()
    sel = train_idx[torch.randint(0, n_train, (batch,), device=device)]
    x = feats.index_select(0, sel)
    y = labels.index_select(0, sel)
    head.train()
    first = last = float("nan")
    for step in range(steps):
        pred = head(x)
        loss = nn.functional.mse_loss(pred, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
    head.eval()
    with torch.no_grad():
        pred = head(x).cpu().numpy()
    yv = y.cpu().numpy()
    pred_std = float(np.std(pred))
    print(f"[p0][overfit:{tag}] single-batch n={batch} first_mse={first:.5f} "
          f"final_mse={last:.6f} pred_std={pred_std:.4f} label_std={float(np.std(yv)):.4f}")
    return {"tag": tag, "batch": int(batch), "steps": int(steps),
            "first_mse": first, "final_mse": last, "pred_std": pred_std,
            "label_std": float(np.std(yv))}


@torch.no_grad()
def _predict_all(head: nn.Module, feats: torch.Tensor, batch: int = 8192) -> torch.Tensor:
    out = []
    head.eval()
    for s in range(0, feats.shape[0], batch):
        out.append(head(feats[s : s + batch]).detach())
    return torch.cat(out, dim=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-demos", type=int, default=0,
                    help="If >0, cap to this many success AND this many fail demos (smoke).")
    ap.add_argument("--train-steps", type=int, default=25000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-6)
    ap.add_argument("--encode-batch", type=int, default=32)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reuse-features", default="",
                    help="Path to a previously saved preencoded_features.pt to skip encoding.")
    args = ap.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    meta = json.loads(META_PATH.read_text())
    img_height = int(meta.get("image_size", meta.get("img_height", 84)))
    print(f"[p0] meta: action_horizon={meta['action_horizon']} cameras={meta['camera_names']} "
          f"image_size={img_height} reward_mode={meta['reward_mode']}")
    assert int(meta["action_horizon"]) == ACTION_HORIZON, "action_horizon mismatch"
    assert list(meta["camera_names"]) == CAMERA_NAMES, "camera_names mismatch"

    # ---------------------------------------------------------------- #
    # 1. Load buffer + freeze post-success tail (mirror warmup)        #
    # ---------------------------------------------------------------- #
    buffer = FlowDaggerReplayBuffer(
        config=ReplayBufferConfig(capacity=max(50000, int(meta["n_transitions"]) + 10), batch_size=args.batch),
        name="p0_mc_probe",
        camera_names=CAMERA_NAMES,
        action_horizon=ACTION_HORIZON,
        image_size=img_height,
        augmentation_config=FlowAugmentationConfig(),
    )
    print(f"[p0] loading buffer {BUFFER_PATH}")
    buffer.load(BUFFER_PATH)
    print(f"[p0] loaded {len(buffer)} transitions, valid_starts={buffer.num_valid_sequences()}")

    if args.max_demos > 0:
        with buffer._lock:  # noqa: SLF001
            buffer._storage = _truncate_to_max_demos(  # noqa: SLF001
                buffer._storage, args.max_demos, args.max_demos
            )
            buffer._rebuild_valid_start_cache_locked()  # noqa: SLF001
        print(f"[p0] post-truncate valid_starts={buffer.num_valid_sequences()}")

    with buffer._lock:  # noqa: SLF001
        n_frozen = _freeze_post_success_tail(buffer._storage)  # noqa: SLF001
        buffer._rebuild_valid_start_cache_locked()  # noqa: SLF001 (rewards changed; valid-starts unchanged but safe)
    print(f"[p0] freeze_post_success: {n_frozen} frames frozen to absorbing anchor (reward=0)")

    storage = list(buffer._storage)  # noqa: SLF001

    # ---------------------------------------------------------------- #
    # 2. valid_starts (in order)                                       #
    # ---------------------------------------------------------------- #
    with buffer._lock:  # noqa: SLF001
        valid_starts = list(buffer._get_valid_start_indices_locked())  # noqa: SLF001
    valid_starts_arr = np.asarray(valid_starts, dtype=np.int64)
    print(f"[p0] valid_starts={len(valid_starts)}")

    # ---------------------------------------------------------------- #
    # 4. MC return-to-go labels (computed before encoding for sanity)  #
    # ---------------------------------------------------------------- #
    g_all = _compute_return_to_go(storage)  # (N,) per-storage-index
    G_label = g_all[valid_starts_arr].astype(np.float32)  # (R,) aligned to rows

    ep_index_per_row = np.asarray([_episode_index_of(storage[s]) for s in valid_starts], dtype=np.int64)
    ep_step_per_row = np.asarray([_episode_step_of(storage[s]) for s in valid_starts], dtype=np.int64)

    # Per-demo success tag: a demo is success iff ANY of its frames has success.
    demo_is_success: dict[int, bool] = {}
    for t in storage:
        ep = _episode_index_of(t)
        if _is_success_frame(t):
            demo_is_success[ep] = True
        demo_is_success.setdefault(ep, False)
    is_success_per_row = np.asarray(
        [demo_is_success.get(int(ep), False) for ep in ep_index_per_row], dtype=bool
    )

    n_succ_rows = int(is_success_per_row.sum())
    n_fail_rows = int((~is_success_per_row).sum())
    print(f"[p0] rows: success={n_succ_rows} fail={n_fail_rows}")

    # ---------------------------------------------------------------- #
    # SANITY CHECK on labels                                           #
    # ---------------------------------------------------------------- #
    print("[p0] ===== G-LABEL SANITY CHECK =====")
    print(f"[p0] G stats overall: min={G_label.min():.4f} max={G_label.max():.4f} "
          f"mean={G_label.mean():.4f}")
    print(f"[p0] G success rows: min={G_label[is_success_per_row].min():.4f} "
          f"max={G_label[is_success_per_row].max():.4f} mean={G_label[is_success_per_row].mean():.4f}")
    print(f"[p0] G fail rows:    min={G_label[~is_success_per_row].min():.4f} "
          f"max={G_label[~is_success_per_row].max():.4f} mean={G_label[~is_success_per_row].mean():.4f}")

    # Pick one success demo and print G along episode_step.
    succ_eps = sorted({int(e) for e, s in demo_is_success.items() if s})
    fail_eps = sorted({int(e) for e, s in demo_is_success.items() if not s})
    sane = True
    if succ_eps:
        ep0 = succ_eps[0]
        mask = ep_index_per_row == ep0
        steps = ep_step_per_row[mask]
        gs = G_label[mask]
        order = np.argsort(steps)
        steps, gs = steps[order], gs[order]
        print(f"[p0] success demo ep={ep0}: n_rows={len(gs)}")
        # show first few, middle, last few
        show = list(range(min(3, len(gs)))) + list(range(max(0, len(gs) - 4), len(gs)))
        for i in sorted(set(show)):
            print(f"     step={steps[i]:4d}  G={gs[i]:+.4f}")
        # Expectation: G near demo END (post-success, frozen reward=0) -> ~0;
        # early steps more negative. Check monotone-ish increase toward 0.
        if gs[-1] > 0.001:
            print("[p0] WARN: tail G of success demo > 0 (expected ~0).")
            sane = False
        if gs[0] >= gs[-1]:
            print("[p0] WARN: success demo early G not more negative than tail G.")
            sane = False
    if fail_eps:
        ep0 = fail_eps[0]
        mask = ep_index_per_row == ep0
        steps = ep_step_per_row[mask]
        gs = G_label[mask]
        order = np.argsort(steps)
        steps, gs = steps[order], gs[order]
        print(f"[p0] fail demo ep={ep0}: n_rows={len(gs)} "
              f"G[start]={gs[0]:+.4f} G[end]={gs[-1]:+.4f}")
        # Fail demos: reward -1 every step, no terminal -> G strongly negative.
        # With r=-0.1/step, gamma=0.99, long horizon -> toward ~ -0.1/(1-0.99)= -10.
        if gs.min() > -1.0:
            print("[p0] WARN: fail demo G not strongly negative (expected toward ~-10).")
    print("[p0] ================================")
    if not sane:
        print("[p0] SANITY CHECK FAILED — stopping before training. Inspect labels above.")
        return

    # ---------------------------------------------------------------- #
    # 3. Preencode features ONCE (or reuse)                            #
    # ---------------------------------------------------------------- #
    if args.reuse_features and Path(args.reuse_features).exists():
        print(f"[p0] reusing features from {args.reuse_features}")
        blob = torch.load(args.reuse_features, map_location="cpu", weights_only=False)
        q_feat = blob["q_chunk_feature"]
        v_feat = blob["v_state_feature"]
        assert q_feat.shape[0] == len(valid_starts), "reused features row mismatch"
    else:
        print(f"[p0] building encoder from {NNPU_CKPT} on {device}")
        encoder = SharedDynamicsEncoder(
            nnpu_ckpt_path=str(NNPU_CKPT),
            encoder_ckpt=None,
            device=device,
            camera_to_view={},
        )
        encoder.bind_policy_cameras(CAMERA_NAMES)
        iql_cfg = IQLConfig(action_horizon=ACTION_HORIZON, device=device,
                            disc_reward_coef=0.0, output_reward_coef=OUTPUT_REWARD_COEF,
                            discount=GAMMA, reward_mode="-1/0")
        replay = IQLReplayBuffer(base_buffer=buffer, cfg=iql_cfg)
        print(f"[p0] preencoding {len(valid_starts)} windows (encode_batch={args.encode_batch})...")
        cache = replay.preencode_step_cache(
            encoder=encoder,
            discriminator=None,
            device=device,
            encode_batch_size=args.encode_batch,
            cache_device="cpu",
            progress_desc="[p0] preencode",
        )
        q_feat = cache.q_chunk_feature  # (R, D_chunk) cpu
        v_feat = cache.v_state_feature  # (R, D_state) cpu
        assert q_feat.shape[0] == len(valid_starts), (
            f"cache rows {q_feat.shape[0]} != valid_starts {len(valid_starts)}"
        )
        print(f"[p0] features: q_chunk={tuple(q_feat.shape)} v_state={tuple(v_feat.shape)}")

    # ---------------------------------------------------------------- #
    # 8. Save preencoded features + labels for reuse                   #
    # ---------------------------------------------------------------- #
    feat_path = OUT_DIR / "preencoded_features.pt"
    torch.save(
        {
            "q_chunk_feature": q_feat,
            "v_state_feature": v_feat,
            "valid_starts": valid_starts_arr,
            "episode_index_per_row": ep_index_per_row,
            "episode_step_per_row": ep_step_per_row,
            "G_label": G_label,
            "is_success_per_row": is_success_per_row,
            "meta": {
                "output_reward_coef": OUTPUT_REWARD_COEF,
                "gamma": GAMMA,
                "action_horizon": ACTION_HORIZON,
                "n_frozen": int(n_frozen),
                "max_demos": int(args.max_demos),
                "buffer_path": str(BUFFER_PATH),
                "nnpu_ckpt": str(NNPU_CKPT),
            },
        },
        feat_path,
    )
    print(f"[p0] saved features+labels -> {feat_path}")

    # ---------------------------------------------------------------- #
    # FEATURE VARIANCE SANITY: confirm features VARY across rows.       #
    # ---------------------------------------------------------------- #
    print("[p0] ===== FEATURE SANITY CHECK =====")
    feat_stats = {}
    for fname, ften in (("q_chunk", q_feat), ("v_state", v_feat)):
        ff = ften.float()
        per_row_mean = ff.mean(dim=1)          # (R,)
        per_dim_std = ff.std(dim=0)            # (D,) variation across rows per feature dim
        # fraction of rows that are exactly identical to row 0 (degenerate cache)
        row0 = ff[0:1]
        identical = bool(torch.allclose(ff, row0.expand_as(ff)))
        feat_stats[fname] = {
            "shape": list(ff.shape),
            "global_mean": float(ff.mean()),
            "global_std": float(ff.std()),
            "per_row_mean_std": float(per_row_mean.std()),  # >0 => rows differ
            "per_dim_std_mean": float(per_dim_std.mean()),   # avg cross-row variation
            "all_rows_identical": identical,
        }
        print(f"[p0] {fname}: shape={tuple(ff.shape)} global_mean={float(ff.mean()):+.4f} "
              f"global_std={float(ff.std()):.4f} per_row_mean_std={float(per_row_mean.std()):.4f} "
              f"per_dim_std_mean={float(per_dim_std.mean()):.4f} all_identical={identical}")
        if identical:
            print(f"[p0] FATAL: {fname} features are ALL IDENTICAL across rows — encode is broken.")
    print("[p0] ================================")

    # ---------------------------------------------------------------- #
    # 5. Train/val split BY DEMO                                       #
    # ---------------------------------------------------------------- #
    rng = np.random.RandomState(args.seed)

    def _split_eps(eps: list[int]) -> tuple[set[int], set[int]]:
        eps = list(eps)
        rng.shuffle(eps)
        n_val = max(1, int(round(len(eps) * args.val_frac))) if eps else 0
        return set(eps[n_val:]), set(eps[:n_val])  # (train, val)

    succ_train, succ_val = _split_eps(succ_eps)
    fail_train, fail_val = _split_eps(fail_eps)
    val_eps = succ_val | fail_val
    print(f"[p0] split: success train/val demos = {len(succ_train)}/{len(succ_val)}; "
          f"fail train/val demos = {len(fail_train)}/{len(fail_val)}")

    is_val_row = np.asarray([int(ep) in val_eps for ep in ep_index_per_row], dtype=bool)
    train_idx = np.where(~is_val_row)[0]

    # ---------------------------------------------------------------- #
    # 6. Train Q and V heads independently (pure MSE, no bootstrap)    #
    # ---------------------------------------------------------------- #
    q_feat_d = q_feat.to(device)
    v_feat_d = v_feat.to(device)
    labels_d = torch.from_numpy(G_label).to(device)
    train_idx_d = torch.from_numpy(train_idx).to(device)

    def _make_q():
        return MLPReg(QChunkNetwork(chunk_feature_dim=q_feat.shape[1], hidden_dims=HIDDEN_DIMS))

    def _make_v():
        return MLPReg(VNetwork(context_dim=v_feat.shape[1], hidden_dims=HIDDEN_DIMS))

    # --- OVERFIT-SANITY: can a fresh head drive ONE batch to ~0 MSE? ---
    print("[p0] ===== OVERFIT-SANITY (single fixed batch, 3000 steps) =====")
    overfit_v = _overfit_sanity(_make_v, v_feat_d, labels_d, train_idx_d,
                                batch=args.batch, steps=3000, lr=1e-3, device=device, tag="V")
    overfit_q = _overfit_sanity(_make_q, q_feat_d, labels_d, train_idx_d,
                                batch=args.batch, steps=3000, lr=1e-3, device=device, tag="Q")

    q_head = _make_q().to(device)
    v_head = _make_v().to(device)

    print(f"[p0] training V-head ({args.train_steps} steps)")
    v_loss_info = _train_head(v_head, v_feat_d, labels_d, train_idx_d, steps=args.train_steps,
                              batch=args.batch, lr=args.lr, wd=args.wd, device=device, tag="V")
    print(f"[p0] training Q-head ({args.train_steps} steps)")
    q_loss_info = _train_head(q_head, q_feat_d, labels_d, train_idx_d, steps=args.train_steps,
                              batch=args.batch, lr=args.lr, wd=args.wd, device=device, tag="Q")

    # ---------------------------------------------------------------- #
    # 7. Evaluate + report                                             #
    # ---------------------------------------------------------------- #
    v_pred = _predict_all(v_head, v_feat_d).cpu().numpy()
    q_pred = _predict_all(q_head, q_feat_d).cpu().numpy()

    # --- correlation of predictions vs labels on TRAIN rows ---
    def _corr(pred: np.ndarray, mask: np.ndarray) -> float:
        p = pred[mask]
        g = G_label[mask]
        if p.size < 2 or np.std(p) < 1e-9 or np.std(g) < 1e-9:
            return float("nan")
        return float(np.corrcoef(p, g)[0, 1])

    train_mask = ~is_val_row
    v_corr_train = _corr(v_pred, train_mask)
    q_corr_train = _corr(q_pred, train_mask)
    v_pred_std_train = float(np.std(v_pred[train_mask]))
    q_pred_std_train = float(np.std(q_pred[train_mask]))
    print("[p0] ===== PRED-vs-LABEL CORRELATION (train rows) =====")
    print(f"[p0] V: corr={v_corr_train:+.4f} pred_std={v_pred_std_train:.4f}")
    print(f"[p0] Q: corr={q_corr_train:+.4f} pred_std={q_pred_std_train:.4f}")
    if (v_pred_std_train < 1e-3) and (v_loss_info["final_loss"] < 1.0):
        print("[p0] WARN: V_pred near-constant yet loss low — likely fit to mean only.")

    def _mae(pred: np.ndarray, mask: np.ndarray) -> float:
        if mask.sum() == 0:
            return float("nan")
        return float(np.abs(pred[mask] - G_label[mask]).mean())

    masks = {
        "success-train": is_success_per_row & ~is_val_row,
        "success-val": is_success_per_row & is_val_row,
        "fail-train": (~is_success_per_row) & ~is_val_row,
        "fail-val": (~is_success_per_row) & is_val_row,
    }
    print("[p0] ===== REGRESSION MAE (pred vs G_label) =====")
    print(f"[p0] {'split':<16}{'n':>8}{'V_MAE':>12}{'Q_MAE':>12}")
    mae_table = {}
    for name, mask in masks.items():
        v_mae = _mae(v_pred, mask)
        q_mae = _mae(q_pred, mask)
        mae_table[name] = {"n": int(mask.sum()), "v_mae": v_mae, "q_mae": q_mae}
        print(f"[p0] {name:<16}{int(mask.sum()):>8}{v_mae:>12.4f}{q_mae:>12.4f}")

    # ---- Adjacent-window V-jump stats per demo ----
    def _jump_stats(eps: set[int]) -> tuple[float, float, int]:
        all_jumps: list[float] = []
        for ep in eps:
            mask = ep_index_per_row == ep
            if mask.sum() < 2:
                continue
            steps = ep_step_per_row[mask]
            vp = v_pred[mask]
            order = np.argsort(steps)
            vp = vp[order]
            jumps = np.abs(np.diff(vp))
            all_jumps.extend(jumps.tolist())
        if not all_jumps:
            return float("nan"), float("nan"), 0
        arr = np.asarray(all_jumps)
        return float(arr.mean()), float(arr.max()), len(all_jumps)

    succ_jmean, succ_jmax, succ_jn = _jump_stats(succ_val if succ_val else set(succ_eps))
    fail_jmean, fail_jmax, fail_jn = _jump_stats(fail_val if fail_val else set(fail_eps))
    print("[p0] ===== ADJACENT-WINDOW V-JUMP |V[t+1]-V[t]| (val demos) =====")
    print(f"[p0] {'demo-set':<16}{'n_jumps':>10}{'mean':>10}{'max':>10}")
    print(f"[p0] {'success':<16}{succ_jn:>10}{succ_jmean:>10.4f}{succ_jmax:>10.4f}")
    print(f"[p0] {'fail':<16}{fail_jn:>10}{fail_jmean:>10.4f}{fail_jmax:>10.4f}")
    print("[p0] (prior IQL run: fail mean~0.05-0.07; success mean~0.53, success max~5-6)")

    # ---------------------------------------------------------------- #
    # 7b. Per-demo CSV + PNG dumps (2 success + 2 fail val demos)      #
    # ---------------------------------------------------------------- #
    dump_succ = sorted(succ_val)[:2] if succ_val else sorted(succ_eps)[:2]
    dump_fail = sorted(fail_val)[:2] if fail_val else sorted(fail_eps)[:2]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        have_mpl = True
    except Exception as exc:  # pragma: no cover
        print(f"[p0] matplotlib unavailable ({exc}); CSV only")
        have_mpl = False

    for tag, ep in [("success", e) for e in dump_succ] + [("fail", e) for e in dump_fail]:
        mask = ep_index_per_row == ep
        steps = ep_step_per_row[mask]
        order = np.argsort(steps)
        steps = steps[order]
        gl = G_label[mask][order]
        vp = v_pred[mask][order]
        qp = q_pred[mask][order]
        csv_path = OUT_DIR / f"demo_{tag}_ep{ep}.csv"
        with csv_path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["episode_step", "G_label", "V_pred", "Q_pred"])
            for s, g, v, q in zip(steps, gl, vp, qp):
                w.writerow([int(s), f"{g:.6f}", f"{v:.6f}", f"{q:.6f}"])
        if have_mpl:
            plt.figure(figsize=(8, 4))
            plt.plot(steps, gl, label="G_label", lw=2)
            plt.plot(steps, vp, label="V_pred", lw=1.5)
            plt.plot(steps, qp, label="Q_pred", lw=1.5, ls="--")
            plt.xlabel("episode_step")
            plt.ylabel("return")
            plt.title(f"P0 MC probe — {tag} demo ep={ep}")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(OUT_DIR / f"demo_{tag}_ep{ep}.png", dpi=110)
            plt.close()
    print(f"[p0] dumped CSV/PNG for demos: success={dump_succ} fail={dump_fail} -> {OUT_DIR}")

    # ---- G-label sanity examples for the report ----
    def _demo_g_examples(ep: int) -> dict:
        mask = ep_index_per_row == ep
        steps_ = ep_step_per_row[mask]
        gs_ = G_label[mask]
        order = np.argsort(steps_)
        steps_, gs_ = steps_[order], gs_[order]
        idxs = sorted(set(list(range(min(3, len(gs_)))) +
                          list(range(max(0, len(gs_) - 3), len(gs_)))))
        return {"episode_index": int(ep), "n_rows": int(len(gs_)),
                "examples": [{"step": int(steps_[i]), "G": float(gs_[i])} for i in idxs]}

    g_examples = {}
    if succ_eps:
        g_examples["success"] = _demo_g_examples(succ_eps[0])
    if fail_eps:
        g_examples["fail"] = _demo_g_examples(fail_eps[0])

    # ---- machine-readable summary ----
    summary = {
        "config": {
            "device": args.device, "train_steps": args.train_steps, "batch": args.batch,
            "lr": args.lr, "wd": args.wd, "encode_batch": args.encode_batch,
            "val_frac": args.val_frac, "seed": args.seed, "max_demos": args.max_demos,
            "gamma": GAMMA, "output_reward_coef": OUTPUT_REWARD_COEF,
            "action_horizon": ACTION_HORIZON, "hidden_dims": list(HIDDEN_DIMS),
        },
        "n_frozen": int(n_frozen),
        "n_valid_starts": len(valid_starts),
        "rows_success": n_succ_rows,
        "rows_fail": n_fail_rows,
        "split": {
            "success_train_demos": len(succ_train), "success_val_demos": len(succ_val),
            "fail_train_demos": len(fail_train), "fail_val_demos": len(fail_val),
        },
        "g_label_stats": {
            "overall": {"min": float(G_label.min()), "max": float(G_label.max()),
                        "mean": float(G_label.mean())},
            "success": {"min": float(G_label[is_success_per_row].min()),
                        "max": float(G_label[is_success_per_row].max()),
                        "mean": float(G_label[is_success_per_row].mean())},
            "fail": {"min": float(G_label[~is_success_per_row].min()),
                     "max": float(G_label[~is_success_per_row].max()),
                     "mean": float(G_label[~is_success_per_row].mean())},
        },
        "g_label_examples": g_examples,
        "feature_stats": feat_stats,
        "mae": mae_table,
        "final_train_loss": {"v": v_loss_info["final_loss"], "q": q_loss_info["final_loss"]},
        "train_loss_curve": {"v": v_loss_info["loss_curve"], "q": q_loss_info["loss_curve"]},
        "pred_vs_label_corr_train": {
            "v": v_corr_train, "q": q_corr_train,
            "v_pred_std": v_pred_std_train, "q_pred_std": q_pred_std_train,
        },
        "overfit_sanity": {"v": overfit_v, "q": overfit_q},
        "v_jump": {
            "success_val": {"mean": succ_jmean, "max": succ_jmax, "n": succ_jn},
            "fail_val": {"mean": fail_jmean, "max": fail_jmax, "n": fail_jn},
            "prior_iql_reference": {"fail_mean": "0.05-0.07", "success_mean": "~0.53",
                                    "success_max": "~5-6"},
        },
        "train_steps": args.train_steps,
        "max_demos": args.max_demos,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUT_DIR / "p0_results.json").write_text(json.dumps(summary, indent=2))
    print(f"[p0] wrote summary -> {OUT_DIR / 'summary.json'}")
    print(f"[p0] wrote results -> {OUT_DIR / 'p0_results.json'}")
    print("[p0] DONE")


if __name__ == "__main__":
    main()
