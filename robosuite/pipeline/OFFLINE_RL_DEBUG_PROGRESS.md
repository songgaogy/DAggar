# Offline IQL Warmup — Debug Progress (P0 / P1)

Branch: `dipole-rl/v1-debug` · Task: `PickPlaceCereal` · Companion to `OFFLINE_RL_DEBUG.md`.

This note records the experiments run so far, where their results live, and what they mean. It is a handoff doc — read `OFFLINE_RL_DEBUG.md` first for the original symptom and hypotheses.

---

## 1. The problem under investigation

Offline IQL Q/V warmup on 50 `success_rollout` + 50 `fail_rollout` demos (expert=0):

- **fail** rollouts are fit well — they collapse to the constant Bellman fixed point `V ≈ -10` (`= -0.0773 / (1 - 0.99^8)` with `reward_mode=-1/0`, `output_reward_coef=0.1`, `γ=0.99`, `H=8`).
- **success** rollouts are fit poorly: high Q-TD MAE (~0.89) and large local V discontinuities (reported mean adjacent-window V jump ~0.53, max ~5–6) on the `success_rollout-val` split.

Goal of this round: **locate the root cause only** (no algorithm change yet).

Agreed diagnostic plan: **P0 → P1 → P2 → P4**.
- P0 — MC return-to-go regression sanity check.
- P1 — target_mode comparison (`iql_v | sarsa_q | mc_return`).
- P2 — pessimism / ensemble ablation.
- P4 — is IQL itself unsuitable for this (fixed-rollout evaluation) task.

Shared facts (verified in code):
- Reward each step ≈ `-0.1` (`0` on success frames).
- `freeze_post_success` collapses each success demo's post-success drift into one frozen absorbing `(s_{t_s}, a_{t_s})` anchor before IQL consumes it.
- Reusable assets on disk:
  - Raw assembled transitions: `data/PickPlaceCereal/offline_data/iql_offline_transitions.pt` (40000 transitions / 39300 valid windows; reloadable via `FlowDaggerReplayBuffer.load`).
  - nnPU ckpt: `checkpoints/dyn_disc/pu_bce_eval_robosuite/run_20260619_194127_PickPlaceCereal/checkpoints/pu_bce_head.pth`
  - flow init ckpt: `checkpoints/multitask_6/flow_multi_ep0100.pt`
  - True held-out splits (NOT yet used): `data/PickPlaceCereal/success_rollout-val/`, `data/PickPlaceCereal/fail_rollout-val/` (separate 2026-06-18 recordings, 50 each).

---

## 2. Experiment P0 — MC return-to-go regression (capacity sanity check)

**Question.** Remove bootstrapping entirely: directly regress the same Q/V head architectures (on frozen encoder features) to the Monte-Carlo return-to-go `G_t = Σ_{k≥t} γ^{k-t} r_k`. If this fits, head + representation capacity is sufficient and the problem is bootstrapping; if not, the problem is upstream (features / head / optimization).

**Method.** Reloaded the saved transitions (skips HDF5+robosuite), applied `_freeze_post_success_tail`, encoded every valid window ONCE with the frozen `SharedDynamicsEncoder` (~4 min DINOv3), and regressed `VNetwork`/`QChunkNetwork` to `G_t` via MSE (25000 steps, batch 256, lr 3e-4). Train/val split by demo (40 train / 10 val per class, seed 42).

**Script (additive, uncommitted):** `robosuite/pipeline/algorithms/q_learning/utils/mc_return_probe.py`

**Re-run:**
```
cd <repo-root> && MUJOCO_GL=egl PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 \
  python -m robosuite.pipeline.algorithms.q_learning.utils.mc_return_probe \
  --device cuda:0 --train-steps 25000 --batch 256 --lr 3e-4
# reuse encoding: --reuse-features outputs/dipole_rl-debug/p0_mc_probe/preencoded_features.pt
```

**Results saved at:**
- `outputs/dipole_rl-debug/p0_mc_probe/p0_results.json`
- `outputs/dipole_rl-debug/p0_mc_probe/preencoded_features.pt` (**4.18 GB**: per-window `q_chunk_feature` (14272-d), `v_state_feature` (12288-d), `valid_starts`, `episode_index_per_row`, `episode_step_per_row`, `G_label`, `is_success_per_row`)
- `outputs/dipole_rl-debug/p0_mc_probe/demo_{success,fail}_ep*.csv/.png`

**Findings:**

| metric | value |
|---|---|
| train MAE V/Q (success) | 0.022 / 0.032 |
| train MAE V/Q (fail) | 0.040 / 0.052 |
| pred-vs-label corr (train) | **0.9999 / 0.9998** |
| single-batch overfit | MSE 43 → **6.8e-7** |
| feature variance | per-dim std ~0.40; `all_rows_identical=False` |
| **val MAE V (success / fail)** | **0.64 / 1.65** |

G-labels verified correct (success ramps to 0 at the frozen tail; fail ranges −9.82 → −0.77, never reaching 0).

**P0 conclusion.** Head capacity + wiring are fine (train fits to ~0, overfits a single batch). **The original P0 verdict "capacity sufficient ⇒ bootstrapping is the culprit" was premature** — it ignored that val MAE was already 0.64 (a train→val gap), and it never compared against bootstrap arms. P1 supplies that comparison.

---

## 3. Experiment P1 — target_mode comparison (`iql_v | sarsa_q | mc_return`)

**Question.** On the SAME fixed windows, does the value-learning target matter? Per `OFFLINE_RL_DEBUG.md` mapping: `mc good + sarsa good + iql bad` ⇒ blame the IQL action-free V-bootstrap; `mc good + sarsa bad` ⇒ TD propagation itself is unstable.

**Method (no re-encoding, no 16 GB reload).** Reused `p0_mc_probe/preencoded_features.pt`. Reconstructed per-window chunk reward and the next-state index from cached labels via the telescoping identity `r_chunk[s] = G[s] − γ^H · G[s+H]` (verified exact: max error 0.0 over 31k non-success chunks), and `s' = s_{t+H}` = the cached row at `(episode e, step t+H)` — exactly production's `_next_obs_for` primary path.
- `iql_v`: imports the production `IQLLearner`/`IQLConfig`/`IQLStepBatch` with the exact `train_dipole_rl.yaml` config (K=10, subset=2, τ=0.7, polyak 0.005) and the production schedule (20000 value-only + 10000 full). Faithful reproduction.
- `sarsa_q`: FQE target `r_chunk + γ^H (1-done) Q_target(s', a'_recorded)`; fail tail mirrors production (never `done`, keep bootstrapping; 400 unmapped fail tail rows dropped).
- `mc_return`: P0's supervised regression — the no-bootstrap oracle / upper bound.
- Eval split identical to P0 (success_val = [13,17,19,25,26,30,32,39,45,48]).

**Script (additive, uncommitted):** `robosuite/pipeline/algorithms/q_learning/utils/p1_target_mode.py`

**Re-run:**
```
cd <repo-root> && MUJOCO_GL=egl PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 \
  python -m robosuite.pipeline.algorithms.q_learning.utils.p1_target_mode --device cuda:0
```

**Results saved at:**
- `outputs/dipole_rl-debug/p1_target_mode/p1_results.json`
- `outputs/dipole_rl-debug/p1_target_mode/demo_{success,fail}_ep*.csv/.png`

**Findings.** The three arms are **statistically indistinguishable** on success-val:

| metric | iql_v | sarsa_q | mc_return (oracle) |
|---|---|---|---|
| success-val MAE vs G | **0.589** | 0.623 | 0.625 |
| success jump mean / max | 0.102 / 5.34 | 0.156 / 5.84 | 0.098 / 4.62 |
| fail jump mean / max | 0.107 / 4.20 | 0.110 / 6.31 | 0.151 / 4.83 |
| fail mean value | −8.94 | −9.30 | −7.15 (truncation-biased) |

Neither DEBUG.md branch fired — `iql_v` is actually the *best* and its V is smooth.

**Decisive per-window evidence (ep13, t_s=171).** True `G ≈ -1.0` (only 11 steps from success), but ALL THREE arms — including the bootstrap-free MC oracle — collapse to deeply pessimistic and mutually inconsistent values:

| step | G_label (truth) | iql_v | sarsa_q | mc_return |
|---|---|---|---|---|
| 160 | −1.05 | −6.93 | −9.68 | −6.21 |
| 161 | −0.96 | −5.92 | −6.27 | −5.03 |

**Structural edge case (ep46, kept).** ep46 *does* succeed — first success at `t_s=393`, 7 post-success frames (393–399), `freeze_post_success` applied — but with `H=8` and a 400-frame rollout, steps ≥393 cannot form a valid chunk start (need 8 frames, only 7 remain). No valid window has `G=0`; P1 infers `t_s=∞` via `G==0` and drops the last 8 windows (steps 385–392, `done==0`, no `next_row`). All other 49 success demos are fine (same tail windows reach `done==1` via frozen anchor). ep46 is in **train**, not val — does not affect P1 metrics. **Decision: keep ep46; be aware it contributes no terminal-anchor windows.**

> **Data-specific (c) — no action needed.** ep46-class demos are not mislabeled: success simply occurs too late in the rollout (first success near the episode end, fewer than `H=8` frames remain after `t_s`). With `freeze_post_success`, no valid chunk start can reach `G=0` / a frozen terminal anchor. This pattern appears only in this dataset's recordings, not a pipeline bug. **Ignore for debugging; do not clean or change code/data for it.**

**P1 conclusion.** The value-learning target is **not** the lever. The residual error is target-mode-independent and concentrated on specific held-out demos that **even the no-bootstrap oracle cannot fit** ⇒ the bottleneck is upstream of the target: the frozen-encoder feature generalization on held-out success demos.

---

## 4. Combined synthesis (P0 + P1)

Layer-by-layer elimination:
- **Ruled out — head capacity** (P0: train fits to ~0, single-batch overfit).
- **Ruled out — target mode / bootstrapping structure** (P1: iql_v ≈ sarsa_q ≈ mc_return on val; error concentrated where even the oracle fails).
- **Points to — a train→val generalization gap** (MAE 0.02 → 0.6) driven by the frozen encoder features + limited data (40 train demos), concentrated on hard held-out demos (e.g. ep13 confidently mis-predicted near success).

**Reframes the original symptom.** The "large V discontinuity" (max jump ~5) reproduces *identically* across all three arms including the MC oracle ⇒ it is the **real, expected success cliff** (terminal value jumping to 0), not instability.

**Partial answer to P4.** IQL's expectile-V objective is most likely **not** the primary culprit for this fixed-rollout evaluation task.

---

## 4b. Experiment P1.5 — TRUE held-out reproduction + data-size sweep

**Question.** Two things at once: (1) does the production symptom reproduce on the **separate** 2026-06-18 recording (`success_rollout-val` / `fail_rollout-val`) that P0/P1 never touched? (2) Split the gap: (a) data-limited overfit (held-out MAE falls with more train demos) vs (b) frozen-encoder representation ceiling (held-out MAE stuck at a floor — even the MC oracle).

**Method (no train re-encode; held-out encoded once).**
- Held-out assembled by driving the production `warmup` with 0 training steps + `save_data` over the `-val` splits → `data/PickPlaceCereal/offline_data_heldout/iql_offline_transitions.pt` (50 success + 50 fail, 40000 transitions). Confirmed **same 128×128 image resolution** as train; `buffer.load` restores `image_size`, so train (P0 cache) and held-out are encoded identically — no resolution confound.
- "Reuse P0/P1 heads" = retrain the SAME architectures + recipes + seed 42 **deterministically** (P0/P1 never saved head weights). Train features are **subset by demo** from P0's `preencoded_features.pt` (covers all 50+50 train demos) — no train re-encode. Only the held-out recording is freshly DINOv3-encoded, once (4:23, 39300 windows).
- Sweep N ∈ {10,20,30,40,50} success+fail demos (nested sorted-ep prefixes); all three arms (`iql_v`/`sarsa_q`/`mc_return`, imported from P1); every point evaluated on the **same full held-out** set.

**Script (additive, uncommitted):** `robosuite/pipeline/algorithms/q_learning/utils/p1_5_heldout_sweep.py`

**Re-run:**
```
cd <repo-root> && MUJOCO_GL=egl PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
  python -m robosuite.pipeline.algorithms.q_learning.utils.p1_5_heldout_sweep --device cuda:0
# reuse held-out encoding: --reuse-heldout-features outputs/dipole_rl-debug/p1_5_heldout/heldout_features.pt
```
Held-out assembly (one-off, needs robosuite env; encoding itself does not):
```
python -m robosuite.pipeline.algorithms.q_learning.warmup seed=42 env.environment=PickPlaceCereal \
  data.task_name=PickPlaceCereal runtime.init_checkpoint=checkpoints/multitask_6/flow_multi_ep0100.pt \
  algorithm.discriminator.checkpoint=<NNPU_CKPT> algorithm.q_learning.config.device=cuda:0 \
  algorithm.q_learning.warmup_value_steps=0 algorithm.q_learning.warmup_full_steps=0 \
  +warmup.output_path=outputs/dipole_rl-debug/p1_5_heldout/throwaway_iql_state.pt +warmup.batch_size=128 \
  'warmup.demo_splits=[success_rollout-val,fail_rollout-val]' \
  warmup.num_trajectories.save_data=true warmup.num_trajectories.save_dir=offline_data_heldout
```

**Results saved at:**
- `outputs/dipole_rl-debug/p1_5_heldout/p1_5_results.json`
- `outputs/dipole_rl-debug/p1_5_heldout/{heldout_features.pt (4.18 GB), sweep_curve.png, demo_{success,fail}_ep*.csv/.png}`

**Finding 1 — production symptom REPRODUCES on the true held-out.**

| metric | production | true held-out (N=50) | verdict |
|---|---|---|---|
| success Q-TD MAE | ~0.89 | **0.61** | same high ballpark ✓ |
| fail Q-TD MAE | ~0.19 | **0.19** | exact ✓ |
| success **max** V-jump | ~5–6 | **6.7** | reproduces ✓ |
| success **mean** V-jump | ~0.53 | 0.12 | not reproduced — known freeze-vs-raw measurement artifact (see §5); the meaningful **max** does reproduce |

The success/fail asymmetry holds in the correct metric: **fail generalizes** (Q-TD 0.19, the constant fixed point), **success does not** (Q-TD 0.61; MAE-vs-G 0.66–0.88 = **33–44× the train MAE 0.02**).

**Finding 2 — data-size sweep ⇒ (b) representation ceiling, not (a) data-limited.** Held-out success MAE-vs-G:

| N (demos/class) | iql_v | sarsa_q | **mc_return (oracle)** |
|---|---|---|---|
| 10 | 0.95 | 1.32 | 0.81 |
| 20 | 0.82 | 1.26 | 0.74 |
| 30 | 0.89 | 1.13 | 0.75 |
| 40 | 0.95 | 1.17 | 0.80 |
| 50 | 0.88 | 1.13 | **0.66** |

The **MC oracle** (no bootstrap; fits train to 0.02; the best-possible head on these features) plateaus at **0.66–0.81** across the N=10→50 sweep, with only a weak, **non-monotone** drift (drop 0.15, rebounds at N=40). `iql_v`/`sarsa_q` are flat-to-noisy (drops 0.07 / 0.19) and strictly worse than the oracle on held-out.

**What this DOES establish.** A real train→held-out generalization gap that is **target-mode-independent** (even the bootstrap-free MC oracle hits it) and **not head capacity** (train fits to 0.02). The production symptom is genuine, not an in-distribution artifact.

**What this does NOT establish (correction — earlier draft overreached).** It does **not** isolate the *cause* of the gap, and in particular it does **not** prove an "encoder representation ceiling." Two reasons the encoder is almost certainly **not** the lever:
- The encoder is a **strong, fixed DINOv3** pretrained on a large in-environment dynamics dataset, and it is **not finetuneable for RL** in this setup — so "finetune the encoder" is neither likely-correct nor actionable. (User feedback, 2026-06-23.)
- The data-size sweep being flat does **not** imply "more data can't help / features are the ceiling." The sweep added demos **from the same recording**, and **frames within a demo are highly correlated** → 50 demos ≪ 50 independent samples. The flatness is equally consistent with the gap being driven by something the sweep never varied.

**The unexamined, more likely confound: an over-parameterized, under-regularized head.** The oracle head takes a **14272-d (q_chunk) / 12288-d (v_state)** feature into a **512×512 MLP** at **weight_decay=1e-6** (≈none), 25k steps — millions of params on few *effective* independent samples. Fitting train to 0.02 with a held-out floor of 0.66 is textbook head overfitting, and the right axis to test it is **head regularization / capacity / feature compression**, NOT #same-recording demos. This is fully consistent with "the encoder is fine" — the overfit lives in the Q/V head, which production shares.

**P1.5 conclusion (revised).** The warmup failure is a **genuine, target-mode-independent generalization gap**, but its **cause is not yet isolated**. The leading (untested) hypothesis is **head-side overfitting under near-zero regularization** on a very high-dim feature with few effective samples — an encoder-free problem. Alternatives still open: aleatoric under-determination of `V(s)` (return depends on future stochastic actions, not on `s` alone) and recording-to-recording covariate shift. **Encoder finetuning is de-prioritised** per the above. Next step = head-side / data-diversity diagnostics (see §6), **not** encoder work.

---

## 5. ✅ Caveat resolved (by P1.5)

The original open caveat — *P0/P1's "val" was in-distribution (held-out demos cut from the SAME 50-demo recording); the production symptom on the separate `success_rollout-val` file had never been touched* — is **now closed by P1.5 §4b**: the symptom reproduces on that separate recording (success Q-TD MAE 0.61, fail 0.19, max V-jump 6.7).

The 5× **mean**-jump discrepancy (P1 in-dist 0.10 vs production 0.53) is **not** a "true held-out is harder" effect: the held-out mean jump is also low (0.12) under `freeze_post_success`. The 0.53 is a measurement-context artifact (raw post-success drift / different window set), confirming the original-symptom reframe in §4 — the big jump is the **real success cliff** (max ~5–7), not instability.

(Note: an earlier subagent claim that "0.53 came from the online pipeline" is a factual error — that number is an offline-warmup figure.)

---

## 6. Hypothesis — gap confirmed, CAUSE still open (encoder ruled out as the lever)

P1.5 confirms a real, target-mode-independent gap but does **not** resolve its cause. Revised candidate table:

| candidate cause | signature to test | status after P1.5 |
|---|---|---|
| head capacity | train can't fit | **rejected** (P0/P1.5: train→0.02, single-batch overfit) |
| target mode / bootstrapping | arms differ | **rejected** (P1 + P1.5: even MC oracle fails) |
| **head overfit (over-param + under-reg)** | held-out MAE **falls** as weight_decay↑ / head shrinks / features compressed (PCA) / early-stop | **LEADING, UNTESTED** — 14272-d→512×512 MLP at wd=1e-6 is wildly over-parameterized for the effective sample count |
| aleatoric `V(s)` under-determination | Q(s,a) generalizes ≫ V(s) on held-out | **open** — return depends on future stochastic actions, not on `s` alone |
| recording-to-recording covariate shift | held-out features OOD vs train (kNN dist / linear separability) | **open** — held-out is a separate 2026-06-18 recording |
| encoder representation ceiling | held-out floor persists *after* the above are exhausted | **de-prioritised** — strong fixed DINOv3, not RL-finetuneable; NOT the lever (user feedback 2026-06-23) |
| label noise (ep46-class) | — | data-specific, ignored (see §3) |

**Why the data-size sweep did NOT settle (a):** it varied #demos from the *same* recording (correlated frames → few effective samples), not the axes that actually move head overfit (regularization / capacity / feature dim) or covariate shift (data *diversity*).

**Actionable next levers — all encoder-free (reuse cached features, no re-encode):**
1. **Head regularization / capacity sweep** (decisive for the leading hypothesis): fix the MC oracle, sweep `weight_decay ∈ {1e-6,1e-3,1e-1}` × head width {512,128,32} × dropout/early-stop × PCA-reduced features. Held-out MAE falling ⇒ head overfit (encoder fine); flat ⇒ escalate.
2. **Q vs V on held-out**: if Q(s,a) ≫ V(s) in generalization, the residual is aleatoric V under-determination, not representation.
3. **Feature-space OOD probe**: train-vs-held-out separability + kNN distance ⇒ is it covariate shift (fix = data diversity, still encoder-free)?
4. De-prioritised: encoder finetune/replace; more same-recording demos; RL-objective changes (ruled out).

---

## 7. Status

- [x] P0 — MC return-to-go sanity check (capacity sufficient).
- [x] P1 — target_mode comparison (target mode is not the lever).
- [x] **P1.5 — true held-out reproduction + data-size sweep.** Symptom reproduces; gap is real & target-mode-independent. **Cause NOT isolated** — earlier "(b) encoder ceiling" verdict retracted (see §4b/§6).
- [ ] P2 — pessimism ablation. **De-prioritised** — value-target is not the lever (P1).
- [x] P4 — IQL suitability: **not the primary culprit** (even the MC oracle fails on held-out). Closed.
- [ ] **Next (encoder-free)** — head regularization/capacity sweep (leading hypothesis: under-regularized over-param head overfit), then Q-vs-V + feature-OOD probes (see §6). Encoder finetune de-prioritised.

**Artifacts root:** `outputs/dipole_rl-debug/{p0_mc_probe,p1_target_mode,p1_5_heldout}/`
**New scripts (uncommitted):** `robosuite/pipeline/algorithms/q_learning/utils/{mc_return_probe,p1_target_mode,p1_5_heldout_sweep}.py`
