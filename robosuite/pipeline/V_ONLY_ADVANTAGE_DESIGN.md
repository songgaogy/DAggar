# V-only RL Advantage Estimation for DIPOLE

Design note. Estimate an **optimistic state value `V*(s)` by an RL (bootstrapped) backup**, then turn it into per-step advantage via **GAE / multi-step TD**. No Q head. Consumed by DIPOLE advantage-weighted BC.

I am trying to fix the problem, and doing experiments while making/changing plan.  
The plan and TODOs and method are all changable, update in a loop.

---

## 1. Problem

Given trajectories with **known success/fail labels**, score a per-step **advantage** that:

- upweights good steps in success trajectories,
- **localizes the failure onset** in fail trajectories (keep the good prefix, down/negative-weight from the mistake onward),

and is used as the weight `w = f(A_t)` in DIPOLE's weighted-BC objective.

We evaluate **on the training data itself** (in-distribution). This is a **credit-assignment / per-step scoring** problem on labeled, in-support data — *not* an off-policy generalization problem.

## 2. Setting (what is actually true here)

- **Single-action-per-state coverage.** States are visual demo frames, ~never revisited → exactly one logged action per state. In-sample `max_a`/expectile *over actions at a fixed state* is degenerate.
- **Deterministic per-transition dynamics.** MuJoCo/robosuite: given `(s,a)`, `s'` is deterministic. (Only the initial state is randomized across episodes.)
- **Sparse reward.** Current runs use `disc_reward_coef=0` → only the terminal env-success reward. `γ=0.99`, chunk bootstrap `γ^H`.
- **Frozen strong encoder** (DINOv3). Not the lever; but its generalization is what makes cross-state stitching possible.
- **Episode structure is retained** in the offline buffer (`episode_index`, `episode_step`) → n-step / λ-returns and leave-one-trajectory-out are computable.
- **Measured ceiling.** Success-vs-fail states are only marginally separable in encoder features (pooled Mahalanobis AUROC ≈ 0.74, single-frame). Per-step signal quality is bounded by this; the temporal/propagation axis is the lever, not the encoder.

### What went wrong before (from `offline_iql_qv-v2-01` logs)

- Q−V advantage is **unidentifiable**: single-action coverage → `Q−V` is noise. The signal lives in `V`/return (temporal), not in the per-state action counterfactual.
- The value-only warmup uses **MSE TD(0)** → it fits the **mean = behavior value `V^β`**. `v_loss≈1e-3` but that is a **flat, uninformative self-consistent fixed point** under sparse reward, not a good value. `V` is "small" exactly at critical (often-failing) states, so it cannot separate good vs bad steps.

## 3. Key insight — why Q is redundant (not merely hard) here

IQL is two steps:

```
V-step:  L_V = expectile_τ( Q(s,a) − V(s) )
Q-step:  L_Q = MSE( Q(s,a),  r + γ V(s') )
```

Q exists **only to integrate over stochastic dynamics** ("integrate over the dynamics to avoid excessive optimism" — IQL). Under **deterministic** transitions, `r + γV(s')` is an exact function of `(s,a)`, so `Q(s,a) = r + γV(s')` holds exactly and the Q-step is trivial. Substituting into the V-step:

```
L_V = expectile_τ( r + γ V(s') − V(s) )          # V-only expectile-TD
```

**Q cancels identically** — not an approximation. So "only estimate V by RL, no Q" is *justified* in this setting, not just a shortcut. (The one condition: determinism. With transition noise, expectile-on-V is over-optimistic about lucky transitions — that is precisely when you would need Q.)

### Why an *optimistic* RL-V (not MC / behavior-V) is required

- **Behavior value `V^β`** (what MC / kNN / MSE-TD / a supervised success-classifier give) = *average* outcome. Low at recoverable-but-often-failed critical states → cannot make a bad step stand out.
- **Dataset-optimal value `V*`** (RL backup with an upper expectile) = value of the *best locally-reachable continuation*. High at recoverable states.
- With an optimistic baseline, the TD residual discriminates the action through the state it leads to:
  ```
  A_t = r_t + γ V*(s_{t+1}) − V*(s_t)
  ```
  - good action → `s'` recoverable → `V*(s')` high → `A_t ≈ 0` / positive;
  - bad action → `s'` doomed → `V*(s')` low → `A_t` strongly **negative**.

  The negativity comes from falling below an *optimistic* baseline; a behavior-value baseline would hide it (`A≈0`, "as bad as expected").

- This is offline RL's **stitching**: only Bellman backups propagate credit across the horizon to correctly value recoverable states — BC / MC / sequence models cannot. In V-only form, the "max over actions" is realized implicitly as **expectile over locally-indistinguishable states' continuations**, via the frozen encoder's generalization.

## 4. Possible Planned Method

**Current design (updated after the TODO-5 runs; pure RL, no Q, no fail terminal).** The step-2 baseline confirmed V is a flat "negative-steps-to-success clock" (`V^β`) — pinned at the living-penalty floor `−7.7255/(1−γ⁸)≈−99.9` on fail trajectories, so the TD/GAE advantage is signal-free noise at the true failure onset (see §7-3). Two axes were tried against that floor: **expectile optimism (`τ`)** and a **soft-LCB V-ensemble**. **Empirically only the ensemble helps** — the expectile knob gave no gain and *degraded* performance, so it is dropped (`τ=0.5`, i.e. plain MSE; see rejected alternatives). What remains is a **1-step MSE-TD backup onto an N-head ensemble soft-LCB target**. Propagation stays implicit (1-step target; n-step also tried and reverted). `λ` survives only in the read-out GAE for the advantage.

```
target_t = r_chunk + γ^H·(1−done)·V_lcb(s')      # 1-step (macro-step) TD target
V_lcb(s) = mean_k V̄_k(s) − β·std_k V̄_k(s)        # N target heads V̄_k, β≈0.5
L_V      = MSE( target_t − V_k(s_t) )             # per online head k (τ=0.5 ⇒ MSE)
A_t      = GAE(λ) on V_lcb                         # per-step advantage (read-out only)
w_t      = f(A_t)                                  # DIPOLE weighted BC
```

Knobs, with current values/decisions:

- **`τ=0.5` (optimism OFF — MSE)** — the value fit is plain MSE-TD. The expectile `τ>0.5` "optimism" hypothesis (pull recoverable fail frames' V up toward the success ramp) was **tested empirically and rejected**: no gain, degraded results (see rejected alternatives). At `τ=0.5` the `expectile_v_loss` weight is `0.5` on both sides → `0.5·MSE` (the constant only halves the effective LR; same fixed point as MSE), so the code path is unchanged but the asymmetry is inert.
- **`β≈0.5` soft-LCB (the useful lever)** — `V_lcb = mean_k − β·std_k` over an **N=5**-head V-ensemble. **Soft LCB, not hard min**: the recoverable-state signal lives in the high-epistemic-uncertainty (high head-disagreement) region, which hard-min would suppress exactly where the signal is. Used in **both** the 1-step bootstrap target and the advantage read-out. This is what empirically breaks the flat floor.
- **`λ≈0.95` (read-out only)** — GAE(λ) is applied **only when reading the advantage off `V_lcb`**, **not** in the value target (the n-step/λ-return value target gave no improvement → the TD-loss target stays 1-step).
- **reward stays sparse** — `reward_mode="-1/0"`, `disc_reward_coef=0`. Pure RL; no discriminator failure score injected as dense reward.

N independent V heads (default **N=5**), each own state projector + polyak target; heads are diversified by **independent init + per-head Bernoulli bootstrap mask** on the loss (`ensemble_bootstrap_prob=0.5`) or the std collapses and LCB→mean (inert). Advantage is GAE(λ) on `V_lcb` at read-out — orthogonal to how the value is learned (plain 1-step MSE-TD).

### Alternatives considered / rejected for this setting

- **expectile optimism (`τ>0.5`)** — implemented (TODO §7-5, swap MSE→`expectile_v_loss`, `τ=0.85`) and **tested empirically: no gain, degraded performance** → dropped by setting `τ=0.5` (= plain MSE; the expectile code path stays but the asymmetry is inert). The hypothesis was that a high expectile pulls recoverable fail frames' V up toward the success ramp (turning `V^β→V*`); in practice the optimism did not help and hurt. The ensemble soft-LCB (below) is the axis that actually earns its keep. Escalation if a better value is still needed: reward densification.
- **reward densification (`disc_reward_coef>0`)** — the discriminator `failure_score` already localizes onset, so injecting it as dense reward is arguably the strongest single lever. **Not taken for now** by decision — keep the method pure-RL; this is the first escalation now that expectile has been ruled out.
- **fail terminal (absorbing failure penalty)** — rejected: an episode-end penalty is discounted to ≈0 over the ~295 steps back to onset (cannot lift the prefix). No-terminal localizes the correct event (mistake onset, not proximity-to-end).
- **n-step / λ-return value target** — implemented (TODO §7-4, n=3) and **tested empirically: no improvement, failed to even detect fail** → reverted by setting `value_n_step=1` (the identity path; machinery stays, inert). The TD-loss target stays **1-step** (`r + γ^H·V_lcb(s')`); `λ` is retained only in the read-out GAE for the advantage. Propagation is not the lever here — optimism (expectile-`τ`) is.
- **Q / IQL full phase** — redundant under determinism (§3); Q−V unidentifiable under single-action coverage.
- **MC / kNN / supervised success-classifier V** — gives `V^β` (behavior/mean), "small" at critical states, mis-credits the good prefix of fail trajectories. Useful only as a *cheap baseline* to check whether the RL machinery earns its keep.
- **Distributional / quantile V** — its upper quantile is optimism over **transition stochasticity**, not over actions; in a near-deterministic sim the quantile ≈ the mean, so it gives the *wrong kind* of optimism. Use only if the target is risk/uncertainty.
- **IDQL / CRR (flow-policy-proposed counterfactual actions)** — the only route to *true action-level* counterfactual, and it reuses the existing flow policy; reserved as an escalation if V-only optimism proves too weak. Reintroduces OOD → needs ensemble pessimism + BC-anchored sampling.

## 5. Caveats / ceilings

- **Expectile optimism was the intended lever but empirically failed** (dropped, `τ=0.5`/MSE; §4). The theory: optimism is realized only through generalization over locally-diverse continuations, bounded by local continuation diversity × encoder grouping (~0.74) — in practice the gain was absent/negative, likely because near a critical state the data lacks a distinctly better continuation to be optimistic over (so `V → V^β` anyway).
- **Determinism is load-bearing** for any future re-introduction of optimism-on-V (per-transition stochasticity makes expectile-on-V over-optimistic — the case Q was built for). Not active now that `τ=0.5`.
- **Bootstrapping overestimation** (deadly triad) is mitigated by the existing polyak target + grad clip plus the **soft V-ensemble LCB target** (`mean−β·std`, β≈0.5 — *not* hard min, which would suppress the recoverable-state lift; §4). Milder than standard offline RL because there is no OOD-action `max`. This is now the primary — and empirically the only useful — value-side lever.
- **Leave-one-trajectory-out** discipline applies to any neighbor-based baseline (disjoint fail-pool invariant): never let a state's own trajectory answer for it.

## 6. Implementation plan (localized)

Touch-points for the §4 method (coded — TODO §7-5):

1. `common.py` — `v_ensemble_size` (default 2 in code; runs use **N=5** via `init_iql_qv.sh`), `ensemble_lcb_beta` (0.5), `ensemble_bootstrap_prob` (0.5); `expectile_tau` (code default 0.85 but runs use **0.5=MSE** — expectile dropped, see §4); `reward_mode="-1/0"` / `disc_reward_coef=0` unchanged. (No value-target `λ`: TD-loss is 1-step; read-out `gae_lambda` lives in `vis_qv.py`.)
2. `networks.py` — `VEnsemble` (ModuleList of N independent `VStateNetwork`, `forward → (B,N)`).
3. `iql.py::update` — `expectile_v_loss(target − V_k, τ)` per head with the per-head bootstrap `mask` as weights (at `τ=0.5` this is `0.5·MSE`); `target` is the 1-step `_bootstrap_target` bootstrapped with `V_lcb` (mean−β·std over target heads); `compute_td_advantage` uses `V_lcb`. `state_dict`/`load_state_dict` carry `v_ensemble_size`; schema v5.
4. `init_iql_qv.sh` — exposes `EXPECTILE_TAU` (0.5) / `LCB_BETA` (0.5) / `V_ENSEMBLE_SIZE` (5) / `BOOTSTRAP_PROB` (0.5); full Q phase stays disabled (V-only).
5. `vis_qv.py` — plots `V_lcb` + ensemble ±std band (read-out GAE already implemented); accepts schema v5.

> **Note — code defaults vs run values.** The in-code defaults (`common.py`/`train_dipole_rl.yaml`: `expectile_tau=0.85`, `v_ensemble_size=2`) still encode the *original* hypothesis. The *decided* values after the TODO-5 runs are `expectile_tau=0.5` (MSE) and `v_ensemble_size=5`, currently applied only via `init_iql_qv.sh` env overrides. Sync the code/config defaults if these become permanent.

Open implementation detail (settled at coding time):

- **Head diversification** — **resolved (with user): independent random init + per-head Bernoulli bootstrap mask** on the loss (keep-prob `ensemble_bootstrap_prob`, default 0.5). Each head is a fully independent `VStateNetwork` (its own Token/Group projector; Kaiming init draws differ per head), and the per-sample-per-head mask makes the heads see different effective data so the LCB std does not collapse (verified: `v_std_mean` grows to >0 within tens of updates). The final Linear stays zero-init, so std starts at 0 and grows — LCB is inert only at initialization.

## 7. TODOs

Every time after modification, update current docs `TODOs` section (clearify implementation detail) and [readme](README.md) 

- [x] 1. Refactor: remove Q from the whole codebase; learn the value **V-only** by a TD backup. Done — implementation detail:
  - `iql.py`: dropped the Q ensemble / chunk+action projectors / `q_optim` / `_q_values` / `compute_advantage_for_batch`. `update()` is now a single **V-only MSE-TD** step (absorbs `warmup_value_only`); added `compute_td_advantage` = `r + γ^H·(1-done)·target_V(s') − V(s)`. `state_dict` is V-only; `load_state_dict` rejects any Q-containing checkpoint. Warmup checkpoint schema `3 → 4`.
  - `losses.py`: removed `bellman_q_loss` / `compute_advantage` / `compute_ensemble_advantage`; kept `expectile_v_loss` (unused, reserved for step 2). `networks.py`: removed `QHead` / `ActionProjector`. `common.py`: removed `q_lr` / `q_ensemble_size` / `v_subset_size` / `chunk_proj_dim` / `action_proj_dim`; kept `expectile_tau` (reserved).
  - `warmup.py`: collapsed the value-only + full-IQL loops into one V-only loop (`warmup_value_steps + warmup_full_steps`).
  - `vis_qv.py` / `vis_iql_qv.sh`: removed the Q curves and the best-of-n Q diagnostic; plot V / target-V / TD-advantage / reward only; accept schema v4.
  - `trainer.py` advantage metric → `compute_td_advantage`. Online `AdvantageGProvider` **stubbed** (`NotImplementedError`): the online `DipoleBatch` carries no next-state/reward; the offline path (`OfflineAdvantageGProvider`) already serves the precomputed TD residual.
  - **Deviation from §6-step1 (intentional):** the value loss is plain **MSE-TD**, *not* expectile. This gives the behavior value `V^β` baseline (the §2 "flat" case) on purpose, as the step-1 reference to visualize before adding the optimism knob.
- [x] 2. Check baseline run result in `outputs/dipole_rl-iql/offline_iql_qv-v2_explore-baseline`; if healthy, add GAE advantage visualization. Done — findings + implementation detail:
  - **Baseline verdict: no fatal flaw, proceed.** Config: `bs=128`, `value_steps=10000 + full_steps=5000 = 15000`, `v_lr=1e-4`, `γ=0.99`, `polyak=0.005`, `grad_clip=2`, `reward_mode="-1/0"` (per-step −1 living penalty), `disc_reward_coef=0`, `action_horizon=8`. `v_loss` 44→~2 by step 3k, one transient bump (~9) at step ~7.5k that self-heals, settles ~1.6–1.9. `v_mean` 0→≈−65 and plateaus (±6) after ~9k; `target_mean` tracks `v_mean` tightly (TD self-consistent); `td_error_abs≈0.8`. Self-consistency check: `min chunk reward = −7.7255` = exactly H=8 all-−1 discounted-aggregated, and `V≈−65` satisfies `−65 ≈ −5.8 + γ^8·(−65)` → the fit is the behavior value `V^β` (steps-to-success), i.e. the intended step-1 MSE reference.
  - **Two deviations from §2 noted (not blockers):** the baseline ran `reward_mode="-1/0"` (dense −1 living penalty), *not* the doc's "terminal-only sparse reward"; and `expectile_tau=0.7` is logged but the step-1 code path is plain MSE, so τ is inert here.
  - **GAE viz (decisions locked with user):** *overlay* GAE(λ) on the existing advantage subplot (`axes[2]`), keeping the 4-subplot layout — no new/replaced panel; **chunk-macro-step** recursion (matches the MDP `V` was trained on, `γ_eff = γ^H`), evaluated *densely* at every stride-1 window start so the curve aligns with the TD one; single `λ=0.95` exposed as a CLI/env knob.
  - `vis_qv.py`: added `--gae-lambda` (default 0.95). Key identity used: the existing `advantage_td1[t]` **is** the one-macro-step TD residual `δ_t = r_chunk(t) + γ^H·mask_t·V(s_{t+H}) − V(s_t)`, so GAE is computed directly on `δ` with no extra encoder passes. Standard backward recursion on the H-strided chain `A_t = δ_t + (γ^H λ)·mask_t·A_{t+H}` (mask = `has_valid_next`, so the bootstrap and the propagation are both dropped at terminal/OOD windows — never propagates through an off-support `next_v`). Written to every row as `advantage_gae` (CSV + plotted). Plot overlays it on `axes[2]` (blue) beside the TD advantage (red), NaN-masked identically where `has_valid_next==0`. `summary.json` records `gae_lambda`. The `_nonoverlap` (stride-H) plot reuses the same per-row GAE, which on the phase-0 chain is exactly the disjoint-chunk GAE.
  - `vis_iql_qv.sh`: added `GAE_LAMBDA` env var (default 0.95) → `--gae-lambda`, documented in the header and echoed at startup.
  - **Verified end-to-end** on the baseline ckpt (episode 58, fail split): GAE curve renders below the TD curve (accumulated future living-penalty residuals) and converges onto TD at the trajectory/window-cap edge where the forward chain is a single δ — matches the recursion; unit-tested the recursion against a brute-force reference (exact match).
- [x] 3. step-2 GAE works (slight improvement, `outputs/dipole_rl-iql/offline_iql_qv-v2_explore-GAE`). Brainstorm/critique done, design locked with user — no code this step. **Diagnosis** (numbers, `.../vis/*/steps.csv`): V is a flat "negative-steps-to-success clock" (`V^β`) — *success* ep8 ramps −88→−6, *fail* ep58 pinned at ≈−99 (living-penalty floor, mathematically forced) → TD/GAE advantage is `±1` noise with zero response at the true onset (frame ~98); the discriminator `failure_score` localizes onset but `disc_reward_coef=0` keeps it out of V; GAE just smooths a signal-free residual. **Critique, finalized method + locked decisions, rejected alternatives → §4**; **implementation touch-points + open details → §6**.
- [x] 4. implement n-step return, default 3. Done — decisions + implementation detail:
  - **Locked decisions (with user):** (1) **n counts chunk-macro-steps**, not env-frames — `n=3` bootstraps 3 chunks = 3H=24 frames: `target = Σ_{k=0}^{n_eff-1} γ^{kH}·R_k + γ^{n_eff·H}·(1−done)·V_target(s_{+n_eff})`. (2) **MSE only** (no expectile/ensemble yet — that stays step 5); single V head. (3) **Variable n_eff** — accumulate to the last full in-episode chunk, bootstrap at the last valid state; a genuine `success` terminal stops the chain and masks the bootstrap; a fail rollout running off the recording boundary is a *truncation* (n_eff shortens, bootstrap kept), matching the 1-step done semantics. (4) **Advantage read-out stays 1-step** — `compute_td_advantage` and the `vis_qv` GAE(λ) curve are **unchanged**; only the value *training target* is n-step.
  - **Consequence:** `IQLStepBatch` now carries BOTH the single-chunk fields (`rewards`/`dones`/`next_v_state_feature`, fed to the 1-step advantage) AND four new n-step fields (`nstep_rewards`, `nstep_bootstrap_feature`, `nstep_dones`, `nstep_discount`, fed to the target). `value_n_step=1` makes them coincide exactly → n-step reduces to the legacy 1-step backup.
  - `common.py`: added `IQLConfig.value_n_step:int=3` (+ `__post_init__ >=1` assert; left the misnamed unused `n_step_aggregate` untouched to avoid a semantic clash); added the 4 fields to `IQLStepBatch` + `.to()`.
  - `replay.py`: new `_nstep_chain(start)` walks `s, s+H, …` ≤ `value_n_step` chunks along the episode (validity = full in-episode window; per-chunk reward via the same `Σγ^i·r` as `aggregate_chunk_reward`; done via the existing per-step `success` semantics), returning `(partial_reward=Σ_{k≥1}(γ^H)^k R_k, nstep_discount=γ^{n_eff·H}, nstep_done, bootstrap_obs)`. `_next_obs_for` refactored to a lock-free `_resolve_next_obs` reused by the walk. `_build_step_batch` keeps the 1-chunk fields, adds the walk, encodes the bootstrap state (`encoder.encode_state`), and sets `nstep_rewards = rewards + partial` (so the k=0 term reuses the exact 1-chunk reward → guarantees the n=1 identity). The preencode cache **inherits** the new fields for free: they were added to `field_names` and to `IQLPreencodedReplayCache` (ctor/shape-check/`sample_step_batch`), since the cache is built by calling `_build_step_batch`.
  - `iql.py`: `_bootstrap_target` now reads the `nstep_*` fields; `update`/MSE/polyak and the 1-step `compute_td_advantage` unchanged (module docstring updated to flag the intentional n-step-target vs 1-step-advantage split).
  - `warmup.py`: `schema_version 4 → 5`, records `value_n_step` in `encoder_meta`. `config/train_dipole_rl.yaml`: `value_n_step: 3`. `init_iql_qv.sh`: `N_STEP` env (default 3) → `algorithm.q_learning.config.value_n_step`. `utils/vis_qv.py`: accepts schema v4 **or** v5 (V net identical).
  - **Tests:** `tests/test_nstep_return.py` (new) — n=1 identity, n=3 brute-force interior, boundary truncation (n_eff<n), success-terminal stop+mask; `tests/test_iql.py` — fixture carries the new fields + a `_bootstrap_target` wiring test; `tests/test_preencode_cache.py` — n-step fields added to the cache==live equivalence check (and repaired a pre-existing stale `_FakeEncoder` missing `encode_state_and_chunk`). All 19 green.
  - **Empirical result (n=3): no good — REVERTED to 1-step.** n-step gave no improvement and **could not even detect fail** (the propagation axis alone, on the MSE `V^β` target, does not break the flat floor). Decision with user: **keep the TD-loss target 1-step** → set `value_n_step=1` (the built-in n=1 identity path; the n-step machinery stays but is inert at the default). Propagation is not the lever here; the optimism (expectile-τ) is. `λ` for propagation survives only in the read-out GAE.
- [x] 5. Implement the §4 finalized method. Done — decisions + implementation detail:
  - **Prereq revert.** The whole n-step machinery (TODO 4) was empirically bad and reverted **at the source level** (`git revert` of the n-step commit), not left inert at `value_n_step=1`. So the TD target is inherently **1-step** again (`r + γ^H·(1−done)·V(s')`) and `IQLStepBatch` carries no `nstep_*` fields. This design doc (with the n-step lesson) was preserved across the revert.
  - **Two axes added on the 1-step target:** (1) **expectile-`τ`=0.85 optimism** — `update()` swaps `F.mse_loss` → `expectile_v_loss(target − V_k, τ)` per head. (2) **N=2 V-ensemble soft-LCB** — value is now `VEnsemble` of N independent `VStateNetwork` heads; the scalar value consumed everywhere is `V_lcb = mean_k − β·std_k` (β=0.5, population std so N=1 ⇒ std=0). LCB is used in **both** the bootstrap target (`_bootstrap_target` → `target_v_lcb`) and the advantage read-out (`compute_td_advantage` → online/target `V_lcb`). `λ` stays read-out-only (existing GAE, unchanged).
  - **Head diversification (resolved, §6):** independent Kaiming init + **per-head Bernoulli bootstrap mask** on the loss (`ensemble_bootstrap_prob`=0.5, applied via `expectile_v_loss(weights=mask)`). Verified `v_std_mean`>0 within tens of updates → LCB non-degenerate.
  - `networks.py`: new `VEnsemble(nn.Module)` (ModuleList of N `VStateNetwork`, `forward → (B,N)`); `VStateNetwork` unchanged. `common.py`: `IQLConfig` gains `v_ensemble_size=2`, `ensemble_lcb_beta=0.5`, `ensemble_bootstrap_prob=0.5`; `expectile_tau` default `0.7→0.85`; `__post_init__` asserts. `iql.py`: `v`/`target_v` → `VEnsemble`; `_lcb` + public `v_lcb`/`target_v_lcb`; expectile+mask `update` with new `v_std_mean` metric; `state_dict` records `v_ensemble_size`, `load_state_dict` rejects N-mismatch (and old single-head v4). `losses.py::expectile_v_loss` unchanged (already handles `(B,N)`+weights).
  - Consumers → LCB: `offline/utils/advantage.py` (precompute), `utils/vis_qv.py` (`V_lcb` + ensemble ±std band; accepts **schema v5**). `train_offline_dipole.py::_freeze_iql` needs no change (`VEnsemble` is an `nn.Module`; `.parameters()` covers all heads). `warmup.py`: schema `4→5`, records ensemble/τ in `encoder_meta` + TB. Config `train_dipole_rl.yaml` + `init_iql_qv.sh` (`EXPECTILE_TAU`/`V_ENSEMBLE_SIZE`/`LCB_BETA`/`BOOTSTRAP_PROB`).
  - **Tests:** `tests/test_iql.py` — ensemble fixture + forward shape, `V_lcb` reduction (N=1 ⇒ std=0), `_bootstrap_target` uses LCB, `update` diversifies heads (`v_std_mean`>0), state_dict round-trip + N-mismatch rejection. `tests/test_preencode_cache.py` — repaired the pre-existing stale `_FakeEncoder` (missing `encode_state_and_chunk`); no n-step fields.
  - **Not run at implementation time:** training/vis deferred (implementation + unit tests only).
- [x] 6. Empirical follow-up on the two axes (TODO-5 runs). Findings + decision:
  - **Expectile optimism (`τ`): rejected.** `τ=0.85` gave **no gain and degraded** performance vs MSE → set `τ=0.5` (plain MSE; the `expectile_v_loss` path stays but is symmetric/inert). Optimism-on-V is not the lever here (consistent with the §5 ceiling: near critical states the data lacks a distinctly better continuation to be optimistic over). Moved to §4 rejected alternatives.
  - **Ensemble soft-LCB: kept, it works.** This is the axis that empirically breaks the flat `V^β` floor. Head count raised **N=2 → N=5** (`V_ENSEMBLE_SIZE=5`) for a smoother/robuster LCB std; β=0.5, bootstrap-prob=0.5 unchanged.
  - **Applied via `init_iql_qv.sh`** (`EXPECTILE_TAU=0.5`, `V_ENSEMBLE_SIZE=5`); in-code defaults (`common.py`/`train_dipole_rl.yaml`: `expectile_tau=0.85`, `v_ensemble_size=2`) still hold the original hypothesis — sync if these become permanent (see §6 note). No code-logic change: `τ=0.5` and larger N are pure config.

## 8. References

- IQL — Kostrikov et al., *Offline RL with Implicit Q-Learning*, arXiv:2110.06169.
- IDQL — *Implicit Q-Learning as an Actor-Critic with Diffusion Policies*.
- UDQL — *Bridging MSE Loss and the Optimal Value Function*, arXiv:2406.03324.
- In-sample offline RL (no OOD actions) — openreview `ueYYgo2pSSU`.
- Stitching / RL-vs-BC — BAIR "Should I use offline RL or imitation learning?"; *Model-based Trajectory Stitching*, arXiv:2211.11603.
- GAE — Schulman et al., *High-Dimensional Continuous Control Using GAE*.
- Credit assignment (context) — RUDDER (arXiv openreview `ryeUtP5Il7`), RRD; DWBC (arXiv:2207.10050); VIP (arXiv:2210.00030).
