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

**Finalized design (locked with user after the step-3 diagnosis; pure RL, no Q, no fail terminal).** The step-2 baseline confirmed V is a flat "negative-steps-to-success clock" (`V^β`) — pinned at the living-penalty floor `−7.7255/(1−γ⁸)≈−99.9` on fail trajectories, so the TD/GAE advantage is signal-free noise at the true failure onset (see §7-3). The recipe below adds optimism (expectile) + a soft pessimism guardrail (ensemble LCB) to break that floor. **Propagation stays implicit**: the TD-loss target is **1-step** (an n-step / λ-return value target was tried and gave no improvement — could not even detect fail — so it is reverted; see rejected alternatives), and optimism still propagates across the horizon via iterated 1-step expectile backups. `λ` survives only in the read-out GAE for the advantage.

```
target_t = r_chunk + γ^H·(1−done)·V_lcb(s')      # 1-step (macro-step) TD target
V_lcb(s) = mean_k V̄_k(s) − β·std_k V̄_k(s)        # N target heads V̄_k, β≈0.5
L_V      = expectile_τ( target_t − V_k(s_t) )     # τ=0.85, per online head k
A_t      = GAE(λ) on V_lcb                         # per-step advantage (read-out only)
w_t      = f(A_t)                                  # DIPOLE weighted BC
```

Knobs, with values/decisions locked for the first run:

- **`τ=0.85` (optimism)** — expectile, not MSE, in the value fit. Turns `V^β → V*`: recoverable fail frames sit (in encoder space) near success frames → expectile pulls their V up toward the success ramp; doomed frames don't → stay on the floor → the advantage drops at onset. Bounded by the 0.74 recoverable-vs-doomed separability (§5). Data precondition (confirmed with user): success trajectories pass through near-failure regions, so the generalization lift has material. Single fixed τ first, **not** a sweep.
- **`λ≈0.95` (read-out only)** — GAE(λ) is applied **only when reading the advantage off `V_lcb`** (the step-2 machinery), **not** in the value target. Decision after the step-3 finding: an n-step / λ-return *value target* gave no improvement (couldn't even detect fail) → **the TD-loss target stays 1-step** `r + γ^H·V_lcb(s')`. Optimism still propagates across the horizon via iterated 1-step expectile backups (value iteration), one macro-step per update.
- **`β≈0.5` soft-LCB (pessimism guardrail)** — `V_lcb = mean_k − β·std_k` over an N-head V-ensemble. **Soft LCB, not hard min**: the recoverable-state lift lives in the high-epistemic-uncertainty (high head-disagreement) region, which hard-min would suppress exactly where the signal is. Used in **both** the 1-step bootstrap target and the advantage read-out.
- **reward stays sparse** — `reward_mode="-1/0"`, `disc_reward_coef=0`. Pure RL: localization must come from expectile (iterated 1-step backups), not from injecting the discriminator failure score.

N independent V heads (default **N=2**), each own state projector + polyak target; heads must be diversified (independent init + minibatch order / per-head bootstrap mask) or std collapses and LCB→mean (inert). Advantage is GAE(λ) on `V_lcb` at read-out — orthogonal to how the value is learned (which is now plain 1-step expectile-TD).

**Falsifiable acceptance criterion:** on the fail traj, V dips and advantage turns strongly negative at onset (~frame 98) while the prefix stays ≈0; on success V stays high / ramps. If τ=0.85 fails this → escalate (sweep τ; then reconsider reward densification).

### Alternatives considered / rejected for this setting

- **reward densification (`disc_reward_coef>0`)** — the discriminator `failure_score` already localizes onset, so injecting it as dense reward is arguably the strongest single lever (and the *prerequisite* the sparse-reward critique points to: with living-penalty-only reward there is no localization signal for `τ`/`λ` to propagate). **Not taken for now** by decision — keep the method pure-RL so the RL machinery earns its keep; densification is the first escalation if pure expectile is too weak.
- **fail terminal (absorbing failure penalty)** — rejected: an episode-end penalty is discounted to ≈0 over the ~295 steps back to onset (cannot lift the prefix), and *any* pessimistic terminal fights the high-τ expectile. No-terminal is self-consistent with expectile and localizes the correct event (mistake onset, not proximity-to-end).
- **n-step / λ-return value target** — implemented (TODO §7-4, n=3) and **tested empirically: no improvement, failed to even detect fail** → reverted by setting `value_n_step=1` (the identity path; machinery stays, inert). The TD-loss target stays **1-step** (`r + γ^H·V_lcb(s')`); `λ` is retained only in the read-out GAE for the advantage. Propagation is not the lever here — optimism (expectile-`τ`) is.
- **Q / IQL full phase** — redundant under determinism (§3); Q−V unidentifiable under single-action coverage.
- **MC / kNN / supervised success-classifier V** — gives `V^β` (behavior/mean), "small" at critical states, mis-credits the good prefix of fail trajectories. Useful only as a *cheap baseline* to check whether the RL machinery earns its keep.
- **Distributional / quantile V** — its upper quantile is optimism over **transition stochasticity**, not over actions; in a near-deterministic sim the quantile ≈ the mean, so it gives the *wrong kind* of optimism. Use only if the target is risk/uncertainty.
- **IDQL / CRR (flow-policy-proposed counterfactual actions)** — the only route to *true action-level* counterfactual, and it reuses the existing flow policy; reserved as an escalation if V-only optimism proves too weak. Reintroduces OOD → needs ensemble pessimism + BC-anchored sampling.

## 5. Caveats / ceilings

- **Optimism is realized only through generalization** over locally-diverse continuations. If, near a critical state, the data contains only one quality of continuation, the expectile has nothing to be optimistic over and `V → V^β`. Bounded by local continuation diversity × encoder grouping (~0.74).
- **Determinism is load-bearing.** Any per-transition stochasticity makes expectile-on-V over-optimistic (the case Q was built for).
- **Bootstrapping overestimation** (deadly triad) can still compound with high `τ`. Mitigate with the existing polyak target + grad clip, plus a **soft V-ensemble LCB target** (`mean−β·std`, β≈0.5 — *not* hard min, which would suppress the recoverable-state lift; §4). Milder than standard offline RL because there is no OOD-action `max`.
- **Leave-one-trajectory-out** discipline applies to any neighbor-based baseline (disjoint fail-pool invariant): never let a state's own trajectory answer for it.

## 6. Implementation plan (localized)

Touch-points for the §4 finalized method (step-4, not yet coded):

1. `common.py` — add `v_ensemble_size` (default 2), `ensemble_lcb_beta` (default 0.5); set `expectile_tau=0.85` default; `reward_mode="-1/0"` / `disc_reward_coef=0` unchanged. (No new value-target `λ`: the TD-loss stays 1-step; the read-out `gae_lambda` already exists in `vis_qv.py`.)
2. `networks.py` — `VStateNetwork` → N-head ensemble (independent nets, each own state projector).
3. `iql.py::update` — swap `F.mse_loss` → `expectile_v_loss(target − V_k, τ)` per head, where `target` is the **unchanged 1-step** `_bootstrap_target` but bootstrapped with `V_lcb` (mean−β·std over target heads) instead of a single `target_v`; `compute_td_advantage` → uses `V_lcb`. `state_dict`/`load_state_dict` schema bump for the ensemble. (`losses.py::expectile_v_loss` already exists — now used.)
4. `init_iql_qv.sh` — expose `EXPECTILE_TAU` / `LCB_BETA` / `V_ENSEMBLE_SIZE`; the full Q phase stays disabled (V-only).
5. `vis_qv.py` — plot `V_lcb` and the ensemble std band alongside the existing curves (read-out GAE already implemented).

Open implementation detail to settle before coding:

- **Head diversification** (init + data ordering vs per-head bootstrap mask) to keep the LCB std non-degenerate. *(open)*

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
- [ ] 5. Implement the §4 finalized method (code — not started). Scope after the TODO-4 n-step finding: **keep the 1-step TD target** (`value_n_step=1`) and add the two axes that actually matter — **expectile-`τ`=0.85 optimism** (swap MSE→`expectile_v_loss`) + **N=2 V-ensemble soft-LCB** (`mean−β·std`, β≈0.5) in both the 1-step bootstrap target and the advantage read-out. **No λ-return value target** (reverted). `λ` stays read-out-only (existing GAE). See §6 for touch-points; one open detail remains (head diversification).

## 8. References

- IQL — Kostrikov et al., *Offline RL with Implicit Q-Learning*, arXiv:2110.06169.
- IDQL — *Implicit Q-Learning as an Actor-Critic with Diffusion Policies*.
- UDQL — *Bridging MSE Loss and the Optimal Value Function*, arXiv:2406.03324.
- In-sample offline RL (no OOD actions) — openreview `ueYYgo2pSSU`.
- Stitching / RL-vs-BC — BAIR "Should I use offline RL or imitation learning?"; *Model-based Trajectory Stitching*, arXiv:2211.11603.
- GAE — Schulman et al., *High-Dimensional Continuous Control Using GAE*.
- Credit assignment (context) — RUDDER (arXiv openreview `ryeUtP5Il7`), RRD; DWBC (arXiv:2207.10050); VIP (arXiv:2210.00030).
