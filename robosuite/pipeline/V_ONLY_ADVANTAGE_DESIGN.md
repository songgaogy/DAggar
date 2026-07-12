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

Full recipe, **no Q**:

```
target_t = λ-return with bootstrapped V           # propagation knob  (λ)
L_V      = expectile_τ( target_t − V(s_t) )        # optimism knob     (τ)
A_t      = GAE(λ) on V*                            # per-step advantage
w_t      = f(A_t)                                  # DIPOLE weighted BC
```

Three orthogonal knobs:

- **`τ` (optimism)** — expectile instead of MSE in the value fit. `τ≈0.8–0.9`. Turns `V^β → V*`. This is the single change that unlocks good/bad separation.
- **`λ` (propagation)** — λ-return / n-step **bootstrapped** target instead of one-step. `λ≈0.9–0.95`. Full value iteration propagates credit across the horizon (one-step "stitches the wrong segments" at critical states). **Keep `λ<1`**: `λ=1` is pure MC and *loses stitching*.
- **reward densification** — set `disc_reward_coef>0` so propagation is not starved by the sparse terminal reward.

Advantage is GAE / multi-step TD on `V*` — orthogonal to how `V*` is learned.

### Alternatives considered / rejected for this setting

- **Q / IQL full phase** — redundant under determinism (§3); Q−V unidentifiable under single-action coverage.
- **MC / kNN / supervised success-classifier V** — gives `V^β` (behavior/mean), "small" at critical states, mis-credits the good prefix of fail trajectories. Useful only as a *cheap baseline* to check whether the RL machinery earns its keep.
- **Distributional / quantile V** — its upper quantile is optimism over **transition stochasticity**, not over actions; in a near-deterministic sim the quantile ≈ the mean, so it gives the *wrong kind* of optimism. Use only if the target is risk/uncertainty.
- **IDQL / CRR (flow-policy-proposed counterfactual actions)** — the only route to *true action-level* counterfactual, and it reuses the existing flow policy; reserved as an escalation if V-only optimism proves too weak. Reintroduces OOD → needs ensemble pessimism + BC-anchored sampling.

## 5. Caveats / ceilings

- **Optimism is realized only through generalization** over locally-diverse continuations. If, near a critical state, the data contains only one quality of continuation, the expectile has nothing to be optimistic over and `V → V^β`. Bounded by local continuation diversity × encoder grouping (~0.74).
- **Determinism is load-bearing.** Any per-transition stochasticity makes expectile-on-V over-optimistic (the case Q was built for).
- **Bootstrapping overestimation** (deadly triad) can still compound with high `τ`. Mitigate with the existing polyak target + grad clip, plus a **V-ensemble pessimistic-min target** or a light CQL-style regularizer. Milder than standard offline RL because there is no OOD-action `max`.
- **Leave-one-trajectory-out** discipline applies to any neighbor-based baseline (disjoint fail-pool invariant): never let a state's own trajectory answer for it.

## 6. Implementation plan (localized)

1. `iql.py::warmup_value_only` — swap `F.mse_loss` → `expectile_v_loss(target − v_pred, τ)`; `τ` from config (default ~0.85).
2. `iql.py::_bootstrap_target` — one-step → **n-step / λ-return** bootstrapped target, accumulated along `episode_index` boundaries (`Σ γ^{H·k} r_k` mixed by `λ`); reconcile with `freeze_post_success` absorbing anchor.
3. `init_iql_qv.sh` — set `disc_reward_coef>0`; expose `expectile_tau`, `lambda`, `n_step` overrides; the full Q phase can be disabled entirely (V-only).
4. (optional) V-ensemble pessimistic target for overestimation control.
5. Advantage consumer: GAE(λ) on `V*` feeding `AdvantageGProvider`.

Open implementation details to settle before coding: exact λ-return accumulation on the chunk-based buffer across episode boundaries, and its interaction with the freeze-absorbing anchor.

## 7. TODOs

Every time after modification, update current docs `TODOs` section (clearify implementation detail) and [readme](README.md) 

- [x] 1. Refactor: remove Q from the whole codebase; learn the value **V-only** by a TD backup. Done — implementation detail:
  - `iql.py`: dropped the Q ensemble / chunk+action projectors / `q_optim` / `_q_values` / `compute_advantage_for_batch`. `update()` is now a single **V-only MSE-TD** step (absorbs `warmup_value_only`); added `compute_td_advantage` = `r + γ^H·(1-done)·target_V(s') − V(s)`. `state_dict` is V-only; `load_state_dict` rejects any Q-containing checkpoint. Warmup checkpoint schema `3 → 4`.
  - `losses.py`: removed `bellman_q_loss` / `compute_advantage` / `compute_ensemble_advantage`; kept `expectile_v_loss` (unused, reserved for step 2). `networks.py`: removed `QHead` / `ActionProjector`. `common.py`: removed `q_lr` / `q_ensemble_size` / `v_subset_size` / `chunk_proj_dim` / `action_proj_dim`; kept `expectile_tau` (reserved).
  - `warmup.py`: collapsed the value-only + full-IQL loops into one V-only loop (`warmup_value_steps + warmup_full_steps`).
  - `vis_qv.py` / `vis_iql_qv.sh`: removed the Q curves and the best-of-n Q diagnostic; plot V / target-V / TD-advantage / reward only; accept schema v4.
  - `trainer.py` advantage metric → `compute_td_advantage`. Online `AdvantageGProvider` **stubbed** (`NotImplementedError`): the online `DipoleBatch` carries no next-state/reward; the offline path (`OfflineAdvantageGProvider`) already serves the precomputed TD residual.
  - **Deviation from §6-step1 (intentional):** the value loss is plain **MSE-TD**, *not* expectile. This gives the behavior value `V^β` baseline (the §2 "flat" case) on purpose, as the step-1 reference to visualize before adding the optimism knob. The expectile swap is deferred to step 2.
- [ ] 2. Decide the knobs from the step-1 V^β baseline (visualize via `vis_iql_qv.sh`). Candidates, in priority order:
  1. **optimism `τ`** — swap V's MSE for `expectile_v_loss(target − V, τ)` (`τ≈0.8–0.9`), turning `V^β → V*` (§4). `expectile_v_loss` + `expectile_tau` are already wired/reserved.
  2. **propagation `λ`** — extend `_bootstrap_target` from one-step to an n-step / λ-return target accumulated along `episode_index` (keep `λ<1`), reconciled with the `freeze_post_success` anchor.
  3. **reward densification** — `disc_reward_coef>0` so propagation is not starved by the sparse terminal reward.
  - Gate each addition on whether it empirically sharpens good-vs-bad separation over the MSE-TD baseline.

## 8. References

- IQL — Kostrikov et al., *Offline RL with Implicit Q-Learning*, arXiv:2110.06169.
- IDQL — *Implicit Q-Learning as an Actor-Critic with Diffusion Policies*.
- UDQL — *Bridging MSE Loss and the Optimal Value Function*, arXiv:2406.03324.
- In-sample offline RL (no OOD actions) — openreview `ueYYgo2pSSU`.
- Stitching / RL-vs-BC — BAIR "Should I use offline RL or imitation learning?"; *Model-based Trajectory Stitching*, arXiv:2211.11603.
- GAE — Schulman et al., *High-Dimensional Continuous Control Using GAE*.
- Credit assignment (context) — RUDDER (arXiv openreview `ryeUtP5Il7`), RRD; DWBC (arXiv:2207.10050); VIP (arXiv:2210.00030).
