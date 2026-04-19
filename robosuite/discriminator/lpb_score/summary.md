# SσDC v1 — Single-σ Diffusion Classifier (lpb_score)

## 1. Goal

`lpb_score` trains a task-conditioned latent DSM and uses it as an offline
anomaly detector on trajectory prefixes. SσDC v1 replaces the previous
`T1 / T2 / T3 + action` detector with **one probabilistically grounded score**
over two factors (state and dynamics). The action sequence is treated as
observed conditioning — it is never denoised.

## 2. Factorization

For each transition `τ = (z_t, a, z_{t+H})` and class label `c ∈ {0: success,
1: failure}`:

```
p(τ | c) = p(z_t | c) · p(z_{t+H} | z_t, a, c)
```

Two factors. No action factor. The old `p(a | z_t, c)` term from uni-dsm is
dropped entirely in both training and inference.

## 3. SσDC identity

For fixed noise scale σ, the DSM training objective is an unbiased estimator
of the conditional marginal log-density up to a constant that does not depend
on `c`:

```
-log p_σ(x | c) = (1 / 2σ²) · E_ε ‖ g_θ(x + σε, c) − x ‖²  +  C(x, σ)
```

Because `C(x, σ)` is independent of `c`, the conditional
log-likelihood ratio reduces to the difference of per-class reconstruction
energies:

```
LLR(x) = log [ p(x | c=0) / p(x | c=1) ]
       = (1 / 2σ²) · ( E⁽¹⁾(x) − E⁽⁰⁾(x) )
```

where `E⁽ᶜ⁾(x) = E_ε ‖ g_θ(x + σε, c) − x ‖²`. At inference we use the
single-sample estimator with `add_noise=False`, `K=1`, `ε=0`:

```
Ê⁽ᶜ⁾(x) = ‖ g_θ(x, c) − x ‖²
```

This estimator is biased but low-variance and empirically preferable to MC
Tweedie on this data.

## 4. Score formula and probabilistic reading

Let `b ∈ {state, dynamics}` and let `z(·)` denote per-task, per-branch
z-score standardization on a success calibration bank. The per-step SσDC
score is:

```
u(x) = Σ_b [ α_b · z(Ê_b⁽⁰⁾(x)) + β_b · ( z(Ê_b⁽⁰⁾(x)) − z(Ê_b⁽¹⁾(x)) ) ]
```

Probabilistic reading:

- `α_b · z(Ê_b⁽⁰⁾)` — per-task calibrated NLL under the success class; an
  OOD prior on the factor `b`.
- `β_b · ( z(Ê_b⁽⁰⁾) − z(Ê_b⁽¹⁾) )` — per-task calibrated signed LLR; a
  Bayes classifier term.

The sum is a scaled posterior log-odds with a tunable OOD prior weight. The
per-task z-score is a gauge-fixing operation that removes the task-dependent
`C(x, σ)` term in the LLR derivation.

## 5. Training recipe

- Fixed `σ = 0.08` (`model.noise_scale` in `config/train.yaml`).
- `DSMModel.forward(..., add_noise=True)` for training;
  `add_noise=False` at inference (see `compute_conditional_energies`).
- Loss: `state_loss_weight · E_state + dynamics_loss_weight · E_dynamics`,
  with defaults `1.0` and `1.0`.
- Positive vs. failure balanced sampling via `WeightedRandomSampler`
  (`positive_sampling_ratio`).
- Optimizer LR multipliers: `shared = state_branch = dynamics_branch = 1.0`.

## 6. Architecture note — shared-backbone CFG

`UnifiedConditionedDSM` is a single transformer parameterized by
`φ(c, task, route)` through AdaLN-Zero modulation. Each factor has its own
routed target encoder and head, a global condition vector combining
`(route_embedding, type_embedding, task_name_embedding, context_feature)`,
and runs through the **same shared DiT blocks**. Factor independence holds
because each `forward_task` call consumes only its own `context_feature` and
target token; there is no cross-factor attention inside the backbone.

## 7. Default weights

```
α_state = α_dynamics = 0,   β_state = β_dynamics = 1
```

These defaults replicate the R5 recipe (β-only, state + dynamics). Per-task
α, β is future work.

## 8. Monitoring — class-signal-collapse

`DSMModel.class_conditional_diff` returns per-factor
`‖ g_θ(x, c=0) − g_θ(x, c=1) ‖²` with `add_noise=False`. `Trainer` calls it
once per epoch on a training batch and prints:

```
[collapse] epoch=<N> state_diff_l2=<val> dynamics_diff_l2=<val>
```

A rolling deque of the last 3 values per factor triggers a
`WARNING[collapse]` line when either factor stays below `1e-4` for 3
consecutive epochs. This is the class-signal-collapse signature that
wrecked the sibling `uni-dsm-new` branch — hold commits and report if it
fires.
