# D4-Disc

Bellman-Bootstrapped Dichotomous Discriminator. A single AdaLN-conditional
latent dynamics network trained with classifier-free-guidance (CFG) label
dropout + a Bellman-consistent bootstrap loop (advantage-gated soft labels)
that replaces D3-Disc's static F3 filter. Design spec:
`.claude/discriminator/d4_disc_design.md`. Implementation spec:
`.claude/discriminator/d4_implementation_plan.md`.

## Package layout

```
robosuite/discriminator/d4disc/
├── adaln.py               AdaLN-Zero + ConditionEmbedder
├── model.py               ConditionalDynamicsPredictor
├── ema.py                 ModelEMA
├── schedule.py            D4Schedule (alpha/kappa/eta)
├── dataset.py             LatentFlowDynamicsDatasetD4 (+ is_fail_raw, gamma buffer)
├── filter.py              compute_advantage_gate
├── monitor.py             CollapseDetector + D4Health
├── trainer.py             D4Trainer (phase A / phase B state machine)
├── train_d4.py            CLI entry
├── detector.py            D4Detector (CFG residual scoring)
├── dynamics_feature.py    D4FeatureExtractor (frame assembly)
├── d4_benchmark.py        D4BenchmarkDiscriminator
├── visualize.py           PDF + video renderer with (r+, r-, A) panels
├── tests/
│   ├── test_adaln.py
│   └── test_advantage_gate.py
└── scripts/
    ├── train_d4.sh
    ├── run_d4_benchmark.sh
    ├── sweep_omega_d4.sh
    ├── ablate_schedule.sh
    └── visualize_d4.sh
```

## Training

Train one checkpoint; one checkpoint serves all inference-time `omega`.

```bash
bash robosuite/discriminator/d4disc/scripts/train_d4.sh
# overrides (examples):
WARM_UP_EPOCHS=4 BOOTSTRAP_EPOCHS=20 TASKS="PickPlaceBread" \
  bash robosuite/discriminator/d4disc/scripts/train_d4.sh
```

Outputs to `checkpoints/d4disc/dynamics/<RUN_NAME>/d4_dynamics.pt`. WandB runs
offline under `project=d4disc`, entity from `WANDB_NAME` env.

## Benchmark

```bash
D4_CKPT=checkpoints/d4disc/dynamics/<RUN>/d4_dynamics.pt \
  bash robosuite/discriminator/d4disc/scripts/run_d4_benchmark.sh
```

Omega sweep (no retraining):

```bash
D4_CKPT=checkpoints/d4disc/dynamics/<RUN>/d4_dynamics.pt \
  bash robosuite/discriminator/d4disc/scripts/sweep_omega_d4.sh
```

## Visualization

```bash
D4_CKPT=checkpoints/d4disc/dynamics/<RUN>/d4_dynamics.pt TASK=PickPlaceBread \
  bash robosuite/discriminator/d4disc/scripts/visualize_d4.sh
```

## Unit tests

```bash
/home/dodo/miniconda3/envs/daggar/bin/python -m pytest \
  robosuite/discriminator/d4disc/tests -v
```

## Stability invariants (plan §7)

- AdaLN heads are zero-init; at step 0 the predictor is c-invariant.
- Phase A (K_warm=8 by default): fail_raw samples never routed to `c=-`.
- Phase B: advantage gate runs on EMA weights; gamma is EMA-smoothed;
  `alpha_k` is capped by `alpha_cap_rate / held_out_r_plus` to enforce
  the design (C4) coupling.
- Collapse monitor reverts to EMA and halves alpha0 on trip.
- Benchmark discriminator loads `ema_state_dict` (not raw weights).
- Inference clamps `omega` to 2.0 (plan §9.3 CFG divergence).

## Symmetric fixed-point failure mode (run d4dyn_20260422_050952)

The first full run landed in a `gamma=0.5` **symmetric fixed point** and
benchmarked at random (AUROC≈0.50). Symptoms:

- `mean_gamma ≈ 0.495` throughout bootstrap (stuck near 0.5)
- `conditional_delta ≈ 0.003` (`f(+) ≈ f(-)` despite 40 epochs of training)
- `separation_gap ≈ 0.05` (tiny; not growing)
- `alpha_used ≈ 1.0` (cap-bound by `alpha_cap_rate=1.0 / held_out_r_plus`)

Mechanism: at the start of Phase B, `f(+)≈f(-)` (AdaLN-Zero kept c=- at
identity through Phase A), so the advantage `A = r⁻ − r⁺ ≈ 0` → gate returns
`γ ≈ 0.5` → fail samples routed 50/50 to {+, −} → both branches see the
same data → they converge to the same function → `A` stays 0 → closed loop.

### Fixes applied (defaults updated accordingly)

1. **F3-based γ warm-start** (`filter.py::warm_start_gamma_from_knn`): at
   end of Phase A, seed γ with D3-style KNN distance to the clean success
   bank. Off-manifold fail frames start at γ≈1, on-manifold prefix frames
   at γ≈0. One-shot replace, no EMA smoothing at the warm-start step.
   Toggle via `--f3-warm-start/--no-f3-warm-start` (default on).
2. **`gate_freeze_epochs=2`**: skip the advantage-gate update for the first
   two bootstrap epochs after warm-start so `f(-)` has time to differentiate
   before the gate takes over. Otherwise the first gate call (with
   `f(+)≈f(-)` still) would immediately reset γ to 0.5.
3. **`alpha_cap_rate: 1.0 → 5.0`** (python + shell default): gives α real
   dynamic range so γ can sharpen once `separation_gap` grows.
4. **`adaln_init_std: 0.0 → 0.02`** (shell default, class default still
   0.0 for backward compat): small non-zero init on AdaLN modulation heads
   so `f(+) ≠ f(-)` structurally from step 0. This addresses the root
   cause — zero-init gates grow only through gradient and fail data alone
   is a weak training signal for c=-. At std=0.02 the initial
   `conditional_delta` is already comparable to what 40 epochs of
   zero-init training reached in the failed run.
5. **γ quantile logging** (q05/q25/q50/q75/q95 per bootstrap epoch) — lets
   you spot symmetric collapse in the first few epochs instead of 40.

### What to watch in subsequent runs

- `warmstart/mean_gamma` should land in [0.4, 0.8] (fail_raw populations
  usually have both on-manifold prefix and off-manifold tail). `q05`
  should be <0.3 and `q95` >0.7 — if they're both near 0.5 the clean
  bank is too uniform.
- First two `gate_frozen=1` epochs should drop `latent_mse` meaningfully
  while leaving γ quantiles unchanged.
- From epoch 3 onward, look for: `conditional_delta` growing >10×,
  `separation_gap` growing (not shrinking), `mean_gamma` moving away from
  0.5 toward whatever the F3 warm-start set. If instead γ drifts back to
  0.5, try `ADALN_INIT_STD=0.1` and/or raise `GATE_FREEZE_EPOCHS=4`.

### Knobs (env var | CLI flag | description)

| env | CLI | default | notes |
|---|---|---|---|
| `F3_WARM_START` | `--f3-warm-start/--no-f3-warm-start` | 1 | F3 γ seed at phase A→B |
| `F3_WARM_START_K` | `--f3-warm-start-k` | 1 | KNN order for F3 |
| `F3_WARM_START_MAX_CLEAN` | `--f3-warm-start-max-clean` | 100000 | subsample clean bank |
| `GATE_FREEZE_EPOCHS` | `--gate-freeze-epochs` | 2 | hold γ after warm-start |
| `ALPHA_CAP_RATE` | `--alpha-cap-rate` | 5.0 | `α_k * held_out_r+ ≤ cap` |
| `ADALN_INIT_STD` | `--adaln-init-std` | 0.02 | AdaLN mod init (0 = AdaLN-Zero) |
