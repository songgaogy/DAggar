# LPB Score — Single-σ Diffusion Classifier (SσDC)

Offline failure detection on **latent transitions** produced by a **frozen
multitask flow policy encoder**. The pipeline trains a **task-conditioned
latent DSM** shared across manipulation tasks, then scores trajectory
prefixes with the **SσDC step score** over two factors: state and dynamics.
The action sequence is observed conditioning — it is never denoised.

Full derivation, formula, defaults, and monitoring: see `summary.md`.

## Layout

```
lpb_score/
├── config/
│   └── train.yaml
├── core/
│   ├── dataset.py               — cached latent trajectory + transition datasets
│   ├── model.py                 — UnifiedConditionedDSM + DSMModel (state + dynamics)
│   ├── trainer.py               — AdamW + optional EMA + collapse monitor
│   └── dsm_discriminator.py     — SσDC detector (single score formula)
├── analysis/
│   └── attribution.py           — (state, dynamics) term ordering for plots
├── app/
│   ├── pipeline.py              — Hydra helpers: encoder + detector + datasets
│   ├── train.py                 — Hydra train entrypoint
│   └── visualize.py             — visualization entrypoint (minimal-touch; may need a follow-up)
├── scripts/
│   ├── train_lpb_score_dsm.sh   — train wrapper
│   ├── run_lpb_score_benchmark.sh — benchmark wrapper (SσDC weights only)
│   └── visualize_lpb_score_dsm_failures.sh
├── train.py                     — Hydra shim
├── visualize_failures.py        — failure-visualization shim
└── summary.md                   — full SσDC derivation and defaults
```

## Data

Reuses the multitask benchmark layout. Per task:

```
./data/<TASK>/
├── expert/                  — scripted success demos
├── success_rollout/         — replay-verified success rollouts
└── fail_rollout/            — failure rollouts (traj_type = 1)
```

`config/train.yaml` splits per task into `train / val / test` by trajectory
count. Failures receive `traj_type = 1`; everything else is `traj_type = 0`.

## Model

`UnifiedConditionedDSM` is a single shared-backbone DiT:

- Two routed target encoders + heads (`state`, `dynamics`) on the same
  latent dim.
- One shared `action_context_encoder` feeding the dynamics branch only.
- Global condition vector = `route_embedding(route) + type_embedding(c) +
  task_name_embedding(task)` combined with `context_feature` through
  `condition_mlp`.
- `AdaLN-Zero` blocks stacked `num_layers` times.

`DSMModel` adds standardization (per-dim latent/action mean and variance),
fixed-σ Gaussian noise on state and residual-dynamics targets at training,
and three inference entry points:

| Method | Purpose |
| --- | --- |
| `compute_dsm_loss(...)` | Training objective: `w_state · E_state + w_dyn · E_dyn` under noised targets. |
| `compute_conditional_energies(...)` | Per-factor positive / negative energies and signed margin at `add_noise=False`. |
| `class_conditional_diff(...)` | Diagnostic `‖ g(x, c=0) − g(x, c=1) ‖²` per factor for collapse monitoring. |

Action is never an inference-time prediction target.

## Detector

`DSMDiscriminator` computes the SσDC step score

```
u(x) = Σ_b [ α_b · z(E_b⁺) + β_b · (z(E_b⁺) − z(E_b⁻)) ],   b ∈ {state, dynamics}
```

where `z(·)` is per-task z-score standardization estimated from the success
calibration bank. Aggregation into the λ score is either rolling mean or
rolling max (`lambda_mode`, `lambda_window_size`), and calibration sets a
per-task percentile threshold via `delta`.

Default weights (replicate R5):

```
α_state = α_dynamics = 0,   β_state = β_dynamics = 1
```

## Training

Conda env `daggar`:

```bash
bash robosuite/discriminator/lpb_score/scripts/train_lpb_score_dsm.sh
```

or directly:

```bash
/home/dodo/miniconda3/envs/daggar/bin/python -m robosuite.discriminator.lpb_score.train \
  data.transition_horizon=1 \
  training.epochs=50 \
  training.batch_size=256
```

Watch stdout per epoch for:

```
[collapse] epoch=<N> state_diff_l2=<val> dynamics_diff_l2=<val>
```

Both values should stay `> 1e-3`; a `WARNING[collapse]` line after 3
consecutive epochs below `1e-4` means the class-signal has collapsed.
Hold commits and investigate before continuing.

## Benchmark

```bash
bash robosuite/discriminator/lpb_score/scripts/run_lpb_score_benchmark.sh
```

Environment variables override the SσDC weights:

| var | default | meaning |
| --- | --- | --- |
| `ALPHA_STATE`, `ALPHA_DYNAMICS` | `0.0, 0.0` | OOD prior weights on each factor. |
| `BETA_STATE`, `BETA_DYNAMICS`   | `1.0, 1.0` | LLR weights on each factor (β-only = R5 recipe). |
| `LAMBDA_MODE`                   | `mean`      | Rolling aggregator over step scores: `mean` \| `max`. |
| `LAMBDA_WINDOW_SIZE`            | `-1`        | `-1` uses full-prefix aggregation. |
| `DELTA`                         | `10.0`      | Percentile for detector threshold calibration. |

The old action-head checkpoint (`checkpoints/multitask_6/lpb_dipole-new/...`)
is **incompatible** after this refactor. Retrain under the SσDC recipe and
point `DSM_CKPT` to the new checkpoint before benchmarking.

## Verification targets

- `trajectory_level.auroc ≥ 0.96` on `[PickPlaceCan, PickPlaceBread,
  PickPlaceCereal, PickPlaceMilk]` with default weights.
- Collapse metric per factor stays `> 1e-3` throughout training.
