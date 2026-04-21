# Conversation Summary — lpb_score SσDC Refactor

## Context

- **Branch**: `uni-dsm` (verified via `git branch --show-current`)
- **Goal**: collapse the pre-refactor `T1/T2/T3 + action branch` detector into a single **Single-σ Diffusion Classifier (SσDC)** score over two factors (state + dynamics); target trajectory AUROC ≥ 0.96 on `[Can, Bread, Cereal, Milk]`.

---

## Phase 1 — Exploration & Ambiguities

Read all target files from remote. Found three mismatches with the refactor prompt:

| # | Issue | Resolution (user Q&A) |
|---|---|---|
| 1 | §2.4 assumes `aux_contrastive_weight / aux_contrastive_margin` exist — grep confirmed they **do not** on `uni-dsm` | **Skip §2.4 entirely** |
| 2 | §1.3 new formula `α·z(E⁺) + β·(z(E⁺)−z(E⁻))` is NOT a monotonic transform of old T3 `α·z(E⁺) − β·z(margin)`; R5 replication not guaranteed | **Strictly implement §1.3**; accept ±0.01–0.03 AUROC drift |
| 3 | `analysis/attribution.py` (12 action refs) not listed in §4 constraints | **Bring it in scope**; `app/visualize.py` stays minimal-touch |

---

## Approved Plan

Saved at `/Users/gaoyuan/.claude/plans/follow-agents-md-task-discriminator-refa-reflective-kernighan.md`. Per-section diff plan for model, trainer, discriminator, config, scripts, benchmark entry, analysis, visualize, docs. Commit granularity = one commit per section.

---

## Execution — 8 commits on `uni-dsm`

| # | Hash | Scope |
|---|---|---|
| 1 | `dab62e1` | `core/model.py` — drop action factor; rename `compute_fisher_score→compute_conditional_energies`, `fisher_components→pairwise_energy_gap`; add `DSMModel.class_conditional_diff` for collapse monitoring |
| 2 | `3616019` | `core/trainer.py` + `app/train.py` — drop `action_branch_lr_multiplier`, action logs; add `_log_collapse_diagnostic` (1e-4 × 3-epoch warning) |
| 3 | `bf60e5e` | `core/dsm_discriminator.py` — single SσDC formula; rename `next_state_*→dynamics_*`; drop `SCORE_T1/T2/T3`, `alpha_action/beta_action`, `ewma_alpha` |
| 4 | `6739f53` | `config/train.yaml` + `app/pipeline.py` + `scripts/run_lpb_score_benchmark.sh` — strip action loss/LR and T1/T2/T3 flags |
| 5 | `1ee221d` | `analysis/attribution.py` — `TERM_ORDER = (state, dynamics)` |
| 6 | `dd0fd61` | `summary.md` + `README.md` — full rewrite around SσDC derivation |
| 7 | `ed2b613` | `scripts/visualize_lpb_score_dsm_failures.sh` — drop `SCORE_MODE/ALPHA_ACTION/BETA_ACTION`; defaults match R5 |
| 8 | `e5249b9` | `app/visualize.py` + `config/visualize.yaml` — port to SσDC single-score semantics (per-branch contribution plot replaces T1/T2/T3 mode_ax) |

### Smoke tests passed
- All lpb_score modules import cleanly.
- `python -m …train --cfg job --resolve` prints clean config (no `action_loss_weight`, no `action_branch_lr_multiplier`).
- `visualize_failures --cfg job --resolve` prints clean `detector:` block (only `alpha_state/alpha_dynamics/beta_state/beta_dynamics`).

### Known limitations
- `data/utils/benchmark/examples/run_lpb_score.py` is gitignored (`.gitignore:134` excludes `data/`). File rewritten on disk, **not committed**.
- Old checkpoint `checkpoints/multitask_6/lpb_dipole-new/...` is intentionally incompatible (action head gone) — retrain required before benchmarking.

### Not yet executed
- **§3.2 retrain** (50 epoch) — awaiting user go-ahead.
- **§3.3 benchmark** — depends on retrain.

---

## Architecture Discussion

User asked how state and dynamics denoising are put together. Answered with a detailed walkthrough:

- **Shared backbone** (`shared_blocks` × `num_layers` AdaLN-Zero DiT), **routed per-factor heads** (`state_target_encoder + state_head`, `dynamics_target_encoder + dynamics_head`).
- Two forward passes through the same backbone per sample; `route_embedding(STATE=0 / DYNAMICS=1)` disambiguates.
- Dynamics pass conditions on `(z_t, a)` via `dynamics_context_encoder([state_context_encoder(z_t), action_context_encoder(a)])`; state pass uses `context = zeros`.
- All conditions (route, type=CFG, task, context) collapsed to one vector via `condition_mlp([context, route + type + task_name])` → drives AdaLN shift/scale/gate for all layers.
- Inference (`compute_conditional_energies`) = 4 backbone passes per transition (2 classes × 2 factors).

---

## Open Critique → Architecture Cleanup Proposals

User pushed back: **"压成一个向量还是很脏"**. Agreed. Precise dirty points:

1. CFG `type_emb(c)` summed with `route_emb + task_emb` — class signal dilutable → collapse risk.
2. Continuous `(z_t, a)` double-bottlenecked through two MLPs before AdaLN.
3. Shared backbone factor disambiguation relies entirely on `route_emb` contrast.

### Three cleanup levels proposed

| Level | Change | Cost | Breakage |
|---|---|---|---|
| **L1** | Separate CFG channel: each DiT block gets two AdaLN streams (`c_class` from type only, `c_context` from route+task+context). | ~1.5× AdaLN params | retrain only |
| **L2** | Cross-attention to `cond_seq = [route_tok, type_tok, task_tok, z_t_tok, a_tok]`; drop `condition_mlp` single-vector bottleneck. | ~1.8× per-layer params, ~1.6× FLOPs | retrain only |
| **L3** | Full MMDiT-style joint attention. | ~2.5× params, ~2× FLOPs | larger code rewrite |

### Current state: awaiting user decision on

1. Run §3.2 retrain on current (dirty-but-working) architecture first to verify AUROC ≥ 0.96, **then** decide L1/L2? Or fold architecture change into this refactor?
2. Accept another retrain as the cost of L1/L2?
3. L1 (conservative) vs. L2 (modern DiT) preference?
