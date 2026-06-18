# Conversation Context — D³-Disc Design

> **Purpose**: a self-contained briefing for a fresh agent who will implement D³-Disc without having seen the prior conversation. Read this end-to-end before starting.

---

## 1. Who the user is, what they want

- Researcher (Songgao) working on imitation-learning failure detection in robosuite environments.
- Native Chinese speaker; uses WandB offline (`WANDB_MODE=offline`, `WANDB_NAME=songgao-personal`); conda env `daggar`; Python at `/home/dodo/miniconda3/envs/dagger/bin/python`.
- Follows `AGENTS.md` rules strictly: minimal-modifications, English code comments, no unprompted refactors, no silent behavior changes.
- Task framing: **improve the LPB discriminator from an RL theoretical perspective** to make it more principled while retaining its SOTA empirical advantage.

---

## 2. Repository context

- Repo root: `/home/dodo/Documents/DAggar/robosuite`.
- Current branch at conversation time: `discriminator-baseline`.
- The user stated (D10) that they will create a new branch themselves; the implementer should just modify code in-place.
- Relevant directories:
  - `robosuite/discriminator/lpb/` — current LPB (non-parametric KNN on latent dynamics features). Reference for API conventions.
  - `robosuite/discriminator/lpb_score/` — a prior codebase (most files deleted, only `__pycache__` remaining in working tree). The cache files under `data/.lpb_score_cache/` were produced by this pipeline. The reference encoder logic survives in git history at commit `5056929` under `robosuite/discriminator/lpb_score/core/policy_encoder.py`.
  - `robosuite/discriminator/float/`, `robosuite/discriminator/logpZO/` — other baselines (FLOAT, logpZO).
  - `robosuite/policy/flow_multi/` — the pretrained multi-task flow-matching policy. **Its encoder path (ImageEncoder → Fusion → AGG) is what we reuse for D³-Disc features.**
  - `data/<task>/expert/`, `data/<task>/success_rollout/`, `data/<task>/fail_rollout/` — HDF5 datasets, one file may contain multiple demos.
  - `data/.lpb_score_cache/<task>/<sha1>.npz` — pre-computed flow-encoder latents (256-D) + actions (7-D), 1.1 GB total, ~650 demos per task.
  - `checkpoints/multitask_6/policy/flow-20/flow_multi_ep0100_*.pt` — the flow policy checkpoint used to populate the cache.

---

## 3. Method evolution across this conversation

### Iteration 0 — Current LPB (pre-existing)

- Trains a ResNet-18 encoder + Transformer dynamics predictor `f_phi` on expert + rollout HDF5 data.
- Features `φ(s,a)` = concat of visual latent + proprio + mean action chunk (projected).
- Detector: per-task KNN bank of success `φ`'s; score = min squared distance; percentile threshold on held-out success calibration.
- **Does not use fail_rollout data.**
- Empirically SOTA on the benchmark; the user wants a theoretical upgrade, not a redesign.

### Iteration 1 — SFV-Disc (proposed then discarded)

- Proposed a "Successor-Feature Value Discriminator": `Q_E = Σ γ^k log p̂_+(rollout step k)` along K-step latent rollouts via `f_phi`.
- RL pitch: treats `−d²_knn` as a reward, computes a discounted return.
- **Rejected by the user** as "too similar to KNN methods, just wrapped in complicated theory"; also fails to use fail data. Deprecated.

### Iteration 2 — D³-Disc (current, DIPOLE-inspired, *to be implemented*)

- The user pointed to the DIPOLE paper (ICLR 2026, `arxiv:2601.00898`, local PDF at `.claude/discriminator/DIPOLE.pdf`).
- DIPOLE decomposes a KL-regularized RL policy into two bounded sigmoid-weighted components with `(1+ω)` / `ω` exponents, naturally degenerating to a positive-only policy when ω=0.
- Applied to discrimination: build $\hat p_+$ (success KDE) **and** $\hat p_-$ (fail KDE), then score
  $$\lambda_t = -(1+\omega)\log \hat p_+(\varphi_t) + \omega \log \hat p_-(\varphi_t).$$
- `G(s,a) := \log p_+ − \log p_-` is the **IRL-optimal reward** (AIRL): this is the principled RL content, not a KNN sugar coating.
- Degeneration: `ω=0` exactly recovers LPB (scale-invariant under percentile threshold).
- The user's critical constraint: **fail_rollout data is dirty** (step-level labels are missing; prefix frames of fail trajectories are expert-like). Cannot be used raw.

### The "dirty fail" fix: F3 self-consistent soft-weighting

- Use only $\hat p_+$ to assign each fail frame a soft weight $w^-_j = \sigma(\beta[d^2_{knn}(\varphi_j, D_+) - \kappa])$.
- Fail frames that happen to lie on the success manifold get $w^- \to 0$ and are softly rejected.
- Fail frames that are genuinely off-manifold get $w^- \to 1$ and are fully used in $\hat p_-$.
- Weighted KDE for $\hat p_-$ folds soft weights as additive log penalties in effective distances.

### F3 consistency theorem (added in the final spec revision)

- Under standard KDE consistency + feature separability + $\beta_N \to \infty$ / $\kappa_N \to 0^+$ schedules, the static F3 weight converges to the ground-truth off-support indicator $y^\star(\varphi) := \mathbb{1}[\varphi \notin \text{supp}(\rho_+)]$.
- Interpretation: in the infinite-data limit, F3 exactly identifies the correct support of fail relevant to the AIRL-reward denominator. No EM iteration needed.
- Proof sketch: on-support ⇒ $d²_{knn} \to 0 \le \kappa$ ⇒ $w^- \to 0$. Off-support ⇒ $d²_{knn} \ge \delta_0^2 > \kappa_N \to 0$ ⇒ $w^- \to 1$.

---

## 4. Final locked decisions (D1–D12)

These were extracted turn-by-turn from the user. **All locked.**

| ID | Choice |
|---|---|
| D1 | Sweep $\omega \in \{0, 0.1, 0.2, 0.5, 1\}$ |
| D2 | $K = 0$ in phase 1 (step-wise only, no dynamics rollout) |
| D3 | F3 filter only (drop F1 GT-mask and F2 expert-only-dynamics) |
| D4 | Shared feature encoder across $\hat p_+$ and $\hat p_-$; use **flow_multi policy encoder** |
| D5 | 1-NN min sq-distance default; $k$ exposed as bash CLI parameter |
| D6 | Conformal calibration uses the full D³-Disc score |
| D7 | Fail data from `data/<task>/fail_rollout/*.hdf5` (no step-level labels); success data from `data/<task>/success_rollout/*.hdf5`. **PandaLift has no fail_rollout.** |
| D8 | Shared $\hat p_+$ **and** shared $\hat p_-$ across all tasks (single multi-task detector); per-task calibration $\tau$ |
| D9 | $\omega$ default = 0.5 |
| D10 | User will create the git branch; implementer just edits code |
| D11 | Static F3 only (no EM iteration) |
| D12 | Add F3 consistency theorem to the design doc (done) |

---

## 5. Key technical details (verified)

### 5.1 Cache format (reuse as-is)

- Directory: `data/.lpb_score_cache/<task_name>/<sha1>.npz`
- Keys: `latents (T, 256) float32`, `actions (T, 7) float32`, `task_name`, `file_path`, `demo_key`
- Hash key (for cache reuse): `sha1("task|resolved_file_path|demo_key|image_size|camera_names|file_mtime|checkpoint_path")`
- Populated by the (now-deleted) `lpb_score.core.policy_encoder.FlowMultitaskEncoder` using `flow_multi` policy's `encode_context_tensors`.

### 5.2 Flow-multi encoder

- Location: `robosuite/policy/flow_multi/model.py::MultiModalFlowPolicy.encode_context`.
- Output: `task_scene_cond ∈ ℝ^256` per frame (the `condition_aggregator` output).
- Checkpoint payload contains: `camera_names`, `model_cfg`, `act_mean`, `act_std`, `prop_mean`, `prop_std`, `task_prompt_map`, `task_metadata_map`, optional `lora_cfg`, `ema_model` / `model` state dicts.
- Preprocessing: images resized to `image_size=128`, normalized with ImageNet mean/std; proprio normalized with `prop_mean/prop_std` from checkpoint.
- Language: resolved per task via `resolve_language_instruction(task_name)`.

### 5.3 Dataset coverage (at conversation time)

| Task | cache entries | success hdf5 | fail hdf5 | expert hdf5 |
|---|---:|---:|---:|---:|
| PandaLift | 465 | 2 | **0** | 10 |
| PandaPickPlaceCan | 659 | 2 | 2 | 9 |
| PandaStack | 655 | 2 | 2 | 9 |
| PickPlaceBread | 646 | 2 | 2 | 9 |
| PickPlaceCereal | 650 | 2 | 2 | 8 |
| PickPlaceMilk | 648 | 2 | 2 | 8 |

Each HDF5 file contains multiple demos, giving the 400–700 cache entries. `PandaLift` cannot contribute to $\hat p_-$ due to lack of fail data.

---

## 6. Files to read, in order, before implementing

1. `.claude/discriminator/conversation_context.md` (this file) — overall orientation.
2. `.claude/discriminator/sfv_discriminator_design.md` — the full D³-Disc specification (math + RL grounding + F3 consistency theorem + decision log).
3. `.claude/discriminator/implementation_plan.md` — the concrete step-by-step implementation plan.
4. `AGENTS.md` — user's coding conventions.
5. `robosuite/discriminator/lpb/knn_discriminator.py` — API conventions to mirror (bank/calibration/threshold pattern).
6. `robosuite/discriminator/lpb/lpb_benchmark.py` — benchmark-facing wrapper pattern.
7. `robosuite/policy/flow_multi/model.py` — encoder to reuse.
8. Git: `git show 5056929:robosuite/discriminator/lpb_score/core/policy_encoder.py` — reference implementation of cache-aware encoder; copy relevant parts into `d3disc/encoder.py`.

---

## 7. What the user explicitly asked NOT to do

- Do not wrap KNN in "theory sugar" — the RL grounding must be substantive (log-odds / IRL-optimal reward, not just "KNN is kind of a log-likelihood").
- Do not implement a plan before user explicit request. (Current status: user has approved D1–D12 and requested this implementation plan; a subsequent fresh agent is authorized to execute.)
- Do not retrain flow_multi or the LPB dynamics model.
- Do not touch `robosuite/discriminator/lpb/` — sibling package `d3disc/` instead.
- Do not regenerate the cache — reuse in place.

---

## 8. Success criteria

- D³-Disc integrated as `D3BenchmarkDiscriminator` with the same benchmark API as `LPBBenchmarkDiscriminator`.
- With $\omega = 0$: metrics numerically match current LPB baseline.
- With $\omega \in \{0.1, 0.2, 0.5, 1\}$: improved or at worst equivalent metrics on tasks that have fail_rollout data, no catastrophic regression on PandaLift.
- Ablation results written to CSV and visualized (PDF + mp4 per task).
- All runs logged to WandB offline.

---

## 9. Hand-off note

Once the implementation is complete:
- Report final per-task metrics table.
- Write a short Chinese summary (per AGENTS.md) of what was built, any deviations from spec, and a link to this doc + the design doc.
- If empirical results suggest extending to K-step rollout (phase 2), do **not** proceed without user approval.
