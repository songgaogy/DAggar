# Task Description: LPB Score Discriminator

## 1) Goal

Build and evaluate an offline trajectory failure discriminator for multi-task robot manipulation rollouts.

Given a trajectory (multi-view RGB + proprio), the discriminator must output:

1. step-level anomaly scores,
2. temporally aggregated scores,
3. binary failure decisions based on calibrated thresholds,
4. visualization artifacts (videos/PDFs) and reproducible run metadata.

The current target implementation is `robosuite/discriminator/lpb_score`.

---

## 2) Problem Formulation

Single-frame detection is ambiguous in long-horizon manipulation. The discriminator therefore scores a **temporal latent chunk** instead of an isolated frame.

For each timestep `t`:

- encode observation into latent `z_t`,
- construct causal window `X_t = [z_{t-W+1}, ..., z_t]` with fixed size `W` (left-padded at prefix),
- run conditional denoising/reconstruction under:
  - positive condition `c=0`,
  - failure condition `c=1`.

Core detector terms:

- positive chunk energy `E_pos(t)`,
- chunk margin `M(t) = E_neg(t) - E_pos(t)`.

T3 step score:

`u_t = alpha * zscore(E_pos(t)) - beta * zscore(M(t))`

Aggregate score `lambda_t` is computed from `u_t` by prefix/sliding `mean` or `max`.

Failure prediction:

`y_t = 1[lambda_t >= tau]`, where `tau` is percentile-calibrated from a success bank.

---

## 3) Data Contract

## 3.1 Input Sources

Data are loaded from HDF5 rollout/demo files across tasks and split types (e.g., expert, success_rollout, fail_rollout, suboptimal).

Each trajectory contains aligned sequences:

- RGB images from multiple cameras,
- proprio vectors,
- action vectors,
- task identity metadata (`task_name`, `task_index`).

## 3.2 Runtime Tensors

Windowed model input:

- `image_window`: `(B, W, V, 3, H, W_img)`
- `proprio_window`: `(B, W, P)`
- `task_index`: `(B,)`
- `traj_type`: `(B,)`, where `0=positive`, `1=failure-conditioned`

For visualization/evaluation:

- per-step labels are optional (`None` allowed),
- frame-level values are derived by mapping step values to frame timeline.

## 3.3 Caching

Use preprocessed disk cache for fast loading:

- `images_chw` (uint8),
- `proprio` (float32),
- `actions` (float32).

Cache invalidation must depend on data path, task metadata, preprocessing parameters, and cache-version token.

---

## 4) Model and Optimization Requirements

## 4.1 Architecture

- Front-end encoder: multitask flow-policy context encoder (optional LoRA tuning).
- Predictor: single-head conditional temporal Conv1d DSM over latent window.
- Conditioning: task embedding + trajectory-type embedding.

## 4.2 Training Objective

For noisy latent window `X_t_tilde = X_t + sigma * eps`:

- predict clean latent window `X_hat_t`,
- optimize chunk reconstruction MSE.

No classification head is required.

## 4.3 Parameter Groups

Optimizer groups should preserve semantic separation:

- `encoder_branch` (encoder/LoRA trainable params),
- `shared` (conditioning + backbone),
- `chunk_branch` (prediction head).

---

## 5) Inference and Detection Requirements

For each timestep/window:

1. encode latent window,
2. run dual condition forward (`c=0`, `c=1`),
3. compute `E_pos`, `E_neg`, `M`,
4. compute standardized T3 step score `u_t`,
5. aggregate to `lambda_t`,
6. threshold with calibrated `tau`,
7. return:
   - step scores,
   - aggregate scores,
   - thresholds,
   - predictions,
   - per-term attribution metadata.

Adaptive thresholding may update `delta -> tau` online when labels are provided.

---

## 6) Calibration Protocol

Calibration requires:

1. a **normal/success bank** to estimate normalization stats for T3 terms,
2. a calibration set to produce `lambda` distribution,
3. percentile threshold:
   - `tau = percentile(lambda, 100 - delta)`.

Must support:

- global threshold,
- task-specific thresholds,
- fallback from task-specific to global stats when needed.

---

## 7) Outputs and Artifacts

Per run, generate:

1. rendered MP4 overlays (score + threshold + prediction),
2. optional per-trajectory PDF summaries,
3. threshold distribution plots by task,
4. machine-readable `summary.json` with:
   - config snapshot,
   - calibration metadata,
   - threshold values,
   - rendered video records,
   - attribution summaries.

---

## 8) Evaluation Requirements

Minimum reporting:

- number of trajectories rendered/evaluated,
- threshold(s) and score mode,
- per-task calibration counts,
- first predicted failure index (if any),
- frame-level predicted failure count.

Recommended extended metrics:

- step-level PR/AUROC,
- false alarm rate under success trajectories,
- time-to-detect on failure trajectories.

---

## 9) Engineering Constraints

- Keep model/checkpoint self-contained for offline visualization (no external policy checkpoint required at inference time).
- Ensure inference window size matches checkpoint window size.
- Avoid assumptions that all trajectories have labels.
- Persist enough metadata for reproducibility.
- Maintain compatibility with Hydra config overrides and provided shell scripts.

---

## 10) Acceptance Criteria

The task is considered complete when:

1. training runs end-to-end with the provided scripts/config system,
2. detector calibration succeeds on a non-empty success bank,
3. visualization produces valid MP4/PDF/summary artifacts,
4. predictions and thresholds are numerically valid (no NaN/Inf in main outputs),
5. outputs are reproducible given fixed seed and fixed data/checkpoint.
