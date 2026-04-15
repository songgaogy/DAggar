# LPB Score: Two-Head Conditional Fisher DSM

Offline failure detection on **latent transitions** produced by a **frozen multitask flow policy encoder**. The pipeline trains a **two-head conditional denoising score model (DSM)** shared across manipulation tasks, then scores trajectories by **contrasting** two trajectory-type conditionals with a **weighted Fisher-type** gap between success-like and failure-like predictions.

This document merges the former user guide and design summary and records the **mathematical formulation** implemented in `core/model.py` and `core/dsm_discriminator.py`.

---

## 1. Problem setup

### 1.1 Latent transition

At time \(t\) with horizon \(H = \texttt{transition\_horizon}\), a training example is

$$
\tau = \bigl(z_t,\, a_{t:t+H-1},\, z_{t+H}\bigr),
$$

where \(z_t, z_{t+H} \in \mathbb{R}^{d_z}\) are policy latents and \(a_{t:t+H-1} \in \mathbb{R}^{H \times d_a}\) is an action chunk. A discrete **task index** \(k \in \{0,\ldots,K-1\}\) identifies the manipulation task (\(K = \texttt{num\_tasks}\)). A binary **trajectory type** \(c \in \{0,1\}\) (denoted `traj_type` in code) indicates whether the source trajectory is treated as **positive** (\(c=0\), expert or success rollout) or **failure-conditioned** (\(c=1\), fail rollout).

### 1.2 Standardization

Let \(\mu_z, \sigma_z\) and \(\mu_a, \sigma_a\) be **per-dimension** mean and standard deviation estimated from **positive training data only** (expert + success rollouts). Define

$$
\bar{z} = \frac{z - \mu_z}{\sigma_z}, \qquad
\bar{a} = \mathrm{flatten}(a_{t:t+H-1}) \;\text{ standardized per action dim with horizon-tiled stats.}
$$

The **dynamics factor** does not predict \(\bar{z}_{t+H}\) directly; it predicts the **normalized residual**

$$
\delta_t = \bar{z}_{t+H} - \bar{z}_t \in \mathbb{R}^{d_z}.
$$

The three **clean normalized targets** used inside the model are:

- **State factor:** \(\bar{z}_t\)
- **Actor factor:** \(\bar{a}\) (flattened dimension \(H d_a\))
- **Dynamics factor:** \(\delta_t\)

---

## 2. Two-head conditional DSM (training)

### 2.1 Denoising targets and noise

The model uses two denoisers:

- **Occupancy head:** joint state-action target \(x_{sa} = [\bar{z}_t, \bar{a}]\)
- **Transition head:** residual target \(x_{\mathrm{dyn}} = \delta_t\)

Training uses **isotropic Gaussian noise** in **normalized** space:

$$
\tilde{x}_{sa} = x_{sa} + \sigma \,\varepsilon_{sa}, \qquad
\tilde{x}_{\mathrm{dyn}} = x_{\mathrm{dyn}} + \sigma \,\varepsilon_{\mathrm{dyn}},
$$

with fixed \(\sigma = \texttt{noise\_scale}\) (see config).

### 2.2 Predictor and conditioning

Let \(g_\theta\) denote the current denoiser (`UnifiedConditionedDSM`, name kept for compatibility). The implementation is now a **two-head decoupled conditional DSM** with **AdaLN-conditioned ResNet-MLP** backbones:

1. **Condition embeddings**

   Learnable embeddings for:

   - trajectory type \(c \in \{0,1\}\),
   - task \(k \in \{0,\ldots,K-1\}\),

   are summed and passed through a small MLP to form a dense condition vector \(e_{\mathrm{cond}}\).

2. **Occupancy denoiser**

   The occupancy head denoises the joint noisy vector

   $$
   \tilde{x}_{sa} = [\tilde{s}, \tilde{a}],
   $$

   and outputs one hidden feature that is decoded into:

   - state prediction \(\hat{s}\),
   - action prediction \(\hat{a}\).

3. **Transition denoiser**

   The transition head denoises noisy residual \(\tilde{\delta}_t\), but conditions on **clean** normalized state-action context. Concretely, clean \([\bar{z}_t, \bar{a}]\) is projected to a context feature, concatenated with \(\tilde{\delta}_t\), and then passed into the transition backbone.

4. **AdaLN residual blocks**

   Each backbone applies condition-dependent scale and shift at every hidden block:

   $$
   h_{l+1} = h_l + \mathrm{MLP}\bigl(\gamma_l(e_{\mathrm{cond}}) \odot \mathrm{LN}(h_l) + \beta_l(e_{\mathrm{cond}})\bigr).
   $$

The config still accepts `model.num_heads` for backward compatibility with older YAMLs, but it is a **compatibility no-op** because the backbone is MLP-only.

$$
\hat{x}_{sa}^{(c)} = g_{\theta,sa}\bigl(\tilde{x}_{sa}, c, k\bigr), \qquad
\hat{\delta}_t^{(c)} = g_{\theta,\mathrm{dyn}}\bigl(\tilde{\delta}_t, \bar{z}_t, \bar{a}, c, k\bigr).
$$

### 2.3 DSM training loss

Define the reconstruction energies

$$
E_{\mathrm{state}} = \frac{1}{d_z}\left\lVert \hat{s} - \bar{z}_t \right\rVert_2^2,
$$
$$
E_{\pi} = \frac{1}{H d_a}\left\lVert \hat{a} - \bar{a} \right\rVert_2^2,
$$
$$
E_{\mathrm{dyn}} = \frac{1}{d_z}\left\lVert \hat{\delta}_t - \delta_t \right\rVert_2^2.
$$

Training minimizes a **weighted** sum of branch energies:

$$
\mathcal{L}_{\mathrm{DSM}}
= \mathbb{E}_{\tau,\,\varepsilon}\left[
w_{\mathrm{state}} E_{\mathrm{state}}
+ w_{\pi} E_{\pi}
+ w_{\mathrm{dyn}} E_{\mathrm{dyn}}
\right],
$$

implemented in `compute_dsm_loss`.

Default branch weights are:

- \(w_{\mathrm{state}} = 1.0\)
- \(w_{\pi} = 0.5\)
- \(w_{\mathrm{dyn}} = 1.0\)

so the action branch contributes **half** the gradient weight of the state and dynamics branches by default. The training logs also keep an `unweighted_score`, which is the raw sum \(E_{\mathrm{state}} + E_{\pi} + E_{\mathrm{dyn}}\), for comparison with older experiments.

Optional **balanced sampling** over \(c=0\) vs \(c=1\) transitions still enforces a target fraction of positives (`training.positive_ratio`).

**Theory:** This is a **conditional DSM / score-matching** style objective: learn to map corrupted targets back to clean ones under explicit \((c,k)\) conditioning, so the model encodes **transition plausibility** under each label.

### 2.4 Branch-wise learning rates

The optimizer uses separate AdamW parameter groups for:

- shared occupancy backbone and occupancy conditioning modules,
- state branch,
- action branch,
- full transition network.

Given base learning rate `training.lr`, the effective per-group learning rates are:

$$
\eta_{\mathrm{group}} = \eta_{\mathrm{base}} \times m_{\mathrm{group}}.
$$

Default multipliers are:

- shared: `1.0`
- state branch: `1.0`
- action branch: `0.5`
- dynamics branch: `1.0`

So by default the action denoising branch is trained with both:

- **half loss weight**, via `model.action_loss_weight = 0.5`
- **half learning rate**, via `training.action_branch_lr_multiplier = 0.5`

relative to the state and dynamics branches.

---

## 3. Inference: weighted Fisher contrast and energies

At inference, **no noise** is added. The same \(\tau\) and task \(k\) are evaluated under **two** type conditionals, \(c=0\) and \(c=1\), producing:

- occupancy outputs \((\hat{s}^{(0)}, \hat{a}^{(0)})\) and \((\hat{s}^{(1)}, \hat{a}^{(1)})\),
- transition outputs \(\hat{\delta}^{(0)}\) and \(\hat{\delta}^{(1)}\).

### 3.1 Fisher-type divergence per factor

Define

$$
\Delta_s = \hat{s}^{(0)} - \hat{s}^{(1)}, \qquad
\Delta_a = \hat{a}^{(0)} - \hat{a}^{(1)}, \qquad
\Delta_{\mathrm{dyn}} = \hat{\delta}^{(0)} - \hat{\delta}^{(1)}.
$$

The raw per-factor Fisher terms are

$$
D_{\mathrm{state}} = \left\lVert \Delta_s \right\rVert_2^2, \qquad
D_{\pi} = \left\lVert \Delta_a \right\rVert_2^2, \qquad
D_{\mathrm{dyn}} = \left\lVert \Delta_{\mathrm{dyn}} \right\rVert_2^2.
$$

The **default step score** is the weighted Fisher score

$$
S^{\mathrm{Fisher}} = D_{\mathrm{state}} + \lambda_a D_{\pi} + \lambda_{\mathrm{trans}} D_{\mathrm{dyn}},
$$

with default detector settings:

- `detector.lambda_a = 0.1`
- `detector.lambda_trans = 1.0`

Large \(S^{\mathrm{Fisher}}\) indicates the two conditionals disagree strongly on the same transition, with the **action occupancy term explicitly suppressed**.

### 3.2 Branch reconstruction energies and margins

Independently, each branch \(c \in \{0,1\}\) yields a **reconstruction energy** per factor (mean coordinate MSE vs. clean target):

$$
E_b^{(c)} = \frac{1}{\dim(x_b)} \left\lVert \hat{x}_b^{(c)} - x_b \right\rVert_2^2 .
$$

**Margins** (used in T2/T3) compare energies between branches, e.g.

$$
M_b = E_b^{(1)} - E_b^{(0)} .
$$

Implementation details, weighted contributions, and auxiliary reconstruction terms are returned by `compute_fisher_score` and packaged into a `TrajectoryScoreBundle`.

---

## 4. Detector: step scores, \(\lambda\), and thresholds

Let \(u_t\) denote a **scalar step score** at time \(t\) chosen from a **score family** (below). The detector maps \(\{u_t\}\) to an aggregate **\(\lambda\)** sequence:

- **Mean mode:** if \(\texttt{lambda\_window\_size} \le 0\), prefix average \(\lambda_t = \frac{1}{t+1}\sum_{i=0}^{t} u_i\). If \(W = \texttt{lambda\_window\_size} > 0\), **sliding arithmetic mean** over the last \(W\) steps.
- **Max mode:** prefix maximum if \(\texttt{lambda\_window\_size} \le 0\), else rolling maximum over a window of size \(W\).

A **threshold** \(\tau\) is calibrated from **normal** trajectories (e.g. success bank) as a high quantile of \(\lambda\) values, controlled by **`delta`** (percentile-style budget). Alarms occur when \(\lambda_t \ge \tau\) (per-task thresholds optional).

### 4.1 Score families (T1 / T2 / T3)

Let \(M_b\) denote the margin terms above. For T1, the implementation uses the **weighted Fisher contributions**:

$$
C_{\mathrm{state}} = D_{\mathrm{state}}, \qquad
C_{\pi} = \lambda_a D_{\pi}, \qquad
C_{\mathrm{dyn}} = \lambda_{\mathrm{trans}} D_{\mathrm{dyn}}.
$$

- **T1 — weighted Fisher:**  
  $$
  u_t^{\mathrm{(T1)}} = C_{\mathrm{state}} + C_{\pi} + C_{\mathrm{dyn}}.
  $$

- **T2 — negative margin:** uses **negative** linear combinations of margin terms. This path is kept for compatibility.

- **T3 — weighted standardized combo:** blends **z-scored** positive energies and margins with weights \(\alpha_b, \beta_b\) (config `alpha_*`, `beta_*`). Normalization statistics for z-scoring are estimated from a **success bank** of trajectories.

The active mode is `detector.score_mode` in `config/visualize.yaml`. The default detector config still points to T1, which now means the **weighted Fisher** score above.

---

## 5. Data and implementation mapping

### 5.1 Data families and splits

| Family | Use |
|--------|-----|
| `expert`, `success_rollout` | Pooled for **positive** trajectory sampling (`num_pos_traj` per split) |
| `fail_rollout` | **Negative** trajectories (`num_neg_traj`) |

Per-task directories are set in `config/train.yaml`. Cached latents live under `data.cache_dir` as `.npz` files.

### 5.2 Dataset tensors (`LatentTransitionDataset`)

| Field | Symbol / role |
|--------|----------------|
| `current_latent` | \(z_t\) |
| `action_sequence` | \(a_{t:t+H-1}\) |
| `target_latent` | \(z_{t+H}\) |
| `task_index` | \(k\) |
| `traj_type` | \(c\) |

### 5.3 Code map

| Component | Path |
|-----------|------|
| Two-head DSM + `DSMModel` | `core/model.py` |
| Training loop | `core/trainer.py` |
| Data + caches | `core/dataset.py` |
| Scorer + detector | `core/dsm_discriminator.py` |
| Hydra train / visualize | `train.py`, `visualize_failures.py` |

---

## 6. Training and inference workflows

**Training:** `train.py` + `config/train.yaml` — builds caches, sets `num_tasks = \|task_to_index\|`, fits \(\mu,\sigma\) from positives, builds the two-head DSM, applies branch-weighted loss and branch-wise optimizer LRs, runs `Trainer.fit`, and saves checkpoints.

**Visualization:** `visualize_failures.py` + `config/visualize.yaml` — loads `model.dsm_ckpt`, calibrates detector from `eval.bank_*`, applies `detector.lambda_a` / `detector.lambda_trans` inside the default Fisher score, and renders failures from `eval.fail_*` or **suboptimal** HDF5 (`visualization.data_source`).

**Shell helpers:** `scripts/train_lpb_score_dsm.sh`, `scripts/visualize_lpb_score_dsm_failures.sh`.

Example:

```bash
NUM_POS=240 NUM_NEG=120 GPU=0 \
  bash robosuite/discriminator/lpb_score/scripts/train_lpb_score_dsm.sh
```

Example with explicit action-branch overrides:

```bash
GPU=0 \
  bash robosuite/discriminator/lpb_score/scripts/train_lpb_score_dsm.sh \
  model.action_loss_weight=0.5 \
  training.action_branch_lr_multiplier=0.5
```

---

## 7. Checkpoints and compatibility

Saved payloads include `model`, `cfg`, `latent_dim`, `action_dim`, `horizon`, `num_tasks`, `task_to_index`, `normalization_stats`, and `model_architecture: two_head_decoupled_conditional_dsm`.

Checkpoints must contain the current two-head architecture. Older checkpoints from the former unified multitask DSM are **not** weight-compatible with the current `UnifiedConditionedDSM`, even though the class name was kept for API compatibility.

---

## References (informal)

- **Denoising / score matching:** learn \(g_\theta\) to invert small Gaussian corruptions (DSM-style objectives).
- **Fisher-type contrast:** compare two conditional predictors at the same input; squared differences act as a **divergence** between manifolds induced by \(c=0\) vs \(c=1\).
- **Failure detection:** high \(\lambda\) vs calibrated \(\tau\) flags transitions inconsistent with the success bank.
