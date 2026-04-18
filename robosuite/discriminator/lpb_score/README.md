# LPB Score: LoRA Encoder + Temporal-Conv Chunk DSM

`lpb_score` is an offline failure detector built from three parts:

1. a multitask flow-policy encoder
2. a single-head conditional denoising score model over temporal latent chunks
3. a T3 detector defined on chunk reconstruction energy and chunk margin

The active implementation is in:

- `core/model.py`
- `core/dsm_discriminator.py`
- `core/dataset.py`
- `app/train.py`
- `app/visualize.py`
- `../../dyn_bce/modules/flow_encoder.py`

This document describes the current implementation only. It does not describe the removed two-head state/action/transition DSM.

## 1. Problem Setting

The detector is designed for the case where a single frame is ambiguous. A state from an early successful phase can overlap geometrically with a later failure state. A pointwise detector cannot resolve this well because it only sees one latent `z_t`.

The current solution evaluates a short trajectory chunk instead of an isolated point. The model reasons over a fixed-length history window, reconstructs that window under positive and failure conditions, and compares the two reconstruction energies.

## 2. End-to-End Pipeline

For each timestep `t`:

1. read raw multi-view RGB and proprio
2. encode each frame into a latent `z_t`
3. build a left-padded temporal window `X_t`
4. normalize the latent window
5. corrupt the normalized window with Gaussian noise during training
6. denoise the whole window with a conditional temporal Conv1d network
7. flatten the reconstructed window only for loss and scoring

At inference, the same latent window is evaluated twice:

- once with the positive condition `c = 0`
- once with the failure condition `c = 1`

The detector then uses:

- positive chunk energy
- chunk margin between failure-conditioned and positive-conditioned energies

## 3. Temporal Chunk Formulation

Let:

- `W` be the temporal window size
- `D` be the encoder latent dimension
- `K` be the number of tasks
- `k in {0, ..., K-1}` be the task index
- `c in {0, 1}` be the trajectory-type condition

At timestep `t`, define the latent history window:

$$
X_t = [z_{t-W+1}, z_{t-W+2}, \dots, z_t] \in \mathbb{R}^{W \times D}.
$$

If `t < W - 1`, the missing left context is padded by repeating the first latent `z_0`. This keeps every sample at shape `(W, D)`.

The current predictor does not flatten the chunk before processing. It treats the chunk as a temporal sequence and applies Conv1d blocks across the window dimension. Flattening is used only after reconstruction so that the loss and detector remain defined on the full chunk vector:

$$
z_{\text{chunk}} = \mathrm{flatten}(X_t) \in \mathbb{R}^{W D}.
$$

## 4. Encoder Front End

### 4.1 Raw inputs

The training dataset returns:

- `image_window` with shape `(B, W, V, 3, H, W_img)`
- `proprio_window` with shape `(B, W, P)`
- `traj_type`
- `task_index`

Here:

- `B` is batch size
- `V` is number of cameras
- `P` is proprio dimension

The raw source is HDF5 demonstration data. The active path does not use latent caches.

### 4.2 Base encoder

The latent encoder is the multitask flow-policy context encoder from `robosuite/policy/flow_multi/model.py`. The `lpb_score` wrapper uses the same modules that produce the policy condition token:

- `image_encoder`
- `proprio_tokenizer`
- `language_encoder`
- `language_guided_modulation`
- `fusion`
- `condition_aggregator`

The `flow_head` is restored for checkpoint compatibility but is never optimized or used by `lpb_score`.

For a single timestep:

$$
z_t = f_{\phi}(o_t, p_t, \ell_k),
$$

where:

- `o_t` is the multi-view image observation
- `p_t` is proprio
- `\ell_k` is the task language prompt
- `f_\phi` is the policy encoder

### 4.3 LoRA fine-tuning

The encoder is no longer fully fine-tuned. When `policy.trainable_encoder=true`, the base encoder weights stay frozen and only LoRA adapters are optimized.

For each target linear layer with pretrained weight `W_0`, the effective weight becomes:

$$
W_{\text{eff}} = W_0 + \frac{\alpha}{r} B A,
$$

where:

- `A \in \mathbb{R}^{r \times d_{\text{in}}}`
- `B \in \mathbb{R}^{d_{\text{out}} \times r}`
- `r` is the LoRA rank
- `\alpha` is the LoRA scaling factor

The base layer output is:

$$
y = W_0 x + \frac{\alpha}{r} B A x.
$$

In code, each adapted `nn.Linear` is replaced by `LoRALinear` in `../../dyn_bce/modules/flow_encoder.py`.

LoRA is injected only into the encoder path used by `encode_context`. The optimizer sees only:

- `lora_a`
- `lora_b`

All original encoder weights remain frozen.

### 4.4 Window encoding

Given a raw window batch:

$$
O \in \mathbb{R}^{B \times W \times V \times 3 \times H \times W_{\text{img}}},
\qquad
P \in \mathbb{R}^{B \times W \times P},
$$

the code reshapes them to `(B * W, ...)`, encodes each timestep independently, and reshapes back:

$$
Z_t = [z_{t-W+1}, \dots, z_t] \in \mathbb{R}^{B \times W \times D}.
$$

This is implemented by `DSMModel.encode_latent_window(...)`.

## 5. Latent Normalization

The DSM operates on normalized latents. Let `\mu` and `\sigma^2` be the mean and variance computed from positive trajectories only. For each latent vector:

$$
\bar{z} = \frac{z - \mu}{\max(\sqrt{\sigma^2}, \sigma_{\min})},
$$

where `\sigma_{\min}` is `std_clamp_min`.

This produces the normalized latent window:

$$
\bar{X}_t = [\bar{z}_{t-W+1}, \dots, \bar{z}_t].
$$

Because the encoder can move during training, these statistics are recomputed from current positive-train encoder outputs at the start of each epoch. If EMA is enabled and used for validation or checkpoint export, EMA statistics are recomputed for the EMA model as well.

## 6. Chunk DSM Architecture

The active denoiser is `ChunkConditionedDSM` in `core/model.py`.

### 6.1 Conditioning

The model uses:

- a trajectory-type embedding for `c`
- a task embedding for `k`

The condition vector is:

$$
e_{\text{cond}} = \mathrm{MLP}(e_c + e_k).
$$

This condition is shared by all timesteps in the chunk.

### 6.2 Per-step input projection

Each normalized latent in the window is projected independently:

$$
h_i^{(0)} = \mathrm{Proj}(\bar{z}_i),
$$

where `Proj` is:

$$
\mathrm{Linear} \rightarrow \mathrm{LayerNorm} \rightarrow \mathrm{SiLU} \rightarrow \mathrm{Dropout} \rightarrow \mathrm{Linear}.
$$

Stacking all projected steps yields:

$$
H^{(0)} \in \mathbb{R}^{B \times C \times W},
$$

where `C` is the hidden embedding dimension.

### 6.3 AdaLN temporal Conv1d blocks

The old MLP-style `AdaLNResidualBlock` has been replaced by `AdaLNTemporalConvBlock`.

For each block:

1. transpose to channel-last for layer normalization
2. apply condition-dependent AdaLN modulation
3. transpose back to `(B, C, W)`
4. apply a residual temporal Conv1d stack

For hidden sequence `H` and condition `e_cond`, the modulation is:

$$
(\beta, \gamma) = W_{\text{mod}} e_{\text{cond}},
$$

$$
\tilde{H} = \mathrm{LN}(H),
$$

$$
\hat{H} = (1 + \gamma) \odot \tilde{H} + \beta.
$$

The temporal residual branch is:

$$
\mathrm{Conv1d}(C \rightarrow F, k)
\rightarrow \mathrm{GELU}
\rightarrow \mathrm{Dropout}
\rightarrow \mathrm{Conv1d}(F \rightarrow C, k)
\rightarrow \mathrm{Dropout},
$$

where:

- `F` is `ffn_dim`
- `k` is `kernel_size`

The block output is:

$$
H' = H + \mathrm{TemporalConv}(\hat{H}).
$$

The kernel size must be a positive odd integer so the temporal length stays unchanged under symmetric padding.

### 6.4 Output head

After `L` temporal blocks and a final `LayerNorm`, the model predicts a reconstructed latent for every timestep:

$$
\hat{X}_t = [\hat{z}_{t-W+1}, \dots, \hat{z}_t]
          \in \mathbb{R}^{W \times D}.
$$

The output head is a linear map from hidden dimension back to latent dimension, applied per timestep.

Flattening is then applied only for reporting and loss computation:

$$
\hat{z}_{\text{chunk}} = \mathrm{flatten}(\hat{X}_t).
$$

### 6.5 About `num_heads`

The config still accepts `model.num_heads`, but the current temporal Conv1d DSM does not use attention. This field is kept only for compatibility with older configs and checkpoint metadata.

## 7. Denoising Objective

### 7.1 Noise model

Gaussian noise is added in normalized latent-window space:

$$
\tilde{X}_t = \bar{X}_t + \sigma \varepsilon,
\qquad
\varepsilon \sim \mathcal{N}(0, I),
$$

with `\sigma = noise_scale`.

### 7.2 Reconstruction target

The conditional denoiser reconstructs the clean normalized window:

$$
\hat{X}_t = g_{\theta}(\tilde{X}_t, c, k).
$$

Equivalently in flattened form:

$$
\hat{z}_{\text{chunk}} = \mathrm{flatten}(\hat{X}_t),
\qquad
z_{\text{chunk}} = \mathrm{flatten}(\bar{X}_t).
$$

### 7.3 Training loss

The current DSM objective is pure chunk reconstruction MSE:

$$
\mathcal{L}_{\text{DSM}}
=
\mathbb{E}_{X_t,\varepsilon,c,k}
\left[
\frac{1}{W D}
\left\|
\hat{z}_{\text{chunk}} - z_{\text{chunk}}
\right\|_2^2
\right].
$$

There are no active:

- action losses
- transition losses
- Fisher branch losses
- multi-branch balancing weights

The optimizer uses three parameter groups:

- `encoder_branch`
- `shared`
- `chunk_branch`

In the current design:

- `encoder_branch` contains LoRA parameters only
- `shared` contains embeddings, condition MLP, input projection, temporal blocks, and final norm
- `chunk_branch` contains the per-step output head

## 8. Inference Scores

At inference, no noise is added. The same clean latent window is scored twice.

### 8.1 Positive chunk energy

$$
E^{(0)}_{\text{chunk}}
=
\frac{1}{W D}
\left\|
g_{\theta}(\bar{X}_t, c=0, k) - \bar{X}_t
\right\|_2^2.
$$

### 8.2 Failure chunk energy

$$
E^{(1)}_{\text{chunk}}
=
\frac{1}{W D}
\left\|
g_{\theta}(\bar{X}_t, c=1, k) - \bar{X}_t
\right\|_2^2.
$$

### 8.3 Chunk margin

$$
M_{\text{chunk}} = E^{(1)}_{\text{chunk}} - E^{(0)}_{\text{chunk}}.
$$

Interpretation:

- low `E^{(0)}_{\text{chunk}}` means the snippet fits the positive manifold
- high `M_{\text{chunk}}` means the failure-conditioned reconstruction is worse than the positive-conditioned one

## 9. T3 Detector

The detector supports one active score family: `t3_weighted_combo`.

### 9.1 Success-bank standardization

Using a bank of positive trajectories, the detector estimates mean and standard deviation for:

- positive chunk energy
- chunk margin

These statistics are computed globally and per task. Task-specific values are used when available; otherwise the detector falls back to global values.

### 9.2 Per-step T3 score

The standardized terms are:

$$
\widetilde{E}^{(0)}_{\text{chunk}}
=
\frac{E^{(0)}_{\text{chunk}} - \mu_E}{\sigma_E},
$$

$$
\widetilde{M}_{\text{chunk}}
=
\frac{M_{\text{chunk}} - \mu_M}{\sigma_M}.
$$

The T3 step score is:

$$
u_t = \alpha \widetilde{E}^{(0)}_{\text{chunk}} - \beta \widetilde{M}_{\text{chunk}}.
$$

The code records this as two signed components:

- `chunk_energy`
- `chunk_margin`

### 9.3 Lambda aggregation

The detector thresholds an aggregated sequence `\lambda_t`, not raw `u_t`.

If `lambda_mode = mean`:

$$
\lambda_t = \frac{1}{|I_t|} \sum_{i \in I_t} u_i.
$$

If `lambda_mode = max`:

$$
\lambda_t = \max_{i \in I_t} u_i.
$$

`I_t` is either:

- the prefix `[0, t]`, or
- a trailing window of size `lambda_window_size`

depending on detector config.

### 9.4 Threshold calibration

Given calibration lambdas and detector percentile parameter `delta`, the threshold is:

$$
\tau = \mathrm{Percentile}(\lambda, 100 - \delta).
$$

A larger `delta` lowers the threshold and makes the detector more aggressive.

## 10. Data Path and Cache

### 10.1 Preprocessed cache

The active fast path uses a disk-backed preprocessed cache, not a latent cache. Each cache file stores:

- `images_chw`
- `proprio`
- `actions`

The current cache version stores:

- images as `uint8`
- proprio as `float32`
- actions as `float32`

This keeps disk usage lower and avoids full-RAM trajectory preload.

### 10.2 Lazy loading

Training no longer keeps every trajectory resident in RAM. The dataset indexes cached trajectories and loads only the needed sample window in `__getitem__`.

This reduces peak memory while preserving fast repeated access after the first cache build.

### 10.3 Cache invalidation

The cache key depends on:

- task name
- task metadata token
- resolved file path
- demo key
- source file modification time
- image size
- camera names
- proprio normalization stats
- cache version token

So the cache is rebuilt automatically if the underlying data or preprocessing contract changes.

## 11. Self-Contained Checkpoints

`lpb_score` checkpoints are self-contained. They store:

- the full joint model state dict
- optional online model weights when EMA is primary
- `policy_checkpoint_payload`
- `latent_dim`
- `window_size`
- `chunk_dim`
- normalization stats
- task vocabulary metadata
- config snapshot
- `model_architecture`

This means visualization and offline detection can rebuild the encoder directly from `model.dsm_ckpt` without a separate `policy.ckpt`.

Old latent-only checkpoints are intentionally incompatible.

## 12. Main Configuration Keys

Important training keys:

- `dataset.window_size`
- `policy.trainable_encoder`
- `policy.lora.rank`
- `policy.lora.alpha`
- `policy.lora.dropout`
- `model.embed_dim`
- `model.num_layers`
- `model.ffn_dim`
- `model.kernel_size`
- `model.noise_scale`
- `model.std_clamp_min`
- `training.encoder_branch_lr_multiplier`
- `training.shared_lr_multiplier`
- `training.chunk_branch_lr_multiplier`

Important detector keys:

- `detector.alpha`
- `detector.beta`
- `detector.lambda_mode`
- `detector.lambda_window_size`
- `detector.delta`

## 13. Scripts

Training:

```bash
bash robosuite/discriminator/lpb_score/scripts/train_lpb_score_dsm.sh
```

Useful overrides:

```bash
TRAIN_ENCODER=1 \
LORA_RANK=8 \
LORA_ALPHA=16.0 \
LORA_DROPOUT=0.0 \
DSM_WINDOW_SIZE=10 \
TEMPORAL_KERNEL_SIZE=3 \
bash robosuite/discriminator/lpb_score/scripts/train_lpb_score_dsm.sh
```

Visualization:

```bash
DSM_CKPT=/abs/path/to/model.pt \
bash robosuite/discriminator/lpb_score/scripts/visualize_lpb_score_dsm_failures.sh
```

## 14. Practical Notes

- `TRAIN_ENCODER=1` means LoRA tuning, not full encoder fine-tuning.
- `TRAIN_ENCODER=0` keeps the encoder fully frozen.
- `dataset.window_size` used at visualize time must match the checkpoint window size.
- The first run can still be slow because preprocessed cache files must be built.
- Large `batch_size`, `encoder_batch_size`, and `num_workers` can still cause the process to be killed if system memory is insufficient.
