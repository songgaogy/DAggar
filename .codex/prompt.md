# Refactoring Prompt: LPB Score - Conditional Score-based Fisher Divergence

**Context:**
We are refactoring the `LPB Score Unified Task-Conditioned DSM` module. The current implementation trains a generative diffusion transformer to estimate the normal manifold using a heteroscedastic NLL loss. We are shifting to a **Conditional Score-based Density Ratio** approach. Instead of calculating a scalar NLL, the model will act as a conditional score predictor, and we will compute the point-wise Fisher Divergence (L2 distance) between the predicted score vectors under two conditions: `expert` (0) and `fail` (1). 

**Core Objective:**
Refactor the dataset pipeline, model architecture, training loop, and inference logic to support trajectory-type conditioning and dual-forward inference. **Do not** add any classification heads or cross-entropy losses. This remains a pure score-matching framework.

Please execute the following refactoring steps:

### 1. Data Pipeline Updates (`app/pipeline.py`, `core/dataset.py`)
* **Include Fail Rollouts in Training:** Update the dataset loaders/samplers so that the training set includes trajectories from `fail_rollout_dir` alongside `expert` and `success_rollout`.
* **Assign Condition Labels:** Inject a binary label `traj_type` into the transition batches:
    * `traj_type = 0` for transitions sourced from `expert` or `success_rollout`.
    * `traj_type = 1` for transitions sourced from `fail_rollout`.
* Ensure this `traj_type` tensor of shape `(batch_size,)` is passed to the model's forward pass.

### 2. Model Architecture Modifications (`core/model.py`)
* **Remove Heteroscedastic Variance:** Remove `pred_logvar` from the model's output head. We are returning to standard deterministic score matching. The output should just be `pred_mean` (the predicted denoised target or predicted noise).
* **Add Condition Embedding:** Add a new embedding layer to `UnifiedConditionedDSM`: `self.type_embedding = nn.Embedding(2, embed_dim)`.
* **Inject Condition:** In the `forward` method, accept the `traj_type` tensor. Extract the `type_token = self.type_embedding(traj_type)`. Add this `type_token` to the existing routing `task_token` (or add it directly to the transformer context tokens). 
* **Loss Function Update:** Replace the heteroscedastic NLL loss with a standard MSE loss: $\mathcal{L} = \text{MSE}(\text{pred\_mean}, \text{target})$. The batch loss is the mean of this MSE across all routed tasks.

### 3. Training Loop Adjustments (`app/train.py`, `core/trainer.py`)
* Remove any code related to logging or optimizing `logvar`.
* Ensure the training loop passes the `traj_type` batch data to the model.
* Retain the current normalized noise injection logic (`add_noise=True`), but ensure the loss is strictly computed as the MSE between the model prediction and the noisy target. 

### 4. Inference & Anomaly Scoring (`core/dsm_discriminator.py`)
* **Implement Dual-Forward Pass:** During `detect_trajectory` or step evaluation (`add_noise=False`), evaluate each transition $x_t$ twice:
    1.  Forward pass with `traj_type = 0` (force tensor of zeros) $\rightarrow$ output `pred_mean_exp`
    2.  Forward pass with `traj_type = 1` (force tensor of ones) $\rightarrow$ output `pred_mean_fail`
* **Compute Fisher Divergence:** The anomaly score for each routed task (state, action, dynamics) is no longer an NLL. It is the squared L2 distance between the two conditional predictions:
    * $E_{state} = ||\text{pred\_mean\_exp}_{state} - \text{pred\_mean\_fail}_{state}||_2^2$
    * $E_{action} = ||\text{pred\_mean\_exp}_{action} - \text{pred\_mean\_fail}_{action}||_2^2$
    * $E_{dynamics} = ||\text{pred\_mean\_exp}_{dynamics} - \text{pred\_mean\_fail}_{dynamics}||_2^2$
* **Final Step Score:** The final `u_t` remains the sum of these three component energies.

### 5. Cleanup & Visualization (`visualize_failures.py`, `app/visualize.py`)
* Update any diagnostic logging to reflect that component energies are now L2 distances (vector field divergences) rather than NLL scalar values.
* Ensure the temporal aggregation (`lambda_t`) logic remains untouched, as it will naturally process the new `u_t` values.