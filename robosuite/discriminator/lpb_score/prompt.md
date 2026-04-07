# Context: Refactoring LPB-New to Denoising Score Matching (DSM)

You are an expert AI and Robotics engineer. We are refactoring the `lpb_new` offline failure detector. 
Currently, it uses a forward-prediction latent world model and a heuristic 4-metric KNN discriminator (Feature KNN, Transition Error, Policy Chunk, Neighbor Dynamics) with learned linear weights. 

We want to upgrade this to a rigorous **Denoising Score Matching (DSM)** framework on the joint manifold of transitions. The core idea is to treat the transition tuple `[z_t, a_{t:t+H-1}, z_{t+H}]` as a single joint vector `tau_t`, inject Gaussian noise during training, and train a Denoising Autoencoder (DAE). During inference, the OOD anomaly score is simply the **Denoising Error** (which is mathematically proportional to the Score Norm / Energy of the distribution).

## Constraints
1. **DO NOT change the policy encoder.** The `FrozenFlowMultitaskEncoder` extracting `z_t = E(o_{\le t})` remains completely unchanged.
2. **Keep code comments short, precise, and ONLY in English.**
3. **Preserve the visualization logic.** We still need to attribute failures and render videos, but the attribution targets will change (see Step 4).

## Step 1: Modify the Model (`core/model.py`)
Rewrite the `LatentWorldModelPredictor` into a `JointManifoldDenoiser`.
* **Input:** Concatenate `z_t`, the action sequence `a_{t:t+H-1}`, and `z_{t+H}` into a single flat continuous vector `tau_t`.
* **Forward Pass (Training):** * Sample Gaussian noise `epsilon ~ N(0, sigma^2 * I)`.
  * Add noise to get `tau_noisy = tau_t + epsilon`.
  * Pass `tau_noisy` through the MLP backbone.
  * Predict the clean `tau_t` (or predict the noise `epsilon` and subtract it).
* **Loss:** Simple Mean Squared Error (MSE) between the predicted `tau` and the ground truth clean `tau_t`. Remove the NLL/uncertainty head.

## Step 2: Modify the Discriminator (`core/knn_discriminator.py`)
Rename to `core/dsm_discriminator.py` (or keep the name but change the class to `DSMDiscriminator`).
* **Remove ALL KNN logic.** Delete `feature_knn`, `policy_chunk`, `neighbor_dynamics`, and `transition_error`. Delete all bank caching logic.
* **New Anomaly Score:** For a given test transition `tau_t` (without adding noise), pass it through the trained DAE. 
  * The raw step score `u_t` is the MSE reconstruction error: `u_t = ||tau_t - D_theta(tau_t)||_2^2`.
* **Temporal Aggregation:** Keep the existing `lambda_mode` (mean/max) and `lambda_window_size` to aggregate `u_t` into `lambda_t` for temporal smoothing.

## Step 3: Simplify the Analyzer (`analyse.py`)
* **Remove Weight Search:** Since we only have ONE score (`u_t`) instead of 4 metrics, we no longer need the BCE linear weighting optimization or simplex constraints.
* **New Task:** The script should now only iterate over the clean validation/calibration dataset to compute the distribution of `lambda_t`. 
* Output a simple JSON saving the quantile threshold `eta` based on the configured `delta` parameter (same logic as the old `AdaptiveKNNDiscriminator.fit()`), along with basic distribution statistics.

## Step 4: Adapt Visualization & Attribution (`visualize_failures.py`)
We still want to know *why* a failure crossed the threshold. 
* Instead of attributing to the old 4 metrics, decompose the scalar denoising error `||tau_t - \hat{tau}_t||^2` into 3 sub-components based on their vector indices:
  1. `state_error`: Reconstruction error of `z_t`.
  2. `action_error`: Reconstruction error of `a_{t:t+H-1}`.
  3. `next_state_error`: Reconstruction error of `z_{t+H}`.
* Feed these 3 components into the existing plotting and video rendering pipeline. If the robot enters an OOD state, `state_error` will spike. If it executes an out-of-distribution action, `action_error` will spike. If the dynamics are violated, `next_state_error` will spike. 

Please provide the updated implementations for `core/model.py`, `core/dsm_discriminator.py`, and the core logic updates for `analyse.py` and `visualize_failures.py`.