# DIPOLE (Dichotomous Diffusion Policy Optimization) Algorithm Reference

This document provides a concise, mathematically rigorous summary of the DIPOLE framework  adapted for parameter-efficient fine-tuning (**PEFT via LoRA/FiLM**) on top of a single pre-trained base policy within a **Robosuite** environment.

---

## 1. Core Mathematical Concept

DIPOLE replaces the unstable exponential weighting ($\exp(\beta G)$) found in traditional KL-regularized RL with a **greedified, value-aware reference policy** governed by a bounded and smooth sigmoid function.

### Objective Function

The optimization problem maximizes the evaluated return $G(s,a)$ while regularizing the policy toward a value-weighted reference distribution:


$$\max_{\pi}\mathbb{E}_{s\sim d^{\pi}(s)}\left[\mathbb{E}_{a\sim\pi(a|s)}[G(s,a)]-\frac{1}{\omega\beta}D_{KL}\left(\pi(\cdot|s)\;\middle\|\;\mu(\cdot|s)\cdot\frac{\sigma(\beta G(s,a))}{\tilde{Z}(s)}\right)\right]$$

Where:

* 
$\mu(a|s)$ is the reference policy (your pre-trained base policy).


* 
$G(s,a)$ is the advantage function $A(s,a) = Q(s,a) - V(s)$.


* 
$\sigma(x) = \frac{1}{1 + \exp(-x)}$ is the sigmoid function.


* 
$\beta$ is the temperature parameter, and $\omega$ is the greediness factor.



### Dichotomous Decomposition

By leveraging the sigmoid identity $\exp(x) = \frac{\sigma(x)}{1-\sigma(x)}$, the optimal closed-form policy solution factorizes into a ratio of a **positive policy ($\pi^+$)** and a **negative policy ($\pi^-$)**:


$$\pi^*(a|s) \propto \frac{[\pi^+(a|s)]^{1+\omega}}{[\pi^-(a|s)]^\omega}$$

Where the standalone components target opposite reward bounds:


$$\pi^+(a|s) \propto \mu(a|s)\cdot\sigma(\beta G(s,a)) \quad \text{(Reward Maximization)}$$

$$\pi^-(a|s) \propto \mu(a|s)\cdot(1-\sigma(\beta G(s,a))) \quad \text{(Reward Minimization)}$$

---

## 2. PEFT Training Objectives (LoRA / FiLM Setup)

Instead of training two independent neural networks from scratch, you instantiate a single frozen pre-trained base policy $\epsilon_{\theta}(a_t, s, t)$. You then append two separate lightweight parameter adapters (e.g., LoRA matrices or FiLM layers, but if you have better idea or approach, tell me. the best way to do this is to use CFG-style because shared backbone can be updated using both pos+neg data and this is data efficiency, but condition injection is difficult):

* 
**$\Delta\theta^+$**: Adapts the base model to output the positive score $\epsilon_{\theta + \Delta\theta^+}^+(a_t, s, t)$.


* 
**$\Delta\theta^-$**: Adapts the base model to output the negative score $\epsilon_{\theta + \Delta\theta^-}^-(a_t, s, t)$.



### Bounded Loss Functions

Both adapters are optimized using weighted diffusion/flow matching regression losses, completely preventing gradient explosion due to the bounded sigmoid weights:

$$\mathcal{L}(\Delta\theta^+) = \mathbb{E}_{t, \epsilon, s, a} \left[ \sigma(\beta G(s,a) + k) \cdot \left\|\epsilon - \epsilon_{\theta + \Delta\theta^+}^+(a_t,s,t)\right\|^2 \right]$$

$$\mathcal{L}(\Delta\theta^-) = \mathbb{E}_{t, \epsilon, s, a} \left[ (1 - \sigma(\beta G(s,a) + k)) \cdot \left\|\epsilon - \epsilon_{\theta + \Delta\theta^-}^-(a_t,s,t)\right\|^2 \right]$$

Note: $k$ is an empirical distribution shift factor used to balance the sample weight assignments.

---

## 3. Inference & Controllable Action Generation

During deployment in Robosuite, action generation is accomplished by linearly combining the noise/velocity predictions of the dual adapter paths. This directly mimics Classifier-Free Guidance (CFG):

$$\tilde{\epsilon}(a_t, s, t) = (1+\omega)\epsilon_{\theta + \Delta\theta^+}^+(a_t, s, t) - \omega \epsilon_{\theta + \Delta\theta^-}^-(a_t, s, t)$$

By adjusting the scale parameter $\omega$, the coding agent can dynamically scale the policy's aggressiveness during execution without retraining the base or adapter weights.

---

## 4. Algorithmic Implementation Flow

### Algorithm 1: PEFT Trajectory Training

```python
# Setup: Base model parameter θ is frozen. Initialize adapters Δθ+ and Δθ-.
# Input: Robosuite replay buffer D, Hyperparameters: β, ω, k

for step in range(total_training_steps):
    # Sample mini-batch from buffer D
    states, actions, rewards, next_states = D.sample(batch_size)
    
    # Compute advantage/return metric
    # For Robosuite, standard advantage: G = Q(s, a) - V(s)
    G = compute_advantage(states, actions, rewards, next_states)
    
    # Sample diffusion timesteps and noise
    t = uniform_sample(0, 1, batch_size)
    noise = normal_sample(0, I, batch_size)
    actions_t = forward_diffusion(actions, t, noise)
    
    # Calculate bounded weights
    pos_weight = sigmoid(beta * G + k)
    neg_weight = 1.0 - pos_weight
    
    # Optimize positive adapter
    loss_pos = pos_weight * MSE(noise, base_model_with_adapter_pos(actions_t, states, t))
    backward_and_update(loss_pos, target_params=Δθ+)
    
    # Optimize negative adapter
    loss_neg = neg_weight * MSE(noise, base_model_with_adapter_neg(actions_t, states, t))
    backward_and_update(loss_neg, target_params=Δθ-)

```

### Algorithm 2: CFG-Guided Action Sampling

```python
# Input: Current environment state 's' from Robosuite, Greediness factor ω
# Initialization
a_t = normal_sample(0, I, action_dim) 

for t_step in reverse_timesteps(T, 0):
    # Forward pass through parallel adapter streams
    eps_pos = base_model_with_adapter_pos(a_t, s, t_step)
    eps_neg = base_model_with_adapter_neg(a_t, s, t_step)
    
    # Linear extrapolation combination (CFG step)
    eps_guided = (1.0 + w) * eps_pos - w * eps_neg
    
    # Execute single denoising reverse step
    a_t = compute_previous_diffusion_step(a_t, eps_guided, t_step)

# Optional Rejection Sampling step (recommended for robotics)
# Sample N candidate actions from the loop above, evaluate them via Q-network, 
# and execute the action maximizing Q(s, a).

```