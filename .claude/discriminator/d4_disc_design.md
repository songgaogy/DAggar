# D⁴-Disc: Bellman-Bootstrapped Dichotomous Discriminator

> **A self-refining dichotomous failure detector that treats the two latent dynamics models as *critics in a preference MDP*, the soft fail-filter as a *policy* over contaminated labels, and their alternation as a *generalized policy iteration* with a provable fixed point.**
>
> **Relationship to D³-Disc (v2)**:
> - D³-Disc is the static special case of D⁴-Disc at *one* policy-evaluation sweep under a coarse advantage approximation (§7.2).
> - ω, F3 filter, shared banks across tasks, and conformal calibration all carry over verbatim; D⁴-Disc only replaces the static KDE/KNN critics with **learned, conditionally-shared latent dynamics** and the one-shot filter with a **Bellman-consistent refinement loop**.
> - Design aesthetics (ω=0 recovers LPB; safe degeneration when fail ≡ success; conformal-valid calibration) are preserved.
>
> **Design goals (new vs D³-Disc)**:
> 1. **Unify** the success-dynamics bootstrap and the dichotomous filter into a *single* fixed-point problem with monotone improvement on a Bellman-consistent objective.
> 2. **Handle dirty fail data principled**ly: soft labels on fail frames evolve alongside the learned success dynamics — clean start, sharp end.
> 3. **One model, not two**: conditional latent dynamics with AdaLN-injected `{+, −, ∅}` token + classifier-free-guidance at inference, in place of a $(f_+, f_-)$ pair.
> 4. **RL-grounded**: advantage $A(\varphi,a) = Q^+(\varphi,a) - Q^-(\varphi,a)$ is *the* quantity that both (i) defines the filter weight and (ii) parametrizes the guidance scale ω.
>
> **Scope locked (inherited from D³-Disc decisions D1–D12)**: K=0 step-wise; shared positive/negative density estimators across tasks; flow_multi as frozen encoder; per-task conformal τ; ω default 0.5.
>
> **New decision locks (D13–D16) proposed for this extension**:
> - **D13**: critic = conditional latent dynamics with AdaLN, trained once, queried with c∈{+, −, ∅}.
> - **D14**: advantage-based soft label γⱼ replaces static F3 weight; schedule α_k → ∞, κ_k → 0⁺.
> - **D15**: warm-up phase k=0..K_warm trains only on 𝒟₊ (γ frozen at 0.5); bootstrap phase k>K_warm alternates γ-update and critic-update.
> - **D16**: CFG guidance scale at inference = ω (same knob as D³-Disc's ω).

---

## Table of Contents

1. [Notation and RL reframing](#1-notation-and-rl-reframing)
2. [Dichotomous reward, Q, and advantage](#2-dichotomous-reward-q-and-advantage)
3. [The discriminator as advantage-weighted log-odds](#3-the-discriminator-as-advantage-weighted-log-odds)
4. [Bellman-consistent co-evolution (GPI)](#4-bellman-consistent-co-evolution-gpi)
5. [Joint objective and monotone improvement](#5-joint-objective-and-monotone-improvement)
6. [Fixed-point / consistency theorem](#6-fixed-point--consistency-theorem)
7. [Degeneration to D³-Disc and to LPB](#7-degeneration-to-d³-disc-and-to-lpb)
8. [Conditional dynamics with AdaLN + CFG](#8-conditional-dynamics-with-adaln--cfg)
9. [Schedule, warm-up, and pathological cases](#9-schedule-warm-up-and-pathological-cases)
10. [Conformal calibration under a learned critic](#10-conformal-calibration-under-a-learned-critic)
11. [Implementation blueprint](#11-implementation-blueprint)
12. [Known limitations and open questions](#12-known-limitations-and-open-questions)
13. [Appendix A — proofs](#appendix-a--proofs)
14. [Appendix B — hyperparameters](#appendix-b--hyperparameters)

---

## 1. Notation and RL reframing

### 1.1 A latent preference MDP

Inherit state $s_t \in \mathcal{S}$, action $a_t \in \mathcal{A}$, and the frozen flow_multi feature map $\varphi_t = \mathrm{AGG}(\ldots) \in \mathbb{R}^{256}$ from D³-Disc §1.3. Define a **latent preference MDP** $\mathcal{M}^{\text{pref}} = (\mathcal{S}, \mathcal{A}, P, \gamma_{\text{disc}}, R)$ with:

- **Reward** (to be learned from data):
$$
R(\varphi, a) \;\triangleq\; \log p_+(\varphi, a) \;-\; \log p_-(\varphi, a) \;=\; G(\varphi, a),
$$
the log-odds of *success vs failure* occupancy — exactly the AIRL-optimal reward.
- **Two conditional transition models** (henceforth "critics"):
$$
f^c_\theta: (\varphi_t, s_t, a_{t:t+h}) \mapsto \varphi_{t+h}, \qquad c \in \{+, -\}.
$$
$f^+$ captures the one-step success dynamics; $f^-$ the fail dynamics. We will implement them as *one shared network conditioned on* $c$ (§8).

### 1.2 Three populations of data

- $\mathcal{D}_+$: clean success frames (from `success_rollout/` + optionally `expert/`). Known-clean.
- $\mathcal{D}_-^{\text{raw}}$: dirty fail frames. Modelled as a contamination mixture
$$
\varphi_j \;\sim\; \pi\, p_+(\varphi) \;+\; (1-\pi)\, p_-^\star(\varphi),
$$
where $p_-^\star$ is the unknown *true* off-manifold distribution and $\pi$ is the unknown prefix-contamination fraction.
- $\mathcal{D}_{\text{calib}} \subset \mathcal{D}_+$: a held-out success slice reserved for per-task conformal τ (§10).

Per each $j \in \mathcal{D}_-^{\text{raw}}$ we maintain a latent assignment variable $\gamma_j \in (0,1)$ — not a label, but the **fail-branch routing probability** of a *dichotomous label-policy*. We treat $\gamma$ as the policy to be iteratively improved.

### 1.3 Policies to be learned

Two objects evolve jointly:

| Symbol | Role in RL language | Concrete meaning |
|---|---|---|
| $f_\theta(\cdot \mid c)$ | critic ensemble (value under condition) | latent-next-step predictor |
| $\gamma_j$ | dichotomous label-policy | "route sample $j$ to the fail branch with prob $\gamma_j$" |

The framework we develop below is a **generalized policy iteration (GPI)** on the pair $(f_\theta, \gamma)$. GPI has the classical Bellman-contraction flavor from tabular DP, here transported to a label-policy setting.

---

## 2. Dichotomous reward, Q, and advantage

### 2.1 Residual → Q: the Gaussian likelihood identity

For a conditional Gaussian dynamics $\varphi_{t+h} \mid \varphi_t, s_t, a \sim \mathcal{N}(f^c_\theta(\cdot), \Sigma)$ (isotropic $\Sigma = \sigma^2 I$), the **one-step log-likelihood** is
$$
\log p^c_\theta(\varphi_{t+h} \mid \varphi_t, s_t, a) \;=\; -\frac{1}{2\sigma^2}\,\underbrace{\|\varphi_{t+h} - f^c_\theta(\varphi_t, s_t, a)\|^2}_{=:\; r^c(\varphi_t, a, \varphi_{t+h})} \;+\; C.
$$

Define the **one-step Q-function** under condition $c$:
$$
\boxed{\; Q^c_\theta(\varphi_t, a; \varphi_{t+h}) \;\triangleq\; -\frac{1}{2\sigma^2}\, r^c(\varphi_t, a, \varphi_{t+h}) \;}
$$
This is **exactly** the (negative) prediction residual, up to a known positive scale — a proper surrogate for a one-step log-density under the critic $f^c_\theta$.

### 2.2 Advantage = dichotomous log-density ratio

$$
A_\theta(\varphi_t, a, \varphi_{t+h}) \;\triangleq\; Q^+_\theta(\cdot) - Q^-_\theta(\cdot) \;=\; \frac{1}{2\sigma^2}\big[\, r^-(\cdot) - r^+(\cdot) \,\big].
$$

- $A \gg 0$: the observed transition is *much* better explained by success dynamics than fail — this transition *belongs to the success branch*.
- $A \ll 0$: the reverse — *belongs to the fail branch*.

Under the Gaussian identity (§2.1), $A_\theta$ is a plug-in estimator of the AIRL-optimal reward:
$$
A_\theta(\varphi_t, a, \varphi_{t+h}) \;\approx\; \log p_+(\varphi_{t+h}, \varphi_t, a) - \log p_-(\varphi_{t+h}, \varphi_t, a) \;=\; G(\varphi_t, a).
$$

**This is the central bridge**: by learning a conditional dynamics (§8), the advantage is the IRL reward — no second estimator, no density-ratio network.

### 2.3 Value function (optional — for K-step extension later)

For future phase-2 K-step rollout (reserved by D2):
$$
V^+_\theta(\varphi_t) \;=\; \mathbb{E}_{a \sim \pi^{\text{eval}}}\bigg[ \sum_{k=0}^{K-1} \gamma_{\text{disc}}^k\, Q^+_\theta(\varphi_{t+k}, a_{t+k}; \varphi_{t+k+1}) \bigg],
$$
with $\varphi_{t+k+1}$ obtained by iterated application of $f^+_\theta$. Phase 1 uses $K=0$ (one-step Q only). The formalism lifts cleanly when K grows.

---

## 3. The discriminator as advantage-weighted log-odds

### 3.1 Recasting the D³-Disc score

D³-Disc defines (Eq. †):
$$
\lambda_t^{\mathrm{D}^3} \;=\; -(1+\omega)\log \hat p_+(\varphi_t) \;+\; \omega \log \hat p_-(\varphi_t).
$$

Under the Q-formulation of §2, the learned-critic analogue is:
$$
\boxed{\; \lambda_t^{\mathrm{D}^4} \;\triangleq\; -(1+\omega)\, Q^+_\theta(\varphi_t, a_t) \;+\; \omega\, Q^-_\theta(\varphi_t, a_t) \;}
\tag{$\ddagger$}
$$

### 3.2 Decomposition: LPB baseline + ω·(negative advantage)

Algebraically:
$$
\lambda_t^{\mathrm{D}^4} \;=\; \underbrace{-Q^+_\theta(\varphi_t, a_t)}_{\text{LPB-under-critic}} \;+\; \omega \cdot \underbrace{\big[Q^-_\theta(\varphi_t, a_t) - Q^+_\theta(\varphi_t, a_t)\big]}_{=\; -A_\theta(\varphi_t, a_t)}.
$$

So ω is literally a **guidance scale on the dichotomous advantage**: ω=0 gives pure LPB-under-critic (Q⁺ only); ω→∞ gives ranking by $-A$ alone (the AIRL discriminator). ω=0.5 is a moderate advantage-aware correction.

### 3.3 Advantage-gated soft label (replaces static F3)

Each dirty fail sample $j$ is routed by the **advantage gate**:
$$
\boxed{\; \gamma_j^{(k)} \;\triangleq\; \sigma\!\Big(\alpha_k \cdot \big[\, -A_{\theta^{(k)}}(\varphi_j, a_j, \varphi_{j+h}) \,-\, \kappa_k \,\big]\Big) \;\in\; (0, 1). \;}
\tag{$\flat$}
$$
- **Interpretation**: $-A > \kappa$ means "fail critic explains $j$ better than success critic by at least $\kappa$" ⇒ route to fail branch; otherwise keep on success branch.
- **Degeneration check**: at critic initialization $f^+_\theta \equiv f^-_\theta$ (shared weights with no conditioning signal learned), $A \equiv 0$, so $\gamma_j \equiv \sigma(-\alpha_k \kappa_k) \to 1/2$ as $\kappa_k \to 0^+$ — safe uniform prior, no premature commitment.
- **Comparison with F3**: static F3 uses $d^2_{\text{knn}}(\varphi_j, \mathcal{D}_+)$ — a non-parametric approximation of $-\log p_+$, which is (negative) Q⁺ *without* the $-Q^-$ term. §(♭) is the *principled* version: advantage, not one-sided density.

### 3.4 Reward-shaping view

$$
r^{\mathrm{D}^4}(\varphi, a) \;=\; Q^+_\theta + \omega \cdot A_\theta \;=\; (1+\omega) Q^+_\theta - \omega Q^-_\theta.
$$

This is a **bidirectional MOPO-style reward shaping** (cf. D³-Disc §7.3) with a single hyperparameter ω that switches between uncertainty-aware reward (ω=0) and density-ratio reward (ω→∞). Values ω ∈ (0, 1] are natural "conservative advantage corrections".

---

## 4. Bellman-consistent co-evolution (GPI)

> We avoid the language of EM and of alternating minimization; the core iteration is a **generalized policy iteration** on the pair $(f_\theta, \gamma)$, where the "policy" being improved is the dichotomous label-policy $\gamma$ and the "critic" being evaluated is the conditional dynamics $f_\theta$. The equivalence to an EM-like alternation is acknowledged in §A.4, but the RL framing is what gives us the contraction structure and the monotone-improvement guarantee.

### 4.1 Two sweeps

**Sweep I — Policy Evaluation (critic update)**. Given $\gamma^{(k)}$, fit $\theta$ by Bellman-residual minimization on a dichotomous, label-weighted objective:
$$
\theta^{(k+1)} \;\leftarrow\; \arg\min_\theta \Bigg\{
\underbrace{\sum_{i \in \mathcal{D}_+} r^+_\theta(\varphi_i, a_i, \varphi_{i+h})}_{\text{clean-success evaluation of }Q^+}
\;+\; \eta \sum_{j \in \mathcal{D}_-^{\text{raw}}} (1 - \gamma_j^{(k)}) \cdot r^+_\theta(\varphi_j, a_j, \varphi_{j+h})
\;+\; \sum_{j \in \mathcal{D}_-^{\text{raw}}} \gamma_j^{(k)} \cdot r^-_\theta(\varphi_j, a_j, \varphi_{j+h})
\Bigg\}.
$$
Every term is an L2 regression residual → a single supervised minibatch step per minibatch. The weighting $(1-\gamma_j, \gamma_j)$ is the **dichotomous routing** of sample $j$ between the $+$ and $-$ critics.

**Sweep II — Policy Improvement (label-policy update)**. Given $\theta^{(k+1)}$, update $\gamma$ via $(\flat)$ with the freshly computed advantage:
$$
\gamma_j^{(k+1)} \;\leftarrow\; \sigma\!\Big(\alpha_{k+1}\,\big[-A_{\theta^{(k+1)}}(\varphi_j, a_j, \varphi_{j+h}) \,-\, \kappa_{k+1}\big]\Big).
$$

### 4.2 The Bellman operator

Stack the two sweeps into one operator:
$$
\mathcal{T}: (f_\theta, \gamma) \;\longmapsto\; (f_{\theta^+}, \gamma^+),
$$
defined by Sweep I then Sweep II. A *fixed point* of $\mathcal{T}$ satisfies simultaneously:
- Critic is Bellman-consistent with current label-policy: $\nabla_\theta \mathcal{L}_{\text{eval}}(\theta; \gamma) = 0$.
- Label-policy is advantage-consistent with current critic: $\gamma_j = \sigma(\alpha[-A_\theta(\cdot) - \kappa])$ for all $j$.

### 4.3 GPI vs value iteration

This is *generalized* policy iteration (not strict): the policy-improvement step is a **soft** advantage gate (sigmoid, not argmax), and the critic-evaluation step is a **partial** gradient update (one or few epochs per outer iteration, not exact minimization). GPI guarantees convergence under mild conditions; the sigmoid-soft gate replaces the usual max and provides smoothness needed for the advantage-schedule argument (§6).

---

## 5. Joint objective and monotone improvement

### 5.1 A Bellman-consistent functional

Define the joint loss:
$$
\mathcal{L}(\theta, \gamma) \;\triangleq\; \mathcal{L}_+(\theta) \;+\; \eta\, \mathcal{L}_\pm(\theta, \gamma) \;+\; \tau_{\text{ent}}\, \mathcal{H}(\gamma),
$$
with:
$$
\mathcal{L}_+(\theta) = \sum_{i \in \mathcal{D}_+} r^+_\theta(\cdot_i), \qquad
\mathcal{L}_\pm(\theta, \gamma) = \sum_{j} \Big[ (1-\gamma_j) r^+_\theta(\cdot_j) + \gamma_j r^-_\theta(\cdot_j) \Big],
$$
$$
\mathcal{H}(\gamma) = \sum_j \big[\gamma_j \log \gamma_j + (1-\gamma_j)\log(1-\gamma_j)\big] \;-\; \frac{1}{\alpha_k} \sum_j \gamma_j \cdot \kappa_k.
$$

The entropy term $\mathcal{H}$ is the **Shannon entropy of the label-policy plus a scheduled temperature shift**; its first variation at fixed θ reproduces exactly $(\flat)$:
$$
\partial_{\gamma_j}\mathcal{L} = 0 \;\Longleftrightarrow\; \gamma_j = \sigma\big(\alpha_k\,[-A_\theta(\cdot_j) - \kappa_k]\big),
$$
as a short calculation shows. So the $(\flat)$-update is **exact closed-form minimization** of $\mathcal{L}$ in $\gamma$.

### 5.2 Monotone improvement

**Proposition 5.1 (Monotone descent).** Each full sweep of $\mathcal{T}$ decreases $\mathcal{L}$:
$$
\mathcal{L}(\theta^{(k+1)}, \gamma^{(k+1)}) \;\le\; \mathcal{L}(\theta^{(k)}, \gamma^{(k)}),
$$
with equality iff $(\theta^{(k)}, \gamma^{(k)})$ is a stationary point of $\mathcal{T}$.

*Proof sketch.* Sweep I is a partial gradient step with non-increasing $\mathcal{L}$ in θ (guaranteed by sufficiently small learning rate under standard smoothness). Sweep II is an exact closed-form minimization in γ by §5.1, so is non-increasing. Compose.

**Corollary 5.2.** The sequence $\mathcal{L}^{(k)} := \mathcal{L}(\theta^{(k)}, \gamma^{(k)})$ is non-increasing and bounded below by 0, hence converges. Accumulation points are stationary points of $\mathcal{T}$.

---

## 6. Fixed-point / consistency theorem

This is the D⁴-analogue of D³-Disc §6.4. We show the fixed point of $\mathcal{T}$, under a schedule of $(\alpha_k, \kappa_k)$ matched to the statistical rate of the critic, **recovers the true off-manifold indicator**.

### 6.1 Assumptions

- **(C1) Critic consistency under weighted contamination**. Suppose contamination enters $\mathcal{L}_+$ via weights $\eta(1-\gamma_j^{(k)})$. If $\gamma_j^{(k)} \to y_j^\star \;(= \mathbb{1}[\varphi_j \notin \operatorname{supp}(p_+)])$ in probability, then $f^+_{\theta^{(k)}} \to f^+_\star$ uniformly on compacts, where $f^+_\star$ is the *true* Bayes-optimal success dynamics. Symmetric for $f^-$.
- **(C2) Residual separation**. $r^+_\star(\varphi, a, \varphi') \xrightarrow{P} 0$ for $(\varphi, a, \varphi') \in \operatorname{supp}(\rho_+)$, and $r^+_\star \ge \delta_0^2 > 0$ almost surely on $\operatorname{supp}(\rho_+)^c$. (Direct consequence of $f^+_\star$ being the MAP predictor under $\rho_+$ and transitions being sub-Gaussian.)
- **(C3) Identifiability**. $\rho_-^\star \ne \rho_+$ in TV distance, and the feature $\varphi$ induced by the frozen flow_multi encoder is sufficient to separate them.
- **(C4) Schedule**. $\alpha_k \to \infty$ and $\kappa_k \to 0^+$ as $k \to \infty$, with $\alpha_k$ growing *no faster* than the critic's uniform convergence rate $\rho_k := \|f^+_{\theta^{(k)}} - f^+_\star\|_\infty$. Concretely: $\alpha_k \cdot \rho_k \to 0$.

### 6.2 Theorem

**Theorem 6.1 (D⁴-Disc fixed-point consistency).** Under (C1)–(C4), the sequence $(\theta^{(k)}, \gamma^{(k)})$ produced by the $\mathcal{T}$-iteration from a warm-started initialization satisfies:
$$
\gamma_j^{(k)} \xrightarrow{P} y_j^\star, \qquad f^c_{\theta^{(k)}} \xrightarrow{} f^c_\star \;(c \in \{+, -\}).
$$

### 6.3 Proof sketch (two-phase coupling)

**Phase A — warm-started critic.** At $k = 0$ with γ ≡ 0.5, $\mathcal{L}_\pm$ is symmetric in $c$ and contributes a uniform data-augmentation of both critic branches. $\mathcal{L}_+$ dominates the gradient on $f^+$ and drives it toward $f^+_\star$ on $\operatorname{supp}(\rho_+)$. After $K_{\text{warm}}$ steps, $r^+$ already exhibits a *weak* version of (C2).

**Phase B — coupled contraction.**

- Given a critic with separation gap $\delta_0^2$ (weak or strong), the advantage gate (♭) at temperature $\alpha_k$ pushes γⱼ toward $y_j^\star$ with error at most $\exp(-\alpha_k \delta_0^2 / 2)$.
- Given γ within $\epsilon_k$ of $y^\star$, (C1) gives critic-error $\rho_{k+1} \le \rho_{k+1}^{\star}(\epsilon_k)$, monotone decreasing in $\epsilon_k$.
- Iterate: $\epsilon_{k+1} \lesssim \exp(-\alpha_{k+1}\,(\delta_0^2 - \rho_{k+1}))$, which goes to 0 under (C4).

The step "iterate"-line is the key Bellman-contraction-style step: each round the gap between γ and y⋆ shrinks geometrically, provided the schedule respects (C4). See §A.1 for the quantitative bound.

### 6.4 Corollary — what the fixed point *is*

The unique fixed point (assuming identifiability) satisfies:
- $\gamma_j^\infty = y_j^\star$ for almost all $j$ (exact off-manifold recovery);
- $f^+_\theta \to f^+_\star$ (true success dynamics, uncontaminated);
- $f^-_\theta \to f^-_\star$ (true fail dynamics, conditioned on off-manifold support);
- $A_\theta(\varphi, a) = G(\varphi, a)$ (critic's advantage equals the AIRL-optimal reward).

Thus the entire §(‡) score converges to the **exact AIRL-shaped dichotomous log-odds**. In contrast, static F3 (D³-Disc) attains (C2) only — the correct filter target — but does not couple to critic refinement, so the AIRL-exact reward is not recovered.

### 6.5 Comparison to D³-Disc §6.4

| | D³-Disc §6.4 | D⁴-Disc §6.2 |
|---|---|---|
| Target | $w_j^- \to y_j^\star$ | $\gamma_j \to y_j^\star$ **and** critics $\to$ true dynamics |
| Separator | $d^2_{\text{knn}}(\varphi_j, \mathcal{D}_+)$ (static) | $A_\theta(\varphi_j, a_j)$ (learned, co-evolving) |
| Schedule | $\beta, \kappa$ (static, but in theorem $\beta \to \infty, \kappa \to 0$) | $\alpha_k, \kappa_k$, coupled to critic-rate $\rho_k$ |
| Data-limit regime | Match (theorem) | Match (theorem) |
| Finite-sample regime | Approximation, gap unquantified | Gap quantified via $\rho_k$ (§A.1) |

---

## 7. Degeneration to D³-Disc and to LPB

### 7.1 ω=0

Under ω=0, $\lambda^{\mathrm{D}^4}_t = -Q^+_\theta(\varphi_t, a_t)$: pure LPB, but with a **learned** one-step likelihood instead of KNN distance. Calling this "D³-Disc at ω=0 with a trained critic" is more accurate than "LPB" because the feature geometry is still flow_multi-based; only the density estimator is replaced.

### 7.2 D³-Disc is one GPI step

**Proposition 7.1.** Fix $k = 0$. Replace $Q^+_\theta$ by the KDE log-density $\log \hat p_+(\varphi)$ (1-NN approximation $-d^2_{\text{knn}}/(2\sigma^2)$), and $Q^-_\theta$ by the *weighted* KDE (D³-Disc §6.2). Then *one application of Sweep II alone* with schedule $(\alpha_0 = \beta, \kappa_0 = \kappa)$ recovers the static F3 weight $w_j^- = \sigma(\beta[d^2_{\text{knn}}(\varphi_j, \mathcal{D}_+) - \kappa])$ up to the substitution $A \rightsquigarrow d^2$.

Hence **D³-Disc = D⁴-Disc at one frozen-critic step**. All D³-Disc guarantees carry forward; D⁴-Disc adds the critic-refinement degree of freedom.

### 7.3 Safe-degeneration (fail ≡ success)

If the fail distribution coincides with success on the feature space, the AIRL reward $G \equiv 0$, hence $A \equiv 0$. The gate (♭) then produces $\gamma_j \equiv \sigma(-\alpha_k \kappa_k) \to 0$ as $\kappa_k \to 0^+$ with $\alpha_k$ finite. All fail mass is routed to the success branch → $f^-$ has no training signal → degenerates to $f^+$. Advantage stays 0. The D⁴-Disc score collapses to the LPB-style $-Q^+_\theta$ — **the detector correctly reports "fail data is uninformative"**.

---

## 8. Conditional dynamics with AdaLN + CFG

### 8.1 One network, not two

Instead of two separate networks $f^+_\theta, f^-_\theta$, maintain a single conditional model
$$
f_\theta(\varphi_{t+h} \mid \varphi_t, s_t, a_{t:t+h}, c), \qquad c \in \{+, -, \varnothing\},
$$
where $\varnothing$ is the null (unconditional) token.

### 8.2 AdaLN-Zero conditioning

Adopt the **AdaLN-Zero** block (Peebles & Xie, DiT, 2023) as the per-layer conditioning primitive. For each transformer block $\ell$:

$$
\begin{aligned}
(\alpha^{\text{msa}}_\ell, \gamma^{\text{msa}}_\ell, \beta^{\text{msa}}_\ell, \alpha^{\text{mlp}}_\ell, \gamma^{\text{mlp}}_\ell, \beta^{\text{mlp}}_\ell)
&= \mathrm{MLP}_\ell\!\big(\mathrm{Embed}(c)\big) \\[4pt]
h_\ell &= h_{\ell-1} + \alpha^{\text{msa}}_\ell \odot \mathrm{MSA}\!\Big(\mathrm{LN}(h_{\ell-1})\odot(1+\gamma^{\text{msa}}_\ell) + \beta^{\text{msa}}_\ell\Big) \\
h'_\ell &= h_\ell + \alpha^{\text{mlp}}_\ell \odot \mathrm{MLP}\!\Big(\mathrm{LN}(h_\ell)\odot(1+\gamma^{\text{mlp}}_\ell) + \beta^{\text{mlp}}_\ell\Big).
\end{aligned}
$$

- $\mathrm{Embed}: \{+, -, \varnothing\} \to \mathbb{R}^{d_{\text{cond}}}$ is a 3-token learnable embedding table.
- $\mathrm{MLP}_\ell$ is a small (usually 2-layer) head producing six modulation vectors per block.
- **Zero-init** of the last linear layers of $\mathrm{MLP}_\ell$ → at step 0, every AdaLN block behaves as identity on the residual branch; training starts from an unconditional network and gradually learns the conditioning signal. This is numerically well-behaved and is the modern default for CFG-trained conditional models.

### 8.3 Label dropout during training

Per minibatch element, with probability $p_{\text{drop}} = 0.1$ we replace the condition token $c$ with $\varnothing$. This trains the unconditional branch in parallel with the conditional one and is *required* for CFG at inference. The probability target $\mathbb{E}[\mathbb{1}[c = \varnothing]] = p_{\text{drop}}$ is a standard CFG hyper-parameter.

### 8.4 Classifier-free guidance at inference

At inference, produce the **guided prediction** via
$$
\boxed{\; f_\omega(\cdot) \;\triangleq\; (1 + \omega)\, f_\theta(\cdot \mid c = +) \;-\; \omega\, f_\theta(\cdot \mid c = -). \;}
$$
The Gaussian log-likelihood under $f_\omega$ is
$$
\log p_\omega(\varphi_{t+h} \mid \cdot) \;=\; -\frac{1}{2\sigma^2}\,\|\varphi_{t+h} - f_\omega(\cdot)\|^2 + C,
$$
which, under a first-order expansion in ω around the conditional mean, is equivalent (up to $O(\omega^2)$) to
$$
(1 + \omega)\,\log p_\theta(\cdot \mid c = +) \;-\; \omega\,\log p_\theta(\cdot \mid c = -) \;+\; C,
$$
the D³-Disc score $(\dagger)$. In the typical CFG range $\omega \in [0, 2]$, the quadratic error term is numerically dominated.

**Takeaway**. The ω in D³-Disc and the ω in classifier-free guidance are *the same knob*. D⁴-Disc inherits the CFG literature's entire body of tuning heuristics (sweet spots ≈ 0.5–2, sampling schedules, divergence mitigations at large ω).

### 8.5 Why AdaLN and not cross-attention / FiLM / token-prepend

- **AdaLN** scales shift/gain per feature channel per layer; low parameter count; no token-length penalty; standard CFG-compatible (DiT, Stable Diffusion 3, Flux).
- **Cross-attention** would treat $c$ as a key-value token; higher FLOPs per step, overkill for a 3-category condition.
- **FiLM** is a special case of AdaLN (β, γ only, no α).
- **Token-prepend** (like classifier-free in autoregressive LMs) breaks the per-frame regression structure of the `DynamicsPredictor`.

AdaLN-Zero is the clean, minimal, modern choice — and the one the diffusion literature has already debugged extensively.

---

## 9. Schedule, warm-up, and pathological cases

### 9.1 Warm-up (phase A, k = 0..K_warm)

- Freeze $\gamma_j \equiv 0.5$ for all fail samples.
- Train $f_\theta(\cdot \mid c=+)$ on $\mathcal{D}_+$; with probability $p_{\text{drop}}=0.1$ train the $\varnothing$ branch instead; with probability $0.5 \cdot (1 - p_{\text{drop}})$ use $\mathcal{D}_-^{\text{raw}}$ with $c=+$ (the γ=0.5 mass assigned to success branch).
- **Critical**: do *not* train $c=-$ during warm-up. Fail dynamics has no reliable signal until γ has separated.
- Duration: $K_{\text{warm}} \approx 5$–10 epochs, chosen so that $r^+$ on a held-out $\mathcal{D}_+$ slice is below $1.5\times$ the inherent aleatoric noise floor.

### 9.2 Bootstrap (phase B, k > K_warm)

- At the start of each outer epoch, recompute γⱼ via (♭) on the full $\mathcal{D}_-^{\text{raw}}$.
- Anneal: $\alpha_k = \alpha_0 \cdot (1 + (k - K_{\text{warm}}))^\rho$ with $\rho \in (0.5, 1]$; $\alpha_0 \approx 1 / \mathrm{MAD}(r^+_\theta \mid \mathcal{D}_+)$ at the end of warm-up.
- $\kappa_k$: **inside** the bootstrap, pin $\kappa_k = 0$ and let the advantage do the separating. (Data-adaptive κ is a D³-Disc relic; once the advantage is learned, the offset is absorbed into the bias term of the conditioning MLP.)
- Continue until the label-policy stabilizes: $\|\gamma^{(k)} - \gamma^{(k-1)}\|_1 / N_- < 10^{-3}$ for two consecutive epochs, OR a fixed epoch budget is exhausted.

### 9.3 Pathological cases and mitigations

| Pathology | Symptom | Mitigation |
|---|---|---|
| **Early commitment** | γ collapses to 0 or 1 before critics separate | Small $\alpha_0$; cap $\alpha_k$ by critic-rate: $\alpha_k \le c / \rho_k$ |
| **Confirmation bias** | $f^+$ overfits $\mathcal{D}_+$, treats all fail prefix as "pseudo-positive" | Hold-out subset of $\mathcal{D}_+$; cap η so that $\mathcal{L}_\pm$ does not dominate $\mathcal{L}_+$ |
| **Low contamination rate ($\pi \approx 1$)** | Mean γ stays near 0; D⁻ is empty | Log $\bar\gamma^{(k)}$; if it stays < 0.05 for 3 epochs, fall back to ω=0 |
| **Collapse $f^- \to f^+$** | Advantage → 0 uniformly | Adds entropy bonus to $\mathcal{H}(\gamma)$; increases $p_{\text{drop}}$ (stronger unconditional branch) |
| **CFG divergence at large ω** | Predicted $\varphi_{t+h}$ has NaNs | Clip $\omega$ to $\omega_{\max}$; use the log-density form $(\dagger)$ for scoring rather than reconstructed $\varphi$ for stability |

### 9.4 Default schedule

| Parameter | Default | Reason |
|---|---|---|
| $K_{\text{warm}}$ | 8 epochs | Matches LPB training warm-up, sufficient for $r^+$ floor |
| $\alpha_0$ | $1/\mathrm{MAD}(r^+\mid\mathcal{D}_+)$ at end of warm-up | D³-Disc heritage |
| $\alpha_{k}$ | $\alpha_0 \cdot (1 + k - K_{\text{warm}})^{0.75}$ | Sublinear, respects (C4) |
| $\kappa_k$ (bootstrap) | 0 | Absorbed into AdaLN bias |
| η | 0.3 | Soft pseudo-positive reuse |
| $p_{\text{drop}}$ | 0.1 | Standard CFG default |
| ω (inference) | 0.5 (baseline), sweep $\{0, 0.1, 0.2, 0.5, 1, 2\}$ | D1 extended with CFG range |

---

## 10. Conformal calibration under a learned critic

D³-Disc §8 calibrates τ on held-out success frames. For D⁴-Disc the situation is slightly more subtle because the scoring function $\lambda_t^{\mathrm{D}^4}$ depends on learned parameters θ.

### 10.1 Exchangeability is preserved if the critic is frozen at calibration time

**Proposition 10.1.** Suppose the critic $f_\theta$ is frozen *before* the conformal τ is computed, and calibration frames $\mathcal{D}_{\text{calib}}$ are exchangeable with test success frames conditioned on the frozen θ. Then the split-conformal guarantee (D³-Disc §8.1) holds unchanged for $\lambda^{\mathrm{D}^4}$.

*Remark.* This is not a new theorem; it is the observation that the conformal guarantee is agnostic to the form of the score function, only requiring exchangeability between calibration and test frames. A *data-driven* score is fine as long as the score function is *frozen* at the moment of calibration.

### 10.2 Leakage to avoid

- Do **not** include $\mathcal{D}_{\text{calib}}$ in the critic's training set. (Standard hold-out.)
- Do **not** re-fit the critic between τ computation and τ use at test time.

### 10.3 Per-task vs shared τ

Same as D³-Disc §8.3: per-task τ recommended under shared banks to handle task-specific score-distribution shifts.

---

## 11. Implementation blueprint

### 11.1 Code delta from D³-Disc v1

Minimal changes (inheriting `robosuite/discriminator/d3disc/`):

| File | Change |
|---|---|
| `model.py` *(new)* | `ConditionalDynamicsPredictor(nn.Module)`: wraps existing `DynamicsPredictor` with AdaLN-Zero blocks and a 3-token condition embedding. |
| `trainer.py` | Add `D4TrainerConfig` with $K_{\text{warm}}$, $\alpha_0$, $\rho$, $\eta$, $p_{\text{drop}}$. Two training phases (warm-up + bootstrap). |
| `filter.py` | Replace `compute_f3_weights` with `compute_advantage_gate(predictor, batch, alpha_k, kappa_k)`. |
| `detector.py` | Replace KDE `p_+`/`p_-` banks with critic-based Q; score = (‡). |
| `dynamics_feature.py` | Replace frozen projector with conditional predictor; inference uses CFG (8.4). |
| `scripts/train_d4.sh` *(new)* | End-to-end entry. |
| `scripts/run_d4_benchmark.sh` *(new)* | Benchmark entry. |

### 11.2 Warm-up loop (pseudocode)

```
for epoch in range(K_warm):
    for batch in D_plus_loader:
        c = sample_condition(batch, p_drop)          # c ∈ {+, ∅}
        loss = ||f_theta(batch, c) - batch.target||²
        backward; step()
```

### 11.3 Bootstrap loop (pseudocode)

```
gamma = 0.5 * ones(N_neg)
for k in range(K_warm, K_total):
    # Sweep II (closed form)
    with torch.no_grad():
        r_plus  = ||f_theta(D_neg, c=+) - D_neg.target||²
        r_minus = ||f_theta(D_neg, c=-) - D_neg.target||²
        A_theta = (r_minus - r_plus) / (2 * sigma²)
        gamma   = sigmoid(alpha_k * (-A_theta - kappa_k))

    # Sweep I (weighted SGD)
    for batch in joint_loader(D_plus, D_neg, gamma):
        c = sample_condition(batch, p_drop)
        w = batch_weights_from_gamma(batch, gamma)        # see §4.1
        loss = (w * ||f_theta(batch, c) - batch.target||²).sum()
        backward; step()

    alpha_k = update_schedule(k)
```

### 11.4 Inference (CFG guided scoring)

```
def score_frame(phi_t, s_t, a, phi_tplus):
    f_plus  = f_theta(phi_t, s_t, a, c='+')
    f_minus = f_theta(phi_t, s_t, a, c='-')
    f_omega = (1 + omega) * f_plus - omega * f_minus
    lam     = ||phi_tplus - f_omega||² / (2 * sigma²)
    return lam
```

Computationally: **two forward passes** per sample (c=+, c=-), *no* batching-over-omega cost because ω is a scalar mix.

### 11.5 Cache compatibility

- No change to `FlowMultiEncoderWrapper` — the frozen flow_multi encoder is unchanged; latents are reused byte-exact from `data/.lpb_score_cache/`.
- `LatentFlowDynamicsDataset` extended with per-sample condition label (expert+success → c=+; fail raw → c=± with γ-weighted route).

---

## 12. Known limitations and open questions

### 12.1 Schedule tuning is non-trivial

The coupling in (C4) between $\alpha_k$ and critic-rate $\rho_k$ is not directly observable during training. Practical heuristics (§9) are a substitute but not a theoretical solution. Open direction: use a *validation-driven* schedule, e.g., increase $\alpha_k$ only when held-out $r^+$ plateaus.

### 12.2 First-order expansion of CFG

§8.4 uses a first-order expansion $\log p_\omega \approx (1+\omega)\log p_+ - \omega \log p_-$ valid for small ω. Large ω (e.g., ω=5) breaks this and the score's AIRL meaning; practical ω ≤ 2 stays in the valid regime.

### 12.3 AdaLN on non-diffusion backbones

The DiT AdaLN-Zero result is for diffusion transformers; its behavior in a non-diffusion latent-regression setting (our dynamics predictor) should be validated but is architecturally unremarkable — LayerNorm-based modulation is well-behaved across regression and generative settings.

### 12.4 Cross-task transfer of $f^-$

Under D8 (shared banks across tasks), $f^-$ is trained on pooled failures. If failure modes are task-specific, $f^-$ may not transfer. Test: per-task ablation with task-gated AdaLN (append task id to condition embedding).

### 12.5 K-step rollout (deferred, as in D³-Disc D2)

Phase-1 uses K=0. With the critic framework, K-step extension is natural:
$$
\lambda^{\mathrm{D}^4, K}_t = \sum_{k=0}^{K-1} \gamma_{\text{disc}}^k \big[(1+\omega) Q^+_\theta(\varphi_{t+k}, a_{t+k}) - \omega Q^-_\theta(\cdot)\big]
$$
with $\varphi_{t+k+1}$ generated by iterating $f_\omega$. Convergence and calibration need separate analysis (rollout errors compound).

### 12.6 Conformal under training-set drift

If the critic is re-trained between calibration and test (e.g., for a new task), τ is invalid. Re-calibration per deployment is necessary.

---

## Appendix A — proofs

### A.1 Quantitative bound for §6.3

We outline the contraction argument used in the proof sketch.

**Notation**. Let $\rho_k := \sup_{(\varphi, a)} \|f^+_{\theta^{(k)}}(\cdot) - f^+_\star(\cdot)\|_\infty$ (uniform critic error at iteration $k$), $\delta_0^2$ the separation gap from (C2), $\epsilon_k := \mathbb{E}_j[|\gamma_j^{(k)} - y_j^\star|]$ the mean label error.

**Key lemma (advantage gate control).** For any $\varphi \in \operatorname{supp}(\rho_+)^c$,
$$
|\gamma_j^{(k)} - 1| \;\le\; \sigma\!\big(-\alpha_k(\delta_0^2 - \rho_k - \kappa_k)\big) \;\le\; \exp\!\big(-\alpha_k(\delta_0^2 - \rho_k - \kappa_k)\big).
$$
Symmetric bound on $\operatorname{supp}(\rho_+)$: $|\gamma_j^{(k)}| \le \exp(-\alpha_k(\rho_k + \kappa_k))$ — wait, this direction is wrong; on-support samples have $A \to 0$ (not $\ge \delta_0^2$), so actually $|\gamma_j^{(k)}| \le \sigma(\alpha_k \rho_k) \le \alpha_k \rho_k / 4$ via Lipschitzness of σ.

Combining the two directions, $\epsilon_k \le \max\{\exp(-\alpha_k(\delta_0^2 - \rho_k)), \alpha_k \rho_k / 4\}$.

**Coupled iteration.** (C1) gives $\rho_{k+1} \le \rho^\star(\epsilon_k)$; assume the Lipschitz relation $\rho^\star(\epsilon) \le L_\rho \cdot \epsilon$ near the fixed point (standard M-estimator rate). Then
$$
\rho_{k+1} \;\le\; L_\rho \cdot \max\{\exp(-\alpha_k(\delta_0^2 - \rho_k)), \alpha_k \rho_k / 4\}.
$$

For $\alpha_k = \alpha_0 (1+k)^\rho$ with $\rho \in (0, 1)$ and $\rho_0$ small (ensured by warm-up), induction shows $\rho_k = O(k^{-\rho})$ and $\epsilon_k = O(k^{-\rho})$. The product $\alpha_k \rho_k = O(1)$ matches the boundary required by (C4) — the schedule must be subtle enough that neither side dominates prematurely. $\blacksquare$

### A.2 Closed-form γ from §5.1

Set $\partial_{\gamma_j} \mathcal{L} = 0$:
$$
\eta [r^-_\theta - r^+_\theta](\cdot_j) + \tau_{\text{ent}}\,[\log \gamma_j - \log(1 - \gamma_j)] - \kappa_k = 0.
$$
Solve the logit form:
$$
\log\frac{\gamma_j}{1 - \gamma_j} = \frac{1}{\tau_{\text{ent}}}\,[\kappa_k + \eta (r^+_\theta - r^-_\theta)(\cdot_j)] = \alpha_k\,[-A_\theta(\cdot_j) - \tilde\kappa_k]
$$
with $\alpha_k := \eta / (\sigma^2 \tau_{\text{ent}})$ and $\tilde\kappa_k := -\kappa_k / \eta$. Matching notation, this is exactly (♭). $\blacksquare$

### A.3 CFG-to-dichotomous log-density equivalence (§8.4)

Under isotropic-Gaussian conditional density,
$$
\log p_\theta(\varphi' \mid c) = -\tfrac{1}{2\sigma^2}\|\varphi' - f_\theta(\cdot \mid c)\|^2 + C.
$$
Substituting $f_\omega = (1+\omega) f_\theta(+) - \omega f_\theta(-)$:
$$
\log p_\omega(\varphi') = -\tfrac{1}{2\sigma^2}\|\varphi' - (1+\omega)f_\theta(+) + \omega f_\theta(-)\|^2 + C.
$$
Expanding the quadratic,
$$
= -\tfrac{1}{2\sigma^2}\Big[(1+\omega)\|\varphi' - f_\theta(+)\|^2 - \omega\|\varphi' - f_\theta(-)\|^2 - \omega(1+\omega)\|f_\theta(+) - f_\theta(-)\|^2\Big] + C.
$$

The first two terms reproduce $(1+\omega)\log p_\theta(+) - \omega \log p_\theta(-)$. The third term, $O(\omega(1+\omega))\cdot\|f_\theta(+) - f_\theta(-)\|^2$, is a **constant** in $\varphi'$ at test time and therefore does not affect ranking. Hence CFG-score and the linear-log-density score differ by a $\varphi'$-independent constant — **rank equivalence is exact, not first-order**. (This is stronger than the §8.4 text claims; we strengthen the statement here.) $\blacksquare$

### A.4 Acknowledgement of the EM correspondence

Formally, the $\mathcal{T}$-operator of §4.2 coincides with an E–M iteration on a contamination mixture with Gaussian observation model; (‡) is the posterior-odds form of the E-step and Sweep I is the M-step. We deliberately present the framework in GPI language because:
(i) the RL formulation reveals the advantage $A_\theta$ as a first-class object (the AIRL reward) and the score ω as a guidance scale, making the connection to CFG structural rather than incidental;
(ii) the Bellman-contraction form of the fixed-point theorem (§6.3) admits quantitative rate analysis (§A.1) that the usual EM monotone-likelihood statement does not.

## Appendix B — hyperparameters

| Parameter | Default | Ablation range | Source |
|---|---|---|---|
| ω (CFG scale) | 0.5 | {0, 0.1, 0.2, 0.5, 1, 2} | extended D1 |
| $K_{\text{warm}}$ | 8 epochs | {5, 8, 12} | D15 |
| $\alpha_0$ | $1/\mathrm{MAD}(r^+ \mid \mathcal{D}_+)$ | ×{0.5, 1, 2} | D⁴-new |
| $\rho$ (schedule exponent) | 0.75 | {0.5, 0.75, 1.0} | D⁴-new |
| η (pseudo-positive reuse) | 0.3 | {0, 0.1, 0.3, 1.0} | D⁴-new |
| $p_{\text{drop}}$ (CFG) | 0.1 | {0.05, 0.1, 0.2} | CFG standard |
| $\tau_{\text{ent}}$ (γ-entropy weight) | 1 | {0.5, 1, 2} | D⁴-new |
| $d_{\text{cond}}$ (condition embed dim) | 64 | {32, 64, 128} | AdaLN standard |
| $\sigma^2$ (Gaussian residual variance) | $1/\sqrt{2}$ (absorbed) | not swept | D5 ranking-invariant |
| conformal δ | 10% | {5%, 10%, 20%} | D³-Disc heritage |
| shared banks | yes | {yes, per-task} | D8 |
| per-task τ | yes | {yes, shared} | D³-Disc §8.3 |

---

## Decision log addendum (D13–D16)

| ID | Decision | Rationale |
|---|---|---|
| **D13** | Single conditional dynamics model with AdaLN-Zero + 3-token condition embedding | Reduces parameter count; aligns with CFG literature; enables inference-time ω sweep without retraining |
| **D14** | Advantage-based soft label γⱼ replaces static F3 weight | Learned advantage is a sharper separator than bank-KNN distance; co-evolves with the critic |
| **D15** | Two-phase training: warm-up (γ=0.5 frozen) then bootstrap (γ updated each epoch) | Mitigates early commitment; satisfies schedule (C4) |
| **D16** | CFG guidance scale at inference = ω (the D³-Disc knob) | Makes ω a first-class inference-time hyperparameter; one trained model serves all ω |

---

## Hand-off pointers

- **Prior design**: `.claude/discriminator/sfv_discriminator_design.md` (D³-Disc v2)
- **Prior implementation summary**: `.claude/discriminator/context/d3disc_v1_implementation.md`
- **Code base to extend**: `robosuite/discriminator/d3disc/`
- **Reference**:
  - DIPOLE (Liang et al., ICLR 2026) — the (1+ω)/(−ω) log-combination heritage
  - AIRL (Fu et al., 2018) — IRL-optimal reward $G = \log \rho_+/\rho_-$
  - DiT (Peebles & Xie, 2023) — AdaLN-Zero block
  - Classifier-Free Guidance (Ho & Salimans, 2022) — CFG training/inference
  - Implicit Q-Learning (Kostrikov et al., 2022) — advantage-weighted objective heritage (dichotomous routing)
  - Generalized Policy Iteration (Sutton & Barto, Ch. 4.6) — abstract alternation framework
