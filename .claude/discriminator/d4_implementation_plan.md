# D⁴-Disc Implementation Plan

> **Audience**: a fresh coding agent that will implement D⁴-Disc **in-place** on the current branch.
>
> **Reference spec**: `.claude/discriminator/d4_disc_design.md` (read the full design first; decisions D13–D16 locked there).
>
> **Prior art to reuse**:
> - `.claude/discriminator/sfv_discriminator_design.md` (D³-Disc design)
> - `.claude/discriminator/context/d3disc_v1_implementation.md` (D³-Disc v1 implementation summary)
> - `.claude/discriminator/implementation_plan.md` (D³-Disc implementation plan — architectural reference)
> - `robosuite/discriminator/d3disc/` (frozen D³-Disc code; **do not modify**)
> - `robosuite/discriminator/lpb/model.py` (`DynamicsPredictor` + `DecoderBlock` — will be subclassed/adapted)
>
> **User preferences**: follow `AGENTS.md` strictly (minimal modifications, English code comments, concise, WandB offline, `daggar` conda env).
>
> **⚠️ Training-stability warning**: this framework involves (i) a conditional network with label-dropout CFG training, (ii) a moving soft-label target γⱼ coupled to the critic, (iii) a schedule αₖ that must be matched to critic-convergence rate. Naively wired, the bootstrap phase can **collapse** (γ → constant, f⁺ ≡ f⁻, advantage → 0) or **diverge** (γ commits to wrong labels before critics separate). Every architectural and training decision below is justified against at least one of these failure modes — follow them as defaults, do not paraphrase away without understanding why.

---

## 1. High-level summary

Build a new discriminator `D4BenchmarkDiscriminator` **sibling** to `D3BenchmarkDiscriminator`, using:

1. **Frozen flow_multi encoder** — reuse `FlowMultiEncoderWrapper` from `d3disc/encoder.py` verbatim (no modifications, no subclassing).
2. **Single conditional latent dynamics predictor** `f_θ(φ_{t+h} | φ_t, s_t, a_{t:t+h}, c)` with `c ∈ {+, −, ∅}`, built on top of `lpb/model.py::DynamicsPredictor` with **AdaLN-Zero blocks replacing the plain LayerNorm** inside each transformer block and a 3-token condition embedding.
3. **Two-phase training**:
   - **Phase A (warm-up)**: γⱼ ≡ 0.5 frozen; train only on (𝒟₊ with c=+) and unconditional branch (c=∅). Fail data **never seen** under c=−.
   - **Phase B (bootstrap)**: γⱼ updated per outer epoch via the advantage gate; fail data flows into c=+ with weight (1-γⱼ) and c=− with weight γⱼ.
4. **Advantage gate** as soft-label update: `γⱼ = σ(αₖ · [−A_θ(φⱼ, aⱼ, φⱼ₊ₕ) − κₖ])`, with `A_θ = Q⁺ − Q⁻` computed from Gaussian-residual surrogates (see §4.2 of design).
5. **CFG-guided scoring** at inference: `λₜ = ‖φₜ₊ₕ − f_ω(·)‖² / (2σ²)` with `f_ω = (1+ω)f_θ(·|c=+) − ω f_θ(·|c=−)`. **One trained model serves all ω** — no retraining for ω sweep.
6. **Per-task conformal τ** on held-out success, unchanged from D³-Disc §8.

**Non-goals (deferred)**: K-step rollout (phase 2); task-gated AdaLN for per-task transfer; alternative condition embeddings beyond 3-token.

---

## 2. File / module layout

Create a new subpackage **parallel to** `d3disc/`:

```
robosuite/discriminator/d4disc/
├── __init__.py
├── README.md
├── adaln.py              # AdaLN-Zero block + condition embedder
├── model.py              # ConditionalDynamicsPredictor (stability-aware wrapper)
├── ema.py                # ModelEMA utility (param exponential moving average)
├── schedule.py           # alpha_k / kappa_k / eta_k schedulers
├── dataset.py            # LatentFlowDynamicsDatasetD4 (adds sample kind + gamma slot)
├── filter.py             # compute_advantage_gate(predictor, dataset, alpha, kappa)
├── monitor.py            # collapse/divergence detectors + wandb logging helpers
├── trainer.py            # D4Trainer (phase A/B state machine)
├── train_d4.py           # CLI entry for training
├── detector.py           # D4Detector (CFG-based scoring + conformal tau)
├── d4_benchmark.py       # D4BenchmarkDiscriminator (benchmark API)
├── dynamics_feature.py   # D4FeatureExtractor (inference-time CFG forward)
├── visualize.py          # adapted from d3disc/visualize.py
└── scripts/
    ├── train_d4.sh
    ├── run_d4_benchmark.sh
    ├── sweep_omega_d4.sh
    ├── ablate_schedule.sh
    └── visualize_d4.sh

data/utils/benchmark/examples/
└── run_d4.py             # benchmark CLI driver (parallel to run_d3.py)
```

**Untouched**:
- `robosuite/discriminator/lpb/` (LPB baseline)
- `robosuite/discriminator/d3disc/` (D³-Disc; D⁴ imports from here but does not modify)

---

## 3. Dependencies and imports

- `torch`, `numpy`, `h5py` (all present)
- `robosuite.discriminator.d3disc.encoder.FlowMultiEncoderWrapper` — **frozen encoder, reuse verbatim**
- `robosuite.discriminator.d3disc.dataset.LatentFlowDynamicsDataset` — extend, not fork
- `robosuite.discriminator.lpb.model.DecoderBlock` — replaced by `AdaLNDecoderBlock` (see §4.1)
- `data.utils.benchmark.{BenchmarkTrajectory, DiscriminatorOutput, FailureBenchmark}`

**Do not** add new package dependencies.

---

## 4. Component specs

### 4.1 `adaln.py` — AdaLN-Zero building block

The single most important stability primitive. Implements DiT-style AdaLN-Zero modulation.

```python
class ConditionEmbedder(nn.Module):
    """3-token embedding: c ∈ {+ (0), − (1), ∅ (2)} → R^d_cond."""
    COND_PLUS, COND_MINUS, COND_NULL = 0, 1, 2

    def __init__(self, d_cond: int = 64) -> None:
        super().__init__()
        self.embed = nn.Embedding(3, d_cond)
        nn.init.normal_(self.embed.weight, std=0.02)  # small init — stability
        self.act = nn.SiLU()

    def forward(self, cond_idx: torch.Tensor) -> torch.Tensor:
        # cond_idx: (B,) long tensor with values in {0,1,2}
        return self.act(self.embed(cond_idx))          # (B, d_cond)


class AdaLNModulation(nn.Module):
    """Produces (scale, shift, gate) for one sub-block. Zero-init the last linear."""
    def __init__(self, d_cond: int, d_model: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_cond, 3 * d_model)
        nn.init.zeros_(self.linear.weight)             # zero-init — CRITICAL
        nn.init.zeros_(self.linear.bias)

    def forward(self, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # cond: (B, d_cond) → three (B, d_model) tensors
        shift, scale, gate = self.linear(cond).chunk(3, dim=-1)
        return shift, scale, gate


class AdaLNDecoderBlock(nn.Module):
    """
    Drop-in replacement for lpb.model.DecoderBlock with AdaLN-Zero modulation.
    Two sub-blocks (MSA + MLP) each get their own (shift, scale, gate).
    Zero-init of gate == identity at step 0; conditioning is learned smoothly.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        d_cond: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model, elementwise_affine=False)   # affine comes from AdaLN
        self.ln_2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True,
        )
        hidden_dim = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )
        self.mod_attn = AdaLNModulation(d_cond, d_model)
        self.mod_mlp  = AdaLNModulation(d_cond, d_model)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, attn_mask) -> torch.Tensor:
        # x: (B, L, d_model)   cond: (B, d_cond)
        shift_a, scale_a, gate_a = self.mod_attn(cond)
        shift_m, scale_m, gate_m = self.mod_mlp(cond)

        h = self.ln_1(x) * (1 + scale_a.unsqueeze(1)) + shift_a.unsqueeze(1)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + gate_a.unsqueeze(1) * h

        h = self.ln_2(x) * (1 + scale_m.unsqueeze(1)) + shift_m.unsqueeze(1)
        h = self.mlp(h)
        x = x + gate_m.unsqueeze(1) * h
        return x
```

**Stability notes (must-keep)**:
- `elementwise_affine=False` on the underlying LN: the affine part is replaced by AdaLN.
- Zero-init of `mod_attn.linear` and `mod_mlp.linear` → all gate values start at 0 → block is identity at step 0 → **initial model is literally the unconditional predictor regardless of `c`**. This is what makes training stable in phase A.
- `ConditionEmbedder` init std=0.02 (not the default 1.0): small embedding magnitudes prevent large gradients once the modulation heads start learning.
- Use `nn.SiLU()` on the embedding, standard DiT choice.

### 4.2 `model.py` — `ConditionalDynamicsPredictor`

Wraps `lpb.model.DynamicsPredictor` with conditioning; minimal changes to the backbone.

```python
class ConditionalDynamicsPredictor(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        proprio_dim: int,
        action_dim: int,
        d_model: int = 512,
        num_layers: int = 6,
        nhead: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_action_horizon: int = 32,
        d_cond: int = 64,
    ) -> None:
        super().__init__()
        # Input projections + query tokens (copied from lpb.model.DynamicsPredictor)
        self.obs_proj     = nn.Linear(latent_dim, d_model)
        self.proprio_proj = nn.Linear(proprio_dim, d_model)
        self.action_proj  = nn.Linear(action_dim, d_model)
        self.pred_latent_token  = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pred_proprio_token = nn.Parameter(torch.zeros(1, 1, d_model))

        max_seq_len = 2 + max_action_horizon + 2
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        self.drop = nn.Dropout(dropout)

        # Conditioning
        self.cond_embed = ConditionEmbedder(d_cond)

        # AdaLN decoder stack
        self.blocks = nn.ModuleList([
            AdaLNDecoderBlock(d_model=d_model, nhead=nhead, d_cond=d_cond,
                              mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm_final = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mod_final  = AdaLNModulation(d_cond, d_model)   # final AdaLN before heads
        self.latent_head  = nn.Linear(d_model, latent_dim)
        self.proprio_head = nn.Linear(d_model, proprio_dim)

        self.latent_dim = latent_dim
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.max_action_horizon = max_action_horizon
        self.d_cond = d_cond

    def forward(self, obs_token, proprio_token, action_tokens, cond_idx) -> dict:
        # cond_idx: (B,) long, values in {0,1,2}
        B, H, _ = action_tokens.shape
        cond = self.cond_embed(cond_idx)                       # (B, d_cond)

        obs  = self.obs_proj(obs_token).unsqueeze(1)
        prop = self.proprio_proj(proprio_token).unsqueeze(1)
        act  = self.action_proj(action_tokens)
        pz = self.pred_latent_token.expand(B, -1, -1)
        ps = self.pred_proprio_token.expand(B, -1, -1)

        x = torch.cat([obs, prop, act, pz, ps], dim=1)
        L = x.size(1)
        x = x + self.pos_embedding[:, :L, :]
        x = self.drop(x)

        mask = torch.triu(torch.full((L, L), float("-inf"), device=x.device), diagonal=1)
        for blk in self.blocks:
            x = blk(x, cond=cond, attn_mask=mask)

        shift_f, scale_f, _ = self.mod_final(cond)
        x = self.norm_final(x) * (1 + scale_f.unsqueeze(1)) + shift_f.unsqueeze(1)

        return {
            "pred_latent":  self.latent_head(x[:, -2, :]),
            "pred_proprio": self.proprio_head(x[:, -1, :]),
        }

    @staticmethod
    def residual_sq(pred, target) -> torch.Tensor:
        # Per-sample squared residual. Latent only; proprio is auxiliary.
        return ((pred["pred_latent"] - target["target_latent"]) ** 2).sum(dim=-1)
```

**Stability notes**:
- Do **not** add a trainable "unconditional" bias separately — the c=∅ token embedding already serves this role via the standard CFG label-dropout mechanism.
- Final pre-head AdaLN is included (DiT-style) to give the heads a conditioning-aware feature.
- Proprio head is kept (auxiliary loss), but the advantage computation in §4.4 uses only the `pred_latent` residual — matches the Gaussian identity in design §2.1.

### 4.3 `ema.py` — parameter EMA

```python
class ModelEMA:
    """Exponential moving average of model params. Used for *inference-time* CFG."""
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        sd = model.state_dict()
        for k, v in sd.items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

    def apply_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}
```

**Why**: CFG inference on the **raw** training weights is known to be noisier than on EMA weights (standard diffusion practice). EMA also dampens the oscillation between Sweep I (critic) and Sweep II (γ) updates.

### 4.4 `schedule.py` — α/κ/η/LR schedulers

```python
@dataclass
class D4Schedule:
    alpha0: float                 # inverse temperature at start of phase B
    alpha_exponent: float = 0.75  # alpha_k = alpha0 * (1 + k)**rho, rho in (0.5, 1]
    kappa0: float = 0.0           # pinned to 0 in bootstrap (design §9.2)
    eta0: float = 0.0             # pseudo-positive reuse weight, phase A = 0
    eta_final: float = 0.3
    eta_ramp_epochs: int = 5      # linear ramp after K_warm
    alpha_cap_rate: float = 1.0   # alpha_k * held_out_r_plus < alpha_cap_rate (safety)

    def alpha(self, k_bootstrap: int) -> float: ...
    def eta(self, k_bootstrap: int) -> float: ...
    def clamped_alpha(self, alpha_raw: float, held_out_r_plus: float) -> float:
        # safety clamp per design (C4): alpha_k * rho_k bounded
        return min(alpha_raw, self.alpha_cap_rate / max(held_out_r_plus, 1e-6))
```

**Design rationale**: `alpha_cap_rate` implements the coupling `α_k · ρ_k = O(1)` from design (C4). Violating it is the single largest driver of early-commitment failure.

### 4.5 `dataset.py` — extend d3disc dataset

```python
class LatentFlowDynamicsDatasetD4(LatentFlowDynamicsDataset):
    """
    Extends d3disc dataset with:
    - per-sample is_fail_raw flag: True if sample came from fail_rollout (the
      contamination pool). Success / expert samples are always "clean positive".
    - a mutable gamma_buffer indexed by sample id, holding the current soft label.
      Updated in Sweep II; read in Sweep I to construct per-sample weights.
    """

    def __init__(self, *, fail_rollout_paths: Sequence[str] = (), **kwargs) -> None:
        super().__init__(**kwargs)      # loads expert + success_rollout as before
        # Scan fail_rollout separately; tag as contaminated.
        ...
        self.gamma_buffer = torch.full((len(self),), 0.5, dtype=torch.float32)
        self._is_fail_raw = torch.tensor([...], dtype=torch.bool)  # (N,)

    def update_gamma(self, indices: torch.Tensor, new_gamma: torch.Tensor,
                     ema_alpha: float = 0.5) -> None:
        """EMA smoothing to prevent γ oscillation. ema_alpha ~= 0.5 is a safe default."""
        idx = indices.cpu()
        old = self.gamma_buffer[idx]
        self.gamma_buffer[idx] = ema_alpha * old + (1 - ema_alpha) * new_gamma.cpu().clamp(1e-3, 1 - 1e-3)

    def __getitem__(self, i: int) -> dict:
        item = super().__getitem__(i)
        item["sample_idx"]   = torch.tensor(i, dtype=torch.long)
        item["is_fail_raw"]  = self._is_fail_raw[i]
        item["gamma"]        = self.gamma_buffer[i]
        return item
```

**Stability notes**:
- γ is **EMA-smoothed** per update (`ema_alpha = 0.5` default). Hard γ replacement → high-frequency gradient oscillation. EMA damps that.
- γ is **clamped** to `[1e-3, 1 - 1e-3]` to keep the routing gradient non-vanishing (pure 0 or 1 → (1-γ) or γ factor = 0 → no gradient to that branch).
- Gamma buffer lives on CPU (one float per sample); updates are cheap.

### 4.6 `filter.py` — advantage gate

```python
@torch.no_grad()
def compute_advantage_gate(
    predictor: ConditionalDynamicsPredictor,
    dataset: LatentFlowDynamicsDatasetD4,
    alpha_k: float,
    kappa_k: float = 0.0,
    sigma_sq: float = 0.5,
    device: str = "cuda",
    batch_size: int = 256,
    ema_alpha: float = 0.5,
) -> dict:
    """
    Recompute gamma for all is_fail_raw samples via advantage gate (design eq. flat).
    Returns diagnostics: mean_gamma, separation_gap, clamp_rate, ...
    """
    predictor.eval()
    fail_indices = dataset._is_fail_raw.nonzero().squeeze(-1)
    loader = make_subset_loader(dataset, fail_indices, batch_size=batch_size)

    new_gammas = torch.empty(len(fail_indices), dtype=torch.float32)
    r_plus_all, r_minus_all = [], []

    for batch in loader:
        # Forward with c=+ and c=-
        c_plus  = torch.full((len(batch),), ConditionEmbedder.COND_PLUS,  device=device)
        c_minus = torch.full((len(batch),), ConditionEmbedder.COND_MINUS, device=device)
        out_plus  = predictor(..., cond_idx=c_plus)
        out_minus = predictor(..., cond_idx=c_minus)

        r_plus  = ((out_plus["pred_latent"]  - batch["target_latent"])**2).sum(-1)
        r_minus = ((out_minus["pred_latent"] - batch["target_latent"])**2).sum(-1)
        A = (r_minus - r_plus) / (2 * sigma_sq)             # advantage
        gamma_new = torch.sigmoid(alpha_k * (-A - kappa_k)) # eq. (♭)
        new_gammas[batch["position_in_subset"]] = gamma_new.cpu()
        r_plus_all.append(r_plus.cpu())
        r_minus_all.append(r_minus.cpu())

    dataset.update_gamma(fail_indices, new_gammas, ema_alpha=ema_alpha)

    return {
        "mean_gamma":     float(dataset.gamma_buffer[fail_indices].mean()),
        "mean_r_plus":    float(torch.cat(r_plus_all).mean()),
        "mean_r_minus":   float(torch.cat(r_minus_all).mean()),
        "separation_gap": float((torch.cat(r_minus_all) - torch.cat(r_plus_all)).mean()),
        "alpha_used":     alpha_k,
    }
```

### 4.7 `monitor.py` — collapse / divergence detectors

```python
@dataclass
class D4Health:
    held_out_r_plus: float          # residual of f_+ on validation success frames
    mean_gamma: float               # mean over fail_raw samples
    separation_gap: float           # mean(r_minus - r_plus), should stay > 0
    conditional_delta: float        # ||f(c=+) - f(c=-)||² on a fixed probe batch

class CollapseDetector:
    """
    Triggers a training halt / fallback if any of the following holds for
    `patience` consecutive checks:
      (a) mean_gamma drifts to {<= 0.02, >= 0.98} (early commitment)
      (b) conditional_delta < 1e-4 (network ignores condition; CFG would return noise)
      (c) held_out_r_plus monotonically increasing over > 3 checks (critic diverging)
      (d) separation_gap < 0 for > 2 checks (f_- is a better success predictor than f_+ !)
    """
    def __init__(self, patience: int = 3): ...
    def update(self, health: D4Health) -> Optional[str]: ...  # reason string if tripped
```

`trainer.py` checks this every K_health_steps steps; on trip, reverts to the last EMA checkpoint and halves `alpha0`.

### 4.8 `trainer.py` — `D4Trainer`

State machine with two phases.

```python
@dataclass
class D4TrainerConfig:
    # Phase structure
    warm_up_epochs: int = 8
    bootstrap_epochs: int = 40

    # Optimizer
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    batch_size: int = 256
    num_workers: int = 4

    # CFG
    p_cond_drop: float = 0.1             # replace c with COND_NULL
    min_batch_balance: float = 0.15      # each of {+, -, ∅} must be >= 15% of batch

    # Schedule
    schedule: D4Schedule = field(default_factory=D4Schedule)

    # Stability
    ema_decay_model: float = 0.999
    ema_alpha_gamma: float = 0.5
    gamma_clamp: float = 1e-3
    held_out_ratio: float = 0.05
    health_check_every: int = 500         # steps

    # Loss weights
    proprio_loss_weight: float = 0.1
    latent_loss_weight:  float = 1.0


class D4Trainer:
    def __init__(self, predictor: ConditionalDynamicsPredictor,
                 dataset: LatentFlowDynamicsDatasetD4, config: D4TrainerConfig,
                 device: str = "cuda"): ...

    def _sample_condition(self, is_fail_raw: torch.Tensor, gamma: torch.Tensor,
                          phase: str) -> torch.Tensor:
        """
        Returns cond_idx per sample. Randomized per-batch.
        - clean positives (is_fail_raw=False): c=+ (w.p. 1 - p_drop) else ∅
        - fail_raw under phase A (warm-up): routed to c=+ w.p. 1 - p_drop else ∅ (NEVER c=-)
        - fail_raw under phase B (bootstrap): c=+ w.p. (1-gamma)(1-p_drop);
                                              c=- w.p. gamma(1-p_drop);
                                              ∅ w.p. p_drop
        """
        ...

    def _loss_for_batch(self, batch: dict, phase: str) -> dict:
        cond_idx = self._sample_condition(batch["is_fail_raw"], batch["gamma"], phase)
        out = self.predictor(
            obs_token=batch["current_latent"],
            proprio_token=batch["current_proprio"],
            action_tokens=batch["action_sequence"],
            cond_idx=cond_idx,
        )
        latent_mse = F.mse_loss(out["pred_latent"], batch["target_latent"])
        proprio_mse = F.mse_loss(out["pred_proprio"], batch["target_proprio"])
        total = (self.cfg.latent_loss_weight * latent_mse +
                 self.cfg.proprio_loss_weight * proprio_mse)
        return {"loss": total, "latent_mse": latent_mse, "proprio_mse": proprio_mse,
                "cond_distribution": _count_per_cond(cond_idx)}

    def run(self) -> dict:
        # Phase A
        for epoch in range(self.cfg.warm_up_epochs):
            self._train_one_epoch(phase="A")
            self._validate_and_log(epoch, phase="A")

        # Snapshot alpha_0 from end-of-phase-A residual MAD
        held_out_r_plus = self._eval_held_out_r_plus()
        self.cfg.schedule.alpha0 = 1.0 / max(mad(self._train_r_plus_distribution()), 1e-6)

        # Phase B
        for k in range(self.cfg.bootstrap_epochs):
            alpha_k_raw = self.cfg.schedule.alpha(k)
            alpha_k = self.cfg.schedule.clamped_alpha(alpha_k_raw, held_out_r_plus)
            gate_diag = compute_advantage_gate(self.ema_model_or_training_model,
                                               self.dataset, alpha_k=alpha_k,
                                               ema_alpha=self.cfg.ema_alpha_gamma)
            self._train_one_epoch(phase="B")
            held_out_r_plus = self._eval_held_out_r_plus()
            health = self._health_snapshot(gate_diag, held_out_r_plus)
            reason = self.collapse_detector.update(health)
            if reason is not None:
                self._rollback_and_halve_alpha(reason)
            self._validate_and_log(k + self.cfg.warm_up_epochs, phase="B", extra=gate_diag)
        ...
```

**Stability notes**:
- `_sample_condition`: in phase A, fail_raw samples **never** get c=−. The c=− branch of the network therefore has no signal at all during phase A, but AdaLN-Zero keeps it as identity, so this is safe.
- `min_batch_balance`: if any condition is under-represented in a batch (e.g., early phase B, γ is still ~0.5 → c=− rare), force-resample. Under-trained branches are a pathway to collapse.
- γ is queried from `batch["gamma"]` which was loaded via dataset — the latest `update_gamma` call from compute_advantage_gate writes through; next epoch's batches see the new γ.
- `ema_model_or_training_model`: use EMA weights for the gate computation. Non-EMA γ updates are noisier.

### 4.9 `detector.py` — `D4Detector`

```python
class D4Detector:
    """
    Inference-time scoring via CFG-guided latent prediction.
    Holds: (i) the trained predictor (EMA weights), (ii) per-task tau.
    """
    def __init__(self, predictor: ConditionalDynamicsPredictor, omega: float = 0.5,
                 sigma_sq: float = 0.5, delta: float = 10.0,
                 lambda_mode: str = "mean", lambda_window_size: int = -1,
                 device: str = "cuda"): ...

    @torch.no_grad()
    def score_frames(self, batch: dict) -> torch.Tensor:
        c_plus  = torch.full((B,), ConditionEmbedder.COND_PLUS,  device=self.device)
        c_minus = torch.full((B,), ConditionEmbedder.COND_MINUS, device=self.device)
        out_p = self.predictor(..., cond_idx=c_plus)
        out_m = self.predictor(..., cond_idx=c_minus)
        f_omega_latent = (1 + self.omega) * out_p["pred_latent"] - self.omega * out_m["pred_latent"]
        lam = ((f_omega_latent - batch["target_latent"]) ** 2).sum(-1) / (2 * self.sigma_sq)
        return lam

    def fit_tau(self, calib_feature_seqs: list[torch.Tensor]) -> float: ...

    def score(self, features: torch.Tensor, ...) -> tuple[np.ndarray, ...]: ...
```

### 4.10 `d4_benchmark.py` — `D4BenchmarkDiscriminator`

Mirror `D3BenchmarkDiscriminator` API exactly:

```python
class D4BenchmarkDiscriminator:
    name = "d4_disc"

    def __init__(self, *, policy_ckpt_path, d4_ckpt_path, omega=0.5, delta=10.0,
                 calib_fraction=0.2, per_task_calibration=True, seed=0,
                 device="cuda", ...):
        self.encoder   = FlowMultiEncoderWrapper(policy_ckpt_path, ...)
        payload        = torch.load(d4_ckpt_path, map_location="cpu")
        self.predictor = ConditionalDynamicsPredictor(**payload["arch_args"])
        self.predictor.load_state_dict(payload["ema_state_dict"])
        self.predictor.to(device).eval()
        self.detector  = D4Detector(self.predictor, omega=omega, ...)

    def fit_on_benchmark(self, trajectories): ...   # reuses d3disc data partitioning
    def score_trajectory(self, traj):     ...
    def calibration_summary(self) -> dict: ...
    def close(self) -> None: ...
```

**Key**: the benchmark discriminator loads the **EMA state dict**, not the raw training weights. Argument `omega` is read at eval time — so one trained checkpoint serves all ω for the sweep.

---

## 5. CLI entry points

### 5.1 `scripts/train_d4.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/daggar/bin/python}"

POLICY_CKPT="${POLICY_CKPT:-checkpoints/multitask_6/policy/flow-20/flow_multi_ep0100_20260320_114720.pt}"
TASKS="${TASKS:-PandaLift PandaPickPlaceCan PandaStack PickPlaceBread PickPlaceCereal PickPlaceMilk}"

# Build expert + success_rollout + fail_rollout path lists from TASKS
EXPERT_PATHS=()
ROLLOUT_PATHS=()
FAIL_PATHS=()
for t in ${TASKS}; do
  [[ -d "data/${t}/expert" ]]           && EXPERT_PATHS+=("data/${t}/expert")
  [[ -d "data/${t}/success_rollout" ]]  && ROLLOUT_PATHS+=("data/${t}/success_rollout")
  [[ -d "data/${t}/fail_rollout" ]]     && FAIL_PATHS+=("data/${t}/fail_rollout")
done

RUN_NAME="${RUN_NAME:-d4dyn_$(date +%Y%m%d_%H%M%S)}"
SAVE_DIR="checkpoints/d4disc/dynamics/${RUN_NAME}"
mkdir -p "${SAVE_DIR}"

WARM_UP_EPOCHS="${WARM_UP_EPOCHS:-8}"
BOOTSTRAP_EPOCHS="${BOOTSTRAP_EPOCHS:-40}"
D_MODEL="${D_MODEL:-512}"
D_COND="${D_COND:-64}"
NUM_LAYERS="${NUM_LAYERS:-6}"
NUM_HEADS="${NUM_HEADS:-8}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-3e-4}"
P_COND_DROP="${P_COND_DROP:-0.1}"
ALPHA_EXPONENT="${ALPHA_EXPONENT:-0.75}"
ALPHA_CAP_RATE="${ALPHA_CAP_RATE:-1.0}"
EMA_DECAY="${EMA_DECAY:-0.999}"
EMA_ALPHA_GAMMA="${EMA_ALPHA_GAMMA:-0.5}"
SEED="${SEED:-0}"

WANDB_MODE="${WANDB_MODE:-offline}" WANDB_NAME="${WANDB_NAME:-songgao-personal}" \
"${PYTHON_BIN}" -m robosuite.discriminator.d4disc.train_d4 \
    --policy-ckpt     "${POLICY_CKPT}" \
    --cache-root      data/.lpb_score_cache \
    --expert-paths    "${EXPERT_PATHS[@]}" \
    --rollout-paths   "${ROLLOUT_PATHS[@]}" \
    --fail-paths      "${FAIL_PATHS[@]}" \
    --warm-up-epochs  "${WARM_UP_EPOCHS}" \
    --bootstrap-epochs "${BOOTSTRAP_EPOCHS}" \
    --d-model         "${D_MODEL}" \
    --d-cond          "${D_COND}" \
    --num-layers      "${NUM_LAYERS}" \
    --num-heads       "${NUM_HEADS}" \
    --batch-size      "${BATCH_SIZE}" \
    --lr              "${LR}" \
    --p-cond-drop     "${P_COND_DROP}" \
    --alpha-exponent  "${ALPHA_EXPONENT}" \
    --alpha-cap-rate  "${ALPHA_CAP_RATE}" \
    --ema-decay       "${EMA_DECAY}" \
    --ema-alpha-gamma "${EMA_ALPHA_GAMMA}" \
    --seed            "${SEED}" \
    --save-dir        "${SAVE_DIR}"
```

### 5.2 `scripts/run_d4_benchmark.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
# ... standard boilerplate ...
D4_CKPT="${D4_CKPT:?must set D4_CKPT}"    # path to d4_dynamics.pt
POLICY_CKPT="${POLICY_CKPT:-...}"

OMEGA="${OMEGA:-0.5}"
DELTA="${DELTA:-10.0}"
SUCC_NUM="${SUCC_NUM:-200}"
FAIL_NUM="${FAIL_NUM:-100}"
TASKS="${TASKS:-PandaLift PandaPickPlaceCan PandaStack PickPlaceBread PickPlaceCereal PickPlaceMilk}"

"${PYTHON_BIN}" -m data.utils.benchmark.examples.run_d4 \
    --policy-ckpt     "${POLICY_CKPT}" \
    --d4-ckpt         "${D4_CKPT}" \
    --fail-root       data/utils/fail_rollout \
    --success-root    data/utils/success_rollout \
    --tasks           ${TASKS} \
    --omega           "${OMEGA}" \
    --delta           "${DELTA}" \
    --max-success-per-task "${SUCC_NUM}" \
    --max-fail-per-task    "${FAIL_NUM}" \
    --save-json       "checkpoints/d4disc/eval/run_$(date +%Y%m%d_%H%M%S)/benchmark.json"
```

### 5.3 `scripts/sweep_omega_d4.sh`

```bash
#!/usr/bin/env bash
# ω-sweep: one checkpoint, multiple ω values at inference, NO retraining.
D4_CKPT="${D4_CKPT:?}"
for OMEGA in 0 0.1 0.2 0.5 1 2; do
  OMEGA="${OMEGA}" D4_CKPT="${D4_CKPT}" \
    RUN_TAG="w${OMEGA}" \
    bash robosuite/discriminator/d4disc/scripts/run_d4_benchmark.sh
done
```

### 5.4 `scripts/ablate_schedule.sh`

Sweep schedule hyperparams (α_exponent, ALPHA_CAP_RATE, EMA_ALPHA_GAMMA, p_cond_drop) — these **require** retraining; keep grid small (3 × 3 × 2 × 2 = 36 runs max; in practice pick a smaller set).

### 5.5 `scripts/visualize_d4.sh`

Mirror `d3disc/scripts/visualize_d3.sh`; additionally render per-frame (r⁺, r⁻, γ, A) traces for fail trajectories.

---

## 6. Concrete task breakdown

### Phase A — Plumbing
1. Create subpackage skeleton `robosuite/discriminator/d4disc/` with `__init__.py` exposing `ConditionalDynamicsPredictor`, `D4Trainer`, `D4TrainerConfig`, `D4BenchmarkDiscriminator`, `D4Detector`.
2. Implement `adaln.py`. **Unit test**: with zero-init modulation heads, for any `cond_idx ∈ {0,1,2}`, the block output equals input (identity). Assert `max abs diff < 1e-8` on random input.
3. Implement `model.py::ConditionalDynamicsPredictor`. **Unit test**: identical outputs across `c=+, c=-, c=∅` at init (zero-init AdaLN → unconditional).
4. Implement `ema.py` and `schedule.py`. Short unit tests for schedule arithmetic.

### Phase B — Dataset & filter
5. Extend `d3disc/dataset.py::LatentFlowDynamicsDataset` → `d4disc/dataset.py::LatentFlowDynamicsDatasetD4`:
   - Scan `fail_rollout/*.hdf5` (new kwarg `fail_rollout_paths`).
   - Per-sample `is_fail_raw` tensor.
   - γ buffer with EMA `update_gamma`.
6. Implement `filter.py::compute_advantage_gate`. **Unit test**: synthetic case with two Gaussian residual distributions (r⁺ ~ 𝒩(0, 0.1), r⁻ ~ 𝒩(1, 0.1)) → γ should concentrate near 0 for "on-support" samples (r⁻ > r⁺) and near 1 otherwise; sweep α and check `γ` sharpness scales correctly.

### Phase C — Trainer
7. Implement `D4Trainer` in `trainer.py`:
   - `_sample_condition` with the per-phase routing table (§4.8).
   - `_train_one_epoch(phase)`; gradient clip; AdamW.
   - `_eval_held_out_r_plus` over a reserved 5% success-slice.
   - `_health_snapshot` + `CollapseDetector` integration.
   - EMA step per batch.
   - wandb offline logging: `wandb.init(project='d4disc', mode='offline', name=RUN_NAME)`.
     Scalars: `loss`, `latent_mse`, `held_out_r_plus`, `mean_gamma`, `separation_gap`, `conditional_delta`, `alpha_used`, `eta_used`, `cond_dist/{+,-,∅}`.
8. Implement `train_d4.py` CLI that builds encoder / dataset / predictor / trainer and launches.
9. **Smoke test** (single task, 10 traj/kind, 2 warm-up + 3 bootstrap epochs):
   - Loss decreases in phase A.
   - `mean_gamma` transitions from 0.5 → somewhere in (0.1, 0.9) in phase B.
   - `conditional_delta` > 1e-4 by end (AdaLN has learned to respond to `c`).
   - No collapse trigger.

### Phase D — Detector & benchmark
10. Implement `detector.py::D4Detector`.
11. Implement `d4_benchmark.py::D4BenchmarkDiscriminator`. Mirror `D3BenchmarkDiscriminator.fit_on_benchmark` data partitioning (share_banks flag inherited; reuse the split / calib logic from d3disc).
12. Implement `data/utils/benchmark/examples/run_d4.py`.

### Phase E — Sanity & ablation
13. **Sanity 1 (ω=0 baseline)**: train D⁴-Disc, score with ω=0. Expect: comparable to D³-Disc ω=0 (learned critic vs KNN; no identity, but same ballpark).
14. **Sanity 2 (phase-A-only ablation)**: skip phase B entirely → model behaves like a success-conditional dynamics trained on D+. Expected: monotonically worse than D³-Disc ω=0.5, because the model loses the advantage signal.
15. **ω sweep** from one checkpoint: {0, 0.1, 0.2, 0.5, 1, 2}. Report per-task + overall AUROC / AUPRC / F1 / first-failure-delay.
16. **Schedule ablation** (select subset, not full grid): `(α_exponent ∈ {0.5, 0.75, 1.0}) × (ALPHA_CAP_RATE ∈ {0.5, 1.0, 2.0})`. Use wandb sweeps.
17. **Visualization**: 4 fail trajectories per task; plot (r⁺, r⁻, γ, A, λ_ω) across frames.

### Phase F — Docs
18. Populate `robosuite/discriminator/d4disc/README.md`: how to train, how to run benchmark, how to sweep ω.
19. Create `.claude/discriminator/context/d4disc_v1_implementation.md` summarizing what was actually built (mirror `d3disc_v1_implementation.md` format).

---

## 7. Training-stability playbook

This is a concentrated reference for the most common failure modes. Each has a symptom, a diagnostic, and a default fix. Consult this **before** re-running a failed training run.

| Failure | Symptom (wandb scalar) | Diagnostic check | Default fix |
|---|---|---|---|
| **Early commitment** | `mean_gamma` snaps to 0 or 1 within first 1–2 bootstrap epochs | `alpha_used / held_out_r_plus` >> 1 | Halve `ALPHA_CAP_RATE`; extend warm-up by 4 epochs |
| **Gate oscillation** | `mean_gamma` oscillates between epochs with amplitude > 0.2 | `|γ^(k) - γ^(k-1)|` diagnostic high | Reduce `EMA_ALPHA_GAMMA` to 0.3 (more smoothing) |
| **Conditional collapse** | `conditional_delta` < 1e-4 after several epochs | inspect AdaLN gate magnitudes | Increase `p_cond_drop` to 0.15 (more unconditional training); verify `min_batch_balance` is enforced |
| **Critic divergence** | `held_out_r_plus` increasing over 3+ checks | inspect gradient norm pre/post clip | Halve LR; increase `GRAD_CLIP_NORM` constraint to 0.5 |
| **Branch starvation** | c=− condition count < 10% of batch for many epochs | `cond_dist/-` on wandb | In phase B, pin minimum γ-sum floor; allow hard sampling of `c=-` with prob ≥ 0.15 regardless of γ |
| **NaNs in CFG score** | `lambda` is NaN at inference | ω > 2 in `f_omega = (1+ω)f⁺ - ωf⁻` | Cap ω ≤ 2 at inference; clamp `pred_latent` norm |
| **Fail-data vacuum** | `mean_gamma` stays at 0.5 for entire phase B | compare `r⁺` vs `r⁻` on fail samples | Either (a) dataset has no informative fails (e.g., PandaLift-only) — fallback to ω=0; (b) warm-up too short — f⁺ has not yet separated |

### 7.1 Mandatory invariants (enforced by `monitor.py`)

- `gamma_buffer.min()` ≥ 1e-3 and `gamma_buffer.max()` ≤ 1 − 1e-3 — prevents dead branches.
- `conditional_delta > 1e-4` sustained over 5 consecutive health checks once phase B begins.
- `separation_gap ≥ 0` by end of phase B (fail samples should on average be harder for `f⁺` than for `f⁻`).

Violations log a wandb alert; repeated violations trigger rollback.

### 7.2 Initialization order (must follow)

1. Build `ConditionalDynamicsPredictor` with zero-init AdaLN heads — sanity check identity at `c ∈ {+, -, ∅}`.
2. Initialize `ModelEMA(model, decay=0.999)` **after** the predictor is on device.
3. Freeze `FlowMultiEncoderWrapper` — it has no trainable parameters, but explicitly call `encoder.model.eval()`.
4. Load dataset with `gamma_buffer = 0.5`. **Do not** warm-start γ from any prior checkpoint — stale γ can collide with a freshly initialized critic.

### 7.3 What goes into the saved checkpoint

```python
payload = {
    "raw_state_dict": model.state_dict(),
    "ema_state_dict": ema.shadow,
    "arch_args": {d_model, num_layers, nhead, d_cond, ...},
    "config":    asdict(config),
    "gamma_buffer": dataset.gamma_buffer,     # for diagnostic / resumption
    "is_fail_raw":  dataset._is_fail_raw,
    "sample_refs":  [(file_path, demo_key, t) ... ],
    "schedule_state": {alpha_k, kappa_k, eta_k, step, epoch},
    "health_history": [D4Health... ],
    "policy_ckpt_path": str(policy_ckpt),
}
```

Loading for benchmark: always the **EMA** state dict. Loading for resume: the **raw** state dict plus `schedule_state` plus `gamma_buffer`.

---

## 8. Things NOT to do

- Do **not** modify `robosuite/discriminator/d3disc/` or `robosuite/discriminator/lpb/`.
- Do **not** retrain or unfreeze the flow_multi encoder.
- Do **not** regenerate `data/.lpb_score_cache/`.
- Do **not** remove the warm-up phase or replace AdaLN-Zero with plain AdaLN (non-zero init). The zero-init is load-bearing for stability.
- Do **not** apply weight decay to the pos-embedding parameter or the condition embedding; they are small and WD destabilizes both.
- Do **not** remove the EMA model or use raw weights for either (a) γ-gate computation or (b) inference-time scoring. Both regress noticeably without EMA.
- Do **not** use ω > 2 at inference. The first-order-like CFG behavior breaks.
- Do **not** train with mixed precision (FP16) on the MSE residual term — the tail of the distribution drives advantage, and small differences are where the signal lives. BF16 is fine; FP32 is safest.
- Do **not** rebalance γ per batch with any hard rule. Use EMA + dataset-level update only.
- Do **not** implement task-gated AdaLN in this iteration — one global `{+, -, ∅}` triple suffices; task-shift ablation is deferred.
- Do **not** write documentation in `.md` files outside what §6 step 18/19 specifies. Project memory + design files are authoritative.

---

## 9. Deliverables checklist

- [ ] `robosuite/discriminator/d4disc/` package with all modules per §2.
- [ ] Unit tests for AdaLN zero-init identity (§6 step 2) and advantage-gate correctness (§6 step 6).
- [ ] Smoke training (10 traj/kind, 2 warm-up + 3 bootstrap) completes without collapse.
- [ ] Full training on all 6 tasks (≥ 8 warm-up + 40 bootstrap epochs) produces a valid checkpoint with `conditional_delta > 1e-3` and `separation_gap > 0`.
- [ ] Benchmark: one checkpoint, ω ∈ {0, 0.1, 0.2, 0.5, 1, 2} swept at inference. Per-task + overall AUROC/AUPRC/F1 recorded in JSON.
- [ ] Schedule ablation: at least `α_exponent ∈ {0.5, 0.75, 1.0} × ALPHA_CAP_RATE ∈ {0.5, 1.0, 2.0}` (9 runs) recorded.
- [ ] Visualization PDFs for 4 fail trajectories per task, both at ω=0.5 and ω=1.0.
- [ ] WandB offline logs under `WANDB_NAME=songgao-personal`.
- [ ] `robosuite/discriminator/d4disc/README.md`.
- [ ] `.claude/discriminator/context/d4disc_v1_implementation.md` (hand-off summary).

---

## 10. Open questions to raise during implementation (only if blocked)

Do not pause on these unless you hit a real ambiguity.

1. **Shared vs per-task banks** for calibration τ: inherit D³-Disc default (`share_banks=1`, per-task τ). Revisit only if a task shows systematic τ mis-calibration.
2. **Phase A data composition**: include `expert/` alongside `success_rollout/`? Default **yes** — matches D³-Disc v1 implementation summary §1.3.3, phase A in our framework is essentially a D³-Disc-style pretrained success-dynamics model. Revisit if ablation shows expert data hurts.
3. **γ initialization for resumed runs**: inherit `gamma_buffer` from checkpoint (payload §7.3). If `dataset._is_fail_raw` has grown (new fail data), initialize new entries to 0.5 only.
4. **Classifier-free guidance score form**: use the residual form `λ = ‖φ_{t+h} − f_ω‖²` (§4.9) rather than the log-density form `(1+ω)log p_+ - ω log p_-`. Both are rank-equivalent (design §A.3), but the residual form is numerically simpler (one reconstruction, one squared-diff).

---

## 11. Final check before merging

Run, from repo root with `daggar` env activated:

```bash
# 1. Unit tests
/home/dodo/miniconda3/envs/daggar/bin/python -m pytest robosuite/discriminator/d4disc/tests -v

# 2. Smoke training
SMOKE=1 WARM_UP_EPOCHS=2 BOOTSTRAP_EPOCHS=3 \
    bash robosuite/discriminator/d4disc/scripts/train_d4.sh
# expect: phase A latent_mse drops; phase B mean_gamma moves off 0.5;
#         conditional_delta > 1e-4 by end.

# 3. Full training
bash robosuite/discriminator/d4disc/scripts/train_d4.sh
# expect: separation_gap > 0, conditional_delta > 1e-3 at end.

# 4. ω sweep eval
D4_CKPT=checkpoints/d4disc/dynamics/<RUN_NAME>/d4_dynamics.pt \
    bash robosuite/discriminator/d4disc/scripts/sweep_omega_d4.sh
# expect: ω=0 results comparable to D³-Disc ω=0 baseline (same ballpark, not identical);
#         ω=0.5 results >= ω=0 on average;
#         ω=2 results may regress (CFG divergence region) — known caveat §9.3 design.

# 5. Visualization
D4_CKPT=... bash robosuite/discriminator/d4disc/scripts/visualize_d4.sh \
    --task PickPlaceBread --num-trajs 4
```

Report:
- Chinese summary per `AGENTS.md` transparency rule.
- Per-task metric table at ω ∈ {0, 0.5, 1} (primary comparison points).
- Health trace plots: `mean_gamma`, `separation_gap`, `conditional_delta` over training.
- Any collapse / rollback events encountered, with final schedule parameters used.

---

## Appendix — quick pseudocode reference

### A. Phase A training step

```python
batch = next(loader_A)               # is_fail_raw entries routed to {+, ∅} only
cond_idx = sample_condition(batch, phase="A")
out   = predictor(**batch, cond_idx=cond_idx)
loss  = mse(out["pred_latent"], batch["target_latent"]) + 0.1 * mse(out["pred_proprio"], batch["target_proprio"])
loss.backward();  clip_grad_norm_(1.0);  optimizer.step();  ema.update(predictor)
```

### B. Phase B outer loop

```python
for k in range(bootstrap_epochs):
    held_out_r_plus = eval_r_plus(predictor_ema, held_out_success_loader)
    alpha_k = schedule.clamped_alpha(schedule.alpha(k), held_out_r_plus)
    gate_diag = compute_advantage_gate(predictor_ema, dataset, alpha_k, ema_alpha=0.5)

    for batch in loader_B:
        cond_idx = sample_condition(batch, phase="B", gamma=batch["gamma"])
        out = predictor(**batch, cond_idx=cond_idx)
        loss = mse_latent + 0.1 * mse_proprio
        loss.backward();  clip_grad_norm_(1.0);  step();  ema.update(predictor)

    health = snapshot(gate_diag, held_out_r_plus, probe_conditional_delta())
    if collapse_detector.update(health):
        rollback_to_ema();  schedule.halve_alpha()
```

### C. CFG inference step

```python
out_p = predictor_ema(**batch, cond_idx=torch.full_like(..., COND_PLUS))
out_m = predictor_ema(**batch, cond_idx=torch.full_like(..., COND_MINUS))
f_omega_latent = (1 + omega) * out_p["pred_latent"] - omega * out_m["pred_latent"]
lambda_t = ((f_omega_latent - batch["target_latent"])**2).sum(-1) / (2 * sigma_sq)
```

---

## Decision addendum (implementation-level, not in design)

| ID | Decision | Reason |
|---|---|---|
| **I1** | `d_cond = 64`, `num_cond_tokens = 3` | Minimal capacity; CFG literature suggests 64–128 works for similar model sizes |
| **I2** | EMA decay = 0.999 for model, EMA α_γ = 0.5 for γ-buffer | Separate decays — model is quasi-continuous (per-batch), γ is discrete (per-epoch) |
| **I3** | Warm-up 8 epochs as default | Matches D³-Disc v1 reasonable minimum for `r⁺` floor |
| **I4** | Residual-form scoring (§10 Q4) | Numerical simplicity; rank-equivalent to log-density form |
| **I5** | `elementwise_affine=False` on LN | Required for AdaLN to have well-defined scale+shift on a normalized input |
| **I6** | No FP16; BF16 optional | Residual-magnitude tail carries advantage signal; FP16 under-resolves it |
| **I7** | Condition token count = 3 (not task-gated) | Defer per-task conditioning to a later iteration |
| **I8** | WD excluded from pos-embed and cond-embed | Small params; WD de-stabilizes them |
