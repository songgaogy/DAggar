"""D4Trainer: two-phase state machine for the conditional dynamics critic.

Phase A (warm-up, k < K_warm):
    gamma frozen at 0.5; condition routing is restricted so that fail_raw
    samples are only ever seen with c in {+, null}; the c=- branch receives
    zero signal and stays identity-like (AdaLN-Zero).

Phase B (bootstrap, k >= K_warm):
    At the start of each outer epoch,
        alpha_k_raw   = schedule.alpha(k - K_warm)
        alpha_k       = schedule.clamped_alpha(alpha_k_raw, held_out_r_plus)
        gamma         = compute_advantage_gate(ema_predictor, ...)
    Per batch the condition for each fail_raw sample is sampled as
        + w.p. (1 - gamma)(1 - p_drop)
        - w.p.  gamma     (1 - p_drop)
        null w.p. p_drop

Optimizer: AdamW with no weight decay on the position embedding, the
condition embedding table, biases, and normalization-layer affine params.

Precision: FP32 or BF16 only. FP16 under-resolves the residual tail that
carries advantage signal.

On every collapse trigger the trainer rolls back to the last EMA snapshot
and halves schedule.alpha0.
"""

from __future__ import annotations

import contextlib
import math
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .adaln import ConditionEmbedder
from .dataset import LatentFlowDynamicsDatasetD4
from .ema import ModelEMA
from .filter import (
    _collate,
    _gamma_quantile_diag,
    compute_advantage_gate,
    warm_start_gamma_from_knn,
)
from .model import ConditionalDynamicsPredictor
from .monitor import CollapseDetector, D4Health
from .schedule import D4Schedule


try:
    import wandb  # type: ignore
except ImportError:
    wandb = None  # type: ignore


_NO_WD_SUBSTRINGS = ("pos_embedding", "cond_embed.embed.weight")


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

    # CFG / routing
    p_cond_drop: float = 0.1
    min_batch_balance: float = 0.15

    # Schedule (bootstrap)
    schedule: D4Schedule = field(default_factory=D4Schedule)

    # Stability
    ema_decay_model: float = 0.999
    ema_alpha_gamma: float = 0.5
    gamma_clamp: float = 1e-3
    held_out_ratio: float = 0.05

    # Loss
    proprio_loss_weight: float = 0.1
    latent_loss_weight: float = 1.0
    sigma_sq: float = 0.5

    # Precision
    use_bfloat16: bool = False

    # Logging / saving
    log_every: int = 50
    save_freq: int = 0          # 0 => final only
    run_name: str = ""
    wandb_project: str = "d4disc"
    wandb_mode: str = "offline"
    disable_wandb: bool = False

    # Probe batch for conditional_delta.
    probe_batch_size: int = 64

    # F3-based gamma warm-start at end of Phase A (breaks symmetric fixed point).
    f3_warm_start: bool = True
    f3_warm_start_k: int = 1
    f3_warm_start_max_clean: int = 100_000
    # "rank" = quantile-normalize fail→clean d² so γ ~ U(0,1) regardless of
    # scale (robust to the saturation observed in run d4dyn_20260422_054915).
    # "sigmoid" = sigmoid auto-calibrated on the fail→clean d² distribution
    # itself (NOT the D3 clean-self calibration that caused the saturation).
    warm_start_mode: str = "rank"
    # Skip the gate update for the first ``gate_freeze_epochs`` bootstrap
    # epochs after warm-start, so the asymmetric warm-start gamma drives the
    # initial training and lets f(-) differentiate before the gate takes over.
    # Without this, the first gate call happens when f(+) ≈ f(-) still, so
    # it replaces the warm-start with a uniform sigma(0)=0.5 gate and the
    # system reverts to the symmetric fixed point.
    gate_freeze_epochs: int = 2
    # Skip Phase B entirely: train f(+) only in Phase A, warm-start gamma,
    # save ckpt. Use with OMEGA=0 at benchmark time to run the pure "learned
    # clean-dynamics residual" baseline (no f(-) refinement). Useful when
    # the bootstrap phase cannot extract additional signal beyond the
    # warm-start (Markov dynamics + informative encoder).
    skip_bootstrap: bool = False

    def __post_init__(self) -> None:
        if self.schedule is None:
            self.schedule = D4Schedule()


def _partition_params_for_wd(predictor: ConditionalDynamicsPredictor) -> Tuple[list, list]:
    wd_params: list = []
    no_wd_params: list = []
    for name, p in predictor.named_parameters():
        if not p.requires_grad:
            continue
        if any(s in name for s in _NO_WD_SUBSTRINGS):
            no_wd_params.append(p)
        elif name.endswith(".bias") or "ln_" in name or "norm_final" in name:
            no_wd_params.append(p)
        else:
            wd_params.append(p)
    return wd_params, no_wd_params


def _mad(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return 1.0
    med = values.median()
    mad = (values - med).abs().median()
    return float(mad.item())


class D4Trainer:
    def __init__(
        self,
        predictor: ConditionalDynamicsPredictor,
        dataset: LatentFlowDynamicsDatasetD4,
        *,
        config: Optional[D4TrainerConfig] = None,
        device: Optional[str] = None,
    ) -> None:
        self.predictor = predictor
        self.dataset = dataset
        self.cfg = config or D4TrainerConfig()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.predictor.to(self.device)

        wd_params, no_wd_params = _partition_params_for_wd(self.predictor)
        self.optimizer = torch.optim.AdamW(
            [
                {"params": wd_params, "weight_decay": float(self.cfg.weight_decay)},
                {"params": no_wd_params, "weight_decay": 0.0},
            ],
            lr=float(self.cfg.lr),
        )

        self.ema = ModelEMA(self.predictor, decay=float(self.cfg.ema_decay_model))

        self.collapse_detector = CollapseDetector()

        (
            self._train_indices,
            self._train_indices_clean_only,
            self._held_out_indices,
        ) = self._split_clean_positives(ratio=float(self.cfg.held_out_ratio))
        # Phase B loader (clean + fail). Phase A uses clean-only loader so
        # fail_raw never contaminates f(+).
        self._train_loader = self._build_loader(self._train_indices, shuffle=True)
        self._train_loader_phase_a = (
            self._build_loader(self._train_indices_clean_only, shuffle=True)
            if len(self._train_indices_clean_only) > 0
            else self._train_loader
        )
        self._held_out_loader = (
            self._build_loader(self._held_out_indices, shuffle=False)
            if len(self._held_out_indices) > 0
            else None
        )

        self._probe_batch: Optional[dict] = None

        self.history: Dict[str, Any] = {"phase_A": [], "phase_B": []}
        self.health_history: List[D4Health] = []

        self._wandb_run = None
        self._maybe_init_wandb()

    # ------------------------------------------------------------------ #
    # Setup                                                              #
    # ------------------------------------------------------------------ #

    def _maybe_init_wandb(self) -> None:
        if bool(self.cfg.disable_wandb) or wandb is None:
            return
        if os.environ.get("WANDB_DISABLED", "").lower() in {"1", "true"}:
            return
        try:
            self._wandb_run = wandb.init(
                project=str(self.cfg.wandb_project),
                name=(str(self.cfg.run_name) or None),
                mode=str(self.cfg.wandb_mode),
                config=asdict(self.cfg),
                reinit=True,
            )
        except Exception as exc:
            print(f"[d4_trainer] wandb init failed, continuing without logging: {exc}")
            self._wandb_run = None

    def _log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if self._wandb_run is None:
            return
        try:
            self._wandb_run.log(
                {k: v for k, v in metrics.items() if not isinstance(v, str)},
                step=step,
            )
        except Exception:
            pass

    def _split_clean_positives(
        self, ratio: float
    ) -> Tuple[List[int], List[int], List[int]]:
        """Return (train_all, train_clean_only, held_out).

        - ``train_all``: clean_train + fail_raw (used in Phase B).
        - ``train_clean_only``: clean_train only (used in Phase A so f(+) is
          never exposed to fail_raw).
        - ``held_out``: clean frames reserved for alpha0 / held_out_r_plus.
        """
        all_idx = list(range(len(self.dataset)))
        is_fail = self.dataset.is_fail_raw.tolist()
        clean_idx = [i for i in all_idx if not is_fail[i]]
        fail_idx = [i for i in all_idx if is_fail[i]]
        n_clean = len(clean_idx)
        if n_clean == 0:
            return all_idx, [], []
        n_hold = int(round(float(ratio) * n_clean))
        n_hold = min(max(n_hold, 1), max(n_clean - 1, 0))
        gen = torch.Generator().manual_seed(0xD4D15C + n_clean)
        perm = torch.randperm(n_clean, generator=gen).tolist()
        held = [clean_idx[i] for i in perm[:n_hold]]
        train_clean = sorted([clean_idx[i] for i in perm[n_hold:]])
        train_all = sorted(train_clean + fail_idx)
        return train_all, train_clean, sorted(held)

    def _build_loader(self, indices: List[int], shuffle: bool = True) -> DataLoader:
        subset = Subset(self.dataset, list(indices))
        nw = int(self.cfg.num_workers)
        return DataLoader(
            subset,
            batch_size=int(self.cfg.batch_size),
            shuffle=bool(shuffle),
            num_workers=nw,
            pin_memory=(self.device.type == "cuda"),
            collate_fn=_collate,
            drop_last=False,
            persistent_workers=(nw > 0),
            prefetch_factor=(4 if nw > 0 else None),
        )

    @contextlib.contextmanager
    def _ema_weights(self):
        """Context manager: swap EMA weights into ``self.predictor`` for the
        duration of the block, then restore raw weights.

        Stash raw weights on CPU. The trainer enters this context ~3× per
        Phase B epoch (gate eval, held-out eval, cond_delta probe); stashing
        on GPU doubled the resident model size each time and contributed to
        OOMs on 24 GB cards."""
        raw_sd = {k: v.detach().to("cpu", copy=True) for k, v in self.predictor.state_dict().items()}
        self.ema.apply_to(self.predictor)
        try:
            yield self.predictor
        finally:
            self.predictor.load_state_dict(raw_sd, strict=True)
            del raw_sd

    # ------------------------------------------------------------------ #
    # Condition sampling                                                 #
    # ------------------------------------------------------------------ #

    def _sample_condition(
        self,
        is_fail_raw: torch.Tensor,
        gamma: torch.Tensor,
        phase: str,
    ) -> torch.Tensor:
        """Per-sample cond_idx in {+, -, null}.

        Phase A: every sample routed to c=+ unconditionally. The Phase A
            loader excludes fail_raw (see ``_split_clean_positives``), so
            f(+) is a clean-only supervised dynamics predictor. No null/CFG
            dropout — the null branch is unused in our inference path
            (detector reads r+/r-, never f(∅)) and wasting batch slots on
            it only slows f(+) convergence.
        Phase B: fail_raw routed by gamma with CFG dropout.
        Clean positives in Phase B routed to {+, null}.
        """
        B = is_fail_raw.shape[0]
        device = is_fail_raw.device
        p_drop = float(self.cfg.p_cond_drop)
        cond = torch.full((B,), ConditionEmbedder.COND_PLUS, dtype=torch.long, device=device)
        null_fill = torch.full_like(cond, ConditionEmbedder.COND_NULL)
        plus_fill = torch.full_like(cond, ConditionEmbedder.COND_PLUS)
        minus_fill = torch.full_like(cond, ConditionEmbedder.COND_MINUS)

        if phase == "A":
            # Phase A loader is clean-only; every sample trains f(+).
            return cond

        u = torch.rand(B, device=device)

        # Phase B: clean positives get c=+ w.p. (1 - p_drop), else null.
        cond = torch.where((~is_fail_raw) & (u < p_drop), null_fill, cond)

        # Phase B: fail_raw routed by gamma.
        fail_threshold = p_drop + (1.0 - p_drop) * (1.0 - gamma.to(u.dtype))
        fail_cond = torch.where(
            u < p_drop,
            null_fill,
            torch.where(u < fail_threshold, plus_fill, minus_fill),
        )
        cond = torch.where(is_fail_raw, fail_cond, cond)
        return cond

    def _enforce_batch_balance(
        self, cond: torch.Tensor, is_fail_raw: torch.Tensor, phase: str
    ) -> torch.Tensor:
        if phase != "B":
            return cond
        B = cond.shape[0]
        if B == 0:
            return cond
        target = float(self.cfg.min_batch_balance)
        count_minus = int((cond == ConditionEmbedder.COND_MINUS).sum().item())
        needed = int(math.ceil(target * B)) - count_minus
        if needed <= 0:
            return cond
        promotable = (
            is_fail_raw & (cond != ConditionEmbedder.COND_MINUS)
        ).nonzero(as_tuple=False).squeeze(-1)
        if promotable.numel() == 0:
            return cond
        k = int(min(needed, promotable.numel()))
        perm = torch.randperm(promotable.numel(), device=cond.device)[:k]
        chosen = promotable[perm]
        cond[chosen] = ConditionEmbedder.COND_MINUS
        return cond

    # ------------------------------------------------------------------ #
    # Step / epoch                                                       #
    # ------------------------------------------------------------------ #

    def _dtype_context(self):
        if bool(self.cfg.use_bfloat16) and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _move_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                out[k] = v.to(self.device, non_blocking=True)
        return out

    def _run_step(self, batch: Dict[str, torch.Tensor], phase: str, train: bool) -> Dict[str, float]:
        b = self._move_batch(batch)
        cond = self._sample_condition(b["is_fail_raw"], b["gamma"], phase=phase)
        cond = self._enforce_batch_balance(cond, b["is_fail_raw"], phase=phase)

        if train:
            self.optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train), self._dtype_context():
            out = self.predictor(
                b["current_latent"],
                b["current_proprio"],
                b["action_sequence"],
                cond_idx=cond,
            )
            latent_mse = F.mse_loss(out["pred_latent"], b["target_latent"])
            proprio_mse = F.mse_loss(out["pred_proprio"], b["target_proprio"])
            loss = (
                float(self.cfg.latent_loss_weight) * latent_mse
                + float(self.cfg.proprio_loss_weight) * proprio_mse
            )
            if train:
                loss.backward()
                if self.cfg.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.predictor.parameters(), float(self.cfg.grad_clip_norm)
                    )
                self.optimizer.step()
                self.ema.update(self.predictor)

        n_plus = int((cond == ConditionEmbedder.COND_PLUS).sum().item())
        n_minus = int((cond == ConditionEmbedder.COND_MINUS).sum().item())
        n_null = int((cond == ConditionEmbedder.COND_NULL).sum().item())
        return {
            "loss": float(loss.detach().item()),
            "latent_mse": float(latent_mse.detach().item()),
            "proprio_mse": float(proprio_mse.detach().item()),
            "cond_plus": n_plus,
            "cond_minus": n_minus,
            "cond_null": n_null,
        }

    def _train_one_epoch(self, epoch: int, phase: str) -> Dict[str, float]:
        self.predictor.train()
        logs: List[Dict[str, float]] = []
        loader = self._train_loader_phase_a if phase == "A" else self._train_loader
        for step, batch in enumerate(loader):
            out = self._run_step(batch, phase=phase, train=True)
            logs.append(out)
            if int(self.cfg.log_every) > 0 and step % int(self.cfg.log_every) == 0:
                print(
                    f"[d4_train][{phase}] epoch={epoch:03d} step={step:05d}  "
                    f"loss={out['loss']:.4f} lat={out['latent_mse']:.4f} prop={out['proprio_mse']:.4f}  "
                    f"c+={out['cond_plus']} c-={out['cond_minus']} cN={out['cond_null']}"
                )
        if not logs:
            return {}
        keys = sorted(logs[0].keys())
        return {k: float(sum(m[k] for m in logs) / len(logs)) for k in keys}

    # ------------------------------------------------------------------ #
    # Probes (always run under EMA weights context)                      #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _eval_r_plus_on_held_out(self, model: ConditionalDynamicsPredictor) -> Tuple[float, torch.Tensor]:
        if self._held_out_loader is None:
            return 0.0, torch.zeros(0)
        model.eval()
        residuals: list[torch.Tensor] = []
        for batch in self._held_out_loader:
            b = self._move_batch(batch)
            B = b["current_latent"].shape[0]
            cond = torch.full((B,), ConditionEmbedder.COND_PLUS, dtype=torch.long, device=self.device)
            out = model(b["current_latent"], b["current_proprio"], b["action_sequence"], cond_idx=cond)
            r = ((out["pred_latent"] - b["target_latent"]) ** 2).sum(dim=-1)
            residuals.append(r.detach().cpu())
        if not residuals:
            return 0.0, torch.zeros(0)
        flat = torch.cat(residuals, dim=0)
        return float(flat.mean().item()), flat

    @torch.no_grad()
    def _eval_conditional_delta(self, model: ConditionalDynamicsPredictor) -> float:
        if self._probe_batch is None:
            return 0.0
        model.eval()
        b = self._move_batch(self._probe_batch)
        B = b["current_latent"].shape[0]
        cp = torch.full((B,), ConditionEmbedder.COND_PLUS, dtype=torch.long, device=self.device)
        cm = torch.full((B,), ConditionEmbedder.COND_MINUS, dtype=torch.long, device=self.device)
        out_p = model(b["current_latent"], b["current_proprio"], b["action_sequence"], cond_idx=cp)
        out_m = model(b["current_latent"], b["current_proprio"], b["action_sequence"], cond_idx=cm)
        return float(((out_p["pred_latent"] - out_m["pred_latent"]) ** 2).mean().item())

    def _cache_probe_batch(self) -> None:
        try:
            batch = next(iter(self._train_loader))
        except StopIteration:
            return
        B = int(min(self.cfg.probe_batch_size, batch["current_latent"].shape[0]))
        self._probe_batch = {
            k: (v[:B].clone() if torch.is_tensor(v) else v) for k, v in batch.items()
        }

    # ------------------------------------------------------------------ #
    # Checkpoint                                                         #
    # ------------------------------------------------------------------ #

    def build_payload(self) -> Dict[str, Any]:
        return {
            "raw_state_dict": self.predictor.state_dict(),
            "ema_state_dict": self.ema.shadow,
            "ema_decay": self.ema.decay,
            "arch_args": self.predictor.arch_args(),
            "config": asdict(self.cfg),
            "gamma_buffer": self.dataset.gamma_buffer.clone(),
            "is_fail_raw": self.dataset.is_fail_raw.clone(),
            "schedule_state": asdict(self.cfg.schedule),
            "history": self.history,
            "health_history": [asdict(h) for h in self.health_history],
            "sigma_sq": float(self.cfg.sigma_sq),
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        torch.save(self.build_payload(), path)
        print(f"[d4_trainer] saved checkpoint -> {path}")

    # ------------------------------------------------------------------ #
    # Main loop                                                          #
    # ------------------------------------------------------------------ #

    def run(
        self,
        save_path: str,
        *,
        save_freq: int = 0,
    ) -> Dict[str, Any]:
        K_warm = int(self.cfg.warm_up_epochs)
        K_boot = int(self.cfg.bootstrap_epochs)
        save_freq = int(save_freq or self.cfg.save_freq)

        # Phase A.
        for epoch in range(1, K_warm + 1):
            logs = self._train_one_epoch(epoch, phase="A")
            with self._ema_weights() as ema_model:
                held_mean, _ = self._eval_r_plus_on_held_out(ema_model)
            entry = {"epoch": epoch, **logs, "held_out_r_plus": held_mean}
            self.history["phase_A"].append(entry)
            self._log({f"phaseA/{k}": v for k, v in entry.items()}, step=epoch)
            print(f"[d4_epoch A {epoch:03d}] {entry}")
            if save_freq > 0 and (epoch % save_freq == 0):
                self.save(_with_epoch_suffix(save_path, epoch, prefix="A"))

        # End of Phase A: snapshot alpha0 from MAD of held-out r_plus.
        with self._ema_weights() as ema_model:
            _, held_flat = self._eval_r_plus_on_held_out(ema_model)
        mad = _mad(held_flat)
        auto_alpha0 = 1.0 / max(mad, 1e-6)
        self.cfg.schedule.alpha0 = float(auto_alpha0)
        print(f"[d4_trainer] end of Phase A: held_out_mad={mad:.6f}  alpha0={auto_alpha0:.4f}")

        # F3-based gamma warm-start: break the gamma=0.5 symmetric fixed point
        # by seeding gamma with a D3-style KNN-distance signal. Must run before
        # the first bootstrap iteration; otherwise the critic enters Phase B
        # with identical f(+) and f(-) branches and gets stuck.
        if bool(self.cfg.f3_warm_start):
            warm_diag = warm_start_gamma_from_knn(
                self.dataset,
                k=int(self.cfg.f3_warm_start_k),
                mode=str(self.cfg.warm_start_mode),
                max_clean_bank=int(self.cfg.f3_warm_start_max_clean),
                device=str(self.device),
                batch_size=int(self.cfg.batch_size),
                num_workers=int(self.cfg.num_workers),
                seed=0,
            )
            print(f"[d4_trainer] F3 warm-start: {warm_diag}")
            self._log({f"warmstart/{k}": v for k, v in warm_diag.items()}, step=K_warm)
            self.history.setdefault("warm_start", []).append(warm_diag)
        else:
            print("[d4_trainer] F3 warm-start disabled; gamma starts at 0.5 "
                  "(NOTE: symmetric fixed point risk).")

        if bool(self.cfg.skip_bootstrap):
            print("[d4_trainer] skip_bootstrap=True: saving Phase-A-only "
                  "checkpoint. Use OMEGA=0 at benchmark time for the "
                  "pos-branch-only baseline (f(-) is at init, do NOT use it).")
            self.save(save_path)
            if self._wandb_run is not None:
                try:
                    self._wandb_run.finish()
                except Exception:
                    pass
            return {"history": self.history, "health_history": self.health_history}

        self._cache_probe_batch()

        # Phase B.
        freeze_epochs = int(self.cfg.gate_freeze_epochs) if bool(self.cfg.f3_warm_start) else 0
        for k in range(K_boot):
            outer_epoch = K_warm + k + 1

            # 1) Gate + held-out probe under EMA weights. Skip the gate call
            #    while frozen, so warm-started gamma drives training.
            with self._ema_weights() as ema_model:
                held_before, _ = self._eval_r_plus_on_held_out(ema_model)
                alpha_raw = self.cfg.schedule.alpha(k)
                alpha_k = self.cfg.schedule.clamped_alpha(alpha_raw, held_before)
                kappa_k = self.cfg.schedule.kappa(k)
                if k < freeze_epochs:
                    fail_idx_now = self.dataset.fail_indices()
                    frozen_diag = _gamma_quantile_diag(
                        self.dataset.gamma_buffer[fail_idx_now]
                    )
                    gate_diag = {
                        **frozen_diag,
                        "mean_r_plus": 0.0,
                        "mean_r_minus": 0.0,
                        "separation_gap": 0.0,
                        "alpha_used": 0.0,
                        "kappa_used": float(kappa_k),
                        "num_fail_raw": int(fail_idx_now.numel()),
                        "gate_frozen": 1,
                    }
                else:
                    gate_diag = compute_advantage_gate(
                        ema_model,
                        self.dataset,
                        alpha_k=alpha_k,
                        kappa_k=kappa_k,
                        sigma_sq=float(self.cfg.sigma_sq),
                        device=str(self.device),
                        batch_size=int(self.cfg.batch_size),
                        num_workers=int(self.cfg.num_workers),
                        ema_alpha=float(self.cfg.ema_alpha_gamma),
                    )
                    gate_diag["gate_frozen"] = 0

            eta_k = self.cfg.schedule.eta(k)

            # 2) Train.
            train_logs = self._train_one_epoch(outer_epoch, phase="B")

            # 3) Post-epoch health snapshot under EMA.
            with self._ema_weights() as ema_model:
                held_after, _ = self._eval_r_plus_on_held_out(ema_model)
                cond_delta = self._eval_conditional_delta(ema_model)

            health = D4Health(
                held_out_r_plus=float(held_after),
                mean_gamma=float(gate_diag.get("mean_gamma", 0.5)),
                separation_gap=float(gate_diag.get("separation_gap", 0.0)),
                conditional_delta=float(cond_delta),
                alpha_used=float(gate_diag.get("alpha_used", alpha_k)),
                kappa_used=float(gate_diag.get("kappa_used", kappa_k)),
                eta_used=float(eta_k),
                mean_r_plus=float(gate_diag.get("mean_r_plus", 0.0)),
                mean_r_minus=float(gate_diag.get("mean_r_minus", 0.0)),
                step=0,
                epoch=outer_epoch,
            )
            self.health_history.append(health)

            reason = self.collapse_detector.update(health)
            if reason is not None:
                print(
                    f"[d4_trainer] COLLAPSE TRIGGERED ({reason}) at epoch {outer_epoch}. "
                    f"Rolling back to EMA, halving alpha0."
                )
                self.ema.apply_to(self.predictor)
                self.cfg.schedule.halve_alpha0()
                self.collapse_detector.reset_counters()

            # Gamma quantile snapshot (in addition to mean_gamma from gate_diag).
            fail_idx_now = self.dataset.fail_indices()
            gamma_snapshot = _gamma_quantile_diag(self.dataset.gamma_buffer[fail_idx_now])

            entry = {
                "outer_epoch": outer_epoch,
                "bootstrap_k": k,
                **train_logs,
                **gate_diag,
                **{f"gamma_{k2}": v for k2, v in gamma_snapshot.items() if k2.startswith("gamma_")},
                "conditional_delta": cond_delta,
                "collapse_reason": reason,
            }
            self.history["phase_B"].append(entry)
            self._log(
                {f"phaseB/{k2}": v for k2, v in entry.items() if not isinstance(v, str)},
                step=outer_epoch,
            )
            print(f"[d4_epoch B {outer_epoch:03d}] {entry}")

            if save_freq > 0 and (outer_epoch % save_freq == 0):
                self.save(_with_epoch_suffix(save_path, outer_epoch, prefix="B"))

        self.save(save_path)
        if self._wandb_run is not None:
            try:
                self._wandb_run.finish()
            except Exception:
                pass
        return {"history": self.history, "health_history": self.health_history}


def _with_epoch_suffix(path: str, epoch: int, prefix: str = "") -> str:
    stem, ext = os.path.splitext(path)
    if not ext:
        ext = ".pt"
    suffix = f"_{prefix}ep{epoch:04d}" if prefix else f"_ep{epoch:04d}"
    return f"{stem}{suffix}{ext}"
