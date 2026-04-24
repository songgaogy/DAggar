from __future__ import annotations

import contextlib
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from ..data.dataset import LatentFlowDynamicsDatasetD4
from ..models.adaln import ConditionEmbedder
from ..models.dynamics import ConditionalDynamicsPredictor
from ..models.encoder import Encoder
from .ema import ModelEMA
from .filter import (
    _collate,
    _gamma_quantile_diag,
    build_expert_target_bank,
    compute_advantage_gate,
    warm_start_gamma_from_knn,
)
from .monitor import CollapseDetector, D4Health
from .schedule import D4Schedule

try:
    import wandb  # type: ignore
except ImportError:
    wandb = None  # type: ignore


_NO_WD_SUBSTRINGS = ("pos_embedding", "cond_embed.embed.weight")


@dataclass
class D4TrainerConfig:
    warm_up_epochs: int = 8
    bootstrap_epochs: int = 40

    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    batch_size: int = 256
    num_workers: int = 4
    encoder_freeze: bool = True
    encoder_lr_mult: float = 0.1

    min_batch_balance: float = 0.15

    schedule: D4Schedule = field(default_factory=D4Schedule)

    ema_decay_model: float = 0.999
    ema_alpha_gamma: float = 0.5
    gamma_clamp: float = 1e-3
    held_out_ratio: float = 0.05

    proprio_loss_weight: float = 0.1
    latent_loss_weight: float = 1.0
    sigma_sq: float = 0.5

    use_bfloat16: bool = False

    log_every: int = 200
    save_freq: int = 0
    run_name: str = ""
    wandb_project: str = "d4disc"
    wandb_mode: str = "offline"
    disable_wandb: bool = False

    probe_batch_size: int = 64

    f3_warm_start: bool = True
    f3_warm_start_k: int = 1
    f3_warm_start_max_clean: int = 100_000
    warm_start_mode: str = "rank"
    gate_freeze_epochs: int = 2
    skip_bootstrap: bool = False

    # Phase-B advantage gate mode. "knn" (default) scores each conditional
    # prediction against an expert z_{t+h} bank via min-sqdist, giving an
    # advantage that is exactly log p_+ - log p_- under Gaussian-KNN. This
    # matches the inference-time score_mode="knn" density paradigm and
    # avoids the motion-magnitude confound that raw residual scoring has on
    # Phase-A-only checkpoints (d4disc_0424_debug_lpb_degraded.md).
    # "residual" keeps d4disc_0423.md §5 verbatim for ablations.
    advantage_mode: str = "knn"
    advantage_knn_bank_size: int = 100_000
    advantage_knn_chunk_size: int = 8192
    advantage_knn_k: int = 1

    # Phase-B M-step repel loss (anchor c=- away from B_+ expert bank).
    repel_weight: float = 0.0
    repel_margin: float = -1.0
    repel_margin_percentile: float = 0.5
    repel_on_phase_a: bool = False
    repel_warmup_epochs: int = 0

    horizon: int = 1
    proprio_indices: Optional[List[int]] = None

    def __post_init__(self) -> None:
        if self.schedule is None:
            self.schedule = D4Schedule()
        if self.proprio_indices is not None:
            self.proprio_indices = [int(x) for x in self.proprio_indices]
        self.repel_margin_percentile = float(min(max(self.repel_margin_percentile, 0.0), 1.0))
        self.repel_warmup_epochs = max(int(self.repel_warmup_epochs), 0)


def _partition_params_for_wd(model: torch.nn.Module) -> Tuple[list, list]:
    wd_params: list = []
    no_wd_params: list = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(substr in name for substr in _NO_WD_SUBSTRINGS):
            no_wd_params.append(param)
        elif name.endswith(".bias") or "ln_" in name or "norm_final" in name:
            no_wd_params.append(param)
        else:
            wd_params.append(param)
    return wd_params, no_wd_params


def _mad(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return 1.0
    med = values.median()
    mad = (values - med).abs().median()
    return float(mad.item())


def _chunked_knn_sqdist_autograd(
    query: torch.Tensor,
    bank: torch.Tensor,
    *,
    k: int = 1,
    chunk_size: int = 8192,
) -> torch.Tensor:
    if query.ndim != 2 or bank.ndim != 2:
        raise ValueError(f"query/bank must be 2D, got {tuple(query.shape)} and {tuple(bank.shape)}")
    if query.shape[1] != bank.shape[1]:
        raise ValueError(f"dim mismatch query={query.shape[1]} bank={bank.shape[1]}")
    if int(bank.shape[0]) == 0:
        raise ValueError("bank cannot be empty")

    query_f = query.float()
    # Keep the bank fixed but leave gradients on the query path.
    bank_f = bank.detach().to(device=query.device, dtype=query_f.dtype, non_blocking=True)
    eff_k = max(1, min(int(k), int(bank_f.shape[0])))
    q_norm = (query_f * query_f).sum(dim=1, keepdim=True)
    best = torch.full((int(query_f.shape[0]), eff_k), float("inf"), device=query_f.device, dtype=query_f.dtype)

    for start in range(0, int(bank_f.shape[0]), int(chunk_size)):
        end = min(start + int(chunk_size), int(bank_f.shape[0]))
        chunk = bank_f[start:end]
        # Squared L2 via ||q-b||^2 = ||q||^2 + ||b||^2 - 2 q·b, chunked over the bank.
        b_norm = (chunk * chunk).sum(dim=1).unsqueeze(0)
        d2 = (q_norm + b_norm - 2.0 * (query_f @ chunk.transpose(0, 1))).clamp_min(0.0)
        merged = torch.cat([best, d2], dim=1)
        best = torch.topk(merged, k=eff_k, dim=1, largest=False).values

    return best.mean(dim=1)


class D4Trainer:
    def __init__(
        self,
        predictor: ConditionalDynamicsPredictor,
        encoder: Encoder,
        dataset: LatentFlowDynamicsDatasetD4,
        *,
        config: Optional[D4TrainerConfig] = None,
        device: Optional[str] = None,
    ) -> None:
        self.predictor = predictor
        self.encoder = encoder
        self.dataset = dataset
        self.cfg = config or D4TrainerConfig()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.predictor.to(self.device)
        self.encoder.to(self.device)

        self.encoder_frozen = bool(self.cfg.encoder_freeze)
        if self.encoder_frozen:
            self.encoder.freeze()
        else:
            self.encoder.unfreeze()

        wd_params, no_wd_params = _partition_params_for_wd(self.predictor)
        optim_groups = [
            {"params": wd_params, "weight_decay": float(self.cfg.weight_decay)},
            {"params": no_wd_params, "weight_decay": 0.0},
        ]
        encoder_params = [p for p in self.encoder.parameters() if p.requires_grad]
        if encoder_params:
            optim_groups.append(
                {
                    "params": encoder_params,
                    "weight_decay": float(self.cfg.weight_decay),
                    "lr": float(self.cfg.lr) * float(self.cfg.encoder_lr_mult),
                }
            )
        self.optimizer = torch.optim.AdamW(optim_groups, lr=float(self.cfg.lr))

        self.ema = ModelEMA(self.predictor, decay=float(self.cfg.ema_decay_model))
        self.collapse_detector = CollapseDetector()

        (
            self._train_indices,
            self._train_indices_clean_only,
            self._held_out_indices,
        ) = self._split_clean_positives(ratio=float(self.cfg.held_out_ratio))
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
        # Phase-B KNN bank and repel state are resolved lazily in run().
        self._expert_z_bank: Optional[torch.Tensor] = None
        self._repel_margin: Optional[float] = None
        self._current_bootstrap_k: int = 0

        self.history: Dict[str, Any] = {"phase_A": [], "phase_B": []}
        self.health_history: List[D4Health] = []

        self._wandb_run = None
        self._maybe_init_wandb()

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

    def _split_clean_positives(self, ratio: float) -> Tuple[List[int], List[int], List[int]]:
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
        raw_sd = {k: v.detach().to("cpu", copy=True) for k, v in self.predictor.state_dict().items()}
        self.ema.apply_to(self.predictor)
        try:
            yield self.predictor
        finally:
            self.predictor.load_state_dict(raw_sd, strict=True)
            del raw_sd

    def _sample_condition(self, is_fail_raw: torch.Tensor, gamma: torch.Tensor, phase: str) -> torch.Tensor:
        """Sample per-example condition indices.

        Phase A:
            - Always route to (+). Fail samples are excluded upstream.

        Phase B:
            - For clean positives: always (+).
            - For fail samples: sample (-) with probability gamma_i, else (+).
        """
        batch_size = int(is_fail_raw.shape[0])
        device = is_fail_raw.device
        cond = torch.full((batch_size,), ConditionEmbedder.COND_PLUS, dtype=torch.long, device=device)
        plus_fill = torch.full_like(cond, ConditionEmbedder.COND_PLUS)
        minus_fill = torch.full_like(cond, ConditionEmbedder.COND_MINUS)

        if phase == "A":
            return cond

        u = torch.rand(batch_size, device=device)
        fail_cond = torch.where(u < gamma.to(u.dtype), minus_fill, plus_fill)
        cond = torch.where(is_fail_raw, fail_cond, cond)
        return cond

    def _enforce_batch_balance(self, cond: torch.Tensor, is_fail_raw: torch.Tensor, phase: str) -> torch.Tensor:
        """Ensure at least `min_batch_balance` fraction of (-) samples in Phase B.

        This is a training stability knob: if gamma collapses early (e.g. near 0),
        the (-) branch would get almost no updates and cannot recover.
        """
        if phase != "B":
            return cond
        batch_size = int(cond.shape[0])
        if batch_size == 0:
            return cond
        target = float(self.cfg.min_batch_balance)
        count_minus = int((cond == ConditionEmbedder.COND_MINUS).sum().item())
        needed = int(math.ceil(target * batch_size)) - count_minus
        if needed <= 0:
            return cond
        promotable = (is_fail_raw & (cond != ConditionEmbedder.COND_MINUS)).nonzero(as_tuple=False).squeeze(-1)
        if promotable.numel() == 0:
            return cond
        k = int(min(needed, promotable.numel()))
        perm = torch.randperm(promotable.numel(), device=cond.device)[:k]
        cond[promotable[perm]] = ConditionEmbedder.COND_MINUS
        return cond

    def _dtype_context(self):
        if bool(self.cfg.use_bfloat16) and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _move_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                out[key] = value.to(self.device, non_blocking=True)
        return out

    def _encode_observation(self, image: torch.Tensor, *, allow_grad: bool) -> torch.Tensor:
        if allow_grad:
            return self.encoder(image)
        with torch.no_grad():
            return self.encoder(image)

    def _run_step(self, batch: Dict[str, torch.Tensor], phase: str, train: bool) -> Dict[str, float]:
        b = self._move_batch(batch)
        cond = self._sample_condition(b["is_fail_raw"], b["gamma"], phase=phase)
        cond = self._enforce_batch_balance(cond, b["is_fail_raw"], phase=phase)

        if train:
            self.optimizer.zero_grad(set_to_none=True)

        allow_encoder_grad = train and (not self.encoder_frozen)
        with torch.set_grad_enabled(train), self._dtype_context():
            z_t = self._encode_observation(b["current_image"], allow_grad=allow_encoder_grad)
            z_target = self._encode_observation(b["target_image"], allow_grad=False).detach()
            out = self.predictor(
                z_t,
                b["current_proprio"],
                b["action_sequence"],
                cond_idx=cond,
            )

            # loss: latent_mse + proprio_mse 
            # NOTE: dynamic model outputs latent and proprio
            latent_mse = F.mse_loss(out["pred_latent"], z_target)
            proprio_mse = F.mse_loss(out["pred_proprio"], b["target_proprio"])
            loss = float(self.cfg.latent_loss_weight) * latent_mse + float(self.cfg.proprio_loss_weight) * proprio_mse

            repel_hinge_val = 0.0
            # Repel is an extra M-step term. It never changes routing or the gate.
            repel_active = (
                train
                and float(self.cfg.repel_weight) > 0.0
                and self._expert_z_bank is not None
                and self._repel_margin is not None
                and (phase == "B" or bool(self.cfg.repel_on_phase_a))
                and (phase != "B" or int(self._current_bootstrap_k) >= int(self.cfg.repel_warmup_epochs))
            )

            if repel_active:
                # Repel applies only to fail_rollout rows (`is_fail_raw=True`).
                # Success rollouts / expert positives are clean by construction and must
                # not receive a "push away from the expert bank" gradient.
                fail_mask = b["is_fail_raw"].to(device=z_t.device, dtype=torch.bool).view(-1)
                if fail_mask.any():
                    z_f = z_t[fail_mask]
                    prop_f = b["current_proprio"][fail_mask]
                    act_f = b["action_sequence"][fail_mask]
                    n_fail = int(z_f.shape[0])
                    c_minus_all = torch.full(
                        (n_fail,),
                        ConditionEmbedder.COND_MINUS,
                        dtype=torch.long,
                        device=self.device,
                    )
                    # Re-run fail rows as all-minus so the hinge gives a direct
                    # gradient to the (-) branch even when gamma is uninformative.
                    out_minus_all = self.predictor(
                        z_f,
                        prop_f,
                        act_f,
                        cond_idx=c_minus_all,
                    )
                    # Penalize minus predictions that stay too close to the expert bank.
                    r_minus_to_pos = _chunked_knn_sqdist_autograd(
                        out_minus_all["pred_latent"],
                        self._expert_z_bank,
                        k=int(self.cfg.advantage_knn_k),
                        chunk_size=int(self.cfg.advantage_knn_chunk_size),
                    )
                    repel_hinge = torch.clamp(float(self._repel_margin) - r_minus_to_pos, min=0.0).mean()
                    loss = loss + float(self.cfg.repel_weight) * repel_hinge
                    repel_hinge_val = float(repel_hinge.detach().item())

            if train:
                loss.backward()
                if self.cfg.grad_clip_norm > 0:
                    params = list(self.predictor.parameters())
                    if not self.encoder_frozen:
                        params.extend(self.encoder.parameters())
                    torch.nn.utils.clip_grad_norm_(params, float(self.cfg.grad_clip_norm))
                self.optimizer.step()
                self.ema.update(self.predictor)

        n_plus = int((cond == ConditionEmbedder.COND_PLUS).sum().item())
        n_minus = int((cond == ConditionEmbedder.COND_MINUS).sum().item())
        return {
            "loss": float(loss.detach().item()),
            "latent_mse": float(latent_mse.detach().item()),
            "proprio_mse": float(proprio_mse.detach().item()),
            "cond_plus": n_plus,
            "cond_minus": n_minus,
            "repel_hinge": repel_hinge_val,
        }

    def _train_one_epoch(self, epoch: int, phase: str) -> Dict[str, float]:
        self.predictor.train()
        self.encoder.train(not self.encoder_frozen)
        logs: List[Dict[str, float]] = []
        loader = self._train_loader_phase_a if phase == "A" else self._train_loader
        for step, batch in enumerate(loader):
            out = self._run_step(batch, phase=phase, train=True)
            logs.append(out)
            if int(self.cfg.log_every) > 0 and step % int(self.cfg.log_every) == 0:
                print(
                    f"[d4_train][{phase}] epoch={epoch:03d} step={step:05d}  "
                    f"loss={out['loss']:.4f} lat={out['latent_mse']:.4f} prop={out['proprio_mse']:.4f}  "
                    f"c+={out['cond_plus']} c-={out['cond_minus']}"
                )
        if not logs:
            return {}
        keys = sorted(logs[0].keys())
        return {k: float(sum(m[k] for m in logs) / len(logs)) for k in keys}

    @torch.no_grad()
    def _eval_r_plus_on_held_out(self, model: ConditionalDynamicsPredictor) -> Tuple[float, torch.Tensor]:
        """Evaluate mean r_plus on held-out clean positives.

        Used as:
            - a convergence proxy for alpha clamping (design coupling),
            - a robust scale estimator (MAD) to set alpha0 after Phase A.
        """
        if self._held_out_loader is None:
            return 0.0, torch.zeros(0)
        model.eval()
        self.encoder.eval()
        residuals: list[torch.Tensor] = []
        for batch in self._held_out_loader:
            b = self._move_batch(batch)
            z_t = self.encoder(b["current_image"])
            z_target = self.encoder(b["target_image"]).detach()
            batch_size = int(z_t.shape[0])
            cond = torch.full((batch_size,), ConditionEmbedder.COND_PLUS, dtype=torch.long, device=self.device)
            out = model(z_t, b["current_proprio"], b["action_sequence"], cond_idx=cond)
            residuals.append(((out["pred_latent"] - z_target) ** 2).sum(dim=-1).detach().cpu())
        flat = torch.cat(residuals, dim=0) if residuals else torch.zeros(0)
        return float(flat.mean().item()) if flat.numel() > 0 else 0.0, flat

    @torch.no_grad()
    def _eval_conditional_delta(self, model: ConditionalDynamicsPredictor) -> float:
        """Diagnostic: mean squared difference between f(+) and f(-) predictions."""
        if self._probe_batch is None:
            return 0.0
        model.eval()
        self.encoder.eval()
        b = self._move_batch(self._probe_batch)
        z_t = self.encoder(b["current_image"])
        batch_size = int(z_t.shape[0])
        cp = torch.full((batch_size,), ConditionEmbedder.COND_PLUS, dtype=torch.long, device=self.device)
        cm = torch.full((batch_size,), ConditionEmbedder.COND_MINUS, dtype=torch.long, device=self.device)
        out_p = model(z_t, b["current_proprio"], b["action_sequence"], cond_idx=cp)
        out_m = model(z_t, b["current_proprio"], b["action_sequence"], cond_idx=cm)
        return float(((out_p["pred_latent"] - out_m["pred_latent"]) ** 2).mean().item())

    def _cache_probe_batch(self) -> None:
        try:
            batch = next(iter(self._train_loader))
        except StopIteration:
            return
        batch_size = int(min(self.cfg.probe_batch_size, batch["current_image"].shape[0]))
        self._probe_batch = {k: (v[:batch_size].clone() if torch.is_tensor(v) else v) for k, v in batch.items()}

    @torch.no_grad()
    def _build_expert_target_bank(self) -> torch.Tensor:
        return build_expert_target_bank(
            self.dataset,
            self.encoder,
            max_bank_size=int(self.cfg.advantage_knn_bank_size),
            batch_size=int(self.cfg.batch_size),
            num_workers=int(self.cfg.num_workers),
            device=str(self.device),
            seed=0,
        )

    @torch.no_grad()
    def _compute_repel_margin(self, bank: torch.Tensor, percentile: float) -> float:
        if int(bank.shape[0]) == 0:
            raise RuntimeError("repel margin requires a non-empty expert bank")

        fail_idx = self.dataset.fail_indices()
        if int(fail_idx.numel()) == 0:
            return 0.0

        encoder_was_training = self.encoder.training
        self.encoder.eval()

        # Auto-margin uses fail current-image latents against the same expert bank
        # used by the Phase-B KNN gate, so the scale matches the training metric.
        subset = Subset(self.dataset, fail_idx.tolist())
        nw = int(self.cfg.num_workers)
        loader = DataLoader(
            subset,
            batch_size=int(self.cfg.batch_size),
            shuffle=False,
            num_workers=nw,
            pin_memory=(self.device.type == "cuda"),
            collate_fn=_collate,
            drop_last=False,
            persistent_workers=(nw > 0),
            prefetch_factor=(4 if nw > 0 else None),
        )

        d2_all: list[torch.Tensor] = []
        for batch in loader:
            b = self._move_batch(batch)
            z_t = self.encoder(b["current_image"])
            d2 = _chunked_knn_sqdist_autograd(
                z_t,
                bank,
                k=int(self.cfg.advantage_knn_k),
                chunk_size=int(self.cfg.advantage_knn_chunk_size),
            )
            d2_all.append(d2.detach().cpu())

        if encoder_was_training:
            self.encoder.train()

        flat = torch.cat(d2_all, dim=0) if d2_all else torch.zeros(0)
        if flat.numel() == 0:
            return 0.0
        return float(torch.quantile(flat, float(percentile)).item())

    def _resolve_repel_margin(self, bank: torch.Tensor) -> tuple[float, str]:
        if float(self.cfg.repel_margin) >= 0.0:
            return float(self.cfg.repel_margin), "manual"
        # Negative config value means "derive once from fail-vs-bank diagnostics".
        margin = self._compute_repel_margin(bank, float(self.cfg.repel_margin_percentile))
        return margin, f"auto, p={float(self.cfg.repel_margin_percentile):.3g}"

    def _log_repel_activation(self) -> None:
        if float(self.cfg.repel_weight) <= 0.0 or self._repel_margin is None:
            return
        mode = "manual" if float(self.cfg.repel_margin) >= 0.0 else f"auto, p={float(self.cfg.repel_margin_percentile):.3g}"
        print(
            f"[d4_trainer] repel loss active: weight={float(self.cfg.repel_weight):.2f} "
            f"margin={float(self._repel_margin):.6f} ({mode})"
        )

    def build_payload(self) -> Dict[str, Any]:
        return {
            "raw_state_dict": self.predictor.state_dict(),
            "ema_state_dict": self.ema.shadow,
            "ema_decay": self.ema.decay,
            "arch_args": self.predictor.arch_args(),
            "encoder_state_dict": self.encoder.state_dict(),
            "encoder_config": {
                "latent_dim": int(self.encoder.latent_dim),
                "normalize_input": bool(getattr(self.encoder, "normalize_input", True)),
                "freeze": bool(self.encoder_frozen),
            },
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

    def run(self, save_path: str, *, save_freq: int = 0) -> Dict[str, Any]:
        warm_epochs = int(self.cfg.warm_up_epochs)
        bootstrap_epochs = int(self.cfg.bootstrap_epochs)
        save_freq = int(save_freq or self.cfg.save_freq)

        self._expert_z_bank = None
        self._repel_margin = None
        self._current_bootstrap_k = 0

        if float(self.cfg.repel_weight) > 0.0 and bool(self.cfg.repel_on_phase_a):
            # Optional: cold-start the minus branch during Phase A with the same bank metric.
            self._expert_z_bank = self._build_expert_target_bank()
            self._repel_margin, _ = self._resolve_repel_margin(self._expert_z_bank)
            self._log_repel_activation()

        # -----------------------------------------------------
        # Phase A: training dynamics model on only success data
        # -----------------------------------------------------

        for epoch in range(1, warm_epochs + 1):
            logs = self._train_one_epoch(epoch, phase="A")
            with self._ema_weights() as ema_model:
                held_mean, _ = self._eval_r_plus_on_held_out(ema_model)

            entry = {"epoch": epoch, **logs, "held_out_r_plus": held_mean}
            self.history["phase_A"].append(entry)
            self._log({f"phaseA/{k}": v for k, v in entry.items()}, step=epoch)
            print(f"[d4_epoch A {epoch:03d}] {entry}")

            if save_freq > 0 and (epoch % save_freq == 0):
                self.save(_with_epoch_suffix(save_path, epoch, prefix="A"))

        # Use only Phase-A
        if bool(self.cfg.skip_bootstrap):
            print(
                "[d4_trainer] skip_bootstrap=True: saving Phase-A-only checkpoint. "
                "Use OMEGA=0 at benchmark time for the pos-branch-only baseline."
            )
            self.save(save_path)
            if self._wandb_run is not None:
                try:
                    self._wandb_run.finish()
                except Exception:
                    pass
            return {"history": self.history, "health_history": self.health_history}

        # -----------------------------------------------------------------
        # Phase B: training dynamics model on both success and failure data
        # -----------------------------------------------------------------

        with self._ema_weights() as ema_model:
            _, held_flat = self._eval_r_plus_on_held_out(ema_model)     # use ema_model to evaluate, more stable
        # MAD: Median Absolute Deviation
        mad = _mad(held_flat)
        # NOTE: if residual is large, indicates model is not able to predict the future state --> large error & noise, gate should be soft
        auto_alpha0 = 1.0 / max(mad, 1e-6)
        self.cfg.schedule.alpha0 = float(auto_alpha0)
        print(f"[d4_trainer] end of Phase A: held_out_mad={mad:.6f}  alpha0={auto_alpha0:.4f}")

        if bool(self.cfg.f3_warm_start):
            # warmup gamma from knn distance
            warm_diag = warm_start_gamma_from_knn(
                self.dataset,
                self.encoder,
                k=int(self.cfg.f3_warm_start_k),
                mode=str(self.cfg.warm_start_mode),
                max_clean_bank=int(self.cfg.f3_warm_start_max_clean),
                device=str(self.device),
                batch_size=int(self.cfg.batch_size),
                num_workers=int(self.cfg.num_workers),
                seed=0,
            )
            print(f"[d4_trainer] F3 warm-start: {warm_diag}")
            self._log({f"warmstart/{k}": v for k, v in warm_diag.items()}, step=warm_epochs)
            self.history.setdefault("warm_start", []).append(warm_diag)
        else:
            print("[d4_trainer] F3 warm-start disabled; gamma starts at 0.5 (NOTE: symmetric fixed point risk).")

        self._cache_probe_batch()

        # Build expert z_{t+h} bank once for KNN-mode advantage gate. 
        # NOTE: Encoder is frozen during Phase-B so the bank is stable across outer epochs.
        need_expert_bank = (str(self.cfg.advantage_mode) == "knn") or (float(self.cfg.repel_weight) > 0.0)
        if need_expert_bank:
            self._expert_z_bank = self._build_expert_target_bank()
            if float(self.cfg.repel_weight) > 0.0:
                # Resolve the repel margin after Phase A so it reflects the final bank.
                self._repel_margin, _ = self._resolve_repel_margin(self._expert_z_bank)
                self._log_repel_activation()
            else:
                self._repel_margin = None
        else:
            self._expert_z_bank = None
            self._repel_margin = None

        if str(self.cfg.advantage_mode) == "knn" and self._expert_z_bank is not None:
            print(
                f"[d4_trainer] built KNN advantage bank: "
                f"{int(self._expert_z_bank.shape[0])} expert target latents "
                f"(dim={int(self._expert_z_bank.shape[1])})"
            )
        elif float(self.cfg.repel_weight) > 0.0 and self._expert_z_bank is not None:
            print(
                f"[d4_trainer] built repel expert bank: "
                f"{int(self._expert_z_bank.shape[0])} expert target latents "
                f"(dim={int(self._expert_z_bank.shape[1])})"
            )

        # After warm-start, keep gamma frozen for a few epochs so the (-) branch
        # can differentiate before the gate re-enters the loop.
        freeze_epochs = int(self.cfg.gate_freeze_epochs) if bool(self.cfg.f3_warm_start) else 0
        for k in range(bootstrap_epochs):
            outer_epoch = warm_epochs + k + 1

            with self._ema_weights() as ema_model:
                held_before, _ = self._eval_r_plus_on_held_out(ema_model)
                alpha_raw = self.cfg.schedule.alpha(k)
                alpha_k = self.cfg.schedule.clamped_alpha(alpha_raw, held_before)
                kappa_k = self.cfg.schedule.kappa(k)
                if k < freeze_epochs:
                    # NOTE: here we do NOT update gamma, in order to avoid the risk of symmetric fixed point
                    fail_idx_now = self.dataset.fail_indices()
                    frozen_diag = _gamma_quantile_diag(self.dataset.gamma_buffer[fail_idx_now])
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
                    # Gate step (E-step): update dataset.gamma_buffer using EMA weights.
                    gate_diag = compute_advantage_gate(
                        ema_model,
                        self.encoder,
                        self.dataset,
                        alpha_k=alpha_k,
                        kappa_k=kappa_k,
                        sigma_sq=float(self.cfg.sigma_sq),
                        device=str(self.device),
                        batch_size=int(self.cfg.batch_size),
                        num_workers=int(self.cfg.num_workers),
                        ema_alpha=float(self.cfg.ema_alpha_gamma),
                        advantage_mode=str(self.cfg.advantage_mode),
                        expert_z_bank=self._expert_z_bank,
                        knn_k=int(self.cfg.advantage_knn_k),
                        knn_chunk_size=int(self.cfg.advantage_knn_chunk_size),
                    )
                    gate_diag["gate_frozen"] = 0

            eta_k = self.cfg.schedule.eta(k)
            # _run_step() reads this to delay repel during the first bootstrap epochs.
            self._current_bootstrap_k = k
            # Train step (M-step): sample conditions using updated gamma and optimize.
            train_logs = self._train_one_epoch(outer_epoch, phase="B")

            with self._ema_weights() as ema_model:
                held_after, _ = self._eval_r_plus_on_held_out(ema_model)
                cond_delta = self._eval_conditional_delta(ema_model)

            # -----------------------------------------------------------------------------
            # FIXME: failure analysis; remove when code should be released
            # -----------------------------------------------------------------------------
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
                # Store the epoch-mean hinge so postmortem analysis can read it from ckpts.
                repel_hinge_loss=float(train_logs.get("repel_hinge", 0.0)),
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
            # -----------------------------------------------------------------------------

            fail_idx_now = self.dataset.fail_indices()
            gamma_snapshot = _gamma_quantile_diag(self.dataset.gamma_buffer[fail_idx_now])

            entry = {
                "outer_epoch": outer_epoch,
                "bootstrap_k": k,
                **train_logs,
                **gate_diag,
                **{f"gamma_{k2}": v for k2, v in gamma_snapshot.items() if k2.startswith("gamma_")},
                "conditional_delta": cond_delta,
                "repel_weight": float(self.cfg.repel_weight),
                "repel_margin": float(self._repel_margin) if self._repel_margin is not None else 0.0,
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
