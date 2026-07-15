"""CUDA-only warm-start training for an offline nnPU head."""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import numpy as np
import torch

from robosuite.discriminator.dyn_disc.detectors.pu_bce import (
    PUCalibStats,
    PUBCEDiscriminator,
    pu_risk,
)


class PUBCEDiscriminatorFT(PUBCEDiscriminator):
    """Pipeline-owned nnPU head that preserves loaded weights during finetuning."""

    def _nnpu_optimization_step(
        self,
        positive_logits: torch.Tensor,
        unlabeled_logits: torch.Tensor,
        *,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        pi_p: float,
        nn_correction: bool,
        beta: float,
    ) -> Dict[str, torch.Tensor]:
        parts = pu_risk(
            positive_logits,
            unlabeled_logits,
            pi_p=float(pi_p),
            surrogate=self._surrogate,
            nn_correction=bool(nn_correction),
            beta=float(beta),
        )
        optimizer.zero_grad(set_to_none=True)
        parts["risk"].backward()
        optimizer.step()
        scheduler.step()
        return parts

    def _calibrate_success_thresholds(
        self,
        success_calib_per_task: Dict[str, Sequence[torch.Tensor]],
        *,
        delta: float,
        verbose: bool,
    ) -> Dict[str, float]:
        self.head.eval()
        self.thresholds = {}
        self.calib_stats = {}
        quantile = 100.0 * (1.0 - float(delta) / 100.0)
        for task, calib_seqs in success_calib_per_task.items():
            seqs = [tensor for tensor in calib_seqs if tensor.numel() > 0]
            if not seqs:
                raise ValueError(
                    f"Task {task!r}: success_calib_per_task[{task!r}] is empty; "
                    "cannot calibrate threshold."
                )
            with torch.no_grad():
                calib_features = torch.cat(
                    [
                        tensor.detach()
                        .to(self.device, dtype=torch.float32, non_blocking=True)
                        .reshape(-1, tensor.shape[-1])
                        for tensor in seqs
                    ],
                    dim=0,
                )
            failure_score = -self._logits_np(calib_features)
            threshold = float(
                np.percentile(failure_score.astype(np.float64), q=quantile)
            )
            self.thresholds[str(task)] = threshold
            self.calib_stats[str(task)] = PUCalibStats(
                threshold=threshold,
                num_calib_frames=int(failure_score.size),
                calib_score_min=float(failure_score.min()),
                calib_score_max=float(failure_score.max()),
                calib_score_mean=float(failure_score.mean()),
                calib_score_std=float(failure_score.std()),
            )
            if verbose:
                print(
                    f"[pu_bce][calib] task={task} tau={threshold:.5f} "
                    f"n_calib={failure_score.size}",
                    flush=True,
                )
        return dict(self.thresholds)

    def finetune(
        self,
        positive_features: Sequence[torch.Tensor],
        unlabeled_features: Sequence[torch.Tensor],
        success_calib_per_task: Dict[str, Sequence[torch.Tensor]],
        *,
        pi_p: float,
        epochs: int = 10,
        lr: float = 3e-5,
        weight_decay: float = 1e-4,
        batch_size: int = 512,
        delta: float = 10.0,
        seed: int = 0,
        loss_surrogate: str = "logistic",
        nn_correction: bool = True,
        beta: float = 0.0,
        verbose: bool = True,
        metric_callback: Optional[Callable[[Dict[str, float]], None]] = None,
    ) -> Dict[str, float]:
        """Warm-start nnPU optimization without reinitializing the head."""
        if self.device.type != "cuda":
            raise ValueError(
                f"PUBCEDiscriminatorFT.finetune requires a CUDA device, got {self.device}."
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "PUBCEDiscriminatorFT.finetune requires CUDA, but "
                "torch.cuda.is_available() is False."
            )
        if delta < 0.0 or delta > 100.0:
            raise ValueError(f"delta must be in [0, 100], got {delta}")
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}")
        if batch_size < 2:
            raise ValueError(f"batch_size must be >= 2, got {batch_size}")
        if int(batch_size) % 2 != 0:
            raise ValueError(
                "batch_size must be even for exact 50/50 P/U sampling, "
                f"got {batch_size}"
            )
        if not (0.0 < float(pi_p) < 1.0):
            raise ValueError(f"pi_p (class prior) must be in (0, 1), got {pi_p}")
        if loss_surrogate not in ("sigmoid", "logistic"):
            raise ValueError(
                "loss_surrogate must be 'sigmoid' or 'logistic', "
                f"got {loss_surrogate!r}"
            )

        p_seqs = [tensor.detach() for tensor in positive_features if tensor.numel() > 0]
        u_seqs = [tensor.detach() for tensor in unlabeled_features if tensor.numel() > 0]
        if not p_seqs:
            raise ValueError("positive_features is empty after filtering")
        if not u_seqs:
            raise ValueError("unlabeled_features is empty after filtering")

        with torch.no_grad():
            z_p = torch.cat(
                [
                    tensor.to(
                        self.device, dtype=torch.float32, non_blocking=True
                    ).reshape(-1, tensor.shape[-1])
                    for tensor in p_seqs
                ],
                dim=0,
            )
            z_u = torch.cat(
                [
                    tensor.to(
                        self.device, dtype=torch.float32, non_blocking=True
                    ).reshape(-1, tensor.shape[-1])
                    for tensor in u_seqs
                ],
                dim=0,
            )
        if z_p.shape[1] != self.in_dim or z_u.shape[1] != self.in_dim:
            raise ValueError(
                f"feature dim mismatch: in_dim={self.in_dim}, "
                f"P={z_p.shape[1]}, U={z_u.shape[1]}"
            )

        self._delta = float(delta)
        self._pi_p = float(pi_p)
        self._surrogate = str(loss_surrogate)
        self.head = self.head.to(self.device)

        optimizer = torch.optim.AdamW(
            self.head.parameters(),
            lr=float(lr),
            weight_decay=float(weight_decay),
        )
        n_p = int(z_p.shape[0])
        n_u = int(z_u.shape[0])
        steps_per_epoch = max(1, int(2 * (n_p + n_u) // int(batch_size)))
        total_steps = int(epochs) * steps_per_epoch
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=0.0,
        )
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
        positive_batch_size = int(batch_size) // 2
        unlabeled_batch_size = positive_batch_size

        self.head.train()
        self._train_history = []
        for epoch in range(int(epochs)):
            totals = {
                "risk": 0.0,
                "pos_risk": 0.0,
                "neg_risk": 0.0,
                "neg_risk_used": 0.0,
                "nn_correction_batches": 0.0,
            }
            for _ in range(steps_per_epoch):
                p_indices = torch.randint(
                    n_p,
                    (positive_batch_size,),
                    device=self.device,
                    generator=generator,
                )
                u_indices = torch.randint(
                    n_u,
                    (unlabeled_batch_size,),
                    device=self.device,
                    generator=generator,
                )
                parts = self._nnpu_optimization_step(
                    self.head(z_p[p_indices]),
                    self.head(z_u[u_indices]),
                    optimizer=optimizer,
                    scheduler=scheduler,
                    pi_p=float(pi_p),
                    nn_correction=bool(nn_correction),
                    beta=float(beta),
                )
                totals["risk"] += float(parts["risk"].detach().item())
                totals["pos_risk"] += float(parts["pos_risk"].item())
                totals["neg_risk"] += float(parts["neg_risk"].item())
                totals["neg_risk_used"] += float(parts["neg_risk_used"].item())
                if float(parts["neg_risk"].item()) < float(-beta):
                    totals["nn_correction_batches"] += 1.0

            metrics = {
                "epoch": float(epoch),
                "risk": totals["risk"] / float(steps_per_epoch),
                "pos_risk": totals["pos_risk"] / float(steps_per_epoch),
                "neg_risk": totals["neg_risk"] / float(steps_per_epoch),
                "neg_risk_used": totals["neg_risk_used"] / float(steps_per_epoch),
                "nn_correction_batches": float(totals["nn_correction_batches"]),
                "nn_correction_fraction": (
                    totals["nn_correction_batches"] / float(steps_per_epoch)
                ),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "steps": float(steps_per_epoch),
                "num_positive_frames": float(n_p),
                "num_unlabeled_frames": float(n_u),
            }
            self._train_history.append(dict(metrics))
            if metric_callback is not None:
                metric_callback(dict(metrics))
            if verbose:
                nn_corr_batches = int(metrics["nn_correction_batches"])
                nn_corr_text = f"{nn_corr_batches}/{int(metrics['steps'])}"
                print(
                    f"[pu_bce][fit] epoch={epoch + 1}/{int(epochs)} "
                    f"risk={metrics['risk']:.5f} neg_risk={metrics['neg_risk']:+.5f} "
                    f"nn_corr_batches={nn_corr_text} "
                    f"lr={metrics['lr']:.2e} Np={n_p} Nu={n_u} "
                    f"pi_p={float(pi_p):.3f} surrogate={self._surrogate} "
                    f"nn_correction={bool(nn_correction)} beta={float(beta):.3g}",
                    flush=True,
                )

        return self._calibrate_success_thresholds(
            success_calib_per_task,
            delta=float(delta),
            verbose=bool(verbose),
        )


def require_cuda_device(device: str | torch.device) -> torch.device:
    """Resolve an available CUDA device without permitting CPU fallback."""
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError(f"Discriminator finetuning requires CUDA, got {device!r}.")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Discriminator finetuning requires CUDA, but torch.cuda.is_available() is False."
        )
    if resolved.index is not None and resolved.index >= torch.cuda.device_count():
        raise ValueError(
            f"CUDA device {resolved} is unavailable; visible device count is "
            f"{torch.cuda.device_count()}."
        )
    return resolved


def finetune_warmstart_detector(
    detector: PUBCEDiscriminatorFT,
    *,
    positive_features: Sequence[torch.Tensor],
    unlabeled_features: Sequence[torch.Tensor],
    calibration_features: Sequence[torch.Tensor],
    task_name: str,
    parent_payload: dict,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    seed: int,
    metric_callback: Callable[[dict[str, float]], None] | None = None,
    verbose: bool = True,
) -> dict[str, float]:
    """Continue nnPU training with checkpoint loss semantics and fresh optimizer."""
    state = dict(parent_payload.get("pu_bce_detector", {}))
    pi_p_raw = parent_payload.get("pi_p")
    if pi_p_raw is None:
        pi_p_raw = state.get("pi_p")
    if pi_p_raw is None:
        raise KeyError("Parent nnPU checkpoint is missing pi_p.")
    delta_raw = parent_payload.get("delta")
    if delta_raw is None:
        delta_raw = state.get("delta")
    if delta_raw is None:
        raise KeyError("Parent nnPU checkpoint is missing calibration delta.")
    surrogate = str(
        parent_payload.get("loss_surrogate")
        if parent_payload.get("loss_surrogate") is not None
        else state.get("loss_surrogate", "logistic")
    )
    nn_correction = parent_payload.get("nn_correction")
    if nn_correction is None:
        nn_correction = state.get("nn_correction", True)
    beta_raw = parent_payload.get("beta")
    if beta_raw is None:
        beta_raw = state.get("beta", 0.0)
    beta = float(beta_raw)
    return detector.finetune(
        positive_features=positive_features,
        unlabeled_features=unlabeled_features,
        success_calib_per_task={str(task_name): list(calibration_features)},
        pi_p=float(pi_p_raw),
        epochs=int(epochs),
        lr=float(lr),
        weight_decay=float(weight_decay),
        batch_size=int(batch_size),
        delta=float(delta_raw),
        seed=int(seed),
        loss_surrogate=surrogate,
        nn_correction=nn_correction,
        beta=beta,
        verbose=bool(verbose),
        metric_callback=metric_callback,
    )


__all__ = [
    "PUBCEDiscriminatorFT",
    "finetune_warmstart_detector",
    "require_cuda_device",
]
