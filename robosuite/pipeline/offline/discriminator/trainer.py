"""CUDA-only warm-start training for replay nnPU plus supervised GT BCE."""

from __future__ import annotations

import math
import numbers
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np
import torch

from robosuite.discriminator.dyn_disc.detectors.pu_bce import (
    PUCalibStats,
    PUBCEDiscriminator,
)

from .objectives import NNPUParameters, build_objective


MetricCallback = Callable[[Dict[str, float]], None]


class PUBCEDiscriminatorFT(PUBCEDiscriminator):
    """Pipeline-owned nnPU head updated by a composable warm-start objective."""

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
                    f"Task {task!r}: success calibration is empty; "
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

    @staticmethod
    def _cuda_feature_pools(
        feature_pools: Mapping[str, Sequence[torch.Tensor]],
        *,
        device: torch.device,
        in_dim: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
        cuda_pools: dict[str, torch.Tensor] = {}
        pool_sizes: dict[str, int] = {}
        for name, tensors in feature_pools.items():
            sequences = [tensor.detach() for tensor in tensors if tensor.numel() > 0]
            invalid_shapes = [tuple(tensor.shape) for tensor in sequences if tensor.ndim != 2]
            if invalid_shapes:
                raise ValueError(
                    f"Feature pool {name!r} tensors must have shape (N, D), "
                    f"got {invalid_shapes}."
                )
            pool_sizes[str(name)] = int(sum(int(tensor.shape[0]) for tensor in sequences))
            if not sequences:
                continue
            latent_dims = {int(tensor.shape[-1]) for tensor in sequences}
            if latent_dims != {int(in_dim)}:
                raise ValueError(
                    f"Feature pool {name!r} must have latent dim {in_dim}, "
                    f"got {sorted(latent_dims)}."
                )
            with torch.no_grad():
                cuda_pools[str(name)] = torch.cat(
                    [
                        tensor.to(
                            device=device,
                            dtype=torch.float32,
                            non_blocking=True,
                        ).reshape(-1, in_dim)
                        for tensor in sequences
                    ],
                    dim=0,
                )
        return cuda_pools, pool_sizes

    def _mean_logits(self, features: torch.Tensor, *, batch_size: int = 4096) -> float:
        if features.device.type != "cuda":
            raise ValueError("Full-pool score summaries require CUDA features.")
        was_training = self.head.training
        self.head.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for start in range(0, int(features.shape[0]), int(batch_size)):
                logits = self.head(features[start : start + int(batch_size)])
                total += float(logits.sum().item())
                count += int(logits.numel())
        if was_training:
            self.head.train()
        if count == 0:
            raise ValueError("Cannot summarize logits for an empty feature pool.")
        return total / float(count)

    def finetune(
        self,
        feature_pools: Mapping[str, Sequence[torch.Tensor]],
        success_calib_per_task: Dict[str, Sequence[torch.Tensor]],
        *,
        objective_config: Mapping[str, Any],
        pi_p: float,
        epochs: int = 10,
        lr: float = 3e-5,
        weight_decay: float = 1e-4,
        delta: float = 10.0,
        seed: int = 0,
        loss_surrogate: str = "logistic",
        nn_correction: bool = True,
        beta: float = 0.0,
        log_interval: int = 10,
        verbose: bool = True,
        metric_callback: Optional[MetricCallback] = None,
        step_metric_callback: Optional[MetricCallback] = None,
    ) -> Dict[str, float]:
        """Optimize the joint replay-nnPU and supervised-GT objective."""
        if self.device.type != "cuda":
            raise ValueError(
                f"PUBCEDiscriminatorFT.finetune requires a CUDA device, got {self.device}."
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "PUBCEDiscriminatorFT.finetune requires CUDA, but "
                "torch.cuda.is_available() is False."
            )
        if not 0.0 <= float(delta) <= 100.0:
            raise ValueError(f"delta must be in [0, 100], got {delta}.")
        if int(epochs) < 1 or int(log_interval) < 1:
            raise ValueError("epochs and log_interval must be positive.")
        if float(lr) < 0.0 or float(weight_decay) < 0.0:
            raise ValueError("Learning rate and weight decay must be non-negative.")

        self._delta = float(delta)
        self._pi_p = float(pi_p)
        self._surrogate = str(loss_surrogate)
        self.head = self.head.to(self.device)
        for parameter in self.head.parameters():
            parameter.requires_grad_(True)

        cuda_pools, pool_sizes = self._cuda_feature_pools(
            feature_pools,
            device=self.device,
            in_dim=self.in_dim,
        )
        objective = build_objective(
            objective_config,
            pools=cuda_pools,
            nnpu_parameters=NNPUParameters(
                pi_p=float(pi_p),
                surrogate=str(loss_surrogate),
                nn_correction=bool(nn_correction),
                beta=float(beta),
            ),
            device=self.device,
            seed=int(seed),
        )
        steps_per_epoch = int(objective.steps_per_epoch)
        optimizer = torch.optim.AdamW(
            self.head.parameters(),
            lr=float(lr),
            weight_decay=float(weight_decay),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(epochs) * steps_per_epoch,
            eta_min=0.0,
        )
        torch.cuda.manual_seed_all(int(seed))

        self.head.train()
        self._train_history = []
        global_step = 0
        interval_totals: dict[str, float] = {}
        interval_steps = 0
        for epoch in range(int(epochs)):
            totals: dict[str, float] = {}
            for epoch_step in range(steps_per_epoch):
                optimizer.zero_grad(set_to_none=True)
                result = objective(self.head)
                result.loss.backward()
                optimizer.step()
                scheduler.step()
                global_step += 1

                step_metrics = {
                    key: float(value.detach().item())
                    for key, value in result.metrics.items()
                }
                for key, value in step_metrics.items():
                    totals[key] = totals.get(key, 0.0) + value
                    interval_totals[key] = interval_totals.get(key, 0.0) + value
                interval_steps += 1
                if (
                    step_metric_callback is not None
                    and global_step % int(log_interval) == 0
                ):
                    running_metrics = {
                        key: value / float(interval_steps)
                        for key, value in interval_totals.items()
                    }
                    step_metric_callback(
                        {
                            "epoch": float(epoch),
                            "epoch_step": float(epoch_step + 1),
                            "global_step": float(global_step),
                            "lr": float(optimizer.param_groups[0]["lr"]),
                            **running_metrics,
                        }
                    )
                    interval_totals.clear()
                    interval_steps = 0

            metrics = {
                "epoch": float(epoch),
                "global_step": float(global_step),
                "steps": float(steps_per_epoch),
                "lr": float(optimizer.param_groups[0]["lr"]),
                **{
                    key: value / float(steps_per_epoch)
                    for key, value in totals.items()
                },
                **{
                    f"data/{name}_frames": float(size)
                    for name, size in pool_sizes.items()
                },
            }
            if "nnpu/clamped" in metrics:
                metrics["nnpu/clamp_fraction"] = metrics["nnpu/clamped"]
            if "offline_positive" in cuda_pools and "offline_gt_negative" in cuda_pools:
                positive_mean = self._mean_logits(cuda_pools["offline_positive"])
                negative_mean = self._mean_logits(cuda_pools["offline_gt_negative"])
                metrics["scores/offline_positive_logit_mean"] = positive_mean
                metrics["scores/gt_negative_logit_mean"] = negative_mean
                metrics["scores/delta_gt"] = positive_mean - negative_mean
            self._train_history.append(dict(metrics))
            if metric_callback is not None:
                metric_callback(dict(metrics))
            if verbose:
                print(
                    f"[pu_bce][fit] epoch={epoch + 1}/{int(epochs)} "
                    f"loss={metrics['loss/total']:.5f} "
                    f"nnpu={metrics.get('loss/nnpu_replay/raw', float('nan')):.5f} "
                    f"gt_bce={metrics.get('loss/supervised_gt_bce/raw', float('nan')):.5f} "
                    f"lr={metrics['lr']:.2e}",
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


def _required_parent_value(
    parent_payload: Mapping[str, Any], state: Mapping[str, Any], key: str
) -> Any:
    value = parent_payload.get(key)
    if value is None:
        value = state.get(key)
    if value is None:
        raise KeyError(f"Parent nnPU checkpoint is missing {key}.")
    return value


def resolve_parent_nnpu_semantics(
    parent_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve and strictly validate immutable parent loss/calibration semantics."""
    state = dict(parent_payload.get("pu_bce_detector", {}))
    raw_delta = _required_parent_value(parent_payload, state, "delta")
    if isinstance(raw_delta, bool) or not isinstance(raw_delta, numbers.Real):
        raise TypeError(
            f"Parent nnPU delta must be a real number, got {raw_delta!r}."
        )
    delta = float(raw_delta)
    if not math.isfinite(delta) or not 0.0 <= delta <= 100.0:
        raise ValueError(f"Parent nnPU delta must be in [0, 100], got {raw_delta!r}.")
    raw_pi_p = _required_parent_value(parent_payload, state, "pi_p")
    raw_beta = _required_parent_value(parent_payload, state, "beta")
    for key, value in (("pi_p", raw_pi_p), ("beta", raw_beta)):
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError(
                f"Parent nnPU {key} must be a real number, got {value!r}."
            )
    parameters = NNPUParameters(
        pi_p=raw_pi_p,
        surrogate=_required_parent_value(parent_payload, state, "loss_surrogate"),
        nn_correction=_required_parent_value(
            parent_payload, state, "nn_correction"
        ),
        beta=raw_beta,
    )
    return {
        "pi_p": parameters.pi_p,
        "loss_surrogate": parameters.surrogate,
        "nn_correction": parameters.nn_correction,
        "beta": parameters.beta,
        "delta": delta,
    }


def finetune_warmstart_detector(
    detector: PUBCEDiscriminatorFT,
    *,
    feature_pools: Mapping[str, Sequence[torch.Tensor]],
    calibration_features: Sequence[torch.Tensor],
    objective_config: Mapping[str, Any],
    task_name: str,
    parent_payload: Mapping[str, Any],
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
    log_interval: int,
    metric_callback: MetricCallback | None = None,
    step_metric_callback: MetricCallback | None = None,
    verbose: bool = True,
) -> dict[str, float]:
    """Warm-start with immutable parent nnPU semantics and the configured objectives."""
    semantics = resolve_parent_nnpu_semantics(parent_payload)
    return detector.finetune(
        feature_pools,
        {str(task_name): list(calibration_features)},
        objective_config=objective_config,
        pi_p=float(semantics["pi_p"]),
        epochs=int(epochs),
        lr=float(lr),
        weight_decay=float(weight_decay),
        delta=float(semantics["delta"]),
        seed=int(seed),
        loss_surrogate=str(semantics["loss_surrogate"]),
        nn_correction=semantics["nn_correction"],
        beta=float(semantics["beta"]),
        log_interval=int(log_interval),
        verbose=bool(verbose),
        metric_callback=metric_callback,
        step_metric_callback=step_metric_callback,
    )


__all__ = [
    "PUBCEDiscriminatorFT",
    "finetune_warmstart_detector",
    "require_cuda_device",
    "resolve_parent_nnpu_semantics",
]
