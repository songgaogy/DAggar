"""CUDA-only warm-start training for replay nnPU plus separate GT risks."""

from __future__ import annotations

import math
import numbers
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import torch

from robosuite.discriminator.dyn_disc.detectors.pu_bce import (
    PUCalibStats,
    PUBCEDiscriminator,
)

from .objectives import FixedLogitNormalizer, NNPUParameters, build_objective


MetricCallback = Callable[[Dict[str, float]], None]


def compute_fixed_robust_normalizer(
    head: torch.nn.Module,
    calibration_features: Sequence[torch.Tensor],
    *,
    center: float,
    device: torch.device,
    batch_size: int = 4096,
) -> FixedLogitNormalizer:
    """Estimate a fixed IQR scale from held-out success logits on CUDA."""
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Fixed robust logit normalization requires CUDA.")
    if not math.isfinite(float(center)):
        raise ValueError(f"Normalizer center must be finite, got {center!r}.")
    if int(batch_size) < 1:
        raise ValueError("Normalizer batch_size must be positive.")
    sequences = [tensor.detach() for tensor in calibration_features if tensor.numel() > 0]
    if not sequences:
        raise ValueError("Held-out success calibration features are empty.")
    was_training = head.training
    head.eval()
    logits: list[torch.Tensor] = []
    with torch.no_grad():
        for sequence in sequences:
            if sequence.ndim != 2:
                raise ValueError(
                    "Calibration feature tensors must have shape (N, D), got "
                    f"{tuple(sequence.shape)}."
                )
            features = sequence.to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            for start in range(0, int(features.shape[0]), int(batch_size)):
                batch_logits = head(features[start : start + int(batch_size)]).reshape(-1)
                if not bool(torch.isfinite(batch_logits).all().item()):
                    raise FloatingPointError("Calibration logits contain non-finite values.")
                logits.append(batch_logits)
    if was_training:
        head.train()
    all_logits = torch.cat(logits, dim=0)
    quartiles = torch.quantile(
        all_logits,
        torch.tensor([0.25, 0.75], device=device, dtype=all_logits.dtype),
    )
    scale_tensor = (quartiles[1] - quartiles[0]) / 1.349
    if not bool(torch.isfinite(scale_tensor).item()) or not bool(scale_tensor > 0.0):
        raise ValueError(
            "Held-out success calibration produced a non-positive or non-finite "
            f"robust scale: {float(scale_tensor.item())!r}."
        )
    return FixedLogitNormalizer(
        enabled=True,
        center=float(center),
        scale=float(scale_tensor.item()),
    )


@torch.no_grad()
def fold_fixed_logit_normalizer_(
    head: torch.nn.Module,
    normalizer: FixedLogitNormalizer,
) -> None:
    """Fold ``(g-center)/scale`` into the final Linear and logit center."""
    if not normalizer.enabled:
        return
    linear_layers = [module for module in head.modules() if isinstance(module, torch.nn.Linear)]
    if not linear_layers:
        raise TypeError("Cannot fold logit normalization: head has no Linear layer.")
    final_linear = linear_layers[-1]
    if final_linear.out_features != 1 or final_linear.bias is None:
        raise TypeError(
            "Cannot fold logit normalization: final Linear must have one output and bias."
        )
    final_linear.weight.div_(normalizer.scale)
    final_linear.bias.div_(normalizer.scale)
    if not hasattr(head, "logit_center") or not hasattr(head, "set_logit_center"):
        final_linear.bias.sub_(normalizer.center / normalizer.scale)
        return
    old_center = getattr(head, "logit_center")
    head.set_logit_center((old_center + normalizer.center) / normalizer.scale)


class PUBCEDiscriminatorFT(PUBCEDiscriminator):
    """Pipeline-owned nnPU head updated by a composable warm-start objective."""

    def state_dict(self) -> Dict[str, object]:
        state = super().state_dict()
        state.update(
            {
                "logit_normalization": dict(
                    getattr(
                        self,
                        "_logit_normalization",
                        {"enabled": False, "folded": False},
                    )
                ),
                "quadratic_cap_c": getattr(self, "_quadratic_cap_c", None),
                "quadratic_cap_lambda": float(
                    getattr(self, "_quadratic_cap_lambda", 0.0)
                ),
            }
        )
        return state

    def load_state_dict(self, state: Dict[str, object]) -> None:
        super().load_state_dict(state)
        raw_normalization = state.get("logit_normalization", {})
        self._logit_normalization = (
            dict(raw_normalization)
            if isinstance(raw_normalization, Mapping)
            else {"enabled": False, "folded": False}
        )

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
            with torch.no_grad():
                failure_score = -self.head(calib_features).reshape(-1)
                if not bool(torch.isfinite(failure_score).all().item()):
                    raise FloatingPointError(
                        f"Task {task!r}: calibration scores contain non-finite values."
                    )
                threshold_tensor = torch.quantile(
                    failure_score,
                    failure_score.new_tensor(quantile / 100.0),
                )
            threshold = float(threshold_tensor.item())
            self.thresholds[str(task)] = threshold
            self.calib_stats[str(task)] = PUCalibStats(
                threshold=threshold,
                num_calib_frames=int(failure_score.numel()),
                calib_score_min=float(failure_score.min().item()),
                calib_score_max=float(failure_score.max().item()),
                calib_score_mean=float(failure_score.mean().item()),
                calib_score_std=float(failure_score.std(unbiased=False).item()),
            )
            if verbose:
                print(
                    f"[pu_bce][calib] task={task} tau={threshold:.5f} "
                    f"n_calib={failure_score.numel()}",
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

    def _logit_health(
        self,
        features: torch.Tensor,
        *,
        normalizer: FixedLogitNormalizer,
        batch_size: int = 4096,
    ) -> dict[str, float]:
        if features.device.type != "cuda":
            raise ValueError("Full-pool score summaries require CUDA features.")
        was_training = self.head.training
        self.head.eval()
        raw_parts: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, int(features.shape[0]), int(batch_size)):
                raw_parts.append(
                    self.head(features[start : start + int(batch_size)]).reshape(-1)
                )
        if was_training:
            self.head.train()
        if not raw_parts:
            raise ValueError("Cannot summarize logits for an empty feature pool.")
        raw = torch.cat(raw_parts)
        effective = normalizer(raw)
        if not bool(torch.isfinite(raw).all().item()) or not bool(
            torch.isfinite(effective).all().item()
        ):
            raise FloatingPointError("Full-pool logit summary contains non-finite values.")
        return {
            "raw_mean": float(raw.mean().item()),
            "raw_abs_p99": float(torch.quantile(raw.abs(), 0.99).item()),
            "raw_saturation_fraction": float((raw.abs() > 9.21).float().mean().item()),
            "effective_mean": float(effective.mean().item()),
            "effective_abs_p99": float(torch.quantile(effective.abs(), 0.99).item()),
            "effective_saturation_fraction": float(
                (effective.abs() > 9.21).float().mean().item()
            ),
        }

    def finetune(
        self,
        feature_pools: Mapping[str, Sequence[torch.Tensor]],
        success_calib_per_task: Dict[str, Sequence[torch.Tensor]],
        *,
        objective_config: Mapping[str, Any],
        positive_safety_boundary: float | None = None,
        logit_normalization_center: float | None = None,
        pi_p: float,
        epochs: int = 10,
        scheduler_horizon_epochs: int = 20,
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
        """Optimize the joint replay-nnPU and separate supervised-GT objective."""
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
        if int(scheduler_horizon_epochs) < int(epochs):
            raise ValueError(
                "scheduler_horizon_epochs must be >= epochs, got "
                f"{scheduler_horizon_epochs} < {epochs}."
            )
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
        raw_normalization = objective_config.get("logit_normalization", {})
        if raw_normalization is None:
            raw_normalization = {}
        if not isinstance(raw_normalization, Mapping):
            raise TypeError("objective.logit_normalization must be a mapping or null.")
        unknown_normalization = set(raw_normalization) - {"enabled", "method"}
        if unknown_normalization:
            raise ValueError(
                "Unknown objective.logit_normalization options: "
                f"{sorted(unknown_normalization)}."
            )
        normalization_enabled = raw_normalization.get("enabled", False)
        if not isinstance(normalization_enabled, bool):
            raise TypeError("objective.logit_normalization.enabled must be bool.")
        normalization_method = str(raw_normalization.get("method", "fixed_robust_iqr"))
        if normalization_method != "fixed_robust_iqr":
            raise ValueError(
                "objective.logit_normalization.method must be 'fixed_robust_iqr', "
                f"got {normalization_method!r}."
            )
        if normalization_enabled:
            if logit_normalization_center is None:
                raise ValueError(
                    "logit_normalization_center is required when fixed robust "
                    "normalization is enabled."
                )
            normalizer = compute_fixed_robust_normalizer(
                self.head,
                [item for items in success_calib_per_task.values() for item in items],
                center=float(logit_normalization_center),
                device=self.device,
            )
        else:
            normalizer = FixedLogitNormalizer()
        self._logit_normalization = {
            "enabled": bool(normalizer.enabled),
            "method": normalization_method,
            "center": float(normalizer.center),
            "scale": float(normalizer.scale),
            "folded": False,
        }
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
            positive_safety_boundary=positive_safety_boundary,
            logit_normalizer=normalizer,
        )
        self._scheduler_horizon_epochs = int(scheduler_horizon_epochs)
        self._completed_epochs = 0
        self._completed_steps = 0
        if objective.quadratic_logit_cap is not None:
            cap_config = objective.quadratic_logit_cap
            self._quadratic_cap_c = float(cap_config.cap)
            self._quadratic_cap_lambda = (
                float(cap_config.weight) if cap_config.enabled else 0.0
            )
        else:
            self._quadratic_cap_c = None
            self._quadratic_cap_lambda = 0.0
        steps_per_epoch = int(objective.steps_per_epoch)
        optimizer = torch.optim.AdamW(
            self.head.parameters(),
            lr=float(lr),
            weight_decay=float(weight_decay),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(scheduler_horizon_epochs) * steps_per_epoch,
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
                if not bool(torch.isfinite(result.loss).item()):
                    raise FloatingPointError(
                        f"Non-finite discriminator loss at epoch={epoch + 1}, "
                        f"step={epoch_step + 1}."
                    )
                result.loss.backward()
                optimizer.step()
                scheduler.step()
                global_step += 1
                self._completed_steps = int(global_step)

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
                "scheduler/horizon_epochs": float(scheduler_horizon_epochs),
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
            metrics.update(
                {
                    "normalization/enabled": float(normalizer.enabled),
                    "normalization/center": float(normalizer.center),
                    "normalization/scale": float(normalizer.scale),
                }
            )
            pool_health: dict[str, dict[str, float]] = {}
            for pool_name, pool_features in cuda_pools.items():
                pool_health[pool_name] = self._logit_health(
                    pool_features,
                    normalizer=normalizer,
                )
                metrics.update(
                    {
                        f"health/{pool_name}/{key}": value
                        for key, value in pool_health[pool_name].items()
                    }
                )
            if "offline_positive" in cuda_pools and "offline_gt_negative" in cuda_pools:
                positive_health = pool_health["offline_positive"]
                negative_health = pool_health["offline_gt_negative"]
                metrics.update(
                    {
                        f"scores/offline_positive_{key}": value
                        for key, value in positive_health.items()
                    }
                )
                metrics.update(
                    {
                        f"scores/gt_negative_{key}": value
                        for key, value in negative_health.items()
                    }
                )
                metrics["scores/delta_gt"] = (
                    positive_health["effective_mean"]
                    - negative_health["effective_mean"]
                )
            self._train_history.append(dict(metrics))
            self._completed_epochs = int(epoch + 1)
            if metric_callback is not None:
                metric_callback(dict(metrics))
            if verbose:
                positive_bce = metrics.get(
                    "gt/positive_bce",
                    metrics.get("gt/positive_logistic", float("nan")),
                )
                print(
                    f"[pu_bce][fit] epoch={epoch + 1}/{int(epochs)} "
                    f"loss={metrics['loss/total']:.5f} "
                    f"nnpu={metrics.get('loss/nnpu_replay/raw', float('nan')):.5f}/"
                    f"{metrics.get('loss/nnpu_replay/weighted', float('nan')):.5f} "
                    f"gt_p={metrics.get('loss/gt_positive/raw', float('nan')):.5f}/"
                    f"{metrics.get('loss/gt_positive/weighted', float('nan')):.5f} "
                    f"p_bce={positive_bce:.5f} "
                    f"p_safe={metrics.get('gt/positive_safety_margin', float('nan')):.5f} "
                    f"p_violate={metrics.get('safety/margin_violation_fraction', float('nan')):.3f} "
                    f"gt_n={metrics.get('loss/gt_negative/raw', float('nan')):.5f}/"
                    f"{metrics.get('loss/gt_negative/weighted', float('nan')):.5f} "
                    f"cap={metrics.get('regularization/quadratic_logit_cap', 0.0):.5f}/"
                    f"{metrics.get('regularization/quadratic_logit_cap_weighted', 0.0):.5f} "
                    f"batch={int(metrics.get('batch/pretrain_positive', 0.0))}/"
                    f"{int(metrics.get('batch/pretrain_unlabeled', 0.0))}/"
                    f"{int(metrics.get('batch/offline_positive', 0.0))}/"
                    f"{int(metrics.get('batch/offline_gt_negative', 0.0))} "
                    f"lr={metrics['lr']:.2e}",
                    flush=True,
                )

        fold_fixed_logit_normalizer_(self.head, normalizer)
        self._logit_normalization["folded"] = bool(normalizer.enabled)
        self.head.eval()
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


def resolve_parent_success_boundary(
    parent_payload: Mapping[str, Any], task_name: str
) -> float:
    """Return the parent success-logit boundary ``m = -tau_task``."""
    state = parent_payload.get("pu_bce_detector", {})
    if not isinstance(state, Mapping):
        raise TypeError("Parent pu_bce_detector state must be a mapping.")
    thresholds = state.get("thresholds", parent_payload.get("thresholds", {}))
    if not isinstance(thresholds, Mapping) or str(task_name) not in thresholds:
        raise KeyError(
            f"Parent nnPU checkpoint has no threshold for task {task_name!r}."
        )
    threshold = thresholds[str(task_name)]
    if isinstance(threshold, bool) or not isinstance(threshold, numbers.Real):
        raise TypeError(
            f"Parent threshold for task {task_name!r} must be real, got {threshold!r}."
        )
    threshold = float(threshold)
    if not math.isfinite(threshold):
        raise ValueError(
            f"Parent threshold for task {task_name!r} must be finite, got {threshold!r}."
        )
    return -threshold


def finetune_warmstart_detector(
    detector: PUBCEDiscriminatorFT,
    *,
    feature_pools: Mapping[str, Sequence[torch.Tensor]],
    calibration_features: Sequence[torch.Tensor],
    objective_config: Mapping[str, Any],
    positive_safety_boundary: float | None = None,
    task_name: str,
    parent_payload: Mapping[str, Any],
    epochs: int,
    scheduler_horizon_epochs: int,
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
    normalization_config = objective_config.get("logit_normalization", {})
    normalization_enabled = bool(
        isinstance(normalization_config, Mapping)
        and normalization_config.get("enabled", False)
    )
    normalization_center = (
        resolve_parent_success_boundary(parent_payload, str(task_name))
        if normalization_enabled
        else None
    )
    return detector.finetune(
        feature_pools,
        {str(task_name): list(calibration_features)},
        objective_config=objective_config,
        positive_safety_boundary=positive_safety_boundary,
        logit_normalization_center=normalization_center,
        pi_p=float(semantics["pi_p"]),
        epochs=int(epochs),
        scheduler_horizon_epochs=int(scheduler_horizon_epochs),
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
    "compute_fixed_robust_normalizer",
    "finetune_warmstart_detector",
    "fold_fixed_logit_normalizer_",
    "require_cuda_device",
    "resolve_parent_nnpu_semantics",
    "resolve_parent_success_boundary",
]
