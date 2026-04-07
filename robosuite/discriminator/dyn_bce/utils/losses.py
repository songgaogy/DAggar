from __future__ import annotations

import torch
import torch.nn.functional as F


def weighted_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sample_weights: torch.Tensor,
) -> torch.Tensor:
    logits = logits.reshape(-1)
    targets = targets.reshape(-1).to(dtype=logits.dtype)
    sample_weights = sample_weights.reshape(-1).to(dtype=logits.dtype)
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    weighted = loss * sample_weights
    return weighted.sum() / sample_weights.sum().clamp_min(1e-6)


def _masked_weighted_mean(
    values: torch.Tensor,
    weights: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor | None:
    mask = mask.reshape(-1).to(dtype=torch.bool)
    if int(mask.sum().item()) <= 0:
        return None
    values = values.reshape(-1)
    weights = weights.reshape(-1).to(dtype=values.dtype)
    sel_values = values[mask]
    sel_weights = weights[mask]
    return (sel_values * sel_weights).sum() / sel_weights.sum().clamp_min(1e-6)


def occupancy_pu_loss(
    logits: torch.Tensor,
    data_type_index: torch.Tensor,
    sample_weights: torch.Tensor,
    positive_prior: float,
    unlabeled_index: int = 2,
    nnpu: bool = True,
    return_details: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logits = logits.reshape(-1)
    data_type_index = data_type_index.reshape(-1)
    sample_weights = sample_weights.reshape(-1).to(dtype=logits.dtype)

    positive_mask = data_type_index != int(unlabeled_index)
    unlabeled_mask = data_type_index == int(unlabeled_index)

    # For nnPU, the fail-rollout unlabeled pool should reflect the raw empirical marginal.
    # BCE loss
    positive_loss = F.binary_cross_entropy_with_logits(
        logits,
        torch.ones_like(logits),
        reduction="none",
    )
    negative_loss = F.binary_cross_entropy_with_logits(
        logits,
        torch.zeros_like(logits),
        reduction="none",
    )

    positive_term = _masked_weighted_mean(positive_loss, sample_weights, positive_mask)
    negative_from_positive = _masked_weighted_mean(negative_loss, sample_weights, positive_mask)
    negative_from_unlabeled = _masked_weighted_mean(negative_loss, sample_weights, unlabeled_mask)

    zero = logits.new_zeros(())
    prior = float(positive_prior)
    positive_term_value = positive_term if positive_term is not None else zero
    negative_from_positive_value = negative_from_positive if negative_from_positive is not None else zero
    negative_from_unlabeled_value = negative_from_unlabeled if negative_from_unlabeled is not None else zero

    # deal with degenerate batch cases
    if positive_term is None and negative_from_unlabeled is None:
        negative_risk_before_clamp = zero
        negative_risk_after_clamp = zero
        loss = zero
    elif positive_term is None:
        negative_risk_before_clamp = negative_from_unlabeled_value
        negative_risk_after_clamp = negative_from_unlabeled_value
        loss = negative_from_unlabeled_value  # pragma: no cover
    elif negative_from_unlabeled is None:
        negative_risk_before_clamp = zero
        negative_risk_after_clamp = zero
        loss = prior * positive_term_value
    else:
        negative_risk_before_clamp = negative_from_unlabeled_value - prior * negative_from_positive_value
        negative_risk_after_clamp = negative_risk_before_clamp

        # clamp, avoid predicting all positive
        if bool(nnpu):
            negative_risk_after_clamp = torch.clamp(negative_risk_after_clamp, min=0.0)
            
        loss = prior * positive_term_value + negative_risk_after_clamp

    if not bool(return_details):
        return loss

    details = {
        "positive_term": positive_term_value.detach(),
        "negative_from_positive": negative_from_positive_value.detach(),
        "negative_from_unlabeled": negative_from_unlabeled_value.detach(),
        "negative_risk_before_clamp": negative_risk_before_clamp.detach(),
        "negative_risk_after_clamp": negative_risk_after_clamp.detach(),
        "nnpu_clamped": (negative_risk_before_clamp.detach() < 0.0).to(dtype=logits.dtype),
    }
    return loss, details


def beta_nll_loss(
    pred_mean: torch.Tensor,
    pred_logvar: torch.Tensor,
    target: torch.Tensor,
    sample_weights: torch.Tensor,
    beta: float = 0.5,
) -> torch.Tensor:
    if pred_mean.ndim != 3:
        raise ValueError(f"pred_mean must be (M,B,D), got {pred_mean.shape}")
    if pred_logvar.shape != pred_mean.shape:
        raise ValueError("pred_logvar must match pred_mean shape")
    if target.ndim != 2:
        raise ValueError(f"target must be (B,D), got {target.shape}")

    target_expanded = target.unsqueeze(0).expand_as(pred_mean)
    inv_var = torch.exp(-pred_logvar)
    squared_error = (pred_mean - target_expanded) ** 2
    per_dim = squared_error * inv_var + pred_logvar
    if float(beta) > 0.0:
        per_dim = per_dim * torch.exp(pred_logvar.detach() * float(beta))
    per_sample = per_dim.mean(dim=-1).mean(dim=0)
    weights = sample_weights.reshape(-1).to(dtype=per_sample.dtype)
    return (per_sample * weights).sum() / weights.sum().clamp_min(1e-6)


def cross_covariance_penalty(
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("cross_covariance_penalty expects 2D tensors")
    if left.shape[0] != right.shape[0]:
        raise ValueError("left/right batch dimensions must match")
    if left.shape[0] <= 1:
        return left.new_zeros(())

    left_centered = left - left.mean(dim=0, keepdim=True)
    right_centered = right - right.mean(dim=0, keepdim=True)
    left_norm = left_centered / left_centered.std(dim=0, keepdim=True).clamp_min(1e-6)
    right_norm = right_centered / right_centered.std(dim=0, keepdim=True).clamp_min(1e-6)
    covariance = left_norm.transpose(0, 1) @ right_norm / float(left.shape[0] - 1)
    return covariance.square().mean()
