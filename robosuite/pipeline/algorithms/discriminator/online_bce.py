"""Online BCE discriminator with frozen encoder + trainable single-frame head.

Replaces the frozen `LPBV2GProvider` in RL mode. The encoder is INJECTED
(see `SharedFrozenEncoder`) — this class never owns or trains it.

Single-frame design (aligned with the lpb v2 reference BCE detector at
`robosuite/discriminator/lpb_v2/detectors/bce.py`):

    head input:  context (B, D_ctx)        # per-frame frozen latent
    logit     :  head(context)             # (B,)

The encoder already absorbs the (image, proprio, real_action) concat
internally — see `SharedFrozenEncoder.encode(..., action_real=...)`. The
head only consumes the resulting latent, matching how the lpb v2 BCE
head was trained.

Sign convention (matches ``robosuite/discriminator/lpb_v2/detectors/bce.py``):

    g(z)            = head(z)           # expert-likeness; higher = more expert
    failure_score   = -g(z)             # higher = more failure-like
    tau             = ``bce_youden_threshold`` on failure_score (from meta.json)
    pred_failure    iff failure_score >= tau

``intrinsic_reward`` returns ``-sigmoid(failure_score - tau)`` ∈ (-1, 0):
expert-like frames (low failure_score) → ~0; failure-like → ~-1.

Online ``update()`` maps replay labels (1 = intervention/failure) to LPB
BCE targets via ``expert_target = 1 - label``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch.optim import AdamW

from .base import DiscriminatorBase, DiscriminatorBatch, DiscriminatorOutput
from .bce_head import TrainableBCEHead, read_lpb_bce_head_hparams
from .losses import bce_with_logits_loss
from .lpb_v2_scorer import load_bce_youden_threshold

if TYPE_CHECKING:
    from .encoder import SharedFrozenEncoder


@dataclass
class DiscriminatorConfig:
    """Config for the online BCE discriminator.

    Fields:
        lr:                    AdamW learning rate for the head.
        weight_decay:          AdamW weight decay.
        hidden:                head hidden width.
        num_layers:            head depth.
        batch_size:            disc update batch size.
        balance_ratio:         expert : policy sample ratio (default 1:1).
        update_freq:           BCE head updates per learner tick (each resamples).
        threshold_ema:         EMA factor for the decision threshold.
        warm_start_ckpt:       optional path to pre-fitted bce_head.pth.
        initial_threshold:     operational tau; if None, read from sibling
                               meta.json when warm_start_ckpt is set.
        meta_json_path:        optional override for meta.json location.
        label_smoothing:       BCE label smoothing in [0, 0.5).
        grad_clip_norm:        max norm for head grad clipping.
        device:                cuda:1 by convention.
    """

    lr: float = 3e-4
    weight_decay: float = 1e-6
    hidden: int = 256
    num_layers: int = 2
    batch_size: int = 64
    balance_ratio: float = 1.0
    update_freq: int = 1
    threshold_ema: float = 0.99
    warm_start_ckpt: str | None = None
    initial_threshold: float | None = None
    meta_json_path: str | None = None
    label_smoothing: float = 0.0
    grad_clip_norm: float = 1.0
    device: str = "cuda:1"


class OnlineBCEDiscriminator(DiscriminatorBase):
    """Online-trainable single-frame BCE discriminator on top of a frozen encoder.

    Args:
        cfg:          DiscriminatorConfig.
        encoder:      SharedFrozenEncoder — encode is called externally;
                      this object reads encoder.context_dim only.
        context_dim:  D_ctx (must equal encoder.context_dim; passed
                      explicitly to allow encoder lazy initialization).
        action_dim:   per-step action dim D_a. Retained as metadata for
                      checkpoint compatibility / future extensions; the
                      single-frame head does NOT consume the action — it
                      enters the latent via the encoder.
    """

    def __init__(
        self,
        cfg: DiscriminatorConfig,
        encoder: "SharedFrozenEncoder",
        context_dim: int,
        action_dim: int,
    ) -> None:
        if context_dim <= 0:
            raise ValueError(f"context_dim must be positive, got {context_dim}")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")

        self.cfg = cfg
        self._encoder_ref = encoder
        self.context_dim = int(context_dim)
        self.action_dim = int(action_dim)

        head_hidden = int(cfg.hidden)
        head_layers = int(cfg.num_layers)
        if cfg.warm_start_ckpt:
            ckpt_hparams = read_lpb_bce_head_hparams(cfg.warm_start_ckpt)
            if int(ckpt_hparams["in_dim"]) != self.context_dim:
                raise ValueError(
                    "OnlineBCEDiscriminator: encoder.context_dim "
                    f"({self.context_dim}) != warm_start ckpt in_dim "
                    f"({ckpt_hparams['in_dim']})."
                )
            head_hidden = int(ckpt_hparams["hidden"])
            head_layers = int(ckpt_hparams["num_layers"])

        in_dim = self.context_dim
        self.head = TrainableBCEHead(
            in_dim=in_dim, hidden=head_hidden, num_layers=head_layers
        ).to(cfg.device)
        self.optim = AdamW(
            self.head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.threshold: float = 0.0
        self.threshold_source: str = "unset"
        self._step: int = 0

        if cfg.warm_start_ckpt:
            self.warm_start_from_lpb_bce_ckpt(cfg.warm_start_ckpt)
        self._init_operational_threshold()
        self.head.eval()

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #

    def _device(self) -> torch.device:
        return next(self.head.parameters()).device

    def _init_operational_threshold(self) -> None:
        """Seed decision threshold from config or ``meta.json`` (Youden tau).

        Online ``update()`` continues to EMA-track this value from the
        non-failure logit pool; warmup keeps the head frozen but still uses
        this tau inside ``intrinsic_reward``.
        """
        if self.cfg.initial_threshold is not None:
            self.threshold = float(self.cfg.initial_threshold)
            self.threshold_source = "config.initial_threshold"
            return
        if not self.cfg.warm_start_ckpt:
            self.threshold_source = "default_zero"
            return
        ckpt_path = Path(str(self.cfg.warm_start_ckpt)).resolve()
        meta_path = (
            Path(str(self.cfg.meta_json_path)).resolve()
            if self.cfg.meta_json_path
            else None
        )
        tau, used = load_bce_youden_threshold(
            ckpt_path,
            meta_json_path=meta_path,
            required=True,
        )
        self.threshold = float(tau)
        self.threshold_source = f"meta_json({used})"

    def _featurize(self, context: torch.Tensor) -> torch.Tensor:
        """Validate + cast a per-frame context tensor."""
        if context.dim() != 2 or context.shape[-1] != self.context_dim:
            raise ValueError(
                f"context must be (B, {self.context_dim}); got {tuple(context.shape)}"
            )
        device = self._device()
        dtype = next(self.head.parameters()).dtype
        return context.to(device=device, dtype=dtype)

    @staticmethod
    def _failure_score(expert_logit: torch.Tensor) -> torch.Tensor:
        """LPB step score: ``failure_score = -g(z)`` (higher = more failure)."""
        return -expert_logit

    # ------------------------------------------------------------------ #
    # Inference                                                           #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def score(
        self,
        *,
        context: torch.Tensor,
    ) -> DiscriminatorOutput:
        head_in = self._featurize(context)
        was_training = self.head.training
        self.head.eval()
        expert_logit = self.head(head_in)
        if was_training:
            self.head.train()
        failure_score = self._failure_score(expert_logit)
        threshold_tensor = torch.full_like(failure_score, float(self.threshold))
        decision = failure_score >= threshold_tensor
        metadata: dict[str, Any] = {
            "threshold": float(self.threshold),
            "expert_logit": expert_logit.detach(),
            "failure_score": failure_score.detach(),
            "normalized_margin": (failure_score - threshold_tensor).abs(),
            "prediction": decision.to(torch.long),
        }
        return DiscriminatorOutput(
            logit=expert_logit,
            prob_failure=torch.sigmoid(failure_score - threshold_tensor),
            decision=decision,
            metadata=metadata,
        )

    @torch.no_grad()
    def intrinsic_reward(
        self,
        *,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Per-frame intrinsic reward in (-1, 0).

        ``r_disc = -sigmoid(failure_score - tau)`` with
        ``failure_score = -expert_logit`` (LPB / ``visualize_bce`` convention).
        """
        head_in = self._featurize(context)
        was_training = self.head.training
        self.head.eval()
        expert_logit = self.head(head_in)
        if was_training:
            self.head.train()
        failure_score = self._failure_score(expert_logit)
        threshold = torch.tensor(
            float(self.threshold), dtype=failure_score.dtype, device=failure_score.device
        )
        return -torch.sigmoid(failure_score - threshold)

    # ------------------------------------------------------------------ #
    # Training                                                            #
    # ------------------------------------------------------------------ #

    def update(self, batch: DiscriminatorBatch) -> dict[str, float]:
        head_in = self._featurize(batch.context)
        labels = batch.label.to(device=head_in.device, dtype=head_in.dtype).view(-1)
        # Replay: 1 = failure/intervention. LPB head trains with 1 = expert.
        expert_targets = 1.0 - labels

        self.head.train()
        expert_logit = self.head(head_in)
        loss = bce_with_logits_loss(
            expert_logit, expert_targets, label_smoothing=self.cfg.label_smoothing
        )
        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        if self.cfg.grad_clip_norm and self.cfg.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.head.parameters(), max_norm=float(self.cfg.grad_clip_norm)
            )
        self.optim.step()
        self.head.eval()

        detached = expert_logit.detach()
        failure_mask = labels > 0.5
        non_failure_mask = ~failure_mask
        failure_score = self._failure_score(detached)

        non_failure_scores = failure_score[non_failure_mask]
        if non_failure_scores.numel() > 0:
            new_thr = float(non_failure_scores.quantile(0.5).item())
            ema = float(self.cfg.threshold_ema)
            self.threshold = ema * self.threshold + (1.0 - ema) * new_thr

        self._step += 1

        failure_logit_mean = (
            float(detached[failure_mask].mean().item()) if failure_mask.any() else 0.0
        )
        non_failure_logit_mean = (
            float(detached[non_failure_mask].mean().item())
            if non_failure_mask.any()
            else 0.0
        )
        preds = (failure_score >= float(self.threshold)).to(dtype=labels.dtype)
        targets = (labels > 0.5).to(dtype=labels.dtype)
        disc_acc = float((preds == targets).to(dtype=labels.dtype).mean().item())

        return {
            "disc_loss": float(loss.item()),
            "disc_acc": disc_acc,
            "failure_logit_mean": failure_logit_mean,
            "non_failure_logit_mean": non_failure_logit_mean,
            "threshold": float(self.threshold),
        }

    # ------------------------------------------------------------------ #
    # Persistence                                                         #
    # ------------------------------------------------------------------ #

    def state_dict(self) -> dict:
        return {
            "head": self.head.state_dict(),
            "optim": self.optim.state_dict(),
            "threshold": float(self.threshold),
            "step": int(self._step),
            "cfg": asdict(self.cfg),
            "context_dim": int(self.context_dim),
            "action_dim": int(self.action_dim),
        }

    def load_state_dict(self, sd: dict, strict: bool = True) -> None:
        for key in ("context_dim", "action_dim"):
            saved = int(sd[key])
            current = int(getattr(self, key))
            if saved != current:
                raise ValueError(
                    f"OnlineBCEDiscriminator.load_state_dict: '{key}' mismatch "
                    f"(saved={saved}, current={current})."
                )
        # Backward-compat: older checkpoints carried `action_horizon`; the
        # single-frame head no longer uses it, so silently ignore.
        self.head.load_state_dict(sd["head"], strict=strict)
        self.optim.load_state_dict(sd["optim"])
        self.threshold = float(sd.get("threshold", 0.0))
        self._step = int(sd.get("step", 0))
        self.head.eval()

    def warm_start_from_lpb_bce_ckpt(self, ckpt_path: str) -> None:
        """Convenience pass-through to `TrainableBCEHead.warm_start_*`.

        With the head's `in_dim` now matching the lpb v2 ckpt's `D_ctx`,
        warm-start is expected to load EVERY layer including the first
        Linear. The strict check lives in the head itself.
        """
        self.head.warm_start_from_lpb_bce_ckpt(ckpt_path)
