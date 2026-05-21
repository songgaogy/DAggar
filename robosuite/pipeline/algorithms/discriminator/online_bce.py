"""Online BCE discriminator with frozen encoder + trainable head.

Replaces the frozen `LPBV2GProvider` in RL mode. The encoder is INJECTED
(see `SharedFrozenEncoder`) — this class never owns or trains it.

Input to the head:
    head_in = concat(context, flatten(action_chunk))    # (B, D_ctx + H*D_a)
    logit = head(head_in)                                # (B,)

Sign convention (matches lpb_v2 BCE warm-start):
    higher logit = more FAILURE-like (intervention needed)
    label = 1    = intervention chunk      (positive class)
    label = 0    = expert demo / non-intervention rollout (negative class)

Threshold for the `decision` field is maintained as an EMA over recent
mini-batches (default percentile = 50th of non-failure-pool logits).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch.optim import AdamW

from .base import DiscriminatorBase, DiscriminatorBatch, DiscriminatorOutput
from .bce_head import TrainableBCEHead
from .losses import bce_with_logits_loss

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
        update_every_n_steps:  trainer skips disc.update for N-1 of every
                               N learner ticks (throughput knob).
        threshold_ema:         EMA factor for the decision threshold.
        warm_start_ckpt:       optional path to pre-fitted bce_head.pth.
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
    update_every_n_steps: int = 1
    threshold_ema: float = 0.99
    warm_start_ckpt: str | None = None
    label_smoothing: float = 0.0
    grad_clip_norm: float = 1.0
    device: str = "cuda:1"


class OnlineBCEDiscriminator(DiscriminatorBase):
    """Online-trainable BCE discriminator on top of a frozen encoder.

    Args:
        cfg:          DiscriminatorConfig.
        encoder:      SharedFrozenEncoder — encode is called externally;
                      this object reads encoder.context_dim only.
        context_dim:  D_ctx (must equal encoder.context_dim; passed
                      explicitly to allow encoder lazy initialization).
        action_dim:   per-step action dim D_a.
        action_horizon: H.
    """

    def __init__(
        self,
        cfg: DiscriminatorConfig,
        encoder: "SharedFrozenEncoder",
        context_dim: int,
        action_dim: int,
        action_horizon: int,
    ) -> None:
        if context_dim <= 0:
            raise ValueError(f"context_dim must be positive, got {context_dim}")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
        if action_horizon <= 0:
            raise ValueError(f"action_horizon must be positive, got {action_horizon}")

        self.cfg = cfg
        self._encoder_ref = encoder
        self.context_dim = int(context_dim)
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)

        in_dim = self.context_dim + self.action_dim * self.action_horizon
        self.head = TrainableBCEHead(
            in_dim=in_dim, hidden=cfg.hidden, num_layers=cfg.num_layers
        ).to(cfg.device)
        self.optim = AdamW(
            self.head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.threshold: float = 0.0
        self._step: int = 0

        if cfg.warm_start_ckpt:
            self.warm_start_from_lpb_bce_ckpt(cfg.warm_start_ckpt)
        self.head.eval()

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #

    def _device(self) -> torch.device:
        return next(self.head.parameters()).device

    def _featurize(
        self, context: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        if context.dim() != 2 or context.shape[-1] != self.context_dim:
            raise ValueError(
                f"context must be (B, {self.context_dim}); got {tuple(context.shape)}"
            )
        if action_chunk.dim() != 3:
            raise ValueError(
                f"action_chunk must be (B, H, D_a); got {tuple(action_chunk.shape)}"
            )
        if action_chunk.shape[1] != self.action_horizon:
            raise ValueError(
                f"action_chunk H mismatch: head expects {self.action_horizon}, "
                f"got {action_chunk.shape[1]}"
            )
        if action_chunk.shape[2] != self.action_dim:
            raise ValueError(
                f"action_chunk D_a mismatch: head expects {self.action_dim}, "
                f"got {action_chunk.shape[2]}"
            )
        device = self._device()
        dtype = next(self.head.parameters()).dtype
        ctx = context.to(device=device, dtype=dtype)
        ach = action_chunk.to(device=device, dtype=dtype).flatten(start_dim=1)
        return torch.cat([ctx, ach], dim=-1)

    # ------------------------------------------------------------------ #
    # Inference                                                           #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def score(
        self,
        *,
        context: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> DiscriminatorOutput:
        head_in = self._featurize(context, action_chunk)
        was_training = self.head.training
        self.head.eval()
        logit = self.head(head_in)
        if was_training:
            self.head.train()
        prob_failure = torch.sigmoid(logit)
        threshold_tensor = torch.full_like(logit, float(self.threshold))
        decision = logit > threshold_tensor
        metadata: dict[str, Any] = {
            "threshold": float(self.threshold),
            "normalized_margin": (logit - threshold_tensor).abs(),
            "prediction": decision.to(torch.long),
        }
        return DiscriminatorOutput(
            logit=logit,
            prob_failure=prob_failure,
            decision=decision,
            metadata=metadata,
        )

    @torch.no_grad()
    def intrinsic_reward(
        self,
        *,
        context: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        head_in = self._featurize(context, action_chunk)
        was_training = self.head.training
        self.head.eval()
        logit = self.head(head_in)
        if was_training:
            self.head.train()
        return logit

    # ------------------------------------------------------------------ #
    # Training                                                            #
    # ------------------------------------------------------------------ #

    def update(self, batch: DiscriminatorBatch) -> dict[str, float]:
        head_in = self._featurize(batch.context, batch.action_chunk)
        labels = batch.label.to(device=head_in.device, dtype=head_in.dtype).view(-1)

        self.head.train()
        logit = self.head(head_in)
        loss = bce_with_logits_loss(
            logit, labels, label_smoothing=self.cfg.label_smoothing
        )
        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        if self.cfg.grad_clip_norm and self.cfg.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.head.parameters(), max_norm=float(self.cfg.grad_clip_norm)
            )
        self.optim.step()
        self.head.eval()

        detached = logit.detach()
        failure_mask = labels > 0.5
        non_failure_mask = ~failure_mask

        non_failure_logits = detached[non_failure_mask]
        if non_failure_logits.numel() > 0:
            new_thr = float(non_failure_logits.quantile(0.5).item())
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
        preds = (detached > self.threshold).to(dtype=labels.dtype)
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
            "action_horizon": int(self.action_horizon),
        }

    def load_state_dict(self, sd: dict, strict: bool = True) -> None:
        for key in ("context_dim", "action_dim", "action_horizon"):
            saved = int(sd[key])
            current = int(getattr(self, key))
            if saved != current:
                raise ValueError(
                    f"OnlineBCEDiscriminator.load_state_dict: '{key}' mismatch "
                    f"(saved={saved}, current={current})."
                )
        self.head.load_state_dict(sd["head"], strict=strict)
        self.optim.load_state_dict(sd["optim"])
        self.threshold = float(sd.get("threshold", 0.0))
        self._step = int(sd.get("step", 0))
        self.head.eval()

    def warm_start_from_lpb_bce_ckpt(self, ckpt_path: str) -> None:
        """Convenience pass-through to `TrainableBCEHead.warm_start_*`."""
        self.head.warm_start_from_lpb_bce_ckpt(ckpt_path)
