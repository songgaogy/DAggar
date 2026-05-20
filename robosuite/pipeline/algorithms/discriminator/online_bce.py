"""Online BCE discriminator with frozen encoder + trainable head.

Replaces the frozen `LPBV2GProvider` in RL mode. The encoder is INJECTED
(see `SharedFrozenEncoder`) — this class never owns or trains it.

Input to the head:
    head_in = concat(context, flatten(action_chunk))    # (B, D_ctx + H·D_a)
    logit = head(head_in)                                # (B,)
Higher logit = more expert-like.

Threshold for the `decision` field is maintained as an EMA over recent
mini-batches (default percentile = 50th of policy-pool logits).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .base import DiscriminatorBase, DiscriminatorBatch, DiscriminatorOutput

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
        self.cfg = cfg
        self._encoder_ref = encoder
        self.context_dim = context_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        # In implementation:
        #   self.head = TrainableBCEHead(in_dim=context_dim + action_dim*H,
        #                                hidden=cfg.hidden, num_layers=cfg.num_layers)
        #   self.optim = AdamW(head.params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        #   if cfg.warm_start_ckpt: head.warm_start_from_lpb_bce_ckpt(...)
        #   self.threshold: float = 0.0          # EMA-tracked
        self.head = None
        self.optim = None
        self.threshold: float = 0.0

    # ------------------------------------------------------------------ #
    # Inference (DiscriminatorBase)                                       #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def score(
        self,
        *,
        context: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> DiscriminatorOutput:
        """Compute logit + prob + decision under current threshold."""
        raise NotImplementedError

    @torch.no_grad()
    def intrinsic_reward(
        self,
        *,
        context: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        """Returns (B,) reward. Default sign: r = +logit (higher = expert);
        Q-learner config decides whether to take the negation."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Training                                                            #
    # ------------------------------------------------------------------ #

    def update(self, batch: DiscriminatorBatch) -> dict[str, float]:
        """One BCEWithLogitsLoss step on the head. Updates EMA threshold
        from this batch's logits. Returns metrics:
            disc_loss, disc_acc, expert_logit_mean, policy_logit_mean,
            threshold.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Persistence                                                         #
    # ------------------------------------------------------------------ #

    def state_dict(self) -> dict:
        raise NotImplementedError

    def load_state_dict(self, sd: dict, strict: bool = True) -> None:
        raise NotImplementedError

    def warm_start_from_lpb_bce_ckpt(self, ckpt_path: str) -> None:
        """Convenience pass-through to `TrainableBCEHead.warm_start_*`."""
        raise NotImplementedError
