"""Small result type for the frozen nnPU discriminator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class DiscriminatorOutput:
    """Per-sample nnPU scoring output.

    Shapes:
        logit:         failure_score = -g(z); higher = more failure-like.
        prob_failure:  sigmoid(failure_score - calibrated threshold).
        decision:      (B,) bool — predicted "human should intervene".
        metadata:      dict — threshold, normalized margin, etc.
    """

    logit: torch.Tensor
    prob_failure: torch.Tensor
    decision: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)
