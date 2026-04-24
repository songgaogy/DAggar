"""Collapse / divergence detector for D4 bootstrap training.

Implements the pathology checks from implementation plan §7:

    (a) early commitment:    mean_gamma <= 0.02 or >= 0.98
    (b) conditional collapse: ||f(c=+) - f(c=-)||^2 < 1e-4
    (c) critic divergence:    held_out_r_plus monotonically up for 3+ checks
    (d) branch starvation:    separation_gap < 0 for 2+ checks

On trigger the trainer reverts to the last EMA checkpoint and halves
``schedule.alpha0``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Deque, List, Optional

from collections import deque


@dataclass
class D4Health:
    held_out_r_plus: float = float("nan")
    mean_gamma: float = 0.5
    separation_gap: float = 0.0
    conditional_delta: float = 0.0
    alpha_used: float = 0.0
    kappa_used: float = 0.0
    eta_used: float = 0.0
    mean_r_plus: float = 0.0
    mean_r_minus: float = 0.0
    repel_hinge_loss: float = 0.0
    step: int = 0
    epoch: int = 0


@dataclass
class CollapseDetector:
    patience: int = 3
    commit_low: float = 0.02
    commit_high: float = 0.98
    conditional_floor: float = 1e-4
    separation_floor: float = 0.0
    history_size: int = 8

    _commit_counter: int = 0
    _collapse_counter: int = 0
    _separation_counter: int = 0
    _rp_hist: Deque[float] = field(default_factory=lambda: deque(maxlen=8))

    def update(self, health: D4Health) -> Optional[str]:
        # (a) Early commitment.
        if health.mean_gamma <= self.commit_low or health.mean_gamma >= self.commit_high:
            self._commit_counter += 1
        else:
            self._commit_counter = 0
        if self._commit_counter >= self.patience:
            return "early_commitment"

        # (b) Conditional collapse.
        if health.conditional_delta < self.conditional_floor:
            self._collapse_counter += 1
        else:
            self._collapse_counter = 0
        if self._collapse_counter >= self.patience + 2:
            return "conditional_collapse"

        # (c) Critic divergence: monotonically increasing held_out_r_plus.
        self._rp_hist.append(float(health.held_out_r_plus))
        if len(self._rp_hist) >= 4:
            tail: List[float] = list(self._rp_hist)[-4:]
            if all((b > a) for a, b in zip(tail[:-1], tail[1:])):
                return "critic_divergence"

        # (d) Branch starvation: separation_gap < floor for multiple checks.
        if health.separation_gap < self.separation_floor:
            self._separation_counter += 1
        else:
            self._separation_counter = 0
        if self._separation_counter >= 2:
            return "branch_starvation"

        return None

    def reset_counters(self) -> None:
        self._commit_counter = 0
        self._collapse_counter = 0
        self._separation_counter = 0
        self._rp_hist.clear()
