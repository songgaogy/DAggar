"""Alpha / kappa / eta schedules for the D4 bootstrap phase.

The coupling
    alpha_k * rho_k = O(1)
from design (C4) is enforced by ``clamped_alpha``: alpha is capped by the
inverse of the held-out r_plus rate (a proxy for critic-convergence rho_k).
Violating this cap is the single largest driver of early-gamma-commitment
failure documented in the implementation plan.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class D4Schedule:
    alpha0: float = 1.0
    alpha_exponent: float = 0.75        # rho in (0.5, 1]
    kappa0: float = 0.0                 # pinned to 0 in bootstrap (design §9.2)
    eta0: float = 0.0                   # pseudo-positive reuse weight, Phase A = 0
    eta_final: float = 0.3
    eta_ramp_epochs: int = 5
    alpha_cap_rate: float = 5.0         # alpha_k * held_out_r_plus < alpha_cap_rate
    # Default bumped 1.0 -> 5.0 after a failed run where alpha was pinned at
    # ~1/held_out_r_plus ≈ 1 and gamma never escaped 0.5. With 5.0 the cap
    # gives alpha enough dynamic range to sharpen once F3 warm-start breaks
    # the symmetric fixed point.

    def alpha(self, k_bootstrap: int) -> float:
        k = max(int(k_bootstrap), 0)
        return float(self.alpha0) * (1.0 + float(k)) ** float(self.alpha_exponent)

    def kappa(self, k_bootstrap: int) -> float:
        # Pinned to 0 in bootstrap; kept as a hook for future schedules.
        return float(self.kappa0)

    def eta(self, k_bootstrap: int) -> float:
        k = max(int(k_bootstrap), 0)
        ramp = max(int(self.eta_ramp_epochs), 1)
        t = min(k / float(ramp), 1.0)
        return float(self.eta0) + (float(self.eta_final) - float(self.eta0)) * t

    def clamped_alpha(self, alpha_raw: float, held_out_r_plus: float) -> float:
        cap = float(self.alpha_cap_rate) / max(float(held_out_r_plus), 1e-6)
        return float(min(float(alpha_raw), cap))

    def halve_alpha0(self) -> None:
        self.alpha0 = float(self.alpha0) * 0.5
