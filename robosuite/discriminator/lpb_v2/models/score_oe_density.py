"""Outlier-exposed (OE) density score on top of the frozen WAM latent.

This module is the *primary* method committed in
``PLAN_score_based_discriminator.md`` §4.A--§4.I and originally specified in
``PROMPT_oe_density.md`` §3--§4.2.

**Loss history (read NOTES_oe_density_phase2.md for the full story).**

1. Original PROMPT: hinge ``L_fail = lam * relu(log p_theta(z_f) - m)`` with
   ``m`` set to the succ-side quantile each epoch. Saturated to zero after
   one epoch (run_20260512_205026), reducing the model to PCA-on-success.

2. Anti-density penalty: ``L_fail = lam * mean log p_theta(z_f)``. Always
   carries gradient, but at lam = 1 the total loss
   ``L_total = -mean log p_succ + mean log p_fail = log(p_fail / p_succ)``
   is **mathematically unbounded** under a Gaussian: moving mu along the
   discriminative axis drives ``q_fail >> q_succ`` without limit. The
   W1-row-normalisation patch only blocked the W1-scale flat direction;
   mu and the off-diagonal of L remained unconstrained. End-to-end run
   (run_20260512_211335) diverged to L_total ~ -2e21 by epoch 200.

3. **Current form (2026-05-12): logistic discriminative loss on the score.**
   Let ``s(z) = -log p_theta(f(z))``. The objective is binary logistic
   regression with a learnable threshold ``tau``:

       L_succ = mean( softplus( s(z_s) - tau ) )      # wants s_s < tau
       L_fail = lam * mean( softplus( tau - s(z_f) ) ) # wants s_f > tau
       L_total = L_succ + L_fail + weight_decay * ||W1||_F^2

   Each softplus term is bounded below by 0 and saturates as the gap
   ``|s - tau|`` grows in the correct direction, so the loss can no
   longer escape to -infinity. This is exactly the §4.J #2 / PROMPT §7
   fallback (low-rank logistic on the score), with the same (W1, b1,
   mu, L_raw) parameterisation as the density form so the test-time
   ``score()`` and the calibration adapter are unchanged.

The module defines:

* a linear projection ``f(z_tilde) = W1 z_tilde + b1`` with ``W1 in R^{k x (D+T)}``;
* a parametric density ``p_theta`` on f-space (default: full-covariance Gaussian
  parameterised by Cholesky factor ``L`` so ``Sigma = L L^T`` is positive
  definite);
* a learnable threshold ``tau`` and the logistic loss above.

Test-time score is ``s(z) = -log p_theta(f(z_tilde))`` -- larger means more
failure-like (matches the ``score >= tau`` convention used by
``LPBV2KNN.score`` and ``TwoBankKNN.score``). Note that the per-task
threshold calibrated by the benchmark adapter is *not* the same as
``self.tau``; the model's ``tau`` is a training-time logistic-regression
parameter, while the adapter computes a per-task percentile or Youden
threshold on calibration scores.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


_VALID_DENSITIES = ("gaussian", "gmm2", "gmm4")


class OEDensityScore(nn.Module):
    """Linear projection + Gaussian score + logistic discriminative loss.

    Trainable parameters:
      * ``W1`` -- linear projection rows, shape ``(score_k, in_dim)``.
      * ``b1`` -- projection bias, shape ``(score_k,)``.
      * ``mu`` -- Gaussian mean on f-space, shape ``(score_k,)``.
      * ``L_raw`` -- packed Cholesky factor (full ``k x k`` storage; strict
        lower triangle is taken verbatim, diag goes through ``softplus + 1e-2``
        to stay strictly positive). Upper triangle is ignored at forward time.
      * ``tau`` -- learnable scalar threshold used by the logistic loss.

    Adapter responsibilities (task one-hot concat, per-task threshold
    calibration on held-out succ scores) are external to this module.
    """

    def __init__(
        self,
        in_dim: int,
        score_k: int = 16,
        density: str = "gaussian",
        weight_decay: float = 1e-4,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        if density not in _VALID_DENSITIES:
            raise ValueError(
                f"density must be one of {_VALID_DENSITIES}, got {density!r}"
            )
        if density != "gaussian":
            raise NotImplementedError(
                f"density={density!r} is reserved for Phase 3 (PROMPT §3.2 "
                "escalation path); not implemented in Phase 2 Priority A."
            )
        if int(score_k) <= 0 or int(in_dim) <= 0:
            raise ValueError(f"score_k and in_dim must be positive (got {score_k}, {in_dim})")

        self.in_dim = int(in_dim)
        self.score_k = int(score_k)
        self.density = str(density)
        self.weight_decay = float(weight_decay)
        # Cache the requested device string; PyTorch handles cuda-availability fallback elsewhere.
        self.device_str = str(device)

        # Projection: small init scale to keep the initial Gaussian fit well-conditioned.
        self.W1 = nn.Parameter(torch.empty(self.score_k, self.in_dim))
        nn.init.normal_(self.W1, mean=0.0, std=1.0 / math.sqrt(float(self.in_dim)))
        self.b1 = nn.Parameter(torch.zeros(self.score_k))

        # Gaussian parameters. mu starts at 0 and L_raw at identity; both are
        # overwritten by :meth:`init_density_from_batch` before training.
        self.mu = nn.Parameter(torch.zeros(self.score_k))
        self.L_raw = nn.Parameter(torch.eye(self.score_k))

        # Learnable logistic-loss threshold. Set by init_density_from_batch
        # to the median of s(z_succ) so the loss starts in its linear regime.
        self.tau = nn.Parameter(torch.zeros(()))

        target = torch.device(
            self.device_str
            if (self.device_str != "cuda" or torch.cuda.is_available())
            else "cpu"
        )
        self.to(target)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _build_L(self) -> torch.Tensor:
        """Reconstruct lower-triangular Cholesky factor ``L`` from ``L_raw``.

        ``L`` is strictly lower-triangular off-diagonal (taken from ``L_raw``)
        plus a positive diagonal (``softplus(diag(L_raw)) + 1e-4``). This makes
        ``Sigma = L L^T`` positive-definite for any value of ``L_raw``.
        """
        L_raw = self.L_raw
        strict_lower = torch.tril(L_raw, diagonal=-1)
        # Floor diag(L) at 1e-2 (not 1e-4): with the W1-normalised forward, the
        # remaining flat direction in the loss is shrinking Sigma toward 0.
        # A modest floor keeps log_det / quad numerics in a sane range without
        # affecting the discriminative ordering.
        diag_positive = F.softplus(torch.diagonal(L_raw)) + 1e-2
        return strict_lower + torch.diag(diag_positive)

    def _normalized_W1(self) -> torch.Tensor:
        """Return ``W1`` with each row L2-normalised to unit norm.

        Anchors the projection scale so the (W1, Sigma) parameterisation has a
        unique optimum. Without this, the joint loss has a flat direction
        (``W1 -> alpha W1``, ``Sigma -> alpha^2 Sigma``) at lam = 1, and the
        optimiser drifts to ``||W1|| -> infinity`` / ``Sigma -> 0`` without
        actually aligning the projection with the discriminative direction
        (observed empirically on the first OE-density run, 2026-05-12).
        """
        norms = self.W1.norm(dim=1, keepdim=True).clamp(min=1e-6)
        return self.W1 / norms

    @staticmethod
    def _inv_softplus(x: torch.Tensor) -> torch.Tensor:
        """Inverse of ``softplus``: ``y = log(exp(x) - 1)`` (numerically stable)."""
        x = x.clamp(min=1e-6)
        return torch.log(torch.expm1(x))

    # ------------------------------------------------------------------ #
    # Forward / density                                                  #
    # ------------------------------------------------------------------ #

    def forward(self, z_tilde: torch.Tensor) -> torch.Tensor:
        return z_tilde @ self._normalized_W1().T + self.b1

    def log_p(self, z_tilde: torch.Tensor) -> torch.Tensor:
        """Return ``log p_theta(f(z_tilde))`` per PROMPT §3.2.

        Uses ``||L^{-1} (u - mu)||^2`` for the quadratic form so we never form
        ``Sigma^{-1}`` explicitly. ``log|Sigma| = 2 sum log diag(L)``.
        """
        u = self.forward(z_tilde)  # (N, k)
        L = self._build_L()  # (k, k) lower-triangular, positive diag
        diff = (u - self.mu).unsqueeze(-1)  # (N, k, 1)
        y = torch.linalg.solve_triangular(L, diff, upper=False).squeeze(-1)  # (N, k)
        quad = (y * y).sum(dim=-1)  # (N,)
        log_det_sigma = 2.0 * torch.log(torch.diagonal(L)).sum()
        k = float(self.score_k)
        log_2pi = math.log(2.0 * math.pi)
        return -0.5 * quad - 0.5 * k * log_2pi - 0.5 * log_det_sigma

    @torch.no_grad()
    def init_density_from_batch(self, z_tilde_succ: torch.Tensor) -> None:
        """One-shot initialisation of ``(mu, L)`` from empirical moments of ``f(z_tilde_succ)``.

        Adds a small diagonal shrinkage (``1e-3 * I``) for numerical safety
        before factoring. After this call, ``mu`` equals the empirical mean and
        ``L`` Cholesky-factorises the shrunk empirical covariance.
        """
        if z_tilde_succ.dim() != 2 or z_tilde_succ.shape[1] != self.in_dim:
            raise ValueError(
                f"expected z_tilde_succ of shape (N, {self.in_dim}), got {tuple(z_tilde_succ.shape)}"
            )
        z = z_tilde_succ.to(self.W1.device, dtype=self.W1.dtype)
        u = self.forward(z)  # (N, k)
        if u.shape[0] < 2:
            raise ValueError(
                "init_density_from_batch requires at least 2 samples to estimate covariance"
            )
        mu_emp = u.mean(dim=0)
        centered = u - mu_emp
        sigma_emp = (centered.T @ centered) / float(u.shape[0] - 1)
        sigma_emp = sigma_emp + 1e-3 * torch.eye(
            self.score_k, device=u.device, dtype=u.dtype
        )
        L_emp = torch.linalg.cholesky(sigma_emp)

        new_L_raw = torch.zeros_like(self.L_raw)
        # Strict lower triangle: copy verbatim.
        tri_mask = torch.tril(torch.ones_like(self.L_raw), diagonal=-1).bool()
        new_L_raw[tri_mask] = L_emp[tri_mask]
        # Diagonal: invert the softplus + diag_floor transform so _build_L()
        # reproduces L_emp. diag_floor must match _build_L().
        diag_emp = torch.diagonal(L_emp)
        raw_diag = self._inv_softplus((diag_emp - 1e-2).clamp(min=1e-6))
        idx = torch.arange(self.score_k, device=u.device)
        new_L_raw[idx, idx] = raw_diag

        self.L_raw.copy_(new_L_raw)
        self.mu.copy_(mu_emp)

        # Seed tau at the median of s(z_succ_train) so the logistic loss starts
        # in its linear regime (succ samples straddle tau roughly 50/50 at init).
        # We use a no-grad recompute of log_p with the freshly-set (mu, L).
        u = z.detach() @ self._normalized_W1().T + self.b1
        L = self._build_L()
        diff = (u - self.mu).unsqueeze(-1)
        y = torch.linalg.solve_triangular(L, diff, upper=False).squeeze(-1)
        quad = (y * y).sum(dim=-1)
        log_det_sigma = 2.0 * torch.log(torch.diagonal(L)).sum()
        k = float(self.score_k)
        log_2pi = math.log(2.0 * math.pi)
        log_p_succ = -0.5 * quad - 0.5 * k * log_2pi - 0.5 * log_det_sigma
        s_succ = -log_p_succ
        self.tau.copy_(s_succ.median())

    # ------------------------------------------------------------------ #
    # Loss                                                               #
    # ------------------------------------------------------------------ #

    def loss(
        self,
        z_tilde_succ: torch.Tensor,
        z_tilde_fail: Optional[torch.Tensor],
        lam: float,
    ) -> Dict[str, torch.Tensor]:
        """Logistic discriminative loss on the score (see module docstring).

        Let ``s(z) = -log p_theta(f(z))`` (larger = more failure-like).

            L_succ = mean softplus( s(z_s) - tau )
            L_fail = lam * mean softplus( tau - s(z_f) )
            L_reg  = weight_decay * ||W1||_F^2
            L_total = L_succ + L_fail + L_reg

        Each softplus term is bounded below by 0 and saturates as the data
        is correctly classified; this replaces the (unbounded) anti-density
        log-ratio loss that diverged on real data (NOTES §4 / module docstring).

        When ``lam == 0`` the fail forward pass is skipped (zero gradient from
        fail data); ``z_tilde_fail`` is ignored. ``lam > 0`` requires a
        non-None ``z_tilde_fail``.

        Returns a dict with keys ``total, succ, fail, reg`` (loss components)
        and diagnostics ``score_succ_mean, score_fail_mean, tau``.
        """
        log_p_succ = self.log_p(z_tilde_succ)
        s_succ = -log_p_succ
        L_succ = F.softplus(s_succ - self.tau).mean()

        if float(lam) == 0.0:
            L_fail = torch.zeros((), device=L_succ.device, dtype=L_succ.dtype)
            s_fail_mean = torch.zeros((), device=L_succ.device, dtype=L_succ.dtype)
        else:
            if z_tilde_fail is None:
                raise ValueError("loss(...) with lam > 0 requires non-None z_tilde_fail")
            log_p_fail = self.log_p(z_tilde_fail)
            s_fail = -log_p_fail
            L_fail = float(lam) * F.softplus(self.tau - s_fail).mean()
            s_fail_mean = s_fail.detach().mean()

        L_reg = self.weight_decay * (self.W1 ** 2).sum()
        L_total = L_succ + L_fail + L_reg
        return {
            "total": L_total,
            "succ": L_succ,
            "fail": L_fail,
            "reg": L_reg,
            "score_succ_mean": s_succ.detach().mean(),
            "score_fail_mean": s_fail_mean,
            "tau": self.tau.detach(),
        }

    @torch.no_grad()
    def score(self, z_tilde: torch.Tensor) -> torch.Tensor:
        """Test-time score: ``s(z) = -log p_theta(f(z_tilde))``.

        Larger = more failure-like (matches ``LPBV2KNN.score`` / ``TwoBankKNN.score``).
        """
        return -self.log_p(z_tilde)
