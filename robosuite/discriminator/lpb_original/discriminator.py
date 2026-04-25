"""Faithful port of the original LPB KNN OOD discriminator.

Mirrors `tmp/lpb-main/dyn_model/planner_libero.py:compute_nn_reward` exactly:

  feature_t  = [encoder(o_t) ; proprio_encoder(s_t)]
               then per-dim weighting: visual * visual_weight, proprio * proprio_weight
  score_t    = -min_{j} ||feature_t - bank_j||_2     (non-squared L2, k=1)

For the benchmark we add a thin classification layer on top of `score_t`:
  - Calibrate a percentile threshold `tau` on a disjoint set of success demos:
        tau = np.percentile(calibration min_dist values, 100 - delta)
  - Predict failure at frame t when  min_dist_t  >=  tau.

This matches the original LPB reward semantics (lower min_dist = more
in-distribution = lower failure score) while letting us emit binary preds.

The dynamics model that produces the encoder + proprio_encoder is loaded
from a checkpoint via `dyn_model.plan.load_model`.
"""

from __future__ import annotations

# Trigger sys.path injection so `import dyn_model` and `import diffusion_policy`
# resolve to the vendored copies.
import robosuite.discriminator.lpb_original  # noqa: F401

import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf

from dyn_model.plan import load_model

from diffusion_policy.model.common.normalizer import LinearNormalizer


# --------------------------------------------------------------------------- #
# KNN distance helper                                                         #
# --------------------------------------------------------------------------- #


@torch.no_grad()
def knn_min_l2_dist(
    query: torch.Tensor,
    bank: torch.Tensor,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """Chunked nearest-neighbor non-squared L2 distance from query to bank.

    Args:
        query: (B, D) tensor.
        bank:  (N, D) tensor.
        chunk_size: number of bank rows processed per cdist call (memory-bound).
    Returns:
        (B,) tensor of min L2 distances (non-squared).
    """
    if query.ndim != 2 or bank.ndim != 2:
        raise ValueError(f"query/bank must be 2D, got {tuple(query.shape)} and {tuple(bank.shape)}")
    if query.shape[1] != bank.shape[1]:
        raise ValueError(f"dim mismatch query={query.shape[1]} bank={bank.shape[1]}")
    if bank.shape[0] == 0:
        raise ValueError("bank cannot be empty")

    q = query
    best = torch.full((q.shape[0],), float("inf"), device=q.device, dtype=q.dtype)
    csz = int(chunk_size)
    for start in range(0, bank.shape[0], csz):
        end = min(start + csz, bank.shape[0])
        chunk = bank[start:end].to(device=q.device, dtype=q.dtype, non_blocking=True)
        d = torch.cdist(q, chunk, p=2.0)  # (B, chunk)
        cur = torch.min(d, dim=1).values
        best = torch.minimum(best, cur)
    return best


# --------------------------------------------------------------------------- #
# Result struct                                                               #
# --------------------------------------------------------------------------- #


@dataclass
class DetectionResult:
    step_scores: np.ndarray   # per-frame min L2 distance (positive; bigger = more OOD)
    thresholds: np.ndarray    # broadcasted threshold per frame (constant unless adaptive)
    preds: np.ndarray         # (T,) int: 1 if failure (min_dist >= threshold), else 0


# --------------------------------------------------------------------------- #
# Encoder wrapper                                                             #
# --------------------------------------------------------------------------- #


class LPBOriginalEncoder:
    """Loads the original LPB dynamics model and exposes (visual+proprio) encoding.

    Reads:
      <ckpt_dir>/hydra.yaml       # full training config (used by dyn_model.plan.load_model)
      <ckpt_dir>/normalizer.pth   # saved LinearNormalizer state_dict (image + state stats)

    `model_ckpt` is the actual `.pth` produced by the training loop (e.g. checkpoints/model_50.pth).
    """

    def __init__(
        self,
        model_ckpt: str,
        device: str = "cuda",
    ) -> None:
        self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")

        ckpt_path = Path(model_ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"LPB dynamics ckpt not found: {ckpt_path}")
        # The hydra config + normalizer live two levels up from `checkpoints/model_*.pth`.
        run_dir = ckpt_path.parent.parent
        cfg_path = run_dir / "hydra.yaml"
        norm_path = run_dir / "normalizer.pth"
        if not cfg_path.exists():
            raise FileNotFoundError(f"hydra.yaml not found next to ckpt: {cfg_path}")
        if not norm_path.exists():
            raise FileNotFoundError(f"normalizer.pth not found next to ckpt: {norm_path}")

        self.cfg = OmegaConf.load(cfg_path)

        self.model = load_model(ckpt_path, self.cfg, device=self.device)
        self.model.eval()

        normalizer = LinearNormalizer()
        normalizer.load_state_dict(torch.load(norm_path, map_location=self.device))
        self.normalizer = normalizer.to(self.device)

        self.view_names: List[str] = list(self.cfg.env.view_names)
        self.original_img_size: int = int(self.cfg.env.original_img_size)
        self.cropped_img_size: int = int(self.cfg.env.cropped_img_size)
        self.use_crop: bool = bool(getattr(self.cfg, "use_crop", True))
        self.proprio_emb_dim: int = int(self.cfg.env.proprio_emb_dim)
        self.visual_emb_dim_total: int = int(self.model.encoder.emb_dim) * len(self.view_names)

        if self.use_crop:
            from dyn_model.datasets.img_transforms import get_eval_crop_transform_resnet
            self.img_transform = get_eval_crop_transform_resnet(
                original_img_size=self.original_img_size,
                cropped_img_size=self.cropped_img_size,
            )
        else:
            self.img_transform = lambda x: x

    @torch.no_grad()
    def encode_batch(
        self,
        images_per_view: Dict[str, torch.Tensor],
        proprio: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a batch of (visual, proprio) timesteps into (B, visual_emb + proprio_emb).

        Args:
            images_per_view[view]: (B, 3, H, W) float tensor in [0, 1]; H=W=original_img_size.
            proprio: (B, proprio_dim) float tensor (same layout as the train-time concat).
        """
        B = next(iter(images_per_view.values())).shape[0]

        visual_in: Dict[str, torch.Tensor] = {}
        for v in self.view_names:
            x = images_per_view[v].to(self.device, non_blocking=True)
            x = self.normalizer[v].normalize(x)  # per-view image normalize
            x = self.img_transform(x.view(-1, 3, self.original_img_size, self.original_img_size))
            x = x.view(B, 1, 3, self.cropped_img_size, self.cropped_img_size)  # add T=1
            visual_in[v] = x

        proprio_in = self.normalizer["state"].normalize(proprio.to(self.device, non_blocking=True))
        if proprio_in.dim() == 2:
            proprio_in = proprio_in.unsqueeze(1)  # (B, 1, D)

        obs = {"visual": visual_in, "proprio": proprio_in}
        enc = self.model.encode_obs(obs)
        v = enc["visual"].squeeze(1) if enc["visual"].dim() == 3 else enc["visual"]
        p = enc["proprio"].squeeze(1) if enc["proprio"].dim() == 3 else enc["proprio"]
        if v.dim() > 2:
            v = v.reshape(v.shape[0], -1)
        if p.dim() > 2:
            p = p.reshape(p.shape[0], -1)
        return torch.cat([v, p], dim=-1)


# --------------------------------------------------------------------------- #
# KNN OOD discriminator                                                       #
# --------------------------------------------------------------------------- #


class LPBOriginalKNN:
    """Per-task KNN OOD detector on top of the original LPB encoder.

    Workflow:
        det = LPBOriginalKNN(visual_dim, proprio_emb_dim, ...)
        det.fit(expert_features=[(N1, D), ...], calibration_features=[(M1, D), ...])
        det.score_trajectory(features)  -> DetectionResult
    """

    def __init__(
        self,
        visual_dim: int,
        proprio_dim: int,
        visual_weight: float = 1.0,
        proprio_weight: float = 2.0,
        delta: float = 10.0,
        chunk_size: int = 2048,
        device: str = "cuda",
    ) -> None:
        if delta < 0.0 or delta > 100.0:
            raise ValueError(f"delta must be in [0, 100], got {delta}")
        self.visual_dim = int(visual_dim)
        self.proprio_dim = int(proprio_dim)
        self.visual_weight = float(visual_weight)
        self.proprio_weight = float(proprio_weight)
        self.delta = float(delta)
        self.chunk_size = int(chunk_size)
        self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")

        self.bank: Optional[torch.Tensor] = None
        self.threshold: Optional[float] = None
        self._weights: Optional[torch.Tensor] = None
        self._calib_min_dists: Optional[np.ndarray] = None

    def _make_weights(self) -> torch.Tensor:
        # Per-dim weight vector replicating original LPB:
        #   weights = [visual_weight] * visual_dim ++ [proprio_weight] * proprio_dim
        if self._weights is not None:
            return self._weights
        w = torch.cat([
            torch.full((self.visual_dim,), self.visual_weight, device=self.device, dtype=torch.float32),
            torch.full((self.proprio_dim,), self.proprio_weight, device=self.device, dtype=torch.float32),
        ])
        self._weights = w
        return w

    def _apply_weights(self, feats: torch.Tensor) -> torch.Tensor:
        w = self._make_weights()
        if feats.shape[-1] != w.shape[0]:
            raise ValueError(
                f"feature dim mismatch: feat.shape[-1]={feats.shape[-1]}, "
                f"expected visual_dim+proprio_dim={w.shape[0]}"
            )
        return feats * w.unsqueeze(0)

    @torch.no_grad()
    def fit(
        self,
        expert_features: Sequence[torch.Tensor],
        calibration_features: Optional[Sequence[torch.Tensor]] = None,
    ) -> float:
        """Build the KNN bank from expert demos and calibrate the threshold.

        Args:
            expert_features:      list of (T_i, D) per-trajectory feature tensors.
            calibration_features: list of (T_j, D) per-trajectory feature tensors,
                                  *disjoint* from `expert_features`.
        Returns:
            The calibrated threshold (higher = more permissive).
        """
        if len(expert_features) == 0:
            raise ValueError("fit requires non-empty expert_features")

        seqs = [x.detach().to(self.device, dtype=torch.float32) for x in expert_features if x.numel() > 0]
        if not seqs:
            raise ValueError("All expert sequences are empty")

        bank = torch.cat(seqs, dim=0)
        bank_w = self._apply_weights(bank)
        self.bank = bank_w

        if calibration_features is None or len(calibration_features) == 0:
            raise ValueError("calibration_features required for threshold calibration")

        cseqs = [x.detach().to(self.device, dtype=torch.float32) for x in calibration_features if x.numel() > 0]
        if not cseqs:
            raise ValueError("All calibration sequences are empty")

        all_min: List[np.ndarray] = []
        for seq in cseqs:
            seq_w = self._apply_weights(seq)
            d = knn_min_l2_dist(seq_w, self.bank, chunk_size=self.chunk_size)
            all_min.append(d.detach().cpu().numpy().astype(np.float32))
        self._calib_min_dists = np.concatenate(all_min, axis=0)

        # Higher delta => fewer false alarms => higher threshold.
        # Match the convention of robosuite/discriminator/lpb: tau = percentile(values, 100 - delta).
        q = 100.0 * (1.0 - self.delta / 100.0)
        self.threshold = float(np.percentile(self._calib_min_dists.astype(np.float64), q=q))
        return self.threshold

    @torch.no_grad()
    def score(self, features: torch.Tensor) -> DetectionResult:
        if self.bank is None or self.threshold is None:
            raise RuntimeError("Call fit(...) before score(...)")
        f = features.reshape(-1, features.shape[-1]).to(self.device, dtype=torch.float32)
        f_w = self._apply_weights(f)
        d = knn_min_l2_dist(f_w, self.bank, chunk_size=self.chunk_size).detach().cpu().numpy().astype(np.float32)
        ths = np.full_like(d, self.threshold, dtype=np.float32)
        preds = (d >= self.threshold).astype(np.int64)
        return DetectionResult(step_scores=d, thresholds=ths, preds=preds)
