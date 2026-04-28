"""Faithful port of the original LPB KNN OOD discriminator.

Mirrors `tmp/lpb-main/dyn_model/planner_libero.py:compute_nn_reward` exactly:

  feature_t  = [encoder(o_t) ; proprio_encoder(s_t) ; action_encoder(a_t)]
               then per-dim weighting: visual * visual_weight,
               proprio * proprio_weight, action * action_weight
  score_t    = -min_{j} ||feature_t - bank_j||_2     (non-squared L2, k=1)

For the benchmark we add a thin classification layer on top of `score_t`:
  - Calibrate a percentile threshold `tau` on a disjoint set of success demos:
        tau = np.percentile(calibration min_dist values, 100 - delta)
  - Predict failure at frame t when  min_dist_t  >=  tau.

This matches the original LPB reward semantics (lower min_dist = more
in-distribution = lower failure score) while letting us emit binary preds.

The dynamics model that produces the encoder + proprio_encoder is loaded
from a checkpoint via `robosuite.discriminator.lpb_v2.model_loader.load_model`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf

from robosuite.discriminator.lpb_v2.model_loader import load_model
from robosuite.discriminator.lpb_v2.utils.normalizer import LinearNormalizer


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


def _resolve_dataset_class_path(path: str) -> str:
    mapping = {
        "robosuite.discriminator.lpb_original.datasets.HDF5DynamicsModelDataset":
            "robosuite.discriminator.lpb_v2.data.hdf5_dynamics_dataset.HDF5DynamicsModelDataset",
        "robosuite.discriminator.lpb_original.datasets.PreprocessedCacheDynamicsModelDataset":
            "robosuite.discriminator.lpb_v2.data.preprocessed_cache_dataset.PreprocessedCacheDynamicsModelDataset",
        "robosuite.discriminator.lpb_original.datasets.hdf5_dynamics_dataset.HDF5DynamicsModelDataset":
            "robosuite.discriminator.lpb_v2.data.hdf5_dynamics_dataset.HDF5DynamicsModelDataset",
        "robosuite.discriminator.lpb_original.datasets.preprocessed_cache_dataset.PreprocessedCacheDynamicsModelDataset":
            "robosuite.discriminator.lpb_v2.data.preprocessed_cache_dataset.PreprocessedCacheDynamicsModelDataset",
    }
    return mapping.get(str(path), str(path))


# --------------------------------------------------------------------------- #
# Encoder wrapper                                                             #
# --------------------------------------------------------------------------- #


class LPBV2Encoder:
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

        # ------------------------------------------------------------------ #
        # Locate saved Hydra config + normalizer across common layouts.
        #
        # New (this repo's `lpb_original/train.py`):
        #   <run_dir>/{hydra.yaml, normalizer.pth, checkpoints/model_*.pth}
        #
        # Legacy (Hydra default output dir):
        #   <run_dir>/.hydra/{hydra.yaml, config.yaml, overrides.yaml}
        #   <run_dir>/checkpoints/model_*.pth
        #
        # Some older runs store `model_*.pth` directly under <run_dir>.
        # ------------------------------------------------------------------ #
        run_dir_candidates: List[Path] = [
            ckpt_path.parent,
            ckpt_path.parent.parent,
        ]
        # If there's a single preprocessing subdir that contains `.hydra/hydra.yaml`,
        # prefer it as a config source.
        try:
            for child in ckpt_path.parent.iterdir():
                if child.is_dir() and (child / ".hydra" / "hydra.yaml").exists():
                    run_dir_candidates.insert(0, child)
        except Exception:
            pass

        cfg_path: Optional[Path] = None
        cfg_candidates: List[Path] = []
        for rd in run_dir_candidates:
            cfg_candidates.extend([
                rd / "hydra.yaml",
                rd / ".hydra" / "config.yaml",
                rd / ".hydra" / "hydra.yaml",
            ])
        for p in cfg_candidates:
            if p.exists():
                cfg_path = p
                break
        if cfg_path is None:
            tried = "\n  - " + "\n  - ".join(str(p) for p in cfg_candidates[:12])
            raise FileNotFoundError(
                "Could not locate a Hydra config for the LPB dynamics checkpoint.\n"
                f"Checkpoint: {ckpt_path}\n"
                f"Tried (first few):{tried}"
            )

        self.cfg = OmegaConf.load(cfg_path)

        # Normalizer: prefer a saved `normalizer.pth` next to the run dir; if missing,
        # rebuild it from the dataset specified in the saved config.
        norm_path: Optional[Path] = None
        norm_candidates: List[Path] = []
        for rd in run_dir_candidates:
            norm_candidates.extend([
                rd / "normalizer.pth",
                rd / ".hydra" / "normalizer.pth",
            ])
        for p in norm_candidates:
            if p.exists():
                norm_path = p
                break

        self.model = load_model(ckpt_path, self.cfg, device=self.device)
        self.model.eval()

        normalizer = LinearNormalizer()
        if norm_path is not None:
            normalizer.load_state_dict(torch.load(norm_path, map_location=self.device))
        else:
            # Fallback: rebuild the normalizer from the training dataset config.
            # This is deterministic for the same dataset and matches the original training code.
            try:
                from hydra.utils import get_class, to_absolute_path

                env = self.cfg.env
                dataset_class_path = str(getattr(env, "dataset_class"))
                DatasetCls = get_class(_resolve_dataset_class_path(dataset_class_path))

                    # Mirror the dataset kwargs used in original LPB training.
                kwargs: Dict[str, Any] = dict(
                    zarr_path=str(getattr(env, "train_data_path")),
                    num_hist=int(getattr(self.cfg, "num_hist")),
                    num_pred=int(getattr(self.cfg, "num_pred")),
                    frameskip=int(getattr(self.cfg, "frameskip")),
                    view_names=list(getattr(env, "view_names")),
                    abs_action=bool(getattr(self.cfg, "abs_action")),
                    use_crop=bool(getattr(self.cfg, "use_crop", True)),
                    train=True,
                    original_img_size=int(getattr(env, "original_img_size")),
                    cropped_img_size=int(getattr(env, "cropped_img_size")),
                    action_dim=int(getattr(env, "action_dim")),
                )
                # Optional knobs present in some dataset classes / configs.
                if hasattr(env, "use_cache"):
                    kwargs["use_cache"] = bool(getattr(env, "use_cache"))
                if hasattr(env, "cache_dir") and getattr(env, "cache_dir") is not None:
                    kwargs["cache_dir"] = to_absolute_path(str(getattr(env, "cache_dir")))
                if hasattr(env, "shape_obs") and getattr(env, "shape_obs") is not None:
                    kwargs["shape_obs"] = OmegaConf.to_container(getattr(env, "shape_obs"), resolve=True)
                if hasattr(env, "cache_root") and getattr(env, "cache_root") is not None:
                    kwargs["cache_root"] = to_absolute_path(str(getattr(env, "cache_root")))
                if hasattr(env, "tasks") and getattr(env, "tasks") is not None:
                    kwargs["tasks"] = list(getattr(env, "tasks"))
                if hasattr(env, "train_sources") and getattr(env, "train_sources") is not None:
                    kwargs["train_sources"] = list(getattr(env, "train_sources"))
                if hasattr(env, "camera_to_view") and getattr(env, "camera_to_view") is not None:
                    kwargs["camera_to_view"] = OmegaConf.to_container(
                        getattr(env, "camera_to_view"), resolve=True
                    )
                if hasattr(env, "max_cached_episodes") and getattr(env, "max_cached_episodes") is not None:
                    kwargs["max_cached_episodes"] = int(getattr(env, "max_cached_episodes"))
                if hasattr(env, "load_all_into_ram"):
                    kwargs["load_all_into_ram"] = bool(getattr(env, "load_all_into_ram"))
                if hasattr(env, "metadata_cache_root") and getattr(env, "metadata_cache_root") is not None:
                    kwargs["metadata_cache_root"] = to_absolute_path(str(getattr(env, "metadata_cache_root")))
                if hasattr(env, "num_expert"):
                    kwargs["num_expert"] = int(getattr(env, "num_expert"))
                if hasattr(env, "num_success"):
                    kwargs["num_success"] = int(getattr(env, "num_success"))
                if hasattr(env, "num_fail"):
                    kwargs["num_fail"] = int(getattr(env, "num_fail"))
                if getattr(self.cfg, "proprio_indices", None):
                    kwargs["proprio_indices"] = list(getattr(self.cfg, "proprio_indices"))
                if getattr(self.cfg, "max_trajectories", None):
                    kwargs["max_trajectories"] = int(getattr(self.cfg, "max_trajectories"))

                ds = DatasetCls(**kwargs)
                rebuilt = ds.get_normalizer()
                normalizer.load_state_dict(rebuilt.state_dict())
                print(
                    "[lpb_v2] WARNING: normalizer.pth not found; rebuilt normalizer from dataset config. "
                    "For exact reproducibility, re-export `normalizer.pth` next to the checkpoint."
                )
            except Exception as e:
                raise FileNotFoundError(
                    "normalizer.pth not found near the checkpoint, and rebuilding the normalizer from the "
                    f"saved Hydra config failed.\nCheckpoint: {ckpt_path}\nConfig: {cfg_path}\n"
                    f"Original error: {type(e).__name__}: {e}"
                ) from e
        self.normalizer = normalizer.to(self.device)

        self.view_names: List[str] = list(self.cfg.env.view_names)
        self.original_img_size: int = int(self.cfg.env.original_img_size)
        self.cropped_img_size: int = int(self.cfg.env.cropped_img_size)
        self.use_crop: bool = bool(getattr(self.cfg, "use_crop", True))
        self.proprio_emb_dim: int = int(self.cfg.env.proprio_emb_dim)
        self.action_emb_dim: int = int(self.cfg.env.action_emb_dim)
        self.visual_emb_dim_total: int = int(self.model.encoder.emb_dim) * len(self.view_names)
        self.frameskip: int = int(getattr(self.cfg, "frameskip", 1))
        self.action_dim_per_step: int = int(
            getattr(self.cfg, "action_dim_per_step", getattr(self.cfg.env, "action_dim", 7))
        )
        self.action_input_dim: int = int(
            getattr(self.model.action_encoder, "in_chans", self.action_dim_per_step * self.frameskip)
        )

        if self.use_crop:
            from robosuite.discriminator.lpb_v2.data.img_transforms import get_eval_crop_transform_resnet
            self.img_transform = get_eval_crop_transform_resnet(
                original_img_size=self.original_img_size,
                cropped_img_size=self.cropped_img_size,
            )
        else:
            self.img_transform = lambda x: x

    def prepare_actions(self, actions: np.ndarray, t_len: int) -> np.ndarray:
        """Build flattened action windows matching the train-time action encoder input."""
        act = np.asarray(actions[:t_len], dtype=np.float32)
        if act.ndim == 1:
            act = act.reshape(-1, 1)
        if act.shape[1] < self.action_dim_per_step:
            pad = np.zeros((act.shape[0], self.action_dim_per_step - act.shape[1]), dtype=np.float32)
            act = np.concatenate([act, pad], axis=1)
        elif act.shape[1] > self.action_dim_per_step:
            act = act[:, : self.action_dim_per_step]

        out = np.zeros((int(t_len), self.frameskip, self.action_dim_per_step), dtype=np.float32)
        for t in range(int(t_len)):
            end = min(int(t_len), t + self.frameskip)
            chunk = act[t:end]
            out[t, : chunk.shape[0]] = chunk
            if chunk.shape[0] < self.frameskip:
                pad_value = chunk[-1] if chunk.shape[0] > 0 else np.zeros((self.action_dim_per_step,), dtype=np.float32)
                out[t, chunk.shape[0] :] = pad_value

        flat = out.reshape(int(t_len), -1)
        if flat.shape[1] < self.action_input_dim:
            pad = np.zeros((flat.shape[0], self.action_input_dim - flat.shape[1]), dtype=np.float32)
            flat = np.concatenate([flat, pad], axis=1)
        elif flat.shape[1] > self.action_input_dim:
            flat = flat[:, : self.action_input_dim]
        return flat.astype(np.float32, copy=False)

    @torch.no_grad()
    def encode_batch(
        self,
        images_per_view: Dict[str, torch.Tensor],
        proprio: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a batch of (visual, proprio, action) timesteps.

        Args:
            images_per_view[view]: (B, 3, H, W) float tensor in [0, 1]; H=W=original_img_size.
            proprio: (B, proprio_dim) float tensor (same layout as the train-time concat).
            actions: (B, action_dim * frameskip) raw action windows.
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

        action_in = actions.to(self.device, dtype=torch.float32, non_blocking=True)
        action_in = action_in.view(B, self.frameskip, self.action_dim_per_step)
        action_in = self.normalizer["act"].normalize(action_in)
        action_in = action_in.reshape(B, 1, self.action_input_dim)
        a = self.model.encode_act(action_in).squeeze(1)
        if a.dim() > 2:
            a = a.reshape(a.shape[0], -1)
        return torch.cat([v, p, a], dim=-1)


# --------------------------------------------------------------------------- #
# KNN OOD discriminator                                                       #
# --------------------------------------------------------------------------- #


class LPBV2KNN:
    """Per-task KNN OOD detector on top of the original LPB encoder.

    Workflow:
        det = LPBV2KNN(visual_dim, proprio_emb_dim, ...)
        det.fit(expert_features=[(N1, D), ...], calibration_features=[(M1, D), ...])
        det.score_trajectory(features)  -> DetectionResult
    """

    def __init__(
        self,
        visual_dim: int,
        proprio_dim: int,
        action_dim: int,
        visual_weight: float = 1.0,
        proprio_weight: float = 2.0,
        action_weight: float = 1.0,
        delta: float = 10.0,
        chunk_size: int = 2048,
        device: str = "cuda",
    ) -> None:
        if delta < 0.0 or delta > 100.0:
            raise ValueError(f"delta must be in [0, 100], got {delta}")
        self.visual_dim = int(visual_dim)
        self.proprio_dim = int(proprio_dim)
        self.action_dim = int(action_dim)
        self.visual_weight = float(visual_weight)
        self.proprio_weight = float(proprio_weight)
        self.action_weight = float(action_weight)
        self.delta = float(delta)
        self.chunk_size = int(chunk_size)
        self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")

        self.bank: Optional[torch.Tensor] = None
        self.threshold: Optional[float] = None
        self._weights: Optional[torch.Tensor] = None
        self._calib_min_dists: Optional[np.ndarray] = None

    def _make_weights(self) -> torch.Tensor:
        # Per-dim weight vector:
        #   [visual_weight] * visual_dim ++ [proprio_weight] * proprio_dim ++ [action_weight] * action_dim
        if self._weights is not None:
            return self._weights
        w = torch.cat([
            torch.full((self.visual_dim,), self.visual_weight, device=self.device, dtype=torch.float32),
            torch.full((self.proprio_dim,), self.proprio_weight, device=self.device, dtype=torch.float32),
            torch.full((self.action_dim,), self.action_weight, device=self.device, dtype=torch.float32),
        ])
        self._weights = w
        return w

    def _apply_weights(self, feats: torch.Tensor) -> torch.Tensor:
        w = self._make_weights()
        if feats.shape[-1] != w.shape[0]:
            raise ValueError(
                f"feature dim mismatch: feat.shape[-1]={feats.shape[-1]}, "
                f"expected visual_dim+proprio_dim+action_dim={w.shape[0]}"
            )
        # Weighting is equivalent to a diagonal Mahalanobis metric with fixed per-block scales.
        # We apply it to both bank and queries so `torch.cdist` still computes standard L2.
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

        # Threshold calibration:
        #   delta in [0, 100] is a percentile-based false-alarm budget.
        #   - delta=0   -> tau = 100th percentile (max)   -> most permissive, almost no alarms
        #   - delta=10  -> tau = 90th percentile          -> allow ~10% of calib frames to exceed tau
        # Match lpb convention: tau = percentile(values, 100 - delta).
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


LPBOriginalEncoder = LPBV2Encoder
LPBOriginalKNN = LPBV2KNN
