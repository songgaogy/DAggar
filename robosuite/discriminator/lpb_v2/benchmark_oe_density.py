"""Benchmark adapter for the OE-density score-based failure discriminator.

Implements the primary method committed to in
``PLAN_score_based_discriminator.md`` §4.A--§4.I and detailed in
``PROMPT_oe_density.md``. Subclasses :class:`LPBV2BenchmarkDiscriminator` so
the frozen-encoder pipeline (``_encode``, normalization, image preprocessing,
feature caching) is reused verbatim -- only the per-task detector is swapped.

Per-task workflow (mirrors :class:`TwoBankBenchmarkDiscriminator`):

1. ``fit_on_benchmark(trajectories)``:
   - Assert disjointness between the eval pool, the f-train fail pool, and the
     optional fail-calib pool by ``video_id``.
   - Split per-task success demos into bank vs calibration (same RNG-seeded
     scheme as the parent class).
   - Encode bank-side success frames and concatenate task one-hot rows; train
     one *pooled* :class:`OEDensityScore` across tasks with the logistic
     discriminative loss (see ``models/score_oe_density.py`` docstring).
   - Calibrate one threshold ``tau`` per task on its own calibration slice
     (``success_percentile`` by default; ``two_class_youden`` requires
     non-empty fail-calib). NB: this calibration tau is the *adapter*-level
     per-task threshold, distinct from the model's training-time
     ``self._model.tau`` (logistic-regression parameter).
2. ``score_trajectory(traj)``: encode, append the task one-hot, score via the
   pooled OE model, threshold with the task's ``tau``.

Hard invariant (defence-in-depth): no ``video_id`` from
``fail_train_trajectories`` or ``fail_calib_trajectories`` may appear in the
eval set.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from benchmark.core import BenchmarkTrajectory, DiscriminatorOutput

from .benchmark import LPBV2BenchmarkDiscriminator, _pad_to_length
from .benchmark_two_bank import _fail_frame_range
from .models.score_oe_density import OEDensityScore
from .two_bank_knn import _youden_threshold


_VALID_DENSITIES = ("gaussian", "gmm2", "gmm4")
_VALID_CALIB_MODES = ("success_percentile", "two_class_youden")


class OEDensityBenchmarkDiscriminator(LPBV2BenchmarkDiscriminator):
    """Score-based failure detector using an OE-density head on the WAM latent."""

    name = "lpb_v2_oe_density"

    def __init__(
        self,
        *,
        model_ckpt: str,
        fail_train_trajectories: Sequence[BenchmarkTrajectory],
        fail_calib_trajectories: Optional[Sequence[BenchmarkTrajectory]] = None,
        ftrain_last_k: int = 60,
        fail_use_all_after_gt: bool = True,
        fail_prefix_to_succ: bool = True,
        score_k: int = 16,
        lam: float = 1.0,
        weight_decay: float = 1e-4,
        density: str = "gaussian",
        calib_mode: str = "success_percentile",
        num_epochs: int = 200,
        batch_size: int = 4096,
        learning_rate: float = 3e-3,
        # forwarded to the single-bank parent for parity
        device: str = "cuda",
        encode_batch_size: int = 32,
        proprio_indices: Optional[Sequence[int]] = None,
        camera_to_view: Optional[Dict[str, str]] = None,
        visual_weight: float = 1.0,
        proprio_weight: float = 2.0,
        action_weight: float = 1.0,
        delta: float = 10.0,
        knn_chunk_size: int = 2048,
        feature_source: str = "transformer",
        transformer_layer: int = 1,
        calib_fraction: float = 0.2,
        seed: int = 0,
        verbose_fit: bool = True,
    ) -> None:
        super().__init__(
            model_ckpt=model_ckpt,
            device=device,
            encode_batch_size=encode_batch_size,
            proprio_indices=proprio_indices,
            camera_to_view=camera_to_view,
            visual_weight=visual_weight,
            proprio_weight=proprio_weight,
            action_weight=action_weight,
            delta=delta,
            knn_chunk_size=knn_chunk_size,
            feature_source=feature_source,
            transformer_layer=transformer_layer,
            calib_fraction=calib_fraction,
            seed=seed,
            verbose_fit=verbose_fit,
        )
        if density not in _VALID_DENSITIES:
            raise ValueError(f"density must be one of {_VALID_DENSITIES}, got {density!r}")
        if calib_mode not in _VALID_CALIB_MODES:
            raise ValueError(
                f"calib_mode must be one of {_VALID_CALIB_MODES}, got {calib_mode!r}"
            )
        if int(num_epochs) <= 0:
            raise ValueError(f"num_epochs must be positive, got {num_epochs}")
        if int(batch_size) <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if float(lam) < 0.0:
            raise ValueError(f"lam must be >= 0, got {lam}")

        self.fail_train_trajectories: List[BenchmarkTrajectory] = list(fail_train_trajectories)
        self.fail_calib_trajectories: List[BenchmarkTrajectory] = (
            list(fail_calib_trajectories) if fail_calib_trajectories else []
        )
        self.ftrain_last_k = int(ftrain_last_k)
        self.fail_use_all_after_gt = bool(fail_use_all_after_gt)
        self.fail_prefix_to_succ = bool(fail_prefix_to_succ)
        self.score_k = int(score_k)
        self.lam = float(lam)
        self.weight_decay = float(weight_decay)
        self.density = str(density)
        self.calib_mode = str(calib_mode)
        self.num_epochs = int(num_epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)

        # Replace parent's per-task detector store with our pooled-model layout.
        self._detectors_per_task: Dict[str, OEDensityScore] = {}  # never used; kept for compatibility
        self._calibration_stats: Dict[str, dict] = {}
        self._model: Optional[OEDensityScore] = None
        self._tau_per_task: Dict[str, float] = {}
        self._calib_method_per_task: Dict[str, str] = {}
        self._task_to_onehot: Dict[str, np.ndarray] = {}
        self._known_tasks: List[str] = []
        self._train_history: List[dict] = []
        self._ftrain_index_ranges: Dict[str, Dict[str, List[int]]] = {}
        self._ftrain_prefix_index_ranges: Dict[str, Dict[str, List[int]]] = {}

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _assert_disjoint(
        eval_trajs: Sequence[BenchmarkTrajectory],
        ftrain_trajs: Sequence[BenchmarkTrajectory],
        fcalib_trajs: Sequence[BenchmarkTrajectory],
    ) -> None:
        eval_keys = {str(t.video_id) for t in eval_trajs}
        ftrain_keys = {str(t.video_id) for t in ftrain_trajs}
        fcalib_keys = {str(t.video_id) for t in fcalib_trajs}
        overlap_ftrain = sorted(eval_keys & ftrain_keys)
        overlap_fcalib = sorted(eval_keys & fcalib_keys)
        ftrain_fcalib_overlap = sorted(ftrain_keys & fcalib_keys)
        problems = []
        if overlap_ftrain:
            problems.append(f"eval ∩ ftrain = {overlap_ftrain}")
        if overlap_fcalib:
            problems.append(f"eval ∩ fail_calib = {overlap_fcalib}")
        if ftrain_fcalib_overlap:
            problems.append(f"ftrain ∩ fail_calib = {ftrain_fcalib_overlap}")
        if problems:
            raise RuntimeError(
                "OEDensityBenchmarkDiscriminator disjointness invariant violated:\n  - "
                + "\n  - ".join(problems)
            )

    def _encode_fail_subset(
        self,
        trajectory: BenchmarkTrajectory,
        last_k: int,
        index_ranges: Optional[Dict[str, Dict[str, List[int]]]] = None,
        warn_tag: str = "ftrain",
    ) -> Optional[torch.Tensor]:
        """Encode a failure trajectory and slice to ``[first_gt, first_gt+last_k)``."""
        idx_range = _fail_frame_range(
            num_frames=int(trajectory.num_frames),
            first_gt=trajectory.first_gt_failure_frame(),
            last_k=int(last_k),
        )
        if idx_range is None:
            print(
                f"[oe_density] WARNING: skipping {warn_tag} trajectory with no "
                f"first_gt_failure_frame: video_id={trajectory.video_id}"
            )
            return None
        feats = self._encode(trajectory)
        T = int(feats.shape[0])
        start = min(idx_range.start, T)
        end = min(idx_range.stop, T)
        if end <= start:
            print(
                f"[oe_density] WARNING: empty {warn_tag} slice after re-clip "
                f"(video_id={trajectory.video_id}, T={T}, range={idx_range})"
            )
            return None
        sub = feats[start:end]
        if index_ranges is not None:
            per_task = index_ranges.setdefault(str(trajectory.task_name), {})
            per_task[str(trajectory.video_id)] = [int(start), int(end)]
        return sub

    def _encode_fail_split(
        self,
        trajectory: BenchmarkTrajectory,
        use_all_after_gt: bool,
        last_k: int,
        prefix_to_succ: bool,
        suffix_index_ranges: Optional[Dict[str, Dict[str, List[int]]]] = None,
        prefix_index_ranges: Optional[Dict[str, Dict[str, List[int]]]] = None,
        warn_tag: str = "ftrain",
    ):
        """Encode a failure trajectory once and split into succ-prefix / fail-suffix.

        Returns a tuple ``(prefix_feats, suffix_feats)`` where each may be ``None`` if
        the corresponding slice is empty or disabled.

        - suffix (fail): ``[first_gt, T)`` when ``use_all_after_gt`` is True,
          else ``[first_gt, first_gt + last_k)`` clipped to ``T``.
        - prefix (succ): ``[0, first_gt)`` when ``prefix_to_succ`` is True,
          else ``None``.
        """
        first_gt = trajectory.first_gt_failure_frame()
        if first_gt is None:
            print(
                f"[oe_density] WARNING: skipping {warn_tag} trajectory with no "
                f"first_gt_failure_frame: video_id={trajectory.video_id}"
            )
            return None, None

        feats = self._encode(trajectory)
        T = int(feats.shape[0])
        gt = int(first_gt)
        gt = max(0, min(T, gt))

        # Fail suffix.
        if use_all_after_gt:
            s_start, s_end = gt, T
        else:
            s_start = gt
            s_end = min(gt + int(last_k), T)
        suffix = feats[s_start:s_end] if s_end > s_start else None
        if suffix is None:
            print(
                f"[oe_density] WARNING: empty {warn_tag} fail-suffix slice "
                f"(video_id={trajectory.video_id}, T={T}, first_gt={gt})"
            )
        elif suffix_index_ranges is not None:
            per_task = suffix_index_ranges.setdefault(str(trajectory.task_name), {})
            per_task[str(trajectory.video_id)] = [int(s_start), int(s_end)]

        # Succ prefix.
        prefix = None
        if prefix_to_succ:
            if gt > 0:
                prefix = feats[0:gt]
                if prefix_index_ranges is not None:
                    per_task = prefix_index_ranges.setdefault(str(trajectory.task_name), {})
                    per_task[str(trajectory.video_id)] = [0, int(gt)]
            else:
                print(
                    f"[oe_density] WARNING: empty {warn_tag} succ-prefix slice "
                    f"(video_id={trajectory.video_id}, first_gt={gt})"
                )

        return prefix, suffix

    def _append_onehot(self, feats: torch.Tensor, task: str) -> torch.Tensor:
        """Append the task one-hot row to every frame of ``feats``."""
        onehot = self._task_to_onehot[task]
        N = int(feats.shape[0])
        if N == 0:
            T = int(onehot.shape[0])
            return torch.empty((0, int(feats.shape[1]) + T), dtype=feats.dtype, device=feats.device)
        onehot_t = torch.from_numpy(onehot).to(feats.device, dtype=feats.dtype)
        onehot_b = onehot_t.unsqueeze(0).expand(N, -1)
        return torch.cat([feats, onehot_b], dim=1)

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def fit_on_benchmark(self, trajectories: List[BenchmarkTrajectory]) -> None:
        self._assert_disjoint(
            eval_trajs=trajectories,
            ftrain_trajs=self.fail_train_trajectories,
            fcalib_trajs=self.fail_calib_trajectories,
        )

        # ---------------- Task indexing ---------------- #
        task_to_success: Dict[str, List[BenchmarkTrajectory]] = {}
        for traj in trajectories:
            if bool(traj.is_failure):
                continue
            task_to_success.setdefault(str(traj.task_name), []).append(traj)
        if not task_to_success:
            raise RuntimeError(
                "OE-density calibration requires success trajectories per task."
            )

        known_tasks = sorted(task_to_success.keys())
        self._known_tasks = known_tasks
        T_tasks = len(known_tasks)
        eye = np.eye(T_tasks, dtype=np.float32)
        self._task_to_onehot = {task: eye[i] for i, task in enumerate(known_tasks)}

        task_to_ftrain: Dict[str, List[BenchmarkTrajectory]] = {}
        for t in self.fail_train_trajectories:
            task_to_ftrain.setdefault(str(t.task_name), []).append(t)
        task_to_fcalib: Dict[str, List[BenchmarkTrajectory]] = {}
        for t in self.fail_calib_trajectories:
            task_to_fcalib.setdefault(str(t.task_name), []).append(t)

        # ---------------- Per-task success bank/calib split + fail-train ---------------- #
        # We do the same RNG-seeded split as the parent class, then keep the
        # per-task calib features for threshold calibration and pool the bank
        # features into one big tensor for OE training.
        device = torch.device(
            self.device
            if (self.device != "cuda" or torch.cuda.is_available())
            else "cpu"
        )

        pooled_succ_train: List[torch.Tensor] = []
        pooled_succ_calib: List[torch.Tensor] = []
        per_task_succ_calib: Dict[str, torch.Tensor] = {}
        per_task_bank_count: Dict[str, int] = {}
        per_task_calib_count: Dict[str, int] = {}
        per_task_succ_traj_count: Dict[str, int] = {}
        per_task_bank_traj_count: Dict[str, int] = {}
        per_task_calib_traj_count: Dict[str, int] = {}

        pooled_ftrain: List[torch.Tensor] = []
        per_task_ftrain_count: Dict[str, int] = {}
        per_task_ftrain_traj_count: Dict[str, int] = {}
        per_task_succ_from_fail_prefix_count: Dict[str, int] = {}
        per_task_succ_from_fail_prefix_traj_count: Dict[str, int] = {}

        per_task_fcalib_feats: Dict[str, torch.Tensor] = {}
        per_task_fcalib_count: Dict[str, int] = {}
        per_task_fcalib_traj_count: Dict[str, int] = {}

        for task in known_tasks:
            succ_list = task_to_success[task]
            if len(succ_list) < 2:
                raise RuntimeError(
                    f"Task {task!r} has only {len(succ_list)} success trajectory; "
                    "need at least 2 for disjoint bank + calibration split."
                )
            rng = np.random.default_rng(int(self.seed))
            perm = rng.permutation(len(succ_list))
            n_calib = int(round(self.calib_fraction * len(succ_list)))
            n_calib = max(1, min(len(succ_list) - 1, n_calib))
            calib_idx = set(perm[:n_calib].tolist())
            bank_trajs = [t for i, t in enumerate(succ_list) if i not in calib_idx]
            calib_trajs = [t for i, t in enumerate(succ_list) if i in calib_idx]

            bank_feats_task: List[torch.Tensor] = []
            calib_feats_task: List[torch.Tensor] = []
            bank_total = 0
            calib_total = 0
            for t in bank_trajs:
                f = self._encode(t)
                z_tilde = self._append_onehot(f, task)
                bank_feats_task.append(z_tilde)
                bank_total += int(f.shape[0])
            for t in calib_trajs:
                f = self._encode(t)
                z_tilde = self._append_onehot(f, task)
                calib_feats_task.append(z_tilde)
                calib_total += int(f.shape[0])

            if bank_feats_task:
                pooled_succ_train.extend(bank_feats_task)
            if calib_feats_task:
                calib_cat = torch.cat(calib_feats_task, dim=0)
                pooled_succ_calib.append(calib_cat)
                per_task_succ_calib[task] = calib_cat
            else:
                per_task_succ_calib[task] = torch.empty((0, 0), dtype=torch.float32)

            per_task_succ_traj_count[task] = int(len(succ_list))
            per_task_bank_traj_count[task] = int(len(bank_trajs))
            per_task_calib_traj_count[task] = int(len(calib_trajs))
            per_task_bank_count[task] = int(bank_total)
            per_task_calib_count[task] = int(calib_total)

            # Fail-train slice for this task. We encode each fail-train trajectory
            # once and split it into:
            #   - fail suffix  (frames >= first_gt) -> pooled_ftrain
            #   - succ prefix  (frames <  first_gt) -> pooled_succ_train (augmentation)
            # The prefix augmentation gives the model paired same-trajectory
            # supervision (succ vs fail at matched scene/setup), avoiding the
            # trajectory-level shortcut that dominated the previous run.
            ftrain_list = task_to_ftrain.get(task, [])
            ftrain_feats_task: List[torch.Tensor] = []
            ftrain_total = 0
            ftrain_kept = 0
            fprefix_total = 0
            fprefix_kept = 0
            for t in ftrain_list:
                prefix, suffix = self._encode_fail_split(
                    t,
                    use_all_after_gt=self.fail_use_all_after_gt,
                    last_k=self.ftrain_last_k,
                    prefix_to_succ=self.fail_prefix_to_succ,
                    suffix_index_ranges=self._ftrain_index_ranges,
                    prefix_index_ranges=self._ftrain_prefix_index_ranges,
                    warn_tag="ftrain",
                )
                if suffix is not None:
                    ftrain_feats_task.append(self._append_onehot(suffix, task))
                    ftrain_total += int(suffix.shape[0])
                    ftrain_kept += 1
                if prefix is not None:
                    pooled_succ_train.append(self._append_onehot(prefix, task))
                    fprefix_total += int(prefix.shape[0])
                    fprefix_kept += 1
            if ftrain_feats_task:
                pooled_ftrain.extend(ftrain_feats_task)
            per_task_ftrain_traj_count[task] = ftrain_kept
            per_task_ftrain_count[task] = ftrain_total
            per_task_succ_from_fail_prefix_count[task] = fprefix_total
            per_task_succ_from_fail_prefix_traj_count[task] = fprefix_kept

            # Fail-calib slice for this task.
            fcalib_list = task_to_fcalib.get(task, [])
            fcalib_feats_task: List[torch.Tensor] = []
            fcalib_total = 0
            fcalib_kept = 0
            for t in fcalib_list:
                sub = self._encode_fail_subset(
                    t,
                    last_k=self.ftrain_last_k,
                    warn_tag="fail_calib",
                )
                if sub is None:
                    continue
                z_tilde = self._append_onehot(sub, task)
                fcalib_feats_task.append(z_tilde)
                fcalib_total += int(sub.shape[0])
                fcalib_kept += 1
            if fcalib_feats_task:
                per_task_fcalib_feats[task] = torch.cat(fcalib_feats_task, dim=0)
            per_task_fcalib_traj_count[task] = fcalib_kept
            per_task_fcalib_count[task] = fcalib_total

        # ---------------- Pool and move to device ---------------- #
        succ_train = torch.cat(pooled_succ_train, dim=0).to(device, dtype=torch.float32)
        succ_calib = (
            torch.cat(pooled_succ_calib, dim=0).to(device, dtype=torch.float32)
            if pooled_succ_calib
            else torch.empty((0, succ_train.shape[1]), device=device, dtype=torch.float32)
        )

        if self.lam > 0.0:
            if not pooled_ftrain:
                raise RuntimeError(
                    "OE-density requires at least one usable fail-train trajectory "
                    "when lam > 0. Pass --lambda 0 for the ablation, or supply "
                    "fail-train trajectories with first_gt_failure_frame annotated."
                )
            fail_train = torch.cat(pooled_ftrain, dim=0).to(device, dtype=torch.float32)
        else:
            # Allocate an empty placeholder so downstream code can branch on size.
            fail_train = torch.empty(
                (0, succ_train.shape[1]), device=device, dtype=torch.float32
            )

        feat_dim_z_tilde = int(succ_train.shape[1])
        D = feat_dim_z_tilde - T_tasks
        if D <= 0:
            raise RuntimeError(
                f"Inferred latent dim D={D} non-positive: feat_dim_z_tilde="
                f"{feat_dim_z_tilde}, num_tasks={T_tasks}."
            )

        if self.verbose_fit:
            n_prefix_frames = sum(per_task_succ_from_fail_prefix_count.values())
            n_prefix_trajs = sum(per_task_succ_from_fail_prefix_traj_count.values())
            print(
                f"[oe_density][fit] tasks={known_tasks} T={T_tasks} D={D} k={self.score_k}  "
                f"succ_train={succ_train.shape[0]} frames "
                f"(+{n_prefix_frames} from {n_prefix_trajs} fail-trajectory prefixes), "
                f"succ_calib={succ_calib.shape[0]} frames, "
                f"fail_train={fail_train.shape[0]} frames "
                f"(use_all_after_gt={self.fail_use_all_after_gt}, prefix_to_succ={self.fail_prefix_to_succ})  "
                f"feature_source={self.feature_source} layer={self.transformer_layer}  "
                f"lam={self.lam} (logistic form) epochs={self.num_epochs}",
                flush=True,
            )

        # ---------------- Build + train OE model ---------------- #
        torch.manual_seed(int(self.seed))
        model = OEDensityScore(
            in_dim=feat_dim_z_tilde,
            score_k=self.score_k,
            density=self.density,
            weight_decay=self.weight_decay,
            device=str(self.device),
        )
        model.init_density_from_batch(succ_train)

        param_groups = [
            {"params": [model.W1], "weight_decay": 0.0},
            {"params": [model.b1], "weight_decay": 0.0},
            {"params": [model.mu], "weight_decay": 0.0},
            {"params": [model.L_raw], "weight_decay": 0.0},
            {"params": [model.tau], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(
            param_groups, lr=self.learning_rate, betas=(0.9, 0.999)
        )

        sampler = np.random.default_rng(int(self.seed) + 1)
        N_succ = int(succ_train.shape[0])
        N_fail = int(fail_train.shape[0])
        batches_per_epoch = max(1, math.ceil(N_succ / max(1, self.batch_size)))

        self._train_history = []
        for epoch in range(self.num_epochs):
            perm = sampler.permutation(N_succ)
            ep_succ = ep_fail = ep_reg = ep_total = 0.0
            ep_s_succ = ep_s_fail = ep_tau = 0.0
            for b in range(batches_per_epoch):
                start = b * self.batch_size
                end = min(start + self.batch_size, N_succ)
                batch_succ = succ_train[perm[start:end]]

                if self.lam > 0.0 and N_fail > 0:
                    replace = N_fail < self.batch_size
                    fail_idx = sampler.choice(
                        N_fail,
                        size=min(self.batch_size, max(1, N_fail)),
                        replace=replace,
                    )
                    batch_fail = fail_train[fail_idx]
                else:
                    batch_fail = None

                losses = model.loss(batch_succ, batch_fail, self.lam)
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                optimizer.step()
                ep_succ += float(losses["succ"].detach().item())
                ep_fail += float(losses["fail"].detach().item())
                ep_reg += float(losses["reg"].detach().item())
                ep_total += float(losses["total"].detach().item())
                ep_s_succ += float(losses["score_succ_mean"].item())
                ep_s_fail += float(losses["score_fail_mean"].item())
                ep_tau += float(losses["tau"].item())

            self._train_history.append(
                {
                    "epoch": int(epoch),
                    "L_succ": ep_succ / batches_per_epoch,
                    "L_fail": ep_fail / batches_per_epoch,
                    "L_reg": ep_reg / batches_per_epoch,
                    "L_total": ep_total / batches_per_epoch,
                    "score_succ_mean": ep_s_succ / batches_per_epoch,
                    "score_fail_mean": ep_s_fail / batches_per_epoch,
                    "tau": ep_tau / batches_per_epoch,
                }
            )
            if self.verbose_fit and (epoch == 0 or (epoch + 1) % max(1, self.num_epochs // 10) == 0):
                last = self._train_history[-1]
                gap = last["score_fail_mean"] - last["score_succ_mean"]
                print(
                    f"[oe_density][train] epoch={epoch + 1}/{self.num_epochs}  "
                    f"L_total={last['L_total']:.4f}  L_succ={last['L_succ']:.4f}  "
                    f"L_fail={last['L_fail']:.4f}  L_reg={last['L_reg']:.4f}  "
                    f"s_succ={last['score_succ_mean']:.3f}  "
                    f"s_fail={last['score_fail_mean']:.3f}  "
                    f"tau={last['tau']:.3f}  gap={gap:+.3f}",
                    flush=True,
                )

        model.eval()
        self._model = model

        # ---------------- Per-task threshold calibration ---------------- #
        with torch.no_grad():
            for task in known_tasks:
                calib_z = per_task_succ_calib[task]
                if calib_z.numel() == 0:
                    raise RuntimeError(
                        f"Task {task!r} has no success calibration frames; cannot "
                        "set a threshold."
                    )
                calib_z = calib_z.to(device, dtype=torch.float32)
                s_calib = model.score(calib_z).detach().cpu().numpy().astype(np.float64)

                method_used = self.calib_mode
                tau: float
                if self.calib_mode == "two_class_youden":
                    fcalib_z = per_task_fcalib_feats.get(task, None)
                    if fcalib_z is None or fcalib_z.numel() == 0:
                        print(
                            f"[oe_density] WARNING: calib_mode='two_class_youden' but "
                            f"task {task!r} has no fail-calib frames; falling back to "
                            "'success_percentile'."
                        )
                        method_used = "success_percentile"
                    else:
                        fcalib_z = fcalib_z.to(device, dtype=torch.float32)
                        s_fcalib = model.score(fcalib_z).detach().cpu().numpy().astype(np.float64)
                        tau = float(_youden_threshold(s_calib, s_fcalib))
                if method_used == "success_percentile":
                    q = 100.0 * (1.0 - float(self.delta) / 100.0)
                    tau = float(np.percentile(s_calib, q=q))

                self._tau_per_task[task] = float(tau)
                self._calib_method_per_task[task] = method_used

                self._calibration_stats[task] = {
                    "num_success_trajectories": per_task_succ_traj_count[task],
                    "num_bank_trajectories": per_task_bank_traj_count[task],
                    "num_calib_trajectories": per_task_calib_traj_count[task],
                    "num_bank_steps": per_task_bank_count[task],
                    "num_calib_steps": per_task_calib_count[task],
                    "num_fail_train_trajectories": per_task_ftrain_traj_count.get(task, 0),
                    "num_fail_train_frames": per_task_ftrain_count.get(task, 0),
                    "num_succ_from_fail_prefix_trajectories": (
                        per_task_succ_from_fail_prefix_traj_count.get(task, 0)
                    ),
                    "num_succ_from_fail_prefix_frames": (
                        per_task_succ_from_fail_prefix_count.get(task, 0)
                    ),
                    "num_fail_calib_trajectories": per_task_fcalib_traj_count.get(task, 0),
                    "num_fail_calib_frames": per_task_fcalib_count.get(task, 0),
                    "threshold_init": float(tau),
                    "delta_init": float(self.delta),
                    "feat_dim": int(D),
                    "feat_dim_z_tilde": int(feat_dim_z_tilde),
                    "score_k": int(self.score_k),
                    "num_tasks": int(T_tasks),
                    "score_mode": "oe_density",
                    "calib_mode": self.calib_mode,
                    "calib_method_used": method_used,
                    "lam": float(self.lam),
                    "fail_loss_form": "logistic",
                    "score_succ_mean_final": (
                        float(self._train_history[-1]["score_succ_mean"])
                        if self._train_history
                        else float("nan")
                    ),
                    "score_fail_mean_final": (
                        float(self._train_history[-1]["score_fail_mean"])
                        if self._train_history
                        else float("nan")
                    ),
                    "tau_logistic_final": (
                        float(self._train_history[-1]["tau"])
                        if self._train_history
                        else float("nan")
                    ),
                    "density": self.density,
                    "feature_source": self.feature_source,
                    "transformer_layer": int(self.transformer_layer),
                    "calib_score_min": float(s_calib.min()),
                    "calib_score_max": float(s_calib.max()),
                    "calib_score_mean": float(s_calib.mean()),
                    "calib_score_std": float(s_calib.std()),
                    "num_calib_frames": int(s_calib.size),
                }

    def score_trajectory(self, trajectory: BenchmarkTrajectory) -> DiscriminatorOutput:
        task = str(trajectory.task_name)
        if self._model is None or task not in self._tau_per_task:
            raise KeyError(
                f"Task {task!r} is not calibrated. Available: "
                f"{sorted(self._tau_per_task)}"
            )

        T = int(trajectory.num_frames)
        feat = self._encode(trajectory)  # (N, D) CPU float32
        z_tilde_cpu = self._append_onehot(feat, task)
        device = next(self._model.parameters()).device
        z_tilde = z_tilde_cpu.to(device, dtype=torch.float32)

        with torch.no_grad():
            s = self._model.score(z_tilde).detach().cpu().numpy().astype(np.float32)

        tau = float(self._tau_per_task[task])
        ths = np.full_like(s, tau, dtype=np.float32)
        preds_unpadded = (s >= tau).astype(np.int64)

        step_scores = _pad_to_length(s, target_len=T, dtype=np.float32)
        thresholds = _pad_to_length(ths, target_len=T, dtype=np.float32)
        preds = _pad_to_length(preds_unpadded, target_len=T, dtype=np.int64).astype(np.int64)

        positive = np.where(preds == 1)[0]
        first_failure_frame = int(positive[0]) if positive.size > 0 else None

        aux: Dict[str, Any] = {
            "task": task,
            "threshold": tau,
            "delta": float(self.delta),
            "thresholds": thresholds,
            "step_scores_raw": step_scores,
            "feature_len": int(feat.shape[0]),
            "feature_source": self.feature_source,
            "transformer_layer": int(self.transformer_layer),
            "score_mode": "oe_density",
            "calib_mode": self.calib_mode,
            "calib_method_used": self._calib_method_per_task.get(task, self.calib_mode),
            "lam": float(self.lam),
            "fail_loss_form": "logistic",
            "score_k": int(self.score_k),
            "view_names": list(self.encoder.view_names),
        }
        return DiscriminatorOutput(
            step_scores=step_scores,
            predictions=preds,
            first_failure_frame=first_failure_frame,
            aux=aux,
        )

    def calibration_summary(self) -> dict:
        return {
            "per_task": dict(self._calibration_stats),
            "model_ckpt": self.model_ckpt,
            "view_names": list(self.encoder.view_names),
            "camera_to_view": dict(self.camera_to_view),
            "feature_source": self.feature_source,
            "transformer_layer": int(self.transformer_layer),
            "delta": float(self.delta),
            "calib_fraction": float(self.calib_fraction),
            "encode_batch_size": int(self.encode_batch_size),
            "score_mode": "oe_density",
            "calib_mode": self.calib_mode,
            "score_k": int(self.score_k),
            "lam": float(self.lam),
            "fail_loss_form": "logistic",
            "weight_decay": float(self.weight_decay),
            "density": self.density,
            "num_epochs": int(self.num_epochs),
            "batch_size": int(self.batch_size),
            "learning_rate": float(self.learning_rate),
            "num_tasks": int(len(self._known_tasks)),
            "known_tasks": list(self._known_tasks),
            "ftrain_last_k": int(self.ftrain_last_k),
            "fail_use_all_after_gt": bool(self.fail_use_all_after_gt),
            "fail_prefix_to_succ": bool(self.fail_prefix_to_succ),
            "ftrain_index_ranges": dict(self._ftrain_index_ranges),
            "ftrain_prefix_index_ranges": dict(self._ftrain_prefix_index_ranges),
            "tau_per_task": dict(self._tau_per_task),
        }

    @property
    def train_history(self) -> List[dict]:
        return list(self._train_history)
