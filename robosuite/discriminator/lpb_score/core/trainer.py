"""Training loop: dataloaders, AMP, DDP, normalization warmup, DSM + aux loss.

Per-step algorithm (``_run_step``):
    Sample ``sigma`` ~ log-uniform in ``[sigma_min, sigma_max]``, Gaussian noise on
    latent window scaled by ``sigma``. Forward ``DSMModel`` with
    ``return_condition_pair=True`` to obtain reconstructions under traj-type 0 and 1.
    DSM loss uses MSE of predicted vs clean chunk on the branch matching the label;
    auxiliary contrastive loss encourages margin between the two branches. Optional
    Welford warmup on positive samples initializes latent normalization before epoch 1.
"""

from __future__ import annotations

import copy
import io
import inspect
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Callable, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

from .dataset import BatchedTrainLoader, LatentTransitionDataset
from .model import DSMModel
from .profiling import StepProfiler


@dataclass
class TrainerConfig:
    """Hyperparameters for ``Trainer``: optimization, loaders, AMP/DDP, DSM losses, warmup caps."""

    batch_size: int = 64
    num_workers: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 50
    positive_sampling_ratio: float = 0.5
    grad_clip_norm: float = 1.0
    log_every: int = 50
    device: str = "cuda"
    use_ema: bool = False
    ema_decay: float = 0.999
    val_use_ema: bool = True
    save_ema_in_checkpoint: bool = True
    shared_lr_multiplier: float = 1.0
    chunk_branch_lr_multiplier: float = 1.0
    encoder_branch_lr_multiplier: float = 1.0
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4
    profile_enabled: bool = False
    profile_log_every: int = 20
    profile_warmup_steps: int = 0
    profile_cuda_sync: bool = True
    profile_first_step_immediate: bool = True
    profile_norm_progress_every: int = 10
    input_pipeline: str = "batched_iterator"
    val_input_pipeline: str = "legacy_dataloader"
    batch_prefetch_depth: int = 2
    train_prefetch_pinned: bool = True
    train_prefetch_thread: bool = True
    train_batch_build_workers: int = 4
    cuda_prefetch: bool = True
    batched_trajectory_cache_gb: float = 8.0
    amp_enabled: bool = True
    amp_dtype: str = "bf16"
    seed: int = 0
    distributed: bool = False
    distributed_rank: int = 0
    distributed_world_size: int = 1
    distributed_local_rank: int = 0
    ddp_find_unused_parameters: bool = False
    ddp_static_graph: bool = True
    ddp_gradient_as_bucket_view: bool = True
    aux_contrastive_weight: float = 0.1
    aux_contrastive_margin: float = 1e-2
    # Normalization warmup budget: stop at first hit of either bound.
    normalization_warmup_max_samples: int = 5000
    normalization_warmup_max_batches: int = 20


@dataclass
class WelfordState:
    """Online sufficient stats for per-dimension mean and variance (Welford)."""

    count: int
    mean: torch.Tensor
    m2: torch.Tensor

    @classmethod
    def zeros(cls, dim: int, *, device: torch.device) -> "WelfordState":
        return cls(
            count=0,
            mean=torch.zeros((int(dim),), dtype=torch.float64, device=device),
            m2=torch.zeros((int(dim),), dtype=torch.float64, device=device),
        )


class Trainer:
    """Owns optimizer(s), dataloaders (standard or batched trajectory iterator), and ``fit``.

    Training stages:
        1. ``fit`` -> ``_run_normalization_warmup`` (positive-only latent stats).
        2. Each epoch: ``train_one_epoch`` (``_run_step`` with gradients) then ``validate``.
    ``_run_step`` samples one ``sigma`` per step (shared across the batch), builds Gaussian
    noise in latent space, runs ``DSMModel`` with ``return_condition_pair=True``, and combines
    DSM + auxiliary contrastive losses (see module docstring).
    """

    def __init__(
        self,
        model: DSMModel,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        config: Optional[TrainerConfig] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = config or TrainerConfig()
        self.is_distributed = bool(self.cfg.distributed)
        self.rank = int(self.cfg.distributed_rank)
        self.world_size = int(self.cfg.distributed_world_size)
        self.local_rank = int(self.cfg.distributed_local_rank)
        self.is_main_process = self.rank == 0

        # Device: DDP uses LOCAL_RANK; single-process uses cfg.training.device.
        resolved_device = device or self.cfg.device
        if str(resolved_device).lower().startswith("cuda") and not torch.cuda.is_available():
            resolved_device = "cpu"
        self.device = torch.device(resolved_device)
        self.model.to(self.device)
        # Forward/backward target: DDP wrapper in multi-GPU mode, else raw model.
        self._train_model: DSMModel | DistributedDataParallel = self.model
        if self.is_distributed:
            if self.device.type != "cuda":
                raise ValueError("DDP training currently requires CUDA.")
            find_unused_parameters = bool(self.cfg.ddp_find_unused_parameters)
            static_graph = bool(self.cfg.ddp_static_graph) and not find_unused_parameters
            if bool(self.cfg.ddp_static_graph) and find_unused_parameters and self.is_main_process:
                self._log(
                    "[lpb_score] disabling DDP static_graph because "
                    "ddp_find_unused_parameters=true."
                )
            ddp_signature = inspect.signature(DistributedDataParallel.__init__)
            ddp_kwargs: dict[str, object] = {}
            if "gradient_as_bucket_view" in ddp_signature.parameters:
                ddp_kwargs["gradient_as_bucket_view"] = bool(self.cfg.ddp_gradient_as_bucket_view)
            if "static_graph" in ddp_signature.parameters:
                ddp_kwargs["static_graph"] = bool(static_graph)
            self._train_model = DistributedDataParallel(
                self.model,
                device_ids=[int(self.device.index)],
                output_device=int(self.device.index),
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters,
                **ddp_kwargs,
            )
            self._log(
                f"[lpb_score] enabling DDP rank={self.rank} local_rank={self.local_rank} "
                f"world_size={self.world_size} device={self.device} "
                f"find_unused={find_unused_parameters} static_graph={bool(static_graph)} "
                f"bucket_view={bool(self.cfg.ddp_gradient_as_bucket_view)}"
            )

        # Optional EMA weights for validation / checkpointing.
        self.ema_model: Optional[DSMModel] = None
        if self.cfg.use_ema:
            self.ema_model = copy.deepcopy(self.model)
            self.ema_model.to(self.device)
            self.ema_model.eval()

        # Parameter groups: encoder vs chunk DSM vs shared (multipliers from cfg).
        self.optimizer = torch.optim.AdamW(
            self._build_optimizer_param_groups(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        self._amp_enabled = bool(self.cfg.amp_enabled) and self.device.type == "cuda"
        self._amp_dtype = self._resolve_amp_dtype(str(self.cfg.amp_dtype))
        self._use_grad_scaler = self._amp_enabled and self._amp_dtype == torch.float16
        self._grad_scaler = torch.amp.GradScaler(
            enabled=self._use_grad_scaler,
            device=self.device.type,
        )

        # Train: BatchedTrainLoader (default) co-batches windows from same trajectories; val: classic DataLoader.
        self.train_loader = self._build_train_loader()
        self.val_loader = self._build_val_loader()
        self._profile_accumulator: dict[str, dict[str, float]] = {"train": {}, "valid": {}}
        self._profile_counts: dict[str, int] = {"train": 0, "valid": 0}
        self._train_rng = self._new_rng(seed_offset=0)
        self._eval_rng = self._new_rng(seed_offset=100_000)
        self._normalization_warmup_done = False
        self._wandb_module = None
        self.wandb_run = self._init_wandb()
        self._global_train_step = 0
        self._global_valid_step = 0
        # Log-space bin edges for logging sigma strata (four bins via three interior boundaries).
        log_min = math.log(float(self.model.sigma_min))
        log_max = math.log(float(self.model.sigma_max))
        if abs(log_max - log_min) < 1e-12:
            self._sigma_log_boundaries: tuple[float, float, float] = (log_max, log_max, log_max)
        else:
            step = (log_max - log_min) / 4.0
            self._sigma_log_boundaries = (
                log_min + step,
                log_min + 2.0 * step,
                log_min + 3.0 * step,
            )

    def _log(self, message: str) -> None:
        if self.is_main_process:
            print(message)

    def _new_rng(self, *, seed_offset: int) -> torch.Generator:
        seed_value = int(self.cfg.seed) + int(seed_offset) + int(self.rank) * 10_000
        if self.device.type == "cuda":
            generator = torch.Generator(device=self.device)
        else:
            generator = torch.Generator()
        generator.manual_seed(seed_value)
        return generator

    @staticmethod
    def _resolve_amp_dtype(name: str) -> torch.dtype:
        normalized = str(name).strip().lower()
        if normalized in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if normalized in {"fp16", "float16", "half"}:
            return torch.float16
        raise ValueError(f"Unsupported amp_dtype={name!r}. Expected 'bf16' or 'fp16'.")

    def _build_optimizer_param_groups(self) -> list[dict[str, object]]:
        multipliers = {
            "encoder_branch": float(self.cfg.encoder_branch_lr_multiplier),
            "shared": float(self.cfg.shared_lr_multiplier),
            "chunk_branch": float(self.cfg.chunk_branch_lr_multiplier),
        }
        for name, multiplier in multipliers.items():
            if multiplier < 0.0:
                raise ValueError(f"{name} lr multiplier must be non-negative, got {multiplier}")

        groups = self.model.optimizer_parameter_groups()
        param_groups: list[dict[str, object]] = []
        for name in ("encoder_branch", "shared", "chunk_branch"):
            params = groups.get(name, [])
            if not params:
                continue
            param_groups.append(
                {
                    "params": params,
                    "lr": self.cfg.learning_rate * multipliers[name],
                    "weight_decay": self.cfg.weight_decay,
                    "name": name,
                }
            )
        return param_groups

    def _resolve_positive_labels(self, dataset: Dataset) -> Optional[list[bool]]:
        if isinstance(dataset, LatentTransitionDataset):
            return dataset.sample_is_positive
        if isinstance(dataset, Subset) and isinstance(dataset.dataset, LatentTransitionDataset):
            base = dataset.dataset.sample_is_positive
            return [base[i] for i in dataset.indices]
        return None

    def _build_val_loader(self):
        """Validation DataLoader when ``val_input_pipeline`` is ``legacy_dataloader`` and dataset non-empty."""
        if (
            self.val_dataset is None
            or len(self.val_dataset) <= 0
            or str(self.cfg.val_input_pipeline) != "legacy_dataloader"
        ):
            return None
        sampler = None
        if self.is_distributed:
            sampler = DistributedSampler(
                self.val_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
                drop_last=False,
            )
        return DataLoader(
            self.val_dataset,
            batch_size=max(1, int(self.cfg.batch_size) // self.world_size) if self.is_distributed else self.cfg.batch_size,
            shuffle=False,
            sampler=sampler,
            **self._loader_kwargs(),
        )

    def _build_legacy_train_loader(self):
        """Shuffle or weighted sampling over window indices; used when not using ``BatchedTrainLoader``."""
        if self.is_distributed:
            sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=False,
                seed=self.cfg.seed,
            )
            return DataLoader(
                self.train_dataset,
                batch_size=max(1, int(self.cfg.batch_size) // self.world_size),
                sampler=sampler,
                **self._loader_kwargs(),
            )
        labels = self._resolve_positive_labels(self.train_dataset)
        use_balanced = labels is not None
        if labels is None:
            n_pos, n_neg = 0, 0
        else:
            n_pos = int(sum(labels))
            n_neg = int(len(labels) - n_pos)
            use_balanced = (n_pos > 0) and (n_neg > 0)

        if use_balanced:
            ratio = float(self.cfg.positive_sampling_ratio)
            ratio = min(max(ratio, 0.0), 1.0)
            w_pos = ratio / float(n_pos)
            w_neg = (1.0 - ratio) / float(n_neg)
            weights = torch.tensor([w_pos if x else w_neg for x in labels], dtype=torch.double)
            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(weights),
                replacement=True,
            )
            return DataLoader(
                self.train_dataset,
                batch_size=self.cfg.batch_size,
                sampler=sampler,
                **self._loader_kwargs(),
            )

        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            **self._loader_kwargs(),
        )

    def _build_train_loader(self):
        """Return ``BatchedTrainLoader`` (default) or legacy ``DataLoader`` with balanced positive ratio."""
        if str(self.cfg.input_pipeline) == "batched_iterator":
            if not isinstance(self.train_dataset, LatentTransitionDataset):
                raise TypeError("batched_iterator train pipeline requires LatentTransitionDataset.")
            return BatchedTrainLoader(
                self.train_dataset,
                batch_size=self.cfg.batch_size,
                positive_sampling_ratio=self.cfg.positive_sampling_ratio,
                seed=self.cfg.seed,
                pin_memory=bool(self.cfg.train_prefetch_pinned) and self.device.type == "cuda",
                prefetch_depth=int(self.cfg.batch_prefetch_depth),
                use_prefetch_thread=bool(self.cfg.train_prefetch_thread),
                batch_build_workers=int(self.cfg.train_batch_build_workers),
                trajectory_cache_limit_gb=float(self.cfg.batched_trajectory_cache_gb),
                num_replicas=self.world_size if self.is_distributed else 1,
                rank=self.rank if self.is_distributed else 0,
            )

        return self._build_legacy_train_loader()

    def _loader_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "num_workers": int(self.cfg.num_workers),
            "pin_memory": bool(self.cfg.pin_memory) and self.device.type == "cuda",
        }
        if int(self.cfg.num_workers) > 0:
            kwargs["persistent_workers"] = bool(self.cfg.persistent_workers)
            kwargs["prefetch_factor"] = max(2, int(self.cfg.prefetch_factor))
        return kwargs

    def _move_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Copy tensor batch fields to ``self.device`` (non-blocking for overlap with prefetch)."""
        return {
            key: value.to(self.device, non_blocking=True)
            for key, value in batch.items()
            if torch.is_tensor(value)
        }

    class _CUDAPrefetchIterator:
        """Overlap host dataloader with async H2D copy on a side CUDA stream."""

        def __init__(self, trainer: "Trainer", loader) -> None:
            self._trainer = trainer
            self._loader_iter = iter(loader)
            self._stream = torch.cuda.Stream(device=trainer.device)
            self._next_batch: dict[str, torch.Tensor] | None = None
            self._preload()

        def _preload(self) -> None:
            try:
                batch = next(self._loader_iter)
            except StopIteration:
                self._next_batch = None
                return
            with torch.cuda.stream(self._stream):
                self._next_batch = self._trainer._move_batch(batch)

        def __iter__(self):
            return self

        def __next__(self) -> dict[str, torch.Tensor]:
            if self._next_batch is None:
                raise StopIteration
            current_stream = torch.cuda.current_stream(device=self._trainer.device)
            current_stream.wait_stream(self._stream)
            batch = self._next_batch
            for value in batch.values():
                if torch.is_tensor(value):
                    value.record_stream(current_stream)
            self._preload()
            return batch

    def _loader_iter(self, loader):
        """Iterable over ``loader``; optionally CUDA-stream prefetch for faster steps."""
        if bool(self.cfg.cuda_prefetch) and self.device.type == "cuda":
            return self._CUDAPrefetchIterator(self, loader)
        return iter(loader)

    def _new_step_profiler(self) -> StepProfiler:
        return StepProfiler(
            enabled=bool(self.cfg.profile_enabled),
            sync_cuda=bool(self.cfg.profile_cuda_sync) and self.device.type == "cuda",
        )

    def _record_profile(
        self,
        *,
        phase: str,
        epoch: int,
        step: int,
        profiler: StepProfiler,
    ) -> None:
        if not bool(self.cfg.profile_enabled):
            return
        if int(step) < int(self.cfg.profile_warmup_steps):
            return
        timings = profiler.snapshot_ms()
        if not timings:
            return
        accum = self._profile_accumulator[phase]
        for key, value in timings.items():
            accum[key] = float(accum.get(key, 0.0)) + float(value)
        self._profile_counts[phase] += 1
        log_every = max(1, int(self.cfg.profile_log_every))
        should_log = False
        if bool(self.cfg.profile_first_step_immediate) and self._profile_counts[phase] == 1:
            should_log = True
        elif self._profile_counts[phase] % log_every == 0:
            should_log = True
        if not should_log:
            return
        averaged = {
            key: float(value) / float(self._profile_counts[phase])
            for key, value in sorted(accum.items())
        }
        ordered_keys = [
            "dataloader_wait",
            "h2d",
            "forward_total",
            "task_name_build",
            "encoder_total",
            "prompt_resolve",
            "flow_encode_context",
            "flow_image_encoder",
            "flow_language_encoder",
            "lang_tokenizer",
            "lang_backbone",
            "flow_fusion",
            "predictor",
            "backward",
            "optim_step",
        ]
        parts = []
        for key in ordered_keys:
            if key in averaged:
                parts.append(f"{key}={averaged[key]:.1f}ms")
        for key, value in averaged.items():
            if key not in ordered_keys:
                parts.append(f"{key}={value:.1f}ms")
        self._log(
            f"[profile {phase}] epoch={epoch:03d} step={step:05d} "
            + " ".join(parts)
        )

    @torch.no_grad()
    def _update_ema(self) -> None:
        if self.ema_model is None:
            return
        decay = float(self.cfg.ema_decay)
        if not (0.0 <= decay < 1.0):
            raise ValueError(f"ema_decay must be in [0, 1), got {decay}")
        for p_ema, p in zip(self.ema_model.parameters(), self.model.parameters()):
            p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)
        for b_ema, b in zip(self.ema_model.buffers(), self.model.buffers()):
            b_ema.data.copy_(b.data)

    def _sample_sigma(self, *, batch_size: int, train: bool, model: DSMModel) -> torch.Tensor:
        """One scalar ``sigma`` per step (log-uniform in [sigma_min, sigma_max]), broadcast to batch."""
        generator = self._train_rng if train else self._eval_rng
        if str(model.sigma_distribution) != "log_uniform":
            raise ValueError(f"Unsupported sigma_distribution={model.sigma_distribution!r}")
        log_min = math.log(float(model.sigma_min))
        log_max = math.log(float(model.sigma_max))
        sigma_scalar = torch.exp(
            torch.rand((1,), device=self.device, generator=generator, dtype=torch.float32) * (log_max - log_min)
            + log_min
        )
        return sigma_scalar.expand(int(batch_size))

    def _sigma_bin_index(self, sigma_value: float) -> int:
        """Bucket ``sigma`` into 0..3 for wandb stratified metrics."""
        log_sigma = math.log(max(float(sigma_value), 1e-12))
        boundaries = self._sigma_log_boundaries
        return int(log_sigma >= boundaries[0]) + int(log_sigma >= boundaries[1]) + int(log_sigma >= boundaries[2])

    def _init_wandb(self):
        if not self.is_main_process:
            return None
        mode = str(os.environ.get("WANDB_MODE", "offline"))
        if mode.lower() == "disabled":
            return None
        try:
            import wandb
        except Exception:
            self._log("[lpb_score] wandb is unavailable, skipping logging.")
            return None

        entity = str(os.environ.get("WANDB_ENTITY", os.environ.get("WANDB_NAME", "songgao-personal")))
        project = str(os.environ.get("WANDB_PROJECT", "robosuite-lpb-score"))
        os.environ.setdefault("WANDB_MODE", mode)
        os.environ.setdefault("WANDB_ENTITY", entity)
        self._wandb_module = wandb
        return wandb.init(
            project=project,
            entity=entity,
            mode=mode,
            config={
                "trainer": asdict(self.cfg),
                "model": {
                    "sigma_min": float(self.model.sigma_min),
                    "sigma_max": float(self.model.sigma_max),
                    "sigma_distribution": str(self.model.sigma_distribution),
                    "num_mc_samples": int(self.model.num_mc_samples),
                    "std_clamp_min": float(self.model.std_clamp_min),
                },
            },
        )

    def _log_wandb_step(self, *, phase: str, epoch: int, step: int, metrics: dict[str, float]) -> None:
        if self.wandb_run is None:
            return
        payload = {
            f"{phase}/{key}": float(value)
            for key, value in metrics.items()
            if key not in {"sigma_bin_index"}
        }
        sigma_bin_index = int(metrics.get("sigma_bin_index", 0))
        payload[f"{phase}/sigma_bin_{sigma_bin_index}_loss_dsm"] = float(metrics["loss_dsm"])
        payload[f"{phase}/sigma_bin_{sigma_bin_index}_loss_total"] = float(metrics["loss"])
        payload["epoch"] = int(epoch)
        payload[f"{phase}_step"] = int(step)
        self.wandb_run.log(payload)

    def _log_wandb_warmup(
        self,
        *,
        latent_abs_before: torch.Tensor,
        latent_abs_after: torch.Tensor,
        latent_count: int,
    ) -> None:
        if self.wandb_run is None or self._wandb_module is None:
            return
        payload = {
            "warmup/latent_count": float(latent_count),
            "warmup/latent_abs_before_hist": self._wandb_module.Histogram(latent_abs_before.cpu().numpy()),
            "warmup/latent_abs_after_hist": self._wandb_module.Histogram(latent_abs_after.cpu().numpy()),
            "warmup/latent_abs_before_mean": float(latent_abs_before.mean().item()),
            "warmup/latent_abs_after_mean": float(latent_abs_after.mean().item()),
        }
        self.wandb_run.log(payload)

    def _finish_wandb(self) -> None:
        if self.wandb_run is not None:
            self.wandb_run.finish()
            self.wandb_run = None

    def _build_positive_warmup_loader(self):
        """Subset of train windows with positive ``traj_type`` only (for Welford stats)."""
        if not isinstance(self.train_dataset, LatentTransitionDataset):
            raise TypeError("Normalization warmup requires LatentTransitionDataset.")
        refs = self.train_dataset.sample_refs_array
        positive_indices = [int(idx) for idx in torch.nonzero(torch.from_numpy(refs[:, 3] == 0), as_tuple=False).view(-1)]
        if not positive_indices:
            raise RuntimeError("Normalization warmup requires at least one positive sample.")
        if self.is_distributed:
            positive_indices = positive_indices[self.rank :: self.world_size]
        subset = Subset(self.train_dataset, positive_indices)
        return DataLoader(
            subset,
            batch_size=max(1, int(self.cfg.batch_size) // self.world_size) if self.is_distributed else self.cfg.batch_size,
            shuffle=False,
            **self._loader_kwargs(),
        )

    @staticmethod
    def _merge_welford_states(state_a: WelfordState, state_b: WelfordState) -> WelfordState:
        if int(state_b.count) <= 0:
            return state_a
        if int(state_a.count) <= 0:
            return state_b
        total = int(state_a.count + state_b.count)
        delta = state_b.mean - state_a.mean
        mean = state_a.mean + delta * (float(state_b.count) / float(total))
        m2 = (
            state_a.m2
            + state_b.m2
            + torch.square(delta) * (float(state_a.count) * float(state_b.count) / float(total))
        )
        return WelfordState(count=total, mean=mean, m2=m2)

    def _update_welford_state(self, state: WelfordState, flat_latents: torch.Tensor) -> WelfordState:
        if int(flat_latents.shape[0]) <= 0:
            return state
        batch = flat_latents.to(dtype=torch.float64)
        batch_count = int(batch.shape[0])
        batch_mean = batch.mean(dim=0)
        batch_m2 = torch.square(batch - batch_mean).sum(dim=0)
        batch_state = WelfordState(count=batch_count, mean=batch_mean, m2=batch_m2)
        return self._merge_welford_states(state, batch_state)

    def _reduce_welford_state(self, state: WelfordState) -> WelfordState:
        if not self.is_distributed:
            return state
        gathered: list[dict[str, object] | None] = [None] * int(self.world_size)
        dist.all_gather_object(
            gathered,
            {
                "count": int(state.count),
                "mean": state.mean.detach().cpu(),
                "m2": state.m2.detach().cpu(),
            },
        )
        merged = WelfordState.zeros(self.model.latent_dim, device=self.device)
        for item in gathered:
            assert item is not None
            merged = self._merge_welford_states(
                merged,
                WelfordState(
                    count=int(item["count"]),
                    mean=torch.as_tensor(item["mean"], dtype=torch.float64, device=self.device),
                    m2=torch.as_tensor(item["m2"], dtype=torch.float64, device=self.device),
                ),
            )
        return merged

    def _roundtrip_normalization_check(self, sample_latent: torch.Tensor) -> None:
        sample_cpu = sample_latent.detach().cpu()
        reference = self.model.normalize_latent(sample_latent.to(self.device)).detach().cpu()
        payload = io.BytesIO()
        torch.save({"model": self.model.state_dict()}, payload)
        payload.seek(0)
        loaded = torch.load(payload, map_location="cpu")
        state_dict = loaded["model"]
        loaded_mean = torch.as_tensor(state_dict["latent_mean"], dtype=torch.float32)
        loaded_var = torch.as_tensor(state_dict["latent_var"], dtype=torch.float32)
        loaded_enable = bool(torch.as_tensor(state_dict["normalize_enabled"]).item())
        if loaded_enable:
            std = torch.clamp(torch.sqrt(loaded_var), min=float(self.model.std_clamp_min))
            loaded_norm = (sample_cpu - loaded_mean.view(1, 1, -1)) / std.view(1, 1, -1)
        else:
            loaded_norm = sample_cpu
        if not torch.allclose(reference, loaded_norm, atol=1e-6, rtol=1e-6):
            raise RuntimeError("Normalization stats did not round-trip through torch.save / torch.load.")

    @torch.no_grad()
    def _run_normalization_warmup(self) -> None:
        """Estimate ``latent_mean`` / ``latent_var`` from encoded positive windows before epoch 1.

        Uses Welford over flattened latents; DDP merges states. Stops early when
        ``normalization_warmup_max_samples`` or ``normalization_warmup_max_batches`` hits.
        """
        if self._normalization_warmup_done:
            return
        loader = self._build_positive_warmup_loader()
        was_training = self.model.training
        self.model.eval()
        state = WelfordState.zeros(self.model.latent_dim, device=self.device)
        sampled_latents: list[torch.Tensor] = []
        sample_budget = 512
        total_batches = len(loader)
        start_time = time.perf_counter()

        max_samples = int(self.cfg.normalization_warmup_max_samples)
        max_batches = int(self.cfg.normalization_warmup_max_batches)
        for step, batch in enumerate(loader):
            data = self._move_batch(batch)
            latents = self.model.encode_latent_window(
                image_window=data["image_window"],
                proprio_window=data["proprio_window"],
                task_index=data["task_index"],
            )
            flat = latents.reshape(-1, self.model.latent_dim)
            state = self._update_welford_state(state, flat)

            if sample_budget > 0:
                take = min(sample_budget, int(latents.shape[0]))
                sampled_latents.append(latents[:take].detach().cpu())
                sample_budget -= take

            if self.is_main_process and (
                step == 0 or (step + 1) % max(1, int(self.cfg.profile_norm_progress_every)) == 0 or (step + 1) == total_batches
            ):
                elapsed = time.perf_counter() - start_time
                self._log(
                    f"[lpb_score] normalization_warmup step={step + 1}/{total_batches} "
                    f"count={state.count} elapsed_s={elapsed:.1f}"
                )

            # Early-stop once stats are sufficiently estimated: random sample is fine.
            if max_samples > 0 and int(state.count) >= max_samples:
                if self.is_main_process:
                    self._log(
                        f"[lpb_score] normalization_warmup early-stop by samples "
                        f"count={state.count} >= max_samples={max_samples} step={step + 1}"
                    )
                break
            if max_batches > 0 and (step + 1) >= max_batches:
                if self.is_main_process:
                    self._log(
                        f"[lpb_score] normalization_warmup early-stop by batches "
                        f"step={step + 1} >= max_batches={max_batches} count={state.count}"
                    )
                break

        state = self._reduce_welford_state(state)
        if int(state.count) <= 0:
            raise RuntimeError("Normalization warmup observed zero positive latent samples.")
        latent_var = torch.clamp(state.m2 / float(state.count), min=1e-6)

        # perform normalization stats update
        self.model.set_normalization_stats(
            latent_mean=state.mean.to(dtype=torch.float32),
            latent_var=latent_var.to(dtype=torch.float32),
        )
        if was_training:
            self.model.train()

        sampled = (
            torch.cat(sampled_latents, dim=0).to(self.device)
            if sampled_latents
            else torch.zeros((1, self.model.window_size, self.model.latent_dim), dtype=torch.float32, device=self.device)
        )
        latent_abs_before = sampled.abs().mean(dim=(1, 2)).detach().cpu()
        latent_abs_after = self.model.normalize_latent(sampled).abs().mean(dim=(1, 2)).detach().cpu()
        self._roundtrip_normalization_check(sampled[:1])

        # Log normalization warmup stats
        self._log(
            "[lpb_score] normalization_warmup "
            f"count={state.count} latent_abs_before_mean={float(latent_abs_before.mean().item()):.6f} "
            f"latent_abs_after_mean={float(latent_abs_after.mean().item()):.6f}"
        )
        self._log_wandb_warmup(
            latent_abs_before=latent_abs_before,
            latent_abs_after=latent_abs_after,
            latent_count=int(state.count),
        )
        self._normalization_warmup_done = True

    def _run_step(
        self,
        batch: dict[str, torch.Tensor],
        train: bool,
        model: Optional[DSMModel] = None,
        profiler: Optional[StepProfiler] = None,
    ) -> dict[str, float]:
        """Single train/val step: sample ``sigma``, noise latents, dual-branch forward, DSM + aux loss."""
        active = model if model is not None else self.model
        step_profiler = profiler or self._new_step_profiler()
        if self.device.type == "cuda" and all(
            torch.is_tensor(value) and value.device == self.device
            for value in batch.values()
            if torch.is_tensor(value)
        ):
            data = {key: value for key, value in batch.items() if torch.is_tensor(value)}
        else:
            with step_profiler.section("h2d"):
                data = self._move_batch(batch)
        if train:
            self.optimizer.zero_grad(set_to_none=True)

        batch_size = int(data["traj_type"].shape[0])
        sigma_tensor = self._sample_sigma(batch_size=batch_size, train=train, model=active)
        # Isotropic Gaussian noise in latent space, scaled per-sample by sigma (same sigma for whole batch).
        latent_window_noise = (
            torch.randn(
                (batch_size, active.window_size, active.latent_dim),
                device=self.device,
                dtype=torch.float32,
                generator=self._train_rng if train else self._eval_rng,
            )
            * sigma_tensor.view(batch_size, 1, 1)
        )

        # Under DDP, always forward through wrapped module; eval may use EMA copy.
        train_model = self._train_model if active is self.model else active
        with torch.set_grad_enabled(train):
            with torch.amp.autocast(
                enabled=self._amp_enabled,
                device_type=self.device.type,
                dtype=self._amp_dtype,
            ):
                with step_profiler.section("forward_total"):
                    out = train_model(
                        image_window=data["image_window"],
                        proprio_window=data["proprio_window"],
                        traj_type=data["traj_type"],
                        task_index=data["task_index"],
                        sigma=sigma_tensor,
                        latent_window_noise=latent_window_noise,
                        return_condition_pair=True,
                        profiler=step_profiler,
                    )
                    # Branch energies: d0 = recon error if predictor thinks "positive", d1 = "failure".
                    positive_recon = active.reconstruction_components(
                        chunk_clean=out["chunk_clean_norm"],
                        chunk_hat=out["positive"]["chunk_hat_norm"],
                    )
                    negative_recon = active.reconstruction_components(
                        chunk_clean=out["chunk_clean_norm"],
                        chunk_hat=out["negative"]["chunk_hat_norm"],
                    )
                    traj_type = out["traj_type"].reshape(-1)
                    d0 = positive_recon["chunk_energy_per_sample"]
                    d1 = negative_recon["chunk_energy_per_sample"]
                    # DSM: supervise the branch that matches dataset traj_type (0=pos, 1=neg).
                    loss_dsm_per_sample = torch.where(traj_type == 0, d0, d1)
                    margin = float(self.cfg.aux_contrastive_margin)
                    # Aux: hinge margin so the label-consistent branch is lower energy than the other.
                    loss_aux_per_sample = torch.where(
                        traj_type == 0,
                        torch.relu(d0 - d1 + margin),
                        torch.relu(d1 - d0 + margin),
                    )
                    loss_dsm = loss_dsm_per_sample.mean()
                    loss_aux = loss_aux_per_sample.mean()
                    loss = loss_dsm + float(self.cfg.aux_contrastive_weight) * loss_aux
                    stats = {
                        "loss": loss,
                        "score": loss,
                        "unweighted_score": loss_dsm,
                        "chunk_mse": loss_dsm,
                        "chunk_energy": loss_dsm,
                        "loss_dsm": loss_dsm,
                        "loss_aux": loss_aux,
                        "sigma": sigma_tensor[0],
                    }
            if train:
                with step_profiler.section("backward"):
                    self._grad_scaler.scale(stats["loss"]).backward()
                if self.cfg.grad_clip_norm > 0:
                    if self._use_grad_scaler:
                        self._grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
                with step_profiler.section("optim_step"):
                    self._grad_scaler.step(self.optimizer)
                    self._grad_scaler.update()
                if self.ema_model is not None:
                    self._update_ema()

        sigma_value = float(stats["sigma"].detach().item())
        return {
            "loss": float(stats["loss"].detach().item()),
            "score": float(stats["score"].detach().item()),
            "unweighted_score": float(stats["unweighted_score"].detach().item()),
            "chunk_mse": float(stats["chunk_mse"].detach().item()),
            "chunk_energy": float(stats["chunk_energy"].detach().item()),
            "loss_dsm": float(stats["loss_dsm"].detach().item()),
            "loss_aux": float(stats["loss_aux"].detach().item()),
            "sigma": sigma_value,
            "sigma_bin_index": float(self._sigma_bin_index(sigma_value)),
        }

    @staticmethod
    def _mean_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
        if not metrics:
            return {}
        keys = sorted(key for key in metrics[0].keys() if key != "sigma_bin_index")
        return {key: float(sum(item[key] for item in metrics) / len(metrics)) for key in keys}

    def _reduce_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        if not self.is_distributed or not metrics:
            return metrics
        keys = sorted(metrics.keys())
        values = torch.tensor([float(metrics[key]) for key in keys], dtype=torch.float64, device=self.device)
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values /= float(self.world_size)
        return {key: float(value.item()) for key, value in zip(keys, values)}

    def _set_loader_epoch(self, loader, epoch: int) -> None:
        sampler = getattr(loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(int(epoch) - 1)

    def eval_model_for_inference(self) -> DSMModel:
        """Return EMA copy for validation when enabled, else the training module."""
        if self.cfg.use_ema and self.cfg.val_use_ema and self.ema_model is not None:
            return self.ema_model
        return self.model

    def checkpoint_state_dicts(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor] | None]:
        """Primary weights for saving; optionally second state dict (online vs EMA) per cfg."""
        if self.ema_model is None:
            return self.model.state_dict(), None
        if self.cfg.save_ema_in_checkpoint:
            return self.ema_model.state_dict(), self.model.state_dict()
        return self.model.state_dict(), None

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        """Run one full pass over ``train_loader``; mean-reduced metrics (DDP-allreduced)."""
        self.model.train()
        self._set_loader_epoch(self.train_loader, epoch)
        logs: list[dict[str, float]] = []
        loader_iter = self._loader_iter(self.train_loader)
        for step in range(len(self.train_loader)):
            profiler = self._new_step_profiler()
            with profiler.section("dataloader_wait"):
                batch = next(loader_iter)
            out = self._run_step(batch, train=True, profiler=profiler)
            logs.append(out)
            self._record_profile(phase="train", epoch=epoch, step=step, profiler=profiler)
            self._log_wandb_step(phase="train", epoch=epoch, step=self._global_train_step, metrics=out)
            self._global_train_step += 1
            if self.is_main_process and self.cfg.log_every > 0 and step % self.cfg.log_every == 0:
                self._log(
                    f"[train] epoch={epoch:03d} step={step:05d} "
                    f"loss={out['loss']:.6f} loss_dsm={out['loss_dsm']:.6f} "
                    f"loss_aux={out['loss_aux']:.6f} sigma={out['sigma']:.6f} "
                    f"chunk_energy={out['chunk_energy']:.6f}"
                )
        return self._reduce_metrics(self._mean_metrics(logs))

    @torch.no_grad()
    def validate(self, epoch: int) -> dict[str, float]:
        """Same loss computation as train but no backward; optional EMA weights."""
        if self.val_loader is None:
            return {}
        eval_model = self.eval_model_for_inference()
        eval_model.eval()
        self._set_loader_epoch(self.val_loader, epoch)
        logs: list[dict[str, float]] = []
        loader_iter = self._loader_iter(self.val_loader)
        for step in range(len(self.val_loader)):
            profiler = self._new_step_profiler()
            with profiler.section("dataloader_wait"):
                batch = next(loader_iter)
            out = self._run_step(batch, train=False, model=eval_model, profiler=profiler)
            logs.append(out)
            self._record_profile(phase="valid", epoch=epoch, step=step, profiler=profiler)
            self._log_wandb_step(phase="valid", epoch=epoch, step=self._global_valid_step, metrics=out)
            self._global_valid_step += 1
            if self.is_main_process and self.cfg.log_every > 0 and step % self.cfg.log_every == 0:
                self._log(
                    f"[valid] epoch={epoch:03d} step={step:05d} "
                    f"loss={out['loss']:.6f} loss_dsm={out['loss_dsm']:.6f} "
                    f"loss_aux={out['loss_aux']:.6f} sigma={out['sigma']:.6f} "
                    f"chunk_energy={out['chunk_energy']:.6f}"
                )
        return self._reduce_metrics(self._mean_metrics(logs))

    def fit(
        self,
        save_freq: int = 0,
        save_callback: Optional[Callable[[int, dict[str, dict[str, float]]], None]] = None,
    ) -> dict[str, dict[str, float]]:
        """Warmup latent stats, then each epoch train + validate; optional periodic ``save_callback``."""
        history: dict[str, dict[str, float]] = {}
        freq = int(save_freq)
        try:
            # Warmup latent stats, normalization warmup is done before epoch 1.
            self._run_normalization_warmup()

            for epoch in range(1, self.cfg.epochs + 1):
                train_stats = self.train_one_epoch(epoch)
                start = time.perf_counter()
                val_stats = self.validate(epoch)

                if self.is_main_process and bool(self.cfg.profile_enabled):
                    self._log(
                        f"[profile boundary] epoch={epoch:03d} phase=validate "
                        f"wall_ms={(time.perf_counter() - start) * 1000.0:.1f}"
                    )
                history[f"epoch_{epoch:03d}"] = {"train": train_stats, "valid": val_stats}
                self._log(f"[epoch {epoch:03d}] train={train_stats} valid={val_stats}")

                if self.wandb_run is not None:
                    epoch_payload = {f"epoch/train/{k}": v for k, v in train_stats.items()}
                    epoch_payload.update({f"epoch/valid/{k}": v for k, v in val_stats.items()})
                    epoch_payload["epoch"] = int(epoch)
                    self.wandb_run.log(epoch_payload)
                if self.is_main_process and save_callback is not None and freq > 0 and (epoch % freq == 0):
                    save_callback(epoch, history)

            return history
        finally:
            self._finish_wandb()
