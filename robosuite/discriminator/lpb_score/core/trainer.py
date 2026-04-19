"""Training loop for the window-conditioned DSM model."""

from __future__ import annotations

import copy
import inspect
import time
from dataclasses import dataclass
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


class Trainer:
    """Coordinate data loading, mixed-precision training, and validation."""

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

        resolved_device = device or self.cfg.device
        if str(resolved_device).lower().startswith("cuda") and not torch.cuda.is_available():
            resolved_device = "cpu"
        self.device = torch.device(resolved_device)
        self.model.to(self.device)
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

        self.ema_model: Optional[DSMModel] = None
        if self.cfg.use_ema:
            self.ema_model = copy.deepcopy(self.model)
            self.ema_model.to(self.device)
            self.ema_model.eval()

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

        self.train_loader = self._build_train_loader()
        self.val_loader = self._build_val_loader()
        self._profile_accumulator: dict[str, dict[str, float]] = {"train": {}, "valid": {}}
        self._profile_counts: dict[str, int] = {"train": 0, "valid": 0}

    def _log(self, message: str) -> None:
        if self.is_main_process:
            print(message)

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

    def _resolve_positive_trajectories(self):
        if isinstance(self.train_dataset, LatentTransitionDataset):
            return self.train_dataset.iter_positive_trajectories()
        if isinstance(self.train_dataset, Subset) and isinstance(self.train_dataset.dataset, LatentTransitionDataset):
            raise TypeError("Subset normalization refresh is not supported for cached transition datasets.")
        raise TypeError("Trainer normalization refresh requires LatentTransitionDataset or Subset thereof.")

    def _build_val_loader(self):
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
        return {
            key: value.to(self.device, non_blocking=True)
            for key, value in batch.items()
            if torch.is_tensor(value)
        }

    class _CUDAPrefetchIterator:
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

    def _run_step(
        self,
        batch: dict[str, torch.Tensor],
        train: bool,
        model: Optional[DSMModel] = None,
        profiler: Optional[StepProfiler] = None,
    ) -> dict[str, float]:
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
                        add_noise=True,
                        profiler=step_profiler,
                    )
                    recon = self.model.reconstruction_components(
                        chunk_clean=out["chunk_clean"],
                        chunk_hat=out["chunk_hat"],
                    )
                    loss = recon["chunk_energy_per_sample"].mean()
                    stats = {
                        "loss": loss,
                        "score": loss,
                        "unweighted_score": loss,
                        "chunk_mse": recon["chunk_mse_per_sample"].mean(),
                        "chunk_energy": recon["chunk_energy_per_sample"].mean(),
                        "latent_window": out["latent_window"],
                        "traj_type": out["traj_type"],
                        "task_index": out["task_index"],
                        "chunk_clean": out["chunk_clean"],
                        "chunk_input": out["chunk_input"],
                        "chunk_hat": out["chunk_hat"],
                    }
            loss = stats["loss"]
            if train:
                with step_profiler.section("backward"):
                    self._grad_scaler.scale(loss).backward()
                if self.cfg.grad_clip_norm > 0:
                    if self._use_grad_scaler:
                        self._grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
                with step_profiler.section("optim_step"):
                    self._grad_scaler.step(self.optimizer)
                    self._grad_scaler.update()
                if self.ema_model is not None:
                    self._update_ema()

        return {
            "loss": float(stats["loss"].detach().item()),
            "score": float(stats["score"].detach().item()),
            "unweighted_score": float(stats["unweighted_score"].detach().item()),
            "chunk_mse": float(stats["chunk_mse"].detach().item()),
            "chunk_energy": float(stats["chunk_energy"].detach().item()),
        }

    @staticmethod
    def _mean_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
        if not metrics:
            return {}
        keys = sorted(metrics[0].keys())
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
        if self.cfg.use_ema and self.cfg.val_use_ema and self.ema_model is not None:
            return self.ema_model
        return self.model

    def checkpoint_state_dicts(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor] | None]:
        if self.ema_model is None:
            return self.model.state_dict(), None
        if self.cfg.save_ema_in_checkpoint:
            return self.ema_model.state_dict(), self.model.state_dict()
        return self.model.state_dict(), None

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
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
            if self.is_main_process and self.cfg.log_every > 0 and step % self.cfg.log_every == 0:
                self._log(
                    f"[train] epoch={epoch:03d} step={step:05d} "
                    f"loss={out['loss']:.6f} score={out['score']:.6f} "
                    f"unweighted_score={out['unweighted_score']:.6f} "
                    f"chunk_mse={out['chunk_mse']:.6f} "
                    f"chunk_energy={out['chunk_energy']:.6f}"
                )
        return self._reduce_metrics(self._mean_metrics(logs))

    @torch.no_grad()
    def validate(self, epoch: int) -> dict[str, float]:
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
            if self.is_main_process and self.cfg.log_every > 0 and step % self.cfg.log_every == 0:
                self._log(
                    f"[valid] epoch={epoch:03d} step={step:05d} "
                    f"loss={out['loss']:.6f} score={out['score']:.6f} "
                    f"unweighted_score={out['unweighted_score']:.6f} "
                    f"chunk_mse={out['chunk_mse']:.6f} "
                    f"chunk_energy={out['chunk_energy']:.6f}"
                )
        return self._reduce_metrics(self._mean_metrics(logs))

    def fit(
        self,
        save_freq: int = 0,
        save_callback: Optional[Callable[[int, dict[str, dict[str, float]]], None]] = None,
    ) -> dict[str, dict[str, float]]:
        history: dict[str, dict[str, float]] = {}
        freq = int(save_freq)
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
            if self.is_main_process and save_callback is not None and freq > 0 and (epoch % freq == 0):
                save_callback(epoch, history)
        return history
