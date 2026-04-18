"""AdamW training loop for the joint encoder + chunk DSM model."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

from .dataset import LatentTransitionDataset
from .model import DSMModel


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
    normalization_batch_size: int = 256
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4


class Trainer:
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

        resolved_device = device or self.cfg.device
        if str(resolved_device).lower().startswith("cuda") and not torch.cuda.is_available():
            resolved_device = "cpu"
        self.device = torch.device(resolved_device)
        self.model.to(self.device)

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

        self.train_loader = self._build_train_loader()
        self.val_loader = (
            DataLoader(
                self.val_dataset,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                **self._loader_kwargs(),
            )
            if self.val_dataset is not None and len(self.val_dataset) > 0
            else None
        )
        self._online_norm_stats: dict[str, torch.Tensor] | None = None
        self._ema_norm_stats: dict[str, torch.Tensor] | None = None

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

    def _build_train_loader(self) -> DataLoader:
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

    @torch.no_grad()
    def _refresh_normalization_stats(
        self,
        model: DSMModel,
        *,
        target: str,
    ) -> dict[str, torch.Tensor]:
        stats = model.compute_normalization_stats(
            self._resolve_positive_trajectories(),
            batch_size=int(self.cfg.normalization_batch_size),
        )
        model.set_normalization_stats(
            latent_mean=stats["latent_mean"],
            latent_var=stats["latent_var"],
        )
        if target == "online":
            self._online_norm_stats = stats
        elif target == "ema":
            self._ema_norm_stats = stats
        else:
            raise ValueError(f"Unsupported normalization target: {target}")
        print(
            f"[lpb_score] refreshed_{target}_norm_stats latent_count={int(stats['latent_count'].item())}"
        )
        return stats

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
    ) -> dict[str, float]:
        active = model if model is not None else self.model
        data = self._move_batch(batch)
        if train:
            self.optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            stats = active.compute_dsm_loss(
                image_window=data["image_window"],
                proprio_window=data["proprio_window"],
                traj_type=data["traj_type"],
                task_index=data["task_index"],
            )
            loss = stats["loss"]
            if train:
                loss.backward()
                if self.cfg.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
                self.optimizer.step()
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

    def checkpoint_normalization_stats(self) -> dict[str, torch.Tensor]:
        if self.ema_model is None or not self.cfg.save_ema_in_checkpoint:
            if self._online_norm_stats is None:
                self._refresh_normalization_stats(self.model, target="online")
            assert self._online_norm_stats is not None
            return self._online_norm_stats
        if self._ema_norm_stats is None:
            self._refresh_normalization_stats(self.ema_model, target="ema")
        assert self._ema_norm_stats is not None
        return self._ema_norm_stats

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        logs: list[dict[str, float]] = []
        for step, batch in enumerate(self.train_loader):
            out = self._run_step(batch, train=True)
            logs.append(out)
            if self.cfg.log_every > 0 and step % self.cfg.log_every == 0:
                print(
                    f"[train] epoch={epoch:03d} step={step:05d} "
                    f"loss={out['loss']:.6f} score={out['score']:.6f} "
                    f"unweighted_score={out['unweighted_score']:.6f} "
                    f"chunk_mse={out['chunk_mse']:.6f} "
                    f"chunk_energy={out['chunk_energy']:.6f}"
                )
        return self._mean_metrics(logs)

    @torch.no_grad()
    def validate(self, epoch: int) -> dict[str, float]:
        if self.val_loader is None:
            return {}
        eval_model = self.eval_model_for_inference()
        eval_model.eval()
        logs: list[dict[str, float]] = []
        for step, batch in enumerate(self.val_loader):
            out = self._run_step(batch, train=False, model=eval_model)
            logs.append(out)
            if self.cfg.log_every > 0 and step % self.cfg.log_every == 0:
                print(
                    f"[valid] epoch={epoch:03d} step={step:05d} "
                    f"loss={out['loss']:.6f} score={out['score']:.6f} "
                    f"unweighted_score={out['unweighted_score']:.6f} "
                    f"chunk_mse={out['chunk_mse']:.6f} "
                    f"chunk_energy={out['chunk_energy']:.6f}"
                )
        return self._mean_metrics(logs)

    def fit(
        self,
        save_freq: int = 0,
        save_callback: Optional[Callable[[int, dict[str, dict[str, float]]], None]] = None,
    ) -> dict[str, dict[str, float]]:
        history: dict[str, dict[str, float]] = {}
        freq = int(save_freq)
        for epoch in range(1, self.cfg.epochs + 1):
            self._refresh_normalization_stats(self.model, target="online")
            train_stats = self.train_one_epoch(epoch)
            self._refresh_normalization_stats(self.model, target="online")
            if self.ema_model is not None:
                self._refresh_normalization_stats(self.ema_model, target="ema")
            val_stats = self.validate(epoch)
            history[f"epoch_{epoch:03d}"] = {"train": train_stats, "valid": val_stats}
            print(f"[epoch {epoch:03d}] train={train_stats} valid={val_stats}")
            if save_callback is not None and freq > 0 and (epoch % freq == 0):
                save_callback(epoch, history)
        return history
