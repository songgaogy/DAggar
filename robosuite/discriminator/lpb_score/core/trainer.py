"""AdamW training loop for ``DSMModel`` with optional positive vs failure balanced sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

from .dataset import LatentTransitionDataset
from .model import DSMModel


@dataclass
class TrainerConfig:
    """Hyperparameters for ``Trainer`` (batching, optimization, logging, class balance)."""

    batch_size: int = 64
    num_workers: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 50
    positive_sampling_ratio: float = 0.5
    grad_clip_norm: float = 1.0
    log_every: int = 50
    device: str = "cuda"


class Trainer:
    """Trains ``DSMModel`` via ``compute_dsm_loss``; uses ``WeightedRandomSampler`` when both traj classes exist."""

    def __init__(
        self,
        model: DSMModel,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        config: Optional[TrainerConfig] = None,
        device: Optional[str] = None,
    ) -> None:
        """Attach model, build optimizers and loaders, move parameters to ``device``."""
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = config or TrainerConfig()

        resolved_device = device or self.cfg.device
        if str(resolved_device).lower().startswith("cuda") and not torch.cuda.is_available():
            resolved_device = "cpu"
        self.device = torch.device(resolved_device)
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )

        self.train_loader = self._build_train_loader()
        self.val_loader = (
            DataLoader(
                self.val_dataset,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                num_workers=self.cfg.num_workers,
                pin_memory=torch.cuda.is_available(),
            )
            if self.val_dataset is not None and len(self.val_dataset) > 0
            else None
        )

    def _resolve_positive_labels(self, dataset: Dataset) -> Optional[list[bool]]:
        if isinstance(dataset, LatentTransitionDataset):
            return dataset.sample_is_positive
        if isinstance(dataset, Subset) and isinstance(dataset.dataset, LatentTransitionDataset):
            base = dataset.dataset.sample_is_positive
            return [base[i] for i in dataset.indices]
        return None

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
                num_workers=self.cfg.num_workers,
                pin_memory=torch.cuda.is_available(),
            )

        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def _move_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            key: value.to(self.device, non_blocking=True)
            for key, value in batch.items()
            if torch.is_tensor(value)
        }

    def _run_step(self, batch: dict[str, torch.Tensor], train: bool) -> dict[str, float]:
        """Single forward/backward on ``task_index`` and ``traj_type`` batches from ``LatentTransitionDataset``."""
        data = self._move_batch(batch)
        if train:
            self.optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            stats = self.model.compute_dsm_loss(
                current_latent=data["current_latent"],
                action_sequence=data["action_sequence"],
                target_latent=data["target_latent"],
                traj_type=data["traj_type"],
                task_index=data["task_index"],
            )
            loss = stats["loss"]
            if train:
                loss.backward()
                if self.cfg.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
                self.optimizer.step()

        return {
            "loss": float(stats["loss"].detach().item()),
            "score": float(stats["score"].detach().item()),
            "tau_mse": float(stats["tau_mse"].detach().item()),
            "state_mse": float(stats["state_mse"].detach().item()),
            "action_mse": float(stats["action_mse"].detach().item()),
            "next_state_mse": float(stats["next_state_mse"].detach().item()),
            "state_energy": float(stats["state_energy"].detach().item()),
            "action_energy": float(stats["action_energy"].detach().item()),
            "next_state_energy": float(stats["next_state_energy"].detach().item()),
        }

    @staticmethod
    def _mean_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
        if not metrics:
            return {}
        keys = sorted(metrics[0].keys())
        return {key: float(sum(item[key] for item in metrics) / len(metrics)) for key in keys}

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        """One full pass over the training loader; returns mean logged scalars."""
        self.model.train()
        logs: list[dict[str, float]] = []
        for step, batch in enumerate(self.train_loader):
            out = self._run_step(batch, train=True)
            logs.append(out)
            if self.cfg.log_every > 0 and step % self.cfg.log_every == 0:
                print(
                    f"[train] epoch={epoch:03d} step={step:05d} "
                    f"loss={out['loss']:.6f} score={out['score']:.6f} tau_mse={out['tau_mse']:.6f} "
                    f"state_mse={out['state_mse']:.6f} action_mse={out['action_mse']:.6f} "
                    f"next_state_mse={out['next_state_mse']:.6f} "
                    f"state_energy={out['state_energy']:.6f} action_energy={out['action_energy']:.6f} "
                    f"next_state_energy={out['next_state_energy']:.6f}"
                )
        return self._mean_metrics(logs)

    @torch.no_grad()
    def validate(self, epoch: int) -> dict[str, float]:
        """Validation pass; returns ``{}`` if no validation loader."""
        if self.val_loader is None:
            return {}
        self.model.eval()
        logs: list[dict[str, float]] = []
        for step, batch in enumerate(self.val_loader):
            out = self._run_step(batch, train=False)
            logs.append(out)
            if self.cfg.log_every > 0 and step % self.cfg.log_every == 0:
                print(
                    f"[valid] epoch={epoch:03d} step={step:05d} "
                    f"loss={out['loss']:.6f} score={out['score']:.6f} tau_mse={out['tau_mse']:.6f} "
                    f"state_mse={out['state_mse']:.6f} action_mse={out['action_mse']:.6f} "
                    f"next_state_mse={out['next_state_mse']:.6f} "
                    f"state_energy={out['state_energy']:.6f} action_energy={out['action_energy']:.6f} "
                    f"next_state_energy={out['next_state_energy']:.6f}"
                )
        return self._mean_metrics(logs)

    def fit(
        self,
        save_freq: int = 0,
        save_callback: Optional[Callable[[int, dict[str, dict[str, float]]], None]] = None,
    ) -> dict[str, dict[str, float]]:
        """Train for ``epochs``; optional ``save_callback(epoch, history)`` every ``save_freq`` epochs."""
        history: dict[str, dict[str, float]] = {}
        freq = int(save_freq)
        for epoch in range(1, self.cfg.epochs + 1):
            train_stats = self.train_one_epoch(epoch)
            val_stats = self.validate(epoch)
            history[f"epoch_{epoch:03d}"] = {"train": train_stats, "valid": val_stats}
            print(f"[epoch {epoch:03d}] train={train_stats} valid={val_stats}")
            if save_callback is not None and freq > 0 and (epoch % freq == 0):
                save_callback(epoch, history)
        return history
