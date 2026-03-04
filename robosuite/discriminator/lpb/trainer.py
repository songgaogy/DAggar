from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

from .dataset import LatentDynamicsDataset
from .model import DynamicsModel


@dataclass
class TrainerConfig:
    batch_size: int = 64
    num_workers: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 50
    expert_sampling_ratio: float = 0.5
    proprio_loss_weight: float = 0.0
    grad_clip_norm: float = 1.0
    log_every: int = 50


class Trainer:
    """
    Trainer for latent visual dynamics:
      L_dynamics = || h_theta(o_{t+T_p}) - f_phi(h_theta(o_t), A_t) ||_2^2
    """

    def __init__(
        self,
        model: DynamicsModel,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        config: Optional[TrainerConfig] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = config or TrainerConfig()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
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
            if self.val_dataset is not None
            else None
        )

    def _resolve_expert_labels(self, dataset: Dataset) -> Optional[list[bool]]:
        if isinstance(dataset, LatentDynamicsDataset):
            return dataset.sample_is_expert
        if isinstance(dataset, Subset) and isinstance(dataset.dataset, LatentDynamicsDataset):
            base = dataset.dataset.sample_is_expert
            return [base[i] for i in dataset.indices]
        return None

    def _build_train_loader(self) -> DataLoader:
        # Balanced expert / rollout sampling for mixed training.
        labels = self._resolve_expert_labels(self.train_dataset)
        use_balanced = labels is not None
        if labels is None:
            n_exp, n_roll = 0, 0
        else:
            n_exp = int(sum(labels))
            n_roll = int(len(labels) - n_exp)
            use_balanced = (n_exp > 0) and (n_roll > 0)

        if use_balanced:
            ratio = float(self.cfg.expert_sampling_ratio)
            ratio = min(max(ratio, 0.0), 1.0)
            w_exp = ratio / float(n_exp)
            w_roll = (1.0 - ratio) / float(n_roll)
            weights = torch.tensor([w_exp if x else w_roll for x in labels], dtype=torch.double)
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

    def _move_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                out[k] = v.to(self.device, non_blocking=True)
        return out

    def _run_step(self, batch: Dict[str, torch.Tensor], train: bool) -> Dict[str, float]:
        b = self._move_batch(batch)
        if train:
            self.optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            stats = self.model.compute_dynamics_loss(
                current_image=b["current_image"],
                current_proprio=b["current_proprio"],
                action_sequence=b["action_sequence"],
                target_image=b["target_image"],
                target_proprio=b["target_proprio"],
                proprio_loss_weight=self.cfg.proprio_loss_weight,
            )
            loss = stats["loss"]
            if train:
                loss.backward()
                if self.cfg.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
                self.optimizer.step()

        out = {
            "loss": float(stats["loss"].detach().item()),
            "latent_mse": float(stats["latent_mse"].detach().item()),
        }
        if "proprio_mse" in stats:
            out["proprio_mse"] = float(stats["proprio_mse"].detach().item())
        return out

    @staticmethod
    def _mean_metrics(metrics: list[Dict[str, float]]) -> Dict[str, float]:
        if not metrics:
            return {}
        keys = sorted(metrics[0].keys())
        return {k: float(sum(m[k] for m in metrics) / len(metrics)) for k in keys}

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        logs: list[Dict[str, float]] = []
        for step, batch in enumerate(self.train_loader):
            out = self._run_step(batch, train=True)
            logs.append(out)
            if self.cfg.log_every > 0 and step % self.cfg.log_every == 0:
                print(
                    f"[train] epoch={epoch:03d} step={step:05d} "
                    f"loss={out['loss']:.6f} latent_mse={out['latent_mse']:.6f}"
                )
        return self._mean_metrics(logs)

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        if self.val_loader is None:
            return {}
        self.model.eval()
        logs: list[Dict[str, float]] = []
        for step, batch in enumerate(self.val_loader):
            out = self._run_step(batch, train=False)
            logs.append(out)
            if self.cfg.log_every > 0 and step % self.cfg.log_every == 0:
                print(
                    f"[valid] epoch={epoch:03d} step={step:05d} "
                    f"loss={out['loss']:.6f} latent_mse={out['latent_mse']:.6f}"
                )
        return self._mean_metrics(logs)

    def fit(
        self,
        save_freq: int = 0,
        save_callback: Optional[Callable[[int, Dict[str, Dict[str, float]]], None]] = None,
    ) -> Dict[str, Dict[str, float]]:
        history: Dict[str, Dict[str, float]] = {}
        freq = int(save_freq)
        for epoch in range(1, self.cfg.epochs + 1):
            train_stats = self.train_one_epoch(epoch)
            val_stats = self.validate(epoch)
            history[f"epoch_{epoch:03d}"] = {"train": train_stats, "valid": val_stats}
            print(f"[epoch {epoch:03d}] train={train_stats} valid={val_stats}")
            if save_callback is not None and freq > 0 and (epoch % freq == 0):
                save_callback(epoch, history)
        return history
