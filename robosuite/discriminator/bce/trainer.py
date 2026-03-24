from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Optional

import os

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .dataset import TemporalTransitionDataset
from .model import TemporalPUDiscriminator


@dataclass
class TrainerConfig:
    batch_size: int = 256
    num_workers: int = 8
    prefetch_factor: int = 4
    persistent_workers: bool = True
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    epochs: int = 50
    positive_sampling_ratio: float = 0.5
    grad_clip_norm: float = 1.0
    log_every: int = 100
    device: str = "cuda"
    amp: bool = False


class TPUDTrainer:
    def __init__(
        self,
        model: TemporalPUDiscriminator,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        config: Optional[TrainerConfig] = None,
        device: Optional[str] = None,
        cfg=None,
        metadata: Optional[dict] = None,
    ) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = config or TrainerConfig()
        self.full_cfg = cfg
        self.metadata = metadata or {}

        resolved_device = device or self.cfg.device
        if str(resolved_device).lower().startswith("cuda") and not torch.cuda.is_available():
            resolved_device = "cpu"
        self.device = torch.device(resolved_device)
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.cfg.learning_rate),
            weight_decay=float(self.cfg.weight_decay),
        )
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=bool(self.cfg.amp) and self.device.type == "cuda"
        )

        self.train_loader = self._build_train_loader()
        self.val_loader = (
            DataLoader(
                self.val_dataset,
                batch_size=int(self.cfg.batch_size),
                shuffle=False,
                num_workers=int(self.cfg.num_workers),
                pin_memory=self.device.type == "cuda",
                persistent_workers=bool(self.cfg.persistent_workers) and int(self.cfg.num_workers) > 0,
                prefetch_factor=int(self.cfg.prefetch_factor) if int(self.cfg.num_workers) > 0 else None,
                drop_last=False,
            )
            if self.val_dataset is not None and len(self.val_dataset) > 0
            else None
        )
        self.wandb_run = self._init_wandb()

    def _init_wandb(self):
        logging_cfg = getattr(self.full_cfg, "logging", None)
        if logging_cfg is None or not bool(getattr(logging_cfg, "use_wandb", False)):
            return None
        try:
            import wandb
        except Exception:
            print("wandb is unavailable, skipping logging.")
            return None

        entity = str(getattr(logging_cfg, "entity", "songgao-personal"))
        mode = str(getattr(logging_cfg, "mode", "offline"))
        project = str(getattr(logging_cfg, "project", "robosuite-tpud"))
        os.environ.setdefault("WANDB_MODE", mode)
        os.environ.setdefault("WANDB_ENTITY", entity)
        return wandb.init(
            project=project,
            entity=entity,
            mode=mode,
            config={
                "trainer": asdict(self.cfg),
                "metadata": self.metadata,
            },
        )

    def _build_train_loader(self) -> DataLoader:
        labels = None
        if isinstance(self.train_dataset, TemporalTransitionDataset):
            labels = self.train_dataset.sample_is_positive

        use_balanced = labels is not None
        if labels is None:
            n_pos, n_fail = 0, 0
        else:
            n_pos = int(sum(labels))
            n_fail = int(len(labels) - n_pos)
            use_balanced = (n_pos > 0) and (n_fail > 0)

        if use_balanced:
            ratio = float(self.cfg.positive_sampling_ratio)
            ratio = min(max(ratio, 0.0), 1.0)
            w_pos = ratio / float(n_pos)
            w_fail = (1.0 - ratio) / float(n_fail)
            weights = torch.tensor([w_pos if x else w_fail for x in labels], dtype=torch.double)
            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(weights),
                replacement=True,
            )
            return DataLoader(
                self.train_dataset,
                batch_size=int(self.cfg.batch_size),
                sampler=sampler,
                num_workers=int(self.cfg.num_workers),
                pin_memory=self.device.type == "cuda",
                persistent_workers=bool(self.cfg.persistent_workers) and int(self.cfg.num_workers) > 0,
                prefetch_factor=int(self.cfg.prefetch_factor) if int(self.cfg.num_workers) > 0 else None,
                drop_last=False,
            )

        return DataLoader(
            self.train_dataset,
            batch_size=int(self.cfg.batch_size),
            shuffle=True,
            num_workers=int(self.cfg.num_workers),
            pin_memory=self.device.type == "cuda",
            persistent_workers=bool(self.cfg.persistent_workers) and int(self.cfg.num_workers) > 0,
            prefetch_factor=int(self.cfg.prefetch_factor) if int(self.cfg.num_workers) > 0 else None,
            drop_last=False,
        )

    def _move_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            key: value.to(self.device, non_blocking=True)
            for key, value in batch.items()
            if torch.is_tensor(value)
        }

    def _compute_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        outputs = self.model(
            current_latent=batch["current_latent"],
            action_sequence=batch["action_sequence"],
            task_index=batch["task_index"],
        )
        logits = outputs["logits"]
        probs = outputs["probs"]
        targets = batch["soft_target"].float()
        pos_mask = batch["is_positive"].bool()
        fail_mask = ~pos_mask

        zero = logits.new_zeros(())
        pos_loss = zero
        fail_loss = zero
        if bool(pos_mask.any()):
            pos_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits[pos_mask],
                torch.ones_like(logits[pos_mask]),
            )
        if bool(fail_mask.any()):
            fail_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits[fail_mask],
                targets[fail_mask],
            )
        loss = pos_loss + fail_loss

        out = {
            "loss": loss,
            "positive_loss": pos_loss,
            "failure_loss": fail_loss,
            "prob_mean": probs.mean(),
            "ood_mean": (1.0 - probs).mean(),
            "target_mean": targets.mean(),
            "positive_prob_mean": probs[pos_mask].mean() if bool(pos_mask.any()) else zero,
            "failure_prob_mean": probs[fail_mask].mean() if bool(fail_mask.any()) else zero,
            "failure_target_mean": targets[fail_mask].mean() if bool(fail_mask.any()) else zero,
        }
        return out

    def _run_step(self, batch: dict[str, torch.Tensor], train: bool) -> dict[str, float]:
        data = self._move_batch(batch)
        if train:
            self.optimizer.zero_grad(set_to_none=True)

        amp_enabled = bool(self.cfg.amp) and self.device.type == "cuda"
        with torch.set_grad_enabled(train):
            with torch.amp.autocast(device_type=self.device.type, enabled=amp_enabled):
                stats = self._compute_loss(data)
            loss = stats["loss"]
            if train:
                self.scaler.scale(loss).backward()
                if self.cfg.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg.grad_clip_norm))
                self.scaler.step(self.optimizer)
                self.scaler.update()

        return {key: float(value.detach().item()) for key, value in stats.items()}

    @staticmethod
    def _mean_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
        if not metrics:
            return {}
        keys = sorted(metrics[0].keys())
        return {key: float(sum(item[key] for item in metrics) / len(metrics)) for key in keys}

    def _run_epoch(
        self,
        loader: DataLoader,
        epoch: int,
        train: bool,
        split_name: str,
    ) -> dict[str, float]:
        self.model.train(mode=train)
        logs: list[dict[str, float]] = []
        for step, batch in enumerate(loader):
            out = self._run_step(batch, train=train)
            logs.append(out)
            if self.cfg.log_every > 0 and step % int(self.cfg.log_every) == 0:
                print(
                    f"[{split_name}] epoch={epoch:03d} step={step:05d} "
                    f"loss={out['loss']:.6f} positive_loss={out['positive_loss']:.6f} "
                    f"failure_loss={out['failure_loss']:.6f} prob_mean={out['prob_mean']:.6f}"
                )
        return self._mean_metrics(logs)

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        return self._run_epoch(self.train_loader, epoch=epoch, train=True, split_name="train")

    @torch.no_grad()
    def validate(self, epoch: int) -> dict[str, float]:
        if self.val_loader is None:
            return {}
        return self._run_epoch(self.val_loader, epoch=epoch, train=False, split_name="valid")

    def fit(
        self,
        save_freq: int = 0,
        save_callback: Optional[Callable[[int, dict[str, dict[str, dict[str, float]]]], None]] = None,
    ) -> dict[str, dict[str, dict[str, float]]]:
        history: dict[str, dict[str, dict[str, float]]] = {}
        freq = int(save_freq)
        best_val_loss = float("inf")
        for epoch in range(1, int(self.cfg.epochs) + 1):
            train_stats = self.train_one_epoch(epoch)
            val_stats = self.validate(epoch)
            history[f"epoch_{epoch:03d}"] = {"train": train_stats, "valid": val_stats}
            print(f"[epoch {epoch:03d}] train={train_stats} valid={val_stats}")

            if self.wandb_run is not None:
                payload = {f"train/{k}": v for k, v in train_stats.items()}
                payload.update({f"valid/{k}": v for k, v in val_stats.items()})
                payload["epoch"] = epoch
                self.wandb_run.log(payload)

            if save_callback is not None and freq > 0 and (epoch % freq == 0):
                save_callback(epoch, history)

            if val_stats and float(val_stats.get("loss", float("inf"))) < best_val_loss:
                best_val_loss = float(val_stats["loss"])

        if self.wandb_run is not None:
            self.wandb_run.finish()
        return history


__all__ = [
    "TPUDTrainer",
    "TrainerConfig",
]
