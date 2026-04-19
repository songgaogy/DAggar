"""AdamW training loop for ``DSMModel`` with optional positive vs failure balanced sampling."""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

from .dataset import LatentTransitionDataset
from .model import DSMModel


_COLLAPSE_ALERT_THRESHOLD = 1e-4
_COLLAPSE_ALERT_WINDOW = 3


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
    use_ema: bool = False
    ema_decay: float = 0.999
    val_use_ema: bool = True
    save_ema_in_checkpoint: bool = True
    shared_lr_multiplier: float = 1.0
    state_branch_lr_multiplier: float = 1.0
    dynamics_branch_lr_multiplier: float = 1.0


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
                num_workers=self.cfg.num_workers,
                pin_memory=torch.cuda.is_available(),
            )
            if self.val_dataset is not None and len(self.val_dataset) > 0
            else None
        )

        # Class-signal-collapse monitor: sustained near-zero ||g(x, c=0) - g(x, c=1)||^2
        # indicates the CFG embedding has collapsed (sibling uni-dsm-new failure signature).
        self._collapse_history: dict[str, deque[float]] = {
            "state": deque(maxlen=_COLLAPSE_ALERT_WINDOW),
            "dynamics": deque(maxlen=_COLLAPSE_ALERT_WINDOW),
        }

    def _build_optimizer_param_groups(self) -> list[dict[str, object]]:
        multipliers = {
            "shared": float(self.cfg.shared_lr_multiplier),
            "state_branch": float(self.cfg.state_branch_lr_multiplier),
            "dynamics_branch": float(self.cfg.dynamics_branch_lr_multiplier),
        }
        for name, multiplier in multipliers.items():
            if multiplier < 0.0:
                raise ValueError(f"{name} lr multiplier must be non-negative, got {multiplier}")

        groups = self.model.predictor.optimizer_parameter_groups()
        param_groups: list[dict[str, object]] = []
        for name in ("shared", "state_branch", "dynamics_branch"):
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

    @torch.no_grad()
    def _update_ema(self) -> None:
        """EMA: ``ema <- decay * ema + (1 - decay) * online`` on learnable parameters."""
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
        """Single forward/backward on ``task_index`` and ``traj_type`` batches from ``LatentTransitionDataset``."""
        active = model if model is not None else self.model
        data = self._move_batch(batch)
        if train:
            self.optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            stats = active.compute_dsm_loss(
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
                if self.ema_model is not None:
                    self._update_ema()

        return {
            "loss": float(stats["loss"].detach().item()),
            "score": float(stats["score"].detach().item()),
            "unweighted_score": float(stats["unweighted_score"].detach().item()),
            "tau_mse": float(stats["tau_mse"].detach().item()),
            "state_mse": float(stats["state_mse"].detach().item()),
            "next_state_mse": float(stats["next_state_mse"].detach().item()),
            "state_energy": float(stats["state_energy"].detach().item()),
            "next_state_energy": float(stats["next_state_energy"].detach().item()),
        }

    @staticmethod
    def _mean_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
        if not metrics:
            return {}
        keys = sorted(metrics[0].keys())
        return {key: float(sum(item[key] for item in metrics) / len(metrics)) for key in keys}

    def eval_model_for_inference(self) -> DSMModel:
        """Weights used for validation (and checkpoints when ``save_ema_in_checkpoint``)."""
        if (
            self.cfg.use_ema
            and self.cfg.val_use_ema
            and self.ema_model is not None
        ):
            return self.ema_model
        return self.model

    def checkpoint_state_dicts(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor] | None]:
        """Return ``(payload['model'], extra_or_none)`` for ``torch.save``.

        If EMA is enabled and ``save_ema_in_checkpoint``, ``model`` is the EMA weights (recommended
        for ``DSMTransitionScorer``); the second dict is online weights for ``model_online``.
        If ``save_ema_in_checkpoint`` is false, only the online weights are returned as primary.
        """
        if self.ema_model is None:
            return self.model.state_dict(), None
        if self.cfg.save_ema_in_checkpoint:
            return self.ema_model.state_dict(), self.model.state_dict()
        return self.model.state_dict(), None

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
                    f"loss={out['loss']:.6f} score={out['score']:.6f} "
                    f"unweighted_score={out['unweighted_score']:.6f} tau_mse={out['tau_mse']:.6f} "
                    f"state_mse={out['state_mse']:.6f} next_state_mse={out['next_state_mse']:.6f} "
                    f"state_energy={out['state_energy']:.6f} "
                    f"next_state_energy={out['next_state_energy']:.6f}"
                )
        return self._mean_metrics(logs)

    @torch.no_grad()
    def validate(self, epoch: int) -> dict[str, float]:
        """Validation pass; returns ``{}`` if no validation loader."""
        if self.val_loader is None:
            return {}
        self.model.eval()
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
                    f"unweighted_score={out['unweighted_score']:.6f} tau_mse={out['tau_mse']:.6f} "
                    f"state_mse={out['state_mse']:.6f} next_state_mse={out['next_state_mse']:.6f} "
                    f"state_energy={out['state_energy']:.6f} "
                    f"next_state_energy={out['next_state_energy']:.6f}"
                )
        return self._mean_metrics(logs)

    @torch.no_grad()
    def _log_collapse_diagnostic(self, epoch: int) -> dict[str, float]:
        """Run ``class_conditional_diff`` on one training batch; warn when either factor collapses."""
        try:
            batch = next(iter(self.train_loader))
        except StopIteration:
            return {}
        data = self._move_batch(batch)
        eval_model = self.eval_model_for_inference()
        eval_model.eval()
        diag = eval_model.class_conditional_diff(
            current_latent=data["current_latent"],
            action_sequence=data["action_sequence"],
            target_latent=data["target_latent"],
            task_index=data["task_index"],
        )
        means = {key: float(value.mean().item()) for key, value in diag.items()}
        print(
            f"[collapse] epoch={epoch:03d} "
            f"state_diff_l2={means.get('state', float('nan')):.6e} "
            f"dynamics_diff_l2={means.get('dynamics', float('nan')):.6e}"
        )
        for factor, mean_val in means.items():
            history = self._collapse_history.setdefault(factor, deque(maxlen=_COLLAPSE_ALERT_WINDOW))
            history.append(mean_val)
            if (
                len(history) == _COLLAPSE_ALERT_WINDOW
                and all(v < _COLLAPSE_ALERT_THRESHOLD for v in history)
            ):
                print(
                    f"WARNING[collapse] epoch={epoch:03d} {factor} diff < "
                    f"{_COLLAPSE_ALERT_THRESHOLD:.0e} for {_COLLAPSE_ALERT_WINDOW} consecutive epochs"
                )
        return means

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
            collapse_stats = self._log_collapse_diagnostic(epoch)
            history[f"epoch_{epoch:03d}"] = {
                "train": train_stats,
                "valid": val_stats,
                "collapse": collapse_stats,
            }
            print(f"[epoch {epoch:03d}] train={train_stats} valid={val_stats}")
            if save_callback is not None and freq > 0 and (epoch % freq == 0):
                save_callback(epoch, history)
        return history
