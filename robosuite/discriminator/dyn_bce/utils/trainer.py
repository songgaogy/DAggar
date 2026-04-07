from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from robosuite.discriminator.dyn_bce.utils.losses import (
    beta_nll_loss,
    cross_covariance_penalty,
    occupancy_pu_loss,
    weighted_bce_with_logits,
)
from robosuite.discriminator.dyn_bce.utils.metrics import (
    compute_binary_metrics,
    compute_score_ranking_metrics,
    estimate_best_threshold,
)


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@dataclass
class TrainerConfig:
    device: str
    batch_size: int
    num_workers: int
    prefetch_factor: int
    persistent_workers: bool
    epochs: int
    learning_rate: float
    weight_decay: float
    grad_clip_norm: float
    amp: bool
    ema_decay: float
    log_every: int
    save_freq: int
    alpha_dyn: float
    eta_decor: float
    xi_fuse: float
    beta_nll: float
    occupancy_positive_prior: float
    occupancy_nnpu: bool


class DynBCETrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset,
        val_dataset,
        test_dataset,
        config: TrainerConfig,
        metadata: dict,
        cfg,
    ) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = val_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.metadata = metadata
        self.cfg = cfg

        device_name = str(config.device)
        self.device = torch.device(device_name if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        if len(self.train_dataset) <= 0:
            raise RuntimeError("Train dataset has 0 transitions. Please check data availability and split counts.")

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(config.learning_rate),
            weight_decay=float(config.weight_decay),
        )
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=bool(config.amp) and self.device.type == "cuda"
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=int(config.batch_size),
            shuffle=True,
            num_workers=int(config.num_workers),
            pin_memory=self.device.type == "cuda",
            persistent_workers=bool(config.persistent_workers) and int(config.num_workers) > 0,
            prefetch_factor=int(config.prefetch_factor) if int(config.num_workers) > 0 else None,
            drop_last=False,
        )
        self.eval_loader = DataLoader(
            val_dataset,
            batch_size=int(config.batch_size),
            shuffle=False,
            num_workers=int(config.num_workers),
            pin_memory=self.device.type == "cuda",
            persistent_workers=bool(config.persistent_workers) and int(config.num_workers) > 0,
            prefetch_factor=int(config.prefetch_factor) if int(config.num_workers) > 0 else None,
            drop_last=False,
        )
        self.val_loader = self.eval_loader
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=int(config.batch_size),
            shuffle=False,
            num_workers=int(config.num_workers),
            pin_memory=self.device.type == "cuda",
            persistent_workers=bool(config.persistent_workers) and int(config.num_workers) > 0,
            prefetch_factor=int(config.prefetch_factor) if int(config.num_workers) > 0 else None,
            drop_last=False,
        )

        self.best_eval_score = float("-inf")
        self._startup_integrity_reported = False
        self.best_eval_threshold = 0.5
        self.best_val_score = self.best_eval_score
        self.best_state_dict: dict[str, torch.Tensor] | None = None
        self.history: dict[str, list[dict[str, float]]] = {"train": [], "eval": [], "test": []}
        self.wandb_run = self._init_wandb()

    def _display_split_name(self, split_name: str) -> str:
        if split_name == "val":
            return "eval"
        return split_name

    def _format_kv_row(self, items: list[tuple[str, float]], key_width: int = 8) -> str:
        parts = [f"{name:<{key_width}} {value:>8.4f}" for name, value in items]
        return " | ".join(parts)

    def _tensor_debug_summary(self, tensor: torch.Tensor) -> dict[str, float | int]:
        flat = tensor.detach().reshape(-1).float()
        total_count = int(flat.numel())
        finite_mask = torch.isfinite(flat)
        finite_count = int(finite_mask.sum().item())
        nonfinite_count = int(total_count - finite_count)
        summary: dict[str, float | int] = {
            "total": total_count,
            "finite": finite_count,
            "nonfinite": nonfinite_count,
        }
        if finite_count > 0:
            finite_flat = flat[finite_mask]
            summary.update(
                {
                    "mean": float(finite_flat.mean().item()),
                    "std": float(finite_flat.std(unbiased=False).item()) if finite_count > 1 else 0.0,
                    "min": float(finite_flat.min().item()),
                    "max": float(finite_flat.max().item()),
                }
            )
        return summary

    def _scan_named_tensors(self, named_tensors) -> list[tuple[str, dict[str, float | int]]]:
        failing = []
        for name, tensor in named_tensors:
            if tensor is None:
                continue
            detached = tensor.detach()
            if not torch.isfinite(detached).all():
                failing.append((name, self._tensor_debug_summary(detached)))
        return failing

    def _scan_model_nonfinite_state(self) -> dict[str, object] | None:
        parameter_failures = self._scan_named_tensors(self.model.named_parameters())
        buffer_failures = self._scan_named_tensors(self.model.named_buffers())
        if not parameter_failures and not buffer_failures:
            return None
        first_nonfinite = None
        if parameter_failures:
            first_nonfinite = f"parameter:{parameter_failures[0][0]}"
        elif buffer_failures:
            first_nonfinite = f"buffer:{buffer_failures[0][0]}"
        return {
            "first_nonfinite": first_nonfinite,
            "parameters": parameter_failures,
            "buffers": buffer_failures,
        }

    def _scan_gradient_nonfinite_state(self) -> dict[str, object] | None:
        gradient_failures = []
        for name, parameter in self.model.named_parameters():
            if parameter.grad is None:
                continue
            detached_grad = parameter.grad.detach()
            if not torch.isfinite(detached_grad).all():
                gradient_failures.append((name, self._tensor_debug_summary(detached_grad)))
        if not gradient_failures:
            return None
        return {
            "first_nonfinite": f"gradient:{gradient_failures[0][0]}",
            "gradients": gradient_failures,
        }

    def _print_model_nonfinite_state(self, state: dict[str, object], prefix: str) -> None:
        print(f"[{prefix}] first_nonfinite={state['first_nonfinite']}")
        for name, summary in state.get("batch", []):
            line = f"  batch {name}: nonfinite={summary['nonfinite']}/{summary['total']}"
            if "mean" in summary:
                line += (
                    f" mean={summary['mean']:.4f} std={summary['std']:.4f}"
                    f" min={summary['min']:.4f} max={summary['max']:.4f}"
                )
            print(line)
        for name, summary in state.get("parameters", []):
            line = f"  parameter {name}: nonfinite={summary['nonfinite']}/{summary['total']}"
            if "mean" in summary:
                line += (
                    f" mean={summary['mean']:.4f} std={summary['std']:.4f}"
                    f" min={summary['min']:.4f} max={summary['max']:.4f}"
                )
            print(line)
        for name, summary in state.get("buffers", []):
            line = f"  buffer {name}: nonfinite={summary['nonfinite']}/{summary['total']}"
            if "mean" in summary:
                line += (
                    f" mean={summary['mean']:.4f} std={summary['std']:.4f}"
                    f" min={summary['min']:.4f} max={summary['max']:.4f}"
                )
            print(line)
        for name, summary in state.get("gradients", []):
            line = f"  gradient {name}: nonfinite={summary['nonfinite']}/{summary['total']}"
            if "mean" in summary:
                line += (
                    f" mean={summary['mean']:.4f} std={summary['std']:.4f}"
                    f" min={summary['min']:.4f} max={summary['max']:.4f}"
                )
            print(line)

    def _run_startup_integrity_check(self) -> None:
        state = self._scan_model_nonfinite_state()
        self._startup_integrity_reported = True
        if state is None:
            print("[Startup Check] model parameters and buffers are finite.")
            return
        self._print_model_nonfinite_state(state, prefix="Startup Check")
        raise RuntimeError("Startup integrity check failed: model parameters or buffers are non-finite.")

    def _scan_batch_nonfinite_state(self, batch: dict[str, torch.Tensor]) -> dict[str, object] | None:
        batch_failures = self._scan_named_tensors((f"batch.{name}", tensor) for name, tensor in batch.items())
        if not batch_failures:
            return None
        return {
            "first_nonfinite": batch_failures[0][0],
            "batch": batch_failures,
        }

    def _raise_nonfinite_error(
        self,
        *,
        message: str,
        debug: dict[str, object] | None = None,
        batch_state: dict[str, object] | None = None,
        model_state: dict[str, object] | None = None,
        gradient_state: dict[str, object] | None = None,
    ) -> None:
        if debug is not None:
            self._print_nonfinite_debug(debug)
        if batch_state is not None:
            self._print_model_nonfinite_state(batch_state, prefix="Batch NonFinite State")
        if gradient_state is not None:
            self._print_model_nonfinite_state(gradient_state, prefix="Gradient NonFinite State")
        if model_state is not None:
            self._print_model_nonfinite_state(model_state, prefix="Model NonFinite State")
        raise RuntimeError(message)

    def _build_nonfinite_debug(
        self,
        *,
        epoch: int,
        split_name: str,
        batch_idx: int,
        batch: dict[str, torch.Tensor],
        output,
        occ_loss: torch.Tensor,
        fuse_loss: torch.Tensor,
        dyn_loss: torch.Tensor,
        decor_loss: torch.Tensor,
        total_loss: torch.Tensor,
    ) -> dict[str, object] | None:
        ordered_tensors = [
            ("batch.current_latent", batch["current_latent"]),
            ("batch.next_latent", batch["next_latent"]),
            ("batch.action_sequence", batch["action_sequence"]),
            ("batch.risk_target", batch["risk_target"]),
            ("batch.hard_label", batch["hard_label"]),
            ("batch.occ_weight", batch["occ_weight"]),
            ("batch.dyn_weight", batch["dyn_weight"]),
            ("batch.fuse_weight", batch["fuse_weight"]),
            ("batch.task_index", batch["task_index"]),
            ("batch.data_type_index", batch["data_type_index"]),
            ("output.task_embedding", output.task_embedding_debug),
            ("output.current_input", output.current_input_debug),
            ("output.shared_latent", output.shared_latent_debug),
            ("output.occ_private", output.occ_private),
            ("output.raw_occ_logit", output.raw_occ_logit_debug),
            ("output.occ_logit", output.occ_logit),
            ("output.occ_prob", output.occ_prob),
            ("output.occ_evidence", output.occ_evidence),
            ("output.action_feature", output.action_feature_debug),
            ("output.dyn_hidden", output.dyn_hidden_debug),
            ("output.ensemble_mean", output.ensemble_mean),
            ("output.ensemble_logvar", output.ensemble_logvar),
            ("output.ensemble_mean_avg", output.ensemble_mean_avg_debug),
            ("output.ema_target", output.ema_target),
            ("output.dyn_residual", output.dyn_residual),
            ("output.dyn_evidence", output.dyn_evidence),
            ("output.epi_variance", output.epi_variance),
            ("output.epi_evidence", output.epi_evidence),
            ("output.judge_logit", output.judge_logit),
            ("loss.occ", occ_loss),
            ("loss.fuse", fuse_loss),
            ("loss.dyn", dyn_loss),
            ("loss.decor", decor_loss),
            ("loss.total", total_loss),
        ]
        failing = []
        first_nonfinite = None
        for name, tensor in ordered_tensors:
            if not torch.isfinite(tensor.detach()).all():
                summary = self._tensor_debug_summary(tensor)
                failing.append((name, summary))
                if first_nonfinite is None:
                    first_nonfinite = name
        if not failing:
            return None
        return {
            "epoch": int(epoch),
            "split": str(split_name),
            "batch_idx": int(batch_idx),
            "first_nonfinite": first_nonfinite,
            "failing": failing,
            "model_state": self._scan_model_nonfinite_state(),
        }

    def _print_nonfinite_debug(self, debug: dict[str, object]) -> None:
        print(
            f"[NonFinite Debug] epoch={debug['epoch']} split={debug['split']} batch={debug['batch_idx']} "
            f"first_nonfinite={debug['first_nonfinite']}"
        )
        for name, summary in debug["failing"]:
            line = (
                f"  {name}: nonfinite={summary['nonfinite']}/{summary['total']}"
            )
            if "mean" in summary:
                line += (
                    f" mean={summary['mean']:.4f} std={summary['std']:.4f}"
                    f" min={summary['min']:.4f} max={summary['max']:.4f}"
                )
            print(line)
        if debug.get("model_state") is not None:
            self._print_model_nonfinite_state(debug["model_state"], prefix="Model NonFinite State")

    def _print_train_batch_log(
        self,
        *,
        epoch: int,
        split_name: str,
        batch_idx: int,
        num_batches: int,
        outputs: dict[str, torch.Tensor],
    ) -> None:
        display_split = self._display_split_name(split_name)
        print(f"[Epoch {epoch:03d} | {display_split}] batch {batch_idx:>4}/{num_batches:<4}")
        print(
            "  losses  | "
            + self._format_kv_row(
                [
                    ("loss", float(outputs["total_loss"].item())),
                    ("occ", float(outputs["occ_loss"].item())),
                    ("dyn", float(outputs["dyn_loss"].item())),
                    ("fuse", float(outputs["fuse_loss"].item())),
                ],
                key_width=4,
            )
        )
        print(
            "  signal  | "
            + self._format_kv_row(
                [
                    ("occ_raw", float(outputs["occ_risk_mean"].item())),
                    ("occ_evi", float(outputs["occ_evidence_mean"].item())),
                    ("dyn_evi", float(outputs["dyn_evidence_mean"].item())),
                    ("epi_evi", float(outputs["epi_evidence_mean"].item())),
                ],
                key_width=7,
            )
        )

    def _print_eval_summary(self, *, epoch: int, split_name: str, metrics: dict[str, float]) -> None:
        display_split = self._display_split_name(split_name)
        print(f"[Epoch {epoch:03d} | {display_split}]")
        sections = [
            (
                "overview",
                [
                    ("total_loss", "total_loss"),
                    ("judge_auc", "judge_auroc"),
                    ("judge_f1", "judge_f1"),
                    ("occ_auc", "occ_auroc"),
                    ("occ_evi_auc", "occ_evidence_auroc"),
                ],
            ),
            (
                "judge",
                [
                    ("accuracy", "judge_accuracy"),
                    ("precision", "judge_precision"),
                    ("recall", "judge_recall"),
                    ("f1", "judge_f1"),
                    ("auprc", "judge_auprc"),
                    ("pos_rate", "judge_positive_rate"),
                    ("label_rate", "judge_label_rate"),
                    ("thr", "judge_threshold"),
                ],
            ),
            (
                "occ_raw",
                [
                    ("loss", "occ_loss"),
                    ("auroc", "occ_auroc"),
                    ("f1", "occ_f1"),
                    ("risk", "occ_risk_mean"),
                    ("pos_rate", "occ_positive_rate"),
                    ("clamped", "occ_nnpu_clamped"),
                ],
            ),
            (
                "occ_evi",
                [
                    ("auroc", "occ_evidence_auroc"),
                    ("f1", "occ_evidence_f1"),
                    ("mean", "occ_evidence_mean"),
                    ("pos_rate", "occ_evidence_positive_rate"),
                ],
            ),
            (
                "dyn_epi",
                [
                    ("dyn_loss", "dyn_loss"),
                    ("residual", "dyn_residual_mean"),
                    ("dyn_auc", "dyn_score_auroc"),
                    ("dyn_pr", "dyn_score_auprc"),
                    ("dyn_e_auc", "dyn_evidence_auroc"),
                    ("epi_auc", "epi_score_auroc"),
                    ("epi_pr", "epi_score_auprc"),
                    ("epi_e_auc", "epi_evidence_auroc"),
                ],
            ),
            (
                "nnpu",
                [
                    ("pos_term", "occ_positive_term"),
                    ("neg_u", "occ_negative_from_unlabeled"),
                    ("neg_pre", "occ_negative_risk_before_clamp"),
                    ("neg_post", "occ_negative_risk_after_clamp"),
                    ("pos_logit", "occ_positive_logit_mean"),
                    ("unl_logit", "occ_unlabeled_logit_mean"),
                ],
            ),
            (
                "extra",
                [
                    ("fuse", "fuse_loss"),
                    ("decor", "decor_loss"),
                    ("risk_tgt", "risk_target_mean"),
                    ("occ_nf", "occ_nonfinite_rate"),
                    ("occ_e_nf", "occ_evidence_nonfinite_rate"),
                    ("judge_nf", "judge_nonfinite_rate"),
                ],
            ),
        ]
        for section_name, pairs in sections:
            items = [(display_name, float(metrics[key])) for display_name, key in pairs if key in metrics]
            if items:
                print(f"  {section_name:<8}| {self._format_kv_row(items, key_width=10)}")

    def _init_wandb(self):
        logging_cfg = getattr(self.cfg, "logging", None)
        if logging_cfg is None or not bool(getattr(logging_cfg, "use_wandb", False)):
            return None
        try:
            import wandb
        except Exception:
            print("wandb is unavailable, skipping logging.")
            return None

        entity = str(getattr(logging_cfg, "entity", "songgao-personal"))
        mode = str(getattr(logging_cfg, "mode", "offline"))
        project = str(getattr(logging_cfg, "project", "robosuite-dyn-bce"))
        os.environ.setdefault("WANDB_MODE", mode)
        os.environ.setdefault("WANDB_ENTITY", entity)
        return wandb.init(
            project=project,
            entity=entity,
            mode=mode,
            config={
                "trainer": asdict(self.config),
                "metadata": self.metadata,
            },
        )

    def _step(
        self,
        batch: dict[str, torch.Tensor],
        train: bool,
        epoch: int | None = None,
        split_name: str | None = None,
        batch_idx: int | None = None,
    ) -> dict[str, torch.Tensor | dict[str, object] | None]:
        batch_state = self._scan_batch_nonfinite_state(batch)
        if batch_state is not None:
            self._raise_nonfinite_error(
                message=(
                    f"Non-finite batch tensor detected at epoch={epoch} split={split_name} batch={batch_idx}."
                ),
                batch_state=batch_state,
            )

        amp_enabled = bool(self.config.amp) and self.device.type == "cuda"
        autocast_kwargs = {
            "device_type": self.device.type,
            "enabled": amp_enabled,
        }
        if amp_enabled:
            autocast_kwargs["dtype"] = torch.float16
        with torch.amp.autocast(**autocast_kwargs):
            output = self.model(
                current_latent=batch["current_latent"],
                next_latent=batch["next_latent"],
                action_sequence=batch["action_sequence"],
                task_index=batch["task_index"],
            )
            occ_loss, occ_loss_details = occupancy_pu_loss(
                logits=output.occ_logit,
                data_type_index=batch["data_type_index"],
                sample_weights=batch["occ_weight"],
                positive_prior=float(self.config.occupancy_positive_prior),
                nnpu=bool(self.config.occupancy_nnpu),
                return_details=True,
            )
            fuse_loss = weighted_bce_with_logits(
                logits=output.judge_logit,
                targets=batch["risk_target"],
                sample_weights=batch["fuse_weight"],
            )
            dyn_loss = beta_nll_loss(
                pred_mean=output.ensemble_mean,
                pred_logvar=output.ensemble_logvar,
                target=output.ema_target,
                sample_weights=batch["dyn_weight"],
                beta=float(self.config.beta_nll),
            )
            decor_loss = cross_covariance_penalty(
                left=output.occ_private,
                right=output.dyn_private_mean,
            )
            total_loss = (
                occ_loss
                + float(self.config.alpha_dyn) * dyn_loss
                + float(self.config.eta_decor) * decor_loss
                + float(self.config.xi_fuse) * fuse_loss
            )

        positive_mask = batch["data_type_index"] != 2
        unlabeled_mask = batch["data_type_index"] == 2

        def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            mask = mask.reshape(-1)
            if int(mask.sum().item()) <= 0:
                return values.new_zeros(())
            flat = values.reshape(-1)
            return flat[mask].mean()

        nonfinite_debug = None
        if epoch is not None and split_name is not None and batch_idx is not None:
            nonfinite_debug = self._build_nonfinite_debug(
                epoch=int(epoch),
                split_name=str(split_name),
                batch_idx=int(batch_idx),
                batch=batch,
                output=output,
                occ_loss=occ_loss,
                fuse_loss=fuse_loss,
                dyn_loss=dyn_loss,
                decor_loss=decor_loss,
                total_loss=total_loss,
            )
        if nonfinite_debug is not None:
            self._raise_nonfinite_error(
                message=(
                    f"Non-finite forward/loss detected at epoch={epoch} split={split_name} batch={batch_idx}."
                ),
                debug=nonfinite_debug,
            )

        if train:
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            gradient_state = self._scan_gradient_nonfinite_state()
            if gradient_state is not None:
                self._raise_nonfinite_error(
                    message=(
                        f"Non-finite gradients detected at epoch={epoch} split={split_name} batch={batch_idx}."
                    ),
                    gradient_state=gradient_state,
                )
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config.grad_clip_norm))
            self.scaler.step(self.optimizer)
            self.scaler.update()
            model_state = self._scan_model_nonfinite_state()
            if model_state is not None:
                self._raise_nonfinite_error(
                    message=(
                        f"Non-finite model state detected after optimizer step at epoch={epoch} split={split_name} batch={batch_idx}."
                    ),
                    model_state=model_state,
                )
            self.model.update_ema(decay=float(self.config.ema_decay))

        return {
            "total_loss": total_loss.detach(),
            "occ_loss": occ_loss.detach(),
            "dyn_loss": dyn_loss.detach(),
            "decor_loss": decor_loss.detach(),
            "fuse_loss": fuse_loss.detach(),
            "occ_prob": output.occ_prob.detach(),
            "occ_evidence_prob": output.occ_evidence.detach(),
            "dyn_score": output.dyn_residual.detach(),
            "dyn_evidence_prob": output.dyn_evidence.detach(),
            "epi_score": output.epi_variance.detach(),
            "epi_evidence_prob": output.epi_evidence.detach(),
            "judge_prob": torch.sigmoid(output.judge_logit).detach(),
            "labels": batch["hard_label"].detach(),
            "nonfinite_debug": nonfinite_debug,
            "occ_risk_mean": output.occ_prob.detach().mean(),
            "occ_evidence_mean": output.occ_evidence.detach().mean(),
            "occ_positive_risk_mean": _masked_mean(output.occ_prob.detach(), positive_mask),
            "occ_unlabeled_risk_mean": _masked_mean(output.occ_prob.detach(), unlabeled_mask),
            "occ_positive_logit_mean": _masked_mean(output.occ_logit.detach(), positive_mask),
            "occ_unlabeled_logit_mean": _masked_mean(output.occ_logit.detach(), unlabeled_mask),
            "dyn_evidence_mean": output.dyn_evidence.detach().mean(),
            "epi_evidence_mean": output.epi_evidence.detach().mean(),
            "dyn_residual_mean": output.dyn_residual.detach().mean(),
            "epi_variance_mean": output.epi_variance.detach().mean(),
            "dyn_weight_mean": batch["dyn_weight"].detach().mean(),
            "risk_target_mean": batch["risk_target"].detach().mean(),
            "occ_positive_term": occ_loss_details["positive_term"],
            "occ_negative_from_positive": occ_loss_details["negative_from_positive"],
            "occ_negative_from_unlabeled": occ_loss_details["negative_from_unlabeled"],
            "occ_negative_risk_before_clamp": occ_loss_details["negative_risk_before_clamp"],
            "occ_negative_risk_after_clamp": occ_loss_details["negative_risk_after_clamp"],
            "occ_nnpu_clamped": occ_loss_details["nnpu_clamped"],
        }

    def _run_epoch(
        self,
        loader: DataLoader,
        train: bool,
        epoch: int,
        split_name: str,
        judge_threshold: float | None = None,
    ) -> dict[str, float]:
        self.model.train(mode=train)
        if not train:
            self.model.eval()

        loss_sums = {
            "total_loss": 0.0,
            "occ_loss": 0.0,
            "dyn_loss": 0.0,
            "decor_loss": 0.0,
            "fuse_loss": 0.0,
            "occ_risk_mean": 0.0,
            "occ_evidence_mean": 0.0,
            "occ_positive_risk_mean": 0.0,
            "occ_unlabeled_risk_mean": 0.0,
            "occ_positive_logit_mean": 0.0,
            "occ_unlabeled_logit_mean": 0.0,
            "occ_positive_term": 0.0,
            "occ_negative_from_positive": 0.0,
            "occ_negative_from_unlabeled": 0.0,
            "occ_negative_risk_before_clamp": 0.0,
            "occ_negative_risk_after_clamp": 0.0,
            "occ_nnpu_clamped": 0.0,
            "dyn_evidence_mean": 0.0,
            "epi_evidence_mean": 0.0,
            "dyn_residual_mean": 0.0,
            "epi_variance_mean": 0.0,
            "dyn_weight_mean": 0.0,
            "risk_target_mean": 0.0,
        }
        num_batches = 0
        occ_probs: list[np.ndarray] = []
        occ_evidence_probs: list[np.ndarray] = []
        dyn_scores: list[np.ndarray] = []
        dyn_evidence_scores: list[np.ndarray] = []
        epi_scores: list[np.ndarray] = []
        epi_evidence_scores: list[np.ndarray] = []
        judge_probs: list[np.ndarray] = []
        labels: list[np.ndarray] = []

        for batch_idx, batch in enumerate(loader):
            batch = _to_device(batch, self.device)
            with torch.set_grad_enabled(train):
                outputs = self._step(batch=batch, train=train, epoch=epoch, split_name=split_name, batch_idx=batch_idx + 1)

            for key in loss_sums:
                loss_sums[key] += float(outputs[key].item())
            occ_probs.append(outputs["occ_prob"].cpu().numpy())
            occ_evidence_probs.append(outputs["occ_evidence_prob"].cpu().numpy())
            dyn_scores.append(outputs["dyn_score"].cpu().numpy())
            dyn_evidence_scores.append(outputs["dyn_evidence_prob"].cpu().numpy())
            epi_scores.append(outputs["epi_score"].cpu().numpy())
            epi_evidence_scores.append(outputs["epi_evidence_prob"].cpu().numpy())
            judge_probs.append(outputs["judge_prob"].cpu().numpy())
            labels.append(outputs["labels"].cpu().numpy())
            num_batches += 1

            if outputs.get("nonfinite_debug") is not None:
                self._print_nonfinite_debug(outputs["nonfinite_debug"])

            if train and (batch_idx + 1) % max(int(self.config.log_every), 1) == 0:
                self._print_train_batch_log(
                    epoch=epoch,
                    split_name=split_name,
                    batch_idx=batch_idx + 1,
                    num_batches=len(loader),
                    outputs=outputs,
                )

        metrics = {
            key: value / max(num_batches, 1)
            for key, value in loss_sums.items()
        }
        label_np = np.concatenate(labels, axis=0) if labels else np.asarray([], dtype=np.int64)
        occ_np = np.concatenate(occ_probs, axis=0) if occ_probs else np.asarray([], dtype=np.float32)
        occ_evidence_np = (
            np.concatenate(occ_evidence_probs, axis=0) if occ_evidence_probs else np.asarray([], dtype=np.float32)
        )
        dyn_score_np = np.concatenate(dyn_scores, axis=0) if dyn_scores else np.asarray([], dtype=np.float32)
        dyn_evidence_np = (
            np.concatenate(dyn_evidence_scores, axis=0) if dyn_evidence_scores else np.asarray([], dtype=np.float32)
        )
        epi_score_np = np.concatenate(epi_scores, axis=0) if epi_scores else np.asarray([], dtype=np.float32)
        epi_evidence_np = (
            np.concatenate(epi_evidence_scores, axis=0) if epi_evidence_scores else np.asarray([], dtype=np.float32)
        )
        judge_np = np.concatenate(judge_probs, axis=0) if judge_probs else np.asarray([], dtype=np.float32)
        occ_metrics = compute_binary_metrics(occ_np, label_np)
        occ_evidence_metrics = compute_binary_metrics(occ_evidence_np, label_np)
        dyn_score_metrics = compute_score_ranking_metrics(dyn_score_np, label_np)
        dyn_evidence_metrics = compute_score_ranking_metrics(dyn_evidence_np, label_np)
        epi_score_metrics = compute_score_ranking_metrics(epi_score_np, label_np)
        epi_evidence_metrics = compute_score_ranking_metrics(epi_evidence_np, label_np)

        judge_threshold_value = float(judge_threshold) if judge_threshold is not None else 0.5
        if (not train) and split_name == "eval":
            threshold_info = estimate_best_threshold(judge_np, label_np, objective="f1", default_threshold=0.5)
            judge_threshold_value = float(threshold_info["threshold"])
            metrics["judge_threshold_objective"] = float(threshold_info["objective"])
            metrics["judge_threshold_balanced_accuracy"] = float(threshold_info["balanced_accuracy"])
            metrics["judge_threshold_positive_rate"] = float(threshold_info["positive_rate"])
        judge_metrics = compute_binary_metrics(judge_np, label_np, threshold=judge_threshold_value)
        metrics["judge_threshold"] = float(judge_threshold_value)
        for key, value in occ_metrics.items():
            metrics[f"occ_{key}"] = float(value)
        for key, value in occ_evidence_metrics.items():
            metrics[f"occ_evidence_{key}"] = float(value)
        for key, value in dyn_score_metrics.items():
            metrics[f"dyn_score_{key}"] = float(value)
        for key, value in dyn_evidence_metrics.items():
            metrics[f"dyn_evidence_{key}"] = float(value)
        for key, value in epi_score_metrics.items():
            metrics[f"epi_score_{key}"] = float(value)
        for key, value in epi_evidence_metrics.items():
            metrics[f"epi_evidence_{key}"] = float(value)
        for key, value in judge_metrics.items():
            metrics[f"judge_{key}"] = float(value)
        return metrics

    def _format_metric_line(self, metrics: dict[str, float], keys: list[str]) -> str:
        parts = [f"{key}={metrics[key]:.4f}" for key in keys if key in metrics]
        return " ".join(parts)

    def _log_metrics(self, split_name: str, epoch: int, metrics: dict[str, float]) -> None:
        display_split = self._display_split_name(split_name)
        if split_name == "train":
            print(f"[Epoch {epoch:03d} | {display_split}]")
            print(
                "  summary | "
                + self._format_kv_row(
                    [
                        ("loss", float(metrics.get("total_loss", 0.0))),
                        ("judge_auc", float(metrics.get("judge_auroc", 0.0))),
                        ("judge_f1", float(metrics.get("judge_f1", 0.0))),
                        ("judge_thr", float(metrics.get("judge_threshold", 0.5))),
                        ("occ_auc", float(metrics.get("occ_auroc", 0.0))),
                        ("occ_f1", float(metrics.get("occ_f1", 0.0))),
                        ("dyn_auc", float(metrics.get("dyn_score_auroc", 0.0))),
                        ("epi_auc", float(metrics.get("epi_score_auroc", 0.0))),
                        ("dyn_loss", float(metrics.get("dyn_loss", 0.0))),
                    ],
                    key_width=9,
                )
            )
            print(
                "  signal  | "
                + self._format_kv_row(
                    [
                        ("occ_raw", float(metrics.get("occ_risk_mean", 0.0))),
                        ("occ_evi", float(metrics.get("occ_evidence_mean", 0.0))),
                        ("dyn_evi", float(metrics.get("dyn_evidence_mean", 0.0))),
                        ("epi_evi", float(metrics.get("epi_evidence_mean", 0.0))),
                        ("risk_tgt", float(metrics.get("risk_target_mean", 0.0))),
                        ("nnpu", float(metrics.get("occ_nnpu_clamped", 0.0))),
                    ],
                    key_width=8,
                )
            )
        else:
            self._print_eval_summary(epoch=epoch, split_name=split_name, metrics=metrics)
        if self.wandb_run is not None:
            log_payload = {f"{display_split}/{key}": value for key, value in metrics.items()}
            log_payload["epoch"] = int(epoch)
            self.wandb_run.log(log_payload)

    def _checkpoint_payload(self, epoch: int) -> dict:
        return {
            "model": self.model.state_dict(),
            "cfg": self.cfg,
            "trainer_config": asdict(self.config),
            "metadata": self.metadata,
            "epoch": int(epoch),
            "history": self.history,
            "best_eval_threshold": float(self.best_eval_threshold),
        }

    def fit(
        self,
        save_dir: str,
        save_name: str,
    ) -> tuple[dict[str, list[dict[str, float]]], dict[str, float]]:
        os.makedirs(save_dir, exist_ok=True)
        if not self._startup_integrity_reported:
            self._run_startup_integrity_check()
        save_path = os.path.join(save_dir, save_name)
        stem, ext = os.path.splitext(save_name)
        if ext == "":
            ext = ".pt"

        for epoch in range(1, int(self.config.epochs) + 1):
            train_metrics = self._run_epoch(self.train_loader, train=True, epoch=epoch, split_name="train")
            eval_metrics = self._run_epoch(self.eval_loader, train=False, epoch=epoch, split_name="eval")
            self.history["train"].append(train_metrics)
            self.history["eval"].append(eval_metrics)
            self._log_metrics("train", epoch, train_metrics)
            self._log_metrics("eval", epoch, eval_metrics)

            eval_score = float(eval_metrics.get("judge_f1", 0.0) + eval_metrics.get("judge_auroc", 0.0))
            if eval_score > self.best_eval_score:
                self.best_eval_score = eval_score
                self.best_eval_threshold = float(eval_metrics.get("judge_threshold", 0.5))
                self.best_val_score = self.best_eval_score
                self.best_state_dict = copy.deepcopy(self.model.state_dict())
                best_path = os.path.join(save_dir, f"{stem}_best{ext}")
                torch.save(self._checkpoint_payload(epoch), best_path)
                print(f"Saved best checkpoint to: {best_path}")

            if int(self.config.save_freq) > 0 and epoch % int(self.config.save_freq) == 0:
                periodic_path = os.path.join(save_dir, f"{stem}_ep{epoch:04d}{ext}")
                torch.save(self._checkpoint_payload(epoch), periodic_path)
                print(f"Saved periodic checkpoint to: {periodic_path}")

        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict, strict=True)
        test_metrics = self._run_epoch(
            self.test_loader,
            train=False,
            epoch=int(self.config.epochs),
            split_name="test",
            judge_threshold=float(self.best_eval_threshold),
        )
        self.history["test"].append(test_metrics)
        self._log_metrics("test", int(self.config.epochs), test_metrics)

        final_payload = self._checkpoint_payload(int(self.config.epochs))
        final_payload["test_metrics"] = test_metrics
        torch.save(final_payload, save_path)
        print(f"Saved final checkpoint to: {save_path}")

        summary_path = os.path.join(save_dir, f"{stem}_summary.json")
        with open(summary_path, "w", encoding="utf-8") as file_handle:
            json.dump(
                {
                    "history": self.history,
                    "metadata": self.metadata,
                    "best_eval_score": self.best_eval_score,
                    "best_eval_threshold": float(self.best_eval_threshold),
                    "best_val_score": self.best_val_score,
                    "test_metrics": test_metrics,
                },
                file_handle,
                indent=2,
                sort_keys=True,
            )
        print(f"Saved training summary to: {summary_path}")

        if self.wandb_run is not None:
            self.wandb_run.finish()
        return self.history, test_metrics
