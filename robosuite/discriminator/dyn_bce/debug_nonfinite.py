from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from robosuite.discriminator.dyn_bce.modules.model import DynBCEModel
from robosuite.discriminator.dyn_bce.utils.dataset import (
    DynBCETransitionDataset,
    estimate_weighted_occupancy_positive_prior,
)
from robosuite.discriminator.dyn_bce.utils.losses import (
    beta_nll_loss,
    cross_covariance_penalty,
    occupancy_pu_loss,
    weighted_bce_with_logits,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a short dyn_bce training loop and stop on the first non-finite batch/activation/gradient."
    )
    parser.add_argument(
        "--config",
        default="robosuite/discriminator/dyn_bce/config/train.yaml",
        help="Path to the dyn_bce training config.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Explicit split manifest.json to debug. If omitted, auto-picks a train manifest.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Maximum optimizer steps to run.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size for the debug loop.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override device. Defaults to cfg.training.device when CUDA is available, otherwise cpu.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for DataLoader shuffling.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Print one metric line every N steps.",
    )
    return parser.parse_args()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_manifest(cfg, manifest_override: str | None) -> str:
    if manifest_override:
        manifest_path = Path(manifest_override)
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        return str(manifest_path)

    memmap_dir = Path(str(cfg.data.memmap.dir))
    horizon = int(cfg.data.transition_horizon)
    candidates = []
    for manifest_path in glob.glob(str(memmap_dir / "*" / "train" / "manifest.json")):
        with open(manifest_path, "r", encoding="utf-8") as file_handle:
            manifest = json.load(file_handle)
        if int(manifest.get("transition_horizon", -1)) != horizon:
            continue
        candidates.append((os.path.getmtime(manifest_path), manifest_path))

    if not candidates:
        raise FileNotFoundError(
            f"No train manifests found under {memmap_dir} matching transition_horizon={horizon}."
        )

    candidates.sort()
    return str(candidates[-1][1])


def _tensor_summary(tensor: torch.Tensor) -> str:
    flat = tensor.detach().reshape(-1).float()
    finite = torch.isfinite(flat)
    finite_count = int(finite.sum().item())
    total = int(flat.numel())
    if finite_count <= 0:
        return f"finite=0/{total}"
    finite_flat = flat[finite]
    return (
        f"finite={finite_count}/{total} "
        f"mean={float(finite_flat.mean()):.4f} std={float(finite_flat.std(unbiased=False)):.4f} "
        f"min={float(finite_flat.min()):.4f} max={float(finite_flat.max()):.4f}"
    )


def _check_named_tensors(named_tensors) -> list[tuple[str, str]]:
    failures = []
    for name, tensor in named_tensors:
        if tensor is None:
            continue
        detached = tensor.detach()
        if not torch.isfinite(detached).all():
            failures.append((name, _tensor_summary(detached)))
    return failures


def main() -> None:
    args = _parse_args()
    cfg = OmegaConf.load(args.config)
    _seed_everything(int(args.seed))

    manifest_path = _resolve_manifest(cfg, args.manifest)
    dataset = DynBCETransitionDataset(manifest_path)
    batch_size = int(args.batch_size or cfg.training.batch_size)
    device_name = str(args.device or cfg.training.device)
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")

    generator = torch.Generator()
    generator.manual_seed(int(args.seed))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )

    first_batch = next(iter(loader))
    latent_dim = int(first_batch["current_latent"].shape[-1])
    action_dim = int(first_batch["action_sequence"].shape[-1])
    num_tasks = int(len(dataset.task_names))
    prior = estimate_weighted_occupancy_positive_prior(
        fail_onset_ratio=float(cfg.labels.fail_onset_ratio),
        risk_temperature=float(cfg.labels.risk_temperature),
        occ_fail_prefix_min_weight=float(cfg.labels.occ_fail_prefix_min_weight),
    )

    model = DynBCEModel(
        latent_dim=latent_dim,
        action_dim=action_dim,
        num_tasks=num_tasks,
        shared_dim=int(cfg.model.shared_dim),
        occ_private_dim=int(cfg.model.occ_private_dim),
        dyn_private_dim=int(cfg.model.dyn_private_dim),
        task_embed_dim=int(cfg.model.task_embed_dim),
        trunk_hidden_dim=int(cfg.model.trunk_hidden_dim),
        head_hidden_dim=int(cfg.model.head_hidden_dim),
        action_model_dim=int(cfg.model.action_model_dim),
        action_num_layers=int(cfg.model.action_num_layers),
        action_num_heads=int(cfg.model.action_num_heads),
        action_dropout=float(cfg.model.action_dropout),
        max_action_horizon=int(cfg.model.max_action_horizon),
        ensemble_size=int(cfg.model.ensemble_size),
        judge_hidden_dim=int(cfg.model.judge_hidden_dim),
        dyn_model_dim=int(cfg.model.dyn_model_dim),
        dyn_backbone_num_blocks=int(cfg.model.dyn_backbone_num_blocks),
        dyn_head_hidden_dim=int(cfg.model.dyn_head_hidden_dim),
        dyn_head_num_blocks=int(cfg.model.dyn_head_num_blocks),
        trunk_num_blocks=int(cfg.model.trunk_num_blocks),
        head_num_blocks=int(cfg.model.head_num_blocks),
        judge_num_blocks=int(cfg.model.judge_num_blocks),
        swiglu_hidden_ratio=float(cfg.model.swiglu_hidden_ratio),
        occupancy_use_spectral_norm=bool(cfg.model.occupancy_use_spectral_norm),
        judge_use_spectral_norm=bool(cfg.model.judge_use_spectral_norm),
        occ_logit_scale=float(cfg.model.occ_logit_scale),
        occ_logit_temperature=float(cfg.model.occ_logit_temperature),
        zero_init_residual=bool(cfg.model.zero_init_residual),
        occ_calibrator_momentum=float(cfg.model.occ_calibrator_momentum),
        evidence_calibrator_eps=float(cfg.model.evidence_calibrator_eps),
        dropout=float(cfg.model.dropout),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
    )
    model.train()

    print(f"manifest={manifest_path}")
    print(f"device={device} batch_size={batch_size} steps={int(args.steps)} prior={prior:.6f}")
    print(f"latent_dim={latent_dim} action_dim={action_dim} num_tasks={num_tasks}")

    for step, batch in enumerate(loader, start=1):
        if step > int(args.steps):
            print(f"completed {int(args.steps)} steps without non-finite values")
            return

        batch = {key: value.to(device) for key, value in batch.items()}
        batch_failures = _check_named_tensors((f"batch.{name}", tensor) for name, tensor in batch.items())
        if batch_failures:
            print(f"non-finite batch tensors detected at step={step}")
            for name, summary in batch_failures:
                print(f"  {name}: {summary}")
            return

        output = model(
            current_latent=batch["current_latent"],
            next_latent=batch["next_latent"],
            action_sequence=batch["action_sequence"],
            task_index=batch["task_index"],
        )
        occ_loss, _ = occupancy_pu_loss(
            logits=output.occ_logit,
            data_type_index=batch["data_type_index"],
            sample_weights=batch["occ_weight"],
            positive_prior=float(prior),
            nnpu=bool(cfg.loss.occupancy_nnpu),
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
            beta=float(cfg.loss.beta_nll),
        )
        decor_loss = cross_covariance_penalty(
            left=output.occ_private,
            right=output.dyn_private_mean,
        )
        total_loss = (
            occ_loss
            + float(cfg.loss.alpha_dyn) * dyn_loss
            + float(cfg.loss.eta_decor) * decor_loss
            + float(cfg.loss.xi_fuse) * fuse_loss
        )

        forward_failures = _check_named_tensors(
            [
                ("output.occ_logit", output.occ_logit),
                ("output.judge_logit", output.judge_logit),
                ("output.ensemble_mean", output.ensemble_mean),
                ("output.ensemble_logvar", output.ensemble_logvar),
                ("output.ema_target", output.ema_target),
                ("output.dyn_residual", output.dyn_residual),
                ("output.epi_variance", output.epi_variance),
                ("loss.occ", occ_loss),
                ("loss.fuse", fuse_loss),
                ("loss.dyn", dyn_loss),
                ("loss.decor", decor_loss),
                ("loss.total", total_loss),
            ]
        )
        if forward_failures:
            print(f"non-finite forward tensors detected at step={step}")
            for name, summary in forward_failures:
                print(f"  {name}: {summary}")
            return

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()

        grad_failures = _check_named_tensors(
            (f"grad.{name}", parameter.grad)
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        )
        if grad_failures:
            print(f"non-finite gradients detected at step={step}")
            for name, summary in grad_failures[:20]:
                print(f"  {name}: {summary}")
            return

        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.training.grad_clip_norm))
        optimizer.step()
        model.update_ema(decay=float(cfg.training.ema_decay))

        state_failures = _check_named_tensors(model.named_parameters())
        state_failures.extend(_check_named_tensors(model.named_buffers()))
        if state_failures:
            print(f"non-finite model state detected at step={step}")
            for name, summary in state_failures[:20]:
                print(f"  {name}: {summary}")
            return

        if int(args.log_every) > 0 and step % int(args.log_every) == 0:
            print(
                f"step={step:04d} "
                f"total={float(total_loss):.6f} occ={float(occ_loss):.6f} "
                f"dyn={float(dyn_loss):.6f} fuse={float(fuse_loss):.6f} "
                f"decor={float(decor_loss):.6f}"
            )

    print("completed the full DataLoader without non-finite values")


if __name__ == "__main__":
    main()
