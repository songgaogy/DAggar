"""CLI entry for training the D3-Disc dynamics predictor.

Predictor input:  (z_t, s_t, a_{t:t+h})  where z_t = flow_multi.encode_context.
Predictor target: (z_{t+h}, s_{t+h})
Encoder is frozen; this script only trains the DynamicsPredictor.

Example:
    python -m robosuite.discriminator.d3disc.train_dynamics \
        --policy-ckpt checkpoints/multitask_6/policy/flow-20/flow_multi_ep0100_*.pt \
        --expert-paths data/PickPlaceBread/expert data/PickPlaceCereal/expert \
        --rollout-paths data/PickPlaceBread/success_rollout data/PickPlaceCereal/success_rollout \
        --save-dir checkpoints/d3disc/dynamics \
        --epochs 30 --batch-size 256
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import Dict, List, Optional

import torch
from torch.utils.data import random_split

from robosuite.discriminator.lpb.model import DynamicsPredictor

from .dataset import LatentFlowDynamicsDataset
from .encoder import FlowMultiEncoderWrapper
from .trainer import D3DynamicsTrainer, D3TrainerConfig


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-ckpt", required=True,
                        help="flow_multi checkpoint (used for frozen encoder + cache keys).")
    parser.add_argument("--cache-root", type=str, default="data/.lpb_score_cache")
    parser.add_argument("--expert-paths", nargs="*", required=True,
                        help="HDF5 files or directories containing expert demos.")
    parser.add_argument("--rollout-paths", nargs="*", default=[],
                        help="HDF5 files or directories containing success rollouts.")
    parser.add_argument("--proprio-indices", type=int, nargs="*", default=None,
                        help="Optional state indices; default is right-pad per LPB.")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--max-trajectories-per-kind", type=int, default=0,
                        help="<=0 means use all.")

    # Predictor architecture (LPB defaults).
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-action-horizon", type=int, default=32)

    # Training.
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--expert-ratio", type=float, default=0.5)
    parser.add_argument("--proprio-loss-weight", type=float, default=0.1)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)

    # Save.
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--save-name", type=str, default="",
                        help="Default 'd3_dynamics_<tag>.pt' if empty.")
    parser.add_argument("--save-freq", type=int, default=0,
                        help="Save periodic checkpoints every N epochs (0 = final only).")
    return parser.parse_args()


def _build_payload(
    predictor: DynamicsPredictor,
    history: Dict[str, Dict[str, Dict[str, float]]],
    args: argparse.Namespace,
    latent_dim: int,
    action_dim: int,
    proprio_dim: int,
    epoch: int,
) -> dict:
    return {
        "model": predictor.state_dict(),
        "history": history,
        "args": {k: v for k, v in vars(args).items()},
        "latent_dim": int(latent_dim),
        "action_dim": int(action_dim),
        "proprio_dim": int(proprio_dim),
        "horizon": int(args.horizon),
        "d_model": int(args.d_model),
        "num_layers": int(args.num_layers),
        "num_heads": int(args.num_heads),
        "dropout": float(args.dropout),
        "max_action_horizon": int(args.max_action_horizon),
        "policy_ckpt_path": str(args.policy_ckpt),
        "proprio_indices": None if not args.proprio_indices else list(args.proprio_indices),
        "epoch": int(epoch),
    }


def main() -> None:
    args = _parse_args()

    seed = int(args.seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    encoder = FlowMultiEncoderWrapper(
        policy_ckpt_path=str(args.policy_ckpt),
        cache_root=str(args.cache_root),
        device=str(args.device),
    )

    dataset = LatentFlowDynamicsDataset(
        encoder=encoder,
        expert_paths=list(args.expert_paths),
        rollout_paths=list(args.rollout_paths),
        horizon=int(args.horizon),
        proprio_indices=list(args.proprio_indices) if args.proprio_indices else None,
        max_trajectories_per_kind=(
            None if int(args.max_trajectories_per_kind) <= 0 else int(args.max_trajectories_per_kind)
        ),
    )
    print(
        f"[d3_train] dataset size={len(dataset)}  "
        f"expert_samples={dataset.num_expert_samples}  rollout_samples={dataset.num_rollout_samples}  "
        f"action_dim={dataset.action_dim}  proprio_dim={dataset.proprio_dim}  "
        f"latent_dim={dataset.latent_dim}"
    )

    # Forked DataLoader workers cannot safely init CUDA after the parent has. Ensure
    # every demo has an on-disk latent cache before workers run (see dataset docstring).
    if int(args.num_workers) > 0 and getattr(dataset.encoder, "device", None) is not None:
        if dataset.encoder.device.type == "cuda":
            n_new = dataset.materialize_missing_latent_caches()
            if n_new > 0:
                print(f"[d3_train] materialized {n_new} missing latent cache(s) on main process")

    n_total = len(dataset)
    n_val = int(round(float(args.val_ratio) * n_total))
    n_val = min(max(n_val, 0), n_total - 1) if n_total > 1 else 0
    n_train = n_total - n_val
    split_gen = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val], generator=split_gen)

    predictor = DynamicsPredictor(
        latent_dim=dataset.latent_dim,
        proprio_dim=dataset.proprio_dim,
        action_dim=dataset.action_dim,
        d_model=int(args.d_model),
        num_layers=int(args.num_layers),
        nhead=int(args.num_heads),
        dropout=float(args.dropout),
        max_action_horizon=max(int(args.max_action_horizon), int(args.horizon)),
    )

    trainer_cfg = D3TrainerConfig(
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        learning_rate=float(args.lr),
        weight_decay=float(args.weight_decay),
        epochs=int(args.epochs),
        expert_sampling_ratio=float(args.expert_ratio),
        proprio_loss_weight=float(args.proprio_loss_weight),
        grad_clip_norm=float(args.grad_clip_norm),
        log_every=int(args.log_every),
    )
    trainer = D3DynamicsTrainer(
        predictor=predictor,
        train_dataset=train_dataset,
        val_dataset=val_dataset if n_val > 0 else None,
        config=trainer_cfg,
        device=str(args.device),
    )

    save_dir = os.path.abspath(str(args.save_dir))
    os.makedirs(save_dir, exist_ok=True)
    save_name = str(args.save_name).strip()
    if not save_name:
        save_name = f"d3_dynamics_{_now_tag()}.pt"
    save_path_final = os.path.join(save_dir, save_name)
    stem, ext = os.path.splitext(save_name)
    if ext == "":
        ext = ".pt"

    def _save_periodic(epoch: int, hist: Dict[str, Dict[str, Dict[str, float]]]) -> None:
        periodic_name = f"{stem}_ep{epoch:04d}{ext}"
        periodic_path = os.path.join(save_dir, periodic_name)
        payload = _build_payload(
            predictor=predictor,
            history=hist,
            args=args,
            latent_dim=dataset.latent_dim,
            action_dim=dataset.action_dim,
            proprio_dim=dataset.proprio_dim,
            epoch=epoch,
        )
        torch.save(payload, periodic_path)
        print(f"[d3_train] saved periodic checkpoint: {periodic_path}")

    history = trainer.fit(save_freq=int(args.save_freq), save_callback=_save_periodic)

    final_payload = _build_payload(
        predictor=predictor,
        history=history,
        args=args,
        latent_dim=dataset.latent_dim,
        action_dim=dataset.action_dim,
        proprio_dim=dataset.proprio_dim,
        epoch=int(args.epochs),
    )
    torch.save(final_payload, save_path_final)
    print(f"[d3_train] saved final checkpoint: {save_path_final}")


if __name__ == "__main__":
    main()
