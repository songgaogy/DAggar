"""Train a D4-Disc dynamics/critic model from a preprocessed cache.

This script is a thin CLI wrapper around the core training components under
``robosuite.discriminator.d4disc``:

- ``PreprocessedCacheReader`` + ``LatentFlowDynamicsDatasetD4`` load trajectories
  that have already been converted into a compact on-disk cache (images, actions,
  proprio, and latents / flow targets as applicable).
- ``Encoder`` optionally loads / freezes an image encoder used to produce latents.
- ``ConditionalDynamicsPredictor`` is the AdaLN-conditioned transformer backbone
  trained with a two-phase schedule (warm-up, then bootstrap).
- ``D4Trainer`` executes training, evaluation probes, EMA updates, and checkpointing.

The goal is reproducible training from cached data: point this at a cache root
and one or more trajectory lists (expert/rollout/fail), and it will train and
save a model checkpoint under ``--save-dir``.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

import torch

from .data import LatentFlowDynamicsDatasetD4, PreprocessedCacheReader
from .models import ConditionalDynamicsPredictor, Encoder
from .training import D4Schedule, D4Trainer, D4TrainerConfig


def _now_tag() -> str:
    """Return a filesystem-friendly timestamp tag (local time)."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for D4-Disc training.

    The flags are grouped roughly by:
    - data/cache selection
    - encoder warm start / freezing
    - predictor architecture
    - training schedule and optimization
    - stability knobs (EMA, clamps, held-out split)
    - logging / checkpointing
    """
    p = argparse.ArgumentParser()
    p.add_argument("--preprocessed-cache-root", type=str, default="data/.lpb_score_preprocessed_cache")
    p.add_argument("--expert-paths", nargs="*", default=[])
    p.add_argument("--rollout-paths", nargs="*", default=[])
    p.add_argument("--fail-paths", nargs="*", default=[])
    p.add_argument("--proprio-indices", type=int, nargs="*", default=None)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--max-trajectories-per-kind", type=int, default=0)
    p.add_argument("--image-size", type=int, default=128)

    p.add_argument("--encoder-checkpoint", type=str, default="")
    enc_pre = p.add_mutually_exclusive_group()
    enc_pre.add_argument("--encoder-pretrained", dest="encoder_pretrained", action="store_true")
    enc_pre.add_argument("--no-encoder-pretrained", dest="encoder_pretrained", action="store_false")
    p.set_defaults(encoder_pretrained=False)
    enc_frz = p.add_mutually_exclusive_group()
    enc_frz.add_argument("--encoder-freeze", dest="encoder_freeze", action="store_true")
    enc_frz.add_argument("--no-encoder-freeze", dest="encoder_freeze", action="store_false")
    p.set_defaults(encoder_freeze=False)
    p.add_argument("--encoder-lr-mult", type=float, default=0.1)

    # Architecture
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max-action-horizon", type=int, default=32)
    p.add_argument("--d-cond", type=int, default=64)
    p.add_argument("--adaln-init-std", type=float, default=0.0)

    # Two-phase schedule
    p.add_argument("--warm-up-epochs", type=int, default=8)
    p.add_argument("--bootstrap-epochs", type=int, default=40)

    # Optimizer
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--proprio-loss-weight", type=float, default=0.1)
    p.add_argument("--latent-loss-weight", type=float, default=1.0)
    p.add_argument("--sigma-sq", type=float, default=0.5)
    p.add_argument("--use-bfloat16", action="store_true")

    # Branch routing
    p.add_argument("--min-batch-balance", type=float, default=0.15)

    # Schedule knobs
    p.add_argument("--alpha-exponent", type=float, default=0.75)
    p.add_argument("--alpha-cap-rate", type=float, default=1.0)
    p.add_argument("--eta-final", type=float, default=0.3)
    p.add_argument("--eta-ramp-epochs", type=int, default=5)

    # Stability
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--ema-alpha-gamma", type=float, default=0.5)
    p.add_argument("--gamma-clamp", type=float, default=1e-3)
    p.add_argument("--held-out-ratio", type=float, default=0.05)

    # Warm-start
    wm = p.add_mutually_exclusive_group()
    wm.add_argument("--f3-warm-start", dest="f3_warm_start", action="store_true")
    wm.add_argument("--no-f3-warm-start", dest="f3_warm_start", action="store_false")
    p.set_defaults(f3_warm_start=True)
    p.add_argument("--f3-warm-start-k", type=int, default=1)
    p.add_argument("--f3-warm-start-max-clean", type=int, default=100_000)
    p.add_argument("--warm-start-mode", type=str, default="rank", choices=["rank", "sigmoid"])
    p.add_argument("--gate-freeze-epochs", type=int, default=2)
    p.add_argument("--skip-bootstrap", action="store_true")

    # Phase-B advantage gate.
    p.add_argument(
        "--advantage-mode",
        type=str,
        default="knn",
        choices=["knn", "residual"],
        help=(
            "Phase-B advantage score. 'knn' (default): r_c = min-sqdist(f(c), expert z_{t+h} bank); "
            "advantage = log p_+ - log p_- under Gaussian-KNN. 'residual': legacy raw MSE to z_target."
        ),
    )
    p.add_argument("--advantage-knn-bank-size", type=int, default=100_000)
    p.add_argument("--advantage-knn-chunk-size", type=int, default=8192)
    p.add_argument("--advantage-knn-k", type=int, default=1)

    # Logging
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-freq", type=int, default=0)
    p.add_argument("--run-name", type=str, default="")
    p.add_argument("--wandb-project", type=str, default="d4disc")
    p.add_argument("--wandb-mode", type=str, default="offline")
    p.add_argument("--disable-wandb", action="store_true")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-dir", required=True)
    p.add_argument("--save-name", type=str, default="")
    return p.parse_args()


def main() -> None:
    """CLI main.

    Wires dataset + model + trainer, then runs training and writes a checkpoint.
    """
    args = _parse_args()

    seed = int(args.seed)
    # Seed torch RNGs for reproducibility. (Data shuffling and some CUDA kernels
    # may still be nondeterministic depending on your environment.)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Build a cache reader that knows how to locate and decode cached episodes.
    cache_reader = PreprocessedCacheReader(
        cache_root=str(args.preprocessed_cache_root),
        image_size=int(args.image_size),
        camera_index=0,
    )
    # Dataset aggregates trajectories from the provided lists and exposes
    # consistent tensor shapes (latent/proprio/action) for training.
    dataset = LatentFlowDynamicsDatasetD4(
        cache_reader=cache_reader,
        expert_paths=list(args.expert_paths),
        rollout_paths=list(args.rollout_paths),
        fail_rollout_paths=list(args.fail_paths),
        horizon=int(args.horizon),
        proprio_indices=list(args.proprio_indices) if args.proprio_indices else None,
        max_trajectories_per_kind=(
            None
            if int(args.max_trajectories_per_kind) <= 0
            else int(args.max_trajectories_per_kind)
        ),
        image_size=int(args.image_size),
    )
    print(
        f"[d4_train] dataset size={len(dataset)}  "
        f"expert_samples={dataset.num_expert_samples}  "
        f"rollout_samples={dataset.num_rollout_samples}  "
        f"fail_raw_samples={dataset.num_fail_raw_samples}  "
        f"action_dim={dataset.action_dim}  proprio_dim={dataset.proprio_dim}  "
        f"latent_dim={dataset.latent_dim}"
    )

    # Materialize / warm up any on-disk indices so the first epoch does not pay
    # the cache discovery cost.
    n_demos = dataset.preload_preprocessed()
    print(f"[d4_train] preloaded cached demos for {n_demos} trajectory(s)")

    encoder_checkpoint = str(args.encoder_checkpoint).strip() or None
    encoder = Encoder(
        checkpoint_path=encoder_checkpoint,
        # If a checkpoint is provided, treat it as the source of truth. Otherwise,
        # "pretrained" controls whether to initialize from the default pretrained
        # weights (if implemented by the Encoder).
        pretrained=(bool(args.encoder_pretrained) if encoder_checkpoint is None else False),
        freeze=bool(args.encoder_freeze),
        normalize_input=True,
    )

    predictor = ConditionalDynamicsPredictor(
        latent_dim=int(encoder.latent_dim),
        proprio_dim=dataset.proprio_dim,
        action_dim=dataset.action_dim,
        d_model=int(args.d_model),
        num_layers=int(args.num_layers),
        nhead=int(args.num_heads),
        dropout=float(args.dropout),
        max_action_horizon=max(int(args.max_action_horizon), int(args.horizon)),
        d_cond=int(args.d_cond),
        adaln_init_std=float(args.adaln_init_std),
    )

    # Schedule controls phase mixing / importance weights during training.
    schedule = D4Schedule(
        alpha_exponent=float(args.alpha_exponent),
        alpha_cap_rate=float(args.alpha_cap_rate),
        eta_final=float(args.eta_final),
        eta_ramp_epochs=int(args.eta_ramp_epochs),
    )

    # TrainerConfig is a single source of truth for all optimization and
    # stability knobs (EMA, clamps, warm start behavior, logging cadence, etc.).
    trainer_cfg = D4TrainerConfig(
        warm_up_epochs=int(args.warm_up_epochs),
        bootstrap_epochs=int(args.bootstrap_epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        grad_clip_norm=float(args.grad_clip_norm),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        encoder_freeze=bool(args.encoder_freeze),
        encoder_lr_mult=float(args.encoder_lr_mult),
        min_batch_balance=float(args.min_batch_balance),
        schedule=schedule,
        ema_decay_model=float(args.ema_decay),
        ema_alpha_gamma=float(args.ema_alpha_gamma),
        gamma_clamp=float(args.gamma_clamp),
        held_out_ratio=float(args.held_out_ratio),
        proprio_loss_weight=float(args.proprio_loss_weight),
        latent_loss_weight=float(args.latent_loss_weight),
        sigma_sq=float(args.sigma_sq),
        use_bfloat16=bool(args.use_bfloat16),
        log_every=int(args.log_every),
        save_freq=int(args.save_freq),
        run_name=str(args.run_name),
        wandb_project=str(args.wandb_project),
        wandb_mode=str(args.wandb_mode),
        disable_wandb=bool(args.disable_wandb),
        probe_batch_size=64,
        f3_warm_start=bool(args.f3_warm_start),
        f3_warm_start_k=int(args.f3_warm_start_k),
        f3_warm_start_max_clean=int(args.f3_warm_start_max_clean),
        warm_start_mode=str(args.warm_start_mode),
        gate_freeze_epochs=int(args.gate_freeze_epochs),
        skip_bootstrap=bool(args.skip_bootstrap),
        advantage_mode=str(args.advantage_mode),
        advantage_knn_bank_size=int(args.advantage_knn_bank_size),
        advantage_knn_chunk_size=int(args.advantage_knn_chunk_size),
        advantage_knn_k=int(args.advantage_knn_k),
        horizon=int(args.horizon),
        proprio_indices=(list(args.proprio_indices) if args.proprio_indices else None),
    )

    trainer = D4Trainer(
        predictor=predictor,
        encoder=encoder,
        dataset=dataset,
        config=trainer_cfg,
        device=str(args.device),
    )

    save_dir = os.path.abspath(str(args.save_dir))
    os.makedirs(save_dir, exist_ok=True)
    save_name = str(args.save_name).strip() or f"d4_dynamics_{_now_tag()}.pt"
    save_path = os.path.join(save_dir, save_name)
    # `save_freq` governs periodic snapshots; `save_path` is the final (or latest)
    # checkpoint written by the trainer.
    trainer.run(save_path=save_path, save_freq=int(args.save_freq))


if __name__ == "__main__":
    main()
