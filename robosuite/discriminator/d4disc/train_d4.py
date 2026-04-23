"""CLI entry point for training the D4-Disc conditional dynamics critic.

Example (from repo root):
    python -m robosuite.discriminator.d4disc.train_d4 \
        --policy-ckpt checkpoints/multitask_6/policy/flow-20/flow_multi_ep0100_*.pt \
        --expert-paths data/PickPlaceBread/expert \
        --rollout-paths data/PickPlaceBread/success_rollout \
        --fail-paths data/PickPlaceBread/fail_rollout \
        --warm-up-epochs 8 --bootstrap-epochs 40 \
        --save-dir checkpoints/d4disc/dynamics

Prefer ``scripts/train_d4.sh`` for the canonical entry.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

import torch

from robosuite.discriminator.d3disc.encoder import FlowMultiEncoderWrapper

from .dataset import LatentFlowDynamicsDatasetD4
from .model import ConditionalDynamicsPredictor
from .schedule import D4Schedule
from .trainer import D4Trainer, D4TrainerConfig


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--policy-ckpt", required=True)
    p.add_argument("--cache-root", type=str, default="data/.lpb_score_cache")
    p.add_argument("--expert-paths", nargs="*", default=[])
    p.add_argument("--rollout-paths", nargs="*", default=[])
    p.add_argument("--fail-paths", nargs="*", default=[])
    p.add_argument("--proprio-indices", type=int, nargs="*", default=None)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--max-trajectories-per-kind", type=int, default=0)

    # Architecture
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max-action-horizon", type=int, default=32)
    p.add_argument("--d-cond", type=int, default=64)
    p.add_argument("--adaln-init-std", type=float, default=0.0,
                   help="Std of Gaussian init for AdaLN modulation heads. 0 => DiT "
                        "AdaLN-Zero (identity at step 0, safe but slow to grow c=- "
                        "branch). >0 (e.g. 0.02) immediately breaks f(+)=f(-) "
                        "degeneracy — use if gamma stays stuck at 0.5 despite warm-start.")

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

    # CFG
    p.add_argument("--p-cond-drop", type=float, default=0.1)
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

    # F3-based gamma warm-start (breaks gamma=0.5 symmetric fixed point)
    wm = p.add_mutually_exclusive_group()
    wm.add_argument("--f3-warm-start", dest="f3_warm_start", action="store_true")
    wm.add_argument("--no-f3-warm-start", dest="f3_warm_start", action="store_false")
    p.set_defaults(f3_warm_start=True)
    p.add_argument("--f3-warm-start-k", type=int, default=1)
    p.add_argument("--f3-warm-start-max-clean", type=int, default=100_000)
    p.add_argument("--warm-start-mode", type=str, default="rank",
                   choices=["rank", "sigmoid"],
                   help="Warm-start γ calibration: 'rank' (default, uniform "
                        "quantile) or 'sigmoid' (auto β/κ on fail→clean d²). "
                        "The old D3 sigmoid-on-clean-self calibration saturated "
                        "every sample at γ=0.999 and was removed.")
    p.add_argument("--gate-freeze-epochs", type=int, default=2,
                   help="Skip advantage-gate updates for the first N bootstrap "
                        "epochs after warm-start (lets f(-) differentiate).")
    p.add_argument("--skip-bootstrap", action="store_true",
                   help="Skip Phase B entirely: train f(+) in Phase A, warm-start "
                        "gamma, save checkpoint, exit. Use with OMEGA=0 at "
                        "benchmark time for the pos-branch-only baseline.")

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

    dataset = LatentFlowDynamicsDatasetD4(
        encoder=encoder,
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
    )
    print(
        f"[d4_train] dataset size={len(dataset)}  "
        f"expert_samples={dataset.num_expert_samples}  "
        f"rollout_samples={dataset.num_rollout_samples}  "
        f"fail_raw_samples={dataset.num_fail_raw_samples}  "
        f"action_dim={dataset.action_dim}  proprio_dim={dataset.proprio_dim}  "
        f"latent_dim={dataset.latent_dim}"
    )

    if int(args.num_workers) > 0 and dataset.encoder.device.type == "cuda":
        n_new = dataset.materialize_missing_latent_caches()
        if n_new > 0:
            print(f"[d4_train] materialized {n_new} missing latent cache(s)")

    # Preload per-demo states/actions into RAM so __getitem__ never opens
    # hdf5 (was the dominant per-sample cost — 1M+ opens/epoch starved the GPU).
    n_demos = dataset.preload_states_actions()
    print(f"[d4_train] preloaded states/actions for {n_demos} demo(s)")

    predictor = ConditionalDynamicsPredictor(
        latent_dim=dataset.latent_dim,
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

    schedule = D4Schedule(
        alpha_exponent=float(args.alpha_exponent),
        alpha_cap_rate=float(args.alpha_cap_rate),
        eta_final=float(args.eta_final),
        eta_ramp_epochs=int(args.eta_ramp_epochs),
    )

    trainer_cfg_kwargs = dict(
        f3_warm_start=bool(args.f3_warm_start),
        f3_warm_start_k=int(args.f3_warm_start_k),
        f3_warm_start_max_clean=int(args.f3_warm_start_max_clean),
        warm_start_mode=str(args.warm_start_mode),
        gate_freeze_epochs=int(args.gate_freeze_epochs),
        skip_bootstrap=bool(args.skip_bootstrap),
    )
    trainer_cfg = D4TrainerConfig(
        warm_up_epochs=int(args.warm_up_epochs),
        bootstrap_epochs=int(args.bootstrap_epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        grad_clip_norm=float(args.grad_clip_norm),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        p_cond_drop=float(args.p_cond_drop),
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
        **trainer_cfg_kwargs,
    )

    trainer = D4Trainer(
        predictor=predictor,
        dataset=dataset,
        config=trainer_cfg,
        device=str(args.device),
    )

    save_dir = os.path.abspath(str(args.save_dir))
    os.makedirs(save_dir, exist_ok=True)
    save_name = str(args.save_name).strip() or f"d4_dynamics_{_now_tag()}.pt"
    save_path = os.path.join(save_dir, save_name)

    trainer.run(save_path=save_path, save_freq=int(args.save_freq))


if __name__ == "__main__":
    main()
