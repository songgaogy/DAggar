from __future__ import annotations

import os
from datetime import datetime

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from robosuite.discriminator.dyn_bce.utils.dataset import (
    build_datasets,
    estimate_weighted_occupancy_positive_prior,
)
from robosuite.discriminator.dyn_bce.modules.flow_encoder import FrozenFlowMultitaskEncoder
from robosuite.discriminator.dyn_bce.modules.model import DynBCEModel
from robosuite.discriminator.dyn_bce.utils.trainer import DynBCETrainer, TrainerConfig


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _resolve_occupancy_positive_prior(cfg: DictConfig) -> float:
    prior_cfg = cfg.loss.occupancy_positive_prior
    if prior_cfg is None:
        return estimate_weighted_occupancy_positive_prior(
            fail_onset_ratio=float(cfg.labels.fail_onset_ratio),
            risk_temperature=float(cfg.labels.risk_temperature),
            occ_fail_prefix_min_weight=float(cfg.labels.occ_fail_prefix_min_weight),
        )
    return float(prior_cfg)


@hydra.main(version_base="1.2", config_path="./config", config_name="train")
def main(cfg: DictConfig) -> None:
    encoder = FrozenFlowMultitaskEncoder(
        checkpoint_path=str(cfg.policy.ckpt),
        device=str(cfg.policy.device),
        image_size=int(cfg.data.image_size),
        batch_size=int(cfg.policy.encoder_batch_size),
    )

    try:
        datasets, task_to_index, metadata = build_datasets(
            cfg_data=cfg.data,
            cfg_labels=cfg.labels,
            encoder=encoder,
            seed=int(cfg.seed),
        )
        print(f"dyn_bce task_to_index={task_to_index}")
        print(f"dyn_bce metadata={metadata}")
        occupancy_positive_prior = _resolve_occupancy_positive_prior(cfg)
        print(f"dyn_bce resolved_occupancy_positive_prior={occupancy_positive_prior:.4f}")

        model = DynBCEModel(
            latent_dim=int(encoder.latent_dim),
            action_dim=int(encoder.action_dim),
            num_tasks=int(len(task_to_index)),
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
        )
        
        trainer = DynBCETrainer(
            model=model,
            train_dataset=datasets["train"],
            val_dataset=datasets["eval"],
            test_dataset=datasets["test"],
            config=TrainerConfig(
                device=str(cfg.training.device),
                batch_size=int(cfg.training.batch_size),
                num_workers=int(cfg.training.num_workers),
                prefetch_factor=int(cfg.training.prefetch_factor),
                persistent_workers=bool(cfg.training.persistent_workers),
                epochs=int(cfg.training.epochs),
                learning_rate=float(cfg.training.lr),
                weight_decay=float(cfg.training.weight_decay),
                grad_clip_norm=float(cfg.training.grad_clip_norm),
                amp=bool(cfg.training.amp),
                ema_decay=float(cfg.training.ema_decay),
                log_every=int(cfg.training.log_every),
                save_freq=int(cfg.training.save_freq),
                alpha_dyn=float(cfg.loss.alpha_dyn),
                eta_decor=float(cfg.loss.eta_decor),
                xi_fuse=float(cfg.loss.xi_fuse),
                beta_nll=float(cfg.loss.beta_nll),
                occupancy_positive_prior=float(occupancy_positive_prior),
                occupancy_nnpu=bool(cfg.loss.occupancy_nnpu),
            ),
            metadata={
                **metadata,
                "policy_ckpt": to_absolute_path(str(cfg.policy.ckpt)),
                "image_size": int(cfg.data.image_size),
            },
            cfg=cfg,
        )

        save_dir = to_absolute_path(str(cfg.save_dir))
        os.makedirs(save_dir, exist_ok=True)
        if str(cfg.save_name) in {"", "None", "null"}:
            save_name = f"dyn_bce_{_now_tag()}.pt"
        else:
            save_name = str(cfg.save_name)
        trainer.fit(save_dir=save_dir, save_name=save_name)
    finally:
        encoder.close()


if __name__ == "__main__":
    main()
