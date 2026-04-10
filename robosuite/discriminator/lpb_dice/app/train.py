from __future__ import annotations

import os
from typing import Any

import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from robosuite.discriminator.lpb_dice.app.pipeline import (
    build_backbone_training_datasets,
    build_flow_encoder,
    build_pu_trajectory_splits,
    now_tag,
)
from robosuite.discriminator.lpb_dice.core.dataset import build_cached_splits
from robosuite.discriminator.lpb_dice.core.model import (
    LatentDynamicsModel,
    build_latent_dynamics_predictor,
)
from robosuite.discriminator.lpb_dice.core.pu_detector import (
    DetectorConfig,
    PUHeadConfig,
    calibrate_pu_detector,
    train_pu_classifier,
)
from robosuite.discriminator.lpb_dice.core.representation import FrozenTransitionRepresentation
from robosuite.discriminator.lpb_dice.core.trainer import Trainer, TrainerConfig


def _optional_abs_path(value: Any) -> str | None:
    if value in {None, "", "None", "null"}:
        return None
    return to_absolute_path(str(value))


def _torch_load_checkpoint(path: str) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _extract_cfg_model(payload: dict[str, Any]) -> dict[str, Any]:
    if "cfg_model" in payload:
        return dict(payload["cfg_model"])
    cfg = payload.get("cfg", None)
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return dict(cfg.get("model", {}))
    try:
        return dict(cfg.model)
    except Exception:
        return {}


def _build_backbone_payload(
    *,
    model: LatentDynamicsModel,
    cfg: DictConfig,
    latent_dim: int,
    action_dim: int,
    source_ckpt: str | None,
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "latent_dim": int(latent_dim),
        "action_dim": int(action_dim),
        "horizon": int(cfg.data.transition_horizon),
        "cfg_model": {
            key: value
            for key, value in dict(cfg.model).items()
            if key != "backbone_ckpt"
        },
        "source_ckpt": source_ckpt,
    }


def _build_backbone_trainer(
    cfg: DictConfig,
    *,
    model: LatentDynamicsModel,
    train_dataset,
    val_dataset,
) -> Trainer:
    trainer_cfg = TrainerConfig(
        batch_size=int(cfg.training.batch_size),
        num_workers=int(cfg.training.num_workers),
        learning_rate=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
        epochs=int(cfg.training.epochs),
        expert_sampling_ratio=float(cfg.training.expert_ratio),
        grad_clip_norm=float(cfg.training.grad_clip_norm),
        log_every=int(cfg.training.log_every),
        device=str(cfg.training.device),
    )
    return Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=trainer_cfg,
        device=str(cfg.training.device),
    )


def _train_or_load_backbone(
    cfg: DictConfig,
    *,
    cached_splits,
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    backbone_ckpt = _optional_abs_path(getattr(cfg.model, "backbone_ckpt", None))
    if backbone_ckpt is not None:
        print(f"[lpb_dice] Reusing frozen backbone checkpoint: {backbone_ckpt}")
        payload = _torch_load_checkpoint(backbone_ckpt)
        if str(payload.get("format_version", "")) == "lpb_dice_v1" and "backbone" in payload:
            payload = payload["backbone"]
        if "model" not in payload:
            raise RuntimeError(f"Backbone checkpoint missing `model`: {backbone_ckpt}")
        return {
            "model": payload["model"],
            "latent_dim": int(payload["latent_dim"]),
            "action_dim": int(payload["action_dim"]),
            "horizon": int(payload.get("horizon", int(cfg.data.transition_horizon))),
            "cfg_model": _extract_cfg_model(payload),
            "source_ckpt": backbone_ckpt,
        }, {}

    print("[lpb_dice] No backbone checkpoint provided. Training latent dynamics backbone from scratch.")
    datasets = build_backbone_training_datasets(cfg=cfg, cached_splits=cached_splits)
    train_dataset = datasets.train_dataset
    eval_dataset = datasets.eval_dataset

    predictor = build_latent_dynamics_predictor(
        latent_dim=int(train_dataset.latent_dim),
        action_dim=int(train_dataset.action_dim),
        cfg_model={
            key: value
            for key, value in dict(cfg.model).items()
            if key != "backbone_ckpt"
        },
        transition_horizon=int(cfg.data.transition_horizon),
    )
    model = LatentDynamicsModel(predictor=predictor)
    trainer = _build_backbone_trainer(
        cfg=cfg,
        model=model,
        train_dataset=train_dataset,
        val_dataset=eval_dataset,
    )
    history = trainer.fit()
    payload = _build_backbone_payload(
        model=model,
        cfg=cfg,
        latent_dim=int(train_dataset.latent_dim),
        action_dim=int(train_dataset.action_dim),
        source_ckpt=None,
    )
    return payload, history


def _build_detector_configs(cfg: DictConfig) -> tuple[PUHeadConfig, DetectorConfig]:
    return (
        PUHeadConfig(
            hidden_dim=int(cfg.detector_training.hidden_dim),
            num_layers=int(cfg.detector_training.num_layers),
            dropout=float(cfg.detector_training.dropout),
            batch_size=int(cfg.detector_training.batch_size),
            num_workers=int(cfg.detector_training.num_workers),
            epochs=int(cfg.detector_training.epochs),
            learning_rate=float(cfg.detector_training.lr),
            weight_decay=float(cfg.detector_training.weight_decay),
            positive_sampling_ratio=float(cfg.detector_training.positive_ratio),
            grad_clip_norm=float(cfg.detector_training.grad_clip_norm),
            log_every=int(cfg.detector_training.log_every),
            device=str(cfg.detector_training.device),
        ),
        DetectorConfig(
            delta=float(cfg.detector.delta),
            lambda_mode=str(cfg.detector.lambda_mode),
            lambda_window_size=int(cfg.detector.lambda_window_size),
            support_penalty_weight=float(cfg.support_penalty.weight),
            c_min=float(cfg.detector.c_min),
            corrected_prob_eps=float(cfg.detector.corrected_prob_eps),
        ),
    )


def run_train(cfg: DictConfig) -> None:
    seed = int(cfg.seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    encoder = build_flow_encoder(cfg)
    try:
        cached_splits, split_summary, task_to_index = build_cached_splits(
            cfg_data=cfg.data,
            encoder=encoder,
            seed=seed,
        )
        backbone_payload, backbone_history = _train_or_load_backbone(
            cfg=cfg,
            cached_splits=cached_splits,
        )
        representation = FrozenTransitionRepresentation.from_backbone_payload(
            payload=backbone_payload,
            device=str(cfg.representation.device),
            batch_size=int(cfg.representation.batch_size),
            source_ckpt=backbone_payload.get("source_ckpt", None),
        )

        pu_splits = build_pu_trajectory_splits(cfg=cfg, cached_splits=cached_splits)
        if not pu_splits.train:
            raise RuntimeError("PU detector training requires non-empty train trajectories.")
        if not pu_splits.eval:
            raise RuntimeError("PU detector training requires non-empty eval trajectories.")
        if not pu_splits.calibration_positive:
            raise RuntimeError("Positive calibration trajectories cannot be empty.")

        print(
            f"[lpb_dice] pu_train_trajectories={len(pu_splits.train)} "
            f"pu_eval_trajectories={len(pu_splits.eval)} "
            f"calibration_positive={len(pu_splits.calibration_positive)} "
            f"calibration_background={len(pu_splits.calibration_background)}"
        )

        train_sequences = representation.encode_trajectories(list(pu_splits.train))
        eval_sequences = representation.encode_trajectories(list(pu_splits.eval))
        calibration_positive = representation.encode_trajectories(list(pu_splits.calibration_positive))
        calibration_background = representation.encode_trajectories(list(pu_splits.calibration_background))

        head_cfg, detector_cfg = _build_detector_configs(cfg)
        head, detector_history = train_pu_classifier(
            train_sequences=train_sequences,
            val_sequences=eval_sequences,
            cfg=head_cfg,
        )
        calibration = calibrate_pu_detector(
            model=head,
            positive_sequences=calibration_positive,
            background_sequences=calibration_background,
            cfg=detector_cfg,
            device=str(cfg.detector_training.device),
        )

        save_dir = to_absolute_path(str(cfg.save_dir))
        os.makedirs(save_dir, exist_ok=True)
        if str(cfg.save_name) in {"", "None", "null"}:
            save_name = f"lpb_dice_{now_tag()}.pt"
        else:
            save_name = str(cfg.save_name)
        save_path = os.path.join(save_dir, save_name)

        final_payload = {
            "format_version": "lpb_dice_v1",
            "cfg": cfg,
            "backbone": backbone_payload,
            "representation": {
                "batch_size": int(cfg.representation.batch_size),
                "device": str(cfg.representation.device),
                "feature_dim": int(representation.feature_dim),
            },
            "detector": {
                "head_state": head.state_dict(),
                "feature_dim": int(representation.feature_dim),
                "head_config": {
                    "hidden_dim": int(head_cfg.hidden_dim),
                    "num_layers": int(head_cfg.num_layers),
                    "dropout": float(head_cfg.dropout),
                },
                "calibration": calibration.to_payload(),
            },
            "history": {
                "backbone": backbone_history,
                "detector": detector_history,
            },
            "task_to_index": dict(task_to_index),
            "split_summary": split_summary,
            "policy_ckpt": to_absolute_path(str(cfg.policy.ckpt)),
            "image_size": int(cfg.data.image_size),
        }
        torch.save(final_payload, save_path)
        print(f"[lpb_dice] Saved combined checkpoint to: {save_path}")
    finally:
        encoder.close()
