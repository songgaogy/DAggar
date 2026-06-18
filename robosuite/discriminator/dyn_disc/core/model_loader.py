from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from robosuite.discriminator.dyn_disc.models.resnet_encoder import ResNetEncoder

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


_TARGET_ALIASES = {
    "dyn_model.models.proprio.ProprioceptiveEmbedding":
        "robosuite.discriminator.dyn_disc.models.proprio.ProprioceptiveEmbedding",
    "dyn_model.models.vit.ViTPredictor":
        "robosuite.discriminator.dyn_disc.models.vit.ViTPredictor",
    "dyn_model.models.visual_dyn_model.VisualDynamicsModel":
        "robosuite.discriminator.dyn_disc.models.visual_dynamics.VisualDynamicsModel",
    "robosuite.discriminator.lpb_original.dyn_model.models.proprio.ProprioceptiveEmbedding":
        "robosuite.discriminator.dyn_disc.models.proprio.ProprioceptiveEmbedding",
    "robosuite.discriminator.lpb_original.dyn_model.models.vit.ViTPredictor":
        "robosuite.discriminator.dyn_disc.models.vit.ViTPredictor",
    "robosuite.discriminator.lpb_original.dyn_model.models.visual_dyn_model.VisualDynamicsModel":
        "robosuite.discriminator.dyn_disc.models.visual_dynamics.VisualDynamicsModel",
}


def _retarget(node: Any) -> Any:
    # Checkpoints store Hydra configs with `_target_` strings that may refer to
    # legacy module paths (lpb_original / dyn_model). Retarget them to the local
    # dyn_disc implementations so we can instantiate modules without editing old configs.
    # IMPORTANT: `instantiate_local()` is often called on sub-nodes (e.g., cfg.model)
    # that contain interpolations like `${img_size}` pointing to keys in the *parent*
    # config. Once we detach the node into a standalone config, those interpolations
    # would become unresolved and crash instantiation. Resolve them first.
    cfg = OmegaConf.create(OmegaConf.to_container(node, resolve=True))
    target = OmegaConf.select(cfg, "_target_", default=None)
    if target in _TARGET_ALIASES:
        cfg._target_ = _TARGET_ALIASES[str(target)]
    return cfg


def instantiate_local(node: Any, **kwargs):
    return hydra.utils.instantiate(_retarget(node), **kwargs)


def load_ckpt(snapshot_path: Path, device: torch.device):
    with snapshot_path.open("rb") as f:
        return torch.load(f, map_location=device)


def _get_view_names(train_cfg: DictConfig):
    view_names = getattr(train_cfg, "view_names", None)
    if view_names is None and getattr(train_cfg, "env", None) is not None:
        view_names = getattr(train_cfg.env, "view_names", None)
    if view_names is None:
        raise ValueError("Missing view_names in training config")
    return list(view_names)


def _get_train_path(train_cfg: DictConfig) -> str:
    path = getattr(train_cfg, "train_data_path", None)
    if path is None and getattr(train_cfg, "env", None) is not None:
        path = getattr(train_cfg.env, "train_data_path", None)
    return "" if path is None else str(path)


def load_model(model_ckpt: Path, train_cfg: DictConfig, device: torch.device):
    result = {}
    if model_ckpt.exists():
        result = load_ckpt(model_ckpt, device)
        print("result keys in load_model:", result.keys())
        print(f"Resuming from epoch {result['epoch']}: {model_ckpt}")

    policy_ckpt_path = getattr(train_cfg, "policy_ckpt_path", None)
    if policy_ckpt_path not in (None, "", "null", "None"):
        raise ValueError(
            "dyn_disc does not support diffusion-policy policy_ckpt_path. "
            "Set policy_ckpt_path/env.policy_ckpt_path to null, or use lpb_original."
        )

    view_names = _get_view_names(train_cfg)
    encoder = ResNetEncoder(policy_ckpt_path=None, view_names=view_names)

    if "encoder" in result:
        encoder.load_state_dict(result["encoder"])
        print(f"loaded encoder from checkpoint {model_ckpt}")
    elif not train_cfg.model.train_encoder:
        print("using pretrained encoder")
    else:
        raise ValueError("Encoder not found in model checkpoint")

    # Infer input dimensions for the embedding modules.
    #
    # Priority:
    #   1) Explicit dims written by `dyn_disc/train.py` (prior_in_chans, action_dim_per_step)
    #   2) env.{proprio_dim, action_dim} from the saved Hydra config
    #   3) Legacy heuristics based on train_data_path (transport/pusht) and defaults
    action_dim = 10 if train_cfg.abs_action else 7
    prior_in_chans = 9
    train_data_path = _get_train_path(train_cfg)
    if "transport" in train_data_path:
        action_dim = 20
        prior_in_chans = 18
    elif "pusht" in train_data_path:
        action_dim = 2
        prior_in_chans = 2
    elif "libero" in train_data_path:
        raise ValueError("dyn_disc does not support legacy libero language checkpoints.")

    if getattr(train_cfg, "env", None) is not None:
        if getattr(train_cfg.env, "proprio_dim", None) is not None:
            prior_in_chans = int(train_cfg.env.proprio_dim)
        if getattr(train_cfg.env, "action_dim", None) is not None:
            action_dim = int(train_cfg.env.action_dim)
    if getattr(train_cfg, "prior_in_chans", None) is not None:
        prior_in_chans = int(train_cfg.prior_in_chans)
    if getattr(train_cfg, "action_dim_per_step", None) is not None:
        action_dim = int(train_cfg.action_dim_per_step)
    # LPB dynamics conditions on a flattened action window of length `frameskip`.
    total_action_dim = action_dim * train_cfg.frameskip

    action_encoder = instantiate_local(
        train_cfg.action_encoder,
        in_chans=total_action_dim,
        emb_dim=train_cfg.action_emb_dim,
    )
    if "action_encoder" in result:
        action_encoder.load_state_dict(result["action_encoder"])
        print(f"loaded action encoder from checkpoint {model_ckpt}")
    else:
        raise ValueError("Action encoder not found in model checkpoint")

    proprio_encoder = instantiate_local(
        train_cfg.proprio_encoder,
        in_chans=prior_in_chans,
        emb_dim=train_cfg.proprio_emb_dim,
    )
    if "proprio_encoder" in result:
        proprio_encoder.load_state_dict(result["proprio_encoder"])
        print(f"loaded proprio encoder from checkpoint {model_ckpt}")
    else:
        raise ValueError("Proprio encoder not found in model checkpoint")

    predictor = instantiate_local(
        train_cfg.predictor,
        num_patches=1,
        num_frames=train_cfg.num_hist,
        dim=encoder.emb_dim * len(view_names)
        + (proprio_encoder.emb_dim + action_encoder.emb_dim),
        visual_dim=encoder.emb_dim * len(view_names),
        proprio_dim=train_cfg.proprio_emb_dim,
        action_dim=train_cfg.action_emb_dim,
    )
    if "predictor" in result:
        predictor.load_state_dict(result["predictor"])
        print(f"loaded predictor from checkpoint {model_ckpt}")
    else:
        raise ValueError("Predictor not found in model checkpoint")

    model = instantiate_local(
        train_cfg.model,
        encoder=encoder,
        proprio_encoder=proprio_encoder,
        action_encoder=action_encoder,
        predictor=predictor,
        proprio_dim=train_cfg.proprio_emb_dim,
        action_dim=train_cfg.action_emb_dim,
        view_names=view_names,
        use_layernorm=train_cfg.use_layernorm,
        language_encoder=None,
    )
    if train_cfg.has_predictor:
        if hasattr(model, "per_view_norm") and "per_view_norm" in result:
            model.per_view_norm.load_state_dict(result["per_view_norm"])
            print(f"loaded per_view_norm from checkpoint {model_ckpt}")
        if hasattr(model, "fusion_norm") and "fusion_norm" in result:
            model.fusion_norm.load_state_dict(result["fusion_norm"])
            print(f"loaded fusion_norm from checkpoint {model_ckpt}")

    model.to(device)
    return model
