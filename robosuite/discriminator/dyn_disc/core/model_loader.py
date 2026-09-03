from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf


def _resolved_node(node: Any) -> Any:
    """Resolve parent-scoped interpolations before detached instantiation."""
    return OmegaConf.create(OmegaConf.to_container(node, resolve=True))


def instantiate_local(node: Any, **kwargs):
    return hydra.utils.instantiate(_resolved_node(node), **kwargs)


def load_ckpt(snapshot_path: Path, device: torch.device):
    with snapshot_path.open("rb") as file:
        return torch.load(file, map_location=device)


def _view_names(train_cfg: DictConfig, field: str) -> list[str]:
    names = getattr(train_cfg, field, None)
    if names is None and field == "view_names":
        names = getattr(train_cfg.env, "view_names", None)
    if names is None:
        if field == "view_names":
            raise ValueError("Missing view_names in TACO training config.")
        names = _view_names(train_cfg, "view_names")
    return list(names)


def load_model(model_ckpt: Path, train_cfg: DictConfig, device: torch.device):
    """Rebuild a TACO representation model from its complete checkpoint state."""
    if not model_ckpt.exists():
        raise FileNotFoundError(f"TACO checkpoint not found: {model_ckpt}")
    if str(OmegaConf.select(train_cfg, "pretraining_method", default="")) != "taco":
        raise ValueError("Only TACO training configs are supported.")

    result = load_ckpt(model_ckpt, device)
    if result.get("pretraining_method") != "taco":
        raise ValueError("Only checkpoints with pretraining_method='taco' are supported.")
    if "model" not in result:
        raise ValueError("TACO model state not found in checkpoint.")

    policy_ckpt_path = getattr(train_cfg, "policy_ckpt_path", None)
    if policy_ckpt_path not in (None, "", "null", "None"):
        raise ValueError("TACO representation pretraining does not use a policy checkpoint.")
    if getattr(train_cfg, "encoder", None) is None:
        raise ValueError("TACO requires an explicit DINOv3 encoder config.")

    view_names = _view_names(train_cfg, "view_names")
    encoder = instantiate_local(train_cfg.encoder, view_names=view_names)
    prior_in_chans = int(getattr(train_cfg, "prior_in_chans", train_cfg.env.proprio_dim))
    action_dim = int(getattr(train_cfg, "action_dim_per_step", train_cfg.env.action_dim))
    proprio_emb_dim = int(
        OmegaConf.select(train_cfg, "proprio_emb_dim", default=train_cfg.env.proprio_emb_dim)
    )
    proprio_encoder = instantiate_local(
        train_cfg.proprio_encoder,
        in_chans=prior_in_chans,
        emb_dim=proprio_emb_dim,
    )
    model = instantiate_local(
        train_cfg.model,
        encoder=encoder,
        proprio_encoder=proprio_encoder,
        proprio_dim=proprio_emb_dim,
        action_dim_per_step=action_dim,
        frameskip=train_cfg.frameskip,
        view_names=view_names,
        source_view_names=_view_names(train_cfg, "source_view_names"),
        target_view_names=_view_names(train_cfg, "target_view_names"),
    )
    model.load_state_dict(result["model"])
    print(
        f"Loaded TACO representation model from epoch "
        f"{result.get('epoch', 'unknown')}: {model_ckpt}"
    )
    return model.to(device)
