"""Single-GPU Hydra training entry for the cleaned LPB v2 dynamics model.

This is a stripped-down rewrite of `dyn_model/train.py` that:
  * reads HDF5 / preprocessed data via robosuite.discriminator.lpb_v2.data
  * skips Accelerate (single GPU torch loop), keeping the dependency surface small
  * keeps the original LPB model architecture (ResNetEncoder + hydra-instantiated
    proprio/action encoders + ViT predictor + VisualDynamicsModel), so checkpoints
    are loadable by `robosuite.discriminator.lpb_v2.model_loader.load_model`.
  * saves <run_dir>/{checkpoints/model_<epoch>.pth, hydra.yaml, normalizer.pth}
    in the layout the discriminator expects.

Run from repo root:
    bash robosuite/discriminator/lpb_v2/scripts/train_lpb_v2_dynamics.sh
"""

from __future__ import annotations

import logging
import os
import random
import warnings
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader

from robosuite.discriminator.lpb_v2.model_loader import instantiate_local
from robosuite.discriminator.lpb_v2.models.resnet_encoder import ResNetEncoder

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _instantiate_dataset(cfg: DictConfig, train: bool):
    """Hydra-instantiate the dataset class named in cfg.env.dataset_class."""
    DatasetCls = hydra.utils.get_class(cfg.env.dataset_class)
    # Dataset contract (shared by HDF5 + preprocessed-cache backends):
    #   __getitem__ -> (obs, act, state)
    #     obs["visual"][view]: (F, 3, H, W) float in [0, 1]
    #     obs["proprio"]:      (F, P) float
    #     act:                (F * frameskip, A) float
    # where F = num_hist + num_pred.
    kwargs = dict(
        zarr_path=cfg.env.train_data_path if train else cfg.env.val_data_path,
        num_hist=cfg.num_hist,
        num_pred=cfg.num_pred,
        frameskip=cfg.frameskip,
        view_names=list(cfg.env.view_names),
        abs_action=cfg.abs_action,
        use_crop=cfg.use_crop,
        train=train,
        original_img_size=cfg.env.original_img_size,
        cropped_img_size=cfg.env.cropped_img_size,
        action_dim=cfg.env.action_dim,
    )
    if "use_cache" in cfg.env:
        kwargs["use_cache"] = bool(cfg.env.use_cache)
    if "cache_dir" in cfg.env and cfg.env.cache_dir is not None:
        kwargs["cache_dir"] = hydra.utils.to_absolute_path(str(cfg.env.cache_dir))
    if "shape_obs" in cfg.env and cfg.env.shape_obs is not None:
        kwargs["shape_obs"] = OmegaConf.to_container(cfg.env.shape_obs, resolve=True)
    # PreprocessedCacheDynamicsModelDataset-specific params
    if "cache_root" in cfg.env and cfg.env.cache_root is not None:
        kwargs["cache_root"] = hydra.utils.to_absolute_path(str(cfg.env.cache_root))
    if "tasks" in cfg.env and cfg.env.tasks is not None:
        kwargs["tasks"] = list(cfg.env.tasks)
    if "train_sources" in cfg.env and cfg.env.train_sources is not None:
        kwargs["train_sources"] = list(cfg.env.train_sources)
    if "camera_to_view" in cfg.env and cfg.env.camera_to_view is not None:
        kwargs["camera_to_view"] = OmegaConf.to_container(cfg.env.camera_to_view, resolve=True)
    if "max_cached_episodes" in cfg.env and cfg.env.max_cached_episodes is not None:
        kwargs["max_cached_episodes"] = int(cfg.env.max_cached_episodes)
    if "load_all_into_ram" in cfg.env:
        kwargs["load_all_into_ram"] = bool(cfg.env.load_all_into_ram)
    if "metadata_cache_root" in cfg.env and cfg.env.metadata_cache_root is not None:
        kwargs["metadata_cache_root"] = hydra.utils.to_absolute_path(str(cfg.env.metadata_cache_root))
    if "num_expert" in cfg.env:
        kwargs["num_expert"] = int(cfg.env.num_expert)
    if "num_success" in cfg.env:
        kwargs["num_success"] = int(cfg.env.num_success)
    if "num_fail" in cfg.env:
        kwargs["num_fail"] = int(cfg.env.num_fail)
    if cfg.get("proprio_indices", None):
        kwargs["proprio_indices"] = list(cfg.proprio_indices)
    if cfg.get("max_trajectories", None):
        kwargs["max_trajectories"] = int(cfg.max_trajectories)
    return DatasetCls(**kwargs)


def _build_model(cfg: DictConfig, dataset, device: torch.device):
    """Replicates the original LPB model construction with local v2 modules."""
    if cfg.policy_ckpt_path not in (None, "", "null", "None"):
        raise ValueError(
            "lpb_v2 does not support diffusion-policy policy_ckpt_path. "
            "Set env.policy_ckpt_path=null or use lpb_original."
        )
    encoder = ResNetEncoder(policy_ckpt_path=None, view_names=list(cfg.view_names))
    if cfg.encoder_ckpt_path:
        ckpt = torch.load(cfg.encoder_ckpt_path, map_location=device)
        if "encoder" in ckpt:
            encoder.load_state_dict(ckpt["encoder"])
            log.info(f"Loaded encoder weights from {cfg.encoder_ckpt_path}")
    if cfg.use_pretrained_encoder:
        for p in encoder.parameters():
            p.requires_grad = False

    proprio_encoder = instantiate_local(
        cfg.proprio_encoder,
        in_chans=dataset.proprio_dim,
        emb_dim=cfg.env.proprio_emb_dim,
    )
    # The training dataset returns per-step actions, but the LPB dynamics model consumes
    # a flattened action sequence of length frameskip for each observed frame.
    action_encoder = instantiate_local(
        cfg.action_encoder,
        in_chans=cfg.env.action_dim * cfg.frameskip,
        emb_dim=cfg.env.action_emb_dim,
    )

    visual_dim = encoder.emb_dim * len(cfg.view_names)
    predictor = instantiate_local(
        cfg.predictor,
        num_patches=1,
        num_frames=cfg.num_hist,
        dim=visual_dim + (proprio_encoder.emb_dim + action_encoder.emb_dim),
        visual_dim=visual_dim,
        proprio_dim=cfg.env.proprio_emb_dim,
        action_dim=cfg.env.action_emb_dim,
    )

    if cfg.predictor_ckpt_path:
        ckpt = torch.load(cfg.predictor_ckpt_path, map_location=device)
        parts = {
            "predictor": predictor,
            "proprio_encoder": proprio_encoder,
            "action_encoder": action_encoder,
        }
        for k, mod in parts.items():
            if k in ckpt:
                mod.load_state_dict(ckpt[k])
                log.info(f"Loaded {k} weights from {cfg.predictor_ckpt_path}")

    model = instantiate_local(
        cfg.model,
        encoder=encoder,
        proprio_encoder=proprio_encoder,
        action_encoder=action_encoder,
        predictor=predictor,
        proprio_dim=cfg.env.proprio_emb_dim,
        action_dim=cfg.env.action_emb_dim,
        view_names=list(cfg.view_names),
        use_layernorm=cfg.use_layernorm,
        language_encoder=None,
        action_loss_weight=OmegaConf.select(cfg, "action_loss_weight", default=0.0),
    )
    return model.to(device), encoder, proprio_encoder, action_encoder, predictor


def _normalize_batch(batch, normalizer, view_names, device):
    obs, act, _state = batch
    visual = obs["visual"]
    for v in view_names:
        x = visual[v].to(device)
        # x: (B, num_frames, 3, H, W)
        B, F, C, H, W = x.shape
        x_flat = x.view(B * F, C, H, W)
        x_flat = normalizer[v].normalize(x_flat)
        visual[v] = x_flat.view(B, F, C, H, W)
    proprio = obs["proprio"].to(device)
    proprio = normalizer["state"].normalize(proprio)
    act = act.to(device)
    act = normalizer["act"].normalize(act)
    return {"visual": visual, "proprio": proprio}, act


def _save_ckpt(out_dir: Path, epoch: int, parts: dict) -> Path:
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    payload = {"epoch": epoch}
    for k, v in parts.items():
        if isinstance(v, nn.Module):
            payload[k] = v.state_dict()
        else:
            payload[k] = v
    fp = ckpt_dir / f"model_{epoch}.pth"
    torch.save(payload, fp)
    return fp


def _accumulate_loss_components(sums: dict, components: dict) -> None:
    for key, value in components.items():
        if key == "loss":
            continue
        if torch.is_tensor(value):
            sums[key] = sums.get(key, 0.0) + float(value.detach().item())
        elif isinstance(value, (int, float)):
            sums[key] = sums.get(key, 0.0) + float(value)


def _format_loss_components(prefix: str, sums: dict, n_batches: int) -> str:
    keys = ("z_loss", "z_visual_loss", "z_proprio_loss", "z_action_loss")
    parts = []
    denom = max(1, int(n_batches))
    for key in keys:
        if key in sums:
            parts.append(f"{prefix}_{key}={sums[key] / denom:.5f}")
    return " ".join(parts)


@hydra.main(config_path="config", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    _seed_all(int(cfg.training.seed))

    out_dir = Path(os.getcwd())
    log.info(f"Output dir: {out_dir}")

    # Hydra @main does not write hydra.yaml by default; do it ourselves so the
    # discriminator can pick the saved config up.
    train_ds = _instantiate_dataset(cfg, train=True)
    valid_ds = _instantiate_dataset(cfg, train=False)
    log.info(
        f"train: {len(train_ds)} samples, valid: {len(valid_ds)} samples; "
        f"proprio_dim={train_ds.proprio_dim}, action_dim={train_ds.action_dim}"
    )

    with open_dict(cfg):
        # Preserve legacy aliases used by older configs / downstream tooling.
        # Some env configs (e.g. `env=preprocessed`) don't define HDF5 paths.
        train_alias = OmegaConf.select(cfg, "env.train_data_path", default=None)
        if train_alias is None:
            train_alias = OmegaConf.select(cfg, "env.cache_root", default=None)
        cfg.train_data_path = train_alias

        val_alias = OmegaConf.select(cfg, "env.val_data_path", default=None)
        if val_alias is None:
            val_alias = train_alias
        cfg.val_data_path = val_alias

        # Save the actual proprio/action input dims so `lpb_v2.model_loader.load_model`
        # can reconstruct the encoders deterministically.
        cfg.prior_in_chans = int(train_ds.proprio_dim)
        cfg.action_dim_per_step = int(cfg.env.action_dim)
    OmegaConf.save(cfg, out_dir / "hydra.yaml", resolve=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    normalizer = train_ds.get_normalizer().to(device)
    # Persist normalizer alongside the checkpoint so benchmark/KNN encoding matches training.
    torch.save(normalizer.state_dict(), out_dir / "normalizer.pth")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size, shuffle=True,
        num_workers=cfg.training.num_workers, drop_last=True,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=cfg.training.batch_size, shuffle=False,
        num_workers=cfg.training.num_workers, drop_last=False,
    )

    model, encoder, proprio_encoder, action_encoder, predictor = _build_model(cfg, train_ds, device)

    optim_groups = []
    if cfg.model.train_encoder:
        optim_groups.append({"params": [p for p in encoder.parameters() if p.requires_grad],
                             "lr": cfg.training.encoder_lr})
    if cfg.model.train_predictor:
        optim_groups.append({"params": predictor.parameters(), "lr": cfg.training.predictor_lr})
    optim_groups.append({"params": list(proprio_encoder.parameters()) + list(action_encoder.parameters()),
                         "lr": cfg.training.action_encoder_lr})
    optimizer = torch.optim.AdamW([g for g in optim_groups if any(True for _ in g["params"])])

    save_every = max(1, int(cfg.training.save_every_x_epoch))
    view_names = list(cfg.view_names)

    for epoch in range(int(cfg.training.epochs)):
        model.train()
        epoch_loss = 0.0
        train_comp_sums = {}
        n_batches = 0
        for batch in train_loader:
            obs, act = _normalize_batch(batch, normalizer, view_names, device)
            optimizer.zero_grad(set_to_none=True)
            loss, comp = model(obs, act)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            _accumulate_loss_components(train_comp_sums, comp)
            n_batches += 1
        avg_train = epoch_loss / max(1, n_batches)

        # Validation pass.
        model.eval()
        val_loss = 0.0
        val_comp_sums = {}
        n_val = 0
        with torch.no_grad():
            for batch in valid_loader:
                obs, act = _normalize_batch(batch, normalizer, view_names, device)
                loss, comp = model(obs, act)
                val_loss += float(loss.item())
                _accumulate_loss_components(val_comp_sums, comp)
                n_val += 1
        avg_val = val_loss / max(1, n_val)
        train_comp_msg = _format_loss_components("train", train_comp_sums, n_batches)
        val_comp_msg = _format_loss_components("val", val_comp_sums, n_val)
        log.info(
            f"epoch={epoch} train_loss={avg_train:.5f} val_loss={avg_val:.5f} "
            f"{train_comp_msg} {val_comp_msg}"
        )

        if (epoch + 1) % save_every == 0 or epoch == int(cfg.training.epochs) - 1:
            parts = {
                "encoder": encoder,
                "predictor": predictor,
                "proprio_encoder": proprio_encoder,
                "action_encoder": action_encoder,
            }
            if hasattr(model, "per_view_norm"):
                parts["per_view_norm"] = model.per_view_norm
            if hasattr(model, "fusion_norm"):
                parts["fusion_norm"] = model.fusion_norm
            fp = _save_ckpt(out_dir, epoch, parts)
            log.info(f"saved checkpoint to {fp}")


if __name__ == "__main__":
    main()
