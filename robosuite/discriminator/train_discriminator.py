import datetime
import itertools
import os
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
import hydra

from robosuite.discriminator.data import StateActionHdf5Dataset, scan_demos, split_demo_infos
from robosuite.discriminator.model import (
    BCEVisitationDiscriminator,
    PolicyConditionEncoder,
    build_policy_backbone_from_ckpt,
)
from robosuite.policy.utils.env_util import PandaLiftProprioExtractor


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_numpy(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _normalize_images(images: torch.Tensor, normalizer: Normalize) -> torch.Tensor:
    b, k, c, h, w = images.shape
    flat = images.view(b * k, c, h, w)
    return normalizer(flat).view(b, k, c, h, w)


def _make_loader(ds, batch_size: int, workers: int, shuffle: bool):
    kwargs = dict(
        dataset=ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        persistent_workers=bool(workers > 0),
        pin_memory=True,
        drop_last=shuffle,
    )
    if workers > 0:
        kwargs["prefetch_factor"] = 4
    return DataLoader(**kwargs)


def _cycle_loader(loader: DataLoader) -> Iterator[dict[str, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def evaluate(
    model: BCEVisitationDiscriminator,
    pos_loader: DataLoader,
    neg_loader: DataLoader,
    device: torch.device,
    img_normalize: Normalize,
):
    model.eval()

    pos_logits_all = []
    neg_logits_all = []

    for batch in pos_loader:
        images = _normalize_images(batch["images"].to(device, non_blocking=True), img_normalize)
        proprio = batch["proprio"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        pos_logits_all.append(model(images=images, proprio=proprio, actions=actions))

    for batch in neg_loader:
        images = _normalize_images(batch["images"].to(device, non_blocking=True), img_normalize)
        proprio = batch["proprio"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        neg_logits_all.append(model(images=images, proprio=proprio, actions=actions))

    if not pos_logits_all or not neg_logits_all:
        return {
            "loss": float("nan"),
            "objective": float("nan"),
            "acc": float("nan"),
            "acc_pos": float("nan"),
            "acc_neg": float("nan"),
        }

    pos_logits = torch.cat(pos_logits_all, dim=0)
    neg_logits = torch.cat(neg_logits_all, dim=0)

    objective = F.logsigmoid(pos_logits).mean() + F.logsigmoid(-neg_logits).mean()
    loss = -objective

    acc_pos = (torch.sigmoid(pos_logits) >= 0.5).float().mean()
    acc_neg = (torch.sigmoid(neg_logits) < 0.5).float().mean()
    acc = 0.5 * (acc_pos + acc_neg)

    return {
        "loss": float(loss.item()),
        "objective": float(objective.item()),
        "acc": float(acc.item()),
        "acc_pos": float(acc_pos.item()),
        "acc_neg": float(acc_neg.item()),
    }


@hydra.main(version_base="1.2", config_path="./config", config_name="train_discriminator")
def main(cfg: DictConfig):
    set_seed(int(cfg.seed))

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    expert_dir = to_absolute_path(cfg.data.expert_dir)
    unlabeled_dir = to_absolute_path(cfg.data.unlabeled_dir)
    policy_ckpt = to_absolute_path(cfg.policy_ckpt)
    save_dir = to_absolute_path(cfg.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    backbone_build = build_policy_backbone_from_ckpt(
        ckpt_path=policy_ckpt,
        history_len=int(cfg.history_len),
        freeze_backbone=bool(cfg.train.freeze_policy_backbone),
        device=str(device),
    )

    history_len = int(backbone_build.history_len)
    print(f"resolved history_len={history_len} loaded_backbone_params={backbone_build.loaded_params}")

    policy_blob = torch.load(policy_ckpt, map_location="cpu", weights_only=False)
    act_mean = _to_numpy(policy_blob.get("act_mean"))
    act_std = _to_numpy(policy_blob.get("act_std"))
    prop_mean = _to_numpy(policy_blob.get("prop_mean"))
    prop_std = _to_numpy(policy_blob.get("prop_std"))

    if prop_mean is not None and prop_mean.shape[0] != backbone_build.proprio_in_dim:
        prop_mean = None
        prop_std = None
        print("policy proprio stats shape mismatch; disabled proprio normalization")

    extractor = PandaLiftProprioExtractor(
        robots="Panda",
        env_name="Lift",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        camera_names=None,
        reward_shaping=False,
    )

    unlabled_size = int(getattr(cfg.data, "unlabled_size", -1))
    if hasattr(cfg.data, "unlabeled_size") and int(cfg.data.unlabeled_size) > 0:
        unlabled_size = int(cfg.data.unlabeled_size)

    expert_infos = scan_demos(
        data_dir=expert_dir,
        camera_name=cfg.camera,
        history_len=history_len,
        max_trajectories=None,
    )
    unlabeled_infos = scan_demos(
        data_dir=unlabeled_dir,
        camera_name=cfg.camera,
        history_len=history_len,
        max_trajectories=(unlabled_size if unlabled_size > 0 else None),
    )

    pos_train_infos, pos_eval_infos = split_demo_infos(expert_infos, eval_ratio=cfg.eval.eval_ratio, seed=cfg.seed)
    neg_train_infos, neg_eval_infos = split_demo_infos(unlabeled_infos, eval_ratio=cfg.eval.eval_ratio, seed=cfg.seed + 1)

    print(
        f"demos expert total/train/eval={len(expert_infos)}/{len(pos_train_infos)}/{len(pos_eval_infos)} | "
        f"unlabeled total/train/eval={len(unlabeled_infos)}/{len(neg_train_infos)}/{len(neg_eval_infos)}"
    )

    normalize_with_policy_stats = bool(cfg.data.normalize_with_policy_stats)

    pos_train_ds = StateActionHdf5Dataset(
        demo_infos=pos_train_infos,
        proprio_extractor=extractor.extract,
        camera_name=cfg.camera,
        history_len=history_len,
        image_size=cfg.image_size,
        label=1.0,
        normalize=normalize_with_policy_stats,
        prop_mean=prop_mean,
        prop_std=prop_std,
        act_mean=act_mean,
        act_std=act_std,
    )
    neg_train_ds = StateActionHdf5Dataset(
        demo_infos=neg_train_infos,
        proprio_extractor=extractor.extract,
        camera_name=cfg.camera,
        history_len=history_len,
        image_size=cfg.image_size,
        label=0.0,
        normalize=normalize_with_policy_stats,
        prop_mean=prop_mean,
        prop_std=prop_std,
        act_mean=act_mean,
        act_std=act_std,
    )

    if len(pos_train_ds) == 0 or len(neg_train_ds) == 0:
        raise RuntimeError("Insufficient train samples in expert or unlabeled dataset")

    pos_eval_ds = StateActionHdf5Dataset(
        demo_infos=pos_eval_infos,
        proprio_extractor=extractor.extract,
        camera_name=cfg.camera,
        history_len=history_len,
        image_size=cfg.image_size,
        label=1.0,
        normalize=normalize_with_policy_stats,
        prop_mean=prop_mean,
        prop_std=prop_std,
        act_mean=act_mean,
        act_std=act_std,
    )
    neg_eval_ds = StateActionHdf5Dataset(
        demo_infos=neg_eval_infos,
        proprio_extractor=extractor.extract,
        camera_name=cfg.camera,
        history_len=history_len,
        image_size=cfg.image_size,
        label=0.0,
        normalize=normalize_with_policy_stats,
        prop_mean=prop_mean,
        prop_std=prop_std,
        act_mean=act_mean,
        act_std=act_std,
    )

    pos_train_loader = _make_loader(pos_train_ds, batch_size=cfg.train.batch_size, workers=cfg.train.num_workers, shuffle=True)
    neg_train_loader = _make_loader(neg_train_ds, batch_size=cfg.train.batch_size, workers=cfg.train.num_workers, shuffle=True)

    pos_eval_loader = _make_loader(pos_eval_ds, batch_size=cfg.eval.batch_size, workers=max(1, cfg.train.num_workers // 2), shuffle=False)
    neg_eval_loader = _make_loader(neg_eval_ds, batch_size=cfg.eval.batch_size, workers=max(1, cfg.train.num_workers // 2), shuffle=False)

    action_dim = int(pos_train_ds[0]["actions"].numel())
    encoder = PolicyConditionEncoder(backbone_build.backbone)
    model = BCEVisitationDiscriminator(
        encoder=encoder,
        action_dim=action_dim,
        hidden_dim=int(cfg.model.hidden_dim),
        dropout=float(cfg.model.dropout),
    ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    total_steps = cfg.train.epochs * max(len(pos_train_loader), len(neg_train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    img_normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).to(device)

    pos_iter = _cycle_loader(pos_train_loader)
    neg_iter = _cycle_loader(neg_train_loader)

    best_eval_loss = float("inf")
    global_step = 0

    print(
        f"train samples pos/neg={len(pos_train_ds)}/{len(neg_train_ds)} | "
        f"eval samples pos/neg={len(pos_eval_ds)}/{len(neg_eval_ds)}"
    )

    for epoch in range(1, int(cfg.train.epochs) + 1):
        model.train()

        steps = max(len(pos_train_loader), len(neg_train_loader))
        running_loss = 0.0
        running_obj = 0.0

        for _ in range(steps):
            pos_batch = next(pos_iter)
            neg_batch = next(neg_iter)

            pos_images = _normalize_images(pos_batch["images"].to(device, non_blocking=True), img_normalize)
            pos_proprio = pos_batch["proprio"].to(device, non_blocking=True)
            pos_actions = pos_batch["actions"].to(device, non_blocking=True)

            neg_images = _normalize_images(neg_batch["images"].to(device, non_blocking=True), img_normalize)
            neg_proprio = neg_batch["proprio"].to(device, non_blocking=True)
            neg_actions = neg_batch["actions"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            pos_logits = model(images=pos_images, proprio=pos_proprio, actions=pos_actions)
            neg_logits = model(images=neg_images, proprio=neg_proprio, actions=neg_actions)

            objective = F.logsigmoid(pos_logits).mean() + F.logsigmoid(-neg_logits).mean()
            loss = -objective

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=float(cfg.train.grad_clip_norm))
            optimizer.step()
            scheduler.step()

            running_loss += float(loss.item())
            running_obj += float(objective.item())
            global_step += 1

        avg_loss = running_loss / float(steps)
        avg_obj = running_obj / float(steps)

        msg = (
            f"epoch={epoch:04d} step={global_step:07d} train_loss={avg_loss:.6f} "
            f"train_obj={avg_obj:.6f} lr={scheduler.get_last_lr()[0]:.3e}"
        )

        should_eval = (epoch % int(cfg.eval.eval_every) == 0) or (epoch == int(cfg.train.epochs))
        eval_metrics = None

        if should_eval and len(pos_eval_ds) > 0 and len(neg_eval_ds) > 0:
            eval_metrics = evaluate(
                model=model,
                pos_loader=pos_eval_loader,
                neg_loader=neg_eval_loader,
                device=device,
                img_normalize=img_normalize,
            )
            msg += (
                f" | eval_loss={eval_metrics['loss']:.6f} eval_obj={eval_metrics['objective']:.6f} "
                f"eval_acc={eval_metrics['acc']:.4f} "
                f"eval_acc_pos={eval_metrics['acc_pos']:.4f} eval_acc_neg={eval_metrics['acc_neg']:.4f}"
            )

        print(msg)

        save_now = (epoch % int(cfg.save_freq) == 0) or (epoch == int(cfg.train.epochs))
        if save_now:
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "cfg": dict(cfg),
                "history_len": history_len,
                "action_dim": action_dim,
            }
            out = os.path.join(save_dir, f"bce_discriminator_ep{epoch:04d}_{stamp}.pt")
            torch.save(ckpt, out)
            print(f"saved checkpoint: {out}")

        if eval_metrics is not None and eval_metrics["loss"] < best_eval_loss:
            best_eval_loss = eval_metrics["loss"]
            best_path = os.path.join(save_dir, "bce_discriminator_best.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "cfg": dict(cfg),
                    "history_len": history_len,
                    "action_dim": action_dim,
                    "eval": eval_metrics,
                },
                best_path,
            )
            print(f"updated best checkpoint: {best_path}")

    extractor.close()


if __name__ == "__main__":
    main()
