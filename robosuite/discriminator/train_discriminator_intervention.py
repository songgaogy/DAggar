import datetime
import glob
import os
from dataclasses import dataclass
from typing import Iterator

import h5py
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Normalize

from robosuite.discriminator.data import StateActionHdf5Dataset, center_crop_resize, scan_demos, split_demo_infos
from robosuite.discriminator.model import (
    BCEVisitationDiscriminator,
    PolicyConditionEncoder,
    build_policy_backbone_from_ckpt,
)
from robosuite.policy.utils.env_util import PandaLiftProprioExtractor


@dataclass(frozen=True)
class DemoInfo:
    file_path: str
    demo_key: str
    length: int


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_numpy(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _list_hdf5_files(data_dir: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(data_dir, "*.hdf5")))
    if not files:
        raise FileNotFoundError(f"No .hdf5 files found in {data_dir}")
    return files


def _intervention_starts(labels: np.ndarray) -> list[int]:
    starts = []
    prev = 0
    for i, v in enumerate(labels.astype(np.int32)):
        if v == 1 and prev == 0:
            starts.append(i)
        prev = v
    return starts


def _resolve_action_stats(act_mean, act_std, action_dim: int):
    if act_mean is None or act_std is None:
        return None, None

    mean = np.asarray(act_mean, dtype=np.float32).reshape(-1)
    std = np.asarray(act_std, dtype=np.float32).reshape(-1)
    if mean.shape != std.shape:
        return None, None

    if mean.size == action_dim:
        return mean, std

    if mean.size % action_dim == 0:
        chunks = mean.size // action_dim
        mean_2d = mean.reshape(chunks, action_dim)
        std_2d = std.reshape(chunks, action_dim)
        merged_mean = mean_2d.mean(axis=0)
        second_moment = (std_2d**2 + mean_2d**2).mean(axis=0)
        merged_std = np.sqrt(np.maximum(second_moment - merged_mean**2, 1e-12))
        return merged_mean, merged_std

    return None, None


def scan_intervention_demos(data_dir: str, camera_name: str, history_len: int) -> list[DemoInfo]:
    demos = []
    for fp in _list_hdf5_files(data_dir):
        with h5py.File(fp, "r") as f:
            group = f["demos"]
            for dk in sorted(group.keys()):
                demo = group[dk]
                if "intervention_labels" not in demo:
                    continue
                if "observations" not in demo or camera_name not in demo["observations"]:
                    continue
                labels = np.asarray(demo["intervention_labels"][:], dtype=np.uint8)
                t = min(
                    labels.shape[0],
                    int(demo["states"].shape[0]),
                    int(demo["actions"].shape[0]),
                    int(demo["observations"][camera_name]["images"].shape[0]),
                )
                if t < history_len:
                    continue
                if len(_intervention_starts(labels[:t])) == 0:
                    continue
                demos.append(DemoInfo(file_path=fp, demo_key=dk, length=t))
    return demos


def split_demos(demos: list[DemoInfo], eval_ratio: float, seed: int):
    if not demos:
        return [], []
    rng = np.random.default_rng(seed)
    idx = np.arange(len(demos))
    rng.shuffle(idx)

    n_eval = int(round(len(demos) * float(eval_ratio)))
    if len(demos) >= 2:
        n_eval = min(max(n_eval, 1), len(demos) - 1)
    else:
        n_eval = 0

    eval_ids = set(idx[:n_eval].tolist())
    train = [d for i, d in enumerate(demos) if i not in eval_ids]
    evals = [d for i, d in enumerate(demos) if i in eval_ids]
    return train, evals


class InterventionStartWindowPositiveDataset(Dataset):
    def __init__(
        self,
        demos: list[DemoInfo],
        camera_name: str,
        history_len: int,
        image_size: int,
        proprio_extractor,
        window_before: int,
        window_after: int,
        normalize: bool,
        prop_mean,
        prop_std,
        act_mean,
        act_std,
    ):
        self.demos = list(demos)
        self.camera_name = camera_name
        self.history_len = int(history_len)
        self.image_size = int(image_size)
        self.proprio_extractor = proprio_extractor
        self.window_before = int(window_before)
        self.window_after = int(window_after)
        self.normalize = bool(normalize)
        self.prop_mean = prop_mean
        self.prop_std = prop_std
        self.act_mean = act_mean
        self.act_std = act_std

        self._handles: dict[int, dict[str, h5py.File]] = {}
        self._resize_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.samples: list[tuple[int, int]] = []

        self._build_samples()
        self._resolve_action_stats()

    def _get_handle(self, file_path: str) -> h5py.File:
        pid = os.getpid()
        if pid not in self._handles:
            self._handles[pid] = {}
        if file_path not in self._handles[pid]:
            self._handles[pid][file_path] = h5py.File(file_path, "r", libver="latest", swmr=True)
        return self._handles[pid][file_path]

    def _build_samples(self):
        for demo_id, info in enumerate(self.demos):
            with h5py.File(info.file_path, "r") as f:
                demo = f["demos"][info.demo_key]
                labels = np.asarray(demo["intervention_labels"][:], dtype=np.uint8)[: info.length]

            starts = _intervention_starts(labels)
            if not starts:
                continue

            pos_set = set()
            for s in starts:
                lo = max(0, s - self.window_before)
                hi = min(info.length - 1, s + self.window_after)
                for idx in range(lo, hi + 1):
                    if idx >= self.history_len - 1:
                        pos_set.add(idx)

            for idx in sorted(pos_set):
                self.samples.append((demo_id, idx))

    def _infer_action_dim(self):
        if not self.demos:
            return None
        info = self.demos[0]
        with h5py.File(info.file_path, "r") as f:
            ds = f["demos"][info.demo_key]["actions"]
            if ds.ndim < 2:
                return None
            return int(ds.shape[-1])

    def _resolve_action_stats(self):
        action_dim = self._infer_action_dim()
        if action_dim is None:
            self.act_mean, self.act_std = None, None
            return
        self.act_mean, self.act_std = _resolve_action_stats(self.act_mean, self.act_std, action_dim)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        demo_id, t = self.samples[i]
        info = self.demos[demo_id]

        f = self._get_handle(info.file_path)
        demo = f["demos"][info.demo_key]

        t0 = t - self.history_len + 1
        t1 = t + 1

        image_seq = demo["observations"][self.camera_name]["images"][t0:t1]
        state_seq = demo["states"][t0:t1]
        action = demo["actions"][t].astype(np.float32)

        img_list = []
        for k in range(self.history_len):
            img = center_crop_resize(image_seq[k], self.image_size, self._resize_cache)
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            img_list.append(img)
        images = np.stack(img_list, axis=0).astype(np.float32)

        prop_seq = np.stack([self.proprio_extractor(s).astype(np.float32) for s in state_seq], axis=0)
        proprio = prop_seq.reshape(-1).astype(np.float32)

        if self.normalize:
            if self.prop_mean is not None and self.prop_std is not None:
                proprio = (proprio - self.prop_mean) / (self.prop_std + 1e-6)
            if self.act_mean is not None and self.act_std is not None:
                action = (action - self.act_mean) / (self.act_std + 1e-6)

        return {
            "images": torch.from_numpy(images),
            "proprio": torch.from_numpy(proprio),
            "actions": torch.from_numpy(action.astype(np.float32)),
            "label": torch.tensor(1.0, dtype=torch.float32),
        }


def _normalize_images(images: torch.Tensor, normalizer: Normalize) -> torch.Tensor:
    b, k, c, h, w = images.shape
    flat = images.view(b * k, c, h, w)
    return normalizer(flat).view(b, k, c, h, w)


def augment_images(images: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """
    Random shift / crop with strict temporal consistency per sample.
    Expects [B, K, C, H, W].
    """
    if pad <= 0:
        return images

    b, k, c, h, w = images.shape
    flat = images.view(b * k, c, h, w)
    flat_pad = F.pad(flat, (pad, pad, pad, pad), mode="replicate")
    pad_imgs = flat_pad.view(b, k, c, h + 2 * pad, w + 2 * pad)

    w_start = torch.randint(0, 2 * pad + 1, (b,), device=images.device)
    h_start = torch.randint(0, 2 * pad + 1, (b,), device=images.device)

    out = torch.empty((b, k, c, h, w), device=images.device, dtype=images.dtype)
    for i in range(b):
        hs = int(h_start[i].item())
        ws = int(w_start[i].item())
        out[i] = pad_imgs[i, :, :, hs : hs + h, ws : ws + w]
    return out


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
def evaluate(model, pos_loader, neg_loader, device, img_normalize, threshold):
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
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
        }

    pos_logits = torch.cat(pos_logits_all, dim=0)
    neg_logits = torch.cat(neg_logits_all, dim=0)

    objective = F.logsigmoid(pos_logits).mean() + F.logsigmoid(-neg_logits).mean()
    loss = -objective

    pos_probs = torch.sigmoid(pos_logits)
    neg_probs = torch.sigmoid(neg_logits)

    tp = int((pos_probs >= threshold).sum().item())
    fn = int((pos_probs < threshold).sum().item())
    tn = int((neg_probs < threshold).sum().item())
    fp = int((neg_probs >= threshold).sum().item())

    acc_pos = tp / max(1, tp + fn)
    acc_neg = tn / max(1, tn + fp)
    acc = 0.5 * (acc_pos + acc_neg)

    # threshold sweep for better operational point under class imbalance
    best_thr = float(threshold)
    best_bal_acc = float(acc)
    for thr in np.linspace(0.05, 0.95, 19):
        tp_t = int((pos_probs >= thr).sum().item())
        fn_t = int((pos_probs < thr).sum().item())
        tn_t = int((neg_probs < thr).sum().item())
        fp_t = int((neg_probs >= thr).sum().item())
        acc_pos_t = tp_t / max(1, tp_t + fn_t)
        acc_neg_t = tn_t / max(1, tn_t + fp_t)
        bal_acc_t = 0.5 * (acc_pos_t + acc_neg_t)
        if bal_acc_t > best_bal_acc:
            best_bal_acc = float(bal_acc_t)
            best_thr = float(thr)

    return {
        "loss": float(loss.item()),
        "objective": float(objective.item()),
        "acc": float(acc),
        "acc_pos": float(acc_pos),
        "acc_neg": float(acc_neg),
        "best_bal_acc": best_bal_acc,
        "best_thr": best_thr,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


@hydra.main(version_base="1.2", config_path="./config", config_name="train_discriminator_intervention")
def main(cfg: DictConfig):
    set_seed(int(cfg.seed))
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    policy_ckpt = to_absolute_path(cfg.policy_ckpt)
    intervention_dir = to_absolute_path(cfg.data.intervention_dir)
    expert_dir = to_absolute_path(cfg.data.expert_dir)
    save_dir = to_absolute_path(cfg.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    train_eps = int(getattr(cfg, "train_eps", -1))

    backbone_build = build_policy_backbone_from_ckpt(
        ckpt_path=policy_ckpt,
        history_len=int(cfg.history_len),
        freeze_backbone=bool(cfg.train.freeze_policy_backbone),
        device=str(device),
    )
    history_len = int(backbone_build.history_len)

    policy_blob = torch.load(policy_ckpt, map_location="cpu", weights_only=False)
    prop_mean = _to_numpy(policy_blob.get("prop_mean"))
    prop_std = _to_numpy(policy_blob.get("prop_std"))
    act_mean = _to_numpy(policy_blob.get("act_mean"))
    act_std = _to_numpy(policy_blob.get("act_std"))

    extractor = PandaLiftProprioExtractor(
        robots="Panda",
        env_name="Lift",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        camera_names=None,
        reward_shaping=False,
    )

    pos_demos = scan_intervention_demos(data_dir=intervention_dir, camera_name=cfg.camera, history_len=history_len)
    if train_eps > 0:
        pos_demos = pos_demos[:train_eps]
    if len(pos_demos) < 2:
        raise RuntimeError(f"Need at least 2 intervention demos for split, got {len(pos_demos)}")

    pos_train_demos, pos_eval_demos = split_demos(pos_demos, eval_ratio=float(cfg.eval.eval_ratio), seed=int(cfg.seed))

    neg_infos = scan_demos(
        data_dir=expert_dir,
        camera_name=cfg.camera,
        history_len=history_len,
        max_trajectories=None,
    )
    if len(neg_infos) < 2:
        raise RuntimeError(f"Need at least 2 expert demos for split, got {len(neg_infos)}")
    neg_train_infos, neg_eval_infos = split_demo_infos(neg_infos, eval_ratio=float(cfg.eval.eval_ratio), seed=int(cfg.seed) + 1)

    pos_train_ds = InterventionStartWindowPositiveDataset(
        demos=pos_train_demos,
        camera_name=cfg.camera,
        history_len=history_len,
        image_size=int(cfg.image_size),
        proprio_extractor=extractor.extract,
        window_before=int(cfg.labels.window_before),
        window_after=int(cfg.labels.window_after),
        normalize=bool(cfg.data.normalize_with_policy_stats),
        prop_mean=prop_mean,
        prop_std=prop_std,
        act_mean=act_mean,
        act_std=act_std,
    )
    pos_eval_ds = InterventionStartWindowPositiveDataset(
        demos=pos_eval_demos,
        camera_name=cfg.camera,
        history_len=history_len,
        image_size=int(cfg.image_size),
        proprio_extractor=extractor.extract,
        window_before=int(cfg.labels.window_before),
        window_after=int(cfg.labels.window_after),
        normalize=bool(cfg.data.normalize_with_policy_stats),
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
        image_size=int(cfg.image_size),
        label=0.0,
        normalize=bool(cfg.data.normalize_with_policy_stats),
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
        image_size=int(cfg.image_size),
        label=0.0,
        normalize=bool(cfg.data.normalize_with_policy_stats),
        prop_mean=prop_mean,
        prop_std=prop_std,
        act_mean=act_mean,
        act_std=act_std,
    )

    if len(pos_train_ds) == 0 or len(neg_train_ds) == 0:
        raise RuntimeError("Empty train samples from intervention positives or expert negatives")
    if len(pos_eval_ds) == 0 or len(neg_eval_ds) == 0:
        raise RuntimeError("Empty eval samples from intervention positives or expert negatives")

    pos_train_loader = _make_loader(pos_train_ds, batch_size=int(cfg.train.batch_size), workers=int(cfg.train.num_workers), shuffle=True)
    neg_train_loader = _make_loader(neg_train_ds, batch_size=int(cfg.train.batch_size), workers=int(cfg.train.num_workers), shuffle=True)
    pos_eval_loader = _make_loader(pos_eval_ds, batch_size=int(cfg.eval.batch_size), workers=max(1, int(cfg.train.num_workers) // 2), shuffle=False)
    neg_eval_loader = _make_loader(neg_eval_ds, batch_size=int(cfg.eval.batch_size), workers=max(1, int(cfg.train.num_workers) // 2), shuffle=False)

    encoder = PolicyConditionEncoder(backbone_build.backbone)
    action_dim = int(pos_train_ds[0]["actions"].numel())
    model = BCEVisitationDiscriminator(
        encoder=encoder,
        action_dim=action_dim,
        hidden_dim=int(cfg.model.hidden_dim),
        dropout=float(cfg.model.dropout),
    ).to(device)

    if bool(cfg.train.freeze_policy_backbone):
        for p in model.encoder.parameters():
            p.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay))

    total_steps = int(cfg.train.epochs) * max(len(pos_train_loader), len(neg_train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps), eta_min=1e-6)

    img_normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).to(device)
    aug_pad = int(getattr(cfg.train, "augment_pad", 4))
    label_smoothing = float(getattr(cfg.train, "label_smoothing", 0.05))
    label_smoothing = min(max(label_smoothing, 0.0), 0.49)
    early_stop_patience = int(getattr(cfg.train, "early_stop_patience", 12))

    pos_iter = _cycle_loader(pos_train_loader)
    neg_iter = _cycle_loader(neg_train_loader)

    print(
        f"demos pos(intervention) total/train/eval={len(pos_demos)}/{len(pos_train_demos)}/{len(pos_eval_demos)} | "
        f"demos neg(expert) total/train/eval={len(neg_infos)}/{len(neg_train_infos)}/{len(neg_eval_infos)}"
    )
    print(
        f"samples pos train/eval={len(pos_train_ds)}/{len(pos_eval_ds)} | "
        f"samples neg train/eval={len(neg_train_ds)}/{len(neg_eval_ds)}"
    )

    best_bal_acc = -1.0
    no_improve_epochs = 0
    global_step = 0

    for ep in range(1, int(cfg.train.epochs) + 1):
        model.train()
        running_loss = 0.0
        running_obj = 0.0

        steps = max(len(pos_train_loader), len(neg_train_loader))
        for _ in range(steps):
            pos_batch = next(pos_iter)
            neg_batch = next(neg_iter)

            pos_images = pos_batch["images"].to(device, non_blocking=True)
            pos_proprio = pos_batch["proprio"].to(device, non_blocking=True)
            pos_actions = pos_batch["actions"].to(device, non_blocking=True)

            neg_images = neg_batch["images"].to(device, non_blocking=True)
            neg_proprio = neg_batch["proprio"].to(device, non_blocking=True)
            neg_actions = neg_batch["actions"].to(device, non_blocking=True)

            pos_images = _normalize_images(augment_images(pos_images, pad=aug_pad), img_normalize)
            neg_images = _normalize_images(augment_images(neg_images, pad=aug_pad), img_normalize)

            optimizer.zero_grad(set_to_none=True)

            pos_logits = model(images=pos_images, proprio=pos_proprio, actions=pos_actions)
            neg_logits = model(images=neg_images, proprio=neg_proprio, actions=neg_actions)

            objective = F.logsigmoid(pos_logits).mean() + F.logsigmoid(-neg_logits).mean()
            pos_target = torch.full_like(pos_logits, 1.0 - label_smoothing)
            neg_target = torch.full_like(neg_logits, label_smoothing)
            pos_bce = F.binary_cross_entropy_with_logits(pos_logits, pos_target)
            neg_bce = F.binary_cross_entropy_with_logits(neg_logits, neg_target)
            bce_loss = 0.5 * (pos_bce + neg_bce)
            loss = -objective + float(getattr(cfg.train, "bce_aux_weight", 0.2)) * bce_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=float(cfg.train.grad_clip_norm))
            optimizer.step()
            scheduler.step()

            running_loss += float(loss.item())
            running_obj += float(objective.item())
            global_step += 1

        train_loss = running_loss / max(1, steps)
        train_obj = running_obj / max(1, steps)

        eval_metrics = evaluate(
            model=model,
            pos_loader=pos_eval_loader,
            neg_loader=neg_eval_loader,
            device=device,
            img_normalize=img_normalize,
            threshold=float(cfg.eval.threshold),
        )

        print(
            f"epoch={ep:04d} step={global_step:07d} train_loss={train_loss:.6f} train_obj={train_obj:.6f} "
            f"eval_loss={eval_metrics['loss']:.6f} eval_obj={eval_metrics['objective']:.6f} "
            f"eval_acc={eval_metrics['acc']:.4f} eval_acc_pos={eval_metrics['acc_pos']:.4f} eval_acc_neg={eval_metrics['acc_neg']:.4f} "
            f"best_thr={eval_metrics['best_thr']:.2f} best_bal_acc={eval_metrics['best_bal_acc']:.4f}"
        )

        if ep % int(cfg.save_freq) == 0 or ep == int(cfg.train.epochs):
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out = os.path.join(save_dir, f"train_intervention_discriminator_ep{ep:04d}_{stamp}.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": ep,
                    "cfg": dict(cfg),
                    "action_dim": action_dim,
                    "history_len": history_len,
                    "eval": eval_metrics,
                },
                out,
            )
            print(f"saved checkpoint: {out}")

        if eval_metrics["best_bal_acc"] > best_bal_acc:
            best_bal_acc = eval_metrics["best_bal_acc"]
            no_improve_epochs = 0
            best_path = os.path.join(save_dir, "train_intervention_discriminator_best.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": ep,
                    "cfg": dict(cfg),
                    "action_dim": action_dim,
                    "history_len": history_len,
                    "eval": eval_metrics,
                },
                best_path,
            )
            print(f"updated best checkpoint: {best_path}")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= early_stop_patience:
                print(f"early stop at epoch={ep}, no improvement for {no_improve_epochs} epochs")
                break

    extractor.close()


if __name__ == "__main__":
    main()
