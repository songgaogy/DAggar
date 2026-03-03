import datetime
import glob
import os
from dataclasses import dataclass

import h5py
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import Normalize

from robosuite.discriminator.data import center_crop_resize
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


class InterventionWindowDataset(Dataset):
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
        self.samples: list[tuple[int, int, int]] = []

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
                    pos_set.add(idx)

            for idx in range(info.length):
                if idx < self.history_len - 1:
                    continue
                if idx in pos_set:
                    self.samples.append((demo_id, idx, 1))
                elif labels[idx] == 0:
                    self.samples.append((demo_id, idx, 0))

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
        demo_id, t, label = self.samples[i]
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
            "label": torch.tensor(label, dtype=torch.float32),
        }



def _normalize_images(images: torch.Tensor, normalizer: Normalize) -> torch.Tensor:
    b, k, c, h, w = images.shape
    flat = images.view(b * k, c, h, w)
    return normalizer(flat).view(b, k, c, h, w)


@torch.no_grad()
def evaluate(model, loader, device, img_normalize, threshold):
    model.eval()
    logits_all = []
    labels_all = []
    for batch in loader:
        images = _normalize_images(batch["images"].to(device, non_blocking=True), img_normalize)
        proprio = batch["proprio"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        logits = model(images=images, proprio=proprio, actions=actions)
        logits_all.append(logits)
        labels_all.append(labels)

    logits = torch.cat(logits_all, dim=0)
    labels = torch.cat(labels_all, dim=0)

    n_pos = torch.sum(labels == 1).item()
    n_neg = torch.sum(labels == 0).item()
    pos_weight = torch.tensor([n_neg / max(1.0, n_pos)], device=device, dtype=logits.dtype)

    loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    tp = torch.sum((preds == 1) & (labels == 1)).item()
    tn = torch.sum((preds == 0) & (labels == 0)).item()
    fp = torch.sum((preds == 1) & (labels == 0)).item()
    fn = torch.sum((preds == 0) & (labels == 1)).item()

    acc = (tp + tn) / max(1.0, tp + tn + fp + fn)
    pos_acc = tp / max(1.0, tp + fn)
    neg_acc = tn / max(1.0, tn + fp)
    bal_acc = 0.5 * (pos_acc + neg_acc)

    return {
        "loss": float(loss.item()),
        "acc": float(acc),
        "bal_acc": float(bal_acc),
        "pos_acc": float(pos_acc),
        "neg_acc": float(neg_acc),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


@hydra.main(version_base="1.2", config_path="./config", config_name="finetune_discriminator")
def main(cfg: DictConfig):
    set_seed(int(cfg.seed))
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    policy_ckpt = to_absolute_path(cfg.policy_ckpt)
    disc_ckpt = to_absolute_path(cfg.disc_ckpt)
    data_dir = to_absolute_path(cfg.data_dir)
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

    demos = scan_intervention_demos(data_dir=data_dir, camera_name=cfg.camera, history_len=history_len)
    if train_eps > 0:
        demos = demos[:train_eps]
    if len(demos) < 2:
        raise RuntimeError(f"Need at least 2 intervention demos for split, got {len(demos)}")

    train_demos, eval_demos = split_demos(demos, eval_ratio=float(cfg.eval.eval_ratio), seed=int(cfg.seed))

    train_ds = InterventionWindowDataset(
        demos=train_demos,
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

    eval_ds = InterventionWindowDataset(
        demos=eval_demos,
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

    if len(train_ds) == 0 or len(eval_ds) == 0:
        raise RuntimeError("Empty train/eval samples after building intervention window dataset")

    train_labels = np.array([int(train_ds.samples[i][2]) for i in range(len(train_ds.samples))], dtype=np.int32)
    n_pos = int(np.sum(train_labels == 1))
    n_neg = int(np.sum(train_labels == 0))
    if n_pos == 0 or n_neg == 0:
        raise RuntimeError(f"Train labels degenerate: pos={n_pos} neg={n_neg}")

    class_count = np.array([n_neg, n_pos], dtype=np.float64)
    class_weight = 1.0 / np.maximum(class_count, 1.0)
    sample_weights = class_weight[train_labels]
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.train.batch_size),
        sampler=sampler,
        num_workers=int(cfg.train.num_workers),
        persistent_workers=bool(cfg.train.num_workers > 0),
        pin_memory=True,
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=int(cfg.eval.batch_size),
        shuffle=False,
        num_workers=max(1, int(cfg.train.num_workers) // 2),
        persistent_workers=bool(int(cfg.train.num_workers) > 0),
        pin_memory=True,
    )

    encoder = PolicyConditionEncoder(backbone_build.backbone)
    action_dim = int(train_ds[0]["actions"].numel())
    model = BCEVisitationDiscriminator(
        encoder=encoder,
        action_dim=action_dim,
        hidden_dim=int(cfg.model.hidden_dim),
        dropout=float(cfg.model.dropout),
    ).to(device)

    disc_blob = torch.load(disc_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(disc_blob["model"], strict=True)

    if bool(cfg.train.freeze_policy_backbone):
        for p in model.encoder.parameters():
            p.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay))

    total_steps = int(cfg.train.epochs) * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps), eta_min=1e-6)

    pos_weight = torch.tensor([n_neg / max(1.0, n_pos)], dtype=torch.float32, device=device)
    img_normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).to(device)

    print(
        f"demos total/train/eval={len(demos)}/{len(train_demos)}/{len(eval_demos)} | "
        f"samples train/eval={len(train_ds)}/{len(eval_ds)} | train pos/neg={n_pos}/{n_neg}"
    )

    best_bal_acc = -1.0
    global_step = 0
    for ep in range(1, int(cfg.train.epochs) + 1):
        model.train()
        running = 0.0

        for batch in train_loader:
            images = _normalize_images(batch["images"].to(device, non_blocking=True), img_normalize)
            proprio = batch["proprio"].to(device, non_blocking=True)
            actions = batch["actions"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images=images, proprio=proprio, actions=actions)
            loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=float(cfg.train.grad_clip_norm))
            optimizer.step()
            scheduler.step()

            running += float(loss.item())
            global_step += 1

        train_loss = running / max(1, len(train_loader))

        eval_metrics = evaluate(
            model=model, 
            loader=eval_loader, 
            device=device, 
            img_normalize=img_normalize,
            threshold=cfg.eval.threshold
        )
        print(
            f"epoch={ep:04d} step={global_step:07d} train_loss={train_loss:.6f} "
            f"eval_loss={eval_metrics['loss']:.6f} eval_acc={eval_metrics['acc']:.4f} "
            f"eval_bal_acc={eval_metrics['bal_acc']:.4f} "
            f"eval_pos_acc={eval_metrics['pos_acc']:.4f} eval_neg_acc={eval_metrics['neg_acc']:.4f}"
        )

        if ep % int(cfg.save_freq) == 0 or ep == int(cfg.train.epochs):
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out = os.path.join(save_dir, f"finetune_discriminator_ep{ep:04d}_{stamp}.pt")
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

        if eval_metrics["bal_acc"] > best_bal_acc:
            best_bal_acc = eval_metrics["bal_acc"]
            best_path = os.path.join(save_dir, "finetune_discriminator_best.pt")
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

    extractor.close()


if __name__ == "__main__":
    main()
