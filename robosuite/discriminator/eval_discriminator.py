import glob
import os
from dataclasses import dataclass

import h5py
import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Normalize
from tqdm import tqdm

from robosuite.discriminator.data import center_crop_resize
from robosuite.discriminator.model import (
    BCEVisitationDiscriminator,
    PolicyConditionEncoder,
    build_policy_backbone_from_ckpt,
)
from robosuite.policy.utils.env_util import PandaLiftProprioExtractor


@dataclass(frozen=True)
class EvalDemoInfo:
    file_path: str
    demo_key: str
    length: int


def _to_numpy(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


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


class InterventionStartEvalDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        camera_name: str,
        history_len: int,
        image_size: int,
        proprio_extractor,
        window_before: int,
        window_after: int,
        normalize: bool = True,
        prop_mean: np.ndarray | None = None,
        prop_std: np.ndarray | None = None,
        act_mean: np.ndarray | None = None,
        act_std: np.ndarray | None = None,
        max_demos: int | None = None,
    ):
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
        self.demo_infos: list[EvalDemoInfo] = []
        self.samples: list[tuple[int, int, int]] = []

        files = _list_hdf5_files(data_dir)
        for fp in files:
            with h5py.File(fp, "r") as f:
                demos = f["demos"]
                for dk in sorted(demos.keys()):
                    demo = demos[dk]
                    if "intervention_labels" not in demo:
                        continue
                    if "observations" not in demo or self.camera_name not in demo["observations"]:
                        continue

                    labels = np.asarray(demo["intervention_labels"][:], dtype=np.uint8)
                    t = min(
                        labels.shape[0],
                        int(demo["states"].shape[0]),
                        int(demo["actions"].shape[0]),
                        int(demo["observations"][self.camera_name]["images"].shape[0]),
                    )
                    if t < self.history_len:
                        continue

                    labels = labels[:t]
                    starts = _intervention_starts(labels)
                    if not starts:
                        continue

                    demo_id = len(self.demo_infos)
                    self.demo_infos.append(EvalDemoInfo(file_path=fp, demo_key=dk, length=t))

                    pos_set = set()
                    for s in starts:
                        lo = max(0, s - self.window_before)
                        hi = min(t - 1, s + self.window_after)
                        for idx in range(lo, hi + 1):
                            pos_set.add(idx)

                    for idx in range(t):
                        if idx < self.history_len - 1:
                            continue
                        if idx in pos_set:
                            self.samples.append((demo_id, idx, 1))
                        elif labels[idx] == 0:
                            self.samples.append((demo_id, idx, 0))

                    if max_demos is not None and len(self.demo_infos) >= int(max_demos):
                        break
            if max_demos is not None and len(self.demo_infos) >= int(max_demos):
                break

        if self.demo_infos and self.act_mean is not None and self.act_std is not None:
            action_dim = self._infer_action_dim()
            if action_dim is not None:
                self.act_mean, self.act_std = _resolve_action_stats(self.act_mean, self.act_std, action_dim)

    def _infer_action_dim(self):
        if not self.demo_infos:
            return None
        info = self.demo_infos[0]
        with h5py.File(info.file_path, "r") as f:
            action_ds = f["demos"][info.demo_key]["actions"]
            if action_ds.ndim < 2:
                return None
            return int(action_ds.shape[-1])

    def __len__(self):
        return len(self.samples)

    def _get_handle(self, file_path: str) -> h5py.File:
        pid = os.getpid()
        if pid not in self._handles:
            self._handles[pid] = {}
        if file_path not in self._handles[pid]:
            self._handles[pid][file_path] = h5py.File(file_path, "r", libver="latest", swmr=True)
        return self._handles[pid][file_path]

    def __getitem__(self, i):
        demo_id, t, label = self.samples[i]
        info = self.demo_infos[demo_id]

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
            "label": torch.tensor(label, dtype=torch.int64),
        }


def _normalize_images(images: torch.Tensor, normalizer: Normalize) -> torch.Tensor:
    b, k, c, h, w = images.shape
    flat = images.view(b * k, c, h, w)
    return normalizer(flat).view(b, k, c, h, w)


@hydra.main(version_base="1.2", config_path="./config", config_name="eval_discriminator")
def main(cfg: DictConfig):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    disc_ckpt = to_absolute_path(cfg.disc_ckpt)
    policy_ckpt = to_absolute_path(cfg.policy_ckpt)
    data_dir = to_absolute_path(cfg.data_dir)

    backbone_build = build_policy_backbone_from_ckpt(
        ckpt_path=policy_ckpt,
        history_len=int(cfg.history_len),
        freeze_backbone=True,
        device=str(device),
    )
    history_len = int(backbone_build.history_len)

    disc_blob = torch.load(disc_ckpt, map_location="cpu", weights_only=False)
    action_dim = int(disc_blob.get("action_dim", 0))

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

    ds = InterventionStartEvalDataset(
        data_dir=data_dir,
        camera_name=cfg.camera,
        history_len=history_len,
        image_size=int(cfg.image_size),
        proprio_extractor=extractor.extract,
        window_before=int(cfg.eval.window_before),
        window_after=int(cfg.eval.window_after),
        normalize=bool(cfg.eval.normalize_with_policy_stats),
        prop_mean=prop_mean,
        prop_std=prop_std,
        act_mean=act_mean,
        act_std=act_std,
        max_demos=(int(cfg.eval.max_demos) if int(cfg.eval.max_demos) > 0 else None),
    )
    if len(ds) == 0:
        raise RuntimeError("No eval samples found. Check data path and intervention labels.")

    if action_dim <= 0:
        action_dim = int(ds[0]["actions"].numel())

    loader = DataLoader(
        ds,
        batch_size=int(cfg.eval.batch_size),
        shuffle=False,
        num_workers=int(cfg.eval.num_workers),
        persistent_workers=bool(cfg.eval.num_workers > 0),
        pin_memory=True,
    )

    encoder = PolicyConditionEncoder(backbone_build.backbone)
    model = BCEVisitationDiscriminator(
        encoder=encoder,
        action_dim=action_dim,
        hidden_dim=int(cfg.model.hidden_dim),
        dropout=float(cfg.model.dropout),
    ).to(device)
    model.load_state_dict(disc_blob["model"], strict=True)
    model.eval()

    img_normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).to(device)

    y_true_list = []
    y_pred_list = []
    prob_list = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="evaluating..."):
            images = _normalize_images(batch["images"].to(device, non_blocking=True), img_normalize)
            proprio = batch["proprio"].to(device, non_blocking=True)
            actions = batch["actions"].to(device, non_blocking=True)

            logits = model(images=images, proprio=proprio, actions=actions)
            probs = torch.sigmoid(logits)
            preds = (probs >= float(cfg.eval.threshold)).long()

            y_true_list.append(batch["label"].cpu().numpy())
            y_pred_list.append(preds.cpu().numpy())
            prob_list.append(probs.cpu().numpy())

    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)
    probs = np.concatenate(prob_list, axis=0)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    acc = float((tp + tn) / max(1, len(y_true)))
    pos_acc = float(tp / max(1, tp + fn))
    neg_acc = float(tn / max(1, tn + fp))
    bal_acc = 0.5 * (pos_acc + neg_acc)
    precision = float(tp / max(1, tp + fp))
    recall = pos_acc
    f1 = float(2.0 * precision * recall / max(1e-12, precision + recall))

    print("### Discriminator Eval on Intervention-Start Windows")
    print(
        f"history_len={history_len}\n"
        f"window=[-{int(cfg.eval.window_before)}, +{int(cfg.eval.window_after)}]\n"
        f"threshold={float(cfg.eval.threshold):.3f}"
    )
    print(f"num_demos={len(ds.demo_infos)}\nnum_samples={len(ds)}")
    print(f"positive_samples={int(np.sum(y_true==1))}\nnegative_samples={int(np.sum(y_true==0))}")
    print(f"accuracy={acc:.4f}\nbalanced_accuracy={bal_acc:.4f}")
    print(f"precision={precision:.4f}\nrecall={recall:.4f}\nf1={f1:.4f}")
    print(f"mean_prob_pos={float(probs[y_true==1].mean()):.4f}\nmean_prob_neg={float(probs[y_true==0].mean()):.4f}")

    extractor.close()


if __name__ == "__main__":
    main()
