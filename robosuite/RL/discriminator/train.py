import argparse
import os
import random
from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from dataset import PandaLiftH5StepsDataset, collate_fn
from metrics import eval_binary_logits
from models import ClassifierConfig, PRVMClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_indices(n: int, val_frac: float, seed: int) -> Tuple[List[int], List[int]]:
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(round(n * val_frac))
    val_idx = idx[:n_val].tolist()
    train_idx = idx[n_val:].tolist()
    return train_idx, val_idx


def build_balanced_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    # Balanced sampling across classes (PRVM symmetric sampling)
    class_count = np.bincount(labels.astype(np.int64), minlength=2)
    class_weight = 1.0 / np.maximum(class_count, 1)
    sample_weight = class_weight[labels.astype(np.int64)]
    sample_weight = torch.tensor(sample_weight, dtype=torch.double)
    sampler = WeightedRandomSampler(weights=sample_weight, num_samples=len(sample_weight), replacement=True)
    return sampler


def infer_dims(dataset: PandaLiftH5StepsDataset) -> Tuple[int, int]:
    # Find one sample containing state/action
    for i in range(min(len(dataset), 100)):
        s = dataset[i]
        if "state" in s and "action" in s:
            return int(s["state"].shape[0]), int(s["action"].shape[0])
    raise RuntimeError("Could not infer state/action dims. Ensure use_state_action=True and data contains states/actions.")


@torch.no_grad()
def run_eval(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    max_batches: int = -1,
) -> Dict[str, float]:
    model.eval()
    all_logits = []
    all_y = []
    for bi, batch in enumerate(loader):
        if max_batches > 0 and bi >= max_batches:
            break

        y = batch["label"].to(device)
        kwargs = {}
        if "img_agent" in batch:
            kwargs["img_agent"] = batch["img_agent"].to(device)
            kwargs["img_wrist"] = batch["img_wrist"].to(device)
        if "state" in batch:
            kwargs["state"] = batch["state"].to(device)
            kwargs["action"] = batch["action"].to(device)

        logits = model(**kwargs)
        all_logits.append(logits.detach().cpu())
        all_y.append(y.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    y = torch.cat(all_y, dim=0)
    return eval_binary_logits(logits, y, threshold=threshold)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--index_cache", type=str, default="")
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--max_steps_per_demo", type=int, default=0)
    parser.add_argument("--use_images", action="store_true")
    parser.add_argument("--use_state_action", action="store_true")
    parser.add_argument("--image_pretrained", action="store_true")
    parser.add_argument("--train_steps", type=str, required=True,
                    help="Comma-separated step dirs, e.g. step-150k,step-470k")
    parser.add_argument("--eval_steps", type=str, required=True,
                    help="Comma-separated step dirs, e.g. step-960k")
    parser.add_argument("--max_eval_batches", type=int, default=-1)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--logdir", type=str, default="./runs_prvm_classifier")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="prvm-classifier")
    parser.add_argument("--wandb_name", type=str, default="")

    args = parser.parse_args()

    def parse_steps(s: str) -> List[str]:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if len(parts) == 0:
            raise ValueError("Empty steps string.")
        return parts

    if not args.use_images and not args.use_state_action:
        raise ValueError("Set at least one of --use_images or --use_state_action.")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    index_cache = args.index_cache if args.index_cache.strip() else None
    max_steps = args.max_steps_per_demo if args.max_steps_per_demo > 0 else None

    train_steps = parse_steps(args.train_steps)
    eval_steps = parse_steps(args.eval_steps)

    def steps_to_tag(steps: List[str]) -> str:
        return "_".join([x.replace("/", "_") for x in steps])

    train_cache = None
    eval_cache = None
    if args.index_cache.strip():
        base, ext = os.path.splitext(args.index_cache)
        train_cache = f"{base}.train.{steps_to_tag(train_steps)}.npz"
        eval_cache = f"{base}.eval.{steps_to_tag(eval_steps)}.npz"

    train_dataset = PandaLiftH5StepsDataset(
        root_dir=args.data_root,
        success_subdirs=("success",),
        fail_subdirs=("pure_fail",),  # recommended for your current layout
        success_step_dirs=train_steps,
        fail_step_dirs=train_steps,
        use_images=args.use_images,
        use_state_action=args.use_state_action,
        max_steps_per_demo=max_steps,
        seed=args.seed,
        cache_index_path=train_cache,
    )

    eval_dataset = PandaLiftH5StepsDataset(
        root_dir=args.data_root,
        success_subdirs=("success",),
        fail_subdirs=("pure_fail",),
        success_step_dirs=eval_steps,
        fail_step_dirs=eval_steps,
        use_images=args.use_images,
        use_state_action=args.use_state_action,
        max_steps_per_demo=max_steps,
        seed=args.seed + 1,
        cache_index_path=eval_cache,
    )

    train_ds = train_dataset
    val_ds = eval_dataset

    # Build labels for sampler
    labels = np.array([it.label for it in train_dataset.items], dtype=np.int64)
    sampler = build_balanced_sampler(labels)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    state_dim, action_dim = (0, 0)
    if args.use_state_action:
        state_dim, action_dim = infer_dims(train_dataset)

    cfg = ClassifierConfig(
        use_images=args.use_images,
        use_state_action=args.use_state_action,
        image_pretrained=args.image_pretrained,
        img_embed_dim=256,
        sa_embed_dim=256,
        fusion_hidden=512,
        fusion_depth=3,
        dropout=0.1,
    )
    model = PRVMClassifier(state_dim=state_dim, action_dim=action_dim, cfg=cfg).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())
    loss_fn = nn.BCEWithLogitsLoss()

    os.makedirs(args.logdir, exist_ok=True)
    ckpt_path = os.path.join(args.logdir, "best.pt")

    use_wandb = args.wandb
    wb = None
    if use_wandb:
        import wandb
        wb = wandb
        wb.init(
            project=args.wandb_project,
            name=args.wandb_name if args.wandb_name else None,
            config={
                **vars(args),
                "model_cfg": asdict(cfg),
                "state_dim": state_dim,
                "action_dim": action_dim,
                "num_samples": len(train_dataset),
                "num_train": len(train_ds),
                "num_val": len(val_ds),
            },
        )

    best_val_bce = float("inf")
    global_step = 0

    for epoch in range(args.epochs):
        model.train()

        running_loss = 0.0
        running_count = 0

        for batch_idx, batch in enumerate(train_loader):
            y = batch["label"].to(device).float()

            kwargs = {}
            if "img_agent" in batch:
                kwargs["img_agent"] = batch["img_agent"].to(device, non_blocking=True)
                kwargs["img_wrist"] = batch["img_wrist"].to(device, non_blocking=True)
            if "state" in batch:
                kwargs["state"] = batch["state"].to(device, non_blocking=True)
                kwargs["action"] = batch["action"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast(enabled=torch.cuda.is_available(), device_type="cuda"):
                logits = model(**kwargs)
                loss = loss_fn(logits, y)

            scaler.scale(loss).backward()

            if args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(opt)
            scaler.update()

            running_loss += loss.item() * y.size(0)
            running_count += y.size(0)

            if batch_idx % 100 == 0:
                with torch.no_grad():
                    probs = torch.sigmoid(logits)
                    pred = (probs > args.threshold).float()

                    pos_ratio = y.mean().item()
                    pred_pos_ratio = pred.mean().item()
                    logit_mean = logits.mean().item()
                    logit_std = logits.std().item()
                    lr = opt.param_groups[0]["lr"]

                print(
                    f"[epoch {epoch:03d} | batch {batch_idx:04d}] "
                    f"loss={loss.item():.4f} "
                    f"lr={lr:.2e} "
                    f"logit_mean={logit_mean:.3f} "
                    f"logit_std={logit_std:.3f} "
                    f"pos_ratio={pos_ratio:.3f} "
                    f"pred_pos_ratio={pred_pos_ratio:.3f}"
                )

            # wandb
            if use_wandb and (global_step % 50 == 0):
                with torch.no_grad():
                    metrics = eval_binary_logits(
                        logits.detach(),
                        y.detach().long(),
                        threshold=args.threshold
                    )
                wb.log(
                    {
                        "train/loss": float(loss.item()),
                        "train/acc": metrics["acc"],
                        "train/f1": metrics["f1"],
                        "train/auroc": metrics["auroc"],
                        "step": global_step,
                        "epoch": epoch,
                    },
                    step=global_step,
                )

            global_step += 1

        epoch_loss = running_loss / max(running_count, 1)
        print(f"\n[epoch {epoch:03d}] TRAIN mean_loss={epoch_loss:.4f}")

        val_metrics = run_eval(model, val_loader, device=device, threshold=args.threshold, max_batches=args.max_eval_batches)

        print(
            f"[epoch {epoch:03d}] VAL "
            f"bce={val_metrics['bce']:.4f} "
            f"acc={val_metrics['acc']:.4f} "
            f"f1={val_metrics['f1']:.4f} "
            f"auroc={val_metrics['auroc']:.4f} "
            f"(tp={int(val_metrics['tp'])} "
            f"tn={int(val_metrics['tn'])} "
            f"fp={int(val_metrics['fp'])} "
            f"fn={int(val_metrics['fn'])})"
        )

        if use_wandb:
            wb.log(
                {
                    "val/bce": val_metrics["bce"],
                    "val/acc": val_metrics["acc"],
                    "val/f1": val_metrics["f1"],
                    "val/auroc": val_metrics["auroc"],
                    "epoch": epoch,
                },
                step=global_step,
            )

        # ---- save best ----
        if val_metrics["bce"] < best_val_bce:
            best_val_bce = val_metrics["bce"]
            payload = {
                "model": model.state_dict(),
                "cfg": asdict(cfg),
                "state_dim": state_dim,
                "action_dim": action_dim,
                "epoch": epoch,
                "global_step": global_step,
                "val_metrics": val_metrics,
            }
            torch.save(payload, ckpt_path)
            print(f"✓ Saved new best checkpoint (val_bce={best_val_bce:.4f})")

        print("-" * 80)


if __name__ == "__main__":
    main()