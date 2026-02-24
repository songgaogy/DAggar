from typing import Dict, Tuple

import numpy as np
import torch


def binary_confusion(pred: np.ndarray, y: np.ndarray) -> Dict[str, int]:
    assert pred.shape == y.shape
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def f1_from_conf(conf: Dict[str, int]) -> float:
    tp, fp, fn = conf["tp"], conf["fp"], conf["fn"]
    denom = (2 * tp + fp + fn)
    return float(0.0 if denom == 0 else (2 * tp) / denom)


def accuracy_from_conf(conf: Dict[str, int]) -> float:
    tp, tn, fp, fn = conf["tp"], conf["tn"], conf["fp"], conf["fn"]
    total = tp + tn + fp + fn
    return float(0.0 if total == 0 else (tp + tn) / total)


def auroc_binary(probs: np.ndarray, y: np.ndarray) -> float:
    # Simple AUROC implementation without sklearn
    y = y.astype(np.int64)
    probs = probs.astype(np.float64)
    pos = probs[y == 1]
    neg = probs[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Mann–Whitney U statistic
    ranks = probs.argsort().argsort().astype(np.float64) + 1.0
    r_pos = ranks[y == 1].sum()
    n_pos = float(len(pos))
    n_neg = float(len(neg))
    u = r_pos - n_pos * (n_pos + 1.0) / 2.0
    return float(u / (n_pos * n_neg))


@torch.no_grad()
def eval_binary_logits(logits: torch.Tensor, y: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    probs = torch.sigmoid(logits).cpu().numpy()
    yy = y.cpu().numpy().astype(np.int64)
    pred = (probs >= threshold).astype(np.int64)

    conf = binary_confusion(pred, yy)
    acc = accuracy_from_conf(conf)
    f1 = f1_from_conf(conf)
    auroc = auroc_binary(probs, yy)
    bce = float(torch.nn.functional.binary_cross_entropy_with_logits(logits, y.float()).cpu().item())
    return {
        "acc": acc,
        "f1": f1,
        "auroc": auroc,
        "bce": bce,
        "tp": float(conf["tp"]),
        "tn": float(conf["tn"]),
        "fp": float(conf["fp"]),
        "fn": float(conf["fn"]),
    }