import torch
import numpy as np
from sklearn.metrics import roc_auc_score

def evaluate_pu_metrics(preds: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5):
    """
    preds: Probabilities after Sigmoid
    labels: 1.0 for Expert, 0.0 for Unlabeled
    """
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    expert_mask = (labels_np == 1.0)
    unlabeled_mask = (labels_np == 0.0)
    
    expert_preds = preds_np[expert_mask]
    unlabeled_preds = preds_np[unlabeled_mask]
    
    # 1. Expert Recall (How well do we recognize good states?)
    expert_recall = np.mean(expert_preds > threshold)
    
    # 2. Unlabeled Positive Rate (Estimated success rate in rollouts)
    unlabeled_positive_rate = np.mean(unlabeled_preds > threshold)
    
    # 3. AUC (Expert vs Unlabeled)
    auc = roc_auc_score(labels_np, preds_np)
    
    return {
        "expert_recall": expert_recall,
        "unlabeled_positive_rate": unlabeled_positive_rate,
        "pu_auc": auc
    }