# eval.py
from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from utils import save_json

def compute_metrics(eval_pred) -> Dict[str, Any]:
    """
    Expects HF Trainer eval_pred: (logits, labels) or (predictions, labels).
    Returns a dict for Trainer logging.
    """
    logits, labels = eval_pred
    if isinstance(logits, tuple):  # safety
        logits = logits[0]
    probs = softmax_np(logits)
    preds = probs.argmax(axis=1)

    acc = accuracy_score(labels, preds)
    # For precision/recall/F1, compute on the positive class (1)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)
    f1   = f1_score(labels, preds, zero_division=0)

    # ROC-AUC requires probabilities of positive class and both classes present
    try:
        pos_scores = probs[:, 1]
        auc = roc_auc_score(labels, pos_scores)
    except Exception:
        auc = float("nan")

    cm = confusion_matrix(labels, preds, labels=[0,1]).tolist()

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "confusion_matrix": cm,
    }

def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)

def evaluate_and_save(trainer, split_name: str, out_dir: str) -> Dict[str, Any]:
    """
    Runs trainer.evaluate() and saves metrics to {out_dir}/metrics_{split_name}.json
    Returns the metrics dict.
    """
    metrics = trainer.evaluate()
    path = f"{out_dir}/metrics_{split_name}.json"
    save_json({k: float(v) if isinstance(v, (np.floating,)) else v for k, v in metrics.items()}, path)
    return metrics
