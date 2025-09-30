# utils.py
import os, json, random, logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
import torch

LOGGER = logging.getLogger("cloudopsbert")

def setup_logging(level=logging.INFO):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )
    LOGGER.info("Logging initialized.")

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def now_str() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_json(obj: Dict[str, Any], path: str):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def compute_class_weights(labels: List[int]) -> torch.Tensor:
    """
    Returns weights tensor [w_neg, w_pos] for binary classification.
    Heavier weight for minority class to counter imbalance.
    """
    labels = np.array(labels)
    n_pos = (labels == 1).sum()
    n_neg = (labels == 0).sum()
    total = n_pos + n_neg
    # Inverse-frequency with smoothing
    w_pos = total / (2.0 * max(1, n_pos))
    w_neg = total / (2.0 * max(1, n_neg))
    weights = torch.tensor([w_neg, w_pos], dtype=torch.float)
    LOGGER.info(f"Class weights: neg={w_neg:.4f}, pos={w_pos:.4f} (n_neg={n_neg}, n_pos={n_pos})")
    return weights
