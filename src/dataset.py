# dataset.py
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase
from utils import load_jsonl

class CloudOpsLogDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]], tokenizer: PreTrainedTokenizerBase, max_length: int = 256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        label = int(item["label"])
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt"
        )
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        enc["labels"] = torch.tensor(label, dtype=torch.long)
        return enc

def build_datasets(
    train_path: str,
    eval_path: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 256
) -> Tuple[CloudOpsLogDataset, CloudOpsLogDataset, DataCollatorWithPadding, List[int], List[int]]:
    train_json = load_jsonl(train_path)
    eval_json  = load_jsonl(eval_path)
    train_ds = CloudOpsLogDataset(train_json, tokenizer, max_length)
    eval_ds  = CloudOpsLogDataset(eval_json, tokenizer, max_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    y_train = [int(x["label"]) for x in train_json]
    y_eval  = [int(x["label"]) for x in eval_json]
    return train_ds, eval_ds, collator, y_train, y_eval
