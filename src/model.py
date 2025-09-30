# model.py
from typing import Optional, Dict, Any
import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, PreTrainedModel, Trainer

def load_model(model_name_or_path: str, num_labels: int = 2) -> PreTrainedModel:
    return AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        problem_type="single_label_classification"
    )

class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        if self.class_weights is not None and self.args.device.type != self.class_weights.device.type:
            self.class_weights = self.class_weights.to(self.args.device)

    def compute_loss(self, model: PreTrainedModel, inputs: Dict[str, Any], return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        # Standard cross-entropy with optional class weights
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights) if self.class_weights is not None else nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
