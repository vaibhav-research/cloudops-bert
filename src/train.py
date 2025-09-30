# train.py (excerpt)
import argparse, os
from transformers import AutoTokenizer, TrainingArguments
from utils import setup_logging, set_seed, get_device, ensure_dir, compute_class_weights, save_json
from dataset import build_datasets
from model import load_model, WeightedTrainer
from eval import compute_metrics, evaluate_and_save

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["hdfs","bgl"], required=True)
    p.add_argument("--model_name", type=str, required=True)              # e.g., models/distilbert-base-uncased
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--train_bs", type=int, default=16)
    p.add_argument("--eval_bs", type=int, default=32)
    p.add_argument("--weight_classes", action="store_true")
    p.add_argument("--cross_eval_on", choices=["hdfs","bgl"], default=None)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()
    setup_logging()
    set_seed(args.seed)
    device = get_device()

    # Resolve paths based on dataset selection (adapt to your layout)
    train_path = f"data/{args.dataset}/train.jsonl"
    eval_path  = f"data/{args.dataset}/test.jsonl"
    out_dir    = ensure_dir(args.output_dir)

    # Tokenizer & model
    tok  = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = load_model(args.model_name, num_labels=2).to(device)

    # Data
    train_ds, eval_ds, collator, y_train, y_eval = build_datasets(train_path, eval_path, tok, args.max_length)

    # Optional class weights
    class_weights = compute_class_weights(y_train) if args.weight_classes else None

    # Training args
    targs = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",          # set to "tensorboard" if you use TB
    )

    trainer = WeightedTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tok,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    trainer.train()

    # In-domain evaluation
    in_metrics = evaluate_and_save(trainer, split_name="in_domain", out_dir=out_dir)

    # Cross-evaluation (optional)
    if args.cross_eval_on:
        cross_eval_path = f"data/{args.cross_eval_on}/test.jsonl"
        _, cross_ds, _, _, _ = build_datasets(train_path=None, eval_path=cross_eval_path, tokenizer=tok, max_length=args.max_length)
        trainer.eval_dataset = cross_ds
        cross_metrics = evaluate_and_save(trainer, split_name=f"cross_on_{args.cross_eval_on}", out_dir=out_dir)
        # restore trainer eval_dataset
        trainer.eval_dataset = eval_ds

    # Save final model for inference/push
    trainer.save_model(out_dir)    # saves model + tokenizer (if attached)
    tok.save_pretrained(out_dir)

if __name__ == "__main__":
    main()
