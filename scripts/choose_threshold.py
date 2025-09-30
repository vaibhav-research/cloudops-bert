#!/usr/bin/env python
import json, argparse, numpy as np, torch
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_data(path):
    y, texts = [], []
    with open(path, "r") as f:
        for line in f:
            j = json.loads(line)
            texts.append(j["text"]); y.append(int(j["label"]))
    return np.array(y), texts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id_or_dir", required=True, help="HF repo or local dir")
    ap.add_argument("--subfolder", default=None, help="hdfs or bgl (HF only)")
    ap.add_argument("--test_jsonl", required=True)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()

    y_true, texts = load_data(args.test_jsonl)

    if args.subfolder:
        tok = AutoTokenizer.from_pretrained(args.model_id_or_dir, subfolder=args.subfolder, use_fast=True)
        mdl = AutoModelForSequenceClassification.from_pretrained(args.model_id_or_dir, subfolder=args.subfolder).eval()
    else:
        tok = AutoTokenizer.from_pretrained(args.model_id_or_dir, use_fast=True)
        mdl = AutoModelForSequenceClassification.from_pretrained(args.model_id_or_dir).eval()

    if torch.backends.mps.is_available(): mdl.to("mps")
    elif torch.cuda.is_available(): mdl.to("cuda")

    ps=[]
    with torch.no_grad():
        for i in range(0, len(texts), args.batch_size):
            batch=texts[i:i+args.batch_size]
            X=tok(batch, return_tensors="pt", truncation=True, max_length=args.max_length, padding=True)
            if mdl.device.type!="cpu": X={k:v.to(mdl.device) for k,v in X.items()}
            probs=torch.softmax(mdl(**X).logits, dim=-1)[:,1].detach().cpu().numpy()
            ps.append(probs)
    p=np.concatenate(ps)

    best=(0.0,0.0,0.0,0.5)
    for t in np.linspace(0.05,0.95,19):
        yhat=(p>=t).astype(int)
        prec,rec,f1,_=precision_recall_fscore_support(y_true, yhat, average="binary", zero_division=0)
        if f1>best[2]: best=(prec,rec,f1,t)
    print({"best_prec":best[0],"best_rec":best[1],"best_f1":best[2],"best_thr":best[3]})

if __name__ == "__main__":
    main()
