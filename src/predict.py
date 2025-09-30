# src/predict.py
import argparse, json, os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load(model_dir, base_tokenizer="models/distilbert-base-uncased", subfolder=None, anomaly_index_override=None):
    """
    Loads tokenizer and model from either:
      - local directory (subfolder ignored), or
      - Hugging Face repo id with optional subfolder (e.g., vaibhav2507/cloudops-bert + hdfs/bgl)
    Returns: tok, mdl, id2label (dict[int->str]), anomaly_idx (int)
    """
    use_subfolder = subfolder if (subfolder and not os.path.isdir(model_dir)) else None

    # tokenizer
    try:
        if use_subfolder:
            tok = AutoTokenizer.from_pretrained(model_dir, subfolder=use_subfolder, use_fast=True)
        else:
            tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    except Exception:
        # fallback to a local base tokenizer if model_dir doesn't include tokenizer files
        tok = AutoTokenizer.from_pretrained(base_tokenizer, use_fast=True)

    # model
    if use_subfolder:
        mdl = AutoModelForSequenceClassification.from_pretrained(model_dir, subfolder=use_subfolder)
    else:
        mdl = AutoModelForSequenceClassification.from_pretrained(model_dir)

    mdl.eval()
    if torch.cuda.is_available():
        mdl.to("cuda")
    elif torch.backends.mps.is_available():
        mdl.to("mps")

    # id2label may be missing or generic (LABEL_0/LABEL_1). Normalize to int keys.
    id2label = getattr(mdl.config, "id2label", None)
    if not id2label:
        # safe default names if missing
        id2label = {0: "LABEL_0", 1: "LABEL_1"}
    else:
        id2label = {int(k): v for k, v in id2label.items()}

    # Decide which index is "anomaly"
    if anomaly_index_override is not None:
        anomaly_idx = int(anomaly_index_override)
    else:
        labels_lower = [id2label[i].lower() for i in range(mdl.config.num_labels)]
        anomaly_idx = labels_lower.index("anomaly") if "anomaly" in labels_lower else min(1, mdl.config.num_labels - 1)

    return tok, mdl, id2label, anomaly_idx

def predict_line(tok, mdl, id2label, text, max_length=256):
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=max_length)
    if mdl.device.type != "cpu":
        inputs = {k: v.to(mdl.device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = mdl(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].tolist()
    # map to label names
    probs_by_label = {id2label[i]: float(probs[i]) for i in range(len(probs))}
    pred_idx = int(torch.argmax(logits, dim=-1).item())
    pred_label = id2label[pred_idx]
    return probs_by_label, pred_label

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="HF repo id or local directory")
    ap.add_argument("--subfolder", default=None, help="HF subfolder: hdfs or bgl (ignored for local dirs)")
    ap.add_argument("--text", help="Single log line")
    ap.add_argument("--file", help="Path to file with one log line per row")
    ap.add_argument("--threshold", type=float, default=0.5, help="Anomaly threshold on P(anomaly)")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--jsonl_out", help="Write JSONL predictions to this path")
    ap.add_argument("--anomaly_index", type=int, choices=[0,1], default=None,
                    help="Force which logit index is 'anomaly' (overrides config/heuristic)")
    args = ap.parse_args()

    tok, mdl, id2label, anomaly_idx = load(
        args.model_dir,
        subfolder=args.subfolder,
        anomaly_index_override=args.anomaly_index
    )

    def anomaly_prob_from_dict(probs_dict):
        # Prefer explicit "anomaly" key if present (any case)
        for k, v in probs_dict.items():
            if k.lower() == "anomaly":
                return float(v)
        # Fallback: use anomaly_idx
        ordered = [probs_dict[id2label[i]] for i in range(len(id2label))]
        return float(ordered[anomaly_idx])

    def emit(line_text: str):
        probs, _ = predict_line(tok, mdl, id2label, line_text, max_length=args.max_length)
        p_anom = anomaly_prob_from_dict(probs)
        label = "ANOMALY" if p_anom >= args.threshold else "NORMAL"
        out = {"text": line_text, "label": label, "anomaly": p_anom}
        out.update(probs)  # include raw probs as well
        return out

    results = []
    if args.text:
        results = [emit(args.text)]
    elif args.file:
        with open(args.file) as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                results.append(emit(line))
    else:
        ap.error("Provide --text or --file")

    if args.jsonl_out:
        with open(args.jsonl_out, "w") as w:
            for r in results:
                w.write(json.dumps(r) + "\n")
    else:
        for r in results:
            print(json.dumps(r, indent=2))

if __name__ == "__main__":
    main()
