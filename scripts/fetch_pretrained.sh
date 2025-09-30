#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-vaibhav2507/cloudops-bert}"
SUBFOLDER="${SUBFOLDER:-hdfs}"   # choose: hdfs | bgl
OUTDIR="${OUTDIR:-models/cloudops-bert-${SUBFOLDER}}"

mkdir -p "$OUTDIR"

python - <<PY
from transformers import AutoTokenizer, AutoModelForSequenceClassification
mid="${MODEL_ID}"
sub="${SUBFOLDER}"
out="${OUTDIR}"
tok = AutoTokenizer.from_pretrained(mid, subfolder=sub, use_fast=True)
mdl = AutoModelForSequenceClassification.from_pretrained(mid, subfolder=sub)
tok.save_pretrained(out)
mdl.save_pretrained(out)
print(f"✅ Downloaded {mid}/{sub} -> {out}")
PY
