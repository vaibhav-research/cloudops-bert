#!/usr/bin/env bash
set -euo pipefail
# Usage:
#   bash scripts/predict_file.sh INPUT.txt [SUBFOLDER] [MODEL_ID_OR_DIR] [OUT.jsonl]
# Defaults: SUBFOLDER=hdfs, MODEL_ID_OR_DIR=vaibhav2507/cloudops-bert, OUT=preds/predictions.jsonl

INPUT="${1:?Path to text file (one log per line)}"
SUBFOLDER="${2:-hdfs}"  # hdfs | bgl
MODEL="${3:-vaibhav2507/cloudops-bert}"
OUT="${4:-preds/predictions.jsonl}"
THRESH="${THRESH:-0.5}"

mkdir -p "$(dirname "$OUT")"

if [ -d "$MODEL" ]; then
  python src/predict.py --model_dir "$MODEL" --file "$INPUT" --threshold "$THRESH" --jsonl_out "$OUT"
else
  python src/predict.py --model_dir "$MODEL" --subfolder "$SUBFOLDER" --file "$INPUT" --threshold "$THRESH" --jsonl_out "$OUT"
fi

echo "✅ Wrote predictions to $OUT"
