#!/usr/bin/env bash
set -euo pipefail
# Usage:
#   bash scripts/predict_line.sh "LOG LINE" [SUBFOLDER] [MODEL_ID_OR_DIR]
# Defaults: SUBFOLDER=hdfs, MODEL_ID_OR_DIR=vaibhav2507/cloudops-bert

TEXT="${1:?Provide a log line in quotes}"
SUBFOLDER="${2:-hdfs}"  # hdfs | bgl
MODEL="${3:-vaibhav2507/cloudops-bert}"
THRESH="${THRESH:-0.5}"

# If MODEL points to a local dir (e.g., models/cloudops-bert-hdfs), skip subfolder
if [ -d "$MODEL" ]; then
  python src/predict.py --model_dir "$MODEL" --text "$TEXT" --threshold "$THRESH"
else
  # HF repo with subfolder
  python src/predict.py --model_dir "$MODEL" --text "$TEXT" --threshold "$THRESH" --subfolder "$SUBFOLDER"
fi
