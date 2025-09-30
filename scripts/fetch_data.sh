#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/fetch_data.sh            # fetch HDFS + BGL
#   DATASETS="hdfs" bash scripts/fetch_data.sh
#   DATASETS="bgl"  bash scripts/fetch_data.sh

DATASETS="${DATASETS:-hdfs bgl}"

for ds in $DATASETS; do
  echo "⬇️  Fetching dataset: $ds"
  python src/fetch_data.py \
    --dataset "$ds" \
    --out "data/processed/$ds"
  echo "✅ $ds ready at data/processed/$ds"
done

echo "All requested datasets fetched."
