#!/usr/bin/env sh
# Usage: sh run_final_main.sh DATASET METHOD
set -eu
DATASET=$1
METHOD=$2
cd "$(dirname "$0")"
for s in 13 29 47; do
  python3 experiment.py --dataset "$DATASET" --method "$METHOD" --seed "$s" --epochs 12
done
