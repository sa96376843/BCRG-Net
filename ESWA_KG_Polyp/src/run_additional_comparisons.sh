#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
for method in unetpp deeplab; do
  for seed in 13 29 47; do
    python3 experiment.py --dataset kvasir --method "$method" --seed "$seed" --epochs 12
  done
done
for method in unet attention attention_original unetpp deeplab bcskg; do
  for seed in 13 29 47; do
    tag="kvasir_${method}_s${seed}_r0.05_b0.1"
    python3 extended_evaluation.py --tag "$tag" --mode robustness
  done
done
printf 'ADDITIONAL_COMPARISONS_COMPLETE\n'
