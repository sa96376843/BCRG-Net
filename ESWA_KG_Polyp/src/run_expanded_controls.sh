#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
for seed in 13 29 47; do
  tag="results/kvasir_unet_s${seed}_r0.05_b0.1_base10/result.json"
  if [ ! -f "$tag" ]; then
    python3 experiment.py --dataset kvasir --method unet --base 10 --seed "$seed" --epochs 12
  fi
done
for seed in 13 29 47; do
  tag="results/kvasir_bcskg_s${seed}_r0.05_b0.1_branch0.0/result.json"
  if [ ! -f "$tag" ]; then
    python3 experiment.py --dataset kvasir --method bcskg --lambda-branch 0 --seed "$seed" --epochs 12
  fi
done
for seed in 13 29 47; do
  tag="kvasir_unet_s${seed}_r0.05_b0.1_base10"
  if [ ! -f "results/${tag}/robustness.json" ]; then
    python3 extended_evaluation.py --tag "$tag" --mode robustness
  fi
done
printf '%s\n' EXPANDED_CONTROLS_COMPLETE
