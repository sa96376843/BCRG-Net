#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
for variant in dual graph_only relation boundary; do
  for seed in 13 29 47; do
    python3 ablation_experiment.py --variant "$variant" --seed "$seed" --epochs 12
  done
done
for seed in 13 29 47; do
  python3 experiment.py --dataset kvasir --method attention_original --seed "$seed" --epochs 12
 done
for bd in 0 0.05 0.2; do
  python3 experiment.py --dataset kvasir --method bcskg --seed 13 --epochs 12 --lambda-bd "$bd"
done
python3 experiment.py --dataset kvasir --method unet --seed 13 --epochs 12 --size 128
python3 experiment.py --dataset kvasir --method bcskg --seed 13 --epochs 12 --size 128
printf 'REVISION_EXPERIMENTS_COMPLETE\n'
