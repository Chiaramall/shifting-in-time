#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results
for dataset in lastfm1k movielens10m; do
  for model in SASRec BERT4Rec gSASRec GRU4Rec LightSANs DuoRec CL4SRec eSASRec BSARec; do
    python experiments/train_baseline_budget.py \
      --config "config/${dataset}_train.yml" --model "$model" \
      --output-dir "checkpoints/${dataset}/${model}"
  done
  python experiments/catalog_robustness.py \
    --config "config/${dataset}_eval.yml" \
    --output-dir "results/catalog/${dataset}" --device "${DEVICE:-cuda}"
  test -f "results/catalog/${dataset}/SUCCESS"
done
python experiments/report_catalog_ndcg.py \
  --source results/catalog --output-dir results/tables
