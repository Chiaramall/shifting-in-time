# Shifting in Time: Probing Ranking Stability in Sequential Recommendation

Research code for studying how the temporal placement of unseen catalogue items
changes sequential recommendations. Evaluation keeps model parameters fixed and
compares original and perturbed Top-10 rankings with **FRBO@10** (`p = 0.9`) and
**nDCG@10**.

This repository contains the nine models evaluated on LastFM-1K and
MovieLens-10M: SASRec, BERT4Rec, gSASRec, GRU4Rec, LightSANs, DuoRec,
CL4SRec, eSASRec, and BSARec. All models are trained using the **[WarpRec framework](https://arxiv.org/abs/2602.1744)**, with supplied configurations using max_seq_len: 20. Model
parameters are kept fixed during the subsequent temporal perturbation experiments.

## Installation

Use Python 3.12. Run all commands from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
mkdir -p results
```

The YAML files use CUDA by default. For CPU runs, set `general.device: cpu` in
the training and evaluation YAML files and pass `--device cpu` to the evaluator.
Graph-model dependencies and experiment-tracking services are not required.

## Data

Prepare the CSV files described in [data/README.md](data/README.md):

```text
data/lastfm1k/user_events1k.csv
data/movielens10m/movielens10m.csv
```


## Train and evaluate

Train one model (the output path must match `meta.load_from` in the evaluation
configuration):

```bash
python experiments/train_baseline_budget.py \
  --config config/lastfm1k_train.yml --model SASRec \
  --output-dir checkpoints/lastfm1k/SASRec

python experiments/catalog_robustness.py \
  --config config/lastfm1k_eval.yml --models SASRec \
  --output-dir results/catalog/lastfm1k --device cuda
```

For all nine models and both datasets:

```bash
bash scripts/run_experiments.sh
```


After evaluating all models, regenerate the effectiveness and stability tables:

```bash
python experiments/report_catalog_ndcg.py \
  --source results/catalog --output-dir results/tables
```

The table reporter compares 5% and 30%; the evaluator exports all four intensities.


## Implemented protocol

1. Split each user's interactions chronologically, holding out the last 10% for
   testing. The supplied fixed-budget configurations have no validation split.
2. Train on original interactions, with seed 42. The preserved configurations use
   **300 training epochs** and keep the final checkpoint. This epoch count is
   separate from the **20-item** input window.
3. Select a fixed, model-specific cohort with at least one relevant item in the
   original Top-20. Evaluate against the original Top-10.
4. Compute the centroid of distinct items in the full training history. Rank unseen
   catalogue items by **decreasing Euclidean distance**, breaking ties by internal
   item ID. Item selection is model-specific and does not filter against test items.
5. For each visible sequence of length `L`, replace
   `min(L-1, max(1, floor(alpha*L + 0.5)))` occurrences farthest from the centroid
   with unseen catalogue items. Use `alpha = 0.05, 0.10, 0.20, 0.30`.
6. Insert the same selected block at the beginning, middle, or end. Preserve
   sequence length and the relative order of retained interactions. For `L=20`,
   the affected counts are 1, 2, 4, and 6.
7. Keep the original full-training-history recommendation mask fixed. Injected
   items remain eligible for recommendation. Average FRBO and binary nDCG per user.


