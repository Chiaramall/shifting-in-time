# Shifting in Time: Probing Ranking Stability in Sequential Recommendation

Research code for studying how the temporal placement of unseen catalogue items
changes sequential recommendations. Evaluation keeps model parameters fixed and
compares original and perturbed Top-10 rankings with **FRBO@10** (`p = 0.9`) and
**nDCG@10**.

This repository contains the nine models evaluated on **LastFM-1K** and
**MovieLens-10M**: SASRec, BERT4Rec, gSASRec, GRU4Rec, LightSANs, DuoRec,
CL4SRec, eSASRec, and BSARec. All supplied configurations use
**`max_seq_len: 20`**. The value 50 in the manuscript's sequence-length description
is a typo, as confirmed by the authors.

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
data/lastfm1k/user_events_30000.csv
data/movielens10m/movielens10m.csv
```

Data and model weights are not distributed in this repository. Historical data
preprocessing is not fully reconstructible from the retained experiment code;
using a different LastFM subset is a new experimental run, not an exact
reproduction of the paper results.

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

The trainer resumes complete epochs from the same output directory. The evaluator
requires a completed training checkpoint and its `SUCCESS` marker. Use a new
output directory after changing weights, data, or evaluation parameters. Inspect
`BLOCKED.json` and `ERROR.json` when a model has no `SUCCESS` marker.

After evaluating all models, regenerate the effectiveness and stability tables:

```bash
python experiments/report_catalog_ndcg.py \
  --source results/catalog --output-dir results/tables
```

The table reporter compares 5% and 30%; the evaluator exports all four intensities.
A quick training preflight runs up to two batches per model and saves separate
smoke outputs, which must not be used for the reported evaluation:

```bash
python experiments/train_baseline_budget.py \
  --config config/lastfm1k_train.yml --preflight-all \
  --output-dir results/smoke/lastfm1k
```

## Implemented protocol

1. Split each user's interactions chronologically, holding out the last 10% for
   testing. The supplied fixed-budget configurations have no validation split.
2. Train on original interactions, with seed 42. The preserved configurations use
   **50 training epochs** and keep the final checkpoint. This epoch count is
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

**Manuscript alignment:** the supplied draft describes nearest-item selection and
validation-selected checkpoints, while the archived implementation uses farthest
items and final-epoch checkpoints. The code also explicitly replaces occurrences
to preserve length. These choices are preserved here rather than silently changing
the experiment. The MovieLens qualitative example uses 20% (four items), although
the draft caption groups both examples under 30%; LastFM uses 30% (six items).

## Qualitative examples

The archived **SASRec examples** show how recent catalogue perturbations change
recommendation content:

| Dataset | Input window | Perturbation | Change in the Top-10 |
| --- | --- | --- | --- |
| MovieLens-10M | 20 Horror interactions | 4 new items (20%) | Horror: 10 → 0; Drama: 0 → 9 |
| LastFM | 20 Beatles interactions | 6 new items (30%) | Beatles: 5 → 0; Chemistry: 0 → 8 |

Both recent-position examples have **FRBO@10 = 0**. These are illustrative cases
selected after evaluation, not a representative sample.

Read the [examples and full rankings](paper/qualitative/examples.md).
The [example records](paper/qualitative/table_evidence.json) preserve titles,
rankings, and interventions. No checkpoints are needed to inspect these artifacts.
Qualitative search/generation scripts are intentionally not included.

The examples document the archived intensities and FRBO values, including the
remaining differences from the supplied manuscript's caption and middle-position
scores; see the notes at the end of the examples.

## Repository layout

| Path | Contents |
| --- | --- |
| `config/` | Train/evaluation YAML for the two datasets; every window is 20 |
| `experiments/` | Training, catalogue perturbations, shared helpers, metrics, tests |
| `models/` | All nine paper model implementations and shared training objectives |
| `warprec/` | Shared data, training, and evaluation infrastructure |
| `paper/qualitative/` | Two archived SASRec examples and JSON evidence |




## Tests

```bash
python -m unittest discover -s experiments -p 'test_*.py'
```

Tests cover the finite ranking metric, catalogue selection/insertion, and corrected
training objectives. See [NOTICE.md](NOTICE.md) for upstream attribution and
licensing status.
