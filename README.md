# Shifting in Time: Probing Ranking Stability in Sequential Recommendation

Code for evaluating how the temporal position of unseen catalogue items affects
sequential recommendation. The implementation uses [WarpRec](https://arxiv.org/abs/2602.17442)
and includes SASRec, BERT4Rec, gSASRec, GRU4Rec, LightSANs, DuoRec, CL4SRec,
eSASRec, and BSARec.

## Setup

Use Python 3.12. Run commands from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
mkdir -p results
```

Place the prepared datasets at these paths; see [data/README.md](data/README.md)
for the CSV format:

```text
data/lastfm1k/user_events1k.csv
data/movielens10m/movielens10m.csv
```

LastFM uses the complete prepared CSV, without track subsampling. Datasets,
checkpoints, and experiment outputs are excluded from Git.

## Protocol

- Split each user's history chronologically: the latest 10% for testing, then
  the latest 10% of the remainder for validation. Train on the remaining events.
  Exclude validation and test interactions with unknown users or items.
- Train for 300 epochs with seed 42. Evaluate full-catalogue validation nDCG@10
  after each epoch and keep the best checkpoint; ties select the earlier epoch.
  Test data are not used for checkpoint selection.
- Keep model weights fixed during perturbation. Use the latest `L <= 20`
  training interactions as input and retain users with at least one test hit in
  the original Top-20. Each model's cohort is fixed across conditions.
- Compute the centroid of distinct items in the full training history. Select
  unseen catalogue items by decreasing Euclidean distance from that centroid,
  breaking ties by internal item ID. Do not use test relevance for selection.
- For intensity `alpha` in `{0.05, 0.10, 0.20, 0.30}`, replace
  `q = min(L - 1, max(1, floor(alpha * L + 0.5)))` interactions. Replace the
  first `q` positions for **old**, the last `q` for **recent**, or a middle block
  starting at zero-based index `floor(L/2) - floor(q/2)` for **middle**.
  Use the same new items in the same order across the three regions. All other
  original interactions remain at their original positions.
- Mask the full original training history in every recommendation condition.
  Newly introduced items remain eligible for recommendation. Report mean
  FRBO@10 (`p = 0.9`) and binary nDCG@10 over the selected users.

## Train and evaluate

For one model:

```bash
python experiments/train_baseline_budget.py \
  --config config/lastfm1k_train.yml --model SASRec \
  --output-dir checkpoints/lastfm1k/SASRec

python experiments/catalog_robustness.py \
  --config config/lastfm1k_eval.yml --models SASRec \
  --output-dir results/catalog/lastfm1k --device cuda
```

`checkpoint.pt` contains the validation-selected weights. `last.pt` contains the
latest optimizer and random-generator state for resuming training. Evaluation
requires the training `SUCCESS` marker. Repeating the training command resumes
from `last.pt`; use a new output directory after changing data or parameters.
Checkpoints from the earlier final-epoch pipeline are not accepted.

For all nine models on both datasets:

```bash
bash scripts/run_experiments.sh
```

The script also generates the aggregate tables. To regenerate tables from saved
rankings without rerunning model inference:

```bash
python experiments/report_catalog_ndcg.py \
  --source results/catalog --output-dir results/tables
```

The evaluator exports all four intensities; the tables show 5% and 30%.
For CPU execution, set `general.device: cpu` in the YAML files and use
`--device cpu` for evaluation, or `DEVICE=cpu bash scripts/run_experiments.sh`.

## Files and tests

- `models/`: the nine model implementations and shared training objectives.
- `config/`: training and evaluation configurations for both datasets.
- `experiments/`: training, perturbation, reporting, and regression tests.
- `warprec/`: shared data, training, and evaluation infrastructure.

```bash
python -m unittest discover -s experiments -p 'test_*.py'
```

The tests check temporal replacement, FRBO, validation splits, checkpoint
selection, resume behavior, and training objectives. Full experiments with these
updated configurations have not been rerun to verify the paper's numerical
results. The supplied hyperparameters are fixed configurations; this repository
does not reconstruct the preliminary hyperparameter search.
