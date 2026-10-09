"""Regression tests for temporal splits and validation checkpoint selection."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.sparse import csr_matrix
from train_baseline_budget import evaluate_validation, train

from warprec.data.splitting import Splitter
from warprec.utils.enums import SplittingStrategies


class SplitTests(unittest.TestCase):
    def test_two_temporal_holdouts(self):
        events = pd.DataFrame(
            {
                "user_id": [0] * 100,
                "item_id": list(range(10)) * 10,
                "rating": [1.0] * 100,
                "timestamp": np.arange(100) + 1241363400,
            }
        )
        training, validation, test = Splitter().split_transaction(
            events,
            test_strategy=SplittingStrategies.TEMPORAL_HOLDOUT,
            test_ratio=0.1,
            val_strategy=SplittingStrategies.TEMPORAL_HOLDOUT,
            val_ratio=0.1,
        )
        self.assertEqual((len(training), len(validation), len(test)), (81, 9, 10))
        self.assertLess(training.timestamp.max(), validation.timestamp.min())
        self.assertLess(validation.timestamp.max(), test.timestamp.min())
        self.assertFalse(set(training.index) & set(validation.index))
        self.assertFalse(set(training.index) & set(test.index))

    def test_unknown_validation_and_test_items_are_filtered(self):
        events = pd.DataFrame(
            {
                "user_id": [0] * 100,
                "item_id": [0] * 81 + [1] * 9 + [2] * 10,
                "rating": [1.0] * 100,
                "timestamp": range(100),
            }
        )
        training, validation, test = Splitter().split_transaction(
            events,
            test_strategy=SplittingStrategies.TEMPORAL_HOLDOUT,
            test_ratio=0.1,
            val_strategy=SplittingStrategies.TEMPORAL_HOLDOUT,
            val_ratio=0.1,
        )
        self.assertEqual(len(training), 81)
        self.assertTrue(validation.empty)
        self.assertTrue(test.empty)

    def test_configs_match_protocol(self):
        for path in (Path(__file__).resolve().parents[1] / "config").glob("*.yml"):
            config = yaml.safe_load(path.read_text())
            self.assertEqual(
                config["splitter"]["validation_splitting"],
                {"strategy": "temporal_holdout", "ratio": 0.1},
            )
            self.assertEqual(config["reader"]["dtypes"]["timestamp_type"], "int64")
            for model in config["models"].values():
                self.assertEqual((model["max_seq_len"], model["epochs"]), (20, 300))


class TinyModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def train_step(self, batch, epoch):
        return self.weight.square()


def tiny_dataset(positive=1):
    matrix = csr_matrix(([1.0], ([0], [positive])), shape=(1, 12))
    return SimpleNamespace(
        get_inverse_mappings=lambda: ({0: "user"}, {i: str(i) for i in range(12)}),
        info=lambda: {},
        train_set=None,
        train_session=SimpleNamespace(
            _cached_dataset={}, _cached_user_histories={0: [0]}
        ),
        eval_set=SimpleNamespace(get_sparse=lambda: matrix),
    )


class CheckpointTests(unittest.TestCase):
    def test_best_epoch_survives_resume_and_ties(self):
        config = SimpleNamespace(
            models={"Tiny": {"epochs": 3, "batch_size": 1}},
            general=SimpleNamespace(device="cpu"),
            evaluation=SimpleNamespace(batch_size=1),
        )
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.ones(2, 1)), batch_size=1
        )
        # Accessing test labels during training is a failure.
        dataset = tiny_dataset()
        dataset.eval_set = SimpleNamespace(
            get_sparse=lambda: self.fail("Test labels accessed")
        )
        validation = tiny_dataset()
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            with (
                patch(
                    "train_baseline_budget.model_registry.get_class",
                    return_value=TinyModel,
                ),
                patch(
                    "train_baseline_budget.standard_optimizer",
                    side_effect=lambda m: torch.optim.SGD(m.parameters(), lr=0.1),
                ),
                patch("train_baseline_budget.make_loader", return_value=loader),
                patch(
                    "train_baseline_budget.evaluate_validation",
                    side_effect=[0.8, 0.2, 0.8],
                ),
            ):
                train(
                    config,
                    dataset,
                    "Tiny",
                    out,
                    stop_after=2,
                    validation_dataset=validation,
                )
                self.assertFalse((out / "SUCCESS").exists())
                selected = torch.load(out / "checkpoint.pt", weights_only=False)
                resume = torch.load(out / "last.pt", weights_only=False)
                self.assertEqual(selected["epoch"], 1)
                self.assertEqual(resume["epoch"], 2)
                saved = selected["model_state"]["weight"].clone()
                train(config, dataset, "Tiny", out, validation_dataset=validation)
                selected = torch.load(out / "checkpoint.pt", weights_only=False)
                resume = torch.load(out / "last.pt", weights_only=False)
                self.assertEqual(selected["epoch"], 1)
                self.assertEqual(resume["epoch"], 3)
                torch.testing.assert_close(selected["model_state"]["weight"], saved)
                self.assertTrue((out / "SUCCESS").exists())

    def test_validation_uses_training_mask_and_validation_targets(self):
        class Ranker(torch.nn.Module):
            max_seq_len, n_items = 20, 12

            def predict(self, user_indices, user_seq, seq_len):
                self.observed = user_seq[0, : seq_len[0]].tolist()
                return torch.tensor(
                    [[100.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
                )

        model = Ranker()
        self.assertEqual(evaluate_validation(model, tiny_dataset(1), "cpu"), 1.0)
        self.assertEqual(model.observed, [0])
        self.assertFalse(model.training)


if __name__ == "__main__":
    unittest.main()
