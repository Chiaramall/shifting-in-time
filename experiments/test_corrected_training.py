import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest

import torch
from semantic_training import SemanticCollator
from torch.nn import functional as F

from models.corrected_objectives import (
    calibrated_bce,
    contrastive_views,
    info_nce_logits,
)
from warprec.data.entities.train_structures.session_structures import SessionDataset


class CorrectedTrainingTests(unittest.TestCase):
    def test_nce_matches_full_reference(self):
        torch.manual_seed(42)
        a = torch.randn(4, 6, requires_grad=True)
        b = torch.randn(4, 6, requires_grad=True)
        logits, labels = info_nce_logits(a, b, 0.2, "dot")
        self.assertEqual(logits.shape, (8, 7))
        z = torch.cat([a, b])
        scores = z @ z.T / 0.2
        scores = scores.masked_fill(torch.eye(8, dtype=torch.bool), -torch.inf)
        reference = F.cross_entropy(scores, (torch.arange(8) + 4) % 8)
        torch.testing.assert_close(F.cross_entropy(logits, labels), reference)
        reference.backward()
        self.assertTrue(torch.isfinite(a.grad).all())
        normalized, _ = info_nce_logits(a.detach(), b.detach(), 0.2, "cos")
        self.assertFalse(torch.allclose(normalized, logits))

    def test_gbce_matches_author_transform(self):
        logits = torch.tensor(
            [[[0.1, -0.2, 0.8], [0.9, -1.0, 0.3]]],
            dtype=torch.float64,
            requires_grad=True,
        )
        valid = torch.tensor([[True, False]])
        t = 0.75
        n = 100
        beta = 1 - t + t * 2 / (n - 1)
        pos = logits[..., :1].sigmoid()
        transformed = torch.log(1 / (pos.pow(-beta) - 1))
        reference_logits = torch.cat([transformed, logits[..., 1:]], -1)
        labels = torch.zeros_like(reference_logits)
        labels[..., 0] = 1
        expected = (
            F.binary_cross_entropy_with_logits(
                reference_logits, labels, reduction="none"
            ).mean(-1)
            * valid
        ).sum()
        got = calibrated_bce(logits, valid, n, t)
        torch.testing.assert_close(got, expected)
        got.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(logits.grad[0, 1].abs().sum(), 0)
        self.assertNotEqual(
            got.item(), calibrated_bce(logits.detach(), valid, n, 0).item()
        )

    def test_semantic_sampling_excludes_self_except_singletons(self):
        base = SessionDataset(
            torch.arange(20).reshape(5, 4),
            torch.full((5,), 4),
            torch.tensor([1, 2, 1, 3, 2]),
        )
        sampler = SemanticCollator(base)
        ids = torch.arange(5)
        for _ in range(20):
            others = sampler.sample_indices(ids)
            self.assertTrue(
                torch.equal(base.positive_items[ids], base.positive_items[others])
            )
            self.assertTrue((others[[0, 1, 2, 4]] != ids[[0, 1, 2, 4]]).all())
            self.assertEqual(others[3], 3)

    def test_views_do_not_crop_or_shuffle_padding(self):
        x = torch.tensor([[0, 1, 2, 3, 9, 9], [4, 5, 9, 9, 9, 9], [6, 9, 9, 9, 9, 9]])
        lengths = torch.tensor([4, 2, 1])
        for _ in range(30):
            a, al, b, bl = contrastive_views(x, lengths, 9, 10)
            for v, l in [(a, al), (b, bl)]:
                for i, k in enumerate(l.tolist()):
                    self.assertTrue((v[i, k:] == 9).all())
                    self.assertFalse((v[i, :k] == 9).any())
                    self.assertTrue(
                        set(v[i, :k].tolist())
                        <= set(x[i, : lengths[i]].tolist()) | {10}
                    )
            self.assertTrue(torch.equal(a[2], x[2]))
            self.assertTrue(torch.equal(b[2], x[2]))


if __name__ == "__main__":
    unittest.main()


class CompactLoaderTests(unittest.TestCase):
    def test_windows_and_negative_filter(self):
        from semantic_training import CompactHistoryDataset, HistoryCollator

        d = CompactHistoryDataset({0: [0, 1, 2, 3, 4], 1: [5, 6]}, 4, 10)
        self.assertEqual(len(d), 3)
        self.assertEqual(d[0].tolist(), [0, 1, 2, 3])
        self.assertEqual(d[1].tolist(), [1, 2, 3, 4])
        self.assertEqual(d[2].tolist(), [5, 6, 10, 10])
        pos, neg = HistoryCollator(10, 3)([d[i] for i in range(3)])
        for i in range(3):
            for j in range(3):
                if pos[i, j + 1] == 10:
                    self.assertTrue((neg[i, j] == 10).all())
                else:
                    self.assertFalse(
                        set(neg[i, j].tolist()) & set(pos[i, : j + 2].tolist())
                    )


class BertMaskPositionTests(unittest.TestCase):
    def test_real_mask_at_position_zero_is_trained(self):
        from models.bert4rec import BERT4Rec

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.corrected_training = True
                self.padding_token_id = 3
                self.n_items = 3
                self.reg_weight = 0.0
                self.item_embedding = torch.nn.Embedding(5, 4)
                self.out_bias = torch.nn.Parameter(torch.zeros(4))
                self.hidden = torch.nn.Parameter(torch.randn(1, 2, 4))

            def forward(self, x):
                return self.hidden

            def _multi_hot_gather(self, x, indices):
                return x.gather(1, indices[:, :, None].expand(-1, -1, 4))

            def reg_loss(self, x):
                return x.sum() * 0

        m = Tiny()
        loss = BERT4Rec.train_step(
            m,
            (
                torch.tensor([[4, 1]]),
                torch.tensor([[2, 3]]),
                torch.tensor([[[0], [3]]]),
                torch.tensor([[0, 0]]),
            ),
        )
        expected = F.cross_entropy(
            m.hidden[:, 0] @ m.item_embedding.weight[:3].T, torch.tensor([2])
        )
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertGreater(m.hidden.grad[0, 0].abs().sum(), 0)
