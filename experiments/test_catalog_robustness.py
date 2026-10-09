import unittest

import numpy as np
from catalog_robustness import farthest_catalog, inject_catalog
from frbo import finite_rank_biased_overlap


class CatalogTests(unittest.TestCase):
    def test_whole_history_excluded_and_ties(self):
        x = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [-10.0, 0.0], [9.0, 0.0]])
        ids, d = farthest_catalog(x, [[0, 1, 1, 2]], 2)
        self.assertEqual(
            ids, [[3, 4]]
        )  # item 2 excluded although outside a hypothetical input window
        ids, _ = farthest_catalog(x, [[0]], 2)
        self.assertEqual(ids, [[2, 3]])

    def test_replacement_preserves_every_other_position(self):
        sequence = list(range(20))
        new = list(range(100, 106))
        for intensity, count in [(0.05, 1), (0.1, 2), (0.2, 4), (0.3, 6)]:
            for region, start in [
                ("old", 0),
                ("middle", 10 - count // 2),
                ("recent", 20 - count),
            ]:
                changed, removed, inserted, items = inject_catalog(
                    sequence, new, intensity, region
                )
                expected = sequence.copy()
                expected[start : start + count] = new[:count]
                self.assertEqual(changed, expected)
                self.assertEqual(removed, list(range(start, start + count)))
                self.assertEqual(inserted, removed)
                self.assertEqual(items, new[:count])
                self.assertEqual(len(changed), 20)

    def test_recent_replaces_recent_interactions(self):
        self.assertEqual(
            inject_catalog(list("ABCDEF"), list("XY"), 0.3, "recent")[0], list("ABCDXY")
        )

    def test_short_odd_sequences(self):
        for length in [2, 3, 5, 19]:
            for region in ["old", "middle", "recent"]:
                source = list(range(length))
                out, positions, _, new = inject_catalog(
                    source, list(range(100, 120)), 0.3, region
                )
                self.assertEqual(len(out), length)
                self.assertGreaterEqual(len(new), 1)
                self.assertLess(len(new), length)
                self.assertTrue(
                    all(
                        out[i] == source[i] for i in range(length) if i not in positions
                    )
                )

    def test_reject_invalid_inputs(self):
        for sequence, new, intensity, region in [
            ([1, 2], [2], 0.1, "recent"),
            ([1], [3], 0.3, "old"),
            ([1, 2], [3], 0, "old"),
            ([1, 2], [3], 0.3, "unknown"),
        ]:
            with self.assertRaises(ValueError):
                inject_catalog(sequence, new, intensity, region)

    def test_distance_reference(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(71, 13))
        hist = [[0, 1, 1, 5], [3, 7, 12]]
        ids, ds = farthest_catalog(x, hist, 6)
        for h, got, dist in zip(hist, ids, ds):
            d = np.linalg.norm(x - x[sorted(set(h))].mean(0), axis=1)
            expected = sorted(set(range(len(x))) - set(h), key=lambda i: (-d[i], i))[:6]
            self.assertEqual(got, expected)
            np.testing.assert_allclose(dist, d[got])

    def test_frbo(self):
        self.assertEqual(
            finite_rank_biased_overlap(list(range(10)), list(range(10))), 1
        )
        self.assertEqual(
            finite_rank_biased_overlap(list(range(10)), list(range(10, 20))), 0
        )


if __name__ == "__main__":
    unittest.main()
