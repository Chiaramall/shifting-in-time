import itertools
import unittest

import numpy as np
from frbo import finite_rank_biased_overlap as frbo
from rbo_reference import rank_biased_overlap as rbo


class FRBOTests(unittest.TestCase):
    def test_extremes_and_numerical_stability(self):
        for p in [0.01, 0.9, 1 - 1e-15]:
            self.assertEqual(frbo([1, 2, 3], [1, 2, 3], p), 1)
            self.assertEqual(frbo([1, 2], [3, 4], p), 0)
            self.assertEqual(frbo([1], [1], p), 1)

    def test_hand_calculation_not_extrapolated_rbo(self):
        self.assertAlmostEqual(frbo([1, 2], [2, 1]), 0.9 / 1.9)
        self.assertAlmostEqual(frbo([1, 2], [1, 3]), 1.45 / 1.9)
        self.assertNotAlmostEqual(frbo([1, 2], [2, 1]), rbo([1, 2], [2, 1]))

    def test_all_short_permutations_against_formula(self):
        for a in itertools.permutations(range(4), 3):
            for b in itertools.permutations(range(4), 3):
                expected = (
                    0.1
                    / (1 - 0.9**3)
                    * sum(
                        0.9**i * len(set(a[: i + 1]) & set(b[: i + 1])) / (i + 1)
                        for i in range(3)
                    )
                )
                self.assertAlmostEqual(frbo(a, b), expected)
                self.assertAlmostEqual(frbo(a, b), frbo(b, a))

    def test_top_weight_and_numpy(self):
        a = np.arange(10)
        top = a.copy()
        top[[0, 1]] = top[[1, 0]]
        bottom = a.copy()
        bottom[[-1, -2]] = bottom[[-2, -1]]
        self.assertLess(frbo(a, top), frbo(a, bottom))

    def test_invalid_inputs(self):
        for a, b, p in [
            ([], [], 0.9),
            ([1], [1, 2], 0.9),
            ([1, 1], [1, 2], 0.9),
            ([1], [1], 0),
            ([1], [1], 1),
            ([1], [1], float("nan")),
        ]:
            with self.assertRaises(ValueError):
                frbo(a, b, p)


if __name__ == "__main__":
    unittest.main()
