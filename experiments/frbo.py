"""Finite RBO used by Betello et al. (ECIR 2024).

Reference: https://github.com/federicosiciliano/finite_rank_biased_rbo/blob/main/frbo.py
Normalized finite-prefix formula, appropriate to top-k with catalog size >= 2*k.
This is not the extrapolated RBO_EXT formula. Rankings are ordered, unique items.
"""

import math


def finite_rank_biased_overlap(first, second, p=0.9):
    """Return FRBO for equal-length, nonempty finite rankings.

    FRBO = (1-p)/(1-p**k) * sum(p**(d-1) * |S[:d]&T[:d]|/d).
    Normalizing geometric weights directly avoids cancellation when p is near 1.
    Reject incomplete/duplicate rankings rather than silently changing depth.
    """
    if not math.isfinite(p) or not 0 < p < 1:
        raise ValueError("FRBO persistence p must lie in (0, 1)")
    k = len(first)
    if not k or len(second) != k:
        raise ValueError("FRBO requires nonempty rankings of equal length")
    if len(set(first)) != k or len(set(second)) != k:
        raise ValueError("FRBO rankings must contain unique items")
    a, b = set(), set()
    terms, weights = [], []
    for d, (x, y) in enumerate(zip(first, second), 1):
        a.add(x)
        b.add(y)
        weight = p ** (d - 1)
        weights.append(weight)
        terms.append(weight * len(a & b) / d)
    return min(1.0, max(0.0, math.fsum(terms) / math.fsum(weights)))
