"""Extrapolated RBO reference used only for metric regression tests."""


def rank_biased_overlap(first, second, p=0.9):
    """Extrapolated RBO for equal-length, nonempty rankings without duplicates.

    Webber et al. (2010), Eq. 23: (1-p) sum_d A_d p**(d-1) + A_k p**k.
    Input order is the returned recommendation order (ties already resolved).
    """
    if not 0 < p < 1:
        raise ValueError("RBO persistence p must lie in (0, 1)")
    if not first or len(first) != len(second):
        raise ValueError("RBO requires nonempty rankings of equal length")
    if len(set(first)) != len(first) or len(set(second)) != len(second):
        raise ValueError("RBO rankings must contain unique items")
    seen_first, seen_second = set(), set()
    score = 0.0
    for depth, (a, b) in enumerate(zip(first, second), start=1):
        seen_first.add(a)
        seen_second.add(b)
        agreement = len(seen_first & seen_second) / depth
        score += (1 - p) * agreement * p ** (depth - 1)
    return min(1.0, max(0.0, score + agreement * p ** len(first)))
