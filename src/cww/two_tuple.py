"""2-tuple linguistic representation.

Reference
---------
Herrera, F., & Martinez, L. (2000). A 2-tuple fuzzy linguistic representation
    model for computing with words. IEEE TFS 8(6), 746-752.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple


class TwoTuple(NamedTuple):
    """(s_i, alpha): index i of term s_i in S = {s_0..s_g}, alpha in [-0.5, 0.5)."""

    index: int
    alpha: float


def delta(beta: float, g: int) -> TwoTuple:
    """Delta(beta) = (s_i, alpha), i = round(beta), alpha = beta - i, for beta in [0, g].

    Halves round up (i = floor(beta + 0.5)), so alpha lies in [-0.5, 0.5).
    """
    if g < 1:
        raise ValueError('Term set needs g >= 1 (at least 2 terms)')
    if not -1e-9 <= beta <= g + 1e-9:
        raise ValueError(f'beta={beta} outside [0, {g}]')
    beta = min(max(beta, 0.0), float(g))  # absorb floating-point round-off only
    i = min(int(math.floor(beta + 0.5)), g)
    return TwoTuple(i, beta - i)


def delta_inv(t: TwoTuple) -> float:
    """Delta^{-1}(s_i, alpha) = i + alpha."""
    return t.index + t.alpha


def mean(tuples: Sequence[TwoTuple], g: int) -> TwoTuple:
    """Arithmetic mean: Delta((1/n) sum_j Delta^{-1}(r_j, alpha_j))."""
    if len(tuples) == 0:
        raise ValueError('Need at least one 2-tuple')
    return delta(sum(delta_inv(t) for t in tuples) / len(tuples), g)


def weighted_mean(tuples: Sequence[TwoTuple], weights: Sequence[float], g: int) -> TwoTuple:
    """Weighted mean: Delta(sum_j w_j Delta^{-1}(r_j, alpha_j) / sum_j w_j)."""
    if len(tuples) == 0 or len(tuples) != len(weights):
        raise ValueError('Need equally many (>= 1) tuples and weights')
    if any(w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError('Weights must be non-negative with a positive sum')
    total = sum(w * delta_inv(t) for t, w in zip(tuples, weights))
    return delta(total / sum(weights), g)
