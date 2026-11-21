"""Interval Agreement Approach (IAA) and the rater-panel encoder.

Reference
---------
Wagner, C., Miller, S., Garibaldi, J. M., Anderson, D. T., & Havens, T. C.
    (2015). From interval-valued data to general type-2 fuzzy sets.
    IEEE TFS 23(2), 248-269.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .sets import DEFAULT_X, IT2Set, T1Set, TOL, _check_x


def _indicator(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """1[x in [lo, hi]] on the sampled universe (closed interval)."""
    return ((x >= lo - TOL) & (x <= hi + TOL)).astype(float)


def _check_inside(x: np.ndarray, lo: float, hi: float) -> None:
    # Refuse silent truncation: an interval outside X would lose membership mass.
    if lo < x[0] - TOL or hi > x[-1] + TOL:
        raise ValueError(f'Interval [{lo}, {hi}] exceeds the universe [{x[0]}, {x[-1]}]')


def iaa_t1(intervals: Sequence[tuple[float, float]], x=None) -> T1Set:
    """Type-1 IAA (Wagner et al. 2015).

    mu(x) = (1/n) * sum_{i=1..n} 1[x in [l_i, r_i]],
    i.e. the fraction of the n intervals that contain x.
    """
    x = DEFAULT_X if x is None else _check_x(x)
    if len(intervals) == 0:
        raise ValueError('Need at least one interval')
    mu = np.zeros_like(x)
    for lo, hi in intervals:
        if not np.isfinite(lo) or not np.isfinite(hi) or lo > hi:
            raise ValueError(f'Invalid interval [{lo}, {hi}]')
        _check_inside(x, lo, hi)
        mu += _indicator(x, lo, hi)
    return T1Set(x, mu / len(intervals))


def score_interval(score: float, half_width: float = 0.5) -> tuple[float, float]:
    """Likert bin of a score: [s - half_width, s + half_width]."""
    if not np.isfinite(score):
        raise ValueError('Score must be finite; filter missing scores before encoding')
    if half_width < 0:
        raise ValueError('half_width must be >= 0')
    return float(score) - half_width, float(score) + half_width


def encode_panel(runs_per_rater: Sequence[Sequence[float]], half_width: float = 0.5,
                 x=None) -> IT2Set:
    """IT2 encoder for a panel of J raters, rater j having runs r_j1..r_jm_j.

    Each score s becomes the Likert bin [s - h, s + h] (h = ``half_width``).
      judge interval  I_j = [min(r_j) - h, max(r_j) + h]
      UMF(x) = (1/J) sum_j 1[x in I_j]
      LMF(x) = (1/J) sum_j (1/m_j) sum_r 1[x in [r_jr - h, r_jr + h]]
    Every run bin lies inside I_j, so LMF <= UMF. m_j may differ between
    raters. When every rater has a single score (e.g. humans), LMF = UMF and
    the result is the type-1 IAA of the bins.
    """
    x = DEFAULT_X if x is None else _check_x(x)
    if len(runs_per_rater) == 0:
        raise ValueError('Need at least one rater')
    umf = np.zeros_like(x)
    lmf = np.zeros_like(x)
    for runs in runs_per_rater:
        runs = [float(r) for r in runs]
        if len(runs) == 0:
            raise ValueError('Each rater needs at least one score')
        bins = [score_interval(r, half_width) for r in runs]
        lo, hi = min(b[0] for b in bins), max(b[1] for b in bins)
        _check_inside(x, lo, hi)
        umf += _indicator(x, lo, hi)
        lmf += sum(_indicator(x, *b) for b in bins) / len(bins)
    n = len(runs_per_rater)
    return IT2Set(x, lmf / n, umf / n)


def encode_scores(scores: Sequence[float], half_width: float = 0.5, x=None) -> IT2Set:
    """Panel of single-score raters (e.g. humans): type-1 IAA embedded as IT2 (lmf = umf)."""
    return encode_panel([[s] for s in scores], half_width, x)


def vote_degree(k: int, n: int) -> float:
    """Fuzzy crisis degree k/n: the IAA membership of 'crisis' when each of n
    raters casts a crisp verdict and k of them say crisis (RQ1)."""
    if n <= 0 or not 0 <= k <= n:
        raise ValueError('Need n > 0 and 0 <= k <= n')
    return k / n


def majority_vote(k: int, n: int) -> str:
    """Crisp majority: 'crisis' if k > n/2, 'no_crisis' if k < n/2, else 'tie'."""
    degree = vote_degree(k, n)
    return 'crisis' if degree > 0.5 else ('no_crisis' if degree < 0.5 else 'tie')
