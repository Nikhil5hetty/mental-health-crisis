"""Centroids of T1 and IT2 fuzzy sets.

References
----------
Karnik, N. N., & Mendel, J. M. (2001). Centroid of a type-2 fuzzy set.
    Information Sciences 132(1-4), 195-220.
Wu, D., & Mendel, J. M. (2009). Enhanced Karnik-Mendel algorithms.
    IEEE TFS 17(4), 923-934.
"""
from __future__ import annotations

import numpy as np

from .sets import IT2Set, T1Set, as_it2


def centroid_t1(a: T1Set) -> float:
    """c_A = sum_i x_i mu_A(x_i) / sum_i mu_A(x_i)."""
    total = a.mu.sum()
    if total <= 0:
        raise ValueError('Centroid of an empty set is undefined')
    return float(np.dot(a.x, a.mu) / total)


def _prepare(x, lo, hi):
    x = np.asarray(x, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    if not (x.shape == lo.shape == hi.shape) or x.ndim != 1:
        raise ValueError('x, lo and hi must be 1-D arrays of equal length')
    if np.any(lo < 0) or np.any(lo > hi):
        raise ValueError('Require 0 <= lo <= hi')
    order = np.argsort(x, kind='stable')
    x, lo, hi = x[order], lo[order], hi[order]
    # Points with hi == 0 carry no weight in any embedded set; dropping them
    # guarantees a positive denominator for every switch point below.
    keep = hi > 0
    if not keep.any():
        raise ValueError('All upper weights are zero; weighted average undefined')
    return x[keep], lo[keep], hi[keep]


def _switch_index(x, y):
    """0-based k with x[k] <= y < x[k+1], clipped to [0, N-2]."""
    return int(np.clip(np.searchsorted(x, y, side='right') - 1, 0, x.size - 2))


def ekm_left(x, lo, hi) -> float:
    """Minimum of sum(w x)/sum(w) over w_i in [lo_i, hi_i] (EKM, Wu & Mendel 2009).

    The minimiser uses hi weights for x_i <= y_l and lo weights above it:
    y_l = (sum_{i<=k} x_i hi_i + sum_{i>k} x_i lo_i) / (sum_{i<=k} hi_i + sum_{i>k} lo_i).
    Initial switch point k = round(N / 2.4); each iteration updates the sums
    incrementally over the indices whose weight changes.
    """
    x, lo, hi = _prepare(x, lo, hi)
    n = x.size
    if n == 1:
        return float(x[0])
    k = int(np.clip(round(n / 2.4) - 1, 0, n - 2))
    a = np.dot(x[:k + 1], hi[:k + 1]) + np.dot(x[k + 1:], lo[k + 1:])
    b = hi[:k + 1].sum() + lo[k + 1:].sum()
    y = a / b
    for _ in range(n + 1):
        k_new = _switch_index(x, y)
        if k_new == k:
            break
        if k_new > k:
            idx = slice(k + 1, k_new + 1)
            sign = 1.0
        else:
            idx = slice(k_new + 1, k + 1)
            sign = -1.0
        delta = hi[idx] - lo[idx]
        a += sign * np.dot(x[idx], delta)
        b += sign * delta.sum()
        y = a / b
        k = k_new
    return float(y)


def ekm_right(x, lo, hi) -> float:
    """Maximum of sum(w x)/sum(w) over w_i in [lo_i, hi_i] (EKM, Wu & Mendel 2009).

    y_r = (sum_{i<=k} x_i lo_i + sum_{i>k} x_i hi_i) / (sum_{i<=k} lo_i + sum_{i>k} hi_i),
    initial switch point k = round(N / 1.7).
    """
    x, lo, hi = _prepare(x, lo, hi)
    n = x.size
    if n == 1:
        return float(x[0])
    k = int(np.clip(round(n / 1.7) - 1, 0, n - 2))
    a = np.dot(x[:k + 1], lo[:k + 1]) + np.dot(x[k + 1:], hi[k + 1:])
    b = lo[:k + 1].sum() + hi[k + 1:].sum()
    y = a / b
    for _ in range(n + 1):
        k_new = _switch_index(x, y)
        if k_new == k:
            break
        if k_new > k:
            idx = slice(k + 1, k_new + 1)
            sign = -1.0
        else:
            idx = slice(k_new + 1, k + 1)
            sign = 1.0
        delta = hi[idx] - lo[idx]
        a += sign * np.dot(x[idx], delta)
        b += sign * delta.sum()
        y = a / b
        k = k_new
    return float(y)


def centroid_it2(a: T1Set | IT2Set) -> tuple[float, float]:
    """Centroid C_A = [c_l, c_r] of an IT2 set (Karnik & Mendel 2001), via EKM.

    c_l = min over embedded sets of sum x mu / sum mu  (ekm_left with lo=lmf, hi=umf)
    c_r = max over embedded sets                         (ekm_right)
    For lmf = umf this reduces to the T1 centroid (c_l = c_r).
    """
    a = as_it2(a)
    return ekm_left(a.x, a.lmf, a.umf), ekm_right(a.x, a.lmf, a.umf)


def centre(a: T1Set | IT2Set) -> float:
    """Defuzzified value c(A) = (c_l + c_r) / 2 (Mendel & Wu 2010)."""
    c_l, c_r = centroid_it2(a)
    return (c_l + c_r) / 2


def width(a: T1Set | IT2Set) -> float:
    """Centroid width c_r - c_l, an uncertainty measure (Wu & Mendel 2009, Inf. Sci.)."""
    c_l, c_r = centroid_it2(a)
    return c_r - c_l
