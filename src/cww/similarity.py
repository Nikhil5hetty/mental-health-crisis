"""Jaccard similarity for fuzzy sets.

Reference
---------
Wu, D., & Mendel, J. M. (2009). A comparative study of ranking methods,
    similarity measures and uncertainty measures for interval type-2 fuzzy
    sets. Information Sciences 179(8), 1169-1192.
"""
from __future__ import annotations

import numpy as np

from .sets import IT2Set, T1Set, as_it2, same_universe


def jaccard(a: T1Set | IT2Set, b: T1Set | IT2Set) -> float:
    """IT2 Jaccard similarity (Wu & Mendel 2009).

    s_J(A, B) = sum_i [min(umf_A, umf_B) + min(lmf_A, lmf_B)]
                / sum_i [max(umf_A, umf_B) + max(lmf_A, lmf_B)]

    T1 inputs are embedded with lmf = umf, so the measure reduces to
    ``jaccard_t1``. The result lies in [0, 1]; s_J(A, A) = 1; it is symmetric.
    """
    a, b = as_it2(a), as_it2(b)
    same_universe(a, b)
    num = np.minimum(a.umf, b.umf).sum() + np.minimum(a.lmf, b.lmf).sum()
    den = np.maximum(a.umf, b.umf).sum() + np.maximum(a.lmf, b.lmf).sum()
    if den <= 0:
        raise ValueError('Jaccard similarity of two empty sets is undefined')
    return float(num / den)


def jaccard_t1(a: T1Set, b: T1Set) -> float:
    """T1 Jaccard similarity: sum min(mu_A, mu_B) / sum max(mu_A, mu_B)."""
    same_universe(a, b)
    den = np.maximum(a.mu, b.mu).sum()
    if den <= 0:
        raise ValueError('Jaccard similarity of two empty sets is undefined')
    return float(np.minimum(a.mu, b.mu).sum() / den)
