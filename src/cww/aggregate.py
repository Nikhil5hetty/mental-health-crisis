"""Aggregation operators: OWA, fuzzy weighted average (FWA) and LWA.

References
----------
Yager, R. R. (1988). On ordered weighted averaging aggregation operators in
    multicriteria decisionmaking. IEEE Trans. SMC 18(1), 183-190.
Liu, F., & Mendel, J. M. (2008). Aggregation using the fuzzy weighted average
    as computed by the Karnik-Mendel algorithms. IEEE TFS 16(1), 1-12.
Wu, D., & Mendel, J. M. (2007). Aggregation using the linguistic weighted
    average and interval type-2 fuzzy sets. IEEE TFS 15(6), 1145-1161.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .centroid import ekm_left, ekm_right
from .sets import IT2Set, T1Set, TOL, _check_x, as_it2

# ---------------------------------------------------------------- OWA


def _check_weights(weights) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or w.size == 0 or np.any(w < -TOL) or not np.isclose(w.sum(), 1.0):
        raise ValueError('OWA weights must be non-negative and sum to 1')
    return w


def owa(values, weights) -> float:
    """OWA_w(a_1..a_n) = sum_j w_j b_j, b_j the j-th LARGEST value (Yager 1988)."""
    w = _check_weights(weights)
    b = np.sort(np.asarray(values, dtype=float))[::-1]
    if b.shape != w.shape:
        raise ValueError('values and weights must have equal length')
    return float(np.dot(w, b))


def orness(weights) -> float:
    """orness(w) = (1/(n-1)) sum_{j=1..n} (n - j) w_j (Yager 1988); needs n >= 2.

    orness = 1 for max (w = (1,0,..,0)), 0.5 for the mean, 0 for min.
    """
    w = _check_weights(weights)
    n = w.size
    if n < 2:
        raise ValueError('orness is undefined for n < 2')
    return float(np.dot(n - np.arange(1, n + 1), w) / (n - 1))


def owa_weights(n: int, orness_level: float) -> np.ndarray:
    """OWA weights of length n with exactly the requested orness.

    Piecewise-linear mixture of the three reference operators:
      orness >= 0.5:  w = (2*o - 1) * e_max + (2 - 2*o) * u
      orness <  0.5:  w = (1 - 2*o) * e_min + 2*o * u
    with e_max = (1,0,..,0), e_min = (0,..,0,1), u = (1/n,..,1/n).
    orness is linear in w, so orness(w) = o exactly. (The mixture is a design
    choice; Yager 1988 defines OWA and orness, not this weight family.)
    """
    if n < 1:
        raise ValueError('n must be >= 1')
    if not 0 <= orness_level <= 1:
        raise ValueError('orness must lie in [0, 1]')
    if n == 1:
        return np.ones(1)
    u = np.full(n, 1.0 / n)
    e = np.zeros(n)
    if orness_level >= 0.5:
        e[0] = 1.0
        return (2 * orness_level - 1) * e + (2 - 2 * orness_level) * u
    e[-1] = 1.0
    return (1 - 2 * orness_level) * e + 2 * orness_level * u


# ---------------------------------------------------------------- FWA


@dataclass(frozen=True)
class AlphaCuts:
    """A T1 fuzzy number given by alpha-cuts [left(alpha), right(alpha)]."""

    alphas: np.ndarray
    left: np.ndarray
    right: np.ndarray

    @property
    def height(self) -> float:
        return float(self.alphas[-1])

    def to_t1(self, x) -> T1Set:
        """Discretise: mu(x) = max{alpha_k : left_k <= x <= right_k}.

        If a cut contains no grid point (it falls between two samples), the
        grid point nearest its midpoint receives that level, so that crisp
        results are not lost on the grid.
        """
        x = _check_x(x)
        mu = np.zeros_like(x)
        for alpha, lo, hi in zip(self.alphas, self.left, self.right):
            if alpha <= 0:
                continue
            mask = (x >= lo - 1e-9) & (x <= hi + 1e-9)
            if not mask.any():
                mask[np.argmin(np.abs(x - (lo + hi) / 2))] = True
            mu[mask] = np.maximum(mu[mask], alpha)
        return T1Set(x, mu)


def _height(obj) -> float:
    if isinstance(obj, T1Set):
        return obj.height
    return 1.0


def _cut(obj, alpha: float) -> tuple[float, float]:
    """Alpha-cut [l, r] of a crisp number, an interval (lo, hi), or a T1Set.

    For a T1Set the cut is {x : mu(x) >= alpha} on the grid (alpha = 0 means the
    support {mu > 0}); [l, r] is its convex hull, so non-convex inputs are
    convexified.
    """
    if isinstance(obj, T1Set):
        mask = obj.mu > TOL if alpha <= 0 else obj.mu >= alpha - TOL
        if not mask.any():
            raise ValueError(f'Empty alpha-cut at alpha={alpha}')
        idx = np.flatnonzero(mask)
        return float(obj.x[idx[0]]), float(obj.x[idx[-1]])
    if np.isscalar(obj):
        return float(obj), float(obj)
    lo, hi = obj
    if lo > hi:
        raise ValueError(f'Invalid interval [{lo}, {hi}]')
    return float(lo), float(hi)


def fwa_alpha_cut(x_sets: Sequence, w_sets: Sequence, n_alpha: int = 51) -> AlphaCuts:
    """Fuzzy weighted average Y = sum_i W_i X_i / sum_i W_i via alpha-cuts + KM.

    Each X_i and W_i is a crisp number, an interval (lo, hi) or a T1Set.
    For alpha in [0, h_min] (h_min = smallest input height), with
    X_i(alpha) = [a_i, b_i] and W_i(alpha) = [c_i, d_i]:
      y_L(alpha) = min_{w_i in [c_i, d_i]} sum w_i a_i / sum w_i   (KM / EKM)
      y_R(alpha) = max_{w_i in [c_i, d_i]} sum w_i b_i / sum w_i
    (Liu & Mendel 2008). The alpha levels are linspace(0, h_min, n_alpha).
    """
    if len(x_sets) == 0 or len(x_sets) != len(w_sets):
        raise ValueError('Need equally many (>= 1) inputs and weights')
    if n_alpha < 2:
        raise ValueError('n_alpha must be >= 2')
    h_min = min(_height(o) for o in list(x_sets) + list(w_sets))
    if h_min <= 0:
        raise ValueError('An input has height 0; FWA undefined')
    alphas = np.linspace(0.0, h_min, n_alpha)
    left = np.empty(n_alpha)
    right = np.empty(n_alpha)
    for k, alpha in enumerate(alphas):
        xc = np.array([_cut(o, alpha) for o in x_sets])
        wc = np.array([_cut(o, alpha) for o in w_sets])
        if np.any(wc < 0):
            raise ValueError('Weights must be non-negative')
        left[k] = ekm_left(xc[:, 0], wc[:, 0], wc[:, 1])
        right[k] = ekm_right(xc[:, 1], wc[:, 0], wc[:, 1])
    return AlphaCuts(alphas, left, right)


# ---------------------------------------------------------------- LWA


def lwa_cuts(x_sets: Sequence, w_sets: Sequence, n_alpha: int = 51):
    """Alpha-cuts of the LWA's UMF and LMF (Wu & Mendel 2007).

    UMF(Y) = FWA(UMF(X_i), UMF(W_i)),  alpha in [0, min heights of the UMFs]
    LMF(Y) = FWA(LMF(X_i), LMF(W_i)),  alpha in [0, h_min], h_min = min heights
    of the LMFs. Returns (umf_cuts, lmf_cuts); lmf_cuts is None when some LMF
    has height 0 (then LMF(Y) = 0).
    """
    xs = [as_it2(s) for s in x_sets]
    ws = [as_it2(s) for s in w_sets]
    umf = fwa_alpha_cut([s.upper for s in xs], [s.upper for s in ws], n_alpha)
    lowers_x = [s.lower for s in xs]
    lowers_w = [s.lower for s in ws]
    if min(s.height for s in lowers_x + lowers_w) <= 0:
        return umf, None
    return umf, fwa_alpha_cut(lowers_x, lowers_w, n_alpha)


def lwa(x_sets: Sequence[IT2Set], w_sets: Sequence[IT2Set], n_alpha: int = 51,
        x=None) -> IT2Set:
    """Linguistic weighted average Y = sum W_i X_i / sum W_i (Wu & Mendel 2007).

    X_i and W_i are IT2 (or T1) sets; W_i may live on a different universe,
    e.g. [0, 1]. The result is discretised on ``x`` (default: the universe of
    X_1). LMF(Y) keeps its own height h_min. Because discretisation can make
    LMF exceed UMF by one grid step, LMF is clipped to min(LMF, UMF).
    """
    x = as_it2(x_sets[0]).x if x is None else _check_x(x)
    umf_cuts, lmf_cuts = lwa_cuts(x_sets, w_sets, n_alpha)
    umf = umf_cuts.to_t1(x).mu
    lmf = np.zeros_like(x) if lmf_cuts is None else lmf_cuts.to_t1(x).mu
    return IT2Set(x, np.minimum(lmf, umf), umf)
