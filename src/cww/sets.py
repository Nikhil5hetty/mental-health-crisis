"""Discretised type-1 (T1) and interval type-2 (IT2) fuzzy sets.

All sets live on a sampled universe of discourse X = {x_0 < x_1 < ... < x_N}.
Operations between sets (similarity, decoding) require the *same* universe.

References
----------
Zadeh, L. A. (1996). Fuzzy logic = computing with words. IEEE TFS 4(2).
Mendel, J. M., & Wu, D. (2010). Perceptual Computing. Wiley-IEEE Press.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Numerical tolerance used for membership bound checks.
TOL = 1e-12


def universe(lo: float = 0.0, hi: float = 6.0, step: float = 0.01) -> np.ndarray:
    """Return the sampled universe ``[lo, lo+step, ..., hi]``.

    Points are rounded to 10 decimals so that decimal grid values (e.g. 2.5)
    are represented exactly as the nearest double, which keeps closed-interval
    tests such as ``x >= 2.5`` reproducible.
    """
    if not hi > lo or not step > 0:
        raise ValueError('Need hi > lo and step > 0')
    n = int(round((hi - lo) / step))
    if not np.isclose(lo + n * step, hi):
        raise ValueError('(hi - lo) must be a multiple of step')
    return np.round(lo + np.arange(n + 1) * step, 10)


DEFAULT_X = universe()


def _check_x(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size == 0 or np.any(np.diff(x) <= 0):
        raise ValueError('Universe must be a non-empty, strictly increasing 1-D array')
    return x


@dataclass(frozen=True, eq=False)
class T1Set:
    """Type-1 fuzzy set A = {(x, mu_A(x))} sampled on ``x``; 0 <= mu <= 1."""

    x: np.ndarray
    mu: np.ndarray

    def __post_init__(self):
        x = _check_x(self.x)
        mu = np.asarray(self.mu, dtype=float)
        if mu.shape != x.shape:
            raise ValueError('mu must have the same shape as x')
        if np.any(~np.isfinite(mu)) or mu.min() < -TOL or mu.max() > 1 + TOL:
            raise ValueError('Membership must lie in [0, 1]')
        object.__setattr__(self, 'x', x)
        object.__setattr__(self, 'mu', np.clip(mu, 0.0, 1.0))

    @property
    def height(self) -> float:
        """h_A = max_x mu_A(x)."""
        return float(self.mu.max())


@dataclass(frozen=True, eq=False)
class IT2Set:
    """Interval type-2 fuzzy set given by its footprint of uncertainty.

    FOU(A) = union over x of [lmf(x), umf(x)], with 0 <= lmf <= umf <= 1
    (Mendel & Wu 2010, ch. 2).
    """

    x: np.ndarray
    lmf: np.ndarray
    umf: np.ndarray

    def __post_init__(self):
        x = _check_x(self.x)
        lmf = np.asarray(self.lmf, dtype=float)
        umf = np.asarray(self.umf, dtype=float)
        if lmf.shape != x.shape or umf.shape != x.shape:
            raise ValueError('lmf and umf must have the same shape as x')
        if np.any(~np.isfinite(lmf)) or np.any(~np.isfinite(umf)):
            raise ValueError('Membership must be finite')
        if lmf.min() < -TOL or umf.max() > 1 + TOL or np.any(lmf > umf + TOL):
            raise ValueError('Require 0 <= lmf <= umf <= 1')
        lmf = np.clip(lmf, 0.0, 1.0)
        umf = np.clip(umf, 0.0, 1.0)
        object.__setattr__(self, 'x', x)
        object.__setattr__(self, 'umf', umf)
        object.__setattr__(self, 'lmf', np.minimum(lmf, umf))

    @classmethod
    def from_t1(cls, a: T1Set) -> 'IT2Set':
        """Embed a T1 set as an IT2 set with lmf = umf = mu."""
        return cls(a.x, a.mu.copy(), a.mu.copy())

    @property
    def is_type1(self) -> bool:
        return bool(np.allclose(self.lmf, self.umf, atol=TOL))

    @property
    def upper(self) -> T1Set:
        return T1Set(self.x, self.umf)

    @property
    def lower(self) -> T1Set:
        return T1Set(self.x, self.lmf)


def as_it2(a: T1Set | IT2Set) -> IT2Set:
    """Return ``a`` as an IT2 set (T1 sets become lmf = umf)."""
    if isinstance(a, IT2Set):
        return a
    if isinstance(a, T1Set):
        return IT2Set.from_t1(a)
    raise TypeError(f'Expected T1Set or IT2Set, got {type(a).__name__}')


def same_universe(a, b) -> None:
    """Raise ValueError unless two sets are sampled on the identical universe."""
    if a.x.shape != b.x.shape or not np.array_equal(a.x, b.x):
        raise ValueError('Sets are defined on different universes')


def trapezoid_mf(x, a, b, c, d, h=1.0) -> np.ndarray:
    """Trapezoidal membership function.

    mu(x) = h * (x - a)/(b - a)   for a <= x < b
            h                     for b <= x <= c
            h * (d - x)/(d - c)   for c < x <= d
            0                     otherwise,
    with vertical edges when a == b or c == d (Mendel & Wu 2010, ch. 3).
    """
    if not a <= b <= c <= d:
        raise ValueError('Trapezoid needs a <= b <= c <= d')
    if not 0 <= h <= 1:
        raise ValueError('Height h must lie in [0, 1]')
    x = np.asarray(x, dtype=float)
    mu = np.zeros_like(x)
    mu[(x >= b) & (x <= c)] = 1.0
    if b > a:
        rising = (x >= a) & (x < b)
        mu[rising] = (x[rising] - a) / (b - a)
    if d > c:
        falling = (x > c) & (x <= d)
        mu[falling] = (d - x[falling]) / (d - c)
    return h * mu


def trapezoid(a, b, c, d, h=1.0, x=None) -> T1Set:
    """T1 trapezoid (a, b, c, d) of height h on universe ``x`` (default [0, 6], step 0.01)."""
    x = DEFAULT_X if x is None else _check_x(x)
    return T1Set(x, trapezoid_mf(x, a, b, c, d, h))


def it2_trapezoid(umf_params, lmf_params, x=None) -> IT2Set:
    """IT2 set whose UMF and LMF are trapezoids.

    ``umf_params`` and ``lmf_params`` are ``(a, b, c, d)`` or ``(a, b, c, d, h)``.
    The constructor asserts 0 <= lmf <= umf <= 1 on the sampled universe.
    """
    x = DEFAULT_X if x is None else _check_x(x)
    return IT2Set(x, trapezoid_mf(x, *lmf_params), trapezoid_mf(x, *umf_params))
