"""Synthetic tests: sets, T1/IT2 centroids, EKM vs brute force."""
import numpy as np
import pytest

from src.cww import (IT2Set, T1Set, centre, centroid_it2, centroid_t1, it2_trapezoid,
                     trapezoid, universe, width)


def brute_force_centroid(x, lmf, umf):
    """Enumerate every switch point k (incl. all-lmf / all-umf) for c_l and c_r."""
    lefts, rights = [], []
    for k in range(-1, len(x)):
        upto = np.arange(len(x)) <= k
        w_l = np.where(upto, umf, lmf)
        w_r = np.where(upto, lmf, umf)
        if w_l.sum() > 0:
            lefts.append(np.dot(x, w_l) / w_l.sum())
        if w_r.sum() > 0:
            rights.append(np.dot(x, w_r) / w_r.sum())
    return min(lefts), max(rights)


def test_universe_default_and_exact_grid():
    x = universe()
    assert x[0] == 0 and x[-1] == 6 and len(x) == 601
    assert 2.5 in x and 3.5 in x
    with pytest.raises(ValueError):
        universe(0, 1, 0.3)


def test_set_validation():
    x = universe(0, 1, 0.5)
    with pytest.raises(ValueError):
        T1Set(x, [0, 1.5, 0])
    with pytest.raises(ValueError):
        IT2Set(x, [0, 0.8, 0], [0, 0.5, 0])  # lmf > umf
    with pytest.raises(ValueError):
        trapezoid(2, 1, 3, 4)


def test_trapezoid_shape():
    a = trapezoid(1, 2, 3, 4, h=0.8)
    assert a.height == pytest.approx(0.8)
    mu = dict(zip(a.x.round(2), a.mu))
    assert mu[1.5] == pytest.approx(0.4) and mu[2.5] == pytest.approx(0.8) and mu[4.0] == 0
    vertical = trapezoid(2, 2, 3, 3)
    assert set(np.unique(vertical.mu)) == {0.0, 1.0}


def test_t1_centroid_symmetric_trapezoid():
    assert centroid_t1(trapezoid(1, 2, 3, 4)) == pytest.approx(2.5)
    assert centroid_t1(trapezoid(2.5, 2.5, 3.5, 3.5)) == pytest.approx(3.0)


def test_it2_centroid_symmetric_trapezoid():
    a = it2_trapezoid((1, 2, 3, 4), (1.5, 2.2, 2.8, 3.5, 0.7))
    c_l, c_r = centroid_it2(a)
    assert c_l < 2.5 < c_r
    assert centre(a) == pytest.approx(2.5, abs=1e-9)
    assert width(a) == pytest.approx(c_r - c_l)


def test_it2_with_equal_bounds_is_t1_centroid():
    t1 = trapezoid(0.5, 1.7, 2.0, 4.9)
    c_l, c_r = centroid_it2(IT2Set.from_t1(t1))
    assert c_l == pytest.approx(centroid_t1(t1), abs=1e-12)
    assert c_r == pytest.approx(centroid_t1(t1), abs=1e-12)


@pytest.mark.parametrize('seed', range(20))
def test_ekm_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 80))
    x = np.sort(rng.choice(np.arange(0, 6.0001, 0.01), size=n, replace=False))
    umf = rng.random(n) * (rng.random(n) > 0.2)  # include zero-membership points
    umf[rng.integers(n)] = 1.0
    lmf = umf * rng.random(n)
    a = IT2Set(x, lmf, umf)
    expected = brute_force_centroid(x, a.lmf, a.umf)
    assert np.allclose(centroid_it2(a), expected, atol=1e-6)
