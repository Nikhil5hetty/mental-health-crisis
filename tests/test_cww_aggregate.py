"""Synthetic tests: OWA, FWA (alpha-cuts + KM) and LWA."""
import itertools

import numpy as np
import pytest

from src.cww import (T1Set, centre, fwa_alpha_cut, it2_trapezoid, lwa, lwa_cuts, orness, owa,
                     owa_weights, trapezoid, universe)

W_X = universe(0, 1, 0.01)


def brute_force_interval_fwa(xs, ws):
    """Enumerate all endpoint combinations of x_i in [a_i, b_i], w_i in [c_i, d_i]."""
    values = []
    for xv in itertools.product(*xs):
        for wv in itertools.product(*ws):
            wv = np.array(wv)
            if wv.sum() > 0:
                values.append(np.dot(wv, xv) / wv.sum())
    return min(values), max(values)


def test_owa_reference_operators():
    v = [3, 1, 5, 2, 4]
    assert owa(v, owa_weights(5, 0.0)) == pytest.approx(1)
    assert owa(v, owa_weights(5, 0.5)) == pytest.approx(3)
    assert owa(v, owa_weights(5, 1.0)) == pytest.approx(5)


@pytest.mark.parametrize('n', [1, 2, 3, 9])
@pytest.mark.parametrize('o', [0, 0.1, 0.25, 0.5, 0.7, 1])
def test_owa_weights_sum_and_orness(n, o):
    w = owa_weights(n, o)
    assert w.sum() == pytest.approx(1) and np.all(w >= 0)
    if n > 1:
        assert orness(w) == pytest.approx(o)


def test_owa_rejects_bad_weights():
    with pytest.raises(ValueError):
        owa([1, 2], [0.7, 0.7])


def test_fwa_crisp_is_weighted_average():
    cuts = fwa_alpha_cut([2.0, 4.0, 5.0], [0.2, 0.7, 0.1])
    expected = (0.4 + 2.8 + 0.5) / 1.0
    assert np.allclose(cuts.left, expected) and np.allclose(cuts.right, expected)


@pytest.mark.parametrize('seed', range(10))
def test_fwa_intervals_match_brute_force(seed):
    rng = np.random.default_rng(seed)
    a = rng.uniform(1, 5, 4)
    xs = [(lo, lo + rng.uniform(0, 1)) for lo in a]
    c = rng.uniform(0, 0.8, 4)
    ws = [(lo, lo + rng.uniform(0, 0.2)) for lo in c]
    cuts = fwa_alpha_cut(xs, ws, n_alpha=5)
    y_l, y_r = brute_force_interval_fwa(xs, ws)
    assert np.allclose(cuts.left, y_l, atol=1e-9) and np.allclose(cuts.right, y_r, atol=1e-9)


def test_fwa_t1_sets_core_and_support():
    xs = [trapezoid(1, 2, 3, 4), trapezoid(2, 3, 3.5, 5)]
    ws = [trapezoid(0.1, 0.2, 0.3, 0.5, x=W_X), trapezoid(0.4, 0.6, 0.7, 0.9, x=W_X)]
    cuts = fwa_alpha_cut(xs, ws, n_alpha=11)
    assert (cuts.left[-1], cuts.right[-1]) == pytest.approx(
        brute_force_interval_fwa([(2, 3), (3, 3.5)], [(0.2, 0.3), (0.6, 0.7)]))
    # alpha = 0 is the discrete support {mu > 0}: the feet (mu = 0) are excluded,
    # so each support starts one grid step inside the closed support.
    assert (cuts.left[0], cuts.right[0]) == pytest.approx(
        brute_force_interval_fwa([(1.01, 3.99), (2.01, 4.99)], [(0.11, 0.49), (0.41, 0.89)]))
    assert np.all(np.diff(cuts.left) >= -1e-12) and np.all(np.diff(cuts.right) <= 1e-12)


def singleton(value, x):
    return it2_trapezoid((value,) * 4, (value,) * 4, x)


def test_lwa_crisp_singletons_is_weighted_average():
    xs = [singleton(2.0, universe()), singleton(4.0, universe())]
    ws = [singleton(0.2, W_X), singleton(0.7, W_X)]
    expected = (0.2 * 2 + 0.7 * 4) / 0.9
    umf_cuts, lmf_cuts = lwa_cuts(xs, ws)
    for cuts in (umf_cuts, lmf_cuts):
        assert np.allclose(cuts.left, expected) and np.allclose(cuts.right, expected)
    assert centre(lwa(xs, ws)) == pytest.approx(expected, abs=0.01)  # one grid step


def test_lwa_interval_sets_match_brute_force():
    x_umf, x_lmf = [(1, 3), (2.5, 4.5), (4, 5)], [(1.5, 2.5), (3, 4), (4.2, 4.8)]
    w_umf, w_lmf = [(0.1, 0.5), (0.3, 0.9), (0.6, 1.0)], [(0.2, 0.4), (0.5, 0.7), (0.7, 0.9)]
    xs = [it2_trapezoid((a, a, b, b), (c, c, d, d)) for (a, b), (c, d) in zip(x_umf, x_lmf)]
    ws = [it2_trapezoid((a, a, b, b), (c, c, d, d), W_X) for (a, b), (c, d) in zip(w_umf, w_lmf)]
    umf_cuts, lmf_cuts = lwa_cuts(xs, ws, n_alpha=5)
    assert (umf_cuts.left[-1], umf_cuts.right[-1]) == pytest.approx(brute_force_interval_fwa(x_umf, w_umf))
    assert (lmf_cuts.left[-1], lmf_cuts.right[-1]) == pytest.approx(brute_force_interval_fwa(x_lmf, w_lmf))
    y = lwa(xs, ws)
    assert np.all(y.lmf <= y.umf)


def test_lwa_respects_lmf_height():
    xs = [it2_trapezoid((1, 2, 3, 4), (1.5, 2.5, 2.5, 3.5, 0.6)),
          it2_trapezoid((2, 3, 4, 5), (2.5, 3.5, 3.5, 4.5, 0.8))]
    ws = [it2_trapezoid((0.2, 0.4, 0.6, 0.8), (0.3, 0.5, 0.5, 0.7), W_X)] * 2
    umf_cuts, lmf_cuts = lwa_cuts(xs, ws)
    assert umf_cuts.height == 1.0 and lmf_cuts.height == pytest.approx(0.6)
    y = lwa(xs, ws)
    assert y.lmf.max() == pytest.approx(0.6) and y.umf.max() == 1.0
    assert np.all(y.lmf <= y.umf)


def test_alpha_cuts_to_t1_keeps_off_grid_crisp_value():
    cuts = fwa_alpha_cut([2.0, 4.0], [1.0, 2.0])  # 10/3, between grid points
    mu = cuts.to_t1(universe()).mu
    assert isinstance(cuts.to_t1(universe()), T1Set) and mu.max() == 1.0
    assert universe()[mu.argmax()] == pytest.approx(10 / 3, abs=0.005)
