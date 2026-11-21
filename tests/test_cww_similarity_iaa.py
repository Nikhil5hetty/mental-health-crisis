"""Synthetic tests: Jaccard similarity and the IAA / panel encoder."""
import numpy as np
import pytest

from src.cww import (IT2Set, encode_panel, encode_scores, iaa_t1, it2_trapezoid, jaccard,
                     jaccard_t1, majority_vote, trapezoid, universe, vote_degree)


def random_it2(rng, x):
    umf = rng.random(x.size)
    return IT2Set(x, umf * rng.random(x.size), umf)


def test_jaccard_identity_symmetry_range():
    rng = np.random.default_rng(0)
    x = universe(0, 6, 0.1)
    for _ in range(20):
        a, b = random_it2(rng, x), random_it2(rng, x)
        assert jaccard(a, a) == pytest.approx(1.0)
        assert jaccard(a, b) == pytest.approx(jaccard(b, a))
        assert 0.0 <= jaccard(a, b) <= 1.0


def test_jaccard_disjoint_is_zero():
    a = it2_trapezoid((0.5, 1, 1.5, 2), (0.8, 1, 1.5, 1.7))
    b = it2_trapezoid((3, 3.5, 4, 4.5), (3.2, 3.5, 4, 4.2))
    assert jaccard(a, b) == 0.0


def test_jaccard_t1_special_case():
    a, b = trapezoid(1, 2, 3, 4), trapezoid(2, 3, 4, 5)
    assert jaccard(a, b) == pytest.approx(jaccard_t1(a, b))
    assert jaccard_t1(a, a) == pytest.approx(1.0)


def test_jaccard_requires_same_universe():
    with pytest.raises(ValueError):
        jaccard(trapezoid(1, 2, 3, 4), trapezoid(1, 2, 3, 4, x=universe(0, 6, 0.1)))


def test_iaa_identical_intervals_is_indicator():
    x = universe()
    mu = iaa_t1([(2.5, 3.5)] * 4, x).mu
    assert np.array_equal(mu, ((x >= 2.5) & (x <= 3.5)).astype(float))


def test_iaa_counts_fraction():
    x = universe()
    mu = iaa_t1([(1, 3), (2, 4)], x).mu
    at = dict(zip(x.round(2), mu))
    assert at[1.5] == 0.5 and at[2.5] == 1.0 and at[3.5] == 0.5 and at[4.5] == 0.0


def test_panel_identical_runs_is_indicator():
    x = universe()
    s = encode_panel([[3, 3, 3], [3, 3, 3], [3, 3]], x=x)
    indicator = ((x >= 2.5) & (x <= 3.5)).astype(float)
    assert np.array_equal(s.umf, indicator) and np.array_equal(s.lmf, indicator)


def test_panel_uneven_runs_values():
    x = universe()
    s = encode_panel([[2, 3, 4], [3, 5]], x=x)  # one judge has only 2 runs
    i3, i5 = np.flatnonzero(np.isclose(x, 3.0))[0], np.flatnonzero(np.isclose(x, 5.0))[0]
    assert s.umf[i3] == pytest.approx(1.0) and s.lmf[i3] == pytest.approx((1 / 3 + 1 / 2) / 2)
    assert s.umf[i5] == pytest.approx(0.5) and s.lmf[i5] == pytest.approx(0.25)


@pytest.mark.parametrize('seed', range(10))
def test_panel_lmf_below_umf(seed):
    rng = np.random.default_rng(seed)
    runs = [list(rng.integers(1, 6, size=rng.integers(1, 4))) for _ in range(3)]
    s = encode_panel(runs, half_width=float(rng.choice([0.25, 0.5])))
    assert np.all(s.lmf <= s.umf)


def test_single_score_raters_are_type1():
    s = encode_scores([2, 4])
    assert s.is_type1
    assert s.umf.max() == 0.5


def test_panel_rejects_out_of_universe_and_nan():
    with pytest.raises(ValueError):
        encode_panel([[5.8]])
    with pytest.raises(ValueError):
        encode_panel([[float('nan')]])


def test_vote_degree_and_majority():
    assert vote_degree(3, 4) == 0.75
    assert majority_vote(3, 4) == 'crisis'
    assert majority_vote(2, 4) == 'tie'
    assert majority_vote(4, 9) == 'no_crisis'
