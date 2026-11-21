"""Synthetic tests for the RQ2 experiment rules (no data needed)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('run_rq2', ROOT / 'experiments/rq2/run_rq2.py')
rq2 = importlib.util.module_from_spec(_spec)
sys.modules['run_rq2'] = rq2  # dataclasses resolve their module via sys.modules
_spec.loader.exec_module(rq2)

WORDS = rq2.APPROPRIATENESS_WORDS


@pytest.mark.parametrize('runs,expected', [
    ((3, 3, 3), 'High'), ((4,), 'High'), ((2, 3, 3), 'Medium'), ((4, 5), 'Medium'),
    ((1, 3, 3), 'Low'), ((1, 5), 'Low')])
def test_reliability_word(runs, expected):
    assert rq2.reliability_word(runs) == expected


def test_weight_words_are_valid_it2_on_unit_interval():
    for s in rq2.WEIGHT_SETS.values():
        assert s.x[0] == 0 and s.x[-1] == 1 and np.all(s.lmf <= s.umf)


@pytest.mark.parametrize('max_j,top2,expected', [
    (0.49, 0.10, True),   # max Jaccard below 0.5
    (0.70, 0.66, True),   # margin 0.04 < 0.05
    (0.75, 0.625, False),  # margin 0.125 >= 0.05 (exact binary values)
    (0.5625, 0.5, False),  # margin 0.0625, max J above 0.5
    (0.50, 0.30, False),  # 0.5 is not below 0.5
])
def test_contested_flag(max_j, top2, expected):
    assert rq2.is_contested(max_j, top2) is expected


def test_contested_threshold_parameter():
    p = rq2.Params(contested_max_j=0.6)
    assert rq2.is_contested(0.55, 0.1, p) and not rq2.is_contested(0.55, 0.1)


def sims(**values):
    out = dict.fromkeys(WORDS, 0.0)
    out.update({WORDS[int(k[1:])]: v for k, v in values.items()})
    return out


def test_m3b_logic():
    # Already unsafe word -> flagged regardless of contest.
    assert rq2.m3b_flag(1, False, sims(w1=0.8))
    # Safe word, contested, Inappropriate within 0.05 of max -> escalate.
    assert rq2.m3b_flag(2, True, sims(w2=0.45, w1=0.41))
    # Safe word, contested, but unsafe words too far below max -> no flag.
    assert not rq2.m3b_flag(2, True, sims(w2=0.45, w1=0.39))
    # Safe word, close unsafe competitor, but not contested -> no flag.
    assert not rq2.m3b_flag(2, False, sims(w2=0.45, w1=0.44))
    # Harmful (index 0) also counts as an unsafe word.
    assert rq2.m3b_flag(3, True, sims(w3=0.3, w0=0.27))


def test_word_index_rounds_half_up():
    assert rq2.word_index(1.5) == 1 and rq2.word_index(2.49) == 1 and rq2.word_index(4.5) == 4


def test_bootstrap_seed_reproducible():
    a, b = rq2.boot_index(50), rq2.boot_index(50)
    assert np.array_equal(a, b) and a.shape == (rq2.B, 50)
    fn = rq2.m_mean(np.random.default_rng(0).random(50))
    assert rq2.summarise(fn, 50) == rq2.summarise(fn, 50)


def test_metric_helpers():
    pred = np.array([1, 1, 0, 0], bool)
    truth = np.array([1, 0, 1, 0], bool)
    idx = np.arange(4)[None, :]
    assert rq2.m_recall(pred, truth)(idx)[0] == 0.5
    assert rq2.m_precision(pred, truth)(idx)[0] == 0.5
    assert np.isnan(rq2.m_precision(np.zeros(4, bool), truth)(idx)[0])
    a = np.array([0, 1, 2, 3, 4])
    assert rq2.m_qwk(a, a)(np.arange(5)[None, :])[0] == pytest.approx(1.0)
    assert rq2.spearman([1, 2, 2, 3], [1, 3, 3, 4]) == pytest.approx(1.0)
    assert np.isnan(rq2.spearman([1, 1, 1], [1, 2, 3]))


def test_qwk_matches_confusion_matrix_definition():
    rng = np.random.default_rng(1)
    a, b = rng.integers(0, 5, 60), rng.integers(0, 5, 60)
    obs = np.zeros((5, 5))
    np.add.at(obs, (a, b), 1)
    obs /= obs.sum()
    exp = np.outer(obs.sum(1), obs.sum(0))
    w = (np.subtract.outer(np.arange(5), np.arange(5)) ** 2) / 16
    expected = 1 - (w * obs).sum() / (w * exp).sum()
    assert rq2.m_qwk(a, b)(np.arange(60)[None, :])[0] == pytest.approx(expected)
