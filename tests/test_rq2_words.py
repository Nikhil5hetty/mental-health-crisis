"""Synthetic-only tests for the citizen vs LLM word-model comparison (WP4, experiments/words/run_rq2.py)."""
import hashlib

import numpy as np
import pandas as pd
import pytest

from experiments.words import run_rq2 as r
from experiments.words.elicit_llm import TERMS, COLUMNS
from experiments.words.run_rq1 import fit as rq1_fit, write_models
from src.cww.quantile import quantile_trapezoid, envelope
from src.data import capphrase

CENTRE = {t: 5 + 90 * i / (len(TERMS) - 1) for i, t in enumerate(TERMS)}
AGES = ['18-24', '35-44', '65-74']
ENGLISH = ['English is my first language', 'English is not my first language but I am fluent']


def write_capphrase(root, per_group=32, seed=1):
    rng = np.random.default_rng(seed)
    meta, judgements = [], []
    rid = 0
    for english in ENGLISH:
        for age in AGES + ['Under 18']:
            for _ in range(per_group if age != 'Under 18' else 3):
                rid += 1
                meta.append(dict(response_id=f'r{rid}', timestamp='t', age_band=age, english_background=english,
                                 education_level='NA', country_of_residence='NA'))
                for order, term in enumerate(TERMS):
                    v = float(np.clip(np.round(rng.normal(CENTRE[term], 8)), 0, 100))
                    judgements.append(dict(response_id=f'r{rid}', term=term, probability=v, order=order))
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(meta).to_csv(root / 'respondent_metadata.csv', index=False)
    pd.DataFrame(judgements).to_csv(root / 'absolute_judgements.csv', index=False)
    return root


def write_rq1_codebooks(capphrase_root, directory):
    pooled, citizen, groups, _, _ = rq1_fit(capphrase.load_judgements(capphrase_root))
    write_models(pooled, citizen, groups, directory)
    return directory


def answer_rows(models=('m_a', 'm_b'), prompts=('P1', 'P2'), n=40, seed=2, missing=None, constant=None):
    """missing: {(model, prompt, term): number of missing rows}; constant: {(model, term): value}."""
    rng = np.random.default_rng(seed)
    missing, constant = missing or {}, constant or {}
    rows = []
    for model in models:
        for prompt in prompts:
            for term in TERMS:
                k = missing.get((model, prompt, term), 0)
                for s in range(n):
                    if s < k:
                        value, attempts = np.nan, 3
                    elif (model, term) in constant:
                        value, attempts = constant[(model, term)], 1
                    else:
                        value, attempts = float(np.clip(np.round(rng.normal(CENTRE[term], 4)), 0, 100)), 1
                    rows.append(dict(model=model, digest='d' * 64, prompt_id=prompt, term=term, sample=s,
                                     raw_text='' if np.isnan(value) else f'{value:g}', value=value,
                                     attempts=attempts, timestamp='t', done_reason='stop'))
    return pd.DataFrame(rows, columns=COLUMNS)


@pytest.fixture
def setup(tmp_path):
    root = write_capphrase(tmp_path / 'capphrase')
    books = write_rq1_codebooks(root, tmp_path / 'rq1')
    answers = answer_rows(missing={('m_b', 'P1', 'Will Happen'): 11, ('m_b', 'P1', 'Likely'): 10},
                          constant={('m_a', 'About Even'): 50.0})
    path = tmp_path / 'answers.csv'
    answers.to_csv(path, index=False)
    return dict(root=root, books=books, answers=path, tmp=tmp_path)


# --------------------------------------------------------------------------- exclusion rule

def test_exclusion_rule_is_inclusive_at_30_and_counts_only_valid(setup):
    answers = r.load_answers(setup['answers'])
    samples, excluded = r.eligible_samples(answers, 'P1')
    assert excluded == [('m_b', 'Will Happen', 29)]
    assert len(samples[('m_b', 'Likely')]) == 30          # 40 rows - 10 missing = 30: kept
    assert ('m_b', 'Will Happen') not in samples
    p2, ex2 = r.eligible_samples(answers, 'P2')
    assert ex2 == [] and len(p2) == 2 * len(TERMS)
    table = r.invalid_table(answers).set_index(['model', 'prompt_id', 'term'])
    row = table.loc[('m_b', 'P1', 'Will Happen')]
    assert (row.n_rows, row.n_valid, row.n_missing, row.included) == (40, 29, 11, 0)
    assert row.invalid_rate == 11 / 40
    assert table.loc[('m_b', 'P1', 'Likely')].included == 1


def test_excluded_model_is_left_out_of_llm_it2_and_h2b_denominator(setup):
    answers = r.load_answers(setup['answers'])
    citizens = r.Citizens(capphrase.load_judgements(setup['root']))
    fit, _ = citizens.fit()
    samples, _ = r.eligible_samples(answers, 'P1')
    m, t1, bands = r.compute(fit, samples)
    expected = envelope([quantile_trapezoid(samples[('m_a', 'Will Happen')])])
    np.testing.assert_array_equal(bands['Will Happen'].umf, expected.umf)
    assert ('jaccard_t1', 'Will Happen', 'm_b') not in m
    assert m[('h2b_n_terms', 'all', 'm_b')] == len(TERMS) - 1
    assert m[('h2b_n_terms', 'all', 'm_a')] == len(TERMS)


def test_out_of_range_values_rejected(tmp_path):
    a = answer_rows(models=('m',), prompts=('P1',), n=30)
    a.loc[0, 'value'] = 101
    a.to_csv(tmp_path / 'a.csv', index=False)
    with pytest.raises(ValueError):
        r.load_answers(tmp_path / 'a.csv')


# --------------------------------------------------------------------------- bootstrap seed

def test_streams_are_seeded_and_independent():
    c1, l1 = r.make_streams(r.SEED)
    c2, l2 = r.make_streams(r.SEED)
    assert np.array_equal(c1.integers(0, 1000, 50), c2.integers(0, 1000, 50))
    assert np.array_equal(l1['P1'].integers(0, 1000, 50), l2['P1'].integers(0, 1000, 50))
    children = np.random.SeedSequence(20261124).spawn(3)
    c3, _ = r.make_streams()
    assert np.array_equal(c3.integers(0, 1000, 50), np.random.default_rng(children[0]).integers(0, 1000, 50))
    _, l4 = r.make_streams()
    assert not np.array_equal(l4['P1'].integers(0, 1000, 50), l4['P2'].integers(0, 1000, 50))


def test_bootstrap_reproducible_seed_sensitive_and_prompt_independent(setup):
    answers = r.load_answers(setup['answers'])
    citizens = r.Citizens(capphrase.load_judgements(setup['root']))
    both = {p: r.eligible_samples(answers, p)[0] for p in r.PROMPTS}
    key = ('jaccard_it2', 'Likely', r.IT2)
    a, _ = r.bootstrap(citizens, both, b=4)
    b, _ = r.bootstrap(citizens, both, b=4)
    c, _ = r.bootstrap(citizens, both, b=4, seed=r.SEED + 1)
    only_p1, _ = r.bootstrap(citizens, {'P1': both['P1']}, b=4)
    assert np.array_equal(a['P1'][key], b['P1'][key]) and np.array_equal(a['P2'][key], b['P2'][key])
    assert not np.array_equal(a['P1'][key], c['P1'][key])
    assert np.array_equal(a['P1'][key], only_p1['P1'][key])   # P2 draws do not shift the P1 stream
    assert len(np.unique(a['P1'][key])) > 1                    # replicates actually vary


def test_llm_resample_preserves_size_and_support():
    rng = np.random.default_rng(0)
    samples = {('m', 't'): np.arange(30.0), ('m', 'u'): np.full(40, 7.0)}
    out = r.resample_llm(samples, rng)
    assert len(out[('m', 't')]) == 30 and set(out[('m', 't')]) <= set(range(30))
    assert np.all(out[('m', 'u')] == 7.0)


# --------------------------------------------------------------------------- H2a / H2b logic

def test_h2a_difference_and_verdict():
    jac = {t: 0.5 for t in TERMS}
    for t in r.HEDGE:
        jac[t] = 0.2
    for t in r.EXTREME:
        jac[t] = 0.8
    hedge, extreme, d = r.h2a_difference(jac)
    assert hedge == pytest.approx(0.2) and extreme == pytest.approx(0.8) and d == pytest.approx(-0.6)
    assert set(r.HEDGE) == {'Realistic Possibility', 'Could Happen', 'May Happen', 'Might Happen', 'Probable', 'Likely'}
    assert set(r.EXTREME) == {'Almost No Chance', 'Almost Certain', 'Will Happen'}
    assert set(r.HEDGE + r.EXTREME) <= set(TERMS)
    assert r.verdict_h2a(-1e-9) == 'supported'
    assert r.verdict_h2a(0.0) == 'not supported'
    assert r.verdict_h2a(0.3) == 'not supported'


def test_h2b_share_strict_and_verdict():
    citizen = {'a': 10, 'b': 10, 'c': 10, 'd': 0}
    assert r.h2b_share({'a': 5, 'b': 10, 'c': 20, 'd': 0}, citizen) == 0.25   # equal is not narrower
    assert r.h2b_share({'a': 5, 'b': 5}, citizen) == 1.0                      # only eligible terms count
    with pytest.raises(ValueError):
        r.h2b_share({}, citizen)
    assert r.verdict_h2b(0.5) == 'not supported'
    assert r.verdict_h2b(0.5000001) == 'supported'


def test_width_ratio_undefined_when_citizen_iqr_zero():
    assert np.isnan(r.ratio(3.0, 0.0))
    assert r.ratio(3.0, 6.0) == 0.5


# --------------------------------------------------------------------------- collapses

def test_collapse_detection():
    samples = {('m', 'same'): np.full(30, 50.0),
               ('m', 'mostly'): np.r_[np.full(29, 20.0), 21.0],
               ('m', 'spread'): np.linspace(0, 100, 30)}
    rows = r.collapses(samples)
    assert [(x['term'], x['value']) for x in rows] == [('mostly', 20.0), ('same', 50.0)]
    assert rows[1]['share'] == 1.0 and rows[0]['share'] == pytest.approx(29 / 30)


# --------------------------------------------------------------------------- end to end

def test_rq1_codebook_mismatch_stops(setup):
    citizens = r.Citizens(capphrase.load_judgements(setup['root']))
    fit, _ = citizens.fit()
    pooled, bands = r.read_rq1_codebooks(setup['books'])
    r.check_against_rq1(fit, pooled, bands)
    other = write_rq1_codebooks(write_capphrase(setup['tmp'] / 'other', seed=9), setup['tmp'] / 'other_books')
    with pytest.raises(ValueError):
        r.check_against_rq1(fit, *r.read_rq1_codebooks(other))


def digest(directory):
    return {p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob('*')) if p.is_file()}


def test_end_to_end_outputs_and_byte_identical_rerun(setup):
    outs = []
    for name in ['run1', 'run2']:
        out = setup['tmp'] / name
        r.run(setup['answers'], None, setup['root'], setup['books'], out, b=6)
        outs.append(digest(out))
    assert outs[0] == outs[1]
    out = setup['tmp'] / 'run1'
    for f in ['metrics.csv', 'sensitivity.csv', 'invalid.csv', 'SUMMARY.md', 'codebooks/llm_model_t1.csv',
              'codebooks/llm_it2.csv', 'figures/bands_small_multiples.png', 'figures/jaccard_dotplot.png',
              'figures/centroid_shift.png']:
        assert f in outs[0]
    metrics = pd.read_csv(out / 'metrics.csv')
    assert list(metrics.columns) == ['metric', 'term', 'model', 'estimate', 'ci_lo', 'ci_hi']
    finite = metrics.dropna(subset=['estimate'])
    assert (finite.ci_lo <= finite.ci_hi).all()
    text = (out / 'SUMMARY.md').read_text()
    assert 'm_b / Will Happen (n valid = 29)' in text
    assert '| m_a | About Even | 50 |' in text          # collapse reported
    assert '## Deviations' in text and 'H2a:' in text
