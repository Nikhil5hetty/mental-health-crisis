"""Tests for the word-meaning lab (WP6): logic unit tests on a synthetic results folder, consistency with the
committed RQ3 results, and Streamlit AppTest smoke tests of both pages."""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.app import logic
from src.cww import APPROPRIATENESS_WORDS, verbalise as shared
from src.cww.quantile import X
from src.cww.sets import trapezoid_mf

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / 'results/words'
APP = ROOT / 'app/word_lab.py'
TERMS = ['Almost Certain', 'Likely', 'Realistic Possibility', 'Remote Chance', 'Will Happen']
GROUPS = list(logic.GROUP_LABELS)
MODELS = ['m_a', 'm_b']
# Citizens cover 0..100 without gaps; the LLMs leave 15..40 uncovered (an all-zero gap, as in the study).
CIT = {'Remote Chance': (0, 0, 20, 30), 'Realistic Possibility': (20, 30, 50, 60), 'Likely': (50, 60, 75, 85),
       'Almost Certain': (80, 90, 100, 100), 'Will Happen': (85, 95, 100, 100)}
LLM = {'Remote Chance': (0, 0, 10, 15), 'Realistic Possibility': (40, 50, 60, 65), 'Likely': (60, 65, 80, 90),
       'Almost Certain': (85, 95, 100, 100), 'Will Happen': (90, 95, 100, 100)}


def write(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def make_results(root, rq3=True):
    """A small synthetic results/words folder with the same file layout and columns as the real one."""
    long = lambda rows: pd.DataFrame(rows)
    write(root / 'rq1/codebooks/citizen_it2.csv', long([dict(term=t, x=x, umf=u, lmf=u * .8) for t, p in CIT.items()
                                                       for x, u in zip(X, trapezoid_mf(X, *p))]))
    write(root / 'rq1/codebooks/group_t1.csv', long([dict(term=t, group=g, x=x, mu=u * (.8 + .03 * i))
                                                     for t, p in CIT.items() for i, g in enumerate(GROUPS)
                                                     for x, u in zip(X, trapezoid_mf(X, *p))]))
    t1, params, it2, metrics, invalid = [], [], [], [], []
    for prompt in ['P1', 'P2']:
        shift = 0 if prompt == 'P1' else 3   # P2 differs, so a P1 filter bug shows up
        for t, (a, b, c, d) in LLM.items():
            mus = []
            for k, m in enumerate(MODELS):
                q = (a, b, c, d) if not (m == 'm_b' and t == 'Likely') else (70, 70, 70, 70)  # m_b collapses on Likely
                q = tuple(min(100, v + shift * (k + 1)) if v < 100 else v for v in q)
                included = not (m == 'm_b' and t == 'Will Happen')                          # m_b excluded on Will Happen
                invalid.append(dict(model=m, prompt_id=prompt, term=t, n_rows=50, n_valid=50 if included else 12,
                                    n_missing=0 if included else 38, invalid_rate=0, n_reasked=0, n_length=0,
                                    n_superseded=0, included=int(included)))
                if not included:
                    continue
                mu = trapezoid_mf(X, *q); mus.append(mu)
                t1 += [dict(prompt=prompt, model=m, term=t, x=x, mu=v) for x, v in zip(X, mu)]
                params.append(dict(prompt=prompt, model=m, term=t, n_valid=50, q05=q[0], q25=q[1], q75=q[2], q95=q[3]))
            mus = np.array(mus)
            it2 += [dict(prompt=prompt, term=t, n_models=len(mus), x=x, umf=u, lmf=l)
                    for x, u, l in zip(X, mus.max(0), mus.min(0))]
    for i, t in enumerate(sorted(TERMS)):
        metrics += [dict(metric='jaccard_it2', term=t, model='llm_it2', estimate=.1 * (i + 1) if t != 'Likely' else .3,
                         ci_lo=.05, ci_hi=.6),
                    dict(metric='centroid_shift_it2', term=t, model='llm_it2', estimate=2.0 * i - 3, ci_lo=-9, ci_hi=9),
                    dict(metric='citizen_centre_it2', term=t, model='citizen', estimate=50, ci_lo=49, ci_hi=51),
                    dict(metric='llm_centre_it2', term=t, model='llm_it2', estimate=52, ci_lo=50, ci_hi=54)]
        metrics += [dict(metric=k, term=t, model=m, estimate=v, ci_lo=0, ci_hi=1)
                    for m in MODELS for k, v in [('jaccard_t1', .4), ('centroid_shift_t1', 1.5), ('width_ratio', .8)]
                    if not (m == 'm_b' and t == 'Will Happen')]
    write(root / 'rq2/codebooks/llm_model_t1.csv', long(t1))
    write(root / 'rq2/codebooks/llm_model_params.csv', long(params))
    write(root / 'rq2/codebooks/llm_it2.csv', long(it2))
    write(root / 'rq2/metrics.csv', long(metrics))
    write(root / 'rq2/invalid.csv', long(invalid))
    if rq3:
        write(root / 'rq3/metrics.csv', long([dict(metric='action_flip_rate', comparison='llm_vs_citizen',
                                                    estimate=.1, ci_lo=.05, ci_hi=.2, n_items=20)]))
        write(root / 'rq3/per_item.csv', long([dict(item_id=i) for i in range(20)]))
    return root


@pytest.fixture
def results(tmp_path):
    return make_results(tmp_path / 'words')


# --------------------------------------------------------------------------- loading and robustness

def test_missing_files_are_named_with_their_script(tmp_path):
    with pytest.raises(logic.MissingResults) as err:
        logic.load_word_data(tmp_path)
    assert 'rq2/codebooks/llm_it2.csv' in str(err.value) and 'run_rq2.py' in str(err.value)
    with pytest.raises(logic.MissingResults):
        logic.load_simulator(tmp_path)


def test_results_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.delenv(logic.ENV_RESULTS, raising=False)
    assert logic.results_dir() == logic.DEFAULT_RESULTS
    monkeypatch.setenv(logic.ENV_RESULTS, str(tmp_path))
    assert logic.results_dir() == tmp_path


def test_only_prompt_p1_is_used(results):
    d = logic.load_word_data(results)
    assert set(d.params.model) == set(MODELS) and 'prompt' not in d.params
    likely = d.params[(d.params.term == 'Realistic Possibility') & (d.params.model == 'm_a')].iloc[0]
    assert likely.q05 == LLM['Realistic Possibility'][0]          # P2 rows are shifted and must be ignored
    assert len(d.llm_it2[d.llm_it2.term == 'Likely']) == len(X)


# --------------------------------------------------------------------------- page 1

def test_term_curves_bands_models_and_groups(results):
    d = logic.load_word_data(results)
    bands, lines = logic.term_curves(d, 'Likely')
    assert set(bands.series) == {'Citizens (band over 6 groups)', 'LLMs (band over models)'}
    assert (bands.lo <= bands.hi + 1e-12).all()
    assert set(lines.series) == set(MODELS) and set(lines.kind) == {'model'}
    _, with_groups = logic.term_curves(d, 'Likely', show_groups=True, show_models=False)
    assert set(with_groups.series) == set(logic.GROUP_LABELS.values())
    _, nothing = logic.term_curves(d, 'Likely', show_models=False)
    assert nothing.empty
    with pytest.raises(ValueError):
        logic.term_curves(d, 'Not a word')


def test_term_summary_reads_results_and_flags(results):
    d = logic.load_word_data(results)
    s = logic.term_summary(d, 'Likely')
    assert s['jaccard'] == {'estimate': .3, 'ci_lo': .05, 'ci_hi': .6}
    assert s['category'] == 'hedge' and s['collapsed_models'] == ['m_b'] and s['excluded_models'] == []
    w = logic.term_summary(d, 'Will Happen')
    assert w['category'] == 'extreme' and w['excluded_models'] == ['m_b']
    row = w['models'].set_index('model').loc['m_b']
    assert not row.included and row.n_valid == 12 and pd.isna(row.jaccard)


def test_ranking_order_and_categories(results):
    r = logic.ranking(logic.load_word_data(results))
    assert r['rank'].tolist() == list(range(1, len(TERMS) + 1))
    assert r.jaccard.is_monotonic_decreasing
    # 'Likely' and 'Almost Certain' tie at 0.3 -> alphabetical
    tied = r[r.jaccard == .3].term.tolist()
    assert tied == sorted(tied)
    assert dict(zip(r.term, r.category))['Remote Chance'] == 'other'


def test_term_groups_match_the_analysis():
    from experiments.words import run_rq2
    assert logic.HEDGE == run_rq2.HEDGE and logic.EXTREME == run_rq2.EXTREME


def test_plain_language_helpers():
    assert logic.overlap_words(.7) == 'large overlap' and logic.overlap_words(.05) == 'almost no overlap'
    assert 'higher' in logic.shift_words(5) and 'lower' in logic.shift_words(-5) and 'same place' in logic.shift_words(.3)


# --------------------------------------------------------------------------- page 2

def test_simulator_uses_the_rq3_code_not_a_copy():
    from experiments.words import run_rq3
    assert logic.verbalise is shared.verbalise is run_rq3.verbalise
    assert logic.action is shared.action is run_rq3.action


def test_simulate_known_flip_fallback_and_agreement(results):
    sim = logic.load_simulator(results)
    t = logic.simulate(sim, 25, 'Mostly Appropriate')     # LLM gap at 25: fallback "None"
    assert t.codebook.tolist()[:2] == ['citizen', 'llm'] and len(t) == 2 + len(GROUPS)
    cit, llm = t.iloc[0], t.iloc[1]
    assert (cit.level, cit.action) == ('Possible', 1)
    assert (llm.level, llm.action, llm.fallback, llm.flips, llm.direction) == ('None', 0, True, True, 'less urgent')
    h = logic.headline(t)
    assert h['llm_flips'] and h['citizen_action'] == 1 and h['llm_action'] == 0 and h['any_fallback']
    same = logic.simulate(sim, 95, 'Fully Appropriate')
    assert not same.flips.any() and same.direction.eq('same').all()
    assert logic.headline(same)['llm_flips'] is False


@pytest.mark.parametrize('m', [-1, 100.5, float('nan')])
def test_simulate_rejects_bad_degree(results, m):
    with pytest.raises(ValueError):
        logic.simulate(logic.load_simulator(results), m, 'Fully Appropriate')


def test_simulate_rejects_unknown_reply(results):
    with pytest.raises(ValueError):
        logic.simulate(logic.load_simulator(results), 50, 'Brilliant')


def test_simulator_works_without_rq3(tmp_path):
    root = make_results(tmp_path / 'words', rq3=False)
    sim = logic.load_simulator(root)
    assert sim.rq3 is None and len(logic.simulate(sim, 50, 'Inappropriate')) == 2 + len(GROUPS)
    with_rq3 = logic.load_simulator(make_results(tmp_path / 'other'))
    assert with_rq3.rq3 == dict(flip_rate=.1, ci_lo=.05, ci_hi=.2, n_items=20, n_flips=2)


def test_level_curves(results):
    c = logic.level_curves(logic.load_simulator(results))
    assert set(c.codebook) == {logic.CODEBOOK_LABELS['citizen'], logic.CODEBOOK_LABELS['llm']}
    assert c.level.nunique() == 4 and c.mu.between(0, 1).all()


@pytest.mark.skipif(not (REAL / 'rq3/per_item.csv').is_file(), reason='committed RQ3 results not present')
def test_simulator_reproduces_committed_rq3_decisions():
    """Same rule as the study: every RQ3 item gets the same citizen and LLM actions in the app."""
    sim = logic.load_simulator(REAL)
    items = pd.read_csv(REAL / 'rq3/per_item.csv', keep_default_na=False)
    for r in items.itertuples():
        t = logic.simulate(sim, r.m, r.reply_word).set_index('codebook')
        assert t.loc['citizen', 'action'] == r.action_citizen and t.loc['llm', 'action'] == r.action_llm
        assert t.loc['llm', 'level'] == r.level_llm and bool(t.loc['llm', 'fallback']) == bool(r.fallback_llm)


# --------------------------------------------------------------------------- Streamlit smoke tests

AppTest = pytest.importorskip('streamlit.testing.v1').AppTest


def run_app(monkeypatch, results, page=None):
    monkeypatch.setenv(logic.ENV_RESULTS, str(results))
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    if page:
        at.switch_page(page).run()
    return at


def test_app_page1_smoke(monkeypatch, results):
    at = run_app(monkeypatch, results)
    assert not at.exception and at.title[0].value == 'What does a word mean?'
    assert [m.label for m in at.metric][0].startswith('Overlap')
    at.toggle[1].set_value(True).run()                     # show the citizen groups
    at.selectbox[0].set_value('Will Happen').run()
    assert not at.exception and any('m_b' in w.value for w in at.warning)


def test_app_page2_smoke_flip_and_no_rq3(monkeypatch, tmp_path):
    at = run_app(monkeypatch, make_results(tmp_path / 'a', rq3=False), 'views/simulator.py')
    assert not at.exception and at.title[0].value == 'Crisis-line simulator'
    at.slider[0].set_value(25.0).run()
    assert any('Decision flips' in e.value for e in at.error)
    assert any('were not found' in c.value for c in at.caption)
    at.slider[0].set_value(95.0).run()
    assert any('Same decision' in s.value for s in at.success)


def test_app_reports_missing_results(monkeypatch, tmp_path):
    root = make_results(tmp_path / 'b')
    shutil.rmtree(root / 'rq2')
    at = run_app(monkeypatch, root)
    assert not at.exception and 'Missing result files' in at.error[0].value
    at.switch_page('views/simulator.py').run()
    assert not at.exception and 'run_rq2.py' in at.error[0].value


@pytest.mark.skipif(not (REAL / 'rq2/metrics.csv').is_file(), reason='committed results not present')
def test_app_runs_on_committed_results(monkeypatch):
    at = run_app(monkeypatch, REAL)
    assert not at.exception
    at.switch_page('views/simulator.py').run()
    assert not at.exception and any('9 of 190' in c.value for c in at.caption)
