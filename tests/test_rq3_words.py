"""Synthetic-only tests for RQ3 crisis-line flips (WP5, experiments/words/run_rq3.py)."""
import hashlib

import numpy as np
import pandas as pd
import pytest

from experiments.words import run_rq3 as r
from src.cww import APPROPRIATENESS_WORDS
from src.cww.quantile import X
from src.cww.rules import RULE_TABLE, CRISIS_WORDS
from src.cww.sets import IT2Set, trapezoid_mf


def t1(a, b, c, d):
    mu = trapezoid_mf(X, a, b, c, d)
    return IT2Set(X, mu, mu)


# Citizen: None/Possible cross at m = 25 with membership 0.5 each (a tie).
CITIZEN = {'None': t1(0, 0, 20, 30), 'Possible': t1(20, 30, 50, 60),
           'Likely': t1(50, 60, 75, 85), 'Clear': t1(80, 90, 100, 100)}
# LLM: gap between None (ends at 15) and Possible (starts at 40): m = 25 has all-zero memberships.
LLM = {'None': t1(0, 0, 10, 15), 'Possible': t1(40, 50, 60, 65),
       'Likely': t1(60, 65, 80, 90), 'Clear': t1(85, 95, 100, 100)}


def books():
    group = {k: IT2Set(X, v.lmf, v.umf) for k, v in CITIZEN.items()}
    return {r.CITIZEN: CITIZEN, r.LLM: LLM, 'native|18-34': group}


def items(ms, reply='Mostly Appropriate'):
    return pd.DataFrame({'item_id': range(len(ms)), 'k_llm_crisis': [0] * len(ms), 'n_llm_observed': [9] * len(ms),
                         'm': ms, 'reply_word': [reply] * len(ms)})


def test_tie_goes_to_more_severe_word():
    mu = r.memberships(25, CITIZEN)
    assert mu['None'] == mu['Possible'] == 0.5
    assert r.verbalise(25, CITIZEN) == ('Possible', False)


def test_linear_interpolation_off_grid():
    m = 100 * 2 / 9   # 22.22..., not on the 0.5 grid
    mu = r.memberships(m, CITIZEN)
    assert mu['None'] == pytest.approx((30 - m) / 10, abs=1e-12)
    assert mu['Possible'] == pytest.approx((m - 20) / 10, abs=1e-12)


def test_mean_of_umf_and_lmf_is_used():
    s = IT2Set(X, trapezoid_mf(X, 20, 30, 40, 50, h=.4), trapezoid_mf(X, 10, 30, 40, 60))
    book = dict(CITIZEN, Possible=s)
    assert r.memberships(35, book)['Possible'] == pytest.approx(.7)


def test_all_zero_fallback_uses_nearest_centre_and_is_flagged():
    assert max(r.memberships(25, LLM).values()) == 0
    assert r.verbalise(25, LLM) == ('None', True)
    assert r.verbalise(35, LLM) == ('Possible', True)     # nearer the Possible centre
    assert r.verbalise(5, LLM) == ('None', False)


def test_fallback_centre_tie_goes_to_more_severe():
    centres = {'None': 10.0, 'Possible': 40.0, 'Likely': 70.0, 'Clear': 95.0}
    assert r.verbalise(25, LLM, centres) == ('Possible', True)


def test_action_is_rule_table_lookup():
    for level in CRISIS_WORDS:
        for reply in APPROPRIATENESS_WORDS:
            assert r.action(level, reply) == RULE_TABLE[(level, reply)]
    assert set(r.LEVELS) == set(CRISIS_WORDS) and r.SEVERITY == list(CRISIS_WORDS)


def test_known_flip_and_rates():
    table = r.per_item(items([0, 25, 95]), books())
    row = table.iloc[1]
    assert (row.level_citizen, row.action_citizen) == ('Possible', 1)
    assert (row.level_llm, row.action_llm, row.fallback_llm) == ('None', 0, 1)
    assert table.fallback_citizen.tolist() == [0, 0, 0]
    rates = r.rates(table.level_citizen.to_numpy(), table.level_llm.to_numpy(),
                    table.action_citizen.to_numpy(), table.action_llm.to_numpy())
    assert rates == {'word_disagreement_rate': 1 / 3, 'action_flip_rate': 1 / 3,
                     'under_escalation_rate': 1 / 3, 'over_escalation_rate': 0}
    flips = r.flip_matrix(table, books())
    primary = flips[flips.comparison == 'llm_vs_citizen'].set_index(['reference_action', 'llm_action']).n
    assert primary[('A1', 'A0')] == 1 and primary.sum() == 3
    assert set(flips.comparison) == {'llm_vs_citizen', 'llm_vs_native|18-34'}


def test_bootstrap_seed_and_reproducibility():
    idx = r.bootstrap_indices(10, b=5)
    assert np.array_equal(idx, np.random.default_rng(20261124).integers(0, 10, size=(5, 10)))
    table = r.per_item(items(np.linspace(0, 100, 40)), books())
    a = r.metric_table(table, books(), b=50)
    b = r.metric_table(table, books(), b=50)
    c = r.metric_table(table, books(), b=50, seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert not a[['ci_lo', 'ci_hi']].equals(c[['ci_lo', 'ci_hi']])
    assert (a.ci_lo <= a.ci_hi).all()


def test_h3_verdict():
    assert r.verdict_h3(0.001) == 'supported'
    assert r.verdict_h3(0.0) == 'not supported'


def write_books(tmp_path, with_llm=True):
    rq1, rq2 = tmp_path / 'rq1', tmp_path / 'rq2'
    rq1.mkdir(); rq2.mkdir()
    cit = [dict(term=r.LEVELS[l], x=x, umf=u, lmf=lo) for l, s in CITIZEN.items() for x, u, lo in zip(s.x, s.umf, s.lmf)]
    pd.DataFrame(cit).to_csv(rq1 / 'citizen_it2.csv', index=False)
    grp = [dict(term=r.LEVELS[l], group=g, x=x, mu=u) for g in ['b|g', 'a|g'] for l, s in CITIZEN.items()
           for x, u in zip(s.x, s.umf)]
    pd.DataFrame(grp).to_csv(rq1 / 'group_t1.csv', index=False)
    if with_llm:
        rows = [dict(prompt=p, term=r.LEVELS[l], n_models=4, x=x, umf=u if p == 'P1' else 1.0, lmf=lo if p == 'P1' else 0.0)
                for p in ['P1', 'P2'] for l, s in LLM.items() for x, u, lo in zip(s.x, s.umf, s.lmf)]
        pd.DataFrame(rows).to_csv(rq2 / 'llm_it2.csv', index=False)
    return rq1, rq2


def test_missing_llm_codebook_exits_with_message(tmp_path):
    rq1, rq2 = write_books(tmp_path, with_llm=False)
    with pytest.raises(SystemExit, match='WP4'):
        r.load_codebooks(rq1, rq2)


def test_load_codebooks_uses_p1_and_groups_as_t1(tmp_path):
    loaded = r.load_codebooks(*write_books(tmp_path))
    assert list(loaded) == [r.CITIZEN, r.LLM, 'a|g', 'b|g']
    np.testing.assert_array_equal(loaded[r.LLM]['None'].umf, LLM['None'].umf)
    for level in r.SEVERITY:
        np.testing.assert_array_equal(loaded['a|g'][level].umf, loaded['a|g'][level].lmf)


def test_load_items_degree_and_reply_word(tmp_path, monkeypatch):
    from src.data import bhh
    per_item = tmp_path / 'per_item.csv'
    pd.DataFrame({'item_id': [3, 1], 'S3_word': [4, 0], 'other': [0, 0]}).to_csv(per_item, index=False)
    degree = pd.DataFrame({'item_id': [0, 1, 2, 3], 'k_llm_crisis': [0, 2, 9, 7], 'n_llm_observed': [9, 9, 9, 8]})
    monkeypatch.setattr(bhh, 'load_crisis_degree', lambda root=None: degree)
    out = r.load_items(per_item)
    assert out.item_id.tolist() == [1, 3]
    assert out.m.tolist() == [100 * 2 / 9, 100 * 7 / 8]
    assert out.reply_word.tolist() == ['Harmful/Unacceptable', 'Fully Appropriate']


def digest(directory):
    return {p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob('*')) if p.is_file()}


def test_end_to_end_byte_identical(tmp_path):
    data = items(np.r_[np.zeros(5), [25] * 3, np.linspace(30, 100, 12)])
    data['reply_word'] = [APPROPRIATENESS_WORDS[i % 5] for i in range(len(data))]
    hashes = []
    for name in ['a', 'b']:
        r.run(data, books(), tmp_path / name, b=30)
        hashes.append(digest(tmp_path / name))
    assert hashes[0] == hashes[1]
    assert {'per_item.csv', 'metrics.csv', 'flip_matrix.csv', 'SUMMARY.md', 'figures/flip_matrix.png',
            'figures/flip_rates.png', 'figures/level_words.png'} <= set(hashes[0])
    text = (tmp_path / 'a' / 'SUMMARY.md').read_text()
    assert '## Deviations' in text and 'All-zero fallback' in text and '## H3' in text
