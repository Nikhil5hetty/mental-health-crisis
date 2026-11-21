"""Synthetic tests for the RQ1, CounselBench and demo experiment scripts (no data needed)."""
import numpy as np
import pandas as pd
import pytest

from experiments.common import write_csv
from experiments.counselbench import run_cb
from experiments.demo import run_demo
from experiments.rq1 import run_rq1
from src.cww import encode_scores

RNG = np.random.default_rng(7)

# ------------------------------------------------------------------ RQ1 rules


@pytest.mark.parametrize('mu,expected', [(0.51, 'crisis'), (0.5, 'tie'), (0.49, 'no_crisis'), (1, 'crisis'), (0, 'no_crisis')])
def test_majority_rule(mu, expected):
    assert run_rq1.majority(mu) == expected


def test_modal_category_and_ties():
    assert run_rq1.modal_category({'a': 2, 'b': 1, 'c': 1}) == 'a'
    assert run_rq1.modal_category({'a': 2, 'b': 2, 'c': 0}) == 'tie'


def test_fuzzy_category_agreement():
    p = {'a': 0.5, 'b': 0.5, 'c': 0.0}
    assert run_rq1.fuzzy_category_agreement(p, p) == 1.0
    assert run_rq1.fuzzy_category_agreement({'a': 1.0, 'b': 0.0}, {'a': 0.0, 'b': 1.0}) == 0.0
    # sum of mins = 0.25 + 0.25 = 0.5; sum of maxes = 0.75 + 0.75 = 1.5
    assert run_rq1.fuzzy_category_agreement({'a': 0.75, 'b': 0.25}, {'a': 0.25, 'b': 0.75}) == pytest.approx(0.5 / 1.5)


def synthetic_rq1(n=60):
    k_h = RNG.integers(0, 5, n)
    n_obs = RNG.integers(6, 10, n)
    k_l = np.minimum(RNG.integers(0, 10, n), n_obs)
    cats = np.array(['suicidal_ideation', 'no_crisis', 'tie'])
    t = pd.DataFrame({'item_id': np.arange(n), 'n_llm_observed': n_obs, 'n_llm_missing': 9 - n_obs,
                      'mu_H': k_h / 4, 'mu_L': k_l / n_obs,
                      'human_modal_category': RNG.choice(cats, n), 'llm_modal_category': RNG.choice(cats, n),
                      'fuzzy_category_agreement': RNG.random(n)})
    t['human_majority'] = t.mu_H.map(run_rq1.majority)
    t['llm_majority'] = t.mu_L.map(run_rq1.majority)
    t['human_split'] = (t.mu_H > 0) & (t.mu_H < 1)
    return t

# ------------------------------------------------------------------ CounselBench rules


def test_toxicity_codebook_decodes_own_words_and_scale_words():
    book = run_cb.toxicity_codebook()
    assert book.words == run_cb.TOX_WORDS
    for w, s in zip(book.words, book.sets):
        assert book.decode(s)[0] == w
    assert book.decode(encode_scores([5, 5, 5, 5, 5]))[0] == 'Extremely toxic'
    assert book.to_two_tuple(1.6)[0] == 'Somewhat not toxic'


def test_modal_membership_uses_bin_centres():
    assert run_cb.modal_membership(encode_scores([1, 1, 2, 2, 5])) == pytest.approx(0.4)
    assert run_cb.modal_membership(encode_scores([3, 3, 3, 3, 3])) == pytest.approx(1.0)


def synthetic_cb(n_per=12):
    rows = []
    for r in run_cb.RESPONDERS:
        for i in range(n_per):
            row = {'reply_id': f'q{i}|{r}', 'responder': r, 'k_medical_yes': int(RNG.integers(0, 6)),
                   'k_medical_not_sure': int(RNG.integers(0, 3)), 'k_factual_not_sure': int(RNG.integers(0, 3)),
                   'factual_unanimous': bool(RNG.random() < 0.2), 'toxicity_contested': bool(RNG.random() < 0.5),
                   'toxicity_word_differs': bool(RNG.random() < 0.4), 'toxicity_decoded_word': int(RNG.integers(0, 5))}
            row['medical_split'] = 1 <= row['k_medical_yes'] <= 4
            for d in run_cb.LIKERT:
                row[f'{d}_range'] = int(RNG.integers(0, 5))
                row[f'{d}_modal_membership'] = float(RNG.choice([0.2, 0.4, 0.6, 0.8, 1.0]))
            for k in run_cb.RISKS:
                row[f'k_{k}_copied'] = int(RNG.integers(0, 4))
            rows.append(row)
    return pd.DataFrame(rows)

# ------------------------------------------------------------------ demo rules


def synthetic_demo(n=40):
    t = pd.DataFrame({'item_id': np.arange(n), 'oracle_action': RNG.integers(0, 4, n),
                      'p_fuzzy_action': RNG.integers(0, 4, n), 'p_crisp_action': RNG.integers(0, 4, n)})
    for p in ['p_fuzzy', 'p_crisp', 'oracle']:
        t[f'{p}_tied'] = RNG.random(n) < 0.1
    return t


def test_worked_cases_rule():
    t = pd.DataFrame({'item_id': [5, 3, 9, 1, 7, 2, 8, 4, 6, 0],
                      'oracle_action': [3, 3, 3, 3, 0, 0, 1, 2, 0, 0],
                      'p_crisp_action': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                      'p_fuzzy_action': [3, 0, 0, 0, 3, 2, 2, 2, 1, 0],
                      'mu_L': 0.0, 'mu_H': 0.0, 'llm_majority': 'no_crisis', 'S2_word': 'x', 'S3_word': 'x'})
    cases = run_demo.worked_cases(t)
    # oracle - crisp gaps: ids 3, 5, 9 have 3 (ties -> lowest id), then id 1 and 4 have 2 -> id 1.
    assert cases.item_id.tolist()[:4] == [3, 5, 9, 1]
    # fuzzy - crisp among the rest: id 7 (3), then ids 2, 4, 8 (2).
    assert cases.item_id.tolist()[4:] == [7, 2, 4, 8]
    assert cases.item_id.is_unique and len(cases) == 8


def test_demo_metrics_on_known_actions():
    t = pd.DataFrame({'item_id': range(4), 'oracle_action': [3, 2, 0, 1], 'p_fuzzy_action': [3, 1, 0, 2],
                      'p_crisp_action': [1, 1, 0, 1], 'p_fuzzy_tied': False, 'p_crisp_tied': False,
                      'oracle_tied': False})
    m = run_demo.compute_metrics(t).set_index(['metric', 'group', 'stratum']).estimate
    assert m[('under_escalation_rate', 'P-fuzzy', 'oracle_A2_A3')] == 0.5
    assert m[('under_escalation_rate', 'P-crisp', 'oracle_A2_A3')] == 1.0
    assert m[('DIFF_under_escalation_rate', 'P-fuzzy-P-crisp', 'oracle_A2_A3')] == -0.5
    assert m[('over_escalation_rate', 'P-fuzzy', 'overall')] == 0.25
    assert m[('exact_agreement_with_oracle', 'P-crisp', 'overall')] == 0.5

# ------------------------------------------------------------------ seed: reruns are byte-identical


@pytest.mark.parametrize('module,factory', [(run_rq1, synthetic_rq1), (run_cb, synthetic_cb),
                                            (run_demo, synthetic_demo)])
def test_reruns_byte_identical(tmp_path, module, factory):
    t = factory()
    first, second = tmp_path / 'a.csv', tmp_path / 'b.csv'
    write_csv(module.compute_metrics(t), first)
    write_csv(module.compute_metrics(t.copy()), second)
    assert first.read_bytes() == second.read_bytes()
    assert len(first.read_bytes()) > 0
