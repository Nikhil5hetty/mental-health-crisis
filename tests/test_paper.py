"""Paper build checks: numbers come from results/, generated files are current, no hand-typed numbers."""
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'paper/scripts'))

import check_numbers  # noqa: E402
from paper.scripts import export_numbers, export_tables  # noqa: E402

needs_results = pytest.mark.skipif(not (ROOT / 'results/rq2/metrics.csv').is_file(), reason='results absent')


def test_no_hand_typed_numbers_in_paper():
    assert check_numbers.main() == 0


@pytest.mark.parametrize('text,expected', [
    ('We found 0.226 here.', ['0.226']),
    ('There were 190 items.', ['190']),
    ('In math $x = 12$.', ['12']),
    ('Allowed: S2, RQ1, IT2, type-1, 2-tuple, $k/4$, 143, 2026, \\cite{x2020}.', []),
    ('% 0.999 in a comment\n\\RtwoMThreeRecallSTwo{} macro', []),
])
def test_checker_flags_only_hand_typed_numbers(text, expected):
    found = [tok for _, tok, _ in check_numbers.violations(text, check_numbers.load_allowlist())]
    assert found == expected


def test_macro_names():
    assert export_numbers.camel('llama3.1:8b') == 'LlamaThreeOneEightB'
    assert export_numbers.camel('native|18-34') == 'NativeOneEightThreeFour'
    assert export_numbers.word_decimals('centroid_shift_it2') == 2 and export_numbers.word_decimals('jaccard_it2') == 3
    assert export_numbers.camel('M3_recall') == 'MThreeRecall'
    assert export_numbers.camel('S3[1-maxJ]-S2[run_std]') == 'SThreeOneMaxJSTwoRunStd'
    assert export_numbers.fmt_num(-0.0001) == '0.000'
    assert export_numbers.fmt_num(-0.07) == r'\ensuremath{-}0.070'


@needs_results
def test_numbers_tex_is_current():
    committed = (ROOT / 'paper/generated/numbers.tex').read_text(encoding='utf-8')
    assert committed == export_numbers.render(export_numbers.build()), 'Run: python -I paper/scripts/export_numbers.py'


@needs_results
def test_tables_are_current(tmp_path, monkeypatch):
    monkeypatch.setattr(export_tables, 'TAB', tmp_path / 'tables')
    monkeypatch.setattr(export_tables, 'FIG', tmp_path / 'figures')
    export_tables.main()
    for f in sorted((tmp_path / 'tables').glob('*.tex')):
        committed = ROOT / 'paper/generated/tables' / f.name
        assert committed.read_text(encoding='utf-8') == f.read_text(encoding='utf-8'), f.name
    assert sorted(f.name for f in (tmp_path / 'tables').glob('*.tex')) == \
        sorted(f.name for f in (ROOT / 'paper/generated/tables').glob('*.tex'))
    for f in sorted((tmp_path / 'figures').glob('*.png')):
        assert (ROOT / 'paper/figures' / f.name).read_bytes() == f.read_bytes(), f.name


# SUMMARY.md values were formatted from full-precision floats; metrics.csv keeps 6 decimals. These CounselBench
# values sit exactly on a rounding half in the CSV and therefore differ by 0.001 in the third decimal.
KNOWN_ROUNDING_CASES = {
    ('counselbench', ('0.026', '0.019', '0.033')),
    ('counselbench', ('0.035', '0.026', '0.045')),
    ('counselbench', ('0.571', '0.555', '0.586')),
    ('counselbench', ('0.573', '0.556', '0.588')),
    ('counselbench', ('0.597', '0.579', '0.616')),
}


@needs_results
def test_paper_numbers_match_recorded_summaries():
    pat = re.compile(r'(-?\d+\.\d{3}) \[(-?\d+\.\d{3}), (-?\d+\.\d{3})\]')
    fmt = lambda x: export_numbers.fmt_num(x).replace('\\ensuremath{-}', '-')  # noqa: E731
    mismatches = set()
    for part in ['rq1', 'rq2', 'counselbench', 'demo']:
        frames = [pd.read_csv(ROOT / f'results/{part}/metrics.csv')]
        if part == 'rq2':
            frames.append(pd.read_csv(ROOT / 'results/rq2/sensitivity.csv'))
        d = pd.concat(frames).dropna(subset=['ci_lo'])
        paper = {(fmt(r.estimate), fmt(r.ci_lo), fmt(r.ci_hi)) for r in d.itertuples()}
        for t in pat.findall((ROOT / f'results/{part}/SUMMARY.md').read_text(encoding='utf-8')):
            if t not in paper and t[0] != '-0.000':  # the paper prints negative zero as 0.000
                mismatches.add((part, t))
    assert mismatches == KNOWN_ROUNDING_CASES


# --------------------------------------------------------------------------- word-meaning study

needs_words = pytest.mark.skipif(not (ROOT / 'results/words/rq3/metrics.csv').is_file(), reason='word results absent')


@needs_words
def test_word_numbers_match_recorded_summaries():
    """Every "estimate [lo, hi]" in the word-study SUMMARY.md files has the same formatting in the paper macros."""
    fmt = lambda x, d: export_numbers.fmt_num(x, d).replace('\\ensuremath{-}', '-')  # noqa: E731
    paper = set()
    for rel in ['rq2/metrics.csv', 'rq2/sensitivity.csv', 'rq3/metrics.csv', 'rq3/sensitivity_posthoc.csv']:
        d = pd.read_csv(ROOT / 'results/words' / rel).dropna(subset=['estimate'])
        for r in d.itertuples():
            dec = export_numbers.word_decimals(r.metric)
            paper.add((fmt(r.estimate, dec), fmt(r.ci_lo, dec), fmt(r.ci_hi, dec)))
    pat = re.compile(r'(-?\d+\.\d{2,3}) \[(-?\d+\.\d{2,3}), (-?\d+\.\d{2,3})\]')
    missing = set()
    for part in ['rq2', 'rq3']:
        for t in pat.findall((ROOT / f'results/words/{part}/SUMMARY.md').read_text(encoding='utf-8')):
            if t not in paper:
                missing.add((part, t))
    assert not missing


@needs_words
def test_hypothesis_verdicts_are_copied_not_recomputed():
    verdicts = export_tables.recorded_verdicts()
    assert len(verdicts) == 13 and set(verdicts.values()) <= {'supported', 'not supported'}
    table = (ROOT / 'paper/generated/tables/hypotheses.tex').read_text(encoding='utf-8')
    for verdict in verdicts.values():
        assert verdict in table
    assert verdicts[('H2a', 'P1', 'llm_it2')] == 'not supported' and verdicts[('H3', 'P1', 'llm_vs_citizen')] == 'supported'


@needs_words
def test_plan_frozen_before_data():
    m = export_numbers.build()
    order = [m.defs[k] for k in ['PlanWordsUtc', 'CapphraseFetchedUtc', 'ElicitStartUtc', 'ElicitEndUtc']]
    assert order == sorted(order)
