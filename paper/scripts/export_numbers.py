#!/usr/bin/env python3
"""Write paper/generated/numbers.tex: one LaTeX macro per reported number.

Sources (nothing is typed by hand):
- Word-meaning study (main study): results/words/rq1 (metrics, sensitivity, consistency, group Jaccard,
  respondent profile), results/words/rq2 (metrics, sensitivity = prompt P2, invalid answers, model parameters),
  results/words/rq3 (metrics, post-hoc sensitivity, flip matrix, per-item counts).
- Earlier study (motivation only): results/{rq1,rq2,counselbench,demo}/metrics.csv, results/rq2/sensitivity.csv.
- Constants imported from the experiment code (seeds, bootstrap sizes, elicitation settings).
- Git history for the commits that froze the analysis plans; model tags and digests from data/llm_answers.

Word-study macro names: W<one|two|three> + metric + term/comparison + group/model, CamelCase with digits spelled
out (llama3.1:8b -> LlamaThreeOneEightB, native|18-34 -> NativeOneEightThreeFour). For rows with a CI:
  \\X      estimate      \\XCI  [lo, hi]      \\XFull  estimate [lo, hi]
Count rows give only \\X. Shares also get \\XPct (percent, one decimal, no % sign).
Index of every macro and its source: paper/generated/numbers_index.csv.

Run from anywhere:  python -I paper/scripts/export_numbers.py
"""
from __future__ import annotations

import math
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GEN = ROOT / 'paper/generated'
W = ROOT / 'results/words'
DIGITS = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']
PARTS = {  # earlier study (motivation): part -> (macro prefix, metrics file, column holding system/group)
    'rq1': ('Rone', 'results/rq1/metrics.csv', 'group'),
    'rq2': ('Rtwo', 'results/rq2/metrics.csv', 'system'),
    'counselbench': ('Cb', 'results/counselbench/metrics.csv', 'group'),
    'demo': ('Demo', 'results/demo/metrics.csv', 'group'),
}


def camel(text: str) -> str:
    """'M3_recall' -> 'MThreeRecall'; 'llama3.1:8b' -> 'LlamaThreeOneEightB'."""
    out = []
    for token in re.findall(r'[A-Za-z]+|\d', str(text)):
        out.append(DIGITS[int(token)] if token.isdigit() else token[0].upper() + token[1:])
    return ''.join(out)


def fmt_num(x: float, decimals: int = 3) -> str:
    """Format like the experiment SUMMARY.md files (Python float formatting); no "-0.000"."""
    s = f'{x:.{decimals}f}'
    if s.startswith('-'):
        s = s[1:] if float(s) == 0 else r'\ensuremath{-}' + s[1:]
    return s


def fmt_int(n: int) -> str:
    return f'{int(n):,}'.replace(',', '{,}')


def is_count(row) -> bool:
    return math.isnan(row.ci_lo) and float(row.estimate).is_integer()


def word_decimals(metric: str) -> int:
    """Decimals as in results/words/*/SUMMARY.md: percentage-point quantities 2, everything else 3."""
    return 2 if any(k in metric for k in ('shift', 'centre', 'centroid')) else 3


class Macros:
    def __init__(self):
        self.defs: dict[str, str] = {}
        self.index: list[dict] = []

    def add(self, name: str, value: str, source: str) -> None:
        if not re.fullmatch(r'[A-Za-z]+', name):
            raise ValueError(f'Invalid macro name {name!r}')
        if name in self.defs:
            raise ValueError(f'Duplicate macro name {name!r} ({source})')
        self.defs[name] = value
        self.index.append({'macro': '\\' + name, 'value': value, 'source': source})

    def add_row(self, base: str, row, source: str, decimals: int = 3) -> None:
        if is_count(row):
            self.add(base, str(int(row.estimate)), source)
            return
        if math.isnan(row.estimate):
            for suffix in ('', 'CI', 'Full'):
                self.add(base + suffix, 'n/a', source)
            return
        est = fmt_num(row.estimate, decimals)
        self.add(base, est, source)
        if math.isnan(row.ci_lo):
            self.add(base + 'CI', 'n/a', source)
            self.add(base + 'Full', est, source)
        else:
            ci = f'[{fmt_num(row.ci_lo, decimals)}, {fmt_num(row.ci_hi, decimals)}]'
            self.add(base + 'CI', ci, source)
            self.add(base + 'Full', f'{est} {ci}', source)

    def add_share(self, base: str, share: float, source: str) -> None:
        self.add(base, fmt_num(share), source)
        self.add(base + 'Pct', fmt_num(100 * share, 1), source)


def git_first_commit(path: str) -> tuple[str, str]:
    """Short hash and ISO date of the commit that added ``path``."""
    out = subprocess.run(['git', 'log', '--diff-filter=A', '--format=%h %ad', '--date=short', '--', path],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    if len(out) < 2:
        raise RuntimeError(f'No commit adds {path}')
    return out[-2], out[-1]  # oldest add


# ----------------------------------------------------------------------------- word study

def rq1_value(metric: str, value: float) -> str:
    if metric in ('n', 'included', 'n_group_excluded'):
        return fmt_int(value)
    if metric == 'iqr' or metric == 'median':
        return f'{value:g}'
    if metric == 'fou_area':
        return fmt_num(value, 2)
    return fmt_num(value, 2 if 'centr' in metric else 3)


def words_rq1(m: Macros) -> None:
    src = 'results/words/rq1/metrics.csv'
    for r in pd.read_csv(W / 'rq1/metrics.csv').itertuples(index=False):
        group = '' if r.group == 'pooled' else camel(r.group)
        m.add('Wone' + camel(r.metric) + camel(r.term) + group, rq1_value(r.metric, r.value), f'{src}:{r.metric}|{r.term}|{r.group}')
    src = 'results/words/rq1/sensitivity.csv'
    for r in pd.read_csv(W / 'rq1/sensitivity.csv').itertuples(index=False):
        group = '' if r.group == 'pooled' else camel(r.group)
        value = fmt_int(r.value) if r.metric in ('n', 'included', 'n_group_excluded', 'respondents_excluded') \
            else rq1_value(r.metric, r.value)
        m.add('WoneSens' + camel(r.scenario) + camel(r.metric) + camel(r.term) + group, value,
              f'{src}:{r.scenario}|{r.metric}|{r.term}|{r.group}')
    sens = pd.read_csv(W / 'rq1/sensitivity.csv')
    for scenario, g in sens.groupby('scenario'):
        m.add('WoneSens' + camel(scenario) + 'MeanFouArea', fmt_num(g[g.metric == 'fou_area'].value.mean(), 2), f'{src}:{scenario} mean fou_area')
        m.add('WoneSens' + camel(scenario) + 'MeanJaccardPrimary',
              fmt_num(g[g.metric == 'jaccard_primary_it2'].value.mean()), f'{src}:{scenario} mean jaccard_primary_it2')
    removed = sens[sens.metric == 'respondents_excluded'].value
    m.add('WoneInattentionExcluded', fmt_int(removed.iloc[0]), f'{src}:respondents_excluded')

    c = pd.read_csv(W / 'rq1/consistency.csv').iloc[0]
    for col in ['n_pairs', 'consistent', 'ties', 'inconsistent']:
        m.add('WoneConsistency' + camel(col), fmt_int(c[col]), f'results/words/rq1/consistency.csv:{col}')
    for col in ['consistent_share', 'tie_share', 'inconsistent_share']:
        m.add_share('WoneConsistency' + camel(col), c[col], f'results/words/rq1/consistency.csv:{col}')

    gj = pd.read_csv(W / 'rq1/group_jaccard.csv')
    means = gj.groupby(['group_a', 'group_b']).jaccard.mean()
    for (a, b), v in means.items():
        m.add('WoneGroupJaccard' + camel(a) + camel(b), fmt_num(v), f'results/words/rq1/group_jaccard.csv:mean {a}|{b}')
    m.add('WoneGroupJaccardMin', fmt_num(means.min()), 'results/words/rq1/group_jaccard.csv:min of pair means')
    m.add('WoneGroupJaccardMax', fmt_num(means.max()), 'results/words/rq1/group_jaccard.csv:max of pair means')
    m.add('WoneGroupJaccardPairs', fmt_int(len(gj)), 'results/words/rq1/group_jaccard.csv:rows')

    met = pd.read_csv(W / 'rq1/metrics.csv')
    n_resp = met[(met.metric == 'n') & (met.group == 'pooled')].value
    if n_resp.nunique() != 1:
        raise ValueError('Pooled n differs between terms')
    m.add('WoneNRespondents', fmt_int(n_resp.iloc[0]), 'results/words/rq1/metrics.csv:n|pooled (all terms equal)')
    m.add('WoneNJudgements', fmt_int(n_resp.sum()), 'results/words/rq1/metrics.csv:sum of n|pooled')
    m.add('WoneNTerms', str(met.term.nunique()), 'results/words/rq1/metrics.csv:terms')
    m.add('WoneNGroups', str(met[met.group != 'pooled'].group.nunique()), 'results/words/rq1/metrics.csv:groups')
    excl = met[(met.metric == 'included') & (met.group != 'pooled')]
    m.add('WoneGroupCellsExcluded', str(int((excl.value == 0).sum())), 'results/words/rq1/metrics.csv:included==0')
    m.add('WoneNotInGroups', fmt_int(met[met.metric == 'n_group_excluded'].value.iloc[0]),
          'results/words/rq1/metrics.csv:n_group_excluded')
    areas = met[met.metric == 'fou_area'].set_index('term').value
    m.add('WoneWidestTerm', areas.idxmax(), 'results/words/rq1/metrics.csv:max fou_area')
    m.add('WoneNarrowestTerm', areas.idxmin(), 'results/words/rq1/metrics.csv:min fou_area')

    prof = pd.read_csv(W / 'rq1/respondent_profile.csv')
    for r in prof.itertuples(index=False):
        if r.field == 'all':
            continue
        base = 'WoneProfile' + camel(r.field) + camel(r.category.replace('+', ' plus'))
        m.add(base + 'N', fmt_int(r.n), f'results/words/rq1/respondent_profile.csv:{r.field}|{r.category}')
        m.add_share(base, r.share, f'results/words/rq1/respondent_profile.csv:{r.field}|{r.category}')
    edu = prof[(prof.field == 'education') & prof.category.isin(['Postgraduate', 'Bachelor'])]
    m.add_share('WoneProfileEducationDegree', edu.share.sum(), 'results/words/rq1/respondent_profile.csv:Postgraduate+Bachelor')
    uk_us = prof[(prof.field == 'country') & prof.category.isin(['United Kingdom', 'United States'])]
    m.add_share('WoneProfileCountryUkUs', uk_us.share.sum(), 'results/words/rq1/respondent_profile.csv:UK+US')


def words_rq2(m: Macros) -> None:
    for prefix, rel in [('Wtwo', 'rq2/metrics.csv'), ('WtwoPtwo', 'rq2/sensitivity.csv')]:
        for r in pd.read_csv(W / rel).itertuples(index=False):
            m.add_row(prefix + camel(r.metric) + camel(r.term) + camel(r.model), r,
                      f'results/words/{rel}:{r.metric}|{r.term}|{r.model}', word_decimals(r.metric))

    inv = pd.read_csv(W / 'rq2/invalid.csv')
    src = 'results/words/rq2/invalid.csv'
    cols = ['n_rows', 'n_valid', 'n_missing', 'n_reasked', 'n_length', 'n_superseded']
    for (model, prompt), g in inv.groupby(['model', 'prompt_id']):
        base = 'WtwoAnswers' + camel(model) + camel(prompt)
        for col in cols:
            m.add(base + camel(col), fmt_int(g[col].sum()), f'{src}:sum {col}|{model}|{prompt}')
        m.add_share(base + 'InvalidRate', g.n_missing.sum() / g.n_rows.sum(), f'{src}:{model}|{prompt} missing/rows')
    for col in cols:
        m.add('WtwoAnswersTotal' + camel(col), fmt_int(inv[col].sum()), f'{src}:sum {col}')
    m.add_share('WtwoAnswersTotalInvalidRate', inv.n_missing.sum() / inv.n_rows.sum(), f'{src}:missing/rows')
    for prompt, g in inv.groupby('prompt_id'):
        m.add('WtwoExcludedCells' + camel(prompt), str(int((g.included == 0).sum())), f'{src}:included==0|{prompt}')
    m.add('WtwoNModels', str(inv.model.nunique()), f'{src}:models')

    par = pd.read_csv(W / 'rq2/codebooks/llm_model_params.csv')
    src = 'results/words/rq2/codebooks/llm_model_params.csv'
    for prompt, g in par.groupby('prompt'):
        coll = g[g.q05 == g.q95]
        m.add('WtwoCollapses' + camel(prompt), str(len(coll)), f'{src}:q05==q95|{prompt}')
        m.add('WtwoCells' + camel(prompt), str(len(g)), f'{src}:rows|{prompt}')
        for model, gm in g.groupby('model'):
            m.add('WtwoCollapses' + camel(prompt) + camel(model), str(int((gm.q05 == gm.q95).sum())),
                  f'{src}:q05==q95|{prompt}|{model}')


def words_rq3(m: Macros) -> None:
    for r in pd.read_csv(W / 'rq3/metrics.csv').itertuples(index=False):
        m.add_row('Wthree' + camel(r.metric) + camel(r.comparison), r, f'results/words/rq3/metrics.csv:{r.metric}|{r.comparison}')
    for r in pd.read_csv(W / 'rq3/sensitivity_posthoc.csv').itertuples(index=False):
        m.add_row('WthreePosthoc' + camel(r.analysis) + camel(r.metric) + camel(r.comparison), r,
                  f'results/words/rq3/sensitivity_posthoc.csv:{r.analysis}|{r.metric}|{r.comparison}')
    for r in pd.read_csv(W / 'rq3/flip_matrix.csv').itertuples(index=False):
        m.add('WthreeMatrix' + camel(r.comparison) + camel(r.reference_action) + camel(r.llm_action), fmt_int(r.n),
              f'results/words/rq3/flip_matrix.csv:{r.comparison}|{r.reference_action}|{r.llm_action}')
    t = pd.read_csv(W / 'rq3/per_item.csv', keep_default_na=False)
    src = 'results/words/rq3/per_item.csv'
    flip = t.action_citizen != t.action_llm
    fb = (t.fallback_citizen == 1) | (t.fallback_llm == 1)
    m.add('WthreeNItems', fmt_int(len(t)), f'{src}:rows')
    m.add('WthreeNDistinctM', str(t.m.round(6).nunique()), f'{src}:distinct m')
    m.add('WthreeNFlips', str(int(flip.sum())), f'{src}:action_citizen != action_llm')
    m.add('WthreeNUnder', str(int((t.action_llm < t.action_citizen).sum())), f'{src}:action_llm < action_citizen')
    m.add('WthreeNOver', str(int((t.action_llm > t.action_citizen).sum())), f'{src}:action_llm > action_citizen')
    m.add('WthreeNFlipsFallback', str(int((flip & fb).sum())), f'{src}:flips with a fallback')
    m.add('WthreeNFlipsClean', str(int((flip & ~fb).sum())), f'{src}:flips without fallback')
    m.add('WthreeNItemsClean', str(int((~fb).sum())), f'{src}:items without fallback')
    for col in [c for c in t.columns if c.startswith('fallback_')]:
        m.add('WthreeNFallback' + camel(col[len('fallback_'):]), fmt_int(t[col].sum()), f'{src}:sum {col}')
    m.add('WthreeMinObserved', str(int(t.n_llm_observed.min())), f'{src}:min n_llm_observed')
    m.add('WthreeMaxObserved', str(int(t.n_llm_observed.max())), f'{src}:max n_llm_observed')
    m.add('WthreeNZeroOrHundred', fmt_int(t.m.isin([0, 100]).sum()), f'{src}:m in {{0, 100}}')


def words_constants(m: Macros) -> None:
    from experiments.words import elicit_llm, run_rq1, run_rq2, run_rq3  # constants only; nothing read on import
    m.add('WSeed', str(run_rq2.SEED), 'experiments/words/run_rq2.py:SEED')
    m.add('WoneBootB', fmt_int(run_rq1.B), 'experiments/words/run_rq1.py:B')
    m.add('WtwoBootB', fmt_int(run_rq2.B), 'experiments/words/run_rq2.py:B')
    m.add('WthreeBootB', fmt_int(run_rq3.B), 'experiments/words/run_rq3.py:B')
    m.add('WMinN', str(run_rq2.MIN_N), 'experiments/words/run_rq2.py:MIN_N')
    m.add('WSamples', str(elicit_llm.N_SAMPLES), 'experiments/words/elicit_llm.py:N_SAMPLES')
    m.add('WTemperature', f'{elicit_llm.TEMPERATURE:.1f}', 'experiments/words/elicit_llm.py:TEMPERATURE')
    m.add('WSeedBase', str(elicit_llm.SEED_BASE), 'experiments/words/elicit_llm.py:SEED_BASE')
    m.add('WMaxReasks', str(elicit_llm.MAX_REASKS), 'experiments/words/elicit_llm.py:MAX_REASKS')
    m.add('WNumPredict', str(elicit_llm.NUM_PREDICT), 'experiments/words/elicit_llm.py:NUM_PREDICT')
    m.add('WNPromptsPerModel', str(len(elicit_llm.PROMPT_IDS)), 'experiments/words/elicit_llm.py:PROMPT_IDS')
    m.add('WNHedge', str(len(run_rq2.HEDGE)), 'experiments/words/run_rq2.py:HEDGE')
    m.add('WNExtreme', str(len(run_rq2.EXTREME)), 'experiments/words/run_rq2.py:EXTREME')

    answers = pd.read_csv(ROOT / 'data/llm_answers/answers.csv', usecols=['model', 'digest'])
    for model, g in answers.groupby('model'):
        if g.digest.nunique() != 1:
            raise ValueError(f'Several digests for {model}')
        m.add('WModel' + camel(model), r'\texttt{' + model + '}', 'data/llm_answers/answers.csv:model')
        m.add('WDigest' + camel(model), r'\texttt{' + g.digest.iloc[0][:12] + '}', 'data/llm_answers/answers.csv:digest')


# ----------------------------------------------------------------------------- earlier study (motivation)

def earlier_study(m: Macros) -> None:
    for part, (prefix, rel, col) in PARTS.items():
        df = pd.read_csv(ROOT / rel)
        for r in df.itertuples(index=False):
            stratum = '' if r.stratum == 'overall' else camel(r.stratum)
            m.add_row(prefix + camel(r.metric) + camel(getattr(r, col)) + stratum, r, f'{rel}:{r.metric}|{getattr(r, col)}|{r.stratum}')
    sens = pd.read_csv(ROOT / 'results/rq2/sensitivity.csv')
    for r in sens.itertuples(index=False):
        base = 'RtwoSens' + camel(r.parameter) + camel(f'{r.value:g}') + camel(r.metric) + camel(r.system)
        m.add_row(base, r, f'results/rq2/sensitivity.csv:{r.parameter}={r.value:g}|{r.metric}|{r.system}')


def build() -> Macros:
    m = Macros()
    words_rq1(m)
    words_rq2(m)
    words_rq3(m)
    words_constants(m)
    for name, path in [('PlanWords', 'docs/ANALYSIS_PLAN_WORDS.md'), ('PlanRtwo', 'docs/ANALYSIS_PLAN_RQ2.md'),
                       ('PlanRoneDemo', 'docs/ANALYSIS_PLAN_RQ1_DEMO.md')]:
        sha, date = git_first_commit(path)
        m.add(name + 'Commit', r'\texttt{' + sha + '}', f'git log --diff-filter=A {path}')
        m.add(name + 'Date', date, f'git log --diff-filter=A {path}')
    timeline(m)
    earlier_study(m)
    return m


def timeline(m: Macros) -> None:
    """UTC times showing the order: plan frozen, CAPphrase fetched, LLM answers elicited."""
    import json
    out = subprocess.run(['git', 'log', '--diff-filter=A', '--format=%aI', '--', 'docs/ANALYSIS_PLAN_WORDS.md'],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    plan = pd.Timestamp(out[-1]).tz_convert('UTC')
    fetched = pd.Timestamp(json.loads((ROOT / 'data/MANIFEST.json').read_text())['sources']['capphrase']['download_date_utc'])
    stamps = pd.concat([pd.read_csv(ROOT / f'data/llm_answers/{f}', usecols=['timestamp']) for f in
                        ['answers.csv', 'superseded.csv']]).timestamp.map(pd.Timestamp)
    for name, t, src in [('PlanWordsUtc', plan, 'git log docs/ANALYSIS_PLAN_WORDS.md (author time)'),
                         ('CapphraseFetchedUtc', fetched, 'data/MANIFEST.json:sources.capphrase.download_date_utc'),
                         ('ElicitStartUtc', stamps.min(), 'data/llm_answers/*.csv:min timestamp'),
                         ('ElicitEndUtc', stamps.max(), 'data/llm_answers/*.csv:max timestamp')]:
        if t > plan or name == 'PlanWordsUtc':
            m.add(name, t.strftime('%Y-%m-%d %H:%M'), src)
        else:
            raise ValueError(f'{name} precedes the frozen plan')


def render(m: Macros) -> str:
    lines = ['% Generated by paper/scripts/export_numbers.py. Do not edit by hand.']
    lines += [f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in m.defs.items()]
    return '\n'.join(lines) + '\n'


def main() -> None:
    GEN.mkdir(parents=True, exist_ok=True)
    m = build()
    (GEN / 'numbers.tex').write_text(render(m), encoding='utf-8')
    pd.DataFrame(m.index).to_csv(GEN / 'numbers_index.csv', index=False, lineterminator='\n')
    print(f'Wrote {len(m.defs)} macros to paper/generated/numbers.tex')


if __name__ == '__main__':
    main()
