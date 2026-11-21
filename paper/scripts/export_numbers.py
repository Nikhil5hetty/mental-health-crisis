#!/usr/bin/env python3
"""Write paper/generated/numbers.tex: one LaTeX macro per reported number.

Sources (nothing is typed by hand):
- results/{rq1,rq2,counselbench,demo}/metrics.csv and results/rq2/sensitivity.csv
- results/rq1/per_item.csv for two data descriptives (items with missing labels, min observed)
- experiment constants (seed, bootstrap size, resampling draws) imported from the experiment code
- git history for the commits that froze the analysis plans

Naming: <Part><Metric><System/Group><Stratum>, digits spelled out (S2 -> STwo, 0.25 -> ZeroTwoFive),
'overall' stratum omitted. For each row three macros exist:
  \\X      estimate                  e.g. 0.226
  \\XCI    95% CI                    e.g. [0.119, 0.341]
  \\XFull  estimate and CI           e.g. 0.226 [0.119, 0.341]
Count rows (no CI) get only \\X. The macro index is written to paper/generated/numbers_index.csv.

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
DIGITS = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']
PARTS = {  # part -> (macro prefix, metrics file, column holding system/group)
    'rq1': ('Rone', 'results/rq1/metrics.csv', 'group'),
    'rq2': ('Rtwo', 'results/rq2/metrics.csv', 'system'),
    'counselbench': ('Cb', 'results/counselbench/metrics.csv', 'group'),
    'demo': ('Demo', 'results/demo/metrics.csv', 'group'),
}


def camel(text: str) -> str:
    """'M3_recall' -> 'MThreeRecall'; 'S3[1-maxJ]-S2[run_std]' -> 'SThreeOneMaxJSTwoRunStd'."""
    out = []
    for token in re.findall(r'[A-Za-z]+|\d', str(text)):
        out.append(DIGITS[int(token)] if token.isdigit() else token[0].upper() + token[1:])
    return ''.join(out)


def fmt_num(x: float, decimals: int = 3) -> str:
    """Format like the experiment SUMMARY.md files (Python float formatting).

    metrics.csv stores 6 decimals, while the SUMMARY.md files were formatted from full-precision values,
    so a value stored exactly at a half (e.g. 0.357500) can differ by 0.001 in the third decimal.
    tests/test_paper.py lists the known cases.
    """
    s = f'{x:.{decimals}f}'
    if s.startswith('-'):
        s = s[1:] if float(s) == 0 else r'\ensuremath{-}' + s[1:]  # no "-0.000"
    return s


def is_count(row) -> bool:
    return math.isnan(row.ci_lo) and float(row.estimate).is_integer()


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

    def add_row(self, base: str, row, source: str) -> None:
        if is_count(row):
            self.add(base, str(int(row.estimate)), source)
            return
        if math.isnan(row.estimate):
            self.add(base, 'n/a', source)
            self.add(base + 'CI', 'n/a', source)
            self.add(base + 'Full', 'n/a', source)
            return
        est = fmt_num(row.estimate)
        self.add(base, est, source)
        if math.isnan(row.ci_lo):
            self.add(base + 'CI', 'n/a', source)
            self.add(base + 'Full', est, source)
        else:
            ci = f'[{fmt_num(row.ci_lo)}, {fmt_num(row.ci_hi)}]'
            self.add(base + 'CI', ci, source)
            self.add(base + 'Full', f'{est} {ci}', source)


def git_first_commit(path: str) -> tuple[str, str]:
    """Short hash and ISO date of the commit that added ``path``."""
    out = subprocess.run(['git', 'log', '--diff-filter=A', '--format=%h %ad', '--date=short', '--', path],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    if len(out) < 2:
        raise RuntimeError(f'No commit adds {path}')
    return out[-2], out[-1]  # oldest add


def build() -> Macros:
    m = Macros()
    for part, (prefix, rel, col) in PARTS.items():
        df = pd.read_csv(ROOT / rel)
        for r in df.itertuples(index=False):
            stratum = '' if r.stratum == 'overall' else camel(r.stratum)
            m.add_row(prefix + camel(r.metric) + camel(getattr(r, col)) + stratum, r, f'{rel}:{r.metric}|{getattr(r, col)}|{r.stratum}')
    sens = pd.read_csv(ROOT / 'results/rq2/sensitivity.csv')
    for r in sens.itertuples(index=False):
        base = 'RtwoSens' + camel(r.parameter) + camel(f'{r.value:g}') + camel(r.metric) + camel(r.system)
        m.add_row(base, r, f'results/rq2/sensitivity.csv:{r.parameter}={r.value:g}|{r.metric}|{r.system}')

    rq1 = pd.read_csv(ROOT / 'results/rq1/per_item.csv')
    m.add('RoneNItemsWithMissingLabels', str(int((rq1.n_llm_missing > 0).sum())), 'results/rq1/per_item.csv')
    m.add('RoneMinLlmObserved', str(int(rq1.n_llm_observed.min())), 'results/rq1/per_item.csv')
    m.add('RoneMaxLlmObserved', str(int(rq1.n_llm_observed.max())), 'results/rq1/per_item.csv')

    from experiments.rq2 import run_rq2 as rq2  # constants only; no data is read on import
    m.add('BootB', f'{rq2.B:,}'.replace(',', '{,}'), 'experiments/rq2/run_rq2.py:B')
    m.add('BootSeed', str(rq2.SEED), 'experiments/rq2/run_rq2.py:SEED')
    m.add('ResampleDraws', f'{rq2.N_RESAMPLE:,}'.replace(',', '{,}'), 'experiments/rq2/run_rq2.py:N_RESAMPLE')

    for name, path in [('PlanRtwo', 'docs/ANALYSIS_PLAN_RQ2.md'), ('PlanRoneDemo', 'docs/ANALYSIS_PLAN_RQ1_DEMO.md')]:
        sha, date = git_first_commit(path)
        m.add(name + 'Commit', r'\texttt{' + sha + '}', f'git log --diff-filter=A {path}')
        m.add(name + 'Date', date, f'git log --diff-filter=A {path}')
    return m


def main() -> None:
    GEN.mkdir(parents=True, exist_ok=True)
    m = build()
    lines = ['% Generated by paper/scripts/export_numbers.py. Do not edit by hand.']
    lines += [f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in m.defs.items()]
    (GEN / 'numbers.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    pd.DataFrame(m.index).to_csv(GEN / 'numbers_index.csv', index=False)
    print(f'Wrote {len(m.defs)} macros to paper/generated/numbers.tex')


if __name__ == '__main__':
    main()
