"""Shared helpers for the RQ1, CounselBench and demo experiments.

Bootstrap, metric and verdict helpers are imported unchanged from
experiments/rq2/run_rq2.py (RQ2 code and outputs are not modified).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # python -I drops the cwd; trusted project code only

from experiments.rq2 import run_rq2 as rq2  # noqa: E402

SEED, B = rq2.SEED, rq2.B
boot_index, summarise, verdict = rq2.boot_index, rq2.summarise, rq2.verdict
m_mean, m_qwk, m_spearman, spearman, div = rq2.m_mean, rq2.m_qwk, rq2.m_spearman, rq2.spearman, rq2._div

COLUMNS = ['metric', 'group', 'stratum', 'estimate', 'ci_lo', 'ci_hi']


def m_cond_mean(x, mask):
    """Mean of x over items where mask is true, per bootstrap row (NaN if none)."""
    x, mask = np.asarray(x, float), np.asarray(mask, bool)
    return lambda idx: div((x[idx] * mask[idx]).sum(1), mask[idx].sum(1))


def est_row(metric: str, group: str, stratum: str, fn, n: int) -> dict:
    est, lo, hi = summarise(fn, n)
    return dict(metric=metric, group=group, stratum=stratum, estimate=est, ci_lo=lo, ci_hi=hi)


def count_row(metric: str, group: str, stratum: str, value) -> dict:
    return dict(metric=metric, group=group, stratum=stratum, estimate=value, ci_lo=np.nan, ci_hi=np.nan)


def to_frame(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=COLUMNS)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, float_format='%.6f')


class Table:
    """Look up and format rows of a metrics frame for generated summaries."""

    def __init__(self, metrics: pd.DataFrame):
        self.m = metrics.set_index(['metric', 'group', 'stratum'])

    def get(self, metric, group, stratum='overall'):
        return self.m.loc[(metric, group, stratum)]

    def fmt(self, metric, group, stratum='overall', d=3) -> str:
        r = self.get(metric, group, stratum)
        if pd.isna(r.estimate):
            return 'n/a'
        if pd.isna(r.ci_lo):
            return f'{r.estimate:.{d}f}'
        return f'{r.estimate:.{d}f} [{r.ci_lo:.{d}f}, {r.ci_hi:.{d}f}]'

    def count(self, metric, group, stratum='overall') -> int:
        return int(self.get(metric, group, stratum).estimate)


def pyplot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.hashsalt': str(SEED), 'font.size': 8})
    return plt


def savefig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, metadata={'Software': None})
    fig.clf()


VERDICT_RULE = ('Verdict rule (as in RQ2, D15): "supported" = the 95% CI of the stated quantity lies entirely '
                'on the hypothesised side; "not supported" = entirely on the other side; "inconclusive" = the CI '
                'includes the boundary. All tests are exploratory (no multiplicity correction).')
