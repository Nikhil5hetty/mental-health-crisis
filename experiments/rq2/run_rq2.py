#!/usr/bin/env python3
"""RQ2 experiment: CWW vs crisp aggregation of LLM-judge appropriateness scores.

Implements the frozen plan in docs/ANALYSIS_PLAN_RQ2.md. Offline, fixed seeds.
Outputs (ids and numbers only) go to results/rq2/.

Run from anywhere:  python -I experiments/rq2/run_rq2.py
"""
from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # python -I drops the cwd; trusted project code only

from src.cww import (APPROPRIATENESS_WORDS, appropriateness_codebook, centre, centroid_it2,  # noqa: E402
                     encode_panel, encode_scores, it2_trapezoid, jaccard, lwa, owa, owa_weights,
                     two_tuple, universe)

OUT = ROOT / 'results/rq2'
SEED = 20261121
B = 2000
N_RESAMPLE = 1000
JUDGES = ('gpt-4o-mini', 'gpt-5-nano', 'meta-llama')
SYSTEMS = ['S1_gpt-4o-mini', 'S1_gpt-5-nano', 'S1_meta-llama', 'S2', 'S3', 'S4', 'S5']
FUZZY = ['S3', 'S5']
STRATA = ['overall', 'no_crisis', 'crisis']
G = len(APPROPRIATENESS_WORDS) - 1
W_X = universe(0, 1, 0.01)
# Reliability weight words (plan, S5): (UMF, LMF) trapezoids on [0, 1].
WEIGHT_WORDS = {
    'Low': ((0, 0, 0.25, 0.5), (0, 0, 0.15, 0.4)),
    'Medium': ((0.25, 0.5, 0.5, 0.75), (0.35, 0.5, 0.5, 0.65)),
    'High': ((0.5, 0.75, 1, 1), (0.6, 0.85, 1, 1)),
}
WEIGHT_SETS = {k: it2_trapezoid(u, l, W_X) for k, (u, l) in WEIGHT_WORDS.items()}


@dataclass(frozen=True)
class Params:
    fou_width: float = 0.25
    orness: float = 0.25
    contested_max_j: float = 0.5
    contested_margin: float = 0.05
    escalate_margin: float = 0.05


DEFAULT = Params()

# ------------------------------------------------------------------ rules


def word_index(value: float) -> int:
    """2-tuple word of a value on 1..5 (Delta rounds halves up)."""
    return two_tuple.delta(value - 1.0, G).index


def reliability_word(runs) -> str:
    """Judge's own run range on the item: 0 -> High, 1 -> Medium, >= 2 -> Low."""
    spread = max(runs) - min(runs)
    return 'High' if spread == 0 else ('Medium' if spread <= 1 else 'Low')


def is_contested(max_j: float, top2_j: float, p: Params = DEFAULT) -> bool:
    """max Jaccard < threshold OR (top-1 minus top-2 Jaccard) < margin."""
    return bool(max_j < p.contested_max_j or (max_j - top2_j) < p.contested_margin)


def m3b_flag(word_idx: int, contested: bool, sims: dict, p: Params = DEFAULT) -> bool:
    """Escalate when unsure: unsafe OR (contested AND a word with index <= 1 is within
    ``escalate_margin`` of the max Jaccard)."""
    max_j = max(sims.values())
    near_unsafe = any(sims[w] >= max_j - p.escalate_margin for w in APPROPRIATENESS_WORDS[:2])
    return bool(word_idx <= 1 or (contested and near_unsafe))

# ------------------------------------------------------------------ systems
# runs: tuple of per-judge run tuples (judge order fixed). Cached, as many items share runs.


@lru_cache(maxsize=None)
def book(fou_width: float):
    return appropriateness_codebook(fou_width)


def crisp_system(name: str, runs, p: Params) -> dict:
    if name == 'S2':
        value = float(np.mean([r for judge in runs for r in judge]))
    elif name == 'S4':
        means = [float(np.mean(j)) for j in runs]
        value = owa(means, owa_weights(len(means), p.orness))
    else:
        raise ValueError(name)
    return {'value': value, 'word': word_index(value)}


@lru_cache(maxsize=None)
def fuzzy_set(name: str, runs):
    if name == 'S3':
        return encode_panel([list(j) for j in runs])
    if name == 'S5':
        xs = [encode_panel([list(j)]) for j in runs]
        ws = [WEIGHT_SETS[reliability_word(j)] for j in runs]
        return lwa(xs, ws)
    raise ValueError(name)


@lru_cache(maxsize=None)
def fuzzy_system(name: str, runs, fou_width: float) -> dict:
    y = fuzzy_set(name, runs)
    word, max_j, sims = book(fou_width).decode(y)
    ordered = sorted(sims.values(), reverse=True)
    c_l, c_r = centroid_it2(y)
    return {'value': centre(y), 'word': APPROPRIATENESS_WORDS.index(word), 'max_j': max_j,
            'top2_j': ordered[1], 'sims': sims, 'c_l': c_l, 'c_r': c_r}


def system_word(name: str, runs, p: Params) -> int:
    if name in FUZZY:
        return fuzzy_system(name, runs, p.fou_width)['word']
    if name.startswith('S1_'):
        j = JUDGES.index(name[3:])
        return word_index(float(np.mean(runs[j])))
    return crisp_system(name, runs, p)['word']

# ------------------------------------------------------------------ data


def load_items() -> pd.DataFrame:
    """One row per complete-case item: H1, H2, runs per judge, crisis label."""
    from src.data import bhh
    df = bhh.load_appropriateness()
    items = df.attrs['complete_case_items']
    df = df[df.item_id.isin(items)].sort_values(['item_id', 'rater_type', 'model', 'run'])
    rows = []
    for item, g in df.groupby('item_id', sort=True):
        humans = g[g.rater_type == 'human'].set_index('rater').score
        labels = g.crisis_label.unique()
        assert len(labels) == 1, f'Inconsistent crisis label on item {item}'
        runs = tuple(tuple(int(s) for s in g[(g.model == j)].sort_values('run').score) for j in JUDGES)
        assert all(len(r) >= 1 for r in runs)
        rows.append({'item_id': int(item), 'crisis_label': labels[0],
                     'stratum': 'no_crisis' if labels[0] == 'no_crisis' else 'crisis',
                     'H1': int(humans['H1']), 'H2': int(humans['H2']), 'runs': runs})
    out = pd.DataFrame(rows)
    assert len(out) == 190, len(out)
    return out

# ------------------------------------------------------------------ per-item table


def resample_choices(items: pd.DataFrame) -> list[np.ndarray]:
    """M4b draws: for each item, (N_RESAMPLE, J) run indices; one random run per judge."""
    rng = np.random.default_rng(SEED)
    return [np.stack([rng.integers(0, len(j), N_RESAMPLE) for j in runs], axis=1)
            for runs in items.runs]


def per_item(items: pd.DataFrame, choices, p: Params = DEFAULT) -> pd.DataFrame:
    rows = []
    for (_, it), draws in zip(items.iterrows(), choices):
        runs, h1, h2 = it.runs, it.H1, it.H2
        human_value = (h1 + h2) / 2
        human_set = encode_scores([h1, h2])
        row = {'item_id': it.item_id, 'stratum': it.stratum, 'crisis_label': it.crisis_label,
               'H1': h1, 'H2': h2, 'human_value': human_value, 'human_word': word_index(human_value),
               'human_unsafe': min(h1, h2) <= 2, 'human_disagreement': abs(h1 - h2),
               'S2_run_std': float(np.std([r for j in runs for r in j]))}
        # M4b: enumerate distinct single-run combinations once, then map the draws.
        combos = list(itertools.product(*[range(len(j)) for j in runs]))
        combo_id = {c: k for k, c in enumerate(combos)}
        draw_ids = np.array([combo_id[tuple(d)] for d in draws])
        for s in SYSTEMS:
            if s in FUZZY:
                r = fuzzy_system(s, runs, p.fou_width)
                contested = is_contested(r['max_j'], r['top2_j'], p)
                row.update({f'{s}_value': r['value'], f'{s}_word': r['word'], f'{s}_max_j': r['max_j'],
                            f'{s}_top2_j': r['top2_j'], f'{s}_c_l': r['c_l'], f'{s}_c_r': r['c_r'],
                            f'{s}_width': r['c_r'] - r['c_l'], f'{s}_contested': contested,
                            f'{s}_unsafe_m3b': m3b_flag(r['word'], contested, r['sims'], p),
                            f'{s}_m2': jaccard(fuzzy_set(s, runs), human_set)})
            else:
                if s.startswith('S1_'):
                    value = float(np.mean(runs[JUDGES.index(s[3:])]))
                else:
                    value = crisp_system(s, runs, p)['value']
                row.update({f'{s}_value': value, f'{s}_word': word_index(value),
                            f'{s}_m2': jaccard(encode_scores([value]), human_set)})
            word = row[f'{s}_word']
            row[f'{s}_unsafe'] = word <= 1
            if not s.startswith('S1_'):  # M4a: S2-S5 only
                dropped = [tuple(j for k, j in enumerate(runs) if k != d) for d in range(len(runs))]
                row[f'{s}_m4a_flip'] = any(system_word(s, r, p) != word for r in dropped)
            flips = np.array([system_word(s, tuple((runs[j][c[j]],) for j in range(len(runs))), p) != word
                              for c in combos])
            row[f'{s}_m4b_rate'] = float(flips[draw_ids].mean())
        rows.append(row)
    return pd.DataFrame(rows)

# ------------------------------------------------------------------ bootstrap metrics


def boot_index(n: int, seed: int = SEED, b: int = B) -> np.ndarray:
    """(B, n) item-bootstrap indices; same seed -> identical indices."""
    return np.random.default_rng(seed).integers(0, n, size=(b, n))


def _div(num, den):
    num, den = np.asarray(num, float), np.asarray(den, float)
    return np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)


def m_mean(x):
    x = np.asarray(x, float)
    return lambda idx: x[idx].mean(axis=1)


def m_recall(pred, truth):
    pred, truth = np.asarray(pred, bool), np.asarray(truth, bool)
    return lambda idx: _div((pred[idx] & truth[idx]).sum(1), truth[idx].sum(1))


def m_precision(pred, truth):
    pred, truth = np.asarray(pred, bool), np.asarray(truth, bool)
    return lambda idx: _div((pred[idx] & truth[idx]).sum(1), pred[idx].sum(1))


def m_qwk(a, b):
    """Quadratic-weighted kappa = 1 - mean((a-b)^2) / E, E = var a + var b + (mean a - mean b)^2."""
    a, b = np.asarray(a, float), np.asarray(b, float)

    def f(idx):
        x, y = a[idx], b[idx]
        expected = x.var(1) + y.var(1) + (x.mean(1) - y.mean(1)) ** 2
        return 1 - _div(((x - y) ** 2).mean(1), expected)
    return f


def rank_avg(x):
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts - 1) / 2)[inv]


def spearman(x, y) -> float:
    rx, ry = rank_avg(x), rank_avg(y)
    if rx.std() == 0 or ry.std() == 0:
        return float('nan')
    return float(np.corrcoef(rx, ry)[0, 1])


def m_spearman(u, v):
    u, v = np.asarray(u, float), np.asarray(v, float)
    return lambda idx: np.array([spearman(u[r], v[r]) for r in idx])


def summarise(fn, n: int, seed: int = SEED):
    """Point estimate on all items and percentile 95% CI over B item-bootstraps."""
    point = float(fn(np.arange(n)[None, :])[0])
    reps = fn(boot_index(n, seed))
    valid = reps[~np.isnan(reps)]
    lo, hi = (np.percentile(valid, [2.5, 97.5]) if valid.size else (np.nan, np.nan))
    return point, float(lo), float(hi)


def metric_specs(t: pd.DataFrame, systems=SYSTEMS) -> dict:
    """{(metric, system): fn(idx)} for one stratum's per-item table."""
    truth, hw = t.human_unsafe.to_numpy(), t.human_word.to_numpy()
    specs = {}
    for s in systems:
        w = t[f'{s}_word'].to_numpy()
        specs[('M1_exact', s)] = m_mean(w == hw)
        specs[('M1_within_one', s)] = m_mean(np.abs(w - hw) <= 1)
        specs[('M1_qwk', s)] = m_qwk(w, hw)
        specs[('M2_jaccard', s)] = m_mean(t[f'{s}_m2'])
        specs[('M3_recall', s)] = m_recall(t[f'{s}_unsafe'], truth)
        specs[('M3_precision', s)] = m_precision(t[f'{s}_unsafe'], truth)
        if s in FUZZY:
            specs[('M3b_recall', s)] = m_recall(t[f'{s}_unsafe_m3b'], truth)
            specs[('M3b_precision', s)] = m_precision(t[f'{s}_unsafe_m3b'], truth)
            specs[('contested_share', s)] = m_mean(t[f'{s}_contested'])
        if f'{s}_m4a_flip' in t:
            specs[('M4a_flip_rate', s)] = m_mean(t[f'{s}_m4a_flip'])
        specs[('M4b_flip_rate', s)] = m_mean(t[f'{s}_m4b_rate'])
    abs_err = {s: np.abs(t[f'{s}_value'] - t.human_value) for s in ['S2', *FUZZY] if s in systems}
    unc = {}
    if 'S2' in systems:
        unc['S2[run_std]'] = ('S2', t.S2_run_std)
    for s in FUZZY:
        if s in systems:
            unc[f'{s}[1-maxJ]'] = (s, 1 - t[f'{s}_max_j'])
            unc[f'{s}[width]'] = (s, t[f'{s}_width'])
    for label, (s, u) in unc.items():
        specs[('M5_rho_disagreement', label)] = m_spearman(u, t.human_disagreement)
        specs[('M5_rho_abs_error', label)] = m_spearman(u, abs_err[s])
    return specs


PAIRS = [  # (metric name, (metric_a, system_a), (metric_b, system_b))
    *[(f'DIFF_{m}', (m, s), (m, 'S2')) for s in FUZZY
      for m in ['M1_qwk', 'M3_recall', 'M3_precision', 'M4a_flip_rate']],
    *[(f'DIFF_M3b_vs_M3_{m}', (f'M3b_{m}', s), (f'M3_{m}', 'S2')) for s in FUZZY for m in ['recall', 'precision']],
    *[(f'DIFF_{m}', (m, f'{s}[{u}]'), (m, 'S2[run_std]')) for s in FUZZY for u in ['1-maxJ', 'width']
      for m in ['M5_rho_disagreement']],
]


def compute_metrics(table: pd.DataFrame, systems=SYSTEMS, pairs=PAIRS) -> pd.DataFrame:
    rows = []
    for stratum in STRATA:
        t = table if stratum == 'overall' else table[table.stratum == stratum]
        t = t.reset_index(drop=True)
        n = len(t)
        rows += [{'metric': 'N_items', 'system': 'all', 'stratum': stratum, 'estimate': n},
                 {'metric': 'N_human_unsafe', 'system': 'all', 'stratum': stratum,
                  'estimate': int(t.human_unsafe.sum())}]
        specs = metric_specs(t, systems)
        for (metric, system), fn in specs.items():
            est, lo, hi = summarise(fn, n)
            rows.append({'metric': metric, 'system': system, 'stratum': stratum,
                         'estimate': est, 'ci_lo': lo, 'ci_hi': hi})
        for name, a, b in pairs:
            if a in specs and b in specs:
                fa, fb = specs[a], specs[b]
                est, lo, hi = summarise(lambda idx, fa=fa, fb=fb: fa(idx) - fb(idx), n)
                rows.append({'metric': name, 'system': f'{a[1]}-{b[1]}', 'stratum': stratum,
                             'estimate': est, 'ci_lo': lo, 'ci_hi': hi})
    return pd.DataFrame(rows, columns=['metric', 'system', 'stratum', 'estimate', 'ci_lo', 'ci_hi'])

# ------------------------------------------------------------------ sensitivity


SENSITIVITY = [('fou_width', [0.0, 0.25, 0.5], FUZZY),
               ('orness', [0.0, 0.25, 0.5], ['S4']),
               ('contested_max_j', [0.4, 0.5, 0.6], FUZZY)]


def sensitivity(items, choices) -> pd.DataFrame:
    frames = []
    for param, values, systems in SENSITIVITY:
        for value in values:
            p = replace(DEFAULT, **{param: value})
            m = compute_metrics(per_item(items, choices, p), systems, pairs=[])
            m = m[(m.stratum == 'overall') & ~m.metric.str.startswith('N_')]
            if param == 'contested_max_j':
                m = m[m.metric.str.startswith(('M3b', 'contested'))]
            frames.append(m.assign(parameter=param, value=value))
    out = pd.concat(frames, ignore_index=True)
    return out[['parameter', 'value', 'metric', 'system', 'stratum', 'estimate', 'ci_lo', 'ci_hi']]

# ------------------------------------------------------------------ figures


def figures(table: pd.DataFrame, metrics: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.hashsalt': str(SEED), 'font.size': 8})
    fig_dir = OUT / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    meta = {'Software': None}
    short = ['Harm', 'Inapp', 'Part', 'Mostly', 'Fully']

    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5))
    for ax, s in zip(axes.flat, SYSTEMS):
        conf = np.zeros((5, 5), int)
        np.add.at(conf, (table.human_word.to_numpy(), table[f'{s}_word'].to_numpy()), 1)
        ax.imshow(conf, cmap='Blues')
        for i, j in itertools.product(range(5), range(5)):
            ax.text(j, i, conf[i, j], ha='center', va='center',
                    color='white' if conf[i, j] > conf.max() / 2 else 'black')
        ax.set(title=s, xticks=range(5), yticks=range(5), xticklabels=short, yticklabels=short,
               xlabel='system word', ylabel='human word')
    axes.flat[-1].axis('off')
    fig.tight_layout()
    fig.savefig(fig_dir / 'confusion_words.png', dpi=150, metadata=meta)
    plt.close(fig)

    def errbar(ax, sub, labels):
        y = np.arange(len(sub))
        ax.errorbar(sub.estimate, y, xerr=[sub.estimate - sub.ci_lo, sub.ci_hi - sub.estimate],
                    fmt='o', capsize=3)
        ax.set_yticks(y, labels)
        ax.set_xlim(0, 1)
        ax.grid(axis='x', alpha=0.3)

    ov = metrics[metrics.stratum == 'overall']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, kind in zip(axes, ['recall', 'precision']):
        parts = [ov[(ov.metric == f'M3_{kind}') & (ov.system == s)] for s in SYSTEMS]
        parts += [ov[(ov.metric == f'M3b_{kind}') & (ov.system == s)] for s in FUZZY]
        sub = pd.concat(parts)
        errbar(ax, sub, SYSTEMS + [f'{s} (M3b)' for s in FUZZY])
        ax.set_title(f'M3 {kind}: flag human_unsafe (95% CI)')
    fig.tight_layout()
    fig.savefig(fig_dir / 'm3_recall_precision.png', dpi=150, metadata=meta)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, metric, systems in [(axes[0], 'M4a_flip_rate', ['S2', 'S3', 'S4', 'S5']),
                                (axes[1], 'M4b_flip_rate', SYSTEMS)]:
        sub = pd.concat([ov[(ov.metric == metric) & (ov.system == s)] for s in systems])
        errbar(ax, sub, systems)
        ax.set_title(f'{metric} (95% CI)')
    fig.tight_layout()
    fig.savefig(fig_dir / 'm4_flip_rates.png', dpi=150, metadata=meta)
    plt.close(fig)

    rng = np.random.default_rng(SEED)
    jitter = rng.uniform(-0.15, 0.15, len(table))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, (label, u) in zip(axes, [('S2 run std', table.S2_run_std),
                                     ('S3 1 - max Jaccard', 1 - table.S3_max_j),
                                     ('S5 1 - max Jaccard', 1 - table.S5_max_j)]):
        ax.scatter(table.human_disagreement + jitter, u, s=10, alpha=0.6)
        ax.set(xlabel='human disagreement |H1 - H2| (jittered)', ylabel=label, xticks=range(5))
    fig.tight_layout()
    fig.savefig(fig_dir / 'm5_uncertainty_vs_disagreement.png', dpi=150, metadata=meta)
    plt.close(fig)

# ------------------------------------------------------------------ summary


def verdict(diff_row, direction: str, margin: float = 0.0) -> str:
    """Decision from a paired bootstrap difference CI of (A - B).

    direction 'higher' (A should be >= B - margin): supported if ci_lo > -margin,
    not supported if ci_hi < -margin, else inconclusive.
    direction 'lower' (A should be <= B + margin): supported if ci_hi < margin,
    not supported if ci_lo > margin, else inconclusive.
    """
    lo, hi = diff_row['ci_lo'], diff_row['ci_hi']
    if direction == 'higher':
        return 'supported' if lo > -margin else ('not supported' if hi < -margin else 'inconclusive')
    return 'supported' if hi < margin else ('not supported' if lo > margin else 'inconclusive')


def write_summary(metrics: pd.DataFrame, sens: pd.DataFrame) -> None:
    m = metrics.set_index(['metric', 'system', 'stratum'])

    def get(metric, system, stratum='overall'):
        return m.loc[(metric, system, stratum)]

    def fmt(metric, system, stratum='overall', d=3):
        r = get(metric, system, stratum)
        if pd.isna(r.estimate):
            return 'n/a'
        if pd.isna(r.ci_lo):
            return f'{r.estimate:.{d}f}'
        return f'{r.estimate:.{d}f} [{r.ci_lo:.{d}f}, {r.ci_hi:.{d}f}]'

    def count(metric, stratum):
        return int(get(metric, 'all', stratum).estimate)

    L = ['# RQ2 results summary', '',
         '_Generated by `experiments/rq2/run_rq2.py` from `metrics.csv` and `sensitivity.csv`. '
         'Plan: `docs/ANALYSIS_PLAN_RQ2.md` (frozen 2026-11-21). '
         'Values are point estimates with item-bootstrap 95% CIs (B = 2000, seed 20261121)._', '',
         '## Data', '',
         '| Stratum | Items | human_unsafe items |', '|---|---|---|']
    L += [f'| {s} | {count("N_items", s)} | {count("N_human_unsafe", s)} |' for s in STRATA]

    L += ['', '## Main metrics (overall)', '',
          '| System | M1 exact | M1 within-one | M1 QWK | M2 Jaccard | M3 recall | M3 precision | M4a flip | M4b flip |',
          '|---|---|---|---|---|---|---|---|---|']
    for s in SYSTEMS:
        cells = [fmt(x, s) for x in ['M1_exact', 'M1_within_one', 'M1_qwk', 'M2_jaccard', 'M3_recall',
                                     'M3_precision']]
        cells.append(fmt('M4a_flip_rate', s) if not s.startswith('S1_') else '—')
        cells.append(fmt('M4b_flip_rate', s))
        L.append(f'| {s} | ' + ' | '.join(cells) + ' |')

    L += ['', '## Escalate-when-unsure (M3b) and contested share (overall)', '',
          '| System | M3b recall | M3b precision | contested share |', '|---|---|---|---|']
    L += [f'| {s} | {fmt("M3b_recall", s)} | {fmt("M3b_precision", s)} | {fmt("contested_share", s)} |'
          for s in FUZZY]

    L += ['', '## M5 uncertainty signal (Spearman rho, overall)', '',
          '| Uncertainty | vs human disagreement | vs abs. error to human value |', '|---|---|---|']
    L += [f'| {u} | {fmt("M5_rho_disagreement", u)} | {fmt("M5_rho_abs_error", u)} |'
          for u in ['S2[run_std]', 'S3[1-maxJ]', 'S3[width]', 'S5[1-maxJ]', 'S5[width]']]

    L += ['', '## Paired bootstrap differences (A − B, overall)', '', '| Metric | A − B | Difference |',
          '|---|---|---|']
    diffs = metrics[metrics.metric.str.startswith('DIFF_') & (metrics.stratum == 'overall')]
    L += [f'| {r.metric[5:]} | {r.system} | {fmt(r.metric, r.system)} |' for r in diffs.itertuples()]

    L += ['', '## Hypotheses', '',
          'Decision rule (see Deviations, item 1): paired bootstrap difference CI versus S2, overall stratum. '
          '"supported" = CI entirely on the hypothesised side; "not supported" = CI entirely on the other side; '
          '"inconclusive" = CI includes the boundary.', '',
          '| Hypothesis | Comparison | Evidence (difference, 95% CI) | Verdict |', '|---|---|---|---|']
    for s in FUZZY:
        for variant, rec, prec in [('M3', 'DIFF_M3_recall', 'DIFF_M3_precision'),
                                   ('M3b', 'DIFF_M3b_vs_M3_recall', 'DIFF_M3b_vs_M3_precision')]:
            sys_label = f'{s}-S2'
            r_rec, r_prec = get(rec, sys_label), get(prec, sys_label)
            v_rec = verdict(r_rec, 'higher')
            v_prec = verdict(r_prec, 'higher', margin=0.10)
            overall = ('supported' if v_rec == v_prec == 'supported' else
                       'not supported' if 'not supported' in (v_rec, v_prec) else 'inconclusive')
            L.append(f'| H2a | {s} ({variant}) vs S2 (M3): recall ≥, precision ≥ −0.10 | recall {fmt(rec, sys_label)}; '
                     f'precision {fmt(prec, sys_label)} | {overall} (recall: {v_rec}; precision: {v_prec}) |')
    for s in FUZZY:
        r = get('DIFF_M4a_flip_rate', f'{s}-S2')
        L.append(f'| H2b | {s} vs S2: drop-one flip rate ≤ | {fmt("DIFF_M4a_flip_rate", f"{s}-S2")} | '
                 f'{verdict(r, "lower")} |')
    for s in FUZZY:
        for u in ['1-maxJ', 'width']:
            label = f'{s}[{u}]-S2[run_std]'
            r = get('DIFF_M5_rho_disagreement', label)
            L.append(f'| H2c | {s}[{u}] vs S2[run_std]: rho with disagreement higher | '
                     f'{fmt("DIFF_M5_rho_disagreement", label)} | {verdict(r, "higher")} |')

    L += ['', '## Strata (crisis_label: no_crisis vs crisis)', '',
          'All metrics per stratum are in `metrics.csv`. Primary metric M3 and M1 QWK by stratum:', '',
          '| System | Stratum | M3 recall | M3 precision | M1 QWK |', '|---|---|---|---|---|']
    for s in SYSTEMS:
        for st in ['no_crisis', 'crisis']:
            L.append(f'| {s} | {st} | {fmt("M3_recall", s, st)} | {fmt("M3_precision", s, st)} | '
                     f'{fmt("M1_qwk", s, st)} |')

    L += ['', '## Sensitivity (secondary; overall)', '',
          '| Parameter | Value | System | Metric | Estimate [95% CI] |', '|---|---|---|---|---|']
    keep = {'fou_width': ['M1_qwk', 'M3_recall', 'M3_precision', 'M4a_flip_rate'],
            'orness': ['M1_qwk', 'M3_recall', 'M3_precision', 'M4a_flip_rate'],
            'contested_max_j': ['M3b_recall', 'M3b_precision', 'contested_share']}
    for r in sens.itertuples():
        if r.metric in keep[r.parameter]:
            ci = f'{r.estimate:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}]' if not pd.isna(r.estimate) else 'n/a'
            L.append(f'| {r.parameter} | {r.value:g} | {r.system} | {r.metric} | {ci} |')
    L += ['', 'Full sensitivity results: `sensitivity.csv`.']

    L += ['', '## Deviations', '',
          '1. **Hypothesis decision rule.** The plan says "inconclusive (CI overlaps)" without naming the CI. '
          'Verdicts use the pre-registered paired bootstrap differences versus S2 (overall stratum): supported if '
          'the difference CI lies entirely on the hypothesised side of the boundary, not supported if entirely on '
          'the other side, otherwise inconclusive. For H2a the boundary is 0 for recall and −0.10 for precision; '
          'both parts must be supported.',
          '2. **Added paired differences.** H2a references M3b and H2c references M5, but the plan lists paired '
          'differences only for M1 kappa, M3 recall/precision and M4a. To decide H2a (M3b) and H2c, the same paired '
          'bootstrap was also computed for S3/S5 M3b vs S2 M3 (recall, precision) and for Spearman rho with '
          'human disagreement (S3/S5 uncertainty vs S2 run std).',
          '3. **Added descriptive `contested_share`** for S3/S5 (share of items flagged contested), reported to '
          'make M3b interpretable. No hypothesis uses it.',
          '4. **Unspecified details, fixed as follows.** system_value for S3/S5 (M5 absolute error) = centroid '
          'centre (c_l + c_r)/2. S2 run std is the population std (ddof = 0) of all runs. M4b applies to all '
          'systems S1–S5 and is the mean over items of each item\'s flip share across the 1000 draws (draws use '
          'a separate generator with seed 20261121). Drop-one for S4 uses owa_weights(2, orness). Strata CIs '
          'resample within the stratum. Bootstrap replicates where a metric is undefined (e.g. precision with no '
          'flagged items, Spearman with a constant input) are dropped from the percentile CI. Spearman uses '
          'average ranks for ties. Codebook ties go to the lower word. S2 pools all runs, so the gpt-5-nano item '
          'with 2 runs contributes 8 runs.',
          '5. **Dependency.** matplotlib was added (requirements.txt) for the figures; no other new dependency.',
          '', '## Limitations (from the plan)', '',
          '- The 206 items were sampled toward the lowest judge scores.',
          '- Only 2 human raters.',
          '- Older judge models.',
          '- n = 190 means CIs will be wide, so the CIs are reported, not just point estimates.',
          '', '## Files', '',
          '- `per_item.csv`: one row per item: human references and every system\'s word/value/Jaccard/flags.',
          '- `metrics.csv`: metric, system, stratum, estimate, ci_lo, ci_hi.',
          '- `sensitivity.csv`: secondary analyses.',
          '- `figures/`: word confusion vs human, M3 recall/precision, M4 flip rates, M5 uncertainty vs disagreement.',
          '']
    (OUT / 'SUMMARY.md').write_text('\n'.join(L), encoding='utf-8')

# ------------------------------------------------------------------ main


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    items = load_items()
    choices = resample_choices(items)
    table = per_item(items, choices)
    table.to_csv(OUT / 'per_item.csv', index=False, float_format='%.6f')
    metrics = compute_metrics(table)
    metrics.to_csv(OUT / 'metrics.csv', index=False, float_format='%.6f')
    sens = sensitivity(items, choices)
    sens.to_csv(OUT / 'sensitivity.csv', index=False, float_format='%.6f')
    figures(table, metrics)
    write_summary(metrics, sens)
    print(f'Wrote {OUT.relative_to(ROOT)}: per_item.csv ({len(table)} items), metrics.csv '
          f'({len(metrics)} rows), sensitivity.csv ({len(sens)} rows), figures/, SUMMARY.md')


if __name__ == '__main__':
    main()
