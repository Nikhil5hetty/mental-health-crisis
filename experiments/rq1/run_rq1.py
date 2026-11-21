#!/usr/bin/env python3
"""Part A (RQ1): is "crisis" better kept as a degree than a majority label?

Implements Part A of docs/ANALYSIS_PLAN_RQ1_DEMO.md (frozen 2026-11-21).
Offline, seeded; writes ids and numbers only to results/rq1/.

Run from anywhere:  python -I experiments/rq1/run_rq1.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.common import (VERDICT_RULE, Table, count_row, div, est_row, m_cond_mean,  # noqa: E402
                                m_mean, m_spearman, pyplot, savefig, to_frame, verdict, write_csv)

OUT = ROOT / 'results/rq1'
MU_H_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]


def majority(mu: float) -> str:
    """> 0.5 crisis, < 0.5 no_crisis, = 0.5 tie."""
    return 'crisis' if mu > 0.5 else ('no_crisis' if mu < 0.5 else 'tie')


def modal_category(counts: dict) -> str:
    """Category with the most votes; 'tie' if the maximum is shared."""
    top = max(counts.values())
    winners = [c for c, v in counts.items() if v == top]
    return winners[0] if len(winners) == 1 else 'tie'


def fuzzy_category_agreement(p_h: dict, p_l: dict) -> float:
    """sum_c min(p_H(c), p_L(c)) / sum_c max(p_H(c), p_L(c))."""
    num = sum(min(p_h[c], p_l[c]) for c in p_h)
    den = sum(max(p_h[c], p_l[c]) for c in p_h)
    return num / den


def load_items() -> pd.DataFrame:
    from src.data import bhh
    d = bhh.load_crisis_degree()
    assert len(d) == 206
    rows = []
    for r in d.to_dict('records'):  # column names contain '-', so no itertuples
        h ={c: r[f'human_count_{c}'] for c in bhh.CATEGORIES}
        l = {c: r[f'llm_count_{c}'] for c in bhh.CATEGORIES}
        n_h, n_l = sum(h.values()), r['n_llm_observed']
        assert n_h == 4 and sum(l.values()) == n_l > 0
        mu_h, mu_l = r['k_humans_crisis'] / 4, r['k_llm_crisis'] / n_l
        assert majority(mu_h) == r['human_majority']  # loader rule == plan rule
        rows.append({
            'item_id': r['item_id'], 'n_llm_observed': n_l, 'n_llm_missing': r['n_llm_missing'],
            'k_humans_crisis': r['k_humans_crisis'], 'k_llm_crisis': r['k_llm_crisis'],
            'mu_H': mu_h, 'mu_L': mu_l, 'human_majority': r['human_majority'],
            'llm_majority': majority(mu_l), 'human_split': 0 < mu_h < 1,
            'human_modal_category': modal_category(h), 'llm_modal_category': modal_category(l),
            'fuzzy_category_agreement': fuzzy_category_agreement({c: v / n_h for c, v in h.items()},
                                                                 {c: v / n_l for c, v in l.items()}),
        })
    return pd.DataFrame(rows)


def compute_metrics(t: pd.DataFrame) -> pd.DataFrame:
    t = t.reset_index(drop=True)
    n = len(t)
    rows = [count_row('N_items', 'all', 'overall', n),
            count_row('N_llm_missing_labels', 'all', 'overall', int(t.n_llm_missing.sum()))]
    # D-A1
    for level in MU_H_LEVELS:
        rows.append(est_row('share_mu_H_level', f'{level:g}', 'overall', m_mean(t.mu_H == level), n))
    rows.append(est_row('share_human_split', 'all', 'overall', m_mean(t.human_split), n))
    for cat in ['crisis', 'no_crisis', 'tie']:
        rows.append(est_row('share_human_majority', cat, 'overall', m_mean(t.human_majority == cat), n))
        rows.append(est_row('share_llm_majority', cat, 'overall', m_mean(t.llm_majority == cat), n))
    rows.append(est_row('mean_mu_L', 'all', 'overall', m_mean(t.mu_L), n))
    rows.append(est_row('share_mu_L_zero', 'all', 'overall', m_mean(t.mu_L == 0), n))
    rows.append(est_row('share_mu_L_split', 'all', 'overall', m_mean((t.mu_L > 0) & (t.mu_L < 1)), n))
    rows.append(est_row('share_mu_L_one', 'all', 'overall', m_mean(t.mu_L == 1), n))
    # D-A2: calibration, resampling within each mu_H level
    for level in MU_H_LEVELS:
        sub = t[t.mu_H == level].reset_index(drop=True)
        rows.append(count_row('N_items', 'all', f'mu_H={level:g}', len(sub)))
        if len(sub):
            rows.append(est_row('mean_mu_L', 'all', f'mu_H={level:g}', m_mean(sub.mu_L), len(sub)))
    # D-A3: category-level agreement
    h_tie, l_tie = t.human_modal_category == 'tie', t.llm_modal_category == 'tie'
    both = ~h_tie & ~l_tie
    rows += [count_row('N_human_modal_tie', 'all', 'overall', int(h_tie.sum())),
             count_row('N_llm_modal_tie', 'all', 'overall', int(l_tie.sum())),
             count_row('N_either_modal_tie', 'all', 'overall', int((~both).sum()))]
    fa = t.fuzzy_category_agreement
    rows.append(est_row('mean_fuzzy_category_agreement', 'all', 'overall', m_mean(fa), n))
    rows.append(est_row('mean_fuzzy_category_agreement', 'all', 'both_modal_nontie', m_cond_mean(fa, both), n))
    rows.append(est_row('mean_fuzzy_category_agreement', 'all', 'either_modal_tie', m_cond_mean(fa, ~both), n))
    match = (t.human_modal_category == t.llm_modal_category) & both
    rows.append(est_row('crisp_category_match', 'all', 'both_modal_nontie', m_cond_mean(match, both), n))
    # H1a: Spearman among human_split items (resampling within the subset)
    split = t[t.human_split].reset_index(drop=True)
    rows.append(count_row('N_items', 'all', 'human_split', len(split)))
    rows.append(est_row('spearman_mu_H_mu_L', 'all', 'human_split', m_spearman(split.mu_H, split.mu_L), len(split)))
    rows.append(est_row('spearman_mu_H_mu_L', 'all', 'overall', m_spearman(t.mu_H, t.mu_L), n))
    # H1b: crisp disagreement on non-tie items, split vs unanimous (whole-sample bootstrap)
    nontie = (t.human_majority != 'tie') & (t.llm_majority != 'tie')
    disagree = (t.human_majority != t.llm_majority) & nontie
    s_mask, u_mask = t.human_split & nontie, ~t.human_split & nontie
    f_split, f_unan = m_cond_mean(disagree, s_mask), m_cond_mean(disagree, u_mask)
    rows.append(est_row('crisp_disagreement_rate', 'all', 'human_split', f_split, n))
    rows.append(est_row('crisp_disagreement_rate', 'all', 'unanimous', f_unan, n))
    rows.append(est_row('DIFF_crisp_disagreement_rate', 'human_split-unanimous', 'overall',
                        lambda idx: f_split(idx) - f_unan(idx), n))
    for name, mask in [('human_split', t.human_split), ('unanimous', ~t.human_split)]:
        rows += [count_row('N_items', 'all', name, int(mask.sum())) if name == 'unanimous' else None,
                 count_row('N_nontie_items', 'all', name, int((mask & nontie).sum())),
                 count_row('N_human_majority_tie', 'all', name, int((mask & (t.human_majority == 'tie')).sum())),
                 count_row('N_llm_majority_tie', 'all', name, int((mask & (t.llm_majority == 'tie')).sum())),
                 count_row('N_dropped_any_tie', 'all', name, int((mask & ~nontie).sum()))]
    return to_frame([r for r in rows if r is not None])


def figures(t: pd.DataFrame, metrics: pd.DataFrame) -> None:
    plt = pyplot()
    tab = Table(metrics)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    jitter = np.random.default_rng(20261121).uniform(-0.03, 0.03, len(t))
    ax.scatter(t.mu_H + jitter, t.mu_L, s=8, alpha=0.35, color='grey', label='items (jittered)')
    levels = [lv for lv in MU_H_LEVELS if (t.mu_H == lv).any()]
    r = [tab.get('mean_mu_L', 'all', f'mu_H={lv:g}') for lv in levels]
    ax.errorbar(levels, [x.estimate for x in r], yerr=[[x.estimate - x.ci_lo for x in r],
                                                        [x.ci_hi - x.estimate for x in r]],
                fmt='o-', capsize=3, label='mean mu_L (95% CI)')
    ax.plot([0, 1], [0, 1], ':', color='black', lw=0.8)
    ax.set(xlabel='mu_H = k_humans_crisis / 4', ylabel='mu_L = k_llm_crisis / n_observed',
           xlim=(-0.08, 1.08), ylim=(-0.05, 1.05), title='D-A2 calibration')
    ax.legend(loc='upper left')
    savefig(fig, OUT / 'figures/calibration_mu_H_mu_L.png')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    counts = t.mu_H.value_counts().reindex(MU_H_LEVELS, fill_value=0)
    axes[0].bar([f'{v:g}' for v in MU_H_LEVELS], counts.values)
    axes[0].set(title='D-A1 mu_H distribution', xlabel='mu_H', ylabel='items')
    axes[1].hist(t.mu_L, bins=np.linspace(0, 1, 19), edgecolor='white')
    axes[1].set(title='D-A1 mu_L distribution', xlabel='mu_L', ylabel='items')
    savefig(fig, OUT / 'figures/distributions_mu.png')
    plt.close(fig)


def write_summary(metrics: pd.DataFrame) -> None:
    tab = Table(metrics)
    f, c = tab.fmt, tab.count
    L = ['# RQ1 (Part A) results summary', '',
         '_Generated by `experiments/rq1/run_rq1.py` from `metrics.csv`. Plan: `docs/ANALYSIS_PLAN_RQ1_DEMO.md` '
         '(frozen 2026-11-21). Estimates with item-bootstrap 95% CIs (B = 2000, seed 20261121)._', '',
         f'Items: {c("N_items", "all")}; missing LLM labels kept missing: {c("N_llm_missing_labels", "all")}.', '',
         '## D-A1 Distributions', '',
         '| mu_H | Share of items |', '|---|---|']
    L += [f'| {lv:g} | {f("share_mu_H_level", f"{lv:g}")} |' for lv in MU_H_LEVELS]
    L += ['', f'- Share human_split (0 < mu_H < 1): {f("share_human_split", "all")}',
          f'- mu_L: mean {f("mean_mu_L", "all")}; share = 0 {f("share_mu_L_zero", "all")}; '
          f'share in (0, 1) {f("share_mu_L_split", "all")}; share = 1 {f("share_mu_L_one", "all")}', '',
          '| Majority category | Human share | LLM share |', '|---|---|---|']
    L += [f'| {cat} | {f("share_human_majority", cat)} | {f("share_llm_majority", cat)} |'
          for cat in ['crisis', 'no_crisis', 'tie']]
    L += ['', '## D-A2 Calibration (mean mu_L per mu_H level)', '', '| mu_H | Items | Mean mu_L |', '|---|---|---|']
    for lv in MU_H_LEVELS:
        st = f'mu_H={lv:g}'
        L.append(f'| {lv:g} | {c("N_items", "all", st)} | {f("mean_mu_L", "all", st)} |')
    L += ['', '## D-A3 Category-level agreement (7 categories)', '',
          f'- Modal-category ties: human {c("N_human_modal_tie", "all")}, LLM {c("N_llm_modal_tie", "all")}, '
          f'either {c("N_either_modal_tie", "all")} (excluded from the crisp match).',
          f'- Mean fuzzy category agreement, all items: {f("mean_fuzzy_category_agreement", "all")}',
          f'- On items where both modal categories are non-tie: fuzzy {f("mean_fuzzy_category_agreement", "all", "both_modal_nontie")}; '
          f'crisp exact match {f("crisp_category_match", "all", "both_modal_nontie")}',
          f'- On items with a modal tie (crisp match undefined): fuzzy '
          f'{f("mean_fuzzy_category_agreement", "all", "either_modal_tie")}', '',
          '## Hypotheses', '', VERDICT_RULE, '',
          '| Hypothesis | Quantity | Estimate [95% CI] | Verdict |', '|---|---|---|---|']
    h1a = tab.get('spearman_mu_H_mu_L', 'all', 'human_split')
    L.append(f'| H1a | Spearman(mu_H, mu_L), human_split items (n = {c("N_items", "all", "human_split")}); > 0 | '
             f'{f("spearman_mu_H_mu_L", "all", "human_split")} | {verdict(h1a, "higher")} |')
    h1b = tab.get('DIFF_crisp_disagreement_rate', 'human_split-unanimous')
    L.append(f'| H1b | Crisp disagreement rate, human_split − unanimous (non-tie items); > 0 | '
             f'{f("DIFF_crisp_disagreement_rate", "human_split-unanimous")} | {verdict(h1b, "higher")} |')
    L += ['', f'- Crisp disagreement rate: human_split {f("crisp_disagreement_rate", "all", "human_split")}; '
          f'unanimous {f("crisp_disagreement_rate", "all", "unanimous")}.',
          f'- Spearman(mu_H, mu_L), all items (descriptive): {f("spearman_mu_H_mu_L", "all")}', '',
          '| Group | Items | Non-tie items used | Human majority ties | LLM majority ties | Dropped (any tie) |',
          '|---|---|---|---|---|---|']
    for g in ['human_split', 'unanimous']:
        L.append(f'| {g} | {c("N_items", "all", g)} | {c("N_nontie_items", "all", g)} | '
                 f'{c("N_human_majority_tie", "all", g)} | {c("N_llm_majority_tie", "all", g)} | '
                 f'{c("N_dropped_any_tie", "all", g)} |')
    L += ['', '## Deviations', '',
          '1. D13 cited in the plan refers to the missing-label rule; DECISIONS.md numbering was synced after the plan commit.',
          '2. **"Majority category" for D-A3** is the modal (plurality) category over the 7 categories; when the '
          'largest count is shared the item is a tie, excluded from the crisp match and counted.',
          '3. **mu_L distribution** (D-A1) is summarised as its mean and the shares at 0, in (0, 1) and at 1; '
          'the full distribution is in `figures/distributions_mu.png` and `per_item.csv`.',
          '4. **CI resampling units.** D-A2 means and the H1a Spearman resample items within their subset '
          '(mu_H level; human_split items). All other estimates, including the H1b risk difference, resample '
          'all 206 items, and conditional rates are recomputed in each replicate.',
          '5. **Binary majority for H1b.** Disagreement compares the crisis / no_crisis majority labels '
          '(human_majority from the loader, which equals the plan rule on mu_H; LLM majority by the plan rule).',
          '', '## Files', '',
          '- `per_item.csv`: item_id, counts, mu_H, mu_L, majority labels, modal categories, fuzzy category agreement.',
          '- `metrics.csv`: metric, group, stratum, estimate, ci_lo, ci_hi.',
          '- `figures/`: calibration (D-A2) and distributions (D-A1).', '']
    (OUT / 'SUMMARY.md').write_text('\n'.join(L), encoding='utf-8')


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t = load_items()
    write_csv(t, OUT / 'per_item.csv')
    metrics = compute_metrics(t)
    write_csv(metrics, OUT / 'metrics.csv')
    figures(t, metrics)
    write_summary(metrics)
    print(f'Wrote {OUT.relative_to(ROOT)}: per_item.csv ({len(t)} items), metrics.csv ({len(metrics)} rows), '
          'figures/, SUMMARY.md')


if __name__ == '__main__':
    main()
