#!/usr/bin/env python3
"""Part B: CounselBench expert uncertainty (descriptive only; D6).

Implements Part B of docs/ANALYSIS_PLAN_RQ1_DEMO.md (frozen 2026-11-21).
Offline, seeded; writes ids and numbers only to results/counselbench/.

Run from anywhere:  python -I experiments/counselbench/run_cb.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.common import (Table, count_row, est_row, m_cond_mean, m_mean, pyplot,  # noqa: E402
                                rq2, savefig, to_frame, write_csv)
from src.cww import Codebook, appropriateness_codebook, encode_scores  # noqa: E402

OUT = ROOT / 'results/counselbench'
RESPONDERS = ['gpt4', 'llama3', 'gemini', 'human']
STRATA = ['overall', *RESPONDERS]
LIKERT = ['overall', 'empathy', 'specificity', 'toxicity']
RISKS = ['medical', 'factual', 'toxicity']
TOX_WORDS = ['Not toxic at all', 'Somewhat not toxic', 'Neutral', 'Somewhat toxic', 'Extremely toxic']


def toxicity_codebook() -> Codebook:
    """5 toxicity words with the same IT2 sets as appropriateness_codebook(0.25), centred on 1..5."""
    return Codebook(TOX_WORDS, appropriateness_codebook(0.25).sets, first_value=1.0)


def modal_membership(panel) -> float:
    """max over integer levels k of the IAA membership mu(k) (bin centres, unaffected by shared bin edges)."""
    return float(max(panel.umf[np.isclose(panel.x, k)][0] for k in range(1, 6)))


def load_replies() -> pd.DataFrame:
    from src.data import counselbench
    ev = counselbench.load_eval()
    deg = counselbench.load_reply_degree().set_index('reply_id')
    book = toxicity_codebook()
    rows = []
    for reply_id, g in ev.groupby('reply_id', sort=True):
        d = deg.loc[reply_id]
        assert len(g) == 5 and g.responder.nunique() == 1
        row = {'reply_id': reply_id, 'responder': g.responder.iloc[0],
               'k_medical_yes': int(d.k_medical_yes), 'k_medical_not_sure': int(d.k_medical_not_sure),
               'k_factual_not_sure': int(d.k_factual_not_sure),
               'factual_unanimous': g.factual.nunique() == 1}
        row['medical_split'] = 1 <= row['k_medical_yes'] <= 4
        for dim in LIKERT:
            scores = g[dim].astype(int).tolist()
            panel = encode_scores(scores, half_width=0.5)
            row[f'{dim}_range'] = max(scores) - min(scores)
            row[f'{dim}_mean'] = float(np.mean(scores))
            row[f'{dim}_modal_membership'] = modal_membership(panel)
            if dim == 'toxicity':
                word, max_j, sims = book.decode(panel)
                ordered = sorted(sims.values(), reverse=True)
                row.update({'toxicity_decoded_word': TOX_WORDS.index(word), 'toxicity_max_j': max_j,
                            'toxicity_top2_j': ordered[1],
                            'toxicity_contested': rq2.is_contested(max_j, ordered[1]),
                            'toxicity_crisp_word': TOX_WORDS.index(book.to_two_tuple(row['toxicity_mean'])[0])})
                row['toxicity_word_differs'] = row['toxicity_decoded_word'] != row['toxicity_crisp_word']
        for risk in RISKS:
            row[f'k_{risk}_copied'] = int(d[f'k_{risk}_copied'])
        rows.append(row)
    out = pd.DataFrame(rows)
    assert len(out) == 400 and out.responder.value_counts().to_dict() == dict.fromkeys(RESPONDERS, 100)
    return out


def compute_metrics(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stratum in STRATA:
        s = (t if stratum == 'overall' else t[t.responder == stratum]).reset_index(drop=True)
        n = len(s)
        rows.append(count_row('N_replies', 'all', stratum, n))
        # B1 risk dimensions
        for k in range(6):
            rows.append(est_row('share_k_medical_yes', str(k), stratum, m_mean(s.k_medical_yes == k), n))
        rows.append(est_row('share_split_panel', 'medical_advice', stratum, m_mean(s.medical_split), n))
        rows.append(est_row('share_unanimous_panel', 'medical_advice', stratum, m_mean(~s.medical_split), n))
        rows.append(est_row('share_unanimous_panel', 'factual', stratum, m_mean(s.factual_unanimous), n))
        for dim, col in [('medical_advice', 'k_medical_not_sure'), ('factual', 'k_factual_not_sure')]:
            rows.append(est_row('not_sure_rating_share', dim, stratum, m_mean(s[col] / 5), n))
            rows.append(est_row('not_sure_reply_share', dim, stratum, m_mean(s[col] >= 1), n))
        # B2 Likert spread and IAA panel
        for dim in LIKERT:
            rows.append(est_row('share_range_ge2', dim, stratum, m_mean(s[f'{dim}_range'] >= 2), n))
            rows.append(est_row('mean_range', dim, stratum, m_mean(s[f'{dim}_range']), n))
            rows.append(est_row('mean_modal_membership', dim, stratum, m_mean(s[f'{dim}_modal_membership']), n))
        # B3 toxicity decoding
        rows.append(est_row('share_contested', 'toxicity', stratum, m_mean(s.toxicity_contested), n))
        rows.append(est_row('share_crisp_word_differs', 'toxicity', stratum, m_mean(s.toxicity_word_differs), n))
        for i, w in enumerate(TOX_WORDS):
            rows.append(est_row('share_decoded_word', w, stratum, m_mean(s.toxicity_decoded_word == i), n))
        # B4 single-copier evidence
        for risk in RISKS:
            k = s[f'k_{risk}_copied']
            rows.append(count_row('N_k_copied_1', risk, stratum, int((k == 1).sum())))
            rows.append(count_row('N_k_copied_ge1', risk, stratum, int((k >= 1).sum())))
            rows.append(est_row('share_k_copied_1', risk, stratum, m_mean(k == 1), n))
            rows.append(est_row('share_k_copied_ge2', risk, stratum, m_mean(k >= 2), n))
            rows.append(est_row('share_k1_among_flagged', risk, stratum, m_cond_mean(k == 1, k >= 1), n))
    return to_frame(rows)


def figures(t: pd.DataFrame, metrics: pd.DataFrame) -> None:
    plt = pyplot()
    tab = Table(metrics)
    width = 0.2
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    x = np.arange(6)
    for i, r in enumerate(RESPONDERS):
        axes[0].bar(x + (i - 1.5) * width, [tab.get('share_k_medical_yes', str(k), r).estimate for k in range(6)],
                    width, label=r)
    axes[0].set(title='B1 medical advice: k of 5 experts "yes"', xlabel='k_yes', ylabel='share of replies',
                xticks=x)
    axes[0].legend()
    x = np.arange(len(LIKERT))
    for i, r in enumerate(RESPONDERS):
        axes[1].bar(x + (i - 1.5) * width, [tab.get('share_range_ge2', d, r).estimate for d in LIKERT], width, label=r)
    axes[1].set(title='B2 experts span >= 2 levels', ylabel='share of replies', xticks=x)
    axes[1].set_xticklabels(LIKERT)
    x = np.arange(len(RISKS))
    for i, r in enumerate(RESPONDERS):
        axes[2].bar(x + (i - 1.5) * width, [tab.get('share_k_copied_1', k, r).estimate for k in RISKS], width, label=r)
    axes[2].set(title='B4 exactly one expert copied a risky sentence', ylabel='share of replies', xticks=x)
    axes[2].set_xticklabels(RISKS)
    savefig(fig, OUT / 'figures/expert_uncertainty.png')
    plt.close(fig)


def write_summary(metrics: pd.DataFrame) -> None:
    tab = Table(metrics)
    f, c = tab.fmt, tab.count
    L = ['# CounselBench (Part B) expert-uncertainty summary', '',
         '_Descriptive only (D6). Generated by `experiments/counselbench/run_cb.py` from `metrics.csv`. '
         'Plan: `docs/ANALYSIS_PLAN_RQ1_DEMO.md` (frozen 2026-11-21). Estimates with reply-bootstrap 95% CIs '
         '(B = 2000, seed 20261121); responder strata resample within the responder._', '',
         f'Replies: {c("N_replies", "all")} (100 per responder), 5 experts each.', '',
         '## B1 Risk dimensions', '',
         '| k of 5 "yes" (medical advice) | ' + ' | '.join(STRATA) + ' |', '|---|' + '---|' * len(STRATA)]
    L += [f'| {k} | ' + ' | '.join(f('share_k_medical_yes', str(k), s) for s in STRATA) + ' |' for k in range(6)]
    L += ['', '| Quantity | ' + ' | '.join(STRATA) + ' |', '|---|' + '---|' * len(STRATA)]
    for metric, group, label in [('share_split_panel', 'medical_advice', 'Medical advice: split panel (1–4 of 5 yes)'),
                                 ('share_unanimous_panel', 'medical_advice', 'Medical advice: unanimous (0 or 5 yes)'),
                                 ('share_unanimous_panel', 'factual', 'Factual: all 5 experts give the same answer'),
                                 ('not_sure_rating_share', 'medical_advice', 'Medical advice: "not sure" share of ratings'),
                                 ('not_sure_reply_share', 'medical_advice', 'Medical advice: replies with ≥ 1 "not sure"'),
                                 ('not_sure_rating_share', 'factual', 'Factual: "not sure" share of ratings'),
                                 ('not_sure_reply_share', 'factual', 'Factual: replies with ≥ 1 "not sure"')]:
        L.append(f'| {label} | ' + ' | '.join(f(metric, group, s) for s in STRATA) + ' |')
    L += ['', '## B2 Likert dimensions: expert spread and IAA panel', '',
          '| Dimension | Quantity | ' + ' | '.join(STRATA) + ' |', '|---|---|' + '---|' * len(STRATA)]
    for dim in LIKERT:
        for metric, label in [('share_range_ge2', 'share spanning ≥ 2 levels'), ('mean_range', 'mean rating range'),
                              ('mean_modal_membership', 'mean IAA modal membership')]:
            L.append(f'| {dim} | {label} | ' + ' | '.join(f(metric, dim, s) for s in STRATA) + ' |')
    L += ['', '## B3 Toxicity panel decoded with the CounselBench scale words', '',
          '| Quantity | ' + ' | '.join(STRATA) + ' |', '|---|' + '---|' * len(STRATA),
          '| Contested panels (max J < 0.5 or top-1 − top-2 < 0.05) | '
          + ' | '.join(f('share_contested', 'toxicity', s) for s in STRATA) + ' |',
          '| Crisp-mean 2-tuple word ≠ decoded word | '
          + ' | '.join(f('share_crisp_word_differs', 'toxicity', s) for s in STRATA) + ' |']
    L += [f'| Decoded word: {w} | ' + ' | '.join(f('share_decoded_word', w, s) for s in STRATA) + ' |' for w in TOX_WORDS]
    L += ['', '## B4 Replies where exactly one expert copied a risky sentence', '',
          '| Risk | Stratum | Replies with k = 1 | Replies with k ≥ 1 | Share k = 1 (all replies) | '
          'Share k ≥ 2 (all replies) | Share k = 1 among flagged (k ≥ 1) |', '|---|---|---|---|---|---|---|']
    for risk in RISKS:
        for s in STRATA:
            L.append(f'| {risk} | {s} | {c("N_k_copied_1", risk, s)} | {c("N_k_copied_ge1", risk, s)} | '
                     f'{f("share_k_copied_1", risk, s)} | {f("share_k_copied_ge2", risk, s)} | '
                     f'{f("share_k1_among_flagged", risk, s)} |')
    L += ['', '## B5 By responder', '', 'Every table above reports overall and per responder (gpt4, llama3, gemini, human).',
          '', '## Deviations', '',
          '1. **B1 "k_yes" exists only for medical advice** (yes / no / not sure). Factual consistency is rated '
          '1–4 or "not sure" and the local release has no scale definitions, so no level is treated as "yes". '
          'For factual we report panel unanimity (all 5 experts give the same answer) and "not sure" prevalence. '
          'Toxicity is a Likert dimension and is covered by B2/B3. "Not sure" answers count as non-"yes".',
          '2. **"Not sure" prevalence** is reported two ways: share of all ratings, and share of replies with at '
          'least one "not sure".',
          '3. **IAA panel set summary (B2).** The per-reply set is `encode_scores(5 ratings, half_width=0.5)`. '
          'It is summarised by its modal membership, max over k = 1..5 of mu(k) at the bin centres (bins share '
          'edges, so the set height at an edge would add two neighbouring bins). The rating range and mean are '
          'in `per_item.csv`.',
          '4. **B3 codebook** reuses the five IT2 sets of `appropriateness_codebook(0.25)` (centred on 1..5) with '
          'the CounselBench toxicity words. The contested rule is the RQ2 default (`is_contested`, max J < 0.5 '
          'or top-1 − top-2 < 0.05).',
          '5. **B4 shares** are given per all replies and among replies with at least one copier '
          '(k ≥ 1); copier counts use the loader\'s placeholder-cleaned copy fields.',
          '', '## Files', '',
          '- `per_item.csv`: one row per reply (reply_id, responder, counts, ranges, IAA summaries, toxicity decoding). No text.',
          '- `metrics.csv`: metric, group, stratum, estimate, ci_lo, ci_hi.',
          '- `figures/expert_uncertainty.png`: B1, B2 and B4 by responder.', '']
    (OUT / 'SUMMARY.md').write_text('\n'.join(L), encoding='utf-8')


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t = load_replies()
    write_csv(t, OUT / 'per_item.csv')
    metrics = compute_metrics(t)
    write_csv(metrics, OUT / 'metrics.csv')
    figures(t, metrics)
    write_summary(metrics)
    print(f'Wrote {OUT.relative_to(ROOT)}: per_item.csv ({len(t)} replies), metrics.csv ({len(metrics)} rows), '
          'figures/, SUMMARY.md')


if __name__ == '__main__':
    main()
