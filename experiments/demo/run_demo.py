#!/usr/bin/env python3
"""Part C: escalation demo (Perceptual Computer decoder to a public-service action).

Implements Part C of docs/ANALYSIS_PLAN_RQ1_DEMO.md (frozen 2026-11-21).
Offline, seeded; writes ids and numbers only to results/demo/.

Run from anywhere:  python -I experiments/demo/run_demo.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.common import (VERDICT_RULE, Table, count_row, est_row, m_cond_mean, m_mean, m_qwk,  # noqa: E402
                                pyplot, rq2, savefig, to_frame, verdict, write_csv)
from experiments.rq1 import run_rq1  # noqa: E402
from src.cww import APPROPRIATENESS_WORDS, appropriateness_codebook, encode_scores  # noqa: E402
from src.cww.rules import (ACTIONS, CRISIS_WORDS, RULE_TABLE, crisis_memberships,  # noqa: E402
                           crisp_memberships, infer)

OUT = ROOT / 'results/demo'
PIPELINES = ['P-fuzzy', 'P-crisp', 'Oracle']
CRISP_CRISIS = {'crisis': 'Clear', 'no_crisis': 'None', 'tie': 'Possible'}
SHORT = {w: w.split()[0].split('/')[0] for w in APPROPRIATENESS_WORDS}  # Harmful, Inappropriate, ...


def build_items() -> pd.DataFrame:
    rq1 = run_rq1.load_items().set_index('item_id')
    items = rq2.load_items()
    book = appropriateness_codebook(0.25)
    rows = []
    for it in items.itertuples(index=False):
        a = rq1.loc[it.item_id]
        s3 = rq2.fuzzy_system('S3', it.runs, rq2.DEFAULT.fou_width)
        s2_word = APPROPRIATENESS_WORDS[rq2.crisp_system('S2', it.runs, rq2.DEFAULT)['word']]
        oracle_sims = book.decode(encode_scores([it.H1, it.H2]))[2]
        inputs = {
            'P-fuzzy': (crisis_memberships(a.mu_L), s3['sims']),
            'P-crisp': (crisp_memberships(CRISP_CRISIS[a.llm_majority], CRISIS_WORDS),
                        crisp_memberships(s2_word, APPROPRIATENESS_WORDS)),
            'Oracle': (crisis_memberships(a.mu_H), oracle_sims),
        }
        row = {'item_id': it.item_id, 'mu_L': a.mu_L, 'mu_H': a.mu_H, 'llm_majority': a.llm_majority,
               'S2_word': s2_word, 'S3_word': APPROPRIATENESS_WORDS[s3['word']]}
        for p, (crisis, reply) in inputs.items():
            action, strengths = infer(crisis, reply)
            tag = p.replace('-', '_').lower()
            if p != 'P-crisp':
                row.update({f'{tag}_crisis_{w}': v for w, v in crisis.items()})
                row.update({f'{tag}_reply_{SHORT[w]}': v for w, v in reply.items()})
            row.update({f'{tag}_strength_A{i}': s for i, s in enumerate(strengths)})
            row[f'{tag}_action'] = action
            row[f'{tag}_tied'] = sum(s == max(strengths) for s in strengths) > 1
        rows.append(row)
    out = pd.DataFrame(rows)
    assert len(out) == 190
    return out


def compute_metrics(t: pd.DataFrame) -> pd.DataFrame:
    t = t.reset_index(drop=True)
    n = len(t)
    o = t.oracle_action.to_numpy()
    high = o >= 2
    rows = [count_row('N_items', 'all', 'overall', n),
            count_row('N_oracle_A2_A3', 'all', 'overall', int(high.sum()))]
    for p in PIPELINES:
        a = t[f'{p.replace("-", "_").lower()}_action'].to_numpy()
        for i in range(len(ACTIONS)):
            rows.append(est_row('share_action', f'{p}|A{i}', 'overall', m_mean(a == i), n))
        rows.append(count_row('N_tied_strengths', p, 'overall', int(t[f'{p.replace("-", "_").lower()}_tied'].sum())))
        if p == 'Oracle':
            continue
        rows.append(est_row('exact_agreement_with_oracle', p, 'overall', m_mean(a == o), n))
        rows.append(est_row('qwk_with_oracle', p, 'overall', m_qwk(a, o), n))
        rows.append(est_row('under_escalation_rate', p, 'overall', m_mean(a < o), n))
        rows.append(est_row('under_escalation_rate', p, 'oracle_A2_A3', m_cond_mean(a < o, high), n))
        rows.append(est_row('over_escalation_rate', p, 'overall', m_mean(a > o), n))
    fuzzy, crisp = t.p_fuzzy_action.to_numpy(), t.p_crisp_action.to_numpy()
    f_f, f_c = m_cond_mean(fuzzy < o, high), m_cond_mean(crisp < o, high)
    rows.append(est_row('DIFF_under_escalation_rate', 'P-fuzzy-P-crisp', 'oracle_A2_A3',
                        lambda idx: f_f(idx) - f_c(idx), n))
    return to_frame(rows)


def worked_cases(t: pd.DataFrame) -> pd.DataFrame:
    """4 largest oracle - P-crisp gaps, then 4 largest P-fuzzy - P-crisp gaps among the rest (ties: lowest id)."""
    t = t.assign(gap_oracle_minus_crisp=t.oracle_action - t.p_crisp_action,
                 gap_fuzzy_minus_crisp=t.p_fuzzy_action - t.p_crisp_action)
    first = t.sort_values(['gap_oracle_minus_crisp', 'item_id'], ascending=[False, True]).head(4)
    rest = t[~t.item_id.isin(first.item_id)]
    second = rest.sort_values(['gap_fuzzy_minus_crisp', 'item_id'], ascending=[False, True]).head(4)
    cols = ['item_id', 'gap_oracle_minus_crisp', 'gap_fuzzy_minus_crisp', 'mu_L', 'mu_H', 'llm_majority',
            'S2_word', 'S3_word', *[c for c in t.columns if c.startswith(('p_fuzzy_crisis', 'p_fuzzy_reply',
                                                                          'oracle_crisis', 'oracle_reply'))],
            'p_fuzzy_action', 'p_crisp_action', 'oracle_action']
    out = pd.concat([first.assign(selection='largest oracle - P-crisp gap'),
                     second.assign(selection='largest P-fuzzy - P-crisp gap')])
    out['rank'] = [1, 2, 3, 4, 1, 2, 3, 4][:len(out)]
    return out[['selection', 'rank', *cols]].reset_index(drop=True)


def rule_table_frame() -> pd.DataFrame:
    return pd.DataFrame([{'crisis_word': c, 'reply_word': r, 'action_index': a, 'action': ACTIONS[a]}
                         for (c, r), a in RULE_TABLE.items()])


def figures(t: pd.DataFrame, metrics: pd.DataFrame) -> None:
    plt = pyplot()
    tab = Table(metrics)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, p in zip(axes[:2], ['P-fuzzy', 'P-crisp']):
        conf = np.zeros((4, 4), int)
        np.add.at(conf, (t.oracle_action.to_numpy(), t[f'{p.replace("-", "_").lower()}_action'].to_numpy()), 1)
        ax.imshow(conf, cmap='Blues')
        for i, j in itertools.product(range(4), range(4)):
            ax.text(j, i, conf[i, j], ha='center', va='center',
                    color='white' if conf[i, j] > conf.max() / 2 else 'black')
        ax.set(title=f'{p} vs oracle', xticks=range(4), yticks=range(4), xticklabels=['A0', 'A1', 'A2', 'A3'],
               yticklabels=['A0', 'A1', 'A2', 'A3'], xlabel=f'{p} action', ylabel='oracle action')
    width = 0.27
    x = np.arange(4)
    for k, p in enumerate(PIPELINES):
        axes[2].bar(x + (k - 1) * width, [tab.get('share_action', f'{p}|A{i}').estimate for i in range(4)],
                    width, label=p)
    axes[2].set(title='Action distribution', xticks=x, ylabel='share of items')
    axes[2].set_xticklabels(['A0', 'A1', 'A2', 'A3'])
    axes[2].legend()
    savefig(fig, OUT / 'figures/actions.png')
    plt.close(fig)


def write_summary(metrics: pd.DataFrame, cases: pd.DataFrame) -> None:
    tab = Table(metrics)
    f, c = tab.fmt, tab.count
    L = ['# Escalation demo (Part C) summary', '',
         '_Generated by `experiments/demo/run_demo.py` from `metrics.csv`, `worked_cases.csv` and `rule_table.csv`. '
         'Plan: `docs/ANALYSIS_PLAN_RQ1_DEMO.md` (frozen 2026-11-21). Estimates with item-bootstrap 95% CIs '
         '(B = 2000, seed 20261121)._', '',
         '**Caveat.** The oracle uses the same rule table, so this compares input representations (fuzzy vs crisp), '
         'not the rule table itself. The rule table is a design proposal, not clinically validated.', '',
         f'Items: {c("N_items", "all")} (RQ1 crisis degree × RQ2 complete-case reply verdicts); '
         f'oracle A2/A3 items: {c("N_oracle_A2_A3", "all")}.', '',
         '## Rule table (crisis level × reply word → action)', '',
         '| Crisis \\ Reply | ' + ' | '.join(SHORT[w] for w in APPROPRIATENESS_WORDS) + ' |',
         '|---|' + '---|' * len(APPROPRIATENESS_WORDS)]
    L += [f'| {cw} | ' + ' | '.join(f'A{RULE_TABLE[(cw, r)]}' for r in APPROPRIATENESS_WORDS) + ' |' for cw in CRISIS_WORDS]
    L += ['', 'Actions: ' + '; '.join(ACTIONS) + '.', '',
          '## Action distribution', '', '| Pipeline | A0 | A1 | A2 | A3 |', '|---|---|---|---|---|']
    L += [f'| {p} | ' + ' | '.join(f('share_action', f'{p}|A{i}') for i in range(4)) + ' |' for p in PIPELINES]
    L += ['', '## Agreement with the oracle and escalation errors', '',
          '| Pipeline | Exact agreement | QWK (A0–A3) | Under-escalation (all) | Under-escalation (oracle A2/A3) '
          '| Over-escalation (all) |', '|---|---|---|---|---|---|']
    for p in ['P-fuzzy', 'P-crisp']:
        L.append(f'| {p} | {f("exact_agreement_with_oracle", p)} | {f("qwk_with_oracle", p)} | '
                 f'{f("under_escalation_rate", p)} | {f("under_escalation_rate", p, "oracle_A2_A3")} | '
                 f'{f("over_escalation_rate", p)} |')
    h3 = tab.get('DIFF_under_escalation_rate', 'P-fuzzy-P-crisp', 'oracle_A2_A3')
    L += ['', '## Hypothesis', '', VERDICT_RULE, '',
          '| Hypothesis | Quantity | Estimate [95% CI] | Verdict |', '|---|---|---|---|',
          f'| H3 | Under-escalation among oracle A2/A3, P-fuzzy − P-crisp (paired); ≤ 0 | '
          f'{f("DIFF_under_escalation_rate", "P-fuzzy-P-crisp", "oracle_A2_A3")} | {verdict(h3, "lower")} |',
          '', '## Worked cases (selected by rule; ids and numbers only)', '',
          '| Selection | Rank | item_id | mu_L | mu_H | LLM majority | S2 word | S3 word | P-fuzzy | P-crisp | Oracle |',
          '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in cases.itertuples(index=False):
        L.append(f'| {r.selection} | {r.rank} | {r.item_id} | {r.mu_L:.3f} | {r.mu_H:.2f} | {r.llm_majority} | '
                 f'{SHORT[r.S2_word]} | {SHORT[r.S3_word]} | A{r.p_fuzzy_action} | A{r.p_crisp_action} | '
                 f'A{r.oracle_action} |')
    L += ['', 'Crisis-word memberships and reply similarities for these items: `worked_cases.csv`.', '',
          f'Items whose winning action was decided by the tie rule (more severe): P-fuzzy {c("N_tied_strengths", "P-fuzzy")}, '
          f'P-crisp {c("N_tied_strengths", "P-crisp")}, oracle {c("N_tied_strengths", "Oracle")}.',
          '', '## Deviations', '',
          '1. **Worked cases, second rule.** The 4 largest P-fuzzy − P-crisp gaps are taken among items not '
          'already selected by the first rule, so the 8 ids are distinct.',
          '2. **Inputs recomputed with the RQ2 code.** S3 similarities and the S2 word are computed with the '
          'unchanged RQ2 functions and defaults (fou_width 0.25). The oracle reply similarities use the same '
          'codebook. Crisis-word memberships are evaluated at the exact degree (not on a grid).',
          '3. **Tie counts.** "Tied" means at least two actions share the maximum strength; the tie rule then '
          'picks the more severe action. Crisp inputs always give a single strength of 1.',
          '4. **CIs** resample all 190 items; rates among oracle A2/A3 items are recomputed in each replicate, and '
          'the H3 difference is paired (same replicate for both pipelines).',
          '', '## Files', '',
          '- `per_item.csv`: degrees, memberships, similarities, action strengths and actions per pipeline. No text.',
          '- `metrics.csv`: metric, group, stratum, estimate, ci_lo, ci_hi.',
          '- `worked_cases.csv`, `rule_table.csv`, `figures/actions.png`.', '']
    (OUT / 'SUMMARY.md').write_text('\n'.join(L), encoding='utf-8')


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t = build_items()
    write_csv(t, OUT / 'per_item.csv')
    metrics = compute_metrics(t)
    write_csv(metrics, OUT / 'metrics.csv')
    cases = worked_cases(t)
    write_csv(cases, OUT / 'worked_cases.csv')
    write_csv(rule_table_frame(), OUT / 'rule_table.csv')
    figures(t, metrics)
    write_summary(metrics, cases)
    print(f'Wrote {OUT.relative_to(ROOT)}: per_item.csv ({len(t)} items), metrics.csv ({len(metrics)} rows), '
          'worked_cases.csv, rule_table.csv, figures/, SUMMARY.md')


if __name__ == '__main__':
    main()
