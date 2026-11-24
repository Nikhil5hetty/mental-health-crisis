#!/usr/bin/env python3
"""RQ3 of docs/ANALYSIS_PLAN_WORDS.md: does the citizen/LLM word mismatch flip crisis-line actions? (WP5)

Offline and deterministic. Items: the 190 BHH complete-case items of results/rq2/per_item.csv. Crisis degree
m = 100 * k_llm_crisis / n_llm_observed (src/data/bhh.load_crisis_degree); reply word = that item's S3 word.

Each m is verbalised as one of four crisis-level words (None = "Remote Chance", Possible = "Realistic
Possibility", Likely = "Likely", Clear = "Almost Certain") under each codebook: citizen IT2 (RQ1), LLM IT2
(RQ2, prompt P1) and each citizen group's T1 (RQ1, UMF = LMF). Membership of m in a word = (UMF + LMF) / 2,
linearly interpolated on the 0.5 grid; the word with the largest membership wins, ties go to the more severe
word. If all four memberships are 0, the word whose EKM centroid centre is nearest m is chosen (ties: more
severe) and the case is counted as a deviation. Action = src/cww/rules.py table with crisis word and S3
reply word at membership 1.

Comparisons: LLM codebook vs citizen IT2 (primary, H3) and vs each citizen group's T1. Bootstrap over items,
B = 1000, default_rng(20261124), the same item draws for every comparison; percentile 95% CIs.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # trusted project only; works with python -I
import numpy as np
import pandas as pd
from src.cww import APPROPRIATENESS_WORDS
from src.cww.centroid import centre
from src.cww.rules import ACTIONS
# The verbalisation rule lives in src so that the word-meaning lab app applies exactly the same code.
from src.cww.verbalise import (CITIZEN, LEVELS, LLM, LLM_PROMPT, SEVERITY, MissingCodebook, action,  # noqa: F401
                               memberships, more_severe, neighbour_fallback, verbalise)
from src.cww.verbalise import load_codebooks as _load_codebooks

SEED = 20261124
B = 1000
N_ITEMS = 190
RQ1_BOOKS = ROOT / 'results/words/rq1/codebooks'
RQ2_BOOKS = ROOT / 'results/words/rq2/codebooks'
METRICS = ['word_disagreement_rate', 'action_flip_rate', 'under_escalation_rate', 'over_escalation_rate']


# ----------------------------------------------------------------------------- inputs

def load_items(rq2_per_item=ROOT / 'results/rq2/per_item.csv', bhh_root=None):
    """item_id, k/n LLM crisis counts, m (0..100) and the S3 reply word for the complete-case items."""
    from src.data import bhh
    s3 = pd.read_csv(rq2_per_item, usecols=['item_id', 'S3_word'])
    degree = bhh.load_crisis_degree(bhh_root).set_index('item_id')
    missing = set(s3.item_id) - set(degree.index)
    if missing:
        raise ValueError(f'Items without crisis degree: {sorted(missing)[:5]}')
    d = degree.loc[s3.item_id]
    if (d.n_llm_observed <= 0).any():
        raise ValueError('Item without observed LLM crisis labels')
    items = pd.DataFrame({'item_id': s3.item_id.to_numpy(), 'k_llm_crisis': d.k_llm_crisis.to_numpy(),
                          'n_llm_observed': d.n_llm_observed.to_numpy(),
                          'm': 100 * d.k_llm_crisis.to_numpy() / d.n_llm_observed.to_numpy(),
                          'reply_word': [APPROPRIATENESS_WORDS[int(i)] for i in s3.S3_word]})
    return items.sort_values('item_id').reset_index(drop=True)


def load_codebooks(rq1=RQ1_BOOKS, rq2=RQ2_BOOKS, prompt=LLM_PROMPT):
    """{codebook name: {level: IT2Set}} (src/cww/verbalise.load_codebooks with the committed result paths)."""
    return _load_codebooks(rq1, rq2, prompt)



def per_item(items, books):
    table = items[['item_id', 'k_llm_crisis', 'n_llm_observed', 'm', 'reply_word']].copy()
    for name, book in books.items():
        centres = {l: centre(book[l]) for l in SEVERITY}
        res = [verbalise(m, book, centres) for m in table.m]
        tag = column_tag(name)
        table[f'level_{tag}'] = [r[0] for r in res]
        table[f'word_{tag}'] = [LEVELS[r[0]] for r in res]
        table[f'fallback_{tag}'] = [int(r[1]) for r in res]
        table[f'action_{tag}'] = [action(r[0], w) for r, w in zip(res, table.reply_word)]
    return table


def column_tag(name):
    return name.replace('|', '_')


# ----------------------------------------------------------------------------- metrics

def comparisons(books):
    """(label, reference codebook): LLM vs citizen IT2 first (H3), then vs each citizen group."""
    return [(CITIZEN, CITIZEN)] + [(g, g) for g in books if g not in (CITIZEN, LLM)]


def rates(ref_level, llm_level, ref_action, llm_action):
    return {'word_disagreement_rate': np.mean(ref_level != llm_level),
            'action_flip_rate': np.mean(ref_action != llm_action),
            'under_escalation_rate': np.mean(llm_action < ref_action),
            'over_escalation_rate': np.mean(llm_action > ref_action)}


def bootstrap_indices(n, b=B, seed=SEED):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(b, n))


def metric_table(table, books, b=B, seed=SEED):
    idx = bootstrap_indices(len(table), b, seed)
    llm_level = table[f'level_{LLM}'].to_numpy()
    llm_action = table[f'action_{LLM}'].to_numpy()
    rows = []
    for label, ref in comparisons(books):
        ref_level = table[f'level_{column_tag(ref)}'].to_numpy()
        ref_action = table[f'action_{column_tag(ref)}'].to_numpy()
        point = rates(ref_level, llm_level, ref_action, llm_action)
        reps = [rates(ref_level[i], llm_level[i], ref_action[i], llm_action[i]) for i in idx]
        for metric in METRICS:
            values = np.array([r[metric] for r in reps])
            lo, hi = np.percentile(values, [2.5, 97.5], method='linear')
            rows.append(dict(metric=metric, comparison=f'llm_vs_{label}', estimate=float(point[metric]),
                             ci_lo=float(lo), ci_hi=float(hi), n_items=len(table)))
    return pd.DataFrame(rows)


def flip_matrix(table, books):
    rows = []
    for label, ref in comparisons(books):
        ref_action = table[f'action_{column_tag(ref)}']
        for i in range(len(ACTIONS)):
            for j in range(len(ACTIONS)):
                rows.append(dict(comparison=f'llm_vs_{label}', reference_action=f'A{i}', llm_action=f'A{j}',
                                 n=int(((ref_action == i) & (table[f'action_{LLM}'] == j)).sum())))
    return pd.DataFrame(rows)


POSTHOC_NO_FALLBACK = 'a_no_fallback_items_only'
POSTHOC_NEIGHBOUR = 'b_severe_neighbour_fallback'


def neighbour_levels(table, books):
    """Per codebook: levels and actions with the post-hoc 'more severe neighbour' fallback."""
    out = {}
    for name, book in books.items():
        levels = np.array([verbalise(m, book, fallback='neighbour')[0] for m in table.m])
        actions = np.array([action(l, w) for l, w in zip(levels, table.reply_word)])
        out[name] = (levels, actions)
    return out


def posthoc_table(table, books, b=B, seed=SEED):
    """POST-HOC sensitivity (PM review of WP5), same bootstrap draws as the primary metrics.

    (a) flips counted only on items where neither codebook used the all-zero fallback; the rate keeps all items
        in the denominator (plus the share of such items and the rate among them).
    (b) primary metrics recomputed with the alternative 'more severe neighbouring word' fallback.
    """
    idx = bootstrap_indices(len(table), b, seed)
    alt = neighbour_levels(table, books)
    llm_action = table[f'action_{LLM}'].to_numpy()
    llm_fb = table[f'fallback_{LLM}'].to_numpy() == 1
    rows = []

    def add(analysis, metric, label, fn, conditional=False):
        point = fn(np.arange(len(table)))
        values = np.array([fn(i) for i in idx], dtype=float)
        undefined = int((~np.isfinite(values)).sum())
        if undefined and not conditional:
            raise ValueError(f'Undefined bootstrap value for {analysis}/{metric}/{label}')
        if undefined == len(values) or not np.isfinite(point):
            raise ValueError(f'No defined value for {analysis}/{metric}/{label}')
        # A conditional rate is undefined in a replicate without no-fallback items; its CI uses the others.
        lo, hi = np.percentile(values[np.isfinite(values)], [2.5, 97.5], method='linear')
        rows.append(dict(analysis=analysis, metric=metric, comparison=f'llm_vs_{label}', estimate=float(point),
                         ci_lo=float(lo), ci_hi=float(hi), n_items=len(table), n_boot_undefined=undefined))

    for label, ref in comparisons(books):
        tag = column_tag(ref)
        ref_action = table[f'action_{tag}'].to_numpy()
        clean = ~(llm_fb | (table[f'fallback_{tag}'].to_numpy() == 1))
        flip = ref_action != llm_action
        add(POSTHOC_NO_FALLBACK, 'action_flip_rate', label, lambda i: np.mean(flip[i] & clean[i]))
        add(POSTHOC_NO_FALLBACK, 'under_escalation_rate', label, lambda i: np.mean((llm_action[i] < ref_action[i]) & clean[i]))
        add(POSTHOC_NO_FALLBACK, 'over_escalation_rate', label, lambda i: np.mean((llm_action[i] > ref_action[i]) & clean[i]))
        add(POSTHOC_NO_FALLBACK, 'share_items_no_fallback', label, lambda i: np.mean(clean[i]))
        add(POSTHOC_NO_FALLBACK, 'action_flip_rate_among_no_fallback_items', label,
            lambda i: np.sum(flip[i] & clean[i]) / np.sum(clean[i]) if np.sum(clean[i]) else np.nan, conditional=True)
        r_level, r_action = alt[ref]
        l_level, l_action = alt[LLM]
        for metric in METRICS:
            add(POSTHOC_NEIGHBOUR, metric, label,
                lambda i, metric=metric: rates(r_level[i], l_level[i], r_action[i], l_action[i])[metric])
    return pd.DataFrame(rows), alt


def verdict_h3(ci_lo):
    return 'supported' if ci_lo > 0 else 'not supported'


# ----------------------------------------------------------------------------- outputs

def write_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format='%.17g', lineterminator='\n')


def figures(table, books, metrics, flips, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.edgecolor': '#52514e'})
    blue, orange, grey, ink = '#2a78d6', '#eb6834', '#8a8984', '#52514e'

    fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
    for ax, (name, colour, title) in zip(axes, [(CITIZEN, blue, 'Citizen IT2 codebook'),
                                                (LLM, orange, f'LLM IT2 codebook ({LLM_PROMPT})')]):
        for k, level in enumerate(SEVERITY):
            s = books[name][level]
            ax.fill_between(s.x, s.lmf, s.umf, color=colour, alpha=.15 + .15 * k, lw=0)
            ax.plot(s.x, (s.umf + s.lmf) / 2, color=colour, lw=1)
            peak = s.x[np.argmax((s.umf + s.lmf) / 2)]
            ax.text(peak, 1.04, f'{level}\n({LEVELS[level]})', ha='center', va='bottom', fontsize=6.5, color='#0b0b0b')
        counts = table.m.round(6).value_counts().sort_index()
        ax.vlines(counts.index, 0, -.06 - .1 * counts.to_numpy() / counts.max(), color=ink, lw=1.5)
        ax.set_ylim(-.2, 1.3); ax.set_ylabel('Membership'); ax.set_title(title, loc='left', fontsize=8)
    axes[-1].set_xlabel('Crisis degree m = 100 x share of LLM crisis labels (ticks below 0: items, length ~ count)')
    axes[-1].set_xlim(0, 100)
    fig.tight_layout(); fig.savefig(out / 'level_words.png', dpi=150, metadata={'Software': None}); plt.close(fig)

    primary = flips[flips.comparison == f'llm_vs_{CITIZEN}']
    grid = np.zeros((len(ACTIONS), len(ACTIONS)), int)
    for r in primary.itertuples():
        grid[int(r.reference_action[1:]), int(r.llm_action[1:])] = r.n
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(grid, cmap='Blues', vmin=0)
    for i in range(len(ACTIONS)):
        for j in range(len(ACTIONS)):
            ax.text(j, i, str(grid[i, j]), ha='center', va='center',
                    color='white' if grid[i, j] > grid.max() / 2 else '#0b0b0b', fontsize=9)
    ticks = [f'A{i}' for i in range(len(ACTIONS))]
    ax.set_xticks(range(len(ACTIONS)), ticks); ax.set_yticks(range(len(ACTIONS)), ticks)
    ax.set_xlabel('Action under LLM codebook'); ax.set_ylabel('Action under citizen codebook')
    ax.set_title(f'Actions, {len(table)} items (off-diagonal = flips)', fontsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(); fig.savefig(out / 'flip_matrix.png', dpi=150, metadata={'Software': None}); plt.close(fig)

    labels = [c for c in metrics.comparison.unique()]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    styles = {'action_flip_rate': (ink, 'o', 'Action flip'), 'under_escalation_rate': (orange, 's', 'Under-escalation (LLM lower)'),
              'over_escalation_rate': (blue, '^', 'Over-escalation (LLM higher)')}
    for k, (metric, (colour, marker, name)) in enumerate(styles.items()):
        sel = metrics[metrics.metric == metric].set_index('comparison').loc[labels]
        y = np.arange(len(labels)) + (k - 1) * .22
        ax.hlines(y, sel.ci_lo, sel.ci_hi, color=colour, lw=2)
        ax.plot(sel.estimate, y, marker, color=colour, ms=6, mec='white', mew=1, label=name, ls='none')
    ax.set_yticks(range(len(labels)), [l.replace('llm_vs_', 'LLM vs ').replace('|', ' / ') for l in labels])
    ax.invert_yaxis(); ax.set_xlim(left=0)
    ax.set_xlabel('Share of items (95% bootstrap CI)'); ax.grid(axis='x', color='#e4e3df', lw=.6); ax.set_axisbelow(True)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.14), ncol=3, frameon=False)
    fig.tight_layout(); fig.savefig(out / 'flip_rates.png', dpi=150, metadata={'Software': None}); plt.close(fig)


def posthoc_section(table, books, metrics, posthoc, alt):
    """SUMMARY section for the post-hoc sensitivities and the generated paper notes."""
    comp = f'llm_vs_{CITIZEN}'

    def ph(analysis, metric):
        return posthoc[(posthoc.analysis == analysis) & (posthoc.metric == metric) & (posthoc.comparison == comp)].iloc[0]

    def pm(metric):
        return metrics[(metrics.metric == metric) & (metrics.comparison == comp)].iloc[0]

    cit, llm = table[f'action_{CITIZEN}'].to_numpy(), table[f'action_{LLM}'].to_numpy()
    clean = ~((table[f'fallback_{CITIZEN}'] == 1) | (table[f'fallback_{LLM}'] == 1)).to_numpy()
    n_clean_flip = int(((cit != llm) & clean).sum())
    alt_cit, alt_llm = alt[CITIZEN][1], alt[LLM][1]
    lines = ['## Post-hoc sensitivity (added after PM review; not in the frozen plan)', '',
             'Labelled post-hoc: chosen after seeing the primary result, to show how much H3 depends on the all-zero fallback. '
             f'Same {len(table)} items and the same bootstrap draws as the primary analysis. Values for every comparison: '
             '`sensitivity_posthoc.csv`.', '',
             '- (a) Only items decided without a fallback on both sides count as flips (denominator: all items).',
             '- (b) Alternative fallback: the more severe of the two words neighbouring m (left: support ends closest below m; '
             'right: support starts closest above m; at an end of the scale the single neighbour). Applied to every codebook.', '',
             '| LLM vs citizen IT2 | Action flip | Under-escalation | Over-escalation |', '|---|---|---|---|',
             f'| Primary (nearest-centre fallback) | {fmt(pm("action_flip_rate"))} | {fmt(pm("under_escalation_rate"))} | '
             f'{fmt(pm("over_escalation_rate"))} |',
             f'| (a) No-fallback items only | {fmt(ph(POSTHOC_NO_FALLBACK, "action_flip_rate"))} | '
             f'{fmt(ph(POSTHOC_NO_FALLBACK, "under_escalation_rate"))} | {fmt(ph(POSTHOC_NO_FALLBACK, "over_escalation_rate"))} |',
             f'| (b) More severe neighbour fallback | {fmt(ph(POSTHOC_NEIGHBOUR, "action_flip_rate"))} | '
             f'{fmt(ph(POSTHOC_NEIGHBOUR, "under_escalation_rate"))} | {fmt(ph(POSTHOC_NEIGHBOUR, "over_escalation_rate"))} |', '',
             f'(a) {n_clean_flip} of {len(table)} items flip without any fallback; these come from '
             f'{int(clean.sum())} no-fallback items (share {fmt(ph(POSTHOC_NO_FALLBACK, "share_items_no_fallback"))}; '
             f'flip rate among them {fmt(ph(POSTHOC_NO_FALLBACK, "action_flip_rate_among_no_fallback_items"))}). '
             f'(b) {int((alt_cit != alt_llm).sum())} of {len(table)} items flip. '
             f'H3 rule applied descriptively: (a) {verdict_h3(ph(POSTHOC_NO_FALLBACK, "action_flip_rate").ci_lo)}, '
             f'(b) {verdict_h3(ph(POSTHOC_NEIGHBOUR, "action_flip_rate").ci_lo)}.', '']
    gap = table[(table[f'fallback_{LLM}'] == 1) & (table[f'fallback_{CITIZEN}'] == 0)]
    if len(gap):
        parts = []
        for m, g in gap.groupby(gap.m.round(6), sort=True):
            r = g.iloc[0]
            parts.append(f'm = {m:.1f} ({len(g)} item{"s" if len(g) > 1 else ""}): citizen "{r[f"word_{CITIZEN}"]}", LLM has no word '
                         f'(fallback "{r[f"word_{LLM}"]}")')
        lines += ['Gap finding: crisis degrees covered by a citizen word but by no LLM word: ' + '; '.join(parts) + '.', '']
    pairs = sorted({(int(a), int(c)) for a, c in zip(cit, llm) if a != c})
    within = sorted({x for p in pairs for x in p})
    lines += ['## Note for the paper', '']
    if pairs and set(within) <= {0, 1}:
        lines.append(f'All {int((cit != llm).sum())} flips (LLM vs citizen IT2) are between A0 and A1: on this dataset the word '
                     'mismatch changes whether a helpline is suggested, never whether a human is involved (A2/A3). '
                     f'Under the post-hoc fallback (b) the flips are between ' +
                     ', '.join(sorted({f"A{min(a, c)}/A{max(a, c)}" for a, c in zip(alt_cit, alt_llm) if a != c})) + '.')
    else:
        lines.append('Flip pairs (citizen action, LLM action): ' + ', '.join(f'(A{a}, A{c})' for a, c in pairs) + '.')
    lines.append('')
    return lines


def fmt(r):
    return f'{r.estimate:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}]'


def summary(table, books, metrics, flips, b, versions, posthoc=None, alt=None):
    def get(metric, comparison):
        return metrics[(metrics.metric == metric) & (metrics.comparison == comparison)].iloc[0]

    primary = f'llm_vs_{CITIZEN}'
    flip = get('action_flip_rate', primary)
    flipped = (table[f'action_{CITIZEN}'] != table[f'action_{LLM}']).to_numpy()
    fb = ((table[f'fallback_{CITIZEN}'] == 1) | (table[f'fallback_{LLM}'] == 1)).to_numpy()
    n_flip, n_flip_fb = int(flipped.sum()), int((flipped & fb).sum())
    lines = ['# RQ3: Citizen vs LLM word meaning and crisis-line actions', '',
             'Generated by `experiments/words/run_rq3.py` (docs/ANALYSIS_PLAN_WORDS.md, RQ3). Ids and numbers only.', '',
             f'Items: {len(table)} BHH complete-case items (results/rq2/per_item.csv). Crisis degree m = 100 x k_llm_crisis / '
             f'n_llm_observed ({table.m.nunique()} distinct values). Reply word = S3 word.',
             f'Codebooks: citizen IT2 (RQ1), LLM IT2 (RQ2, prompt {LLM_PROMPT}), {len(books) - 2} citizen-group T1 (RQ1).',
             f'Level words: ' + '; '.join(f'{k} = "{v}"' for k, v in LEVELS.items()) + '.',
             f'Bootstrap: B = {b} item resamples, seed {SEED}, same draws for every comparison; percentile 95% CIs.', '',
             '## H3', '',
             f'Action flip rate, LLM vs citizen codebook: {fmt(flip)} ({n_flip} of {len(table)} items). '
             f'Verdict: **{verdict_h3(flip.ci_lo)}** (rule: CI excludes 0, i.e. lower bound > 0).', '',
             f'Flips where the citizen or the LLM word came from the all-zero fallback: {n_flip_fb} of {n_flip}; '
             f'flips with both words chosen by maximum membership: {n_flip - n_flip_fb}. '
             f'Items with no fallback on either side: {int((~fb).sum())} (descriptive; see Deviations).', '',
             '## Rates by comparison', '',
             '| Comparison | Word disagreement | Action flip | Under-escalation (LLM < ref) | Over-escalation (LLM > ref) |',
             '|---|---|---|---|---|']
    for comp in metrics.comparison.unique():
        lines.append(f'| {comp.replace("llm_vs_", "LLM vs ").replace("|", " / ")} | ' +
                     ' | '.join(fmt(get(m, comp)) for m in METRICS) + ' |')

    lines += ['', '## Flip matrix (LLM vs citizen IT2)', '', 'Rows: action under the citizen codebook; columns: under the LLM codebook.', '',
              '| Citizen \\ LLM | ' + ' | '.join(f'A{j}' for j in range(len(ACTIONS))) + ' |',
              '|---|' + '---:|' * len(ACTIONS)]
    pm = flips[flips.comparison == primary]
    for i in range(len(ACTIONS)):
        row = pm[pm.reference_action == f'A{i}'].set_index('llm_action').n
        lines.append(f'| A{i} | ' + ' | '.join(str(int(row[f'A{j}'])) for j in range(len(ACTIONS))) + ' |')
    lines += ['', 'Actions: ' + '; '.join(f'A{i} = {a.split(" ", 1)[1]}' for i, a in enumerate(ACTIONS)) + '.', '']

    lines += ['## Verbalisation of each crisis degree', '',
              '| m | Items | ' + ' | '.join(['Citizen', 'LLM'] + [g.replace('|', ' / ') for g in books if g not in (CITIZEN, LLM)]) + ' |',
              '|---:|---:|' + '---|' * len(books)]
    names = [CITIZEN, LLM] + [g for g in books if g not in (CITIZEN, LLM)]
    for m, g in table.groupby(table.m.round(6), sort=True):
        r = g.iloc[0]
        cells = [r[f'level_{column_tag(n)}'] + ('*' if r[f'fallback_{column_tag(n)}'] else '') for n in names]
        lines.append(f'| {m:.2f} | {len(g)} | ' + ' | '.join(cells) + ' |')
    lines += ['', '`*` = all four memberships were 0; nearest-centre fallback used.', '']

    if posthoc is not None:
        lines += posthoc_section(table, books, metrics, posthoc, alt)
    fallback = {n: int(table[f'fallback_{column_tag(n)}'].sum()) for n in names}
    distinct = {n: int(table.loc[table[f'fallback_{column_tag(n)}'] == 1, 'm'].round(6).nunique()) for n in names}
    lines += ['## Deviations', '',
              '- All-zero fallback (brief: count and report). Items whose m has membership 0 in all four level words, '
              'verbalised by the nearest EKM centroid centre (ties to the more severe word): ' +
              '; '.join(f'{n.replace("|", " / ")} {fallback[n]} items ({distinct[n]} distinct m)' for n in names) + '. '
              'In every codebook "Remote Chance" has membership 0 at m = 0 and "Almost Certain" has membership 0 at m = 100 '
              '(the quantile trapezoids rise from q05 and fall to q95 inside 0..100); intermediate m values can fall in gaps '
              'between level words (e.g. the LLM codebook between "Remote Chance" and "Realistic Possibility"). '
              'The fallback therefore decides many verbalisations, including flips (count in the H3 section). '
              'PM ruling (WP5 review): the nearest-centre fallback stays the primary rule; at m = 0 / 100 it is harmless '
              'because all codebooks agree, and an LLM gap is itself a finding. Post-hoc sensitivities are in their own section.',
              f'- The LLM codebook is the primary-prompt ({LLM_PROMPT}) LLM IT2 from RQ2; P2 is not analysed for RQ3.',
              '- Under- and over-escalation are reported for every comparison; over-escalation (LLM action > reference action) '
              'is not in the plan and is descriptive. Group comparisons are descriptive; H3 uses the citizen IT2 comparison only.',
              '- Max-membership ties go to the more severe word (plan); the rule-table action is computed with rules.infer at '
              'membership 1 and checked against the table for every item.',
              '- Verdict wording: "supported" only if the CI lower bound of the flip rate is above 0 (exploratory, no multiplicity correction).', '',
              'Files: `per_item.csv` (item ids, m, reply word, and per codebook the level, CAPphrase word, fallback flag and action; '
              'read with `keep_default_na=False`, since "None" is a level name), `metrics.csv`, `flip_matrix.csv` (all comparisons), '
              '`sensitivity_posthoc.csv` (post-hoc), `figures/`. `results/words/rq2/codebooks/llm_it2.csv` holds P1 and P2 rows: '
              'filter on `prompt` when reusing it.', '',
              f'Software: numpy {versions["numpy"]}, pandas {versions["pandas"]}, matplotlib {versions["matplotlib"]}. '
              'CSV floats: 17 significant digits.', '']
    return '\n'.join(lines)


def run(items, books, out, b=B, seed=SEED, make_figures=True):
    table = per_item(items, books)
    metrics = metric_table(table, books, b, seed)
    flips = flip_matrix(table, books)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(table, out / 'per_item.csv')
    write_csv(metrics, out / 'metrics.csv')
    write_csv(flips, out / 'flip_matrix.csv')
    posthoc, alt = posthoc_table(table, books, b, seed)
    write_csv(posthoc, out / 'sensitivity_posthoc.csv')
    import matplotlib
    versions = dict(numpy=np.__version__, pandas=pd.__version__, matplotlib=matplotlib.__version__)
    if make_figures:
        figdir = out / 'figures'; figdir.mkdir(exist_ok=True)
        figures(table, books, metrics, flips, figdir)
    (out / 'SUMMARY.md').write_text(summary(table, books, metrics, flips, b, versions, posthoc, alt), encoding='utf-8')
    return table, metrics, flips


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--rq1-codebooks', type=Path, default=RQ1_BOOKS)
    parser.add_argument('--rq2-codebooks', type=Path, default=RQ2_BOOKS)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/words/rq3')
    args = parser.parse_args()
    books = load_codebooks(args.rq1_codebooks, args.rq2_codebooks)
    items = load_items()
    if len(items) != N_ITEMS:
        raise ValueError(f'Expected {N_ITEMS} complete-case items, got {len(items)}')
    table, metrics, _ = run(items, books, args.output)
    flip = metrics[(metrics.metric == 'action_flip_rate') & (metrics.comparison == f'llm_vs_{CITIZEN}')].iloc[0]
    print(f'Wrote {args.output}; flip rate LLM vs citizen = {flip.estimate:.4f} [{flip.ci_lo:.4f}, {flip.ci_hi:.4f}]')


if __name__ == '__main__':
    main()
