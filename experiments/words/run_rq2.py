#!/usr/bin/env python3
"""RQ2 of docs/ANALYSIS_PLAN_WORDS.md: citizen vs LLM probability-word models (WP4).

Offline and deterministic. Inputs: data/llm_answers/answers.csv (+ superseded.csv for reporting),
the CAPphrase raw CSVs (citizen bootstrap) and the RQ1 codebooks in results/words/rq1/codebooks/
(citizen point estimates; the script refits them from the raw data and stops if they differ).

Valid answer = non-empty value in [0, 100] (strict parsing, D19). A model x prompt x term with fewer than
30 valid answers is excluded and reported. Per model x term T1 = quantile trapezoid (q05, q25, q75, q95)
on 0..100 step 0.5 (src/cww/quantile.py, as RQ1). LLM IT2 per term = UMF max / LMF min over the eligible
models. P1 is primary (metrics.csv); P2 is the sensitivity analysis (sensitivity.csv).

Bootstrap: B = 1000, SeedSequence(20261124) spawned into three independent streams (citizen, P1, P2).
Each replicate resamples whole citizen respondents (all 19 values together, as in RQ1) and re-derives the
pooled T1 and the six-group IT2; independently, each model's valid answers are resampled per term. The same
citizen replicates are used for P1 and P2. CIs are percentile (2.5, 97.5; numpy linear).

CSV floats use 17 significant digits; PNG metadata has no software/timestamp field.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # trusted project only; works with python -I
import numpy as np
import pandas as pd
from src.data import capphrase
from src.cww.quantile import quantile_trapezoid, envelope, group_models, X
from src.cww.centroid import centroid_t1, centre
from src.cww.similarity import jaccard
from src.cww.sets import T1Set, IT2Set

SEED = 20261124
B = 1000
MIN_N = 30
PROMPTS = ('P1', 'P2')  # fixed order: stream 0 = citizen, 1 = P1, 2 = P2
PRIMARY, SECONDARY = 'P1', 'P2'
GROUPS = [f'{english}|{age}' for english in ['native', 'non_native'] for age in ['18-34', '35-54', '55+']]
HEDGE = ['Realistic Possibility', 'Could Happen', 'May Happen', 'Might Happen', 'Probable', 'Likely']
EXTREME = ['Almost No Chance', 'Almost Certain', 'Will Happen']
IT2 = 'llm_it2'          # model label for metrics on the LLM IT2 set
POOLED = 'pooled_llm'    # model label for metrics on all eligible models' answers pooled
CITIZEN = 'citizen'


# ----------------------------------------------------------------------------- LLM answers

def load_answers(path):
    df = pd.read_csv(path, keep_default_na=True)
    key = ['model', 'prompt_id', 'term', 'sample']
    if df.duplicated(key).any():
        raise ValueError('Duplicate model/prompt/term/sample rows')
    value = pd.to_numeric(df.value, errors='raise')
    df['valid'] = value.notna() & value.between(0, 100)
    if (value.notna() & ~df.valid).any():
        raise ValueError('Stored value outside [0, 100]')
    df['value'] = value
    return df


def invalid_table(answers, superseded=None, min_n=MIN_N):
    """Per model x prompt x term: rows, valid, missing, re-asked, token-capped, superseded, included."""
    g = answers.groupby(['model', 'prompt_id', 'term'], sort=True)
    out = pd.DataFrame({
        'n_rows': g.size(),
        'n_valid': g.valid.sum().astype(int),
        'n_reasked': g.attempts.apply(lambda a: int((a > 1).sum())),
        'n_length': g.done_reason.apply(lambda r: int((r == 'length').sum())) if 'done_reason' in answers else 0,
    })
    out['n_missing'] = out.n_rows - out.n_valid
    out['invalid_rate'] = out.n_missing / out.n_rows
    if superseded is not None and len(superseded):
        sup = superseded.groupby(['model', 'prompt_id', 'term']).size()
        out['n_superseded'] = sup.reindex(out.index, fill_value=0).astype(int)
    else:
        out['n_superseded'] = 0
    out['included'] = (out.n_valid >= min_n).astype(int)
    cols = ['n_rows', 'n_valid', 'n_missing', 'invalid_rate', 'n_reasked', 'n_length', 'n_superseded', 'included']
    return out[cols].reset_index()


def eligible_samples(answers, prompt, min_n=MIN_N):
    """{(model, term): sorted valid values} for cells with >= min_n valid answers, plus the exclusions."""
    sub = answers[(answers.prompt_id == prompt) & answers.valid]
    samples, excluded = {}, []
    models = sorted(answers.loc[answers.prompt_id == prompt, 'model'].unique())
    terms = sorted(answers.loc[answers.prompt_id == prompt, 'term'].unique())
    for model in models:
        for term in terms:
            values = sub.loc[(sub.model == model) & (sub.term == term)].sort_values('sample').value.to_numpy(float)
            if len(values) >= min_n:
                samples[(model, term)] = values
            else:
                excluded.append((model, term, len(values)))
    return samples, excluded


def quantiles(values):
    return np.clip(np.quantile(np.asarray(values, float), [.05, .25, .75, .95], method='linear'), 0, 100)


def iqr(values):
    q25, q75 = np.quantile(np.asarray(values, float), [.25, .75], method='linear')
    return float(q75 - q25)


def collapses(samples):
    """Model-terms whose trapezoid collapses to a point (q05 == q95)."""
    rows = []
    for (model, term), values in sorted(samples.items()):
        q = quantiles(values)
        if q[0] == q[3]:
            rows.append(dict(model=model, term=term, value=float(q[0]), n=len(values),
                             share=float(np.mean(values == q[0]))))
    return rows


# ----------------------------------------------------------------------------- citizens

class Citizens:
    """Respondent x term matrix with group codes, for exact refits and whole-respondent resampling."""

    def __init__(self, judgements):
        wide = judgements.pivot(index='response_id', columns='term', values='probability')
        wide = wide.sort_index().sort_index(axis=1)
        if wide.isna().any().any():
            raise ValueError('Citizen bootstrap requires complete respondent vectors')
        meta = judgements.drop_duplicates('response_id').set_index('response_id').reindex(wide.index)
        keys = (meta.english_group + '|' + meta.age_group).to_numpy()
        self.terms = list(wide.columns)
        self.values = wide.to_numpy(float)
        self.group = np.array([GROUPS.index(k) if k in GROUPS else -1 for k in keys])

    def fit(self, idx=None):
        """Pooled T1, six-group IT2, pooled IQR, IT2 centre and pooled T1 centroid per term."""
        values = self.values if idx is None else self.values[idx]
        group = self.group if idx is None else self.group[idx]
        out, n_excluded = {}, 0
        for col, term in enumerate(self.terms):
            column = values[:, col]
            models, counts = group_models({g: column[group == k] for k, g in enumerate(GROUPS)}, MIN_N)
            n_excluded += len(counts) - len(models)
            pooled = quantile_trapezoid(column)
            it2 = envelope(models.values())
            out[term] = dict(pooled=pooled, it2=it2, iqr=iqr(column),
                             centre=centre(it2), centroid=centroid_t1(pooled))
        return out, n_excluded


def read_rq1_codebooks(directory):
    t1 = pd.read_csv(directory / 'citizen_pooled_t1.csv')
    it2 = pd.read_csv(directory / 'citizen_it2.csv')
    pooled = {t: T1Set(g.x.to_numpy(), g.mu.to_numpy()) for t, g in t1.groupby('term', sort=True)}
    bands = {t: IT2Set(g.x.to_numpy(), g.lmf.to_numpy(), g.umf.to_numpy()) for t, g in it2.groupby('term', sort=True)}
    return pooled, bands


def check_against_rq1(fit, pooled, bands):
    """The refit must reproduce the committed RQ1 codebooks exactly (17-digit CSV round trip)."""
    if set(fit) != set(pooled) or set(fit) != set(bands):
        raise ValueError('Terms differ from the RQ1 codebooks')
    for term, f in fit.items():
        if not (np.array_equal(f['pooled'].x, X) and np.array_equal(pooled[term].x, X)):
            raise ValueError('Universe differs from 0..100 step 0.5')
        if not (np.array_equal(f['pooled'].mu, pooled[term].mu) and np.array_equal(f['it2'].umf, bands[term].umf)
                and np.array_equal(f['it2'].lmf, bands[term].lmf)):
            raise ValueError(f'Refit citizen model differs from the RQ1 codebook for {term!r}')


# ----------------------------------------------------------------------------- metrics

def h2a_difference(jaccard_it2):
    """D = mean Jaccard(hedge) - mean Jaccard(extreme), from a {term: Jaccard(citizen IT2, LLM IT2)} map."""
    hedge = float(np.mean([jaccard_it2[t] for t in HEDGE]))
    extreme = float(np.mean([jaccard_it2[t] for t in EXTREME]))
    return hedge, extreme, hedge - extreme


def h2b_share(llm_iqr, citizen_iqr):
    """Share of terms (those with an LLM value) where LLM IQR < citizen IQR (strict)."""
    terms = sorted(llm_iqr)
    if not terms:
        raise ValueError('No terms for H2b')
    return float(np.mean([llm_iqr[t] < citizen_iqr[t] for t in terms]))


def verdict_h2a(ci_hi):
    return 'supported' if ci_hi < 0 else 'not supported'


def verdict_h2b(ci_lo):
    return 'supported' if ci_lo > 0.5 else 'not supported'


def ratio(llm_iqr, citizen_iqr):
    """Width ratio; undefined (NaN) when the citizen IQR is 0."""
    return llm_iqr / citizen_iqr if citizen_iqr > 0 else float('nan')


def compute(citizen, samples):
    """All per-term and summary metrics for one citizen fit and one set of LLM samples.

    Returns ({(metric, term, model): value}, model T1 sets, LLM IT2 sets). Used unchanged for the point
    estimate and inside every bootstrap replicate.
    """
    m = {}
    models = sorted({model for model, _ in samples})
    t1 = {key: quantile_trapezoid(v) for key, v in sorted(samples.items())}
    bands = {}
    for term in sorted(citizen):
        c = citizen[term]
        m[('citizen_iqr', term, CITIZEN)] = c['iqr']
        m[('citizen_centre_it2', term, CITIZEN)] = c['centre']
        present = [model for model in models if (model, term) in t1]
        if not present:
            raise ValueError(f'No eligible model for {term!r}')
        bands[term] = envelope(t1[(model, term)] for model in present)
        llm_centre = centre(bands[term])
        m[('jaccard_it2', term, IT2)] = jaccard(c['it2'], bands[term])
        m[('llm_centre_it2', term, IT2)] = llm_centre
        m[('centroid_shift_it2', term, IT2)] = llm_centre - c['centre']
        for model in present:
            a, values = t1[(model, term)], samples[(model, term)]
            m[('jaccard_t1', term, model)] = jaccard(c['pooled'], a)
            m[('centroid_shift_t1', term, model)] = centroid_t1(a) - c['centroid']
            m[('llm_iqr', term, model)] = iqr(values)
            m[('width_ratio', term, model)] = ratio(iqr(values), c['iqr'])
        pooled_values = np.concatenate([samples[(model, term)] for model in present])
        m[('llm_iqr', term, POOLED)] = iqr(pooled_values)
        m[('width_ratio', term, POOLED)] = ratio(iqr(pooled_values), c['iqr'])
    jac = {t: m[('jaccard_it2', t, IT2)] for t in citizen}
    hedge, extreme, d = h2a_difference(jac)
    m[('mean_jaccard_hedge', 'hedge', IT2)] = hedge
    m[('mean_jaccard_extreme', 'extreme', IT2)] = extreme
    m[('h2a_D', 'hedge_minus_extreme', IT2)] = d
    cit_iqr = {t: citizen[t]['iqr'] for t in citizen}
    for model in models + [POOLED]:
        llm = {t: m[('llm_iqr', t, model)] for t in citizen if ('llm_iqr', t, model) in m}
        m[('h2b_share', 'all', model)] = h2b_share(llm, cit_iqr)
        m[('h2b_n_terms', 'all', model)] = len(llm)
    return m, t1, bands


def make_streams(seed=SEED):
    """Independent generators: citizen, then one per prompt in PROMPTS order."""
    children = np.random.SeedSequence(seed).spawn(1 + len(PROMPTS))
    return np.random.default_rng(children[0]), {p: np.random.default_rng(c) for p, c in zip(PROMPTS, children[1:])}


def resample_llm(samples, rng):
    """Independent with-replacement resample of each model-term's answers, in sorted key order."""
    return {key: values[rng.integers(0, len(values), size=len(values))] for key, values in sorted(samples.items())}


def bootstrap(citizens, samples_by_prompt, b=B, seed=SEED):
    """{prompt: {key: np.array of b replicate values}} and the citizen group exclusions over all replicates."""
    citizen_rng, llm_rngs = make_streams(seed)
    n = len(citizens.values)
    reps = {p: {} for p in samples_by_prompt}
    group_excluded = 0
    for i in range(b):
        idx = citizen_rng.integers(0, n, size=n)
        fit, excluded = citizens.fit(idx)
        group_excluded += excluded
        for prompt in PROMPTS:
            if prompt not in samples_by_prompt:
                continue
            m, _, _ = compute(fit, resample_llm(samples_by_prompt[prompt], llm_rngs[prompt]))
            for key, value in m.items():
                reps[prompt].setdefault(key, np.full(b, np.nan))[i] = value
    return reps, group_excluded


def metric_frame(point, reps):
    rows = []
    for key, estimate in point.items():
        if key[0] == 'h2b_n_terms':
            continue
        values = reps[key]
        if np.isnan(estimate):  # undefined metric (width ratio with citizen IQR 0): no CI
            lo = hi = float('nan')
        else:
            if not np.isfinite(values).all():
                raise ValueError(f'Undefined bootstrap value for {key}')
            lo, hi = np.percentile(values, [2.5, 97.5], method='linear')
        rows.append(dict(metric=key[0], term=key[1], model=key[2], estimate=estimate, ci_lo=lo, ci_hi=hi))
    return pd.DataFrame(rows, columns=['metric', 'term', 'model', 'estimate', 'ci_lo', 'ci_hi'])


# ----------------------------------------------------------------------------- outputs

def write_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format='%.17g', lineterminator='\n')


def codebook_frames(prompt, t1, bands, samples):
    long = pd.DataFrame([dict(prompt=prompt, model=m, term=t, x=x, mu=mu)
                         for (m, t), a in sorted(t1.items()) for x, mu in zip(a.x, a.mu)])
    params = pd.DataFrame([dict(prompt=prompt, model=m, term=t, n_valid=len(samples[(m, t)]),
                                q05=q[0], q25=q[1], q75=q[2], q95=q[3])
                           for (m, t) in sorted(t1) for q in [quantiles(samples[(m, t)])]])
    n_models = {t: sum(1 for (_, tt) in t1 if tt == t) for t in bands}
    it2 = pd.DataFrame([dict(prompt=prompt, term=t, n_models=n_models[t], x=x, umf=u, lmf=l)
                        for t, a in sorted(bands.items()) for x, u, l in zip(a.x, a.umf, a.lmf)])
    return long, params, it2


def figures(citizen, bands, frame, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b'})
    blue, orange, grey, ink = '#2a78d6', '#eb6834', '#8a8984', '#52514e'
    order = sorted(citizen, key=lambda t: (citizen[t]['centre'], t))

    fig, axes = plt.subplots(5, 4, figsize=(12, 12), sharex=True, sharey=True)
    for ax, term in zip(axes.flat, order):
        c, l = citizen[term]['it2'], bands[term]
        ax.fill_between(X, c.lmf, c.umf, color=blue, alpha=.35, lw=0, label='Citizen IT2 (FOU)')
        ax.plot(X, c.umf, color=blue, lw=1)
        ax.fill_between(X, l.lmf, l.umf, color=orange, alpha=.35, lw=0, label='LLM IT2 (FOU, P1)')
        ax.plot(X, l.umf, color=orange, lw=1)
        ax.set_title(term, color='#0b0b0b'); ax.set_xlim(0, 100); ax.set_ylim(0, 1.05)
    for ax in list(axes.flat)[len(order):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right', frameon=False)
    fig.supxlabel('Probability (%)'); fig.supylabel('Membership')
    fig.tight_layout(); fig.savefig(out / 'bands_small_multiples.png', dpi=150, metadata={'Software': None})
    plt.close(fig)

    def per_term(metric, model):
        return frame[(frame.metric == metric) & (frame.model == model)].set_index('term')

    jac = per_term('jaccard_it2', IT2)
    ranked = sorted(jac.index, key=lambda t: (jac.loc[t, 'estimate'], t))
    kind = {t: 'hedge' if t in HEDGE else 'extreme' if t in EXTREME else 'other' for t in ranked}
    colour = {'hedge': orange, 'extreme': blue, 'other': grey}
    fig, ax = plt.subplots(figsize=(7, 7))
    for y, t in enumerate(ranked):
        r = jac.loc[t]
        ax.plot([r.ci_lo, r.ci_hi], [y, y], color=colour[kind[t]], lw=2, solid_capstyle='round')
        ax.plot(r.estimate, y, 'o', ms=7, color=colour[kind[t]], mec='white', mew=1.5)
    for k, lab in [('hedge', 'Hedge term (H2a)'), ('extreme', 'Extreme term (H2a)'), ('other', 'Other term')]:
        ax.plot([], [], 'o', color=colour[k], label=lab)
    ax.set_yticks(range(len(ranked)), ranked); ax.set_xlim(0, 1)
    ax.set_xlabel('Jaccard(citizen IT2, LLM IT2), P1, with 95% bootstrap CI')
    ax.grid(axis='x', color='#e4e3df', lw=.6); ax.set_axisbelow(True)
    ax.legend(loc='lower right', frameon=False)
    fig.tight_layout(); fig.savefig(out / 'jaccard_dotplot.png', dpi=150, metadata={'Software': None})
    plt.close(fig)

    shift = per_term('centroid_shift_it2', IT2)
    t1 = frame[frame.metric == 'centroid_shift_t1']
    models = sorted(t1.model.unique())
    markers = dict(zip(models, ['s', '^', 'D', 'v', 'P', 'X']))
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.axvline(0, color=ink, lw=.8)
    for y, t in enumerate(order):
        r = shift.loc[t]
        ax.plot([r.ci_lo, r.ci_hi], [y, y], color=orange, lw=2, solid_capstyle='round')
        ax.plot(r.estimate, y, 'o', ms=8, color=orange, mec='white', mew=1.5)
        for model in models:
            row = t1[(t1.term == t) & (t1.model == model)]
            if len(row):
                ax.plot(row.estimate.iloc[0], y + .25, markers[model], ms=4, color=grey, mfc='none', mew=.8)
    ax.plot([], [], 'o', color=orange, label='LLM IT2 centre - citizen IT2 centre (95% CI)')
    for model in models:
        ax.plot([], [], markers[model], color=grey, mfc='none', label=f'{model} T1 - pooled citizen T1')
    ax.set_yticks(range(len(order)), order)
    ax.set_xlabel('Centroid shift (percentage points), P1; terms ordered by citizen centre')
    ax.grid(axis='x', color='#e4e3df', lw=.6); ax.set_axisbelow(True)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(out / 'centroid_shift.png', dpi=150, metadata={'Software': None})
    plt.close(fig)


def fmt(r, digits=3):
    if np.isnan(r.estimate):
        return 'undefined'
    return f'{r.estimate:.{digits}f} [{r.ci_lo:.{digits}f}, {r.ci_hi:.{digits}f}]'


def summary(answers, invalid, excluded, collapsed, frames, citizens_n, group_excluded, b, versions):
    p1, p2 = frames[PRIMARY], frames.get(SECONDARY)

    def get(frame, metric, term, model):
        return frame[(frame.metric == metric) & (frame.term == term) & (frame.model == model)].iloc[0]

    lines = ['# RQ2: Citizen vs LLM probability-word models', '',
             'Generated by `experiments/words/run_rq2.py` (docs/ANALYSIS_PLAN_WORDS.md, RQ2). Numbers only.', '',
             f'Citizens: CAPphrase, {citizens_n:,} respondents, RQ1 codebooks (refit from raw data and checked identical).',
             f'LLMs: {answers.model.nunique()} models x {answers.prompt_id.nunique()} prompts x '
             f'{answers.term.nunique()} terms x 50 samples = {len(answers):,} rows. Primary prompt P1; P2 (persona) is sensitivity.',
             f'Bootstrap: B = {b}, seed {SEED}; citizen respondents (whole 19-term vectors) and each model x term '
             'sample resampled independently; percentile 95% CIs.', '',
             '| Model | Digest (first 12) |', '|---|---|']
    for model, digest in answers.groupby('model').digest.first().items():
        lines.append(f'| {model} | {digest[:12]} |')

    lines += ['', '## Valid answers', '', '| Model | Prompt | Rows | Valid | Missing | Invalid rate | Re-asked | Token cap (`length`) | Superseded (D19) |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    agg = invalid.groupby(['model', 'prompt_id'])[['n_rows', 'n_valid', 'n_missing', 'n_reasked', 'n_length', 'n_superseded']].sum()
    for (model, prompt), r in agg.iterrows():
        lines.append(f'| {model} | {prompt} | {r.n_rows} | {r.n_valid} | {r.n_missing} | {r.n_missing / r.n_rows:.4f} | '
                     f'{r.n_reasked} | {r.n_length} | {r.n_superseded} |')
    lines += ['', 'Per model x prompt x term counts: `invalid.csv`. Re-asked counts answers that needed more than one attempt '
              '(valid or not); superseded rows are pre-D19 answers moved to superseded.csv and re-elicited.', '',
              f'Excluded model-terms (< {MIN_N} valid answers):']
    any_excl = False
    for prompt in PROMPTS:
        for model, term, n in excluded.get(prompt, []):
            lines.append(f'- {prompt}: {model} / {term} (n valid = {n})'); any_excl = True
    if not any_excl:
        lines.append('- none')

    lines += ['', '## Per-term results (P1)', '',
              '| Term | Jaccard(citizen IT2, LLM IT2) | Centroid shift IT2 (pp) | Width ratio, pooled LLM | Models |',
              '|---|---|---|---|---:|']
    n_models = p1[p1.metric == 'jaccard_t1'].groupby('term').size()
    terms = sorted(p1[p1.metric == 'jaccard_it2'].term)
    for t in terms:
        lines.append(f'| {t} | {fmt(get(p1, "jaccard_it2", t, IT2))} | {fmt(get(p1, "centroid_shift_it2", t, IT2), 2)} | '
                     f'{fmt(get(p1, "width_ratio", t, POOLED))} | {n_models[t]} |')
    lines += ['', '| Model (P1) | Mean Jaccard(pooled citizen T1, model T1) | Mean centroid shift T1 (pp) | Median width ratio |',
              '|---|---:|---:|---:|']
    for model in sorted(p1[p1.metric == 'jaccard_t1'].model.unique()):
        sel = p1[p1.model == model]
        lines.append(f'| {model} | {sel[sel.metric == "jaccard_t1"].estimate.mean():.3f} | '
                     f'{sel[sel.metric == "centroid_shift_t1"].estimate.mean():.2f} | '
                     f'{sel[sel.metric == "width_ratio"].estimate.median():.3f} |')
    lines.append('')
    lines.append('All per-term metrics with CIs (including per-model Jaccard, centroid shift and width ratio): `metrics.csv`.')

    lines += ['', '## Hypotheses', '']
    for prompt, frame in [(PRIMARY, p1), (SECONDARY, p2)]:
        if frame is None:
            continue
        d = get(frame, 'h2a_D', 'hedge_minus_extreme', IT2)
        h = get(frame, 'mean_jaccard_hedge', 'hedge', IT2); e = get(frame, 'mean_jaccard_extreme', 'extreme', IT2)
        tag = 'primary' if prompt == PRIMARY else 'sensitivity'
        lines += [f'### {prompt} ({tag})', '',
                  f'H2a: mean Jaccard hedge = {fmt(h)}; extreme = {fmt(e)}; D = {fmt(d)}. '
                  f'Verdict: **{verdict_h2a(d.ci_hi)}** (rule: CI(D) entirely below 0)'
                  + ('; the CI lies entirely above 0, i.e. the opposite direction.' if d.ci_lo > 0 else '.'), '',
                  '| H2b: share of terms with LLM IQR < citizen IQR | Terms | Share [95% CI] | Verdict |', '|---|---:|---|---|']
        counts = {r.model: int(r.estimate) for r in frames[f'{prompt}_nterms'].itertuples()}
        shares = frame[frame.metric == 'h2b_share']
        for r in shares.itertuples():
            lines.append(f'| {r.model} | {counts[r.model]} | {fmt(r)} | {verdict_h2b(r.ci_lo)} |')
        lines += ['', 'H2b rule: CI of the share entirely above 0.5.', '']

    lines += ['## Collapsed trapezoids (descriptive, no hypothesis)', '',
              'Model-terms where q05 == q95: the model gave the same number in at least 90% of its valid answers '
              '("share" = fraction exactly at that value); the T1 set is a single point.', '']
    for prompt in PROMPTS:
        rows = collapsed.get(prompt, [])
        lines.append(f'{prompt}: {len(rows)} of {len(frames[prompt + "_cells"])} eligible model-terms.')
        lines.append('')
        if rows:
            lines += ['| Model | Term | Value | n valid | Share at value |', '|---|---|---:|---:|---:|']
            lines += [f'| {r["model"]} | {r["term"]} | {r["value"]:g} | {r["n"]} | {r["share"]:.2f} |' for r in rows]
            lines.append('')
    by_model = {p: pd.Series([r['model'] for r in collapsed.get(p, [])]).value_counts().to_dict() for p in PROMPTS}
    lines += ['Per model (P1 / P2): ' + '; '.join(f'{m} {by_model[PRIMARY].get(m, 0)} / {by_model[SECONDARY].get(m, 0)}'
                                                  for m in sorted(answers.model.unique())) + '.', '']

    zero_iqr = sorted(p1[(p1.metric == 'citizen_iqr') & (p1.estimate == 0)].term)
    lines += ['## Deviations', '',
              '- None from the frozen RQ2 analysis plan in the quantities estimated. Implementation choices '
              '(all set before the real-data run except the zero-IQR rule below):',
              '- Valid answer = stored value present and in [0, 100] (strict parsing, D19); no further cleaning.',
              f'- Model-terms with < {MIN_N} valid answers are excluded (WP4 brief). For such a term the LLM IT2 is the envelope of the '
              'remaining models, the pooled-LLM IQR pools only eligible models, and that model\'s H2b share is taken over its eligible '
              'terms (denominator shown in the H2b table).',
              '- Citizen resampling draws whole respondents, as in the RQ1 bootstrap, so cross-term dependence is kept; the six-group '
              f'IT2 is re-derived in every replicate with the n >= {MIN_N} rule (group-term cells dropped over all replicates: {group_excluded}).',
              '- Independent streams: SeedSequence(seed).spawn(3) gives citizen, P1 and P2 generators; P1 and P2 share the citizen replicates.',
              '- H2a uses the IT2 Jaccard of the plan (citizen IT2 vs LLM IT2); H2b uses a strict inequality (equal IQRs count as not narrower).',
              '- Verdicts: "supported" only if the CI rule in the brief holds; otherwise "not supported" (exploratory, no multiplicity correction).',
              '- Centroid shift: IT2 uses EKM centres (c_l + c_r) / 2; per model it is the T1 centroid minus the pooled citizen T1 centroid. '
              'IQR uses numpy linear quantiles of the raw values; width ratio = LLM IQR / citizen pooled IQR.',
              '- Width ratio is undefined (NaN, no CI) for a term whose citizen IQR is 0'
              + (f' ({", ".join(zero_iqr)})' if zero_iqr else '') + '; H2b compares IQRs directly, so such a term '
              'is never counted as narrower and stays in the denominator. This rule was added after the first run on the real '
              'data stopped on a division by zero; it does not change any other number.',
              '- A collapsed model set is a one-point spike on the 0.5 grid; it still enters the IT2 envelope, Jaccard and centroids. '
              'Its IQR is 0, so it counts as narrower in H2b unless the citizen IQR is also 0.',
              '- Percentile CIs of discrete statistics (shares) and of collapsed sets can be degenerate (lo = hi).', '',
              f'Software: numpy {versions["numpy"]}, pandas {versions["pandas"]}, matplotlib {versions["matplotlib"]}. CSV floats: 17 significant digits.', '']
    return '\n'.join(lines)


def run(answers_path, superseded_path, capphrase_root, rq1_codebooks, out, b=B, seed=SEED, make_figures=True):
    answers = load_answers(answers_path)
    superseded = pd.read_csv(superseded_path) if superseded_path and Path(superseded_path).exists() else None
    if set(answers.prompt_id) - set(PROMPTS):
        raise ValueError('Unknown prompt id')
    invalid = invalid_table(answers, superseded)

    judgements = capphrase.load_judgements(capphrase_root)
    citizens = Citizens(judgements)
    if sorted(answers.term.unique()) != citizens.terms:
        raise ValueError('LLM terms differ from CAPphrase terms')
    point_citizen, _ = citizens.fit()
    check_against_rq1(point_citizen, *read_rq1_codebooks(Path(rq1_codebooks)))

    samples, excluded, collapsed, point, books = {}, {}, {}, {}, {}
    for prompt in PROMPTS:
        if prompt in set(answers.prompt_id):
            samples[prompt], excluded[prompt] = eligible_samples(answers, prompt)
            collapsed[prompt] = collapses(samples[prompt])
            point[prompt], t1, bands = compute(point_citizen, samples[prompt])
            books[prompt] = (t1, bands)
    reps, group_excluded = bootstrap(citizens, samples, b, seed)

    out.mkdir(parents=True, exist_ok=True)
    frames = {}
    for prompt in samples:
        frames[prompt] = metric_frame(point[prompt], reps[prompt])
        frames[f'{prompt}_nterms'] = pd.DataFrame([dict(model=k[2], estimate=v) for k, v in point[prompt].items()
                                                   if k[0] == 'h2b_n_terms'])
        frames[f'{prompt}_cells'] = list(samples[prompt])
    write_csv(frames[PRIMARY], out / 'metrics.csv')
    if SECONDARY in frames:
        write_csv(frames[SECONDARY], out / 'sensitivity.csv')
    write_csv(invalid, out / 'invalid.csv')
    parts = [codebook_frames(p, *books[p], samples[p]) for p in samples]
    write_csv(pd.concat([p[0] for p in parts], ignore_index=True), out / 'codebooks/llm_model_t1.csv')
    write_csv(pd.concat([p[1] for p in parts], ignore_index=True), out / 'codebooks/llm_model_params.csv')
    write_csv(pd.concat([p[2] for p in parts], ignore_index=True), out / 'codebooks/llm_it2.csv')
    import matplotlib
    versions = dict(numpy=np.__version__, pandas=pd.__version__, matplotlib=matplotlib.__version__)
    if make_figures:
        figdir = out / 'figures'; figdir.mkdir(exist_ok=True)
        figures(point_citizen, books[PRIMARY][1], frames[PRIMARY], figdir)
    text = summary(answers, invalid, excluded, collapsed, frames, len(citizens.values), group_excluded, b, versions)
    (out / 'SUMMARY.md').write_text(text, encoding='utf-8')
    return frames, excluded, collapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--answers', type=Path, default=ROOT / 'data/llm_answers/answers.csv')
    parser.add_argument('--superseded', type=Path, default=ROOT / 'data/llm_answers/superseded.csv')
    parser.add_argument('--capphrase-root', type=Path, default=None, help='default: data/raw/capphrase')
    parser.add_argument('--rq1-codebooks', type=Path, default=ROOT / 'results/words/rq1/codebooks')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/words/rq2')
    args = parser.parse_args()
    frames, excluded, collapsed = run(args.answers, args.superseded, args.capphrase_root, args.rq1_codebooks, args.output)
    print(f'Wrote {args.output}; excluded={ {p: len(v) for p, v in excluded.items()} }; '
          f'collapsed={ {p: len(v) for p, v in collapsed.items()} }')


if __name__ == '__main__':
    main()
