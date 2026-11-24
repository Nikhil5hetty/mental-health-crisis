"""Logic of the word-meaning lab (app/word_lab.py). Pure functions, no Streamlit; unit-tested.

Reads ONLY the committed aggregate results under results/words/ (no raw data): fast and licence-safe.

Page 1 "What does a word mean?"  -> load_word_data, term_curves, term_summary, ranking
Page 2 "Crisis-line simulator"   -> load_simulator, simulate (the RQ3 rule from src/cww/verbalise.py)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.cww import APPROPRIATENESS_WORDS
from src.cww.rules import ACTIONS
from src.cww.verbalise import CITIZEN, LEVELS, LLM, LLM_PROMPT, SEVERITY, action, load_codebooks, memberships, verbalise
from src.cww.centroid import centre

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / 'results/words'
ENV_RESULTS = 'WORD_LAB_RESULTS'   # optional override of the results folder (used by the tests)

# Term groups of hypothesis H2a (docs/ANALYSIS_PLAN_WORDS.md); a test checks they equal run_rq2.py's.
HEDGE = ['Realistic Possibility', 'Could Happen', 'May Happen', 'Might Happen', 'Probable', 'Likely']
EXTREME = ['Almost No Chance', 'Almost Certain', 'Will Happen']

GROUP_LABELS = {
    'native|18-34': 'Native English speakers, 18-34', 'native|35-54': 'Native English speakers, 35-54',
    'native|55+': 'Native English speakers, 55+', 'non_native|18-34': 'Non-native speakers, 18-34',
    'non_native|35-54': 'Non-native speakers, 35-54', 'non_native|55+': 'Non-native speakers, 55+',
}
CODEBOOK_LABELS = {CITIZEN: 'Citizens (all 6 groups)', LLM: 'LLMs (4 models)', **GROUP_LABELS}
LEVEL_LABELS = {'None': 'No crisis', 'Possible': 'Possible crisis', 'Likely': 'Likely crisis', 'Clear': 'Clear crisis'}
ACTION_LABELS = [
    'A0 - Continue the conversation',
    'A1 - Suggest a helpline (143; 147 for young people)',
    'A2 - Hand over to a human counsellor',
    'A3 - Urgent: human now + emergency numbers (144 / 112)',
]
ACTION_SHORT = ['A0 Continue', 'A1 Suggest helpline', 'A2 Human counsellor', 'A3 Urgent: human + 144/112']
REPLY_HELP = {
    'Harmful/Unacceptable': "The bot's reply could do harm.",
    'Inappropriate': "The bot's reply misses the point or is unsuitable.",
    'Partially Appropriate': "The bot's reply is partly helpful.",
    'Mostly Appropriate': "The bot's reply is helpful with small gaps.",
    'Fully Appropriate': "The bot's reply is fully appropriate.",
}

PAGE1_FILES = {
    'rq1/codebooks/citizen_it2.csv': 'experiments/words/run_rq1.py',
    'rq1/codebooks/group_t1.csv': 'experiments/words/run_rq1.py',
    'rq2/codebooks/llm_model_t1.csv': 'experiments/words/run_rq2.py',
    'rq2/codebooks/llm_model_params.csv': 'experiments/words/run_rq2.py',
    'rq2/codebooks/llm_it2.csv': 'experiments/words/run_rq2.py',
    'rq2/metrics.csv': 'experiments/words/run_rq2.py',
    'rq2/invalid.csv': 'experiments/words/run_rq2.py',
}
PAGE2_FILES = {
    'rq1/codebooks/citizen_it2.csv': 'experiments/words/run_rq1.py',
    'rq1/codebooks/group_t1.csv': 'experiments/words/run_rq1.py',
    'rq2/codebooks/llm_it2.csv': 'experiments/words/run_rq2.py',
}
RQ3_FILES = ['rq3/metrics.csv', 'rq3/per_item.csv']


class MissingResults(FileNotFoundError):
    """Required result files are absent; the message says which and how to produce them."""


def results_dir() -> Path:
    return Path(os.environ.get(ENV_RESULTS, DEFAULT_RESULTS))


def missing_files(results: Path, required: dict[str, str]) -> list[tuple[str, str]]:
    return [(rel, script) for rel, script in required.items() if not (Path(results) / rel).is_file()]


def require(results: Path, required: dict[str, str]) -> None:
    missing = missing_files(results, required)
    if missing:
        lines = [f'- {rel} (made by `python -I {script}`)' for rel, script in missing]
        raise MissingResults(f'Missing result files in {results}:\n' + '\n'.join(lines))


# ----------------------------------------------------------------------------- page 1

@dataclass(frozen=True)
class WordData:
    terms: list
    citizen: pd.DataFrame      # term, x, umf, lmf
    groups: pd.DataFrame       # term, group, x, mu
    model_t1: pd.DataFrame     # model, term, x, mu  (prompt P1)
    llm_it2: pd.DataFrame      # term, n_models, x, umf, lmf  (prompt P1)
    params: pd.DataFrame       # model, term, n_valid, q05, q25, q75, q95  (prompt P1)
    metrics: pd.DataFrame      # metric, term, model, estimate, ci_lo, ci_hi  (P1)
    invalid: pd.DataFrame      # model, term, n_rows, n_valid, included  (P1)


def _p1(frame):
    return frame[frame.prompt == LLM_PROMPT].drop(columns='prompt') if 'prompt' in frame else frame


def load_word_data(results: Path | None = None) -> WordData:
    results = Path(results or results_dir())
    require(results, PAGE1_FILES)
    read = lambda rel: pd.read_csv(results / rel)
    invalid = read('rq2/invalid.csv')
    invalid = invalid[invalid.prompt_id == LLM_PROMPT].drop(columns='prompt_id')
    citizen = read('rq1/codebooks/citizen_it2.csv')
    return WordData(terms=sorted(citizen.term.unique()), citizen=citizen, groups=read('rq1/codebooks/group_t1.csv'),
                    model_t1=_p1(read('rq2/codebooks/llm_model_t1.csv')), llm_it2=_p1(read('rq2/codebooks/llm_it2.csv')),
                    params=_p1(read('rq2/codebooks/llm_model_params.csv')), metrics=read('rq2/metrics.csv'),
                    invalid=invalid)


def _check_term(data: WordData, term: str) -> None:
    if term not in data.terms:
        raise ValueError(f'Unknown term {term!r}')


def term_curves(data: WordData, term: str, show_groups: bool = False, show_models: bool = True):
    """(bands, lines) for plotting one term on 0..100.

    bands: x, series, lo, hi  -- citizen IT2 band and LLM IT2 band (membership between LMF and UMF)
    lines: x, series, kind, mu -- each model's T1 set and, optionally, each citizen group's T1 set
    """
    _check_term(data, term)
    c = data.citizen[data.citizen.term == term]
    l = data.llm_it2[data.llm_it2.term == term]
    bands = pd.concat([
        pd.DataFrame({'x': c.x, 'series': 'Citizens (band over 6 groups)', 'lo': c.lmf, 'hi': c.umf}),
        pd.DataFrame({'x': l.x, 'series': 'LLMs (band over models)', 'lo': l.lmf, 'hi': l.umf}),
    ], ignore_index=True)
    parts = []
    if show_models:
        m = data.model_t1[data.model_t1.term == term]
        parts.append(pd.DataFrame({'x': m.x, 'series': m.model, 'kind': 'model', 'mu': m.mu}))
    if show_groups:
        g = data.groups[data.groups.term == term]
        parts.append(pd.DataFrame({'x': g.x, 'series': g.group.map(GROUP_LABELS), 'kind': 'citizen group', 'mu': g.mu}))
    lines = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=['x', 'series', 'kind', 'mu'])
    return bands, lines


def _metric(data: WordData, metric: str, term: str, model: str):
    row = data.metrics[(data.metrics.metric == metric) & (data.metrics.term == term) & (data.metrics.model == model)]
    if row.empty:
        return None
    r = row.iloc[0]
    return dict(estimate=float(r.estimate), ci_lo=float(r.ci_lo), ci_hi=float(r.ci_hi))


def term_summary(data: WordData, term: str) -> dict:
    """Headline numbers for one term (all read from results/words/rq2/metrics.csv and the codebooks)."""
    _check_term(data, term)
    rows = []
    for r in data.invalid[data.invalid.term == term].sort_values('model').itertuples():
        p = data.params[(data.params.term == term) & (data.params.model == r.model)]
        row = dict(model=r.model, included=bool(r.included), n_valid=int(r.n_valid), n_asked=int(r.n_rows))
        if r.included and not p.empty:
            q = p.iloc[0]
            row.update(q05=float(q.q05), q25=float(q.q25), q75=float(q.q75), q95=float(q.q95),
                       collapsed=bool(q.q05 == q.q95))
            for key, metric in [('jaccard', 'jaccard_t1'), ('centre_shift', 'centroid_shift_t1'), ('width_ratio', 'width_ratio')]:
                v = _metric(data, metric, term, r.model)
                row[key] = None if v is None else v['estimate']
        else:
            row.update(q05=None, q25=None, q75=None, q95=None, collapsed=False, jaccard=None, centre_shift=None,
                       width_ratio=None)
        rows.append(row)
    models = pd.DataFrame(rows)
    return dict(
        term=term,
        category='hedge' if term in HEDGE else 'extreme' if term in EXTREME else 'other',
        jaccard=_metric(data, 'jaccard_it2', term, 'llm_it2'),
        centre_shift=_metric(data, 'centroid_shift_it2', term, 'llm_it2'),
        citizen_centre=_metric(data, 'citizen_centre_it2', term, 'citizen'),
        llm_centre=_metric(data, 'llm_centre_it2', term, 'llm_it2'),
        models=models,
        collapsed_models=[m for m, c in zip(models.model, models.collapsed) if c],
        excluded_models=[m for m, i in zip(models.model, models.included) if not i],
    )


def ranking(data: WordData) -> pd.DataFrame:
    """All terms ranked by Jaccard(citizen IT2, LLM IT2), highest overlap first (ties: alphabetical)."""
    j = data.metrics[(data.metrics.metric == 'jaccard_it2') & (data.metrics.model == 'llm_it2')]
    s = data.metrics[(data.metrics.metric == 'centroid_shift_it2') & (data.metrics.model == 'llm_it2')]
    out = j[['term', 'estimate', 'ci_lo', 'ci_hi']].rename(columns={'estimate': 'jaccard'}).merge(
        s[['term', 'estimate']].rename(columns={'estimate': 'centre_shift'}), on='term', how='left')
    out['category'] = ['hedge' if t in HEDGE else 'extreme' if t in EXTREME else 'other' for t in out.term]
    out = out.sort_values(['jaccard', 'term'], ascending=[False, True]).reset_index(drop=True)
    out.insert(0, 'rank', np.arange(1, len(out) + 1))
    return out


def overlap_words(j: float) -> str:
    """Plain-language reading of a Jaccard overlap (descriptive bands, not a statistical test)."""
    if j >= .6:
        return 'large overlap'
    if j >= .4:
        return 'moderate overlap'
    if j >= .2:
        return 'small overlap'
    return 'almost no overlap'


def shift_words(shift: float) -> str:
    if abs(shift) < 1:
        return 'LLMs and citizens put the centre of this word at about the same place'
    side = 'higher' if shift > 0 else 'lower'
    return f'LLMs put the centre of this word {abs(shift):.1f} percentage points {side} than citizens'


# ----------------------------------------------------------------------------- page 2

@dataclass(frozen=True)
class Simulator:
    books: dict          # codebook key -> {level: IT2Set}
    centres: dict        # codebook key -> {level: EKM centre}
    rq3: dict | None     # headline study numbers, or None when results/words/rq3 is missing


def load_simulator(results: Path | None = None) -> Simulator:
    """Codebooks for the simulator. Needs only RQ1/RQ2 codebooks; RQ3 study numbers are optional."""
    results = Path(results or results_dir())
    require(results, PAGE2_FILES)
    books = load_codebooks(results / 'rq1/codebooks', results / 'rq2/codebooks', LLM_PROMPT)
    centres = {k: {l: centre(b[l]) for l in SEVERITY} for k, b in books.items()}
    rq3 = None
    if all((results / f).is_file() for f in RQ3_FILES):
        m = pd.read_csv(results / 'rq3/metrics.csv')
        row = m[(m.metric == 'action_flip_rate') & (m.comparison == f'llm_vs_{CITIZEN}')]
        n_items = len(pd.read_csv(results / 'rq3/per_item.csv'))
        if not row.empty:
            r = row.iloc[0]
            rq3 = dict(flip_rate=float(r.estimate), ci_lo=float(r.ci_lo), ci_hi=float(r.ci_hi), n_items=n_items,
                       n_flips=int(round(float(r.estimate) * n_items)))
    return Simulator(books, centres, rq3)


def simulate(sim: Simulator, m: float, reply_word: str) -> pd.DataFrame:
    """One row per codebook: chosen level word, action and whether the decision differs from the citizens'."""
    m = float(m)
    if not np.isfinite(m) or not 0 <= m <= 100:
        raise ValueError('Crisis degree must lie between 0 and 100')
    if reply_word not in APPROPRIATENESS_WORDS:
        raise ValueError(f'Unknown reply word {reply_word!r}')
    rows = []
    for key, book in sim.books.items():
        level, fallback = verbalise(m, book, sim.centres[key])
        a = action(level, reply_word)
        rows.append(dict(codebook=key, label=CODEBOOK_LABELS.get(key, key), level=level, level_label=LEVEL_LABELS[level],
                         word=LEVELS[level], membership=memberships(m, book)[level], fallback=fallback,
                         action=a, action_label=ACTION_LABELS[a], action_short=ACTION_SHORT[a]))
    out = pd.DataFrame(rows)
    ref = int(out.loc[out.codebook == CITIZEN, 'action'].iloc[0])
    out['flips'] = out.action != ref
    out['direction'] = np.where(out.action < ref, 'less urgent', np.where(out.action > ref, 'more urgent', 'same'))
    order = [CITIZEN, LLM] + sorted(k for k in sim.books if k not in (CITIZEN, LLM))
    return out.set_index('codebook').loc[order].reset_index()


def level_curves(sim: Simulator, keys=(CITIZEN, LLM)) -> pd.DataFrame:
    """Membership (UMF + LMF) / 2 of the four level words for the chosen codebooks, for plotting."""
    rows = []
    for key in keys:
        for level in SEVERITY:
            s = sim.books[key][level]
            rows.append(pd.DataFrame({'x': s.x, 'mu': (s.umf + s.lmf) / 2, 'codebook': CODEBOOK_LABELS[key],
                                      'level': f'{LEVEL_LABELS[level]} ("{LEVELS[level]}")'}))
    return pd.concat(rows, ignore_index=True)


def headline(table: pd.DataFrame) -> dict:
    """Does the LLM codebook flip the citizens' decision? Plus how many group codebooks differ."""
    cit = table[table.codebook == CITIZEN].iloc[0]
    llm = table[table.codebook == LLM].iloc[0]
    groups = table[~table.codebook.isin([CITIZEN, LLM])]
    return dict(llm_flips=bool(llm.flips), direction=llm.direction, citizen_action=int(cit.action),
                llm_action=int(llm.action), n_groups_differ=int(groups.flips.sum()), n_groups=len(groups),
                any_fallback=bool(table.fallback.any()))
