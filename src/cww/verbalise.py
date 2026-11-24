"""Verbalise a crisis degree with a probability-word codebook (RQ3 of docs/ANALYSIS_PLAN_WORDS.md).

Shared by experiments/words/run_rq3.py and the word-meaning lab app (src/app/logic.py), so both apply the same
rule. Four crisis-level words (None = "Remote Chance", Possible = "Realistic Possibility", Likely = "Likely",
Clear = "Almost Certain"). Membership of m (0..100) in a word = (UMF + LMF) / 2, linearly interpolated on the
codebook grid; the largest membership wins, ties go to the more severe word. If all four memberships are 0,
the word whose EKM centroid centre is nearest m is chosen (ties: more severe) and flagged as a fallback.
Action = src/cww/rules.py table with crisis word and reply word at membership 1.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import APPROPRIATENESS_WORDS
from .centroid import centre
from .rules import CRISIS_WORDS, RULE_TABLE, crisp_memberships, infer
from .sets import IT2Set

LEVELS = {'None': 'Remote Chance', 'Possible': 'Realistic Possibility', 'Likely': 'Likely', 'Clear': 'Almost Certain'}
SEVERITY = list(LEVELS)  # increasing severity, same order as rules.CRISIS_WORDS
CITIZEN, LLM = 'citizen', 'llm'
LLM_PROMPT = 'P1'


class MissingCodebook(SystemExit):
    pass


def _book(frame, lmf='lmf', umf='umf'):
    book = {}
    for level, word in LEVELS.items():
        rows = frame[frame.term == word].sort_values('x')
        if rows.empty:
            raise ValueError(f'Codebook lacks the level word {word!r}')
        book[level] = IT2Set(rows.x.to_numpy(float), rows[lmf].to_numpy(float), rows[umf].to_numpy(float))
    return book


def load_codebooks(rq1, rq2, prompt=LLM_PROMPT):
    """{codebook name: {level: IT2Set}}: citizen IT2, LLM IT2 and each citizen group's T1 (UMF = LMF)."""
    llm_path = Path(rq2) / 'llm_it2.csv'
    if not llm_path.exists():
        raise MissingCodebook(f'LLM codebook not found: {llm_path}. RQ3 needs the WP4 (RQ2) results; '
                              'run experiments/words/run_rq2.py or merge WP4 first.')
    llm = pd.read_csv(llm_path)
    if 'prompt' in llm:
        llm = llm[llm.prompt == prompt]
    books = {CITIZEN: _book(pd.read_csv(Path(rq1) / 'citizen_it2.csv')), LLM: _book(llm)}
    groups = pd.read_csv(Path(rq1) / 'group_t1.csv')
    for group, frame in groups.groupby('group', sort=True):
        books[group] = _book(frame, lmf='mu', umf='mu')
    return books


def memberships(m, book):
    """(UMF + LMF) / 2 of each level word at m, linear interpolation on the codebook grid."""
    out = {}
    for level, s in book.items():
        if not s.x[0] <= m <= s.x[-1]:
            raise ValueError(f'm = {m} outside the codebook universe')
        out[level] = float(np.interp(m, s.x, (s.umf + s.lmf) / 2))
    return out


def more_severe(levels):
    return max(levels, key=SEVERITY.index)


def neighbour_fallback(m, book):
    """POST-HOC alternative fallback: the more severe of the two words neighbouring m.

    Support of a word = grid points with (UMF + LMF) / 2 > 0. Left neighbour = word whose support ends
    closest below m; right neighbour = word whose support starts closest above m (ties: more severe).
    At an end of the scale only one neighbour exists and is chosen.
    """
    left, right = {}, {}
    for level, s in book.items():
        x = s.x[(s.umf + s.lmf) / 2 > 0]
        if x.size == 0:
            raise ValueError(f'Level word {level!r} has empty support')
        if x[-1] < m:
            left[level] = x[-1]
        elif x[0] > m:
            right[level] = x[0]
        else:
            raise ValueError(f'm = {m} lies inside the support of {level!r}; no fallback needed')
    picks = []
    if left:
        best = max(left.values()); picks.append(more_severe([l for l, v in left.items() if v == best]))
    if right:
        best = min(right.values()); picks.append(more_severe([l for l, v in right.items() if v == best]))
    return more_severe(picks)


def verbalise(m, book, centres=None, fallback='centre'):
    """(level, fallback used). Max membership, ties to the more severe level; all-zero -> nearest centre
    (primary) or, for the post-hoc sensitivity, fallback='neighbour' (more severe neighbouring word)."""
    mu = memberships(m, book)
    best = max(mu.values())
    if best > 0:
        return max((l for l in SEVERITY if mu[l] == best), key=SEVERITY.index), False
    if fallback == 'neighbour':
        return neighbour_fallback(m, book), True
    if fallback != 'centre':
        raise ValueError(f'Unknown fallback {fallback!r}')
    centres = centres or {l: centre(book[l]) for l in SEVERITY}
    dist = {l: abs(centres[l] - m) for l in SEVERITY}
    nearest = min(dist.values())
    return max((l for l in SEVERITY if dist[l] == nearest), key=SEVERITY.index), True


def action(level, reply_word):
    a, _ = infer(crisp_memberships(level, CRISIS_WORDS), crisp_memberships(reply_word, APPROPRIATENESS_WORDS))
    if a != RULE_TABLE[(level, reply_word)]:
        raise AssertionError('Max-min inference disagrees with the rule table')
    return a
