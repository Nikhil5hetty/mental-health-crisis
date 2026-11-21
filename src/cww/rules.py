"""Escalation rules: crisis-level words, the action rule table and max-min inference.

This is a design proposal for the escalation demo (docs/ANALYSIS_PLAN_RQ1_DEMO.md,
Part C). It is NOT clinically validated.

Inference is Mamdani-style max-min firing (Mendel & Wu 2010, Perceptual
Computing, ch. 6 on rule-based Per-C engines): rule strength = min(antecedent
memberships); action strength = max over the rules concluding that action.
"""
from __future__ import annotations

from collections.abc import Mapping

from .codebook import APPROPRIATENESS_WORDS
from .sets import trapezoid_mf

#: Crisis-level words with T1 trapezoids (a, b, c, d) on [0, 1].
CRISIS_WORDS = {
    'None': (0.0, 0.0, 0.1, 0.3),
    'Possible': (0.1, 0.3, 0.4, 0.6),
    'Likely': (0.4, 0.6, 0.7, 0.9),
    'Clear': (0.7, 0.9, 1.0, 1.0),
}

#: Actions in increasing severity (index = severity).
ACTIONS = [
    'A0 Continue',
    'A1 Signpost helpline (143; 147 for young people)',
    'A2 Human review / handover to a counsellor',
    'A3 Urgent: human handover and show emergency numbers (144 / 112)',
]

#: Rule table: crisis word -> action index per reply word (Harmful .. Fully).
_TABLE = {
    'None': [2, 2, 0, 0, 0],
    'Possible': [2, 2, 1, 1, 0],
    'Likely': [3, 3, 2, 1, 1],
    'Clear': [3, 3, 3, 2, 1],
}
RULE_TABLE = {(c, r): _TABLE[c][i] for c in CRISIS_WORDS for i, r in enumerate(APPROPRIATENESS_WORDS)}


def crisis_memberships(mu: float) -> dict[str, float]:
    """Membership of each crisis word at the crisis degree ``mu`` in [0, 1]."""
    if not 0.0 <= mu <= 1.0:
        raise ValueError(f'Crisis degree {mu} outside [0, 1]')
    return {w: float(trapezoid_mf([mu], *p)[0]) for w, p in CRISIS_WORDS.items()}


def crisp_memberships(word: str, words) -> dict[str, float]:
    """Membership 1 for ``word`` and 0 for every other word."""
    if word not in words:
        raise ValueError(f'Unknown word {word!r}')
    return {w: float(w == word) for w in words}


def infer(crisis: Mapping[str, float], reply: Mapping[str, float]) -> tuple[int, list[float]]:
    """Max-min inference over the rule table.

    strength(rule c, r) = min(crisis[c], reply[r])
    strength(action a)  = max over rules with RULE_TABLE[c, r] = a
    Returns (action index, strengths). Ties go to the MORE severe action.
    """
    for name, m, words in [('crisis', crisis, CRISIS_WORDS), ('reply', reply, APPROPRIATENESS_WORDS)]:
        if set(m) != set(words):
            raise ValueError(f'{name} memberships must cover exactly {list(words)}')
        if any(not 0.0 <= v <= 1.0 for v in m.values()):
            raise ValueError(f'{name} memberships must lie in [0, 1]')
    strengths = [0.0] * len(ACTIONS)
    for (c, r), action in RULE_TABLE.items():
        strengths[action] = max(strengths[action], min(crisis[c], reply[r]))
    best = max(strengths)
    action = max(i for i, s in enumerate(strengths) if s == best)  # tie -> more severe
    return action, strengths
