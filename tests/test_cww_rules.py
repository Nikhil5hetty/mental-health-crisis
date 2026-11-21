"""Synthetic tests for the escalation rule engine (src/cww/rules.py)."""
import numpy as np
import pytest

from src.cww import APPROPRIATENESS_WORDS
from src.cww.rules import (ACTIONS, CRISIS_WORDS, RULE_TABLE, crisis_memberships, crisp_memberships,
                           infer)


def test_rule_table_complete():
    assert len(RULE_TABLE) == 20
    assert set(RULE_TABLE.values()) <= set(range(len(ACTIONS)))
    assert RULE_TABLE[('None', 'Harmful/Unacceptable')] == 2
    assert RULE_TABLE[('Clear', 'Partially Appropriate')] == 3
    assert RULE_TABLE[('Possible', 'Fully Appropriate')] == 0


@pytest.mark.parametrize('cell', sorted(RULE_TABLE))
def test_crisp_input_reproduces_table_lookup(cell):
    crisis, reply = cell
    action, strengths = infer(crisp_memberships(crisis, CRISIS_WORDS),
                              crisp_memberships(reply, APPROPRIATENESS_WORDS))
    assert action == RULE_TABLE[cell]
    assert strengths[action] == 1.0


@pytest.mark.parametrize('mu', np.linspace(0, 1, 101))
def test_crisis_memberships_in_unit_interval(mu):
    m = crisis_memberships(float(mu))
    assert set(m) == set(CRISIS_WORDS)
    assert all(0.0 <= v <= 1.0 for v in m.values())
    assert max(m.values()) > 0  # the four words cover [0, 1]


def test_crisis_memberships_values():
    assert crisis_memberships(0.0)['None'] == 1.0
    assert crisis_memberships(1.0)['Clear'] == 1.0
    m = crisis_memberships(0.2)
    assert m['None'] == pytest.approx(0.5) and m['Possible'] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        crisis_memberships(1.2)


def test_tie_goes_to_more_severe_action():
    # Clear crisis; reply equally Mostly (-> A2) and Fully (-> A1): tie at 0.6 -> A2.
    crisis = crisp_memberships('Clear', CRISIS_WORDS)
    reply = dict.fromkeys(APPROPRIATENESS_WORDS, 0.0)
    reply['Mostly Appropriate'] = reply['Fully Appropriate'] = 0.6
    action, strengths = infer(crisis, reply)
    assert strengths[1] == strengths[2] == 0.6 and action == 2


def test_all_zero_strengths_tie_to_most_severe():
    crisis = dict.fromkeys(CRISIS_WORDS, 0.0)
    reply = dict.fromkeys(APPROPRIATENESS_WORDS, 0.0)
    assert infer(crisis, reply)[0] == 3


def test_max_min_picks_strongest_rule():
    crisis = {'None': 0.0, 'Possible': 0.3, 'Likely': 0.8, 'Clear': 0.0}
    reply = dict.fromkeys(APPROPRIATENESS_WORDS, 0.0)
    reply['Mostly Appropriate'] = 0.9   # Likely x Mostly -> A1, strength 0.8
    reply['Inappropriate'] = 0.4        # Likely x Inapp -> A3, strength 0.4
    action, strengths = infer(crisis, reply)
    assert action == 1 and strengths[1] == pytest.approx(0.8) and strengths[3] == pytest.approx(0.4)


def test_invalid_memberships_rejected():
    reply = dict.fromkeys(APPROPRIATENESS_WORDS, 0.0)
    with pytest.raises(ValueError):
        infer({'None': 1.5, 'Possible': 0, 'Likely': 0, 'Clear': 0}, reply)
    with pytest.raises(ValueError):
        infer({'None': 1.0}, reply)
