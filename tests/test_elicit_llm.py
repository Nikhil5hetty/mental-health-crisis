"""Synthetic tests for the LLM elicitation harness (no Ollama, no network)."""
import csv
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('elicit_llm', ROOT / 'experiments/words/elicit_llm.py')
el = importlib.util.module_from_spec(_spec)
sys.modules['elicit_llm'] = el
_spec.loader.exec_module(el)

MODELS = ['llama3.1:8b', 'qwen2.5:7b', 'gemma2:9b', 'mistral:7b']


def rows(path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


class Scripted:
    """Replies from a fixed script; records the messages it was sent."""

    def __init__(self, replies, digest='sha256:aaa'):
        self.replies, self.digest, self.calls = list(replies), digest, []

    def tags(self):
        return {'m:1': self.digest}

    def chat(self, model, messages, seed):
        self.calls.append((list(messages), seed))
        return self.replies.pop(0)


class Crashing(el.FakeClient):
    def __init__(self, models, after):
        super().__init__(models)
        self.left = after

    def chat(self, model, messages, seed):
        if self.left == 0:
            raise ConnectionError('server went away')
        self.left -= 1
        return super().chat(model, messages, seed)


@pytest.mark.parametrize('text,expected', [
    ('70', 70.0), ('about 70%', 70.0), ('About 70%', 70.0), ('70-80', 70.0), ('70 - 80', 70.0),
    ('12.5', 12.5), ('100', 100.0), ('0', 0.0), ('I would say 85 percent.', 85.0),
    ('N/A', None), ('', None), (None, None), ('unsure', None),
    ('150', None), ('-5', None), ('150 or maybe 70', None),
])
def test_parse_value(text, expected):
    assert el.parse_value(text) == expected


def test_prompts_follow_plan():
    p1 = el.prompt_text('P1', 'Likely')
    assert p1 == ('On a scale from 0 to 100, what probability (in percent) does the phrase "Likely" convey? '
                  'Answer with a single number only.')
    assert el.prompt_text('P2', 'Likely') == 'You are a member of the public living in Switzerland. ' + p1


def test_reask_until_parsable():
    c = Scripted(['N/A', 'about 70%'])
    assert el.ask(c, 'm:1', 'Q', 7) == ('about 70%', 70.0, 2)
    first, second = c.calls
    assert first == ([{'role': 'user', 'content': 'Q'}], 7)
    assert second[0][1:] == [{'role': 'assistant', 'content': 'N/A'}, {'role': 'user', 'content': 'Q'}]
    assert second[1] == 7  # seed = base + sample index for every attempt


def test_missing_after_two_reasks():
    c = Scripted(['N/A', 'no idea', '200', 'never asked'])
    assert el.ask(c, 'm:1', 'Q', 7) == ('200', None, 3)
    assert len(c.calls) == 3


def test_dry_run_writes_7600_rows(tmp_path):
    out = tmp_path / 'answers.csv'
    assert el.main(['--dry-run', '--out', str(out)]) == 0
    r = rows(out)
    assert len(r) == 7600 == 4 * 2 * 19 * 50
    assert list(r[0]) == el.COLUMNS
    assert len({(x['model'], x['prompt_id'], x['term'], x['sample']) for x in r}) == 7600
    assert {x['model'] for x in r} == set(MODELS)
    assert {x['term'] for x in r} == set(el.TERMS)
    assert all(x['value'] == '' or 0 <= float(x['value']) <= 100 for x in r)
    assert all(x['value'] != '' or x['attempts'] == '3' for x in r)
    # A second run adds nothing.
    assert el.run(MODELS, el.FakeClient(MODELS), out, log=lambda *_: None) == 0
    assert len(rows(out)) == 7600


def test_resume_after_crash_gives_same_rows(tmp_path):
    full, part = tmp_path / 'full.csv', tmp_path / 'part.csv'
    kw = dict(terms=el.TERMS[:3], n_samples=4, log=lambda *_: None)
    el.run(MODELS[:2], el.FakeClient(MODELS), full, **kw)
    with pytest.raises(ConnectionError):
        el.run(MODELS[:2], Crashing(MODELS, after=10), part, **kw)
    n_before = len(rows(part))
    assert 0 < n_before < 48
    assert el.run(MODELS[:2], el.FakeClient(MODELS), part, **kw) == 48 - n_before
    strip = lambda rs: sorted(tuple(v for k, v in x.items() if k != 'timestamp') for x in rs)  # noqa: E731
    assert strip(rows(part)) == strip(rows(full))


def test_refuses_digest_change(tmp_path):
    out = tmp_path / 'a.csv'
    el.run(['m:1'], Scripted(['5'] * 2, digest='sha256:aaa'), out, prompt_ids=['P1'], terms=['Likely'],
           n_samples=2, log=lambda *_: None)
    with pytest.raises(SystemExit, match='digest'):
        el.run(['m:1'], Scripted(['5'], digest='sha256:bbb'), out, prompt_ids=['P1'], terms=['Likely'],
               n_samples=3, log=lambda *_: None)


def test_refuses_unpulled_model(tmp_path):
    with pytest.raises(SystemExit, match='ollama pull'):
        el.run(['absent:1b'], el.FakeClient(MODELS), tmp_path / 'a.csv', log=lambda *_: None)


def test_dry_run_never_writes_real_file():
    with pytest.raises(SystemExit, match='never writes'):
        el.main(['--dry-run', '--out', str(el.OUT)])


def test_config_defaults():
    assert el.load_config(el.CONFIG)['models'] == MODELS


@pytest.mark.skipif(not (ROOT / 'data/raw/capphrase/absolute_judgements.csv').is_file(), reason='raw data absent')
def test_terms_match_capphrase():
    with (ROOT / 'data/raw/capphrase/absolute_judgements.csv').open(newline='', encoding='utf-8') as f:
        assert {r['term'] for r in csv.DictReader(f)} == set(el.TERMS)
