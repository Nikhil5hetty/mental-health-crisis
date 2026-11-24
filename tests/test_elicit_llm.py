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
    ('70', 70.0), ('70%', 70.0), (' 72.5 ', 72.5), ('70.', 70.0), ('70 %', 70.0), ('70%.', 70.0),
    ('\n85\n', 85.0), ('100', 100.0), ('0', 0.0),
    ('## Step 1: Understand the phrase', None), ('1:1', None), ('70-80', None), ('about 70', None),
    ('About 70%', None), ('I would say 85 percent.', None), ('70 percent', None), ('70\n\nBecause ...', None),
    ('N/A', None), ('', None), (None, None), ('150', None), ('-5', None), ('101%', None), ('.5', None),
])
def test_parse_value(text, expected):
    assert el.parse_value(text) == expected


def test_prompts_follow_plan():
    p1 = el.prompt_text('P1', 'Likely')
    assert p1 == ('On a scale from 0 to 100, what probability (in percent) does the phrase "Likely" convey? '
                  'Answer with a single number only.')
    assert el.prompt_text('P2', 'Likely') == 'You are a member of the public living in Switzerland. ' + p1


def test_reask_until_parsable():
    c = Scripted(['N/A', '70%'])
    assert el.ask(c, 'm:1', 'Q', 7) == ('70%', 70.0, 2)
    first, second = c.calls
    assert first == ([{'role': 'user', 'content': 'Q'}], 7)
    assert second[0][1:] == [{'role': 'assistant', 'content': 'N/A'}, {'role': 'user', 'content': 'Q'}]
    assert second[1] == 7  # seed = base + sample index for every attempt


def test_missing_after_two_reasks():
    c = Scripted(['## Step 1: Understand', 'about 70', '200', 'never asked'])
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


def write_answers(path, raws):
    """Rows as the pre-D19 harness wrote them: value = first number in [0, 100], else missing."""
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(el.COLUMNS)
        for i, (term, raw, value, attempts) in enumerate(raws):
            w.writerow(['m:1', 'sha256:aaa', 'P1', term, i, raw, value, attempts, '2026-11-24T12:00:00+00:00'])


PRE_D19 = [
    ('Better than Even', '## Step 1: Understand the phrase', '1', 1),
    ('Better than Even', '55', '55', 1),
    ('Likely', '70%', '70', 1),
    ('Likely', 'About 70%', '70', 2),
    ('Likely', 'no idea', '', 3),
]


def test_requeue_moves_only_strict_failures(tmp_path):
    out = tmp_path / 'answers.csv'
    write_answers(out, PRE_D19)
    before = out.read_text(encoding='utf-8')
    logged = []
    counts = el.requeue_invalid(out, log=logged.append)
    assert counts == {('m:1', 'P1', 'Better than Even'): 1, ('m:1', 'P1', 'Likely'): 1}
    kept, sup = rows(out), rows(tmp_path / 'superseded.csv')
    assert [r['raw_text'] for r in kept] == ['55', '70%', 'no idea']  # order kept, rows unchanged
    assert [r['raw_text'] for r in sup] == ['## Step 1: Understand the phrase', 'About 70%']
    assert list(sup[0]) == el.COLUMNS + ['reason'] and sup[0]['reason'] == el.REQUEUE_REASON
    original = {r['sample']: r for r in csv.DictReader(before.splitlines())}
    assert [{k: r[k] for k in el.COLUMNS} for r in sup] == [original['0'], original['3']]  # copied unchanged
    assert any('2 to requeue' in line for line in logged)
    # Idempotent: nothing more to move, superseded.csv unchanged.
    assert el.requeue_invalid(out, log=lambda *_: None) == {}
    assert len(rows(tmp_path / 'superseded.csv')) == 2


def test_rerun_after_requeue_requests_exactly_the_moved_rows(tmp_path):
    out = tmp_path / 'answers.csv'
    write_answers(out, [('Likely', '## Step 1: Understand', '1', 1), ('Likely', '70', '70', 1),
                        ('Likely', 'About 65%', '65', 1)])
    el.requeue_invalid(out, log=lambda *_: None)
    c = Scripted(['## Step 1: Understand', '60', '65%'])
    c.tags = lambda: {'m:1': 'sha256:aaa'}
    assert el.run(['m:1'], c, out, prompt_ids=['P1'], terms=['Likely'], n_samples=3, log=lambda *_: None) == 2
    assert [seed - el.SEED_BASE for _, seed in c.calls] == [0, 0, 2]  # sample 0 (with one re-ask), sample 2
    by_sample = {r['sample']: (r['value'], r['attempts']) for r in rows(out)}
    assert by_sample == {'0': ('60', '2'), '1': ('70', '1'), '2': ('65', '1')}


def test_requeue_keeps_superseded_append_only(tmp_path):
    out = tmp_path / 'answers.csv'
    write_answers(out, PRE_D19[:1])
    el.requeue_invalid(out, log=lambda *_: None)
    write_answers(out, PRE_D19[:2])  # a later file with the same bad row and a good one
    el.requeue_invalid(out, log=lambda *_: None)
    assert len(rows(tmp_path / 'superseded.csv')) == 1  # same row is not appended twice


def test_requeue_rejects_value_mismatch(tmp_path):
    out = tmp_path / 'answers.csv'
    write_answers(out, [('Likely', '70', '71', 1)])
    with pytest.raises(SystemExit, match='differs'):
        el.requeue_invalid(out, log=lambda *_: None)
