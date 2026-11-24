#!/usr/bin/env python3
"""Elicit LLM probability values for the CAPphrase terms (docs/ANALYSIS_PLAN_WORDS.md, RQ2).

For each model x prompt (P1, P2) x term (19) x sample (50) one answer is requested from a local Ollama
server with temperature 1.0 and seed = 20261124 + sample index. A reply is valid only if, after stripping
white space, it is a bare number with an optional % and an optional trailing period, and the number lies in
[0, 100] (strict parsing, D19). Otherwise the same prompt is asked again in the same conversation (the failed
reply stays in the history, same seed), up to 2 times; after that the value is missing.

Output: data/llm_answers/answers.csv, append-only and resumable. One row per (model, prompt_id, term, sample);
rows already present are skipped, so an interrupted run continues where it stopped. Missing answers are rows
with an empty value and attempts = 3. The digest of each model tag is read from /api/tags, written to every
row, and a run refuses to append answers for a tag whose digest differs from rows already in the file.

    python -I experiments/words/elicit_llm.py              # real run (Ollama must be running)
    python -I experiments/words/elicit_llm.py --dry-run    # fake model, writes data/processed/
    python -I experiments/words/elicit_llm.py --requeue-invalid   # move pre-D19 accepted rows that fail
                                                                  # strict parsing to superseded.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).with_name('models.yaml')
OUT = ROOT / 'data/llm_answers/answers.csv'
SUPERSEDED_NAME = 'superseded.csv'  # next to the answers file
DRY_RUN_OUT = ROOT / 'data/processed/llm_answers_dryrun.csv'

# The 19 CAPphrase terms, spelled as in data/raw/capphrase/absolute_judgements.csv (checked by a test).
TERMS = [
    'About Even', 'Almost Certain', 'Almost No Chance', 'Better than Even', 'Chances are Slight',
    'Could Happen', 'Highly Likely', 'Highly Unlikely', 'Improbable', 'Likely', 'Little Chance',
    'May Happen', 'Might Happen', 'Probable', 'Realistic Possibility', 'Remote Chance', 'Unlikely',
    'Very Good Chance', 'Will Happen',
]
P1 = ('On a scale from 0 to 100, what probability (in percent) does the phrase "{term}" convey? '
      'Answer with a single number only.')
PERSONA = 'You are a member of the public living in Switzerland.'
PROMPT_IDS = ('P1', 'P2')

TEMPERATURE = 1.0
N_SAMPLES = 50
SEED_BASE = 20261124
MAX_REASKS = 2
NUM_PREDICT = 64  # cap on generated tokens; a single number needs far fewer
TIMEOUT_S = 300

COLUMNS = ['model', 'digest', 'prompt_id', 'term', 'sample', 'raw_text', 'value', 'attempts', 'timestamp']
BARE_NUMBER = re.compile(r'^(-?\d+(?:\.\d+)?)\s*%?\s*\.?$')  # D19
REQUEUE_REASON = 'D19: raw_text is not a bare number in [0, 100]'


def prompt_text(prompt_id: str, term: str) -> str:
    p1 = P1.format(term=term)
    if prompt_id == 'P1':
        return p1
    if prompt_id == 'P2':
        return f'{PERSONA} {p1}'
    raise ValueError(f'Unknown prompt id {prompt_id!r}')


def parse_value(text: str | None) -> float | None:
    """Strict parsing (D19): '70', '70%', ' 72.5 ', '70.' -> number; '## Step 1: ...', '70-80', 'about 70' -> None."""
    m = BARE_NUMBER.match((text or '').strip())
    if m is None:
        return None
    value = float(m.group(1))
    return value if 0 <= value <= 100 else None


def ask(client, model: str, prompt: str, seed: int) -> tuple[str, float | None, int]:
    """Ask once, then re-ask the same prompt up to MAX_REASKS times. Returns (raw_text, value, attempts)."""
    messages = [{'role': 'user', 'content': prompt}]
    for attempt in range(1, MAX_REASKS + 2):
        raw = client.chat(model, messages, seed)
        value = parse_value(raw)
        if value is not None:
            return raw, value, attempt
        messages += [{'role': 'assistant', 'content': raw}, {'role': 'user', 'content': prompt}]
    return raw, None, attempt


class OllamaClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')

    def _request(self, path: str, payload: dict | None = None) -> dict:
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(self.base_url + path, data=data, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            return json.load(resp)

    def tags(self) -> dict[str, str]:
        out = {}
        for m in self._request('/api/tags').get('models', []):
            for key in ('name', 'model'):
                if m.get(key):
                    out[m[key]] = m['digest']
        return out

    def chat(self, model: str, messages: list[dict], seed: int) -> str:
        resp = self._request('/api/chat', {
            'model': model, 'messages': messages, 'stream': False,
            'options': {'temperature': TEMPERATURE, 'seed': seed, 'num_predict': NUM_PREDICT},
        })
        return resp['message']['content']


class FakeClient:
    """Deterministic stand-in for Ollama (--dry-run and tests). Some replies are unparseable on purpose."""

    def __init__(self, models):
        self.models = list(models)

    def tags(self) -> dict[str, str]:
        return {m: 'dry-run' for m in self.models}

    def chat(self, model: str, messages: list[dict], seed: int) -> str:
        key = json.dumps([model, messages, seed]).encode()
        h = int.from_bytes(hashlib.sha256(key).digest()[:8], 'big')
        v = h % 101
        return [f'{v}', f'{v}%', f'About {v}%', '## Step 1: Understand the phrase'][(h >> 8) % 4]


def normalise_tag(tag: str) -> str:
    return tag if ':' in tag else tag + ':latest'


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not cfg or not cfg.get('models'):
        raise SystemExit(f'{path}: no models configured')
    return cfg


def read_existing(out: Path) -> tuple[set, dict[str, str]]:
    """Keys already written and the digest recorded per model."""
    done, digests = set(), {}
    if not out.is_file() or out.stat().st_size == 0:
        return done, digests
    with out.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != COLUMNS:
            raise SystemExit(f'{out}: unexpected header {reader.fieldnames}')
        for r in reader:
            key = (r['model'], r['prompt_id'], r['term'], int(r['sample']))
            if key in done:
                raise SystemExit(f'{out}: duplicate row {key}')
            done.add(key)
            if digests.setdefault(r['model'], r['digest']) != r['digest']:
                raise SystemExit(f'{out}: several digests for {r["model"]}')
    return done, digests


def requeue_invalid(out: Path, log=print) -> dict[tuple, int]:
    """Move accepted rows whose raw_text fails strict parsing (D19) from ``out`` to superseded.csv.

    Rows are never edited: each moved row is appended unchanged to superseded.csv with a reason column, then
    ``out`` is rewritten without it, so a normal run re-requests exactly those keys. Rows that are already
    missing (all attempts failed the old, looser rule, hence also the strict one) are the strict outcome and stay.
    Kept rows must parse to their stored value. Returns the moved count per (model, prompt_id, term).
    """
    if not out.is_file():
        raise SystemExit(f'{out} does not exist')
    with out.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != COLUMNS:
            raise SystemExit(f'{out}: unexpected header {reader.fieldnames}')
        rows = list(reader)
    keep, move = [], []
    for r in rows:
        value = parse_value(r['raw_text'])
        if value is None and r['value'] != '':
            move.append(r)
            continue
        if value is not None and r['value'] != f'{value:g}':
            raise SystemExit(f'{out}: stored value {r["value"]} differs from strict parse {value} for {r}')
        keep.append(r)

    counts: dict[tuple, int] = {}
    for r in move:
        k = (r['model'], r['prompt_id'], r['term'])
        counts[k] = counts.get(k, 0) + 1
    n_missing = sum(1 for r in keep if r['value'] == '')
    log(f'{len(rows)} rows in {out}: {len(move)} to requeue, {len(keep)} kept '
        f'({n_missing} already missing after all attempts)')
    for (model, pid, term), n in sorted(counts.items()):
        log(f'  {model}\t{pid}\t{term}\t{n}')
    if not move:
        return counts

    sup = out.with_name(SUPERSEDED_NAME)
    sup_cols = COLUMNS + ['reason']
    seen = set()
    if sup.is_file() and sup.stat().st_size:
        with sup.open(newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != sup_cols:
                raise SystemExit(f'{sup}: unexpected header {reader.fieldnames}')
            seen = {tuple(r[c] for c in COLUMNS) for r in reader}
    new_sup = not sup.is_file() or sup.stat().st_size == 0
    with sup.open('a', newline='', encoding='utf-8') as f:  # append first: a crash never loses a row
        w = csv.writer(f)
        if new_sup:
            w.writerow(sup_cols)
        for r in move:
            if tuple(r[c] for c in COLUMNS) not in seen:  # rerun after a crash between the two writes
                w.writerow([r[c] for c in COLUMNS] + [REQUEUE_REASON])
        f.flush()
        os.fsync(f.fileno())
    tmp = out.with_suffix(out.suffix + '.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerows([r[c] for c in COLUMNS] for r in keep)
    os.replace(tmp, out)
    log(f'Moved {len(move)} rows to {sup}; rerun without --requeue-invalid to request them again')
    return counts


def run(models: list[str], client, out: Path, prompt_ids=PROMPT_IDS, terms=TERMS,
        n_samples: int = N_SAMPLES, log=print) -> int:
    """Request every missing (model, prompt, term, sample) answer and append it to ``out``. Returns rows written."""
    available = {normalise_tag(k): v for k, v in client.tags().items()}
    missing = [m for m in models if normalise_tag(m) not in available]
    if missing:
        raise SystemExit(f'Models not available in Ollama: {missing}. Run: ollama pull <tag>')
    digest = {m: available[normalise_tag(m)] for m in models}

    done, recorded = read_existing(out)
    for m in models:
        if m in recorded and recorded[m] != digest[m]:
            raise SystemExit(f'{m}: digest {digest[m]} differs from {recorded[m]} already in {out}. '
                             'Pull the recorded version or write to a new file.')

    total = len(models) * len(prompt_ids) * len(terms) * n_samples
    todo = total - sum(1 for k in done if k[0] in models and k[1] in prompt_ids and k[2] in terms
                       and k[3] < n_samples)
    log(f'{total} answers planned, {total - todo} already in {out}, {todo} to request')
    out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out.is_file() or out.stat().st_size == 0
    written, start = 0, time.monotonic()
    with out.open('a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(COLUMNS)
        for model in models:
            for pid in prompt_ids:
                for term in terms:
                    for sample in range(n_samples):
                        if (model, pid, term, sample) in done:
                            continue
                        raw, value, attempts = ask(client, model, prompt_text(pid, term), SEED_BASE + sample)
                        writer.writerow([model, digest[model], pid, term, sample, raw,
                                         '' if value is None else f'{value:g}', attempts,
                                         datetime.now(timezone.utc).isoformat(timespec='seconds')])
                        f.flush()
                        written += 1
                        if written % 100 == 0:
                            rate = (time.monotonic() - start) / written
                            log(f'{written}/{todo} written, about {rate * (todo - written) / 60:.0f} min left')
    log(f'Done: {written} rows written to {out}')
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--config', type=Path, default=CONFIG)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--dry-run', action='store_true', help='use a fake model; never touches the real output')
    ap.add_argument('--models', nargs='+', help='subset of the configured models')
    ap.add_argument('--requeue-invalid', action='store_true',
                    help='move accepted rows that fail strict parsing (D19) to superseded.csv, then exit')
    ap.add_argument('--samples', type=int, default=N_SAMPLES, help='samples per term (plan: 50; smaller only to test)')
    args = ap.parse_args(argv)

    if args.requeue_invalid:
        requeue_invalid(args.out or OUT)
        return 0

    cfg = load_config(args.config)
    models = args.models or list(cfg['models'])
    unknown = set(models) - set(cfg['models'])
    if unknown:
        raise SystemExit(f'Not in {args.config}: {sorted(unknown)}')
    if args.samples != N_SAMPLES:
        print(f'WARNING: {args.samples} samples per term; the plan requires {N_SAMPLES}', file=sys.stderr)

    if args.dry_run:
        out = args.out or DRY_RUN_OUT
        if out.resolve() == OUT.resolve():
            raise SystemExit('--dry-run never writes the real answers file')
        client = FakeClient(models)
    else:
        out = args.out or OUT
        client = OllamaClient(cfg.get('base_url', 'http://localhost:11434'))
    run(models, client, out, n_samples=args.samples)
    return 0


if __name__ == '__main__':
    sys.exit(main())
