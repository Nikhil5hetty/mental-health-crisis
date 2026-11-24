#!/usr/bin/env python3
"""Descriptive respondent profile of CAPphrase (for the paper's study design; no hypothesis, no model).

Writes results/words/rq1/respondent_profile.csv: field, category, n, share (share of all respondents).
Fields: country of residence (United Kingdom, United States, other, missing), education level, English
background and age band, using the published categories unchanged; empty/"NA" counts as missing.
Aggregate counts only; no identifiers. Offline and deterministic.

    python -I experiments/words/describe_respondents.py
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # trusted project only; works with python -I
import pandas as pd
from src.data import capphrase

OUT = ROOT / 'results/words/rq1/respondent_profile.csv'
COUNTRIES = ['United Kingdom', 'United States']
FIELDS = {'country_of_residence': 'country', 'education_level': 'education',
          'english_background': 'english_background', 'age_band': 'age_band'}


def profile(meta: pd.DataFrame) -> pd.DataFrame:
    n = len(meta)
    rows = []
    for column, field in FIELDS.items():
        values = meta[column].astype(str).str.strip().replace({'': 'missing', 'NA': 'missing'})
        if column == 'country_of_residence':
            values = values.where(values.isin(COUNTRIES + ['missing']), 'other')
        counts = values.value_counts()
        order = sorted(counts.index, key=lambda c: (c == 'missing', c == 'other', -counts[c], c))
        rows += [dict(field=field, category=c, n=int(counts[c]), share=counts[c] / n) for c in order]
    rows.append(dict(field='all', category='respondents', n=n, share=1.0))
    return pd.DataFrame(rows)


def main():
    meta = capphrase.load_respondents()
    table = profile(meta)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False, float_format='%.17g', lineterminator='\n')
    print(f'Wrote {OUT.relative_to(ROOT)} ({len(meta)} respondents)')


if __name__ == '__main__':
    main()
