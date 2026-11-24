"""Read CAPphrase CSVs without filtering, reordering, or modifying raw values.

Derived english_group: 'English is my first language' -> native; both published
'English is not my first language ...' categories (fluent / not fluent) ->
non_native. Derived age_group: 18-24/25-34 -> 18-34; 35-44/45-54 -> 35-54;
55-64/65-74/75+ -> 55+; Under 18 -> under_18. For either field, literal NA,
empty/whitespace-only strings, or null -> missing. Unknown nonmissing values
raise ValueError. Matching strips whitespace only for the derived comparison;
original demographic strings, including literal NA, remain unchanged.

Group columns are present on all three tables. Joining on response_id requires
unique metadata and complete coverage; no missing respondent is silently dropped.
No outlier removal or analysis is performed. Optional root accepts synthetic CSVs.
"""
from pathlib import Path
import pandas as pd
from .common import RAW

ENGLISH = {
    'English is my first language': 'native',
    'English is not my first language but I am fluent': 'non_native',
    'English is not my first language and I am not fluent': 'non_native',
}
AGE = {'18-24':'18-34','25-34':'18-34','35-44':'35-54','45-54':'35-54',
       '55-64':'55+','65-74':'55+','75+':'55+','Under 18':'under_18'}


def group_value(value, mapping):
    if pd.isna(value):
        return 'missing'
    key = str(value).strip()
    if key in ('', 'NA'):
        return 'missing'
    if key not in mapping:
        raise ValueError(f'Unknown CAPphrase demographic value: {value!r}')
    return mapping[key]


def _read(root, filename, columns):
    root = RAW / 'capphrase' if root is None else Path(root)
    df = pd.read_csv(root / filename, keep_default_na=False)
    if list(df.columns) != columns:
        raise ValueError(f'Unexpected columns in {filename}: {list(df.columns)}')
    if df.response_id.isna().any() or df.response_id.astype(str).str.strip().eq('').any():
        raise ValueError('Missing response_id')
    return df


def load_respondents(root=None):
    df = _read(root, 'respondent_metadata.csv', [
        'response_id','timestamp','age_band','english_background',
        'education_level','country_of_residence'])
    if df.response_id.duplicated().any():
        raise ValueError('Duplicate respondent metadata')
    df['english_group'] = df.english_background.map(lambda x: group_value(x, ENGLISH))
    df['age_group'] = df.age_band.map(lambda x: group_value(x, AGE))
    return df


def _groups(df, root):
    metadata = load_respondents(root)[['response_id','english_group','age_group']]
    result = df.merge(metadata, on='response_id', how='left', sort=False,
                      validate='many_to_one', indicator=True)
    if not result['_merge'].eq('both').all():
        raise ValueError('Response lacks respondent metadata')
    return result.drop(columns='_merge')


def load_judgements(root=None):
    df = _read(root, 'absolute_judgements.csv', ['response_id','term','probability','order'])
    df['probability'] = pd.to_numeric(df.probability, errors='raise')
    df['order'] = pd.to_numeric(df.order, errors='raise')
    return _groups(df, root)


def load_pairs(root=None):
    df = _read(root, 'pairwise_comparisons.csv', ['response_id','pair_id','term1','term2','selected'])
    return _groups(df, root)


def exclude_inattentive(judgements):
    """WP3 sensitivity only: drop respondents with Almost No Chance >= Almost Certain.

    Requires one of each anchor per respondent, rejects missing anchors, and
    returns a new table plus the respondent exclusion count. Other text and
    values are untouched. The primary pooled/group samples never use this rule.
    """
    anchors = judgements[judgements.term.isin(['Almost No Chance', 'Almost Certain'])]
    wide = anchors.pivot(index='response_id', columns='term', values='probability')
    if set(wide.index) != set(judgements.response_id) or wide.isna().any().any():
        raise ValueError('Inattention screening requires both anchors for every respondent')
    excluded = wide.index[wide['Almost No Chance'] >= wide['Almost Certain']]
    return judgements[~judgements.response_id.isin(excluded)].copy(), len(excluded)
