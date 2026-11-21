"""Published row order defines rater_pos, never a stable expert identity.

Only derived *_copy_clean fields replace stripped, case-insensitive placeholders
na, n/a, n/.a, nil, none, nothing wrong with it. with null. Original text stays
unchanged. Counts of newly nulled placeholders are returned in attrs['exclusions'].
"""
import pandas as pd
from .common import RAW

PLACEHOLDERS = {'na', 'n/a', 'n/.a', 'nil', 'none', 'nothing wrong with it.'}
SCALES = {'overall': range(1,6), 'empathy': range(1,6), 'specificity': range(1,6),
          'toxicity': range(1,6), 'medical_advice': ['yes','no','not_sure'],
          'factual': ['1','2','3','4','not_sure']}

def clean_copy(value):
    return None if isinstance(value,str) and value.strip().lower() in PLACEHOLDERS else value

def categorical(value, dimension):
    if dimension == 'medical_advice':
        mapping = {'Yes':'yes', 'No':'no', 'I am not sure':'not_sure'}
    else:
        mapping = {str(i):str(i) for i in range(1,5)} | {'I am not sure':'not_sure'}
    if value not in mapping:
        raise ValueError(f'Unexpected {dimension} score: {value!r}')
    return mapping[value]

def load_eval(root=None):
    root = RAW/'counselbench/CounselBench-Eval' if root is None else root
    # Disable pandas' implicit NA conversion to preserve literal raw text.
    df = pd.read_csv(root/'counselbench_eval.csv', keep_default_na=False, dtype=str)
    out = pd.DataFrame({'reply_id': df.questionID+'|'+df.responder,
                        'responder':df.responder, 'topic':df.topic})
    out['rater_pos'] = out.groupby('reply_id',sort=False).cumcount()+1
    assert out.groupby('reply_id').size().eq(5).all(), 'Expected five expert rows per reply'
    for dimension in ['overall','empathy','specificity','toxicity']:
        values = pd.to_numeric(df[dimension+'_score'],errors='raise')
        assert values.isin(SCALES[dimension]).all(), dimension
        out[dimension] = values.astype('int64')
    out['medical_advice'] = df.medical_advice_score.map(lambda x:categorical(x,'medical_advice'))
    out['factual'] = df.factual_consistency_score.map(lambda x:categorical(x,'factual'))
    counts = {}
    for col in df:
        if col.endswith(('_copy','_reason')):
            out[col] = df[col]
            if col.endswith('_copy'):
                out[col+'_clean'] = df[col].map(clean_copy)
                counts[col] = int(out[col+'_clean'].isna().sum())
    out.attrs['exclusions'] = counts
    return out

def load_reply_degree(root=None):
    df=load_eval(root)
    rows=[]
    for reply, group in df.groupby('reply_id',sort=False):
        row={'reply_id':reply, 'responder':group.responder.iloc[0], 'topic':group.topic.iloc[0]}
        for dimension,scale in SCALES.items():
            for value in scale:
                row[f'{dimension}_count_{value}']=int(group[dimension].eq(value).sum())
        row['k_medical_yes']=row['medical_advice_count_yes']
        row['k_medical_not_sure']=row['medical_advice_count_not_sure']
        row['k_factual_not_sure']=row['factual_count_not_sure']
        for risk in ['medical','factual','toxicity']:
            row[f'k_{risk}_copied']=sum(isinstance(x,str) and bool(x.strip()) for x in group[risk+'_copy_clean'])
        rows.append(row)
    return pd.DataFrame(rows)
