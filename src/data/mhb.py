"""Optional MentalHealthBench, without model/grader calls."""
import json
import pandas as pd
from .common import RAW,files

def records(root=None):
    root=RAW/'mentalhealthbench' if root is None else root
    paths=files(root,'*.jsonl')
    assert len(paths)==1,'Expected one official JSONL'
    with paths[0].open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]

def load_conversations(root=None):
    return pd.DataFrame([{**{k:r[k] for k in ['id','acuity','language','user_profile','has_prior_context']},
                          'n_messages':len(r['conversation']['messages'])} for r in records(root)])

def load_rubric(root=None):
    return pd.DataFrame([{'id':r['id'],'criterion_idx':i,**{k:c[k] for k in ['behavior_axis','criterion_text','points']}}
                         for r in records(root) for i,c in enumerate(r['rubric_items'])])
