"""Read-only BHH loaders implementing D11/D12. Audit tables live in attrs."""
import ast
import difflib
import math
import re
import unicodedata
from pathlib import Path
import pandas as pd
from .common import RAW,read_json,number

CATEGORIES=['suicidal_ideation','self-harm','anxiety_crisis','violent_thoughts',
            'substance_abuse_or_withdrawal','risk_taking_behaviours','no_crisis']
JUDGES=['gpt-4o-mini','gpt-5-nano','meta-llama']
ARTEFACT=re.compile(r'^[A-Z]\+[A-Z]+\d+\+[A-Z]+\d+$')

def root_path(root): return RAW/'between_help_and_harm' if root is None else Path(root)
def is_artefact(value): return not isinstance(value,str) or bool(ARTEFACT.fullmatch(value))
def normalise(value): return ' '.join(unicodedata.normalize('NFKC',value).replace('\r','').split())
def input_text(value):
    if isinstance(value,list):
        assert all(isinstance(x,str) for x in value),'Invalid input list'
        return ' | '.join(value)
    return value

def verify_pair(inputs, answer, label_inputs, reference_answer):
    """Return field status and similarity; artefacts need the other field verified."""
    expected=input_text(label_inputs); actual=input_text(inputs)
    checks={}
    checks['inputs']=('artefact',float('nan')) if is_artefact(actual) else (('exact',1.0) if actual==expected else ('excluded',0.0))
    if is_artefact(answer) or is_artefact(reference_answer): checks['answer']=('artefact',float('nan'))
    elif answer==reference_answer: checks['answer']=('exact',1.0)
    else:
        a,b=normalise(answer),normalise(reference_answer)
        ratio=difflib.SequenceMatcher(None,a,b).ratio()
        checks['answer']=('normalised' if a==b else ('near_identical' if ratio>=.99 else 'excluded'),ratio)
    verified={'exact','normalised','near_identical'}
    original=dict(checks)
    for field,other in [('inputs','answer'),('answer','inputs')]:
        if original[field][0]=='artefact':
            checks[field]=('artefact_accepted' if original[other][0] in verified else 'excluded',float('nan'))
    return checks

def label_files(root):
    return sorted(p for p in (root/'llm_label').glob('*-labeled-sampled_dataset_n_200_merged_n50-noSeed_156-s42-*.json')
                  if not p.name.endswith('merged_labels.json'))

def load_crisis_labels(root=None):
    root=root_path(root)
    baseline=read_json(root/'human_label/H1_labeled_n206_s42.json')
    assert len(baseline)==206
    sources=[]
    for i in range(1,5):
        name=f'H{i}_'+('labelled' if i==4 else 'labeled')+'_n206_s42.json'
        sources.append((root/'human_label'/name,f'H{i}','human',None,None))
    counts={}
    for path in label_files(root):
        model=path.name.split('-labeled-')[0]
        counts[model]=counts.get(model,0)+1
        run=counts[model]
        sources.append((path,f'{model}_run{run}','llm',model,run))
    assert len(counts)==3 and set(counts.values())=={3},'Expected 3 models x 3 independent runs'
    rows=[]
    for path,annotator,kind,model,run in sources:
        records=read_json(path)
        assert len(records)==len(baseline)
        for i,(row,ref) in enumerate(zip(records,baseline)):
            assert row['inputs']==ref['inputs'],f'Label input mismatch: {path.name} item {i}'
            assert row['label'] in CATEGORIES or (kind=='llm' and row['label']==''), 'Unknown crisis label'
            rows.append(dict(item_id=i,annotator=annotator,annotator_type=kind,model=model,run=run,label=row['label'] or None))
    out=pd.DataFrame(rows);out['run']=out['run'].astype('Int64')
    return out

def load_crisis_degree(root=None):
    root=root_path(root);df=load_crisis_labels(root)
    baseline=read_json(root/'human_label/H1_labeled_n206_s42.json')
    rows=[]
    for item,group in df.groupby('item_id',sort=True):
        row={'item_id':item,'item_text':input_text(baseline[item]['inputs'])}
        for kind,prefix in [('human','human'),('llm','llm')]:
            subset=group[group.annotator_type==kind]
            if kind=='llm':
                row['n_llm_observed']=int(subset.label.notna().sum())
                row['n_llm_missing']=int(subset.label.isna().sum())
            subset=subset[subset.label.notna()]
            row['k_humans_crisis' if kind=='human' else 'k_llm_crisis']=int(subset.label.ne('no_crisis').sum())
            for category in CATEGORIES: row[f'{prefix}_count_{category}']=int(subset.label.eq(category).sum())
        # Strict binary majority; ties remain explicit, rather than guessing a category.
        k=row['k_humans_crisis'];row['human_majority']='crisis' if k>2 else ('no_crisis' if k<2 else 'tie')
        rows.append(row)
    return pd.DataFrame(rows)

def resolve_runs(sampled,merged):
    """D12: exact two-field match; one hit; matching mean; retain run list length."""
    hits=[r for r in merged if r['inputs']==sampled['inputs'] and r['answer']==sampled['answer']]
    assert len(hits)==1,f'Expected exactly one merged hit, got {len(hits)}'
    row=hits[0]
    assert abs(number(row['evaluation'])-number(sampled['evaluation']))<1e-9,'Evaluation mismatch'
    values=row['evals']
    if isinstance(values,str):
        # Published Python-style numeric list; literal_eval never executes code.
        values=ast.literal_eval(values)
    assert isinstance(values,list),'evals must be a list'
    return [number(x) for x in values]

def answer_fields(value):
    """Preserve original strings; retain non-string artefact type and repr separately."""
    return {'answer_text':value if isinstance(value,str) else None,
            'answer_artefact_type':type(value).__name__ if is_artefact(value) else None,
            'answer_artefact_value':repr(value) if is_artefact(value) else None}

def load_appropriateness(root=None):
    root=root_path(root)
    labels=read_json(root/'human_label/H1_labeled_n206_s42.json')
    humans={f'H{i}':read_json(root/f'human_label/H{i}_scores_n206_s42.json') for i in [1,2]}
    rows=[];alignment=[];resolution=[];cache={}
    def verify(records,path):
        assert len(records)==len(labels)==206
        accepted=[]
        for i,row in enumerate(records):
            checks=verify_pair(row['inputs'],row['answer'],labels[i]['inputs'],humans['H1'][i]['answer'])
            for field,(status,similarity) in checks.items():
                alignment.append(dict(item_id=i,file=path.relative_to(root).as_posix(),field=field,status=status,similarity=similarity))
            accepted.append(all(s!='excluded' for s,_ in checks.values()))
        return accepted
    for human,records in humans.items():
        path=root/f'human_label/{human}_scores_n206_s42.json'
        for i,ok in enumerate(verify(records,path)):
            if ok:
                r=records[i]
                rows.append(dict(item_id=i,rater=human,rater_type='human',model=None,run=None,
                                 score=number(r['evaluation']),crisis_label=r['label'],**answer_fields(r['answer'])))
    for judge in JUDGES:
        suffix='' if judge=='gpt-4o-mini' else '__'+judge
        path=root/f'llm_evaluator/sampled/sampled_lowest_scores_n200{suffix}.json'
        records=read_json(path);accepted=verify(records,path)
        for i,row in enumerate(records):
            source=row['sampled_from' if judge=='gpt-4o-mini' else 'source_file']
            log=dict(item_id=i,judge=judge,source_file=source,hits=0,status='unscored',n_runs=0)
            if not accepted[i]:log['status']='excluded'
            elif not math.isnan(number(row['evaluation'])):
                assert isinstance(source,str) and Path(source).name==source and '\\' not in source,'Unsafe source filename'
                merged_path=root/'llm_evaluator'/judge/'n_200/merged'/source
                if merged_path not in cache:cache[merged_path]=read_json(merged_path)
                values=resolve_runs(row,cache[merged_path])
                log.update(hits=1,status='resolved',n_runs=len(values))
                for run,score in enumerate(values,1):
                    rows.append(dict(item_id=i,rater=f'{judge}_run{run}',rater_type='llm',model=judge,run=run,
                                     score=score,crisis_label=row['label'],**answer_fields(row['answer'])))
            resolution.append(log)
    out=pd.DataFrame(rows);out['run']=out['run'].astype('Int64')
    out.attrs['alignment_log']=pd.DataFrame(alignment)
    out.attrs['run_resolution_log']=pd.DataFrame(resolution)
    # Count items with both humans and every judge having observed scores (not 9-run completeness).
    sets=[set(out.loc[(out.rater==h)&out.score.notna(),'item_id']) for h in ['H1','H2']]
    sets += [set(out.loc[(out.model==j)&out.score.notna(),'item_id']) for j in JUDGES]
    out.attrs['complete_case_items']=sorted(set.intersection(*sets))
    out.attrs['distinct_scores']={r:sorted(g.score.dropna().unique().tolist()) for r,g in out.groupby('rater')}
    return out

def load_protocol(root=None):
    root=root_path(root);df=pd.read_csv(root/'llm_evaluator/protocol.csv')
    assert len(df)==7 and len(df.columns)==6
    rows=[]
    for _,row in df.iterrows():
        for col in df.columns[1:]:
            match=re.fullmatch(r'\((\d)\) (.+)',col);assert match
            rows.append(dict(category=row['label'],level=int(match[1]),level_name=match[2],description=row[col]))
    return pd.DataFrame(rows)
