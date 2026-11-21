import math
from pathlib import Path
import pytest
from src.data import bhh,counselbench,mentalalign,mhb
from src.data.common import files

@pytest.mark.parametrize('value,expected',[('I am not sure','not_sure'),('Yes','yes'),('No','no')])
def test_medical_scores(value,expected):
    assert counselbench.categorical(value,'medical_advice')==expected

def test_factual_score():
    assert counselbench.categorical('I am not sure','factual')=='not_sure'
    assert counselbench.categorical('4','factual')=='4'
    with pytest.raises(ValueError):counselbench.categorical('5','factual')

@pytest.mark.parametrize('value',['na','N/A',' N/.A ','nil','None','nothing wrong with it.'])
def test_placeholder_rule(value):
    assert counselbench.clean_copy(value) is None
    assert counselbench.clean_copy('There is nothing wrong here')=='There is nothing wrong here'

def test_score_cell_parsing():
    parsed=mentalalign.parse_cell('{"Safety": 6, "Empathy": null, "Guidance": 3}')
    assert parsed['Safety']==(6.,'out_of_range')
    assert parsed['Empathy'][1]=='null'
    assert parsed['Guidance']==(3.,'ok')
    assert parsed['Understanding'][1]=='unparseable'
    assert mentalalign.parse_cell('Safety: 4; Empathy: 2')['Safety']==(4.,'ok')

def test_join_and_alignment():
    result=bhh.verify_pair('one | two','answer',['one','two'],'answer')
    assert result['inputs']==('exact',1.)
    assert bhh.verify_pair('one two','answer',['one','two'],'answer')['inputs'][0]=='excluded'

@pytest.mark.parametrize('value',[float('nan'),42,3.5,'T+D24+B24'])
def test_artefacts(value):
    assert bhh.is_artefact(value)
    assert bhh.verify_pair('input',value,['input'],'answer')['answer'][0]=='artefact_accepted'
    assert bhh.verify_pair(value,value,['input'],'answer')['answer'][0]=='excluded'
    assert not bhh.is_artefact('ordinary text')

def test_cr_and_nfkc_comparison_only():
    original='Ａ\r\n  B '
    result=bhh.verify_pair('input',original,['input'],'A B')
    assert result['answer'][0]=='normalised'
    assert original=='Ａ\r\n  B '

def test_near_identical_answer():
    assert bhh.verify_pair('input','a'*200+'b',['input'],'a'*200+'c')['answer'][0]=='near_identical'

def test_exactly_one_hit_and_mean():
    row={'inputs':['input'],'answer':'answer','evaluation':2.,'evals':[1,3]}
    with pytest.raises(AssertionError,match='exactly one'):bhh.resolve_runs(row,[])
    with pytest.raises(AssertionError,match='exactly one'):bhh.resolve_runs(row,[row,row])
    with pytest.raises(AssertionError,match='Evaluation'):bhh.resolve_runs(row,[row|{'evaluation':3}])

def test_no_padding():
    row={'inputs':['input'],'answer':'answer','evaluation':2.,'evals':'[1, 3]'}
    assert bhh.resolve_runs(row,[row])==[1.,3.]

def test_merged_labels_excluded(tmp_path):
    folder=tmp_path/'llm_label';folder.mkdir()
    stem='gpt-4o-mini-labeled-sampled_dataset_n_200_merged_n50-noSeed_156-s42-'
    (folder/(stem+'20250101-000000.json')).write_text('[]')
    (folder/(stem+'merged_labels.json')).write_text('[]')
    assert [p.name for p in bhh.label_files(tmp_path)]==[stem+'20250101-000000.json']

def test_misaligned_label_raises(tmp_path):
    import json
    (tmp_path/'human_label').mkdir();(tmp_path/'llm_label').mkdir()
    reference=[{'inputs':['synthetic'],'label':'no_crisis'} for _ in range(206)]
    for i in range(1,5):
        spelling='labelled' if i==4 else 'labeled'
        rows=[dict(r) for r in reference]
        if i==2:rows[0]={'inputs':['different'],'label':'no_crisis'}
        (tmp_path/f'human_label/H{i}_{spelling}_n206_s42.json').write_text(json.dumps(rows))
    for model in ['a','b','c']:
        for run in [1,2,3]:
            (tmp_path/f'llm_label/{model}-labeled-sampled_dataset_n_200_merged_n50-noSeed_156-s42-{run}.json').write_text(json.dumps(reference))
    with pytest.raises(AssertionError,match='Label input mismatch'):bhh.load_crisis_labels(tmp_path)

def test_macosx_ignored(tmp_path):
    (tmp_path/'__MACOSX').mkdir()
    (tmp_path/'__MACOSX/extra.jsonl').write_text('{}\n')
    (tmp_path/'data.jsonl').write_text('{}\n')
    assert [p.name for p in files(tmp_path,'*.jsonl')]==['data.jsonl']

def test_missing_verdict_excluded_from_counts(monkeypatch):
    import pandas as pd
    rows=[dict(item_id=0,annotator_type='human',label='no_crisis') for _ in range(4)]
    rows += [dict(item_id=0,annotator_type='llm',label=label) for label in [None,'no_crisis','self-harm']]
    monkeypatch.setattr(bhh,'load_crisis_labels',lambda root:pd.DataFrame(rows))
    monkeypatch.setattr(bhh,'read_json',lambda path:[{'inputs':['synthetic']}])
    row=bhh.load_crisis_degree().iloc[0]
    assert row.k_llm_crisis==1
    assert row.n_llm_observed==2 and row.n_llm_missing==1
    assert row['llm_count_no_crisis']==1

def test_answer_artefacts_parquet_roundtrip(tmp_path):
    import pandas as pd
    values=[' original\r\n text ',42,float('nan'),'T+D24+B24']
    df=pd.DataFrame([bhh.answer_fields(v) for v in values])
    path=tmp_path/'synthetic.parquet';df.to_parquet(path,index=False)
    restored=pd.read_parquet(path)
    assert restored.answer_text.iloc[0]==values[0]
    assert restored.answer_text.iloc[1] is None
    assert restored.answer_artefact_type.iloc[1]=='int'
    assert restored.answer_artefact_value.iloc[1]=='42'
    assert restored.answer_artefact_value.iloc[2]=='nan'
    assert restored.answer_text.iloc[3]==values[3]
