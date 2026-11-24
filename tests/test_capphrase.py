"""Only synthetic fixtures; no published respondent rows are embedded here."""
import pandas as pd
import pytest
from src.data import capphrase

@pytest.fixture
def source(tmp_path):
    pd.DataFrame([
        [1,'2026-01','18-24','English is my first language','example','example'],
        [2,'2026-01','NA','NA','example','example'],
    ], columns=['response_id','timestamp','age_band','english_background','education_level','country_of_residence']).to_csv(tmp_path/'respondent_metadata.csv',index=False)
    pd.DataFrame([[2,'synthetic phrase',0,1],[1,'synthetic phrase',100,1]],
                 columns=['response_id','term','probability','order']).to_csv(tmp_path/'absolute_judgements.csv',index=False)
    pd.DataFrame([[1,1,'a','b','a'],[2,1,'b','a','b']],
                 columns=['response_id','pair_id','term1','term2','selected']).to_csv(tmp_path/'pairwise_comparisons.csv',index=False)
    return tmp_path

@pytest.mark.parametrize('value,expected',list(capphrase.AGE.items())+[(None,'missing'),('NA','missing'),(' ','missing')])
def test_age_mapping(value,expected):
    assert capphrase.group_value(value,capphrase.AGE)==expected

@pytest.mark.parametrize('value,expected',list(capphrase.ENGLISH.items())+[(float('nan'),'missing'),('NA','missing'),('','missing')])
def test_english_mapping(value,expected):
    assert capphrase.group_value(value,capphrase.ENGLISH)==expected

def test_unknown_category_fails():
    with pytest.raises(ValueError,match='Unknown'):capphrase.group_value('unrecognized',capphrase.AGE)

def test_loaders_preserve_rows_order_raw_values_and_bytes(source):
    before={p.name:p.read_bytes() for p in source.iterdir()}
    judgements=capphrase.load_judgements(source)
    pairs=capphrase.load_pairs(source)
    people=capphrase.load_respondents(source)
    assert judgements.response_id.tolist()==[2,1]
    assert judgements.probability.tolist()==[0,100]
    assert judgements.english_group.tolist()==['missing','native']
    assert judgements.age_group.tolist()==['missing','18-34']
    assert people.age_band.tolist()==['18-24','NA']
    assert pairs.selected.tolist()==['a','b']
    assert before=={p.name:p.read_bytes() for p in source.iterdir()}

@pytest.mark.parametrize('kind',['duplicate','missing'])
def test_metadata_coverage_fails(source,kind):
    path=source/'respondent_metadata.csv';df=pd.read_csv(path,keep_default_na=False)
    df=pd.concat([df,df.iloc[[0]]]) if kind=='duplicate' else df.iloc[[0]]
    df.to_csv(path,index=False)
    with pytest.raises(ValueError):capphrase.load_judgements(source)

def test_bad_probability_rejected(source):
    path=source/'absolute_judgements.csv';df=pd.read_csv(path)
    df['probability']='not a number';df.to_csv(path,index=False)
    with pytest.raises(ValueError):capphrase.load_judgements(source)

@pytest.mark.parametrize('value',[-1,101])
def test_verifier_flags_out_of_range_without_changing_source(source,value,monkeypatch):
    import importlib.util
    from pathlib import Path
    path=source/'absolute_judgements.csv';df=pd.read_csv(path);df.loc[0,'probability']=value;df.to_csv(path,index=False)
    before=path.read_bytes()
    script=Path(__file__).resolve().parents[1]/'scripts/verify_data.py'
    spec=importlib.util.spec_from_file_location('verify_capphrase_test',script)
    verifier=importlib.util.module_from_spec(spec);spec.loader.exec_module(verifier)
    # All three loaders still read only these synthetic files.
    for name in ['load_judgements','load_pairs','load_respondents']:
        original=getattr(capphrase,name)
        monkeypatch.setattr(capphrase,name,lambda root,fn=original:fn(source))
    class Capture(verifier.Report):
        def check(self,label,actual,expected):
            self.values[label]=actual
            return super().check(label,actual,expected)
    report=Capture();report.values={}
    verifier.verify_capphrase(report,source)
    assert report.values['CAPphrase probabilities outside 0–100 or missing']==1
    assert report.failed>0
    assert before==path.read_bytes()
