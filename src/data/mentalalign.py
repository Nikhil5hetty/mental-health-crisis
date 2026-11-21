"""D8 exclusions are applied only to human scores, in memory; no source rewrites."""
import json
import re
import math
import pandas as pd
from .common import RAW, files, number

ATTRIBUTES=['Guidance','Informativeness','Relevance','Safety','Empathy','Helpfulness','Understanding']
PROVIDERS={'claude':'anthropic','gpt-4o':'openai','gemini':'google','o4-mini':'openai',
           'original':'human','ds_llama':'deepseek','ds_qwen':'deepseek', 'gpt4o':'openai',
           'gpt4omini':'openai','llama_3':'meta','qwen_2':'alibaba','qwen_3':'alibaba'}

def parse_cell(value):
    if pd.isna(value) or (isinstance(value,str) and not value.strip()):
        return {a:(float('nan'),'null') for a in ATTRIBUTES}
    try:
        obj=json.loads(value)
        if not isinstance(obj,dict): raise ValueError('not object')
    except (ValueError,TypeError):
        # Only explicitly named attribute/value pairs; never eval source cells.
        obj={}
        for attribute in ATTRIBUTES:
            hit=re.search(r'[\"\x27]?'+attribute+r'[\"\x27]?\s*:\s*(null|None|NaN|[-+]?\d+(?:\.\d+)?)\b',str(value))
            if hit: obj[attribute]=None if hit[1] in ['null','None','NaN'] else hit[1]
    result={}
    for a in ATTRIBUTES:
        if a not in obj: result[a]=(float('nan'),'unparseable'); continue
        if obj[a] is None: result[a]=(float('nan'),'null'); continue
        score=number(obj[a])
        flag='unparseable' if math.isnan(score) or isinstance(obj[a],bool) else ('ok' if 1<=score<=5 else 'out_of_range')
        result[a]=(score,flag)
    return result

def load_ratings(root=None):
    root=RAW/'mentalalign' if root is None else root
    paths=files(root/'Codes/Analysis','*_llm_judge_with_human.csv')
    assert len(paths)==4, 'Expected four analysis CSVs'
    rows=[]; reference=None
    for path in paths:
        judge=path.name.removesuffix('_llm_judge_with_human.csv')
        assert judge in ['claude','gpt-4o','gemini','o4-mini']
        df=pd.read_csv(path,keep_default_na=False)
        df.columns=df.columns.str.strip()
        models=[c.removeprefix('human_response_') for c in df if c.startswith('human_response_')]
        assert len(models)==10
        # CSV has no ConvID. Require exact context/response ordering across sources.
        alignment=df[['context','response']].to_dict('records')
        human={}
        for i,record in df.iterrows():
            for model in models:
                hs=parse_cell(record['human_response_'+model]); js=parse_cell(record['judge_response_'+model])
                for attribute in ATTRIBUTES:
                    score,flag=hs[attribute]
                    human[(i,model,attribute)]=(None if math.isnan(score) else score,flag)
                    rows.append({'conv_id':i+1,'response_model':model,'judge':judge,'attribute':attribute,
                                 'human_score':score if flag=='ok' else float('nan'),
                                 'judge_score':js[attribute][0], 'human_score_flag':flag,
                                 'same_provider':PROVIDERS[judge]==PROVIDERS[model]})
        if reference is None: reference=(alignment,human)
        else:
            assert reference[0]==alignment, 'MentalAlign context/response alignment failed'
            assert reference[1]==human, 'Human scores differ across judge files'
    out=pd.DataFrame(rows)
    unique=out.drop_duplicates(['conv_id','response_model','attribute'])
    out.attrs['exclusions']=unique.groupby(['attribute','human_score_flag']).size().to_dict()
    return out
