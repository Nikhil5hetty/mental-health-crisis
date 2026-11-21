#!/usr/bin/env python3
"""Build derived tables offline; raw inputs remain unchanged."""
import sys
from pathlib import Path
import os
if not sys.flags.isolated:
    os.execv(sys.executable,[sys.executable,'-I',str(Path(__file__).resolve()),*sys.argv[1:]])
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))  # Only trusted project source, never downloaded code.
from src.data import bhh,counselbench,mentalalign,mhb

def main():
    output=ROOT/'data/processed';output.mkdir(parents=True,exist_ok=True)
    tables={
        'bhh_crisis_labels':bhh.load_crisis_labels(),
        'bhh_crisis_degree':bhh.load_crisis_degree(),
        'bhh_appropriateness':bhh.load_appropriateness(),
        'bhh_protocol':bhh.load_protocol(),
        'counselbench_eval':counselbench.load_eval(),
        'counselbench_reply_degree':counselbench.load_reply_degree(),
        'mentalalign_ratings':mentalalign.load_ratings(),
    }
    if (ROOT/'data/raw/mentalhealthbench').is_dir():
        tables.update(mhb_conversations=mhb.load_conversations(),mhb_rubric=mhb.load_rubric())
    app=tables['bhh_appropriateness'];alignment=app.attrs['alignment_log'];runs=app.attrs['run_resolution_log']
    alignment.to_csv(output/'bhh_alignment_log.csv',index=False)
    runs.to_csv(output/'bhh_run_resolution_log.csv',index=False)
    print('Alignment status counts:',alignment.groupby(['file','field','status']).size().to_dict(),flush=True)
    print('Near-identical answers:',alignment.loc[alignment.status=='near_identical',['item_id','file']].to_dict('records'),flush=True)
    assert not alignment.status.eq('excluded').any(),'D11 alignment exclusions'
    for human,count in [('H1',206),('H2',205)]:
        subset=alignment[alignment.file.eq(f'human_label/{human}_scores_n206_s42.json')]
        assert len(subset[(subset.field=='inputs')&(subset.status=='exact')])==count
    h2=alignment[alignment.file.eq('human_label/H2_scores_n206_s42.json')]
    assert h2.loc[(h2.field=='inputs')&(h2.status=='artefact_accepted'),'item_id'].tolist()==[45]
    assert h2.loc[(h2.field=='answer')&(h2.status=='artefact_accepted'),'item_id'].tolist()==[21,22,42]
    assert set(alignment.loc[(alignment.field=='answer')&(alignment.status=='near_identical'),'item_id'])=={22}
    assert set(alignment.loc[(alignment.field=='answer')&(alignment.status=='normalised'),'item_id'])=={124}
    for judge,group in runs.groupby('judge'):
        counts=group.status.value_counts().to_dict();print(judge,'resolution:',counts,'runs:',group.n_runs.value_counts().to_dict())
        assert counts=={'resolved':190,'unscored':16}
        expected={3:189,2:1} if judge=='gpt-5-nano' else {3:190}
        assert group.loc[group.status=='resolved','n_runs'].value_counts().to_dict()==expected
    assert len(app.attrs['complete_case_items'])==190
    print('RQ2 complete-case items:',len(app.attrs['complete_case_items']))
    print('Distinct scores per rater:',app.attrs['distinct_scores'])
    crisis=tables['bhh_crisis_degree'].k_humans_crisis.value_counts().sort_index().tolist()
    medical=tables['counselbench_eval'].medical_advice.value_counts().to_dict()
    flags=tables['mentalalign_ratings'].attrs['exclusions']
    assert crisis==[36,33,33,26,78]
    assert medical=={'no':1729,'yes':220,'not_sure':51}
    assert flags.get(('Safety','out_of_range'),0)==34 and flags.get(('Safety','null'),0)==69
    assert tables['bhh_crisis_degree'].n_llm_missing.sum()==27
    print('Missing LLM crisis verdicts:',int(tables['bhh_crisis_degree'].n_llm_missing.sum()))
    print('Crisis by k humans:',crisis);print('Medical advice:',medical)
    print('MentalAlign exclusion counts (unique cells, not repeated across judges):',flags)
    print('CounselBench placeholders nulled:',tables['counselbench_eval'].attrs['exclusions'])
    if 'mhb_conversations' in tables:
        assert len(tables['mhb_conversations'])==1215 and len(tables['mhb_rubric'])==5262
        print('MHB conversations / criteria: 1215 / 5262')
    else: print('MHB optional: absent, skipped')
    for name,df in tables.items():
        # attrs contain audit DataFrames, not Parquet metadata. Do not modify data values.
        clean=df.copy();clean.attrs={}
        clean.to_parquet(output/(name+'.parquet'),index=False)
        print(name,'rows=',len(df),'columns=',len(df.columns))
    print('PASS: all reference and alignment checks')

if __name__=='__main__': main()
