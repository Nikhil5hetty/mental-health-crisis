#!/usr/bin/env python3
"""Frozen citizen-word RQ1; offline deterministic aggregate outputs only.

Bootstrap samples whole respondents (all 19 values together), with replacement,
200 times using default_rng(20261124). Same indices apply to every term.
CSV floats use 17 significant digits. PNG metadata has no timestamps.
"""
from pathlib import Path
import argparse
import itertools
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # trusted project only; works with python -I
import numpy as np
import pandas as pd
from src.data import capphrase
from src.cww.quantile import quantile_trapezoid, envelope, group_models, X
from src.cww.centroid import centroid_it2, centroid_t1
from src.cww.similarity import jaccard

SEED = 20261124
B = 200
GROUPS = [f'{english}|{age}' for english in ['native', 'non_native']
          for age in ['18-34', '35-54', '55+']]


def pair_consistency(judgements, pairs):
    """Strict concordance divided by ALL pair rows; equal absolute values are ties.

    Counts repeated published pairs as rows. Invalid selections, absent values,
    and duplicate respondent/term keys raise; no silent exclusion.
    """
    values = judgements.set_index(['response_id', 'term']).probability
    if not values.index.is_unique:
        raise ValueError('Duplicate respondent/term')
    if not (pairs.selected.eq(pairs.term1) | pairs.selected.eq(pairs.term2)).all():
        raise ValueError('Selected term was not presented')
    a = values.reindex(pd.MultiIndex.from_arrays([pairs.response_id, pairs.term1])).to_numpy()
    b = values.reindex(pd.MultiIndex.from_arrays([pairs.response_id, pairs.term2])).to_numpy()
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Missing absolute probability for pair')
    chosen_first = pairs.selected.eq(pairs.term1).to_numpy()
    consistent = int(((chosen_first & (a > b)) | (~chosen_first & (b > a))).sum())
    tied = int((a == b).sum())
    n = len(pairs)
    if n == 0:
        raise ValueError('No pair observations')
    return dict(n_pairs=n, consistent=consistent, ties=tied, inconsistent=n-consistent-tied,
                consistent_share=consistent/n, tie_share=tied/n,
                inconsistent_share=(n-consistent-tied)/n)


def model_metrics(model):
    cl, cr = centroid_it2(model)
    return dict(fou_area=float(np.sum(model.umf-model.lmf)*.5),
                centroid_left=cl, centroid_right=cr, centre=(cl+cr)/2)


def fit(data):
    pooled, citizen, groups, metrics, exclusions = {}, {}, {}, [], []
    for term, frame in data.groupby('term', sort=True):
        pooled[term] = quantile_trapezoid(frame.probability)
        samples = {group:frame.loc[(frame.english_group==group.split('|')[0]) &
                                   (frame.age_group==group.split('|')[1]), 'probability'].to_numpy()
                   for group in GROUPS}
        models, counts = group_models(samples)
        groups[term] = models
        citizen[term] = envelope(models.values())
        for group, n in counts.items():
            metrics.extend([dict(metric='n',term=term,group=group,value=n),
                            dict(metric='included',term=term,group=group,value=int(group in models))])
            if group not in models:
                exclusions.append((term,group,n))
        q25,q75 = np.quantile(frame.probability,[.25,.75],method='linear')
        info = dict(n=len(frame),iqr=q75-q25,median=float(frame.probability.median()),
                    pooled_centroid=centroid_t1(pooled[term]),
                    n_group_excluded=int(len(frame)-sum(counts.values())),
                    **model_metrics(citizen[term]))
        metrics.extend(dict(metric=k,term=term,group='pooled',value=v) for k,v in info.items())
    return pooled,citizen,groups,pd.DataFrame(metrics),exclusions


def bootstrap_models(data):
    wide = data.pivot(index='response_id',columns='term',values='probability').sort_index().sort_index(axis=1)
    if wide.isna().any().any():
        raise ValueError('Bootstrap requires complete respondent vectors')
    rng = np.random.default_rng(SEED)
    values = wide.to_numpy(); models = {term:[] for term in wide.columns}
    for _ in range(B):
        sample = values[rng.integers(0,len(values),size=len(values))]
        for col,term in enumerate(wide.columns):
            models[term].append(quantile_trapezoid(sample[:,col]))
    return {term:envelope(sets) for term,sets in models.items()}


def write_csv(frame,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(path,index=False,float_format='%.17g',lineterminator='\n')


def write_models(pooled,citizen,groups,directory,prefix='citizen'):
    write_csv(pd.DataFrame([dict(term=t,x=x,mu=mu) for t,a in pooled.items() for x,mu in zip(a.x,a.mu)]),
              directory/f'{prefix}_pooled_t1.csv')
    write_csv(pd.DataFrame([dict(term=t,x=x,umf=u,lmf=l) for t,a in citizen.items()
                           for x,u,l in zip(a.x,a.umf,a.lmf)]),directory/f'{prefix}_it2.csv')
    write_csv(pd.DataFrame([dict(term=t,group=g,x=x,mu=mu) for t,sets in groups.items()
                           for g,a in sets.items() for x,mu in zip(a.x,a.mu)]),directory/'group_t1.csv')


def figures(pooled,citizen,metrics,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8})
    median = metrics[(metrics.metric=='median')].set_index('term').value.to_dict()
    terms = sorted(citizen,key=lambda t:(median[t],t))
    fig,axes = plt.subplots(5,4,figsize=(12,12),sharex=True,sharey=True)
    for ax,term in zip(axes.flat,terms):
        a=citizen[term]
        ax.fill_between(X,a.lmf,a.umf,color='#4589b0',alpha=.4,label='Citizen FOU')
        ax.plot(X,a.umf,color='#24597a',lw=.8);ax.plot(X,a.lmf,color='#24597a',lw=.8)
        ax.plot(X,pooled[term].mu,color='#b34c32',ls='--',lw=.8,label='Pooled T1')
        ax.set_title(term);ax.set_xlim(0,100);ax.set_ylim(0,1.05)
    for ax in list(axes.flat)[len(terms):]:ax.set_visible(False)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower right');fig.supxlabel('Probability (%)');fig.supylabel('Membership')
    fig.tight_layout();fig.savefig(out/'fou_panels.png',dpi=150,metadata={'Software':None});plt.close(fig)
    areas={t:model_metrics(a)['fou_area'] for t,a in citizen.items()}
    ranked=sorted(areas,key=lambda t:(areas[t],t))
    fig,ax=plt.subplots(figsize=(8,7));ax.barh(ranked,[areas[t] for t in ranked],color='#4589b0')
    ax.set_xlabel('FOU area (probability percentage points × membership)')
    fig.tight_layout();fig.savefig(out/'fou_ranking.png',dpi=150,metadata={'Software':None});plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/words/rq1')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    data=capphrase.load_judgements();pairs=capphrase.load_pairs()
    assert data.response_id.nunique()==5174 and len(data)==98306 and data.term.nunique()==19
    assert data.probability.between(0,100).all()
    pooled,citizen,groups,metrics,exclusions=fit(data)
    write_models(pooled,citizen,groups,out/'codebooks')
    write_csv(metrics[['metric','term','group','value']],out/'metrics.csv')
    overlaps=pd.DataFrame([dict(term=term,group_a=a,group_b=b,jaccard=jaccard(sets[a],sets[b]))
                           for term,sets in groups.items() for a,b in itertools.combinations(sets,2)])
    write_csv(overlaps,out/'group_jaccard.csv')
    consistency=pair_consistency(data,pairs)
    write_csv(pd.DataFrame([consistency]),out/'consistency.csv')
    boot=bootstrap_models(data)
    filtered,removed=capphrase.exclude_inattentive(data)
    fp,fc,fg,fmetrics,fexclusions=fit(filtered)
    write_models(fp,fc,fg,out/'codebooks/filtered')
    write_csv(pd.DataFrame([dict(term=t,x=x,umf=u,lmf=l) for t,a in boot.items()
                           for x,u,l in zip(a.x,a.umf,a.lmf)]),out/'codebooks/bootstrap_it2.csv')
    sensitivity=[]
    for scenario,models in [('bootstrap',boot),('inattention_excluded',fc)]:
        for term,a in models.items():
            info=model_metrics(a)|{'jaccard_primary_it2':jaccard(a,citizen[term])}
            if scenario=='inattention_excluded':
                vals=filtered.loc[filtered.term==term,'probability']
                q=np.quantile(vals,[.25,.75],method='linear')
                info.update(pooled_iqr=q[1]-q[0],pooled_centroid=centroid_t1(fp[term]),
                            jaccard_primary_pooled=jaccard(fp[term],pooled[term]),respondents_excluded=removed)
            sensitivity.extend(dict(scenario=scenario,metric=k,term=term,group='pooled',value=v) for k,v in info.items())
    for row in fmetrics.to_dict('records'):
        if row['metric'] in ['n','included','n_group_excluded']:
            sensitivity.append(dict(scenario='inattention_excluded',**row))
    sensitivity=pd.DataFrame(sensitivity);write_csv(sensitivity,out/'sensitivity.csv')
    figdir=out/'figures';figdir.mkdir(exist_ok=True);figures(pooled,citizen,metrics,figdir)
    ranking=sorted(citizen,key=lambda t:(-model_metrics(citizen[t])['fou_area'],t))
    n=metrics[(metrics.metric=='n') & metrics.group.ne('pooled')].groupby('group').value.agg(['min','max'])
    lines=['# RQ1: Citizen probability-word models','',
           f'Pooled: {data.response_id.nunique():,} respondents; {len(data):,} judgements; {len(pooled)} terms.',
           'Universe 0–100, step 0.5. NumPy linear quantiles (q05, q25, q75, q95), clipped after estimation.',
           'Primary IT2: pointwise max/min of eligible demographic T1 models. Centroids use EKM.',
           '', '## Group sizes', '', '| Group | Minimum n per term | Maximum n per term |', '|---|---:|---:|']
    lines += [f'| {g.replace(chr(124), chr(32)+chr(47)+chr(32))} | {int(r["min"])} | {int(r["max"])} |' for g,r in n.iterrows()]
    excluded=int(metrics[(metrics.metric=='n_group_excluded')].value.iloc[0])
    lines += ['',f'{excluded} respondents excluded from demographic groups only (missing metadata or under 18); retained in pooled models.',
              f'Group/term cells excluded for n < 30: {len(exclusions)}.', '', '## FOU ranking', '', '| Rank | Term | FOU area | EKM interval | Centre |','|---:|---|---:|---|---:|']
    for i,t in enumerate(ranking,1):
        v=model_metrics(citizen[t]);lines.append(f'| {i} | {t} | {v["fou_area"]:.6f} | [{v["centroid_left"]:.6f}, {v["centroid_right"]:.6f}] | {v["centre"]:.6f} |')
    lines += ['',f'Widest: {ranking[0]}; narrowest: {ranking[-1]}.', '', '## Group differences', '',
              f'{len(overlaps)} pairwise T1 Jaccards (15 per term when all six groups qualify).',
              '| Group pair | Mean Jaccard across terms |','|---|---:|']
    for (a,b),v in overlaps.groupby(['group_a','group_b']).jaccard.mean().items():lines.append(f'| {a.replace("|"," / ")} vs {b.replace("|"," / ")} | {v:.6f} |')
    lines += ['', '## Pairwise consistency', '',
              f'Of {consistency["n_pairs"]:,} pair rows: {consistency["consistent"]:,} strictly consistent ({consistency["consistent_share"]:.6%}), '
              f'{consistency["ties"]:,} tied ({consistency["tie_share"]:.6%}), and {consistency["inconsistent"]:,} inconsistent ({consistency["inconsistent_share"]:.6%}).',
              'Denominator includes all pair rows, including the repeated pair in each respondent session; ties are not counted as consistent.',
              '', '## Sensitivity', '',f'Bootstrap: {B} whole-respondent resamples with replacement, seed {SEED}; same sampled respondents across all terms.',
              f'Inattention rule excludes {removed} respondents (Almost No Chance >= Almost Certain), leaving {filtered.response_id.nunique()}.',
              f'Filtered group/term cells excluded for n < 30: {len(fexclusions)}.', '',
              '| Scenario | Mean FOU area | Mean Jaccard to primary IT2 |','|---|---:|---:|']
    for scenario,g in sensitivity.groupby('scenario'):
        lines.append(f'| {scenario} | {g.loc[g.metric=="fou_area","value"].mean():.6f} | {g.loc[g.metric=="jaccard_primary_it2","value"].mean():.6f} |')
    lines += ['', '## Deviations', '', 'None from the frozen RQ1 analysis plan. Missing/under-18 respondents remain in pooled and bootstrap samples; the inattention exclusion applies only to its sensitivity scenario.',
              'Implementation choices: strict-choice share uses all rows as denominator; whole-respondent bootstrap preserves cross-term dependence; group eligibility is reassessed after sensitivity exclusion.',
              'CSV files use 17 significant digits. Extra filtered and bootstrap codebooks are saved to support inspection; no individual-level records or identifiers are exported.', '']
    if exclusions or fexclusions:lines += [f'Observed group exclusions: primary={exclusions}; filtered={fexclusions}.']
    (out/'SUMMARY.md').write_text('\n'.join(lines),encoding='utf-8')
    print(f'Wrote {out}; groups excluded={len(exclusions)}; sensitivity respondents excluded={removed}')

if __name__=='__main__':main()
