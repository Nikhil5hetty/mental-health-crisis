"""Synthetic-only tests for frozen citizen-word models."""
import numpy as np
import pandas as pd
import pytest
from src.cww.quantile import X,quantile_trapezoid,envelope,group_models
from src.cww.sets import trapezoid
from src.data.capphrase import exclude_inattentive
from experiments.words.run_rq1 import pair_consistency,bootstrap_models


def test_known_quantiles():
    model=quantile_trapezoid(np.arange(101))
    expected=trapezoid(5,25,75,95,x=X)
    np.testing.assert_array_equal(model.mu,expected.mu)
    assert len(model.x)==201 and model.mu[100]==1


def test_linear_interpolation_and_clipping():
    a=quantile_trapezoid([0,100])
    np.testing.assert_array_equal(a.mu,trapezoid(5,25,75,95,x=X).mu)
    a=quantile_trapezoid([-100,200])
    q=np.clip(np.quantile([-100,200],[.05,.25,.75,.95],method='linear'),0,100)
    np.testing.assert_array_equal(a.mu,trapezoid(*q,x=X).mu)


@pytest.mark.parametrize('values',[[],[np.nan],[np.inf],[[1,2]]])
def test_bad_samples_rejected(values):
    with pytest.raises(ValueError):quantile_trapezoid(values)


def test_constant_and_envelope():
    a=quantile_trapezoid([50]*30);b=quantile_trapezoid([75]*30)
    model=envelope([a,b])
    assert np.all(model.umf>=model.lmf)
    np.testing.assert_array_equal(model.umf,np.maximum(a.mu,b.mu))
    np.testing.assert_array_equal(model.lmf,np.minimum(a.mu,b.mu))
    with pytest.raises(ValueError):envelope([])


def test_group_threshold_inclusive_and_counts_retained():
    models,counts=group_models({'too_small':[20]*29,'eligible':[20]*30,'empty':[]})
    assert list(models)==['eligible']
    assert counts=={'too_small':29,'eligible':30,'empty':0}


def test_pair_consistency_ties_and_strictness():
    data=pd.DataFrame({'response_id':[1,1,2,2,3,3],'term':['a','b']*3,
                       'probability':[80,20,40,40,10,90]})
    pairs=pd.DataFrame({'response_id':[1,2,3],'term1':['a']*3,'term2':['b']*3,'selected':['a']*3})
    result=pair_consistency(data,pairs)
    assert result['consistent']==result['ties']==result['inconsistent']==1
    assert result['consistent_share']==result['tie_share']==1/3
    pairs.loc[0,'selected']='unknown'
    with pytest.raises(ValueError):pair_consistency(data,pairs)


def test_inattention_excludes_whole_respondents_without_mutation():
    data=pd.DataFrame({'response_id':[1,1,2,2,3,3],
                       'term':['Almost No Chance','Almost Certain']*3,
                       'probability':[0,100,50,50,90,10]})
    original=data.copy(deep=True)
    kept,count=exclude_inattentive(data)
    assert count==2 and set(kept.response_id)=={1}
    pd.testing.assert_frame_equal(data,original)


def test_bootstrap_is_reproducible_and_bounds_ordered():
    data=pd.DataFrame({'response_id':np.repeat(np.arange(10),2),
                       'term':['a','b']*10,'probability':np.arange(20)*5})
    first=bootstrap_models(data);second=bootstrap_models(data)
    for term in first:
        np.testing.assert_array_equal(first[term].umf,second[term].umf)
        np.testing.assert_array_equal(first[term].lmf,second[term].lmf)
        assert (first[term].umf>=first[term].lmf).all()
