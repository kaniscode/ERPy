"""Human-reference construction and participant-resampling contracts."""
import numpy as np
import pandas as pd
import pytest
from validation.score_n1_label_free import reference_from_raters, counts, cluster_draws, clean_json
from validation.summarize_n1_nominal_stage import nominal_screen


def test_reference_is_fractional_human_votes_and_ignores_algorithm_columns(tmp_path):
    rows=[]
    for rater,code in [('A',1),('B',0),('C',2),('D',np.nan)]:
        rows.append(dict(subject='UMCU20',stimpair='A-B',channel='C',rater=rater,
            configuration='early_10_90ms',raw_annotation=code,reference_positive=1,
            archived_detected=True,detected=True))
    path=tmp_path/'votes.csv';pd.DataFrame(rows).to_csv(path,index=False)
    result=reference_from_raters(path).iloc[0]
    assert result.reference_positive==pytest.approx(1/3)
    assert result.n_raters==3 and result.positive_votes==1
    np.testing.assert_allclose(counts([1/3,.5],[True,False]),[1/3,.5,.5,2/3,2,2])


def test_duplicate_human_rater_does_not_gain_weight(tmp_path):
    row=dict(subject='UMCU20',stimpair='A-B',channel='C',rater='A',
             configuration='early_10_90ms',raw_annotation=1)
    path=tmp_path/'votes.csv';pd.DataFrame([row,row]).to_csv(path,index=False)
    with pytest.raises(ValueError,match='Duplicate'):
        reference_from_raters(path)


def test_unknown_annotation_cannot_silently_become_missing(tmp_path):
    row=dict(subject='UMCU20',stimpair='A-B',channel='C',rater='A',
             configuration='early_10_90ms',raw_annotation=7)
    path=tmp_path/'votes.csv';pd.DataFrame([row]).to_csv(path,index=False)
    with pytest.raises(ValueError,match='Unknown'):
        reference_from_raters(path)


def test_participant_bootstrap_preserves_site_counts_and_shared_draws():
    subjects=['M1','M2','U1','U2','U3'];site={s:s[0] for s in subjects}
    draws=cluster_draws(subjects,site)
    np.testing.assert_array_equal(draws,cluster_draws(subjects,site))
    np.testing.assert_array_equal(draws[:,:2].sum(axis=1),np.full(2000,2))
    np.testing.assert_array_equal(draws[:,2:].sum(axis=1),np.full(2000,3))
    vector=np.arange(5)
    assert np.all(draws@vector-draws@vector==0)


def test_invalid_reference_probabilities_rejected():
    with pytest.raises(ValueError,match='Human reference'):
        counts([np.nan],[True])


def test_metric_scalar_array_and_undefined_value_serialization():
    assert clean_json({'value':np.array(.5),'undefined':np.array(np.nan)})=={'value':.5,'undefined':None}


def test_nominal_stage_uses_gate_and_joint_p_without_bh_selection():
    frame=pd.DataFrame({'available':[True,True,True,False],
        'morphology_gate':[True,True,False,True],'p_joint':[.05,.051,.001,.001],
        'q_screened':[.9,.9,.9,.9]})
    np.testing.assert_array_equal(nominal_screen(frame),[True,False,False,False])
