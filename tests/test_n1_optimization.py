"""Check selection constraints, paired counts and participant isolation."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'validation'))
from optimize_n1 import choose_candidate, choose_threshold, confusion, fit_model, join_features


def test_joint_feasibility_precedes_high_precision_with_lost_sensitivity():
    y=np.r_[np.ones(10),np.zeros(90)]
    archive=np.zeros(100,bool);archive[:7]=1;archive[10:12]=1
    calls=np.zeros((100,3),bool)
    calls[:2,0]=1  # Perfect PPV, unacceptably low sensitivity.
    calls[:7,1]=1;calls[10,1]=1  # Meets both targets.
    calls[:9,2]=1;calls[10:20,2]=1  # Loses specificity.
    i,detail=choose_candidate(y,calls,archive)
    assert i==1 and detail['policy']=='both_constraints_met'


def test_threshold_ties_preserve_entire_score_group_and_missing_no_calls():
    y=np.array([1,1,0,0,1.])
    score=np.array([.9,.8,.8,.1,np.nan])
    archive=np.array([1,1,0,0,0],bool)
    threshold,detail=choose_threshold(y,score,archive)
    assert threshold==.8
    assert detail['policy']=='specificity_constraint_infeasible'
    call=np.isfinite(score)&(score>=threshold)
    assert confusion(y,call).tolist()==[2.,1.,1.,1.,5.,5.]


def test_soft_votes_have_record_weight_not_rater_multiplicity():
    np.testing.assert_allclose(confusion([.25,.5,1],[1,0,1]),[1.25,.5,.5,.75,3,3])


def test_training_transform_and_coefficients_ignore_heldout_values_and_labels():
    rng=np.random.default_rng(3);x=rng.normal(size=(60,12));y=(x[:,0]>0).astype(float)
    train=np.arange(60)<40;available=np.ones(60,bool)
    a=fit_model(x,y,available,train,False,{'test':'isolation'})
    altered=x.copy();altered[~train]=1e8;labels=y.copy();labels[~train]=1-y[~train]
    b=fit_model(altered,labels,available,train,False,{'test':'isolation'})
    np.testing.assert_array_equal(a.mean,b.mean)
    np.testing.assert_array_equal(a.scale,b.scale)
    np.testing.assert_array_equal(a.coefficients,b.coefficients)
    assert a.intercept==b.intercept


def test_feature_join_matches_frozen_undirected_pair_identity():
    records=pd.DataFrame([dict(subject='MAYO02',stimpair='LG10-LG9',channel='LG3')])
    features=pd.DataFrame([dict(subject='MAYO02',stimpair='LG9-LG10',channel='LG3',feature_available=True)])
    joined=join_features(records,features)
    assert len(joined)==1 and joined.feature_available.iloc[0]
