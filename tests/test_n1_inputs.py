"""Prevent trained early-N1 prediction from silently accepting other evidence."""
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ERPy.crp_energy import run_crp_energy_array
from ERPy.n1 import (INFERENCE_FEATURES, MORPHOLOGY_FEATURES, N1MorphologyModel,
                     extract_n1_features, n1_feature_vector, n1_inference_config,
                     prepare_n1_hybrid_inputs)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks/examples"))
from _n1_inputs import FrozenN1Inputs

ROOT = Path(__file__).resolve().parents[1]


def arrays():
    t = np.arange(-1000, 351)/1000
    rng = np.random.default_rng(103)
    x = 3*np.sin(2*np.pi*7*t) - 80*np.exp(-.5*((t-.04)/.009)**2)
    return x + rng.normal(0, 2, (10, len(t))), t


def model():
    names = MORPHOLOGY_FEATURES + INFERENCE_FEATURES
    n = len(names)
    return N1MorphologyModel(names, (0.,)*n, (1.,)*n, (.1,)*n, -.7, .6, {'status':'test'})


def test_checked_array_path_matches_original_features_and_inference(tmp_path):
    x,t = arrays()
    features, evidence = prepare_n1_hybrid_inputs(x,t,random_state=72)
    np.testing.assert_array_equal(n1_feature_vector(features),n1_feature_vector(extract_n1_features(x,t)))
    normalization = x - np.median(x[:,500:980],axis=1,keepdims=True)
    original = run_crp_energy_array(normalization,t,config=n1_inference_config(72))
    assert evidence.p_reproducibility == original.p_crp
    assert evidence.p_energy == original.p_energy
    assert evidence.rms_ratio_db == original.rms_ratio_db
    vector = n1_feature_vector(features,evidence.validated_values(features))
    expected = float(np.exp(-np.logaddexp(0,-(.1*vector.sum()-.7))))
    result = model().predict(features,evidence)
    assert result['score'] == pytest.approx(expected,abs=1e-15)
    p=tmp_path/'model.json'; model().save(p)
    assert N1MorphologyModel.load(p).predict(features,evidence)==result


@pytest.mark.parametrize('field,value',[
    ('response_window',[.015,.3]), ('artifact_interval',[0.,.015]),
    ('baseline_window',[-.5,-.02]), ('n_permutations',999),
    ('max_exact_reproducibility_trials',16), ('min_clean_trials',4),
])
def test_rejects_nontrained_inference_settings(field,value):
    features,evidence=prepare_n1_hybrid_inputs(*arrays())
    config=json.loads(evidence.configuration_json);config[field]=value
    with pytest.raises(ValueError,match='configuration'):
        replace(evidence,configuration_json=json.dumps(config))


def test_plain_numeric_inference_dictionary_is_rejected():
    features,evidence=prepare_n1_hybrid_inputs(*arrays())
    with pytest.raises(ValueError,match='checked early-window'):
        model().predict(features,evidence.validated_values(features))


def test_numpy_integer_seed_retains_the_same_checked_prediction():
    a=prepare_n1_hybrid_inputs(*arrays(),random_state=42)
    b=prepare_n1_hybrid_inputs(*arrays(),random_state=np.int64(42))
    assert model().predict(*a)==model().predict(*b)
    assert b[1].configuration_json==a[1].configuration_json


def test_checked_array_path_preserves_both_monte_carlo_streams():
    x,t=arrays();x=np.vstack([x,x[:7]+.5])
    features,evidence=prepare_n1_hybrid_inputs(x,t,random_state=908)
    normalized=x-np.median(x[:,500:980],axis=1,keepdims=True)
    result=run_crp_energy_array(normalized,t,config=n1_inference_config(908))
    assert not result.reproducibility_test_exact and not result.energy_test_exact
    assert evidence.p_reproducibility==result.p_crp
    assert evidence.p_energy==result.p_energy
    assert json.loads(evidence.configuration_json)['random_state']==908
    assert model().predict(features,evidence)['available']


def test_separate_declared_clean_masks_are_preserved_and_allowed():
    x,t=arrays();x[0,np.argmin(abs(t+.8))]=np.nan
    features,evidence=prepare_n1_hybrid_inputs(x,t)
    p=json.loads(evidence.provenance_json)
    assert p['feature_clean_trial_indices']==list(range(1,10))
    assert p['inference_clean_trial_indices']==list(range(10))
    assert model().predict(features,evidence)['available']


def test_inclusive_inference_endpoint_can_drop_more_trials_than_morphology():
    x,t=arrays();x[:3,np.argmin(abs(t-.09))]=np.nan
    assert extract_n1_features(x,t)['n_feature_trials']==10
    with pytest.raises(ValueError,match='early-window inference is unavailable'):
        prepare_n1_hybrid_inputs(x,t)


def test_changed_source_or_trial_order_rejects_cross_binding():
    x,t=arrays(); features,evidence=prepare_n1_hybrid_inputs(x,t)
    for changed in (x[::-1],x.copy()):
        changed=changed.copy();changed[0,-1]+=1
        other,_=prepare_n1_hybrid_inputs(changed,t)
        with pytest.raises(ValueError,match='identity mismatch'):
            model().predict(other,evidence)
    with pytest.raises(ValueError,match='identity mismatch'):
        model().predict(dict(features,negative_peak_z=features['negative_peak_z']+1),evidence)


@pytest.fixture(scope='module')
def frozen():
    return FrozenN1Inputs()


def test_verified_frozen_example_matches_saved_score_without_inventing_trials(frozen):
    key=('MAYO02','LG1-LG2','LG4')
    features,evidence=frozen.prepare(frozen.features.loc[key],frozen.inference.loc[key])
    model_path=ROOT/'validation/frozen_results/n1/nested/N1_logistic_hybrid_heldout_MAYO02.json'
    result=N1MorphologyModel.load(model_path).predict(features,evidence)
    assert result['score']==pytest.approx(.6766222915777962,abs=1e-12)
    assert result['detected']
    p=json.loads(evidence.provenance_json)
    assert p['feature_clean_trial_indices'] is None and p['inference_clean_trial_indices'] is None
    assert 'not archived' in p['trial_provenance_scope']


def test_broad_window_swap_is_rejected_even_if_relabelled_early(frozen):
    key=('MAYO02','LG1-LG2','LG4')
    records=pd.read_csv(ROOT/'validation/frozen_results/erdetect/record_predictions_and_labels.csv.gz',low_memory=False,
                        dtype={'random_seed':'string'})
    broad=records.loc[(records.subject=='MAYO02') & (records.stimpair=='LG1-LG2') &
                      (records.channel=='LG4') & (records.configuration=='broad_15_300ms')].iloc[0].to_dict()
    with pytest.raises(ValueError,match='early_10_90ms'):
        frozen.prepare(frozen.features.loc[key],broad)
    broad['configuration']='early_10_90ms'
    with pytest.raises(ValueError,match='verified early inference record'):
        frozen.prepare(frozen.features.loc[key],broad)


@pytest.mark.parametrize('side,field,value',[
    ('feature','epoch_sha256','a'*64),('inference','epoch_sha256','b'*64),
    ('inference','channel','LG5'),('inference','n_trials_clean',9),
    ('inference','random_seed',1),('feature','negative_peak_z',777.),
])
def test_frozen_input_provenance_and_value_mismatch_rejected(frozen,side,field,value):
    key=('MAYO02','LG1-LG2','LG4')
    f=frozen.features.loc[key].to_dict();i=frozen.inference.loc[key].to_dict()
    (f if side=='feature' else i)[field]=value
    with pytest.raises(ValueError): frozen.prepare(f,i)


def test_frozen_random_seed_is_preserved_as_an_exact_integer(frozen):
    key=('MAYO02','LG1-LG2','LG4')
    f=frozen.features.loc[key].to_dict();i=frozen.inference.loc[key].to_dict()
    _,evidence=frozen.prepare(f,i)
    assert json.loads(evidence.configuration_json)['random_state']==1453175015351346450
    i['random_seed']=float(i['random_seed'])
    with pytest.raises(ValueError,match='exact frozen seed'):
        frozen.prepare(f,i)
