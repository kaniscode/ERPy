"""Behavioral checks for a fixed gate and original complete-family inference."""
import numpy as np
import pytest

from ERPy.n1 import (extract_n1_features, N1FeatureConfig, N1_FEATURE_VERSION,
                     feature_config_sha256)
from ERPy.n1_detection import morphology_gate, _apply_complete_family, detect_n1_family


def descriptor(amplitude=170.,noise=20.,present=True,available=True):
    return dict(feature_available=available,negative_peak_present=present,
                negative_peak_uv=amplitude,baseline_std_uv=noise,
                voltage_unit='uV',feature_version=N1_FEATURE_VERSION,
                feature_config_sha256=feature_config_sha256(N1FeatureConfig()))


@pytest.mark.parametrize('noise,threshold',[(20.,170.),(50.,170.),(100.,340.)])
def test_fixed_floor_and_high_noise_boundary(noise,threshold):
    assert morphology_gate(descriptor(threshold,noise))['passed']
    assert not morphology_gate(descriptor(np.nextafter(threshold,0),noise))['passed']
    assert morphology_gate(descriptor(threshold,noise))['threshold_uv']==threshold


def test_baseline_only_ablation_is_prespecified_and_distinct():
    features=descriptor(100.,20.)
    assert not morphology_gate(features)['passed']
    assert morphology_gate(features,baseline_sd_floor_uv=0)['passed']
    with pytest.raises(ValueError,match='fixed'):
        morphology_gate(features,baseline_sd_floor_uv=10)


@pytest.mark.parametrize('field,value', [('voltage_unit','V'),
    ('feature_version','unknown'),('feature_config_sha256','broad_window')])
def test_standalone_gate_rejects_feature_contract_mismatch(field,value):
    with pytest.raises(ValueError,match='contract'):
        morphology_gate(dict(descriptor(),**{field:value}))


def test_complete_family_keeps_gate_failures_and_unavailable_morphology():
    # A selected-only analysis would reject the first contact (.03); the
    # declared three-contact family correctly gives .09 and no call.
    records=[dict(channel='A',p_joint=.03,features=descriptor(250.)),
             dict(channel='B',p_joint=.001,features=descriptor(10.)),
             dict(channel='C',p_joint=.002,features=descriptor(available=False)),
             dict(channel='D',p_joint=np.nan,features=descriptor(250.))]
    rows=_apply_complete_family(records)
    assert rows[0]['q_screened']==pytest.approx(.09)
    assert not any(row['detected'] for row in rows)
    assert rows[1]['p_screened']==rows[2]['p_screened']==1
    assert rows[2]['decision_stage']=='morphology_unavailable'
    assert rows[2]['family_size']==3 and not rows[2]['available']
    assert np.isnan(rows[3]['p_screened']) and rows[3]['decision_stage']=='inference_unavailable'


def test_no_peak_is_observed_gate_failure_with_p_one():
    row=_apply_complete_family([dict(p_joint=.001,features=descriptor(250.,present=False))])[0]
    assert row['available'] and row['p_screened']==1 and not row['detected']
    assert row['decision_stage']=='no_negative_interior_peak'


def test_screening_never_improves_original_p_or_q():
    rng=np.random.default_rng(76)
    p=np.r_[rng.uniform(0,.05,100),[np.nan,.001]]
    rows=_apply_complete_family([dict(p_joint=value,features=descriptor(
        250. if i%3==0 else 50.)) for i,value in enumerate(p)])
    for row in rows:
        if np.isfinite(row['p_joint']):
            assert row['p_screened']>=row['p_joint']
            assert row['q_screened']+1e-14>=row['original_q_joint']
            assert not row['detected'] or row['original_detected']


def test_strict_response_boundary_does_not_become_interior_peak():
    t=np.arange(-500,176)/500
    rng=np.random.default_rng(11)
    x=rng.normal(0,1,(8,len(t)))
    mask=(t>=.010)&(t<.090)
    x[:,mask]=-250+np.arange(mask.sum())*3
    features=extract_n1_features(x,t)
    assert features['feature_available'] and not features['negative_peak_present']
    assert not morphology_gate(features)['passed']


def test_array_api_binds_shared_source_and_separate_masks_and_excludes_stimulation():
    t=np.arange(-500,176)/500
    rng=np.random.default_rng(15)
    x=rng.normal(0,4,(10,len(t),3))
    x[:,:,0]-=250*np.exp(-.5*((t-.05)/.012)**2)
    # Sample lies in full morphology baseline but not latest matched inference
    # baseline; both masks are valid and need not coincide.
    x[0,np.argmin(abs(t+.9)),0]=np.nan
    result=detect_n1_family(x,t,['A','B','STIM'],stimulation_contacts=['STIM'],
                            random_seeds=[1,2,3],family_id='one_pair')
    assert [r['channel'] for r in result['contacts']]==['A','B']
    row=result['contacts'][0]
    assert len(row['provenance']['feature_clean_trial_indices'])==9
    assert len(row['provenance']['inference_clean_trial_indices'])==10
    assert row['provenance']['inference_config']['response_window']==(.01,.09)
    assert row['p_joint']==max(row['p_reproducibility'],row['p_energy'])
    assert row['features']['negative_peak_present'] and row['morphology_gate']
    changed=detect_n1_family(x*2,t,['A','B','STIM'],stimulation_contacts=['STIM'],
                             family_id='one_pair')
    assert row['provenance']['family_input_sha256']!=changed['contacts'][0]['provenance']['family_input_sha256']


@pytest.mark.parametrize('kwargs,match',[
    ({'voltage_unit':'V'},'microvolts'),
    ({'random_seeds':[1]},'one exact'),
    ({'stimulation_contacts':['missing']},'Stimulation'),
    ({'family_id':''},'family_id'),
])
def test_array_input_contract_rejects_mismatches(kwargs,match):
    t=np.arange(-500,176)/500;x=np.ones((8,len(t),2))
    args=dict(family_id='pair');args.update(kwargs)
    with pytest.raises(ValueError,match=match):
        detect_n1_family(x,t,['A','B'],**args)


def test_precomputed_broad_dictionary_cannot_enter_array_api():
    with pytest.raises((TypeError,ValueError)):
        detect_n1_family({'p_reproducibility':.01,'configuration':'broad_15_300ms'},
                         np.arange(10),['A'],family_id='pair')
