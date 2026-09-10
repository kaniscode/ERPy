"""Physiological and numerical checks for descriptive N1 features."""
import numpy as np
import pytest

from ERPy.n1 import (N1FeatureConfig, extract_n1_features, n1_feature_vector,
                     N1MorphologyModel, MORPHOLOGY_FEATURES)


def waveform(polarity=-1, seed=2):
    t = np.arange(-1000, 351)/1000
    rng = np.random.default_rng(seed)
    baseline = 4*np.sin(2*np.pi*7*t)
    response = 120*np.exp(-.5*((t-.04)/.008)**2)
    x = baseline + polarity*response + rng.normal(0, 2, (12,len(t)))
    return x,t


def test_negative_peak_recovers_latency_and_trial_consistency():
    x,t = waveform()
    r = extract_n1_features(x,t)
    assert r['feature_available'] and r['negative_peak_present']
    assert 37 <= r['peak_latency_ms'] <= 43
    assert r['negative_peak_uv'] > 110
    assert r['negative_peak_z'] > 20
    assert r['trial_negative_fraction'] == 1
    assert 12 < r['peak_width_ms'] < 24


def test_positive_response_not_mislabeled_as_large_negative_peak():
    x,t = waveform(polarity=1)
    r = extract_n1_features(x,t)
    assert r['feature_available']
    assert r['negative_peak_z'] < 3.4
    assert r['negative_area_fraction'] < .1


def test_features_invariant_to_trial_offsets_and_dimensionless_scale():
    x,t = waveform()
    a = extract_n1_features(x,t)
    b = extract_n1_features(x + np.arange(len(x))[:,None]*137,t)
    c = extract_n1_features(x*7,t)
    for key in ['negative_peak_z','negative_prominence_z','peak_width_ms',
                'negative_area_fraction','trial_negative_fraction','trial_cosine']:
        assert a[key] == pytest.approx(b[key],abs=1e-10)
        assert a[key] == pytest.approx(c[key],abs=1e-10)
    assert c['negative_peak_uv'] == pytest.approx(7*a['negative_peak_uv'])


def test_missing_trials_and_degenerate_baseline_are_unavailable():
    x,t = waveform()
    x[:5,t>0] = np.nan
    assert not extract_n1_features(x,t)['feature_available']
    assert extract_n1_features(np.zeros((12,len(t))),t)['feature_reason'] == 'degenerate_mean_baseline'


def test_absent_interior_peak_is_observed_not_missing():
    x,t = waveform()
    x[:,t>0] = 10 + 100*t[t>0]
    r = extract_n1_features(x,t)
    assert r['feature_available']
    assert not r['negative_peak_present']
    assert r['feature_reason'] == 'no_negative_interior_peak'


def test_rejects_unavailable_windows_and_irregular_time_grid():
    x,t = waveform()
    with pytest.raises(ValueError,match='unavailable'):
        extract_n1_features(x,t,N1FeatureConfig(baseline_window=(-2,-1)))
    t[300]+=.0001
    with pytest.raises(ValueError,match='uniformly'):
        extract_n1_features(x,t)


def test_values_outside_required_windows_do_not_change_features():
    x,t=waveform();a=extract_n1_features(x,t)
    x[:,t>.15]=np.inf
    b=extract_n1_features(x,t)
    np.testing.assert_allclose(n1_feature_vector(a),n1_feature_vector(b))


def test_portable_model_rejects_wrong_contract_and_invalid_evidence(tmp_path):
    x,t=waveform();features=extract_n1_features(x,t)
    n=len(MORPHOLOGY_FEATURES)
    model=N1MorphologyModel(MORPHOLOGY_FEATURES,(0.,)*n,(1.,)*n,(0.,)*n,0.,.6,
                             {'status':'test'})
    path=tmp_path/'model.json';model.save(path)
    assert N1MorphologyModel.load(path).predict(features)['score']==.5
    assert not model.predict(features)['detected']
    for key,value in [('feature_version','old'),('voltage_unit','V'),('feature_config_sha256','unknown')]:
        with pytest.raises(ValueError,match='contract'):
            model.predict(dict(features,**{key:value}))
    for value in [-.1,np.inf,np.nan,1.1]:
        with pytest.raises(ValueError,match='p_reproducibility'):
            n1_feature_vector(features,dict(p_reproducibility=value,p_energy=.01,rms_ratio_db=2.))
    with pytest.raises(ValueError,match='negative_peak_z'):
        n1_feature_vector(dict(features,negative_peak_z=np.inf))


def test_trial_neighborhood_is_consistent_on_shifted_sample_grid():
    # The ±2ms neighborhood includes both boundary samples, independent of
    # floating representation of the peak time.
    x,t=waveform();x[:]=4*np.sin(2*np.pi*7*t)
    x[:,t>0]=0
    at=lambda ms:np.argmin(abs(t-ms/1000))
    for center in [30,40]:
        y=x.copy();y[:,at(center)]=-100
        y[:,at(center-2)]=200
        y[:,at(center+2)]=200
        r=extract_n1_features(y,t)
        assert r['trial_negative_fraction']==0


@pytest.mark.parametrize('state', ['peak', 'no_peak', 'unavailable'])
def test_exported_timing_distinguishes_measurement_from_model_imputation(state):
    x,t=waveform()
    if state=='no_peak':
        x[:,t>0]=10+100*t[t>0]
    elif state=='unavailable':
        x[:5,t>0]=np.nan
    features=extract_n1_features(x,t)
    n=len(MORPHOLOGY_FEATURES)
    model=N1MorphologyModel(MORPHOLOGY_FEATURES,(0.,)*n,(1.,)*n,(.1,)*n,0.,.6,
                           {'status':'test'})
    prediction=model.predict(features)
    assert features['peak_latency_imputed']==(state=='no_peak')
    assert prediction['peak_latency_imputed']==(state=='no_peak')
    if state=='peak':
        assert features['measured_peak_latency_ms']==features['peak_latency_ms']
        assert prediction['measured_peak_latency_ms']==features['peak_latency_ms']
    else:
        assert features['measured_peak_latency_ms'] is None
        assert prediction['measured_peak_latency_ms'] is None
    if state=='no_peak':
        assert features['peak_latency_ms']==pytest.approx(50)
    # Additional reporting metadata must not change vectors, scores or calls
    # when reading original feature exports with the original presence flag.
    legacy={k:v for k,v in features.items() if k not in
            {'measured_peak_latency_ms','peak_latency_imputed'}}
    assert model.predict(legacy)==prediction
    if features['feature_available']:
        np.testing.assert_array_equal(n1_feature_vector(legacy),n1_feature_vector(features))
