from dataclasses import replace

import numpy as np
import pytest

from ERPy.crp_energy import run_crp_energy_array
from ERPy.multiscale import (
    MultiScaleConfig, conservative_bonferroni, run_multiscale_array,
    run_multiscale_family, window_seed,
    conservative_weighted_bonferroni,
)


def fixture(n=12, channels=1):
    times = np.arange(-300, 161) / 500.0
    rng = np.random.default_rng(71)
    x = rng.normal(size=(n, len(times), channels))
    shape = 8*np.exp(-.5*((times-.04)/.012)**2)
    x[:, :, 0] += shape
    return x, times


def test_window_results_match_existing_core_with_identical_trials_and_seeds():
    x, times = fixture()
    config = MultiScaleConfig()
    result = run_multiscale_array(x[:, :, 0], times, channel="target", config=config)
    for item in result.windows:
        expected = run_crp_energy_array(x[:, :, 0], times, channel="target", config=replace(
            config.component_config, response_window=item.declared_window,
            random_state=window_seed(42, "target", item.declared_window),
        ))
        assert item.p_joint == expected.p_joint
        assert item.result.p_crp == expected.p_crp
        assert item.result.p_energy == expected.p_energy
        np.testing.assert_array_equal(item.result.clean_trial_indices, expected.clean_trial_indices)
    assert result.p_multiscale == min(1, 3*min(w.p_joint for w in result.windows))


@pytest.mark.parametrize("weights", [None, (.1,.1,.8)])
def test_whole_record_polarity_reflection_preserves_components_and_calls(weights):
    x, times = fixture(n=13, channels=2)
    # Thirteen trials exercises Monte Carlo projection with the same streams.
    config=MultiScaleConfig(weights=weights)
    positive = run_multiscale_family(x, times, channels=["A", "B"],config=config)
    negative = run_multiscale_family(-x, times, channels=["A", "B"],config=config)
    for first, second in zip(positive, negative):
        assert first.p_multiscale == second.p_multiscale
        assert first.q_multiscale == second.q_multiscale
        assert first.significant == second.significant
        for a, b in zip(first.windows, second.windows):
            assert a.result.p_crp == b.result.p_crp
            assert a.result.p_energy == b.result.p_energy


def test_unavailable_window_keeps_fixed_three_way_penalty():
    assert conservative_bonferroni([.01, np.nan, np.nan]) == .03
    x, times = fixture()
    short = times <= .1
    result = run_multiscale_array(x[:, short, 0], times[short])
    assert result.n_declared_windows == 3
    assert result.n_available_windows == 1
    assert [w.status for w in result.windows[1:]] == ["declared_window_unavailable"]*2
    assert result.p_multiscale == min(1, 3*result.windows[0].p_joint)
    assert all(w.p_for_combination == 1 for w in result.windows[1:])


def test_common_clean_trials_prevent_narrow_window_eligibility_selection():
    x, times = fixture(n=9)
    x[0, np.flatnonzero(times == .2)[0], 0] = np.nan
    result = run_multiscale_array(x[:, :, 0], times)
    np.testing.assert_array_equal(result.common_clean_trial_indices, np.arange(1, 9))
    assert all(w.result.n_trials_clean == 8 for w in result.windows)
    assert all(0 not in w.result.clean_trial_indices for w in result.windows)


def test_entirely_unavailable_contact_keeps_flag_and_conservative_family_entry():
    x, times = fixture(n=7, channels=2)
    results = run_multiscale_family(x, times, channels=["A", "B"])
    assert all(not r.available and not r.significant for r in results)
    assert all(r.p_multiscale == 1 and r.q_multiscale == 1 for r in results)
    assert all(r.n_available_windows == 0 for r in results)


def test_eight_trial_discrete_floor_blocks_isolated_four_contact_discovery():
    times = np.arange(-300, 161) / 500.
    x = np.zeros((8, len(times), 4))
    shape = 10*np.exp(-.5*((times-.04)/.012)**2)
    x[:, :, 0] = shape
    # Three entirely unavailable contacts deliberately contribute p=1. This
    # isolates the exact resolution, without a stochastic false-positive test.
    results = run_multiscale_family(x, times)
    assert results[0].p_multiscale == 3/128
    assert results[0].q_multiscale == 12/128
    assert not results[0].significant


def test_fixed_broad_weight_allows_the_isolated_eight_trial_discrete_floor():
    times=np.arange(-300,161)/500.
    x=np.zeros((8,len(times),4))
    x[:,:,0]=10*np.exp(-.5*((times-.04)/.012)**2)
    result=run_multiscale_family(x,times,config=MultiScaleConfig(weights=(.1,.1,.8)))[0]
    assert result.p_multiscale == 1/(128*.8)
    assert result.q_multiscale == 4/(128*.8)
    assert result.significant


def test_missing_windows_do_not_redistribute_fixed_weights():
    assert conservative_weighted_bonferroni([.01,np.nan,np.nan],[.1,.1,.8]) == pytest.approx(.1)
    assert conservative_weighted_bonferroni([np.nan,np.nan,.01],[.1,.1,.8]) == pytest.approx(.0125)
    x,t=fixture();short=t<=.1
    result=run_multiscale_array(x[:,short,0],t[short],config=MultiScaleConfig(weights=(.1,.1,.8)))
    assert result.n_available_windows == 1
    assert result.p_multiscale == min(1,result.windows[0].p_joint/.1)


def test_weighted_bank_permutation_preserves_allocations_and_seeds():
    x,t=fixture(n=13)
    config=MultiScaleConfig(weights=(.1,.1,.8))
    a=run_multiscale_array(x[:,:,0],t,config=config)
    b=run_multiscale_array(x[:,:,0],t,config=replace(config,windows=config.windows[::-1],weights=config.weights[::-1]))
    assert a.p_multiscale == b.p_multiscale
    assert {w.declared_window:w.seed for w in a.windows} == {w.declared_window:w.seed for w in b.windows}


@pytest.mark.parametrize("weights", [(), (.5,.5), (0,.2,.8), (-.1,.3,.8), (.1,.1,.7), (np.nan,.1,.9), (np.inf,.1,.9), ((.1,.1,.8),)])
def test_invalid_weight_allocation_rejected(weights):
    with pytest.raises(ValueError):MultiScaleConfig(weights=weights).validate()


def test_bank_and_contact_order_do_not_change_random_streams_or_adjustment():
    x, times = fixture(n=13, channels=2)
    c = MultiScaleConfig()
    forward = run_multiscale_family(x, times, channels=["A", "B"], config=c)
    reverse = run_multiscale_family(x[:, :, ::-1], times, channels=["B", "A"],
                                   config=replace(c, windows=c.windows[::-1]))
    for a, b in zip(forward, reverse[::-1]):
        assert a.p_multiscale == b.p_multiscale
        assert a.q_multiscale == b.q_multiscale
        assert {w.declared_window:w.seed for w in a.windows} == {w.declared_window:w.seed for w in b.windows}


@pytest.mark.parametrize("bad", [[], [.1, -.1], [1.1, .1], [[.1]], [float("inf"), -.1]])
def test_invalid_combination_rejected(bad):
    with pytest.raises(ValueError):
        conservative_bonferroni(bad)


def test_degenerate_responses_are_unavailable_not_false_precision():
    times=np.arange(-300,161)/500.
    r=run_multiscale_array(np.zeros((12,len(times))),times)
    assert r.p_multiscale == 1 and not r.available
    assert all(w.status == "reproducibility_unavailable" for w in r.windows)


@pytest.mark.parametrize("windows", [(), ((.01,.09),(.01,.09)), ((.3,.1),), ((float("nan"),.3),)])
def test_invalid_window_banks_rejected(windows):
    with pytest.raises(ValueError):MultiScaleConfig(windows=windows).validate()


def test_family_api_rejects_duplicate_labels_and_invalid_time_grid():
    x,t=fixture(channels=2)
    with pytest.raises(ValueError):run_multiscale_family(x,t,channels=["A","A"])
    t[20]+=.0001
    with pytest.raises(ValueError):run_multiscale_family(x,t)


def test_predeclared_stress_energy_null_has_joint_window_sign_symmetry():
    from validation.benchmark_multiscale import TIMES, simulate, specs_primary
    specs=specs_primary()
    assert len(specs)==2000
    spec=next(s for s in specs if s['scenario']=='projection_alt_energy_null')
    values,_,_=simulate(spec)
    target=values[:,:,0]
    bi=np.flatnonzero(TIMES==-.1)[0];ri=np.flatnonzero(TIMES==.1)[0]
    a=target[:,ri]/(TIMES[ri]*500);b=target[:,bi]/(TIMES[bi]*500)
    swapped=np.where(TIMES[None,:]<0,a[:,None]*TIMES*500,b[:,None]*TIMES*500)
    for start,stop in MultiScaleConfig().windows:
        r=np.flatnonzero((TIMES>=start)&(TIMES<=stop));base=np.flatnonzero((TIMES>=-.5)&(TIMES<=-.02))[-len(r):]
        def differences(x):
            response=x[:,r]-x[:,r].mean(axis=1,keepdims=True)
            baseline=x[:,base]-x[:,base].mean(axis=1,keepdims=True)
            return np.log(np.sqrt(np.mean(response**2,axis=1))+1e-12)-np.log(np.sqrt(np.mean(baseline**2,axis=1))+1e-12)
        np.testing.assert_allclose(differences(target),-differences(swapped),atol=1e-14,rtol=0)
