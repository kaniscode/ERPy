"""Equation-level regressions for the detector's inference contract."""

from itertools import product

import numpy as np
import pytest

from ERPy.crp import (
    CRPConfig,
    _mean_projection_profile,
    _projection_sample_counts,
    run_crp_array,
    semi_normalized_cross_projections,
)
from ERPy.crp_energy import (
    CRPEnergyConfig,
    _canonical_energy_effects,
    fixed_window_reproducibility_sign_flip_test,
    paired_log_rms_sign_flip_test,
    run_crp_energy_array,
)
from ERPy.inference import exact_sign_flip_test, fdr_bh


@pytest.mark.parametrize("alternative, sign", [("greater", 1), ("less", -1), ("two-sided", 1)])
def test_generic_monte_carlo_counts_observed_assignment(alternative, sign):
    result = exact_sign_flip_test(
        sign * np.ones(21),
        alternative=alternative,
        n_resamples=9,
        random_state=3,
    )
    assert not result.exact
    assert result.n_permutations == 9
    assert result.p_value == 1 / 10


def test_generic_exact_and_monte_carlo_keep_all_ties():
    for exact_limit in (0, 8):
        result = exact_sign_flip_test(
            np.zeros(8), max_exact_observations=exact_limit, n_resamples=9
        )
        assert result.p_value == 1.0
    exact = exact_sign_flip_test(np.ones(8), alternative="greater")
    assert exact.exact
    assert exact.p_value == 1 / 256


@pytest.mark.parametrize("scale", [1.0, 1e-8])
def test_duration_profile_equals_mean_of_ordered_projections(scale):
    voltage = scale * np.array([[1., -1., 2.], [2., 2., -1.], [1., 3., 4.]])
    counts = np.array([2, 3])
    profile = _mean_projection_profile(voltage, counts, 250.0, 1e-12)
    expected = [
        np.mean(semi_normalized_cross_projections(voltage[:count], sfreq=250.0))
        for count in counts
    ]
    np.testing.assert_allclose(profile, expected, rtol=1e-14, atol=0.0)


def test_fixed_window_test_survives_degenerate_descriptive_duration():
    # A zero early prefix wins the descriptive profile over negative later
    # projections. The full response still defines a valid, non-significant
    # randomization test and must remain available to the adjustment family.
    times = np.arange(-16, 17) / 1000.0
    values = np.zeros((8, len(times)))
    response_indices = np.flatnonzero(times >= 0.001)
    values[:, response_indices[10:]] = np.r_[np.ones(4), -np.ones(4)][:, None]
    result = run_crp_energy_array(
        values,
        times,
        config=CRPEnergyConfig(
            response_window=(0.001, 0.016),
            baseline_window=(-0.016, -0.001),
            artifact_interval=(0.0, 0.0),
        ),
    )
    assert result.p_crp == 1.0
    assert result.p_joint == 1.0
    assert result.qc_status == "pass"
    assert result.classification == "energetic_inconsistent"
    assert not result.significant
    assert np.isnan(result.response_duration)
    assert result.canonical_waveform.size == 0
    assert "canonical response is degenerate" in result.notes
    adjusted, _ = fdr_bh([0.02, result.p_joint])
    assert adjusted[0] == pytest.approx(0.04)


def test_undefined_projection_is_insufficient_not_a_negative_component():
    times = np.arange(-16, 17) / 1000.0
    result = run_crp_energy_array(
        np.zeros((8, len(times))),
        times,
        config=CRPEnergyConfig(
            response_window=(0.001, 0.016),
            baseline_window=(-0.016, -0.001),
            artifact_interval=(0.0, 0.0),
        ),
    )
    assert np.isnan(result.p_crp)
    assert np.isnan(result.p_joint)
    assert result.classification == "insufficient_data"
    assert result.qc_status == "reproducibility_unavailable"
    assert not result.significant


def test_zero_training_direction_has_no_canonical_energy_estimate():
    response = np.array([[1., -1.], [0., 0.], [0., 0.]])
    energy, fraction = _canonical_energy_effects(
        response, response_samples=2, cross_validated=True, eps=1e-12
    )
    assert np.isnan(energy)
    assert np.isnan(fraction)


def test_canonical_energy_matches_held_out_projection_equation():
    response = np.random.default_rng(17).normal(size=(8, 12))
    held_out_energies = []
    for i in range(len(response)):
        training = np.delete(response, i, axis=0).T
        canonical = np.linalg.svd(training, full_matrices=False)[0][:, 0]
        held_out_energies.append((response[i] @ canonical) ** 2)
    energy, fraction = _canonical_energy_effects(
        response, response_samples=12, cross_validated=True, eps=1e-12
    )
    assert energy == pytest.approx(sum(held_out_energies) / response.size)
    assert fraction == pytest.approx(sum(held_out_energies) / np.sum(response ** 2))


def test_crp_snr_and_scaled_coefficient_preserve_square_roots():
    times = np.arange(-32, 33) / 1000.0
    values = np.random.default_rng(2).normal(size=(8, len(times)))
    config = CRPConfig(response_window=(0.001, 0.032), baseline_window=(-0.032, -0.001))
    result = run_crp_array(values, times, config=config)
    baseline = (times >= config.baseline_window[0]) & (times <= config.baseline_window[1])
    response = (times >= config.response_window[0]) & (times <= config.response_window[1])
    centered = values - values[:, baseline].mean(axis=1, keepdims=True)
    selected = centered[:, response][:, :len(result.canonical_waveform)]
    residual = selected - np.outer(result.projections, result.canonical_waveform)
    expected_snr = result.projections / np.sqrt(np.maximum(np.sum(residual ** 2, axis=1), config.eps))
    np.testing.assert_allclose(result.snr, expected_snr)
    np.testing.assert_allclose(result.alpha_prime_uv, result.projections / np.sqrt(selected.shape[1]))


def test_primary_exact_energy_agrees_with_complete_enumeration():
    differences = np.array([-2., -1., 0., 1., 2., 3., 4., 5.])
    observed, p_value, exact, draws = paired_log_rms_sign_flip_test(differences)
    null = np.array([np.mean(np.array(signs) * differences) for signs in product((-1, 1), repeat=8)])
    assert observed == differences.mean()
    assert exact and draws == 255
    assert p_value == np.mean(null >= observed)


def test_primary_monte_carlo_resolution_and_projection_scaling():
    waveform = np.linspace(-1.0, 1.0, 20)
    response = np.tile(waveform, (17, 1))
    first = fixed_window_reproducibility_sign_flip_test(response, sfreq=250.0, random_state=4)
    scaled = fixed_window_reproducibility_sign_flip_test(3 * response, sfreq=1000.0, random_state=4)
    assert not first.exact and first.n_randomizations == 5000
    assert first.p_value >= 1 / 5001
    assert scaled.p_value == first.p_value
    assert scaled.statistic == pytest.approx(1.5 * first.statistic)
    _, p_energy, exact, draws = paired_log_rms_sign_flip_test(np.ones(17), random_state=4)
    assert not exact and draws == 5000
    assert p_energy >= 1 / 5001


def test_nominal_crp_sample_grid_uses_nearest_even_and_appends_endpoint():
    counts = _projection_sample_counts(9, 250.0, CRPConfig())
    np.testing.assert_array_equal(counts, np.arange(2, 10))
    counts = _projection_sample_counts(13, 1000.0, CRPConfig())
    np.testing.assert_array_equal(counts, [10, 13])
