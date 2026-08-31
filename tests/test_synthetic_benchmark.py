from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "benchmark_crp_energy",
    ROOT / "validation" / "benchmark_crp_energy.py",
)
benchmark = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)

def test_wilson_interval_handles_boundary_counts() -> None:
    empty_low, empty_high = benchmark.wilson_interval(0, 10)
    full_low, full_high = benchmark.wilson_interval(10, 10)

    assert empty_low == pytest.approx(0.0)
    assert empty_high == pytest.approx(0.2775328, rel=1e-6)
    assert full_low == pytest.approx(1.0 - empty_high)
    assert full_high == pytest.approx(1.0)
    assert benchmark.wilson_interval(0, 0) == (np.nan, np.nan)


def test_simulated_scenario_is_deterministic_and_retains_truth() -> None:
    config = benchmark.BenchmarkConfig(
        n_replicates=1,
        trial_counts=(12,),
        snr_values=(0.0, 2.0),
        artifact_fractions=(0.25,),
        n_family_channels=3,
    )
    spec = benchmark.ScenarioSpec(
        replicate=0,
        n_trials=12,
        snr=2.0,
        artifact_fraction=0.25,
    )

    first_values, first_times, first_channels, first_truth = (
        benchmark.simulate_scenario(config, spec)
    )
    second_values, second_times, second_channels, second_truth = (
        benchmark.simulate_scenario(config, spec)
    )

    np.testing.assert_array_equal(first_values, second_values)
    np.testing.assert_array_equal(first_times, second_times)
    assert first_channels == second_channels
    assert first_truth == second_truth
    assert first_truth["truth"] == "injected"
    assert first_truth["n_artifact_trials"] == 3
    assert first_truth["n_clean_trials_expected"] == 9
    assert first_truth["snr_realized"] > 1.0
    assert np.isnan(first_values[:, :, 0]).any()
    assert not np.isnan(first_values[:, :, 1:]).any()


def test_metric_summary_counts_nonevaluable_as_unconditional_negative() -> None:
    frame = pd.DataFrame(
        {
            "detector": ["d"] * 4,
            "truth": [benchmark.NULL_TRUTH] * 4,
            "call": [False, True, False, False],
            "evaluable": [True, True, False, False],
        }
    )

    result = benchmark.summarize_metrics(
        frame,
        group_columns=["detector", "truth"],
    ).iloc[0]

    assert result["n_scenarios"] == 4
    assert result["n_evaluable"] == 2
    assert result["call_rate_unconditional"] == pytest.approx(0.25)
    assert result["call_rate_given_evaluable"] == pytest.approx(0.5)
    assert result["specificity"] == pytest.approx(0.75)


def test_one_scenario_uses_public_detectors_and_joint_identity() -> None:
    config = benchmark.BenchmarkConfig(
        n_replicates=1,
        trial_counts=(8,),
        snr_values=(0.0, 4.0),
        artifact_fractions=(0.0,),
        n_family_channels=3,
    )
    spec = benchmark.ScenarioSpec(
        replicate=0,
        n_trials=8,
        snr=4.0,
        artifact_fraction=0.0,
    )

    scenario, records = benchmark.run_scenario(config, spec)
    target = [record for record in records if record["is_target"]]
    primary_records = [
        record for record in records if record["detector"] == "crp_energy"
    ]

    assert {record["detector"] for record in target} == set(
        benchmark.DETECTORS
    )
    assert scenario["primary_qc_status"] == "pass"
    assert scenario["primary_p_joint_identity"]
    assert scenario["primary_n_trials_clean"] == 8
    assert scenario["crp__energy__evaluable"]
    assert len(primary_records) == config.n_family_channels
    assert {
        record["crp_energy_random_state_root"] for record in primary_records
    } == {scenario["detector_random_state"]}
    assert len(
        {
            record["crp_energy_random_state_effective"]
            for record in primary_records
        }
    ) == config.n_family_channels
    assert scenario["primary_random_state_effective"] == next(
        record["crp_energy_random_state_effective"]
        for record in primary_records
        if record["is_target"]
    )


def test_family_calibration_keeps_target_and_reference_errors_distinct() -> None:
    rows = []
    for scenario_id, target_truth, calls in (
        ("null_family", benchmark.NULL_TRUTH, [False, True, False]),
        ("mixed_family", benchmark.INJECTED_TRUTH, [True, True, False]),
    ):
        for channel_index, call in enumerate(calls):
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "detector": "crp_energy",
                    "multiplicity_adjustment": benchmark.MULTIPLICITY_ADJUSTMENT[
                        "crp_energy"
                    ],
                    "channel": "target"
                    if channel_index == 0
                    else f"reference_{channel_index}",
                    "is_target": channel_index == 0,
                    "truth": target_truth
                    if channel_index == 0
                    else "null_reference",
                    "call": call,
                    "evaluable": True,
                    "replicate": 0,
                    "n_trials_total": 12,
                    "artifact_fraction": 0.0,
                    "n_clean_trials_expected": 12,
                    "snr_nominal": 0.0
                    if target_truth == benchmark.NULL_TRUTH
                    else 2.0,
                }
            )

    families = benchmark.family_scenario_results(pd.DataFrame(rows)).set_index(
        "family_type"
    )

    assert families.loc["all_null", "n_false_calls"] == 1
    assert families.loc["all_null", "null_channel_call_fraction"] == pytest.approx(
        1 / 3
    )
    assert families.loc["all_null", "false_discovery_proportion"] == 1.0
    assert families.loc["mixed_signal", "target_call"]
    assert families.loc["mixed_signal", "n_false_calls"] == 1
    assert families.loc[
        "mixed_signal", "false_discovery_proportion"
    ] == pytest.approx(0.5)
    summary = benchmark.summarize_family_calibration(
        families.reset_index(),
        group_columns=["detector", "family_type"],
        n_resamples=100,
        random_state=17,
    ).set_index("family_type")
    assert summary.loc["all_null", "probability_any_null_call"] == 1.0
    assert summary.loc["mixed_signal", "target_power_unconditional"] == 1.0
    assert summary.loc[
        "mixed_signal", "mean_false_discovery_proportion"
    ] == pytest.approx(0.5)


def test_family_calibration_separates_generated_and_evaluable_nulls() -> None:
    rows = []
    for channel, call, evaluable in (
        ("target", False, False),
        ("reference_1", True, True),
        ("reference_2", False, True),
        ("reference_3", False, True),
    ):
        rows.append(
            {
                "scenario_id": "masked_null_family",
                "detector": "crp_energy",
                "multiplicity_adjustment": benchmark.MULTIPLICITY_ADJUSTMENT[
                    "crp_energy"
                ],
                "channel": channel,
                "is_target": channel == "target",
                "truth": (
                    benchmark.NULL_TRUTH
                    if channel == "target"
                    else "null_reference"
                ),
                "call": call,
                "evaluable": evaluable,
                "replicate": 0,
                "n_trials_total": 8,
                "artifact_fraction": 0.5,
                "n_clean_trials_expected": 4,
                "snr_nominal": 0.0,
            }
        )

    family = benchmark.family_scenario_results(pd.DataFrame(rows)).iloc[0]
    assert family["n_null_channels"] == 4
    assert family["n_evaluable_null_channels"] == 3
    assert family["generated_null_channel_call_fraction"] == pytest.approx(0.25)
    assert family["null_channel_call_fraction"] == pytest.approx(1 / 3)

    summary = benchmark.summarize_family_calibration(
        pd.DataFrame([family]),
        group_columns=["detector", "family_type"],
        n_resamples=100,
        random_state=19,
    ).iloc[0]
    assert summary["n_generated_null_contacts"] == 4
    assert summary["n_evaluable_null_hypotheses"] == 3
    assert summary["n_null_hypotheses"] == 3


def test_fixed_window_projection_matrix_matches_existing_ordered_quantities() -> None:
    from ERPy.crp import semi_normalized_cross_projections
    from ERPy.crp_energy import (
        _ordered_projection_matrix,
        fixed_window_reproducibility_sign_flip_test,
    )

    rng = np.random.default_rng(9)
    response = rng.normal(size=(5, 20))
    matrix = _ordered_projection_matrix(response, sfreq=250.0)
    ordered = matrix[~np.eye(len(matrix), dtype=bool)]
    existing = semi_normalized_cross_projections(
        response.T,
        sfreq=250.0,
    )

    np.testing.assert_allclose(ordered, existing, rtol=1e-14, atol=1e-14)
    result = fixed_window_reproducibility_sign_flip_test(
        response,
        sfreq=250.0,
        max_exact_trials=12,
    )
    assert result.exact
    assert result.n_randomizations == 2 ** (len(response) - 1) - 1
    assert result.statistic == pytest.approx(matrix.sum() / (5 * 4))


def test_separate_demean_energy_is_invariant_to_segment_dc_offsets() -> None:
    from ERPy.crp_energy import CRPEnergyConfig, run_crp_energy_array

    config = benchmark.BenchmarkConfig(
        n_replicates=1,
        trial_counts=(12,),
        snr_values=(0.0,),
        artifact_fractions=(0.0,),
        n_family_channels=3,
    )
    spec = benchmark.ScenarioSpec(0, 12, 0.0, 0.0)
    values, times, _, truth = benchmark.simulate_scenario(config, spec)
    detector_config = CRPEnergyConfig(
        baseline_window=tuple(config.baseline_window),
        response_window=tuple(config.response_window),
        artifact_interval=tuple(config.artifact_interval),
        min_clean_trials=int(config.min_clean_trials),
        n_permutations=int(config.n_permutations),
        max_exact_trials=int(config.max_exact_trials),
        max_exact_reproducibility_trials=int(
            config.max_exact_reproducibility_trials
        ),
        canonical_energy_cv=False,
        random_state=int(truth["detector_random_state"]),
    )
    original = run_crp_energy_array(
        values[:, :, 0],
        times,
        channel="target",
        config=detector_config,
    )
    shifted = values[:, :, 0].copy()
    response = np.flatnonzero(
        (times >= max(config.response_window[0], config.artifact_interval[1]))
        & (times <= config.response_window[1])
    )
    baseline_candidates = np.flatnonzero(
        (times >= config.baseline_window[0])
        & (times <= config.baseline_window[1])
    )
    baseline = baseline_candidates[-len(response) :]
    shifted[:, baseline] -= np.arange(12, dtype=float)[:, None]
    shifted[:, response] += 2.0 * np.arange(12, dtype=float)[:, None]
    changed = run_crp_energy_array(
        shifted,
        times,
        channel="target",
        config=detector_config,
    )

    assert changed.p_energy == pytest.approx(original.p_energy)
    assert changed.energy_statistic == pytest.approx(
        original.energy_statistic
    )
