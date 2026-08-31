from __future__ import annotations

import json
from itertools import product

import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

import ERPy as ep
from ERPy.crp import (
    CRPConfig,
    balanced_projection_indices,
    run_crp_array,
    semi_normalized_cross_projections,
)


def _epochs(arr: np.ndarray, times: np.ndarray, channels: list[str]) -> ep.Epochs:
    index = pd.MultiIndex.from_product(
        [range(arr.shape[0]), times],
        names=["epoch", "time"],
    )
    return ep.Epochs(
        pd.DataFrame(
            arr.reshape(-1, len(channels)),
            index=index,
            columns=channels,
        ),
        sfreq=float(1.0 / np.median(np.diff(times))),
        tmin=float(times[0]),
        tmax=float(times[-1]),
        baseline=(-0.2, -0.03),
    )


def test_optimized_crp_profile_matches_direct_published_projections() -> None:
    sfreq = 1000.0
    times = np.arange(-0.2, 0.301, 1.0 / sfreq)
    rng = np.random.default_rng(20)
    x = rng.normal(0.0, 0.5, (9, len(times)))
    response = (times >= 0.015) & (times <= 0.16)
    shape = 12.0 * np.exp(-0.5 * ((times[response] - 0.065) / 0.012) ** 2)
    x[:, response] += shape
    config = CRPConfig(
        response_window=(0.015, 0.2),
        baseline_window=(-0.2, -0.03),
    )

    result = run_crp_array(x, times, channel="C1", config=config)

    baseline = (times >= -0.2) & (times <= -0.03)
    centered = x - x[:, baseline].mean(axis=1, keepdims=True)
    response_times = times[(times >= 0.015) & (times <= 0.2)]
    voltage = centered[:, (times >= 0.015) & (times <= 0.2)].T
    counts = np.searchsorted(response_times, result.projection_times) + 1
    direct_profile = np.asarray(
        [
            np.mean(
                semi_normalized_cross_projections(
                    voltage[:count],
                    sfreq=sfreq,
                )
            )
            for count in counts
        ]
    )
    assert result.mean_projection_profile == pytest.approx(
        direct_profile,
        rel=1e-11,
        abs=1e-11,
    )

    selected = semi_normalized_cross_projections(
        voltage[: len(result.canonical_waveform)],
        sfreq=sfreq,
    )[balanced_projection_indices(x.shape[0])]
    assert result.cross_projections == pytest.approx(selected)
    assert result.p_value < 0.05
    assert result.response_duration_s > 0
    assert np.isfinite(result.alpha_prime_uv).all()


def test_primary_shape_magnitude_rule_separates_discordant_responses() -> None:
    sfreq = 500.0
    times = np.arange(-0.2, 0.301, 1.0 / sfreq)
    channels = ["strong", "weak", "inconsistent", "control1", "control2"]
    rng = np.random.default_rng(44)
    arr = rng.normal(0.0, 0.25, (20, len(times), len(channels)))
    response = (times >= 0.005) & (times <= 0.100)
    carrier = np.sin(2 * np.pi * 55.0 * times[response])
    taper = np.sin(np.linspace(0.0, np.pi, response.sum())) ** 0.5
    waveform = carrier * taper
    arr[:, response, 0] += 100.0 * waveform
    arr[:, response, 1] += 6.0 * waveform
    signs = np.where(np.arange(arr.shape[0]) % 2 == 0, 1.0, -1.0)
    arr[:, response, 2] += signs[:, None] * 100.0 * waveform
    epochs = _epochs(arr, times, channels)

    detections = epochs.detect_erp_all(
        methods=["crp_significance", "kundu_rolston"],
        min_consensus=1,
    )
    channel_results = detections.drop_duplicates("channel").set_index("channel")
    crp = detections[detections["method"].eq("crp_significance")].set_index("channel")
    kundu = detections[detections["method"].eq("kundu_rolston")].set_index("channel")

    assert bool(crp.loc["strong", "crp_fdr_significant"])
    assert bool(crp.loc["weak", "crp_fdr_significant"])
    assert crp["crp_inference_status"].eq(
        "exploratory_uncalibrated_selected_duration_shared_trial_t_test"
    ).all()
    assert bool(kundu.loc["strong", "significant"])
    assert not bool(kundu.loc["weak", "significant"])
    assert bool(kundu.loc["inconsistent", "significant"])
    assert not bool(crp.loc["inconsistent", "crp_fdr_significant"])
    assert bool(channel_results.loc["strong", "shape_magnitude_significant"])
    assert not bool(channel_results.loc["weak", "shape_magnitude_significant"])
    assert not bool(channel_results.loc["inconsistent", "shape_magnitude_significant"])
    assert kundu.loc["strong", "kundu_longest_suprathreshold_ms"] >= 15.0
    assert kundu.loc["strong", "kundu_post_median_uv"] > 30.0


def _crp_energy_fixture() -> tuple[ep.Epochs, list[str]]:
    sfreq = 500.0
    times = np.arange(-0.4, 0.351, 1.0 / sfreq)
    channels = [
        "both",
        "small_repeated",
        "energy_noise",
        "polarity_reversal",
        "null1",
        "null2",
    ]
    rng = np.random.default_rng(104)
    arr = rng.normal(0.0, 0.5, (20, len(times), len(channels)))
    response = (times >= 0.015) & (times <= 0.350)
    response_times = times[response]
    waveform = (
        np.exp(-0.5 * ((response_times - 0.090) / 0.018) ** 2)
        - 0.75 * np.exp(-0.5 * ((response_times - 0.180) / 0.030) ** 2)
    )
    arr[:, response, 0] += 4.0 * waveform
    arr[:, response, 1] *= 0.55
    arr[:, response, 1] += 0.75 * waveform
    arr[:, response, 2] *= 3.0
    signs = np.where(np.arange(arr.shape[0]) % 2 == 0, 1.0, -1.0)
    arr[:, response, 3] += signs[:, None] * 4.0 * waveform
    return _epochs(arr, times, channels), channels


def test_crp_energy_recovers_all_four_diagnostic_classes() -> None:
    epochs, _ = _crp_energy_fixture()

    results = ep.detect(
        epochs,
        method="crp_energy",
        baseline_window=(-0.35, -0.015),
        response_window=(0.015, 0.350),
        canonical_energy_cv=False,
        return_arrays=False,
        random_state=42,
    ).set_index("channel")

    assert results.loc["both", "classification"] == "reproducible_energetic"
    assert results.loc["small_repeated", "classification"] == "reproducible_low_energy"
    assert results.loc["energy_noise", "classification"] == "energetic_inconsistent"
    assert results.loc["polarity_reversal", "classification"] == "energetic_inconsistent"
    assert results.loc["null1", "classification"] == "no_response"
    assert bool(results.loc["both", "significant"])
    assert not bool(results.loc["energy_noise", "significant"])
    assert results.loc["both", "p_joint"] == max(
        results.loc["both", "p_crp"],
        results.loc["both", "p_energy"],
    )
    assert results.loc["both", "q_joint"] <= 0.05


def test_crp_energy_excludes_stimulation_contacts_from_fdr_family() -> None:
    epochs, _ = _crp_energy_fixture()
    epochs.stim_ch = ["both-ref"]

    results = ep.detect(
        epochs,
        method="crp_energy",
        baseline_window=(-0.35, -0.015),
        response_window=(0.015, 0.350),
        canonical_energy_cv=False,
        return_arrays=False,
        random_state=42,
    ).set_index("channel")

    excluded = results.loc["both"]
    assert excluded["classification"] == "reproducible_energetic"
    assert np.isfinite(excluded["p_joint"])
    assert np.isnan(excluded["q_joint"])
    assert excluded["qc_status"] == "stimulation_contact_excluded"
    assert not bool(excluded["significant"])
    assert results["testing_family_size"].nunique() == 1
    assert int(results["testing_family_size"].iloc[0]) == int(
        results.drop(index="both")["p_joint"].notna().sum()
    )

    included = ep.detect(
        epochs,
        method="crp_energy",
        baseline_window=(-0.35, -0.015),
        response_window=(0.015, 0.350),
        canonical_energy_cv=False,
        return_arrays=False,
        random_state=42,
        exclude_stimulation_contacts=False,
    ).set_index("channel")
    assert included.loc["both", "qc_status"] == "pass"
    assert np.isfinite(included.loc["both", "q_joint"])
    assert bool(included.loc["both", "significant"])


def test_crp_energy_is_deterministic_and_global_sign_invariant() -> None:
    epochs, _ = _crp_energy_fixture()
    arr, times, channels = epochs.as_array()
    inverted = _epochs(-arr, times, list(channels))
    options = {
        "method": "crp_energy",
        "baseline_window": (-0.35, -0.015),
        "response_window": (0.015, 0.350),
        "canonical_energy_cv": True,
        "return_arrays": False,
        "random_state": 42,
    }

    first = ep.detect(epochs, **options).set_index("channel")
    repeated = ep.detect(epochs, **options).set_index("channel")
    sign_reversed = ep.detect(inverted, **options).set_index("channel")

    for column in (
        "p_crp",
        "p_energy",
        "p_joint",
        "q_joint",
        "rms_ratio_db",
        "canonical_energy",
        "canonical_energy_fraction",
    ):
        assert first[column].to_numpy() == pytest.approx(
            repeated[column].to_numpy(), nan_ok=True
        )
        assert first[column].to_numpy() == pytest.approx(
            sign_reversed[column].to_numpy(), nan_ok=True
        )
    assert first["classification"].tolist() == sign_reversed["classification"].tolist()


def test_crp_energy_returns_auditable_arrays_and_insufficient_data() -> None:
    epochs, _ = _crp_energy_fixture()
    full = ep.detect(
        epochs,
        method="crp_energy",
        baseline_window=(-0.35, -0.015),
        response_window=(0.015, 0.350),
        canonical_energy_cv=True,
        random_state=42,
    ).set_index("channel")

    required = {
        "significant",
        "classification",
        "p_crp",
        "p_energy",
        "p_joint",
        "q_joint",
        "crp_statistic",
        "rms_response",
        "rms_baseline",
        "rms_ratio_db",
        "canonical_energy",
        "canonical_energy_fraction",
        "response_duration",
        "n_trials_total",
        "n_trials_clean",
        "canonical_waveform",
        "trial_coefficients",
        "reproducibility_test_exact",
        "reproducibility_n_randomizations",
        "qc_status",
        "parameters",
        "detector_version",
    }
    assert required.issubset(full.columns)
    assert len(full.loc["both", "canonical_waveform"]) > 1
    assert len(full.loc["both", "trial_coefficients"]) == 20

    arr, times, channels = epochs.as_array()
    arr[:15, :, 0] = np.nan
    sparse = _epochs(arr, times, list(channels))
    insufficient = ep.detect(
        sparse,
        method="crp_energy",
        baseline_window=(-0.35, -0.015),
        response_window=(0.015, 0.350),
    ).set_index("channel")
    assert insufficient.loc["both", "classification"] == "insufficient_data"
    assert insufficient.loc["both", "qc_status"] == "insufficient_data"
    assert not bool(insufficient.loc["both", "significant"])


def test_crp_energy_exact_sign_flip_uses_the_intersection_union_p_value() -> None:
    differences = np.linspace(0.1, 0.8, 8)
    statistic, p_value, exact, n_permutations = ep.paired_log_rms_sign_flip_test(
        differences,
        random_state=999,
    )

    assert statistic == pytest.approx(differences.mean())
    assert p_value == pytest.approx(1.0 / 2**8)
    assert exact
    assert n_permutations == 2**8 - 1


def test_fixed_window_reproducibility_uses_unique_trial_sign_patterns() -> None:
    waveform = np.r_[
        np.linspace(0.0, 1.0, 24),
        np.linspace(1.0, -0.6, 32),
    ]
    response = np.tile(waveform, (8, 1))

    result = ep.fixed_window_reproducibility_sign_flip_test(
        response,
        sfreq=500.0,
        random_state=999,
    )

    assert result.statistic > 0
    assert result.p_value == pytest.approx(1.0 / 2**7)
    assert result.exact
    assert result.n_randomizations == 2**7 - 1


def test_fixed_window_reproducibility_exact_p_value_respects_its_floor() -> None:
    """The observed assignment remains counted despite summation roundoff."""

    rng = np.random.default_rng(40)
    common = rng.normal(size=83)
    response = (common[None, :] + 0.08 * rng.normal(size=(8, 83))) * 37.0

    result = ep.fixed_window_reproducibility_sign_flip_test(
        response,
        sfreq=500.0,
        max_exact_trials=12,
    )

    assert result.exact
    assert result.p_value == pytest.approx(1.0 / 2**7)
    assert result.n_randomizations == 2**7 - 1


def test_fixed_window_reproducibility_matches_brute_force_quadratic_null() -> None:
    sfreq = 250.0
    rng = np.random.default_rng(809)
    response = rng.normal(size=(5, 31))
    result = ep.fixed_window_reproducibility_sign_flip_test(
        response,
        sfreq=sfreq,
        max_exact_trials=12,
    )

    normalized = response / np.linalg.norm(response, axis=1)[:, None]
    projections = normalized @ response.T / np.sqrt(sfreq)
    np.fill_diagonal(projections, 0.0)
    denominator = float(len(response) * (len(response) - 1))
    observed = float(projections.sum() / denominator)
    signs = np.asarray(
        list(product((-1.0, 1.0), repeat=len(response))),
        dtype=float,
    )
    null_statistics = np.asarray(
        [sign @ projections @ sign / denominator for sign in signs],
        dtype=float,
    )

    assert result.statistic == pytest.approx(observed, rel=1e-14, abs=1e-14)
    assert result.p_value == pytest.approx(
        np.mean(null_statistics >= observed - 1e-15)
    )


def test_crp_energy_fixed_window_inference_is_independent_of_duration_grid() -> None:
    epochs, _ = _crp_energy_fixture()
    values, times, channels = epochs.as_array()
    channel_values = values[:, :, channels.index("both")]
    common = {
        "response_window": (0.015, 0.350),
        "baseline_window": (-0.35, -0.015),
        "canonical_energy_cv": False,
        "random_state": 42,
    }

    first = ep.run_crp_energy_array(
        channel_values,
        times,
        config=ep.CRPEnergyConfig(
            projection_start_s=0.010,
            projection_step_s=0.005,
            **common,
        ),
    )
    changed_grid = ep.run_crp_energy_array(
        channel_values,
        times,
        config=ep.CRPEnergyConfig(
            projection_start_s=0.030,
            projection_step_s=0.011,
            **common,
        ),
    )

    assert changed_grid.p_crp == pytest.approx(first.p_crp)
    assert changed_grid.crp_statistic == pytest.approx(first.crp_statistic)


def test_crp_energy_compares_separately_demeaned_matched_windows() -> None:
    sfreq = 200.0
    times = np.arange(-0.4, 0.351, 1.0 / sfreq)
    baseline = (times >= -0.35) & (times <= -0.015)
    response = (times >= 0.015) & (times <= 0.350)
    assert int(baseline.sum()) == int(response.sum())
    rng = np.random.default_rng(48)
    matched_noise = rng.normal(0.0, 1.0, (12, int(response.sum())))
    values = np.zeros((12, len(times), 1), dtype=float)
    values[:, baseline, 0] = matched_noise
    values[:, response, 0] = matched_noise + 25.0

    result = ep.detect(
        _epochs(values, times, ["offset_only"]),
        method="crp_energy",
        baseline_window=(-0.35, -0.015),
        response_window=(0.015, 0.350),
        return_arrays=False,
        random_state=42,
    ).iloc[0]

    assert result["rms_response"] == pytest.approx(result["rms_baseline"])
    assert result["p_energy"] == pytest.approx(1.0)
    assert not bool(result["energy_significant"])


def test_crp_energy_channel_streams_are_distinct_and_order_invariant() -> None:
    epochs, channels = _crp_energy_fixture()
    values, times, _ = epochs.as_array()
    options = {
        "method": "crp_energy",
        "baseline_window": (-0.35, -0.015),
        "response_window": (0.015, 0.350),
        "canonical_energy_cv": False,
        "return_arrays": False,
        "random_state": 42,
    }

    original = ep.detect(epochs, **options).set_index("channel").sort_index()
    reversed_epochs = _epochs(
        values[:, :, ::-1],
        times,
        list(reversed(channels)),
    )
    reordered = (
        ep.detect(reversed_epochs, **options)
        .set_index("channel")
        .sort_index()
    )

    assert original["random_state_root"].eq("42").all()
    assert original["random_state_effective"].is_unique
    assert original["random_stream_derivation"].eq(
        "sha256-v1(root-seed,channel-label;63-bit)"
    ).all()
    assert original["reproducibility_test_exact"].eq(False).all()
    assert original["reproducibility_n_randomizations"].eq(5_000).all()
    assert original["energy_test_exact"].eq(False).all()
    assert original["energy_n_permutations"].eq(5_000).all()
    assert original["random_state_effective"].tolist() == reordered[
        "random_state_effective"
    ].tolist()
    for column in ("p_crp", "p_energy", "p_joint", "q_joint"):
        np.testing.assert_allclose(
            original[column].to_numpy(dtype=float),
            reordered[column].to_numpy(dtype=float),
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )
    for _, row in original.iterrows():
        parameters = json.loads(str(row["parameters"]))
        assert parameters["random_state"] == int(
            row["random_state_effective"]
        )


def test_multimethod_concat_and_metadata_preserve_seed_decimal_exactly(
    tmp_path,
) -> None:
    from ERPy.facade import _detector_parameter_records

    epochs, _ = _crp_energy_fixture()
    detections = epochs.detect_erp_all(
        methods=["kundu_rolston", "crp_energy"],
        baseline_window=(-0.35, -0.015),
        response_window=(0.015, 0.350),
        crp_energy={
            "canonical_energy_cv": False,
            "return_arrays": False,
            "random_state": 42,
        },
    )
    primary = detections[detections["method"].eq("crp_energy")].copy()

    assert primary["random_state_root"].eq("42").all()
    assert primary["random_state_effective"].map(type).eq(str).all()
    assert primary["random_state_effective"].str.fullmatch(r"[0-9]+").all()
    assert primary["random_state_effective"].map(int).gt(2**53).all()
    for _, row in primary.iterrows():
        parameters = json.loads(str(row["parameters"]))
        assert str(parameters["random_state"]) == row[
            "random_state_effective"
        ]

    metadata = {
        "detection": {
            "method_parameter_records": _detector_parameter_records(
                detections
            )
        }
    }
    analysis_result = ep.AnalysisResult(
        epochs=epochs,
        detections=detections,
        artifact_report=ep.ArtifactResponseReport(pd.DataFrame()),
        artifact_summary=pd.DataFrame(),
        qc_detections=detections,
        metadata=metadata,
    )
    saved = analysis_result.save(
        tmp_path / "seed_roundtrip",
        prefix="seed_roundtrip",
    )
    decoded = json.loads(
        saved["metadata"].read_text(encoding="utf-8")
    )
    crp_record = next(
        record
        for record in decoded["detection"]["method_parameter_records"]
        if record["method"] == "crp_energy"
    )
    first = primary.iloc[0]
    assert crp_record["random_state_root"] == "42"
    assert crp_record["random_state_effective"] == first[
        "random_state_effective"
    ]


@pytest.mark.parametrize(
    "config, message",
    [
        (ep.CRPEnergyConfig(n_permutations=0), "n_permutations"),
        (ep.CRPEnergyConfig(eps=np.inf), "eps"),
        (ep.CRPEnergyConfig(random_state=-1), "random_state"),
        (
            ep.CRPEnergyConfig(response_window=(0.2, 0.1)),
            "response_window",
        ),
        (
            ep.CRPEnergyConfig(
                baseline_window=(-0.1, 0.02),
                response_window=(0.015, 0.2),
            ),
            "baseline_window",
        ),
        (
            ep.CRPEnergyConfig(
                response_window=(0.015, 0.2),
                artifact_interval=(0.0, 0.3),
            ),
            "artifact_interval",
        ),
    ],
)
def test_crp_energy_rejects_invalid_configuration_early(
    config: ep.CRPEnergyConfig,
    message: str,
) -> None:
    times = np.arange(-1.0, 1.001, 0.002)
    values = np.ones((8, len(times)), dtype=float)

    with pytest.raises(ValueError, match=message):
        ep.run_crp_energy_array(values, times, config=config)


def test_sign_flip_helpers_bound_enumeration_and_mark_empty_as_not_run() -> None:
    with pytest.raises(ValueError, match="max_exact_trials"):
        ep.paired_log_rms_sign_flip_test(
            np.ones(20),
            max_exact_trials=19,
        )
    with pytest.raises(ValueError, match="n_permutations"):
        ep.fixed_window_reproducibility_sign_flip_test(
            np.ones((8, 20)),
            sfreq=500.0,
            n_permutations=0,
        )
    statistic, p_value, exact, n_randomizations = (
        ep.paired_log_rms_sign_flip_test(np.asarray([np.nan]))
    )
    assert np.isnan(statistic)
    assert np.isnan(p_value)
    assert not exact
    assert n_randomizations == 0

    empty = ep.run_crp_energy_array(
        np.ones((3, 401), dtype=float),
        np.linspace(-1.0, 1.0, 401),
        config=ep.CRPEnergyConfig(min_clean_trials=8),
    )
    assert empty.qc_status == "insufficient_data"
    assert not empty.reproducibility_test_exact
    assert not empty.energy_test_exact
    assert empty.reproducibility_n_randomizations == 0
    assert empty.energy_n_permutations == 0


def test_crp_energy_detector_error_retains_schema_and_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    epochs, channels = _crp_energy_fixture()

    def fail_detector(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("forced detector failure")

    monkeypatch.setattr(
        "ERPy.erp_detection.run_crp_energy_array",
        fail_detector,
    )
    detections = ep.detect(
        epochs,
        method="crp_energy",
        baseline_window=(-0.35, -0.015),
        response_window=(0.015, 0.350),
        return_arrays=False,
        random_state=42,
    )

    assert len(detections) == len(channels)
    assert set(ep.DETECTOR_QUANTITY_COLUMNS["crp_energy"]).issubset(
        detections.columns
    )
    assert detections["qc_status"].eq("detector_error").all()
    assert detections["parameters"].str.len().gt(0).all()
    assert detections["reproducibility_test_exact"].eq(False).all()
    assert detections["energy_test_exact"].eq(False).all()
    assert detections["notes"].str.contains("forced detector failure").all()


def test_crp_energy_null_family_controls_false_positive_rate() -> None:
    sfreq = 500.0
    times = np.arange(-0.25, 0.201, 1.0 / sfreq)
    channels = [f"null{index}" for index in range(48)]
    rng = np.random.default_rng(222)
    arr = rng.normal(0.0, 1.0, (12, len(times), len(channels)))
    epochs = _epochs(arr, times, channels)

    results = ep.detect(
        epochs,
        method="crp_energy",
        baseline_window=(-0.20, -0.015),
        response_window=(0.015, 0.200),
        canonical_energy_cv=False,
        return_arrays=False,
        random_state=42,
    )

    assert results["significant"].mean() <= 0.05
    assert (results["p_joint"] == results[["p_crp", "p_energy"]].max(axis=1)).all()


def test_crp_energy_detects_reproducible_late_multiphasic_response_that_kundu_misses() -> None:
    sfreq = 500.0
    times = np.arange(-0.4, 0.351, 1.0 / sfreq)
    channels = ["late", "control1", "control2", "control3"]
    rng = np.random.default_rng(73)
    arr = rng.normal(0.0, 0.5, (20, len(times), len(channels)))
    late_waveform = 8.0 * (
        np.exp(-0.5 * ((times - 0.190) / 0.025) ** 2)
        - 0.70 * np.exp(-0.5 * ((times - 0.270) / 0.030) ** 2)
    )
    arr[:, :, 0] += late_waveform
    epochs = _epochs(arr, times, channels)

    detections = epochs.detect_erp_all(
        methods=["crp_energy", "kundu_rolston"],
        baseline_window=(-0.35, -0.015),
        response_window=(0.015, 0.350),
        crp_energy={"canonical_energy_cv": False, "return_arrays": False},
    )
    late = detections[detections["channel"].eq("late")].set_index("method")

    assert bool(late.loc["crp_energy", "significant"])
    assert not bool(late.loc["kundu_rolston", "significant"])


def test_crp_energy_receives_saturation_rejected_before_detection() -> None:
    sfreq = 1000.0
    times = np.arange(-0.25, 0.201, 1.0 / sfreq)
    channels = ["saturated", "clean"]
    rng = np.random.default_rng(2)
    arr = rng.normal(0.0, 0.2, (10, len(times), len(channels)))
    plateau = (times >= 0.040) & (times <= 0.090)
    arr[:, plateau, 0] = 4095.0
    epochs = _epochs(arr, times, channels)
    report = epochs.flag_artifacts(
        response_window=(0.015, 0.200),
        baseline_window=(-0.20, -0.015),
        zscore_threshold=99.0,
        saturation_fraction=0.95,
    )
    clean_epochs = epochs.reject_artifacts(report=report, mode="nan_response")

    results = ep.detect(
        clean_epochs,
        method="crp_energy",
        baseline_window=(-0.20, -0.015),
        response_window=(0.015, 0.200),
        canonical_energy_cv=False,
        return_arrays=False,
    ).set_index("channel")

    assert report.table.loc[
        report.table["channel"].eq("saturated"), "bad_response"
    ].all()
    assert results.loc["saturated", "classification"] == "insufficient_data"
    assert results.loc["saturated", "n_trials_clean"] == 0


def test_crp_energy_excludes_the_declared_stimulation_artifact_interval() -> None:
    sfreq = 2000.0
    times = np.arange(-0.2, 0.201, 1.0 / sfreq)
    channels = ["C1", "C2", "C3"]
    rng = np.random.default_rng(918)
    clean = rng.normal(0.0, 0.5, (20, len(times), len(channels)))
    contaminated = clean.copy()
    artifact = (times >= 0.0) & (times < 0.015)
    contaminated[:, artifact, :] += 10_000.0
    options = {
        "method": "crp_energy",
        "baseline_window": (-0.185, -0.015),
        "response_window": (0.0, 0.170),
        "artifact_interval": (0.0, 0.015),
        "canonical_energy_cv": False,
        "return_arrays": False,
        "random_state": 42,
    }

    clean_results = ep.detect(_epochs(clean, times, channels), **options)
    artifact_results = ep.detect(
        _epochs(contaminated, times, channels),
        **options,
    )

    for column in ("p_crp", "p_energy", "p_joint", "q_joint", "rms_ratio_db"):
        np.testing.assert_allclose(
            clean_results[column].to_numpy(dtype=float),
            artifact_results[column].to_numpy(dtype=float),
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )
    starts = artifact_results["response_window"].map(
        lambda window: float(tuple(window)[0])
    )
    stops = artifact_results["response_window"].map(
        lambda window: float(tuple(window)[1])
    )
    np.testing.assert_allclose(starts, 0.015, atol=1.0 / sfreq)
    assert ((stops <= 0.170) & (stops >= 0.170 - 1.0 / sfreq)).all()


def test_adding_crp_energy_does_not_change_existing_detector_outputs() -> None:
    epochs, _ = _crp_energy_fixture()
    existing_methods = [method for method in ep.ALL_METHODS if method != "crp_energy"]
    common_options = {
        "baseline_window": (-0.35, -0.015),
        "response_window": (0.015, 0.100),
        "crowther_gamma": {
            "n_permutations": 24,
            "channel_chunk_size": 3,
            "permutation_chunk_size": 12,
            "random_state": 9,
        },
    }
    before = epochs.detect_erp_all(methods=existing_methods, **common_options)
    after = epochs.detect_erp_all(
        methods=existing_methods + ["crp_energy"],
        crp_energy={"canonical_energy_cv": False, "return_arrays": False},
        **common_options,
    )
    after = after[after["method"].isin(existing_methods)].copy()
    comparison_columns = [
        "channel",
        "method",
        "significant",
        "score",
        "p_value",
        "threshold",
        "n_methods_significant",
        "consensus_ch",
    ]
    for method in existing_methods:
        comparison_columns.extend(ep.DETECTOR_QUANTITY_COLUMNS[method])
    comparison_columns = list(dict.fromkeys(comparison_columns))
    pd.testing.assert_frame_equal(
        before[comparison_columns].reset_index(drop=True),
        after[comparison_columns].reset_index(drop=True),
        check_dtype=False,
    )


def test_keller_polarity_screen_rejects_saturation_like_deflection() -> None:
    sfreq = 1000.0
    times = np.arange(-0.2, 0.301, 1.0 / sfreq)
    channels = ["biphasic", "unidirectional", "control"]
    rng = np.random.default_rng(8)
    arr = rng.normal(0.0, 0.8, (20, len(times), len(channels)))
    negative = -75.0 * np.exp(-0.5 * ((times - 0.035) / 0.008) ** 2)
    positive = 55.0 * np.exp(-0.5 * ((times - 0.085) / 0.012) ** 2)
    arr[:, :, 0] += negative + positive
    plateau = np.zeros_like(times)
    plateau[(times >= 0.010) & (times <= 0.250)] = 90.0
    arr[:, :, 1] += plateau
    epochs = _epochs(arr, times, channels)

    detections = epochs.detect_erp_all(
        methods=["keller_zscore", "peak_amplitude"],
        min_consensus=1,
    )
    keller = detections[detections["method"].eq("keller_zscore")].set_index("channel")
    peak = detections[detections["method"].eq("peak_amplitude")].set_index("channel")

    assert bool(keller.loc["biphasic", "significant"])
    assert bool(keller.loc["biphasic", "keller_polarity_reversal"])
    assert not bool(keller.loc["unidirectional", "significant"])
    assert bool(keller.loc["unidirectional", "keller_saturation_like"])
    assert bool(peak.loc["unidirectional", "significant"])
    assert keller.loc["biphasic", "keller_max_z"] >= 6.0


def test_rms_reports_paired_quantity_effect_and_fdr() -> None:
    sfreq = 1000.0
    times = np.arange(-0.2, 0.301, 1.0 / sfreq)
    channels = ["response", "control"]
    rng = np.random.default_rng(74)
    arr = rng.normal(0.0, 1.0, (18, len(times), len(channels)))
    response = (times >= 0.010) & (times <= 0.100)
    arr[:, response, 0] += 12.0 * np.sin(2 * np.pi * 35.0 * times[response])
    epochs = _epochs(arr, times, channels)

    detections = epochs.detect_erp_all(methods=["rms_response"])
    rms = detections.set_index("channel")

    assert bool(rms.loc["response", "significant"])
    assert bool(rms.loc["response", "rms_paired_fdr_significant"])
    assert rms.loc["response", "rms_response_trial_median_uv"] > rms.loc["response", "rms_baseline_trial_median_uv"]
    assert rms.loc["response", "rms_wilcoxon_p_value"] < 0.05
    assert rms.loc["response", "rms_detection_n1_z"] >= 6.0
    assert not bool(rms.loc["control", "significant"])


def test_rms_paired_test_compares_separately_demeaned_segments() -> None:
    sfreq = 1000.0
    times = np.arange(-0.2, 0.301, 1.0 / sfreq)
    baseline = (times >= -0.096) & (times <= -0.006)
    response = (times >= 0.010) & (times <= 0.100)
    assert int(baseline.sum()) == int(response.sum())
    rng = np.random.default_rng(481)
    matched = rng.normal(0.0, 1.0, (12, int(response.sum())))
    values = np.zeros((12, len(times), 1), dtype=float)
    values[:, baseline, 0] = matched
    values[:, response, 0] = matched + 25.0
    epochs = _epochs(values, times, ["offset_only"])

    rms = epochs.detect_erp_all(
        methods=["rms_response"],
        baseline_window=(-0.096, -0.006),
        rms_response={"response_window": (0.010, 0.100)},
    ).iloc[0]

    assert rms["rms_response_trial_median_uv"] == pytest.approx(
        rms["rms_baseline_trial_median_uv"]
    )
    assert rms["rms_wilcoxon_p_value"] > 0.05
    assert not bool(rms["rms_paired_fdr_significant"])


def test_signi_randomization_is_deterministic_and_detects_induced_gamma() -> None:
    sfreq = 500.0
    times = np.arange(-0.25, 0.301, 1.0 / sfreq)
    channels = ["induced_gamma", "control"]
    rng = np.random.default_rng(19)
    arr = rng.normal(0.0, 0.35, (24, len(times), len(channels)))
    response = (times >= 0.010) & (times <= 0.100)
    response_times = times[response]
    burst_envelope = np.exp(-0.5 * ((response_times - 0.050) / 0.014) ** 2)
    phases = rng.uniform(0.0, 2 * np.pi, size=arr.shape[0])
    arr[:, response, 0] += 14.0 * burst_envelope[None, :] * np.sin(
        2 * np.pi * 100.0 * response_times[None, :] + phases[:, None]
    )
    epochs = _epochs(arr, times, channels)
    options = {
        "n_permutations": 120,
        "random_state": 31,
        "channel_chunk_size": 2,
        "permutation_chunk_size": 24,
    }

    unavailable = epochs.detect_erp_all(
        methods=["crowther_gamma"],
        crowther_gamma=options,
    ).set_index("channel")
    assert not unavailable["method_available"].any()
    assert unavailable["score"].isna().all()
    assert unavailable["notes"].str.contains("wideband_epochs").all()

    first = epochs.detect_erp_all(
        methods=["crowther_gamma"],
        wideband_epochs=epochs,
        crowther_gamma=options,
    ).set_index("channel")
    second = epochs.detect_erp_all(
        methods=["crowther_gamma"],
        wideband_epochs=epochs,
        crowther_gamma=options,
    ).set_index("channel")

    assert first.loc["induced_gamma", "signi_snr"] == pytest.approx(
        second.loc["induced_gamma", "signi_snr"]
    )
    assert first.loc["induced_gamma", "signi_p_value_raw"] == pytest.approx(
        second.loc["induced_gamma", "signi_p_value_raw"]
    )
    assert first.loc["induced_gamma", "signi_n_permutations"] == 120
    assert first.loc["induced_gamma", "signi_snr"] > first.loc["control", "signi_snr"]
    assert first.loc["induced_gamma", "signi_p_value_bonferroni"] < 0.05


def test_published_detector_diagnostic_is_public_and_renders() -> None:
    sfreq = 500.0
    times = np.arange(-0.2, 0.301, 1.0 / sfreq)
    channels = ["response", "reference1", "reference2"]
    rng = np.random.default_rng(91)
    arr = rng.normal(0.0, 0.3, (12, len(times), len(channels)))
    response = (times >= 0.005) & (times <= 0.100)
    waveform = np.sin(2 * np.pi * 55.0 * times[response])
    arr[:, response, 0] += 85.0 * waveform
    epochs = _epochs(arr, times, channels)
    detections = epochs.detect_erp_all(
        methods=["crp_significance", "crp_energy", "kundu_rolston", "keller_zscore", "peak_amplitude", "rms_response"],
        response_window=(0.015, 0.100),
        crp_energy={"canonical_energy_cv": False},
    )

    fig = epochs.plot.detectors(
        "response",
        detections=detections,
        response_window=(0.015, 0.100),
    )

    assert len(fig.axes) == 6
    assert "CRP-RMS reproducibility-energy conjunction" in fig._suptitle.get_text()
    plt.close(fig)


def test_detector_diagnostic_renders_insufficient_matched_trials() -> None:
    sfreq = 500.0
    times = np.arange(-0.25, 0.201, 1.0 / sfreq)
    rng = np.random.default_rng(915)
    arr = rng.normal(0.0, 0.3, (10, len(times), 2))
    arr[:, (times >= 0.015) & (times <= 0.200), 0] = np.nan
    epochs = _epochs(arr, times, ["masked", "control"])
    detections = ep.detect(
        epochs,
        method="crp_energy",
        baseline_window=(-0.20, -0.015),
        response_window=(0.015, 0.200),
        return_arrays=False,
    )

    fig = epochs.plot.detectors(
        "masked",
        detections=detections,
        baseline_window=(-0.20, -0.015),
        response_window=(0.015, 0.200),
    )

    assert "FAIL" in fig._suptitle.get_text()
    assert any(
        "CRP unavailable" in text.get_text()
        for text in fig.axes[2].texts
    )
    assert "RMS unavailable" in fig.axes[4].get_title()
    plt.close(fig)


def test_every_published_detector_exposes_its_native_quantities() -> None:
    sfreq = 500.0
    times = np.arange(-0.25, 0.301, 1.0 / sfreq)
    channels = ["response", "control1", "control2"]
    rng = np.random.default_rng(92)
    arr = rng.normal(0.0, 0.4, (14, len(times), len(channels)))
    response = (times >= 0.010) & (times <= 0.100)
    arr[:, response, 0] += 80.0 * np.sin(2 * np.pi * 55.0 * times[response])
    epochs = _epochs(arr, times, channels)
    detections = epochs.detect_erp_all(
        response_window=(0.015, 0.100),
        wideband_epochs=epochs,
        crp_energy={"canonical_energy_cv": False},
        crowther_gamma={
            "n_permutations": 24,
            "permutation_chunk_size": 12,
            "channel_chunk_size": 3,
        },
    )

    for method, quantity_columns in ep.DETECTOR_QUANTITY_COLUMNS.items():
        rows = detections[detections["method"].eq(method)]
        assert len(rows) == len(channels)
        assert set(quantity_columns).issubset(rows.columns)
        assert rows[list(quantity_columns)].notna().any(axis=None)

    hays = detections[detections["method"].eq("peak_amplitude")].set_index("channel")
    assert hays.loc["response", "hays_n1_z"] == pytest.approx(
        hays.loc["response", "keller_a1_z"]
    )
