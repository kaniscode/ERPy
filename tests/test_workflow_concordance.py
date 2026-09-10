"""Regression checks for workflow availability and QC decision contracts."""

import numpy as np
import pandas as pd
import pytest

from ERPy.epochs import Epochs
from ERPy.erp_detection import DETECTOR_QUANTITY_COLUMNS
from ERPy.waveform_qc import audit_waveforms
import ERPy.waveform_qc as waveform_qc


def _epochs(values, times, channels):
    return Epochs(
        pd.DataFrame(
            values.reshape(-1, len(channels)),
            index=pd.MultiIndex.from_product(
                [range(values.shape[0]), times], names=["epoch", "time"]
            ),
            columns=channels,
        ),
        sfreq=float(1.0 / np.median(np.diff(times))),
        tmin=float(times[0]),
        tmax=float(times[-1]),
        baseline=(float(times[0]), -0.03),
    )


@pytest.mark.parametrize("field", ["peak_amplitude_uv", "peak_latency_ms"])
@pytest.mark.parametrize("source", ["stored", "recomputed"])
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_waveform_qc_rejects_nonfinite_peak_features(monkeypatch, field, source, invalid):
    times = np.arange(-100, 81) / 200.0
    rng = np.random.default_rng(17)
    values = rng.normal(scale=0.1, size=(12, len(times), 1))
    values[:, :, 0] += 80 * np.exp(-((times - 0.08) / 0.025) ** 2)
    epochs = _epochs(values, times, ["response"])
    detections = epochs.detect_erp_all(
        methods=["peak_amplitude"],
        response_window=(0.01, 0.35),
        baseline_window=(-0.5, -0.03),
    )
    if source == "stored":
        detections[field] = invalid
    else:
        original = waveform_qc._feature_table_from_array

        def nonfinite_recomputed(*args, **kwargs):
            result = original(*args, **kwargs)
            result[field] = invalid
            return result

        monkeypatch.setattr(waveform_qc, "_feature_table_from_array", nonfinite_recomputed)
    result = audit_waveforms(epochs, detections, pd.DataFrame()).summary.iloc[0]
    assert result["feature_mismatch"]
    assert not result["analysis_eligible"]
    assert "recomputed_metric_mismatch" in result["review_reasons"]


@pytest.mark.parametrize("failure", ["sampling_rate", "coverage", "filter"])
def test_signi_failures_remain_unavailable_in_public_result(monkeypatch, failure):
    sfreq = 250.0 if failure == "sampling_rate" else 500.0
    start = -0.02 if failure == "coverage" else -0.25
    times = np.arange(start, 0.301, 1.0 / sfreq)
    values = np.random.default_rng(19).normal(size=(12, len(times), 1))
    epochs = _epochs(values, times, ["contact"])
    if failure == "filter":
        def fail_quantities(*args, **kwargs):
            raise ValueError("filter numerical failure")

        monkeypatch.setattr("ERPy.erp_detection._signi_quantities", fail_quantities)
    result = epochs.detect_erp_all(
        methods=["crowther_gamma"],
        wideband_epochs=epochs,
        baseline_window=(start, -0.004),
        crowther_gamma={"n_permutations": 32},
    )
    assert not result["method_available"].any()
    assert not result["significant"].any()
    assert result["n_methods_available"].eq(0).all()
    assert not result["consensus"].any()
    assert result["availability_reason"].str.len().gt(0).all()
    assert result[list(DETECTOR_QUANTITY_COLUMNS["crowther_gamma"])].isna().all().all()


def test_public_detector_keeps_nonsignificant_contact_when_description_fails():
    times = np.arange(-16, 17) / 1000.0
    response_indices = np.flatnonzero(times >= 0.001)
    values = np.zeros((8, len(times), 2))
    wave = np.r_[np.zeros(10), np.arange(1, 7)]
    values[:, response_indices, 0] = wave
    values[:, response_indices, 1] = np.r_[np.ones(4), -np.ones(4)][:, None] * wave
    epochs = _epochs(values, times, ["coherent", "balanced_polarity"])
    result = epochs.detect_erp_all(
        methods=["crp_energy"],
        baseline_window=(-0.016, -0.001),
        response_window=(0.001, 0.016),
        crp_energy={"artifact_interval": (0.0, 0.0), "canonical_energy_cv": False},
    ).set_index("channel")
    assert result["testing_family_size"].eq(2).all()
    assert result.loc["balanced_polarity", "p_joint"] == 1.0
    assert result.loc["balanced_polarity", "qc_status"] == "pass"
    assert np.isnan(result.loc["balanced_polarity", "response_duration"])
    assert result.loc["coherent", "q_joint"] == pytest.approx(2 / 128)
    assert result.loc["coherent", "significant"]
    assert not result.loc["balanced_polarity", "significant"]
