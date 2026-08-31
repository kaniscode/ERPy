import numpy as np
import pandas as pd

from ERPy.epochs import Epochs
from ERPy.waveform_qc import (
    audit_waveforms,
    common_mode_recovery_diagnostic,
)


def make_epochs():
    sfreq = 200.0
    times = np.arange(-0.5, 0.505, 1.0 / sfreq)
    rng = np.random.default_rng(19)
    rows = []
    index = []
    for epoch in range(8):
        response = 80.0 * np.exp(-((times - 0.08) / 0.025) ** 2)
        rows.append(
            np.column_stack(
                [
                    response
                    + epoch * 0.1
                    + rng.normal(scale=0.08, size=len(times)),
                    np.sin(2 * np.pi * 5 * times)
                    + rng.normal(scale=0.03, size=len(times)),
                ]
            )
        )
        index.extend((epoch, float(time)) for time in times)
    frame = pd.DataFrame(
        np.concatenate(rows, axis=0),
        index=pd.MultiIndex.from_tuples(index, names=["epoch", "time"]),
        columns=["A1", "B1"],
    )
    return Epochs(
        frame,
        sfreq=sfreq,
        tmin=-0.5,
        tmax=0.5,
        baseline=(-0.5, -0.03),
    )


def test_waveform_audit_recomputes_metrics_and_writes_review(tmp_path):
    epochs = make_epochs()
    detections = epochs.detect_erp_all(
        methods=["keller_zscore", "crp_significance", "peak_amplitude"],
        response_window=(0.01, 0.35),
        baseline_window=(-0.5, -0.03),
        min_consensus=2,
    )
    artifact_rows = []
    for epoch in range(epochs.n_trials()):
        for channel in epochs.channels:
            artifact_rows.append(
                {
                    "epoch": epoch,
                    "channel": channel,
                    "peak_abs_z": 0.0,
                    "ptp_z": 0.0,
                    "max_gradient_z": 0.0,
                    "late_high_frequency_ratio_z": 0.0,
                    "reason": "",
                    "bad_response": False,
                    "hard_artifact_response": False,
                }
            )
    artifacts = pd.DataFrame(artifact_rows)

    audit = audit_waveforms(
        epochs,
        detections,
        artifacts,
        min_clean_responses=3,
    )
    assert set(audit.summary["channel"]) == {"A1", "B1"}
    assert not audit.summary["feature_mismatch"].any()
    assert audit.summary["crp_q_value"].notna().all()
    assert "strict_detector_pattern" in audit.summary
    assert "strict_detector_consensus" in audit.summary
    eligible_row = audit.summary.set_index("channel").loc["A1"]
    assert bool(eligible_row["artifact_eligible"])
    assert bool(eligible_row["analysis_eligible"])

    mean_waveforms = audit.mean_waveforms(step_ms=10.0)
    assert list(mean_waveforms.columns) == ["A1", "B1"]
    assert mean_waveforms.index.min() >= -100.0
    assert mean_waveforms.index.max() <= 450.0

    output = audit.to_pdf(tmp_path / "review.pdf", channels_per_page=1)
    assert output.is_file()
    assert output.stat().st_size > 0


def test_waveform_audit_keeps_crp_energy_out_of_legacy_consensus() -> None:
    epochs = make_epochs()
    detections = epochs.detect_erp_all(
        methods=[
            "kundu_rolston",
            "crp_significance",
            "crp_energy",
            "peak_amplitude",
        ],
        response_window=(0.01, 0.35),
        baseline_window=(-0.5, -0.03),
        min_consensus=2,
        crp_energy={
            "canonical_energy_cv": False,
            "return_arrays": False,
        },
    )

    audit = audit_waveforms(
        epochs,
        detections,
        pd.DataFrame(),
        min_clean_responses=3,
    )

    assert "detector_crp_energy" in audit.summary
    assert "strict_detector_crp_energy" in audit.summary
    assert not audit.summary["strict_detector_crp_energy"].any()
    assert "primary_detector_pass" in audit.summary
    assert "primary_qc_pass" in audit.summary


def test_waveform_audit_flags_stored_metric_disagreement():
    epochs = make_epochs()
    detections = epochs.detect_erp_all(
        methods=["keller_zscore", "crp_significance"],
        response_window=(0.01, 0.35),
        baseline_window=(-0.5, -0.03),
        min_consensus=2,
    )
    detections.loc[detections["channel"].eq("A1"), "peak_latency_ms"] += 10.0
    artifacts = pd.DataFrame(
        [
            {
                "epoch": epoch,
                "channel": channel,
                "reason": "",
                "bad_response": False,
                "hard_artifact_response": False,
            }
            for epoch in range(epochs.n_trials())
            for channel in epochs.channels
        ]
    )

    audit = audit_waveforms(epochs, detections, artifacts)
    row = audit.summary.set_index("channel").loc["A1"]
    assert bool(row["feature_mismatch"])
    assert not bool(row["analysis_eligible"])
    assert not bool(row["strict_consensus"])
    assert int(row["review_priority"]) == 3


def test_waveform_audit_gates_detector_consensus_at_artifact_cutoff():
    epochs = make_epochs()
    detections = epochs.detect_erp_all(
        methods=["crp_significance", "peak_amplitude"],
        response_window=(0.01, 0.35),
        baseline_window=(-0.5, -0.03),
        min_consensus=2,
    )
    artifact_rows = []
    for epoch in range(epochs.n_trials()):
        for channel in epochs.channels:
            flagged = channel == "A1" and epoch < 2
            artifact_rows.append(
                {
                    "epoch": epoch,
                    "channel": channel,
                    "reason": "sharp_transient" if flagged else "",
                    "bad_response": flagged,
                    "hard_artifact_response": False,
                }
            )

    audit = audit_waveforms(
        epochs,
        detections,
        pd.DataFrame(artifact_rows),
        max_bad_response_fraction=0.25,
        min_clean_responses=3,
    )
    row = audit.summary.set_index("channel").loc["A1"]
    assert bool(row["strict_detector_consensus"])
    assert row["bad_response_fraction"] == 0.25
    assert not bool(row["artifact_eligible"])
    assert not bool(row["analysis_eligible"])
    assert not bool(row["strict_consensus"])
    assert "detector_consensus_excluded_by_qc" in row["review_reasons"]


def test_erp_metrics_and_waveform_audit_are_invariant_to_trial_dc_offsets():
    sfreq = 500.0
    times = np.arange(-0.2, 0.402, 1.0 / sfreq)
    rng = np.random.default_rng(47)
    response = 75.0 * np.exp(-0.5 * ((times - 0.09) / 0.018) ** 2)
    base_rows = []
    shifted_rows = []
    index = []
    for epoch in range(12):
        trial = response + rng.normal(scale=0.6, size=len(times))
        base_rows.append(trial[:, None])
        shifted_rows.append((trial + 400.0 + 17.0 * epoch)[:, None])
        index.extend((epoch, float(time)) for time in times)

    def _epochs(rows):
        return Epochs(
            pd.DataFrame(
                np.vstack(rows),
                index=pd.MultiIndex.from_tuples(
                    index,
                    names=["epoch", "time"],
                ),
                columns=["A1"],
            ),
            sfreq=sfreq,
            tmin=float(times[0]),
            tmax=float(times[-1]),
            baseline=(-0.2, -0.03),
        )

    reference = _epochs(base_rows)
    shifted = _epochs(shifted_rows)
    methods = [
        "kundu_rolston",
        "keller_zscore",
        "peak_amplitude",
        "rms_response",
    ]
    reference_detection = reference.detect_erp_all(
        methods=methods,
        response_window=(0.01, 0.35),
        min_consensus=2,
    ).sort_values("method")
    shifted_detection = shifted.detect_erp_all(
        methods=methods,
        response_window=(0.01, 0.35),
        min_consensus=2,
    ).sort_values("method")

    np.testing.assert_allclose(
        reference_detection[
            [
                "score",
                "peak_amplitude_uv",
                "peak_latency_ms",
                "rms_uv",
                "auc_uv_ms",
            ]
        ],
        shifted_detection[
            [
                "score",
                "peak_amplitude_uv",
                "peak_latency_ms",
                "rms_uv",
                "auc_uv_ms",
            ]
        ],
        rtol=1e-10,
        atol=1e-10,
    )
    assert (
        reference_detection["significant"].to_numpy()
        == shifted_detection["significant"].to_numpy()
    ).all()

    audit = audit_waveforms(
        shifted,
        shifted_detection,
        pd.DataFrame(),
        min_clean_responses=10,
    )
    baseline = audit.mean_waveforms(
        start_s=-0.2,
        stop_s=-0.03,
        step_ms=2.0,
    )
    assert abs(float(baseline["A1"].mean())) < 1e-10


def test_baseline_restoration_excludes_post_artifact_recovery_peak():
    sfreq = 500.0
    times = np.arange(-0.2, 0.402, 1.0 / sfreq)
    response_mask = (times >= 0.01) & (times <= 0.35)
    rows = []
    index = []
    for epoch in range(12):
        waveform = np.full(len(times), 1000.0 + epoch)
        waveform[response_mask] = np.linspace(
            -400.0 + epoch,
            800.0 + epoch,
            int(response_mask.sum()),
        )
        rows.append(waveform[:, None])
        index.extend((epoch, float(time)) for time in times)
    epochs = Epochs(
        pd.DataFrame(
            np.vstack(rows),
            index=pd.MultiIndex.from_tuples(
                index,
                names=["epoch", "time"],
            ),
            columns=["recovery"],
        ),
        sfreq=sfreq,
        tmin=float(times[0]),
        tmax=float(times[-1]),
        baseline=(-0.2, -0.03),
    )
    detections = epochs.detect_erp_all(
        methods=["kundu_rolston", "peak_amplitude"],
        response_window=(0.01, 0.35),
        min_consensus=2,
    )
    audit = audit_waveforms(
        epochs,
        detections,
        pd.DataFrame(),
        response_window=(0.01, 0.35),
        min_clean_responses=10,
    )
    row = audit.summary.set_index("channel").loc["recovery"]

    assert np.isclose(row["peak_latency_ms_recomputed"], 10.0)
    assert bool(row["peak_at_response_boundary_recomputed"])
    assert not bool(row["analysis_eligible"])
    assert not bool(row["strict_consensus"])


def test_unknown_rail_diagnostic_does_not_make_waveform_borderline():
    epochs = make_epochs()
    detections = epochs.detect_erp_all(
        methods=["keller_zscore", "peak_amplitude"],
        response_window=(0.01, 0.35),
        baseline_window=(-0.5, -0.03),
        min_consensus=2,
    )
    artifact_rows = []
    for epoch in range(epochs.n_trials()):
        for channel in epochs.channels:
            artifact_rows.append(
                {
                    "epoch": epoch,
                    "channel": channel,
                    "peak_abs_z": 0.0,
                    "ptp_z": 0.0,
                    "max_gradient_z": 0.0,
                    "late_high_frequency_ratio_z": 0.0,
                    "saturation_fraction": 0.0,
                    "plateau_fraction": 0.0,
                    "plateau_run_s": 0.0,
                    "rail_fraction": 0.02,
                    "rail_limits_known": False,
                    "reason": "",
                    "bad_response": False,
                    "hard_artifact_response": False,
                }
            )

    audit = audit_waveforms(
        epochs,
        detections,
        pd.DataFrame(artifact_rows),
        artifact_thresholds={
            "zscore_threshold": 6.0,
            "late_high_frequency_zscore_threshold": 6.0,
            "saturation_fraction": 0.02,
            "plateau_fraction": 0.015,
            "plateau_min_run_s": 0.003,
            "rail_fraction": 0.02,
        },
        min_clean_responses=3,
    )

    assert not audit.summary["review_reasons"].str.contains(
        "artifact_feature_near_threshold",
        na=False,
    ).any()


def test_common_mode_recovery_diagnostic_requires_shared_early_step():
    channels = [f"C{index:02d}" for index in range(24)]
    contaminated = pd.DataFrame(
        {
            "channel": channels,
            "strict_detector_consensus": True,
            "peak_latency_ms_recomputed": 12.5,
            "response_start_uv_recomputed": -180.0,
            "response_drift_uv_recomputed": 65.0,
        }
    )
    diagnostic = common_mode_recovery_diagnostic(contaminated)

    assert diagnostic["common_mode_recovery_artifact"]
    assert diagnostic["common_mode_detector_channels"] == 24
    assert diagnostic["common_mode_early_channels"] == 24

    physiological = contaminated.copy()
    physiological["peak_latency_ms_recomputed"] = np.linspace(
        20.0,
        180.0,
        len(physiological),
    )
    assert not common_mode_recovery_diagnostic(physiological)[
        "common_mode_recovery_artifact"
    ]


def test_waveform_audit_quarantines_common_mode_recovery_acquisition():
    sfreq = 2000.0
    times = np.arange(-0.1, 0.401, 1.0 / sfreq)
    channels = [f"C{index:02d}" for index in range(24)]
    rng = np.random.default_rng(9)
    rows = []
    index = []
    for epoch in range(12):
        trial_channels = []
        for channel_index in range(len(channels)):
            waveform = rng.normal(0.0, 0.4, len(times))
            response = times >= 0.01
            relative_time = times[response] - 0.01
            recovery = (
                -180.0
                + 120.0 * (1.0 - np.exp(-relative_time / 0.15))
                - 25.0
                * np.exp(
                    -0.5
                    * ((times[response] - 0.0125) / 0.0007) ** 2
                )
            )
            channel_scale = 0.9 + 0.2 * channel_index / (
                len(channels) - 1
            )
            waveform[response] += recovery * channel_scale
            trial_channels.append(waveform)
        rows.append(np.column_stack(trial_channels))
        index.extend((epoch, float(time)) for time in times)
    epochs = Epochs(
        pd.DataFrame(
            np.vstack(rows),
            index=pd.MultiIndex.from_tuples(
                index,
                names=["epoch", "time"],
            ),
            columns=channels,
        ),
        sfreq=sfreq,
        tmin=float(times[0]),
        tmax=float(times[-1]),
        baseline=(-0.1, -0.03),
    )
    detections = epochs.detect_erp_all(
        methods=["keller_zscore", "peak_amplitude", "rms_response"],
        response_window=(0.01, 0.35),
        baseline_window=(-0.1, -0.03),
        min_consensus=2,
    )

    audit = audit_waveforms(
        epochs,
        detections,
        pd.DataFrame(),
        response_window=(0.01, 0.35),
        baseline_window=(-0.1, -0.03),
        min_clean_responses=10,
    )

    assert audit.summary["strict_detector_consensus"].all()
    assert audit.summary["common_mode_recovery_artifact"].all()
    assert not audit.summary["analysis_eligible"].any()
    assert not audit.summary["strict_consensus"].any()
    assert audit.summary["review_reasons"].str.contains(
        "common_mode_recovery_artifact"
    ).all()
