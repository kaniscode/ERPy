"""The convenience layer of the public API: reprs, aliases and shortcuts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

import ERPy as ep
from ERPy.pipeline import STANDARD_PIPELINES


def make_epochs(n_trials: int = 6, n_channels: int = 3, sfreq: float = 200.0) -> ep.Epochs:
    times = np.round(np.arange(-0.25, 0.5 + 1 / sfreq, 1 / sfreq), 9)
    rng = np.random.default_rng(0)
    index = pd.MultiIndex.from_product([range(n_trials), times], names=["epoch", "time"])
    channels = [f"A{number}" for number in range(1, n_channels + 1)]
    frame = pd.DataFrame(rng.normal(size=(len(index), n_channels)), index=index, columns=channels)
    return ep.Epochs(frame, sfreq=sfreq, tmin=-0.25, tmax=0.5, baseline=(-0.25, -0.05), stim_ch=["A1", "A2"])


@pytest.fixture
def patient(tmp_path) -> ep.Patient:
    sfreq = 100.0
    seconds = np.arange(1000) / sfreq
    timestamps = pd.Timestamp("2026-01-01") + pd.to_timedelta(seconds, unit="s")
    pd.DataFrame(
        {"times": timestamps, "A1": np.sin(seconds), "A2": np.cos(seconds), "B1": np.ones(len(seconds))}
    ).to_csv(tmp_path / "signal.csv", index=False)
    tables = {
        "rawdata_meta_path": [dict(patient_id="P1", session_id="A", raw_file="signal.csv", sampling_freq=sfreq)],
        "stim_meta_path": [
            dict(patient_id="P1", session_id="A", stim_pair="A_1_2", stim_start=0, stim_stop=10, stim_freq=0.5)
        ],
        "elec_meta_path": [dict(patient_id="P1", session_id="A", elec_label="B1", bad_channel=False)],
    }
    config = {"rawdata_path": str(tmp_path), "procdata_path": str(tmp_path / "proc")}
    for key, rows in tables.items():
        path = tmp_path / f"{key}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        config[key] = str(path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return ep.Patient("P1", config_path=str(config_path), verbose=False)


def test_epochs_repr_and_shape_describe_the_data():
    epochs = make_epochs()

    assert epochs.shape == (6, 151, 3)
    assert epochs.shape == epochs.as_array()[0].shape
    assert repr(epochs) == "<Epochs | 6 trials x 151 samples x 3 channels | -0.25 to 0.5 s at 200 Hz | stim A1, A2>"


def test_epochs_summary_methods_match_their_retained_aliases():
    epochs = make_epochs()

    pd.testing.assert_frame_equal(epochs.mean(), epochs.get_mean_waveform())
    pd.testing.assert_frame_equal(epochs.std(), epochs.get_std_waveform())
    pd.testing.assert_frame_equal(epochs.sem(), epochs.get_sem_waveform())
    assert list(epochs.mean().columns) == epochs.channels
    assert len(epochs.mean()) == epochs.times.size


def test_epochs_transformations_return_the_same_type_without_mutating():
    epochs = make_epochs()
    original = epochs.epochs_df.copy()

    scaled = epochs.rescale(2.0, reason="unit test")
    rereferenced = epochs.rereference(method="mean", min_reference_channels=1)

    assert type(scaled) is ep.Epochs and type(rereferenced) is ep.Epochs
    np.testing.assert_allclose(scaled.epochs_df.to_numpy(), 2.0 * original.to_numpy())
    pd.testing.assert_frame_equal(epochs.epochs_df, original)
    assert (scaled.sfreq, scaled.tmin, scaled.tmax, scaled.baseline, scaled.stim_ch) == (
        epochs.sfreq,
        epochs.tmin,
        epochs.tmax,
        epochs.baseline,
        epochs.stim_ch,
    )
    assert scaled.metadata["value_scale_reason"] == "unit test"
    assert "value_scale_reason" not in epochs.metadata


def test_epochs_detect_shortcuts_match_the_module_functions():
    epochs = make_epochs(n_trials=10)
    options = dict(baseline_window=(-0.25, -0.05), response_window=(0.01, 0.3))

    one = epochs.detect("peak_amplitude", **options)
    pd.testing.assert_frame_equal(one, ep.detect(epochs, method="peak_amplitude", **options))
    pd.testing.assert_frame_equal(one, epochs.detect_erp("peak_amplitude", **options))

    methods = ["keller_zscore", "peak_amplitude"]
    combined = epochs.detect_all(methods=methods, min_consensus=1, **options)
    pd.testing.assert_frame_equal(combined, epochs.detect_erp_all(methods=methods, min_consensus=1, **options))
    assert set(combined["method"]) == set(methods)


def test_standard_pipelines_are_listed_and_returned_as_independent_copies(patient):
    pipeline = patient.pipeline

    assert "blank_filt" in ep.STANDARD_PIPELINES and ep.STANDARD_PIPELINES is STANDARD_PIPELINES
    steps = pipeline.get_standard_pipeline("blank_filt")
    assert steps == STANDARD_PIPELINES["blank_filt"]
    steps[0][1]["width_s"] = 99.0
    assert pipeline.get_standard_pipeline("blank_filt")[0][1]["width_s"] == 0.003

    assert pipeline.resolve_pipeline("blank_filt") == pipeline.get_standard_pipeline("blank_filt")
    assert pipeline.resolve_pipeline([("car_ref", None)]) == [("car_ref", {})]
    with pytest.raises(KeyError, match="Unknown standard pipeline"):
        pipeline.get_standard_pipeline("missing")
    with pytest.raises(KeyError, match="Unknown pipeline step"):
        pipeline.resolve_pipeline([("missing", {})])


def test_patient_navigation_and_reprs(patient):
    assert repr(patient) == "<Patient P1 | sessions: A>"
    assert repr(patient.dataloader) == "<DataLoader P1 | sessions: A>"
    assert patient.session_ids == patient.sessions() == ["A"]
    assert patient.stim_pairs() == patient.stim_pairs("A") == ["A_1_2"]

    recording = patient["A"]
    assert recording == patient.recording("A") == patient.session("A")
    assert recording.stim_pairs() == ["A_1_2"]
    assert ep.Study is ep.Patient and ep.Session is ep.Recording


def test_recording_event_detection_names_are_interchangeable(patient, monkeypatch):
    calls = []

    def fake_detector(dataloader, session_id, stim_pair, **kwargs):
        calls.append((session_id, stim_pair, kwargs))
        return pd.DataFrame({"times": [1.0]})

    monkeypatch.setattr("ERPy.facade.detect_events_from_artifacts", fake_detector)
    recording = patient["A"]

    recording.detect_events("A_1_2", method="adjacent", stim_freq=0.5)
    recording.events_for("A_1_2", method="adjacent", stim_freq=0.5)
    patient.detect_events("A", "A_1_2", method="adjacent", stim_freq=0.5)

    assert calls[0] == calls[1] == calls[2]
    assert calls[0] == ("A", "A_1_2", {"method": "adjacent", "save_events": True, "stim_freq": 0.5})


def test_dataloader_snake_case_attributes_keep_their_upper_case_aliases(patient, tmp_path):
    loader = patient.dataloader
    aliases = {
        "CONFIG_PATH": "config_path",
        "RAWDATA_PATH": "rawdata_path",
        "PROCDATA_PATH": "procdata_path",
        "RAW_META_PATH": "raw_meta_path",
        "STIM_META_PATH": "stim_meta_path",
        "ELEC_META_PATH": "elec_meta_path",
        "EVENT_PATH": "event_path",
        "RAW_CACHE_FORMAT": "raw_cache_format",
        "RAW_CACHE_DTYPE": "raw_cache_dtype",
        "RAW_STAGING_PATH": "raw_staging_path",
        "RAW_META_DF": "all_raw_meta",
        "STIM_META_DF": "all_stim_meta",
        "ELEC_META_DF": "all_elec_meta",
        "PATIENT_PATH": "patient_path",
        "SESSION_PATHS": "session_paths",
    }
    for legacy, current in aliases.items():
        assert getattr(loader, legacy) is getattr(loader, current), legacy

    assert loader.rawdata_path == str(tmp_path)
    assert loader.patient_path == str(tmp_path / "proc" / "P1")
    loader.RAW_CACHE_FORMAT = "npz"
    assert loader.raw_cache_format == "npz"


def test_analysis_result_repr_and_saved_table_names(patient, tmp_path):
    events = patient.dataloader.get_event_path("A", "A_1_2")
    times = pd.Timestamp("2026-01-01") + pd.to_timedelta([2.0, 4.0, 6.0], unit="s")
    pd.DataFrame({"times": times}).to_csv(events, index=False)

    result = patient.analyze(
        "A",
        "A_1_2",
        pipeline="raw",
        tmin=-0.1,
        tmax=0.2,
        baseline=None,
        zero_time=None,
        methods=["peak_amplitude"],
        min_consensus=1,
        detect_kwargs={"baseline_window": (-0.1, -0.01), "response_window": (0.01, 0.2)},
    )

    assert repr(result) == "<AnalysisResult P1/A/A_1_2 | 3 trials, 3 channels | 0 primary responses after QC>"
    saved = result.save(tmp_path / "tables")
    assert {path.name for path in saved.values()} == {
        "P1_A_A_1_2_detections.csv",
        "P1_A_A_1_2_detections_qc.csv",
        "P1_A_A_1_2_detections_qc_pass.csv",
        "P1_A_A_1_2_detections_primary.csv",
        "P1_A_A_1_2_artifact_response_table.csv",
        "P1_A_A_1_2_artifact_channel_summary.csv",
        "P1_A_A_1_2_zero_anchor_report.csv",
        "P1_A_A_1_2_metadata.json",
    }


def test_legacy_script_hooks_remain_inert_on_every_workflow_class(patient):
    for owner in (ep.Pipeline, ep.Analysis, ep.Events, type(patient.pipeline.epochs)):
        owner.use_standard_scripts()
        assert owner.install_standard_scripts() is None
        assert owner.PYTHON_SCRIPT is None and owner.BASH_SCRIPT is None
    assert ep.Events(patient.dataloader).dataloader is patient.dataloader


def test_version_is_single_sourced():
    from importlib import metadata

    from ERPy import _version

    assert ep.__version__ == _version.__version__
    assert ep.CHECKPOINT_COMPATIBILITY_ID == _version.CHECKPOINT_COMPATIBILITY_ID
    assert metadata.version("erpy-neuro") == ep.__version__
