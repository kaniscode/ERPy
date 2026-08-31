from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import threading
import zlib

import numpy as np
import pandas as pd
import pytest
import yaml

import ERPy as ep
import ERPy.events as event_detection
from ERPy.erp_detection import _trapezoid
from ERPy.facade import _annotate_detection_artifact_qc
from ERPy.utils import preproc


def _write_fixture(tmp_path: Path):
    root = tmp_path / "data"
    raw = root / "raw"
    meta = root / "metadata"
    proc = root / "proc"
    raw.mkdir(parents=True)
    meta.mkdir(parents=True)
    proc.mkdir(parents=True)

    sfreq = 200.0
    start = pd.Timestamp("2026-01-01T00:00:00")
    times = pd.date_range(start, periods=int(10 * sfreq), freq=pd.to_timedelta(1 / sfreq, unit="s"))
    events = [start + pd.to_timedelta(s, unit="s") for s in (2.0, 4.0, 6.0, 8.0)]
    rng = np.random.default_rng(8)
    df = pd.DataFrame(
        {
            "times": times,
            "LACC1": rng.normal(0, 0.2, len(times)),
            "LACC2": rng.normal(0, 0.2, len(times)),
            "LACC3": rng.normal(0, 0.2, len(times)),
            "RSGC1": rng.normal(0, 0.2, len(times)),
            "BAD1": 1000.0,
        }
    )
    for ev in events:
        i = int(np.argmin(np.abs(times - ev)))
        df.loc[i, "LACC3"] += 25
        r0 = i + int(0.03 * sfreq)
        r1 = i + int(0.16 * sfreq)
        df.loc[r0:r1, "RSGC1"] += -5 * np.hanning(r1 - r0 + 1)
    raw_csv = raw / "fixture.csv"
    df.to_csv(raw_csv, index=False)

    pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "session_id": "A",
                "raw_file": "fixture.csv",
                "start_time": start.isoformat(),
                "stop_time": (start + pd.to_timedelta(10, unit="s")).isoformat(),
                "sampling_freq": sfreq,
            }
        ]
    ).to_csv(meta / "raw_metadata.csv", index=False)
    pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "session_id": "A",
                "stim_pair": "LACC_1_2",
                "stim_start": start.isoformat(),
                "stim_stop": (start + pd.to_timedelta(10, unit="s")).isoformat(),
                "stim_freq": 0.5,
            }
        ]
    ).to_csv(meta / "stim_metadata.csv", index=False)
    pd.DataFrame(
        [
            {"patient_id": "P1", "session_id": "A", "elec_label": "LACC1", "anat_label": "L ACC", "mni_x": -4, "mni_y": 28, "mni_z": 30, "bad_channel": False},
            {"patient_id": "P1", "session_id": "A", "elec_label": "LACC2", "anat_label": "L ACC", "mni_x": -3, "mni_y": 27, "mni_z": 31, "bad_channel": False},
            {"patient_id": "P1", "session_id": "A", "elec_label": "LACC3", "anat_label": "L ACC", "mni_x": -2, "mni_y": 26, "mni_z": 32, "bad_channel": False},
            {"patient_id": "P1", "session_id": "A", "elec_label": "BAD1", "anat_label": "bad", "mni_x": 0, "mni_y": 0, "mni_z": 0, "bad_channel": True},
            {"patient_id": "P1", "session_id": "A", "elec_label": "RSGC1", "anat_label": "R SGC", "mni_x": 8, "mni_y": 18, "mni_z": -4, "bad_channel": False},
        ]
    ).to_csv(meta / "electrode_metadata.csv", index=False)
    config = {
        "rawdata_path": str(raw),
        "procdata_path": str(proc),
        "rawdata_meta_path": str(meta / "raw_metadata.csv"),
        "stim_meta_path": str(meta / "stim_metadata.csv"),
        "elec_meta_path": str(meta / "electrode_metadata.csv"),
        "session_ids": {"P1": ["A"]},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path, events


def test_notch_line_skips_frequencies_at_or_above_nyquist():
    sfreq = 100.0
    times = np.arange(400, dtype=float) / sfreq
    frame = pd.DataFrame(
        {"LACC1": np.sin(2 * np.pi * 10 * times)},
        index=times,
    )
    frame.attrs["sfreq"] = sfreq

    filtered = preproc.notch_line(
        frame,
        notch_freq=60.0,
        fs=sfreq,
        harm=True,
    )

    pd.testing.assert_frame_equal(filtered, frame)
    assert filtered.attrs == frame.attrs


def test_default_trigger_matching_does_not_treat_striatal_contacts_as_triggers():
    frame = pd.DataFrame(
        {
            "LSTR1": np.sin(np.linspace(0, 20, 500)),
            "RSTR2": np.cos(np.linspace(0, 20, 500)),
        }
    )
    frame.attrs["sfreq"] = 100.0

    with pytest.raises(ValueError, match="No trigger channels matched"):
        event_detection._detect_trigger_events(frame)


def test_auto_event_detection_reports_the_plausible_detector(tmp_path):
    config_path, _ = _write_fixture(tmp_path)
    dataloader = ep.DataLoader("P1", config_path=config_path)

    events = ep.detect_events_from_artifacts(
        dataloader,
        "A",
        "LACC_1_2",
        method="auto",
        stim_freq=0.5,
        min_peak_height=10.0,
        raw_source="raw",
        save_events=False,
    )

    assert len(events) == 4
    assert events["detection_method"].eq("adjacent").all()
    assert events["auto_validation"].eq("pulse_train_consistent").all()


def _write_low_amplitude_event_fixture(tmp_path: Path, event_offsets: list[float]):
    config_path, _ = _write_fixture(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    sfreq = 200.0
    start = pd.Timestamp("2026-01-01T00:00:00")
    stim_start = start + pd.to_timedelta(2.0, unit="s")
    stim_stop = start + pd.to_timedelta(12.0, unit="s")
    times = pd.date_range(
        start,
        periods=int(22 * sfreq),
        freq=pd.to_timedelta(1 / sfreq, unit="s"),
    )
    rng = np.random.default_rng(83)
    frame = pd.DataFrame(
        {
            "times": times,
            "LACC1": rng.normal(0, 3.0, len(times)),
            "LACC2": rng.normal(0, 3.0, len(times)),
            "POL L ACC3-Ref": rng.normal(0, 3.0, len(times)),
            "RSGC1": rng.normal(0, 3.0, len(times)),
        }
    )
    for offset in event_offsets:
        sample = int(np.argmin(np.abs(times - (start + pd.to_timedelta(offset, unit="s")))))
        frame.loc[sample, "POL L ACC3-Ref"] += 220.0
    # Large post-train peaks must never be treated as stimulation events.
    for offset in (13.0, 15.07, 17.14, 19.21):
        sample = int(np.argmin(np.abs(times - (start + pd.to_timedelta(offset, unit="s")))))
        frame.loc[sample, "POL L ACC3-Ref"] += 2_000.0
    frame.to_csv(Path(config["rawdata_path"]) / "fixture.csv", index=False)

    raw_meta = pd.read_csv(config["rawdata_meta_path"])
    raw_meta.loc[:, "start_time"] = start.isoformat()
    raw_meta.loc[:, "stop_time"] = (start + pd.to_timedelta(22.0, unit="s")).isoformat()
    raw_meta.to_csv(config["rawdata_meta_path"], index=False)

    stim_meta = pd.read_csv(config["stim_meta_path"])
    stim_meta.loc[:, "stim_start"] = stim_start.isoformat()
    stim_meta.loc[:, "stim_stop"] = stim_stop.isoformat()
    stim_meta.to_csv(config["stim_meta_path"], index=False)
    return config_path, stim_start, stim_stop


def test_auto_event_detection_crops_margins_and_adapts_to_decimated_artifacts(tmp_path):
    config_path, stim_start, stim_stop = _write_low_amplitude_event_fixture(
        tmp_path,
        [2.5, 4.57, 6.64, 8.71, 10.78],
    )
    dataloader = ep.DataLoader("P1", config_path=config_path)

    events = ep.detect_events_from_artifacts(
        dataloader,
        "A",
        "LACC_1_2",
        method="auto",
        stim_freq=0.5,
        min_peak_height=1000.0,
        raw_source="raw",
        raw_file="fixture.csv",
        save_events=False,
    )

    assert len(events) == 5
    assert events["detection_method"].eq("adjacent_adaptive").all()
    assert events["detection_adaptive"].all()
    assert events["detection_threshold_uv"].max() < 220.0
    assert pd.to_datetime(events["times"]).between(stim_start, stim_stop).all()


def test_auto_event_detection_rejects_irregular_low_amplitude_peaks(tmp_path):
    config_path, _, _ = _write_low_amplitude_event_fixture(
        tmp_path,
        [2.5, 4.1, 6.5, 8.1, 10.5],
    )
    dataloader = ep.DataLoader("P1", config_path=config_path)

    with pytest.raises(ep.EventDetectionError, match="cadence_consistency"):
        ep.detect_events_from_artifacts(
            dataloader,
            "A",
            "LACC_1_2",
            method="auto",
            stim_freq=0.5,
            min_peak_height=1000.0,
            raw_source="raw",
            raw_file="fixture.csv",
            save_events=False,
        )


def test_event_regularization_removes_cross_channel_duplicate_peaks():
    events = pd.DataFrame(
        {
            "times": [
                0.000,
                0.005,
                2.000,
                2.006,
                4.000,
                4.005,
                8.000,
                8.006,
            ]
        }
    )

    result = event_detection._regularize_event_sequence(events, stim_freq=0.5)

    assert result["times"].tolist() == [0.0, 2.0, 4.0, 8.0]
    assert result["event_count_before_regularization"].eq(8).all()
    assert result["event_count_removed_by_regularization"].eq(4).all()
    assert result["event_regularization_retained_fraction"].eq(0.5).all()


def test_event_regularization_is_datetime_resolution_independent():
    events = pd.DataFrame(
        {
            "times": pd.Series(
                np.array(
                    [
                        "2026-01-01T00:00:02",
                        "2026-01-01T00:00:04",
                        "2026-01-01T00:00:06",
                        "2026-01-01T00:00:08",
                    ],
                    dtype="datetime64[us]",
                ),
                dtype="datetime64[us]",
            )
        }
    )

    result = event_detection._regularize_event_sequence(
        events,
        stim_freq=0.5,
    )

    assert len(result) == 4
    assert result["event_regularization_retained_fraction"].eq(1.0).all()


def test_candidate_channel_matching_is_case_insensitive():
    frame = pd.DataFrame(
        {
            "POL R CMA1-Ref": [0.0],
            "POL R CMA2-Ref": [0.0],
            "POL R CMA5-Ref": [0.0],
            "OTHER1": [0.0],
        }
    )

    assert event_detection._candidate_channels(frame, "RCMa_3_4", "adjacent") == [
        "POL R CMA2-Ref",
        "POL R CMA5-Ref",
    ]


def test_binary_event_conversion():
    marker = pd.DataFrame({"event": [0, 0, 1, 1, 0, 1, 0]})
    _, events = ep.evoked_to_datetime(marker, "2026-01-01T00:00:00", "2026-01-01T00:00:06")
    assert len(events) == 2
    assert list(events.columns) == ["times"]


def test_erp_auc_integration_supports_numpy_old_and_new_api():
    values = np.array([0.0, 1.0, 0.0])
    times = np.array([0.0, 0.5, 1.0])
    assert _trapezoid(values, times) == pytest.approx(0.5)


def test_epochs_hdf_roundtrip_supports_large_audit_metadata(tmp_path):
    index = pd.MultiIndex.from_product([[0], [-0.1, 0.0, 0.1]], names=["epoch", "time"])
    frame = pd.DataFrame({"C1": [0.0, 1.0, 0.0]}, index=index)
    reports = [
        {
            "reference_channels": ("C1," * 400) + str(report_id),
            "anchor_time_s": 0.01 + report_id * 1e-6,
            "method": "test",
        }
        for report_id in range(100)
    ]
    epochs = ep.Epochs(
        frame,
        sfreq=10.0,
        tmin=-0.1,
        tmax=0.1,
        baseline=(-0.1, 0.0),
        stim_ch=["S1"],
        metadata={"zero_anchor_reports": reports, "patient_id": "P1"},
    )
    path = tmp_path / "large_metadata_epochs.h5"

    epochs.to_hdf(path)
    loaded = ep.Epochs.from_hdf(path)

    import tables

    with tables.open_file(path, mode="r") as handle:
        group = handle.get_node("/epochs")
        assert hasattr(group, "erpy_metadata_v2")
        assert not hasattr(group, "erpy_metadata_v1")
        assert "metadata" not in group._v_attrs._f_list()
        assert group._v_attrs.erpy_metadata_storage == "zlib+json-v2"

    assert len(loaded.metadata["zero_anchor_reports"]) == len(reports)
    assert loaded.metadata["zero_anchor_reports"][-1]["reference_channels"].endswith("99")
    assert loaded.metadata["patient_id"] == "P1"
    assert loaded.epochs_df["C1"].dtype == np.dtype("float32")
    assert loaded.metadata["storage_dtype"] == "float32"
    assert not list(tmp_path.glob("*.tmp.h5"))


def test_epochs_json_metadata_payload_is_deterministic():
    from ERPy.epochs import _decode_metadata_payload, _encode_metadata_payload

    first = {
        "zero_anchor_reports": [{"epoch": np.int64(2), "settled": True}],
        "baseline": (-0.1, 0.0),
        "scores": np.array([1.0, np.nan, np.inf], dtype=np.float32),
    }
    second = {
        "scores": np.array([1.0, np.nan, np.inf], dtype=np.float32),
        "baseline": (-0.1, 0.0),
        "zero_anchor_reports": [{"settled": True, "epoch": np.int64(2)}],
    }

    first_payload = _encode_metadata_payload(first)
    second_payload = _encode_metadata_payload(second)

    assert first_payload == second_payload
    restored = _decode_metadata_payload(first_payload)
    assert restored["baseline"] == (-0.1, 0.0)
    np.testing.assert_array_equal(
        restored["scores"],
        first["scores"],
    )


def _write_legacy_epochs_file(path: Path, frame: pd.DataFrame, metadata: dict) -> None:
    """Create the pickle-backed representation used by pre-1.0 ERPy builds."""

    import tables

    with pd.HDFStore(path, mode="w") as store:
        store.put("epochs", frame, format="fixed")
    payload = zlib.compress(pickle.dumps(metadata, protocol=pickle.HIGHEST_PROTOCOL))
    with tables.open_file(path, mode="a") as handle:
        group = handle.get_node("/epochs")
        handle.create_array(
            group,
            "erpy_metadata_v1",
            obj=np.frombuffer(payload, dtype=np.uint8),
        )
        group._v_attrs.erpy_metadata_storage = "zlib+pickle-v1"


def test_epochs_hdf_rejects_legacy_pickle_metadata_by_default(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import ERPy.epochs as epoch_storage

    index = pd.MultiIndex.from_product([[0], [-0.1, 0.0, 0.1]], names=["epoch", "time"])
    frame = pd.DataFrame({"C1": [0.0, 1.0, 0.0]}, index=index)
    path = tmp_path / "legacy_epochs.h5"
    metadata = {
        "sfreq": 10.0,
        "tmin": -0.1,
        "tmax": 0.1,
        "baseline": (-0.1, 0.0),
        "stim_ch": ["S1"],
    }
    _write_legacy_epochs_file(path, frame, metadata)

    def fail_if_unpickled(_payload):
        raise AssertionError("pickle.loads must not run without explicit unsafe opt-in")

    monkeypatch.setattr(epoch_storage, "pickle", SimpleNamespace(loads=fail_if_unpickled))

    with pytest.raises(ep.UnsafeLegacyEpochMetadataError, match="blocked by default"):
        ep.Epochs.from_hdf(path)


def test_epochs_hdf_loads_legacy_pickle_only_with_explicit_unsafe_opt_in(tmp_path):
    index = pd.MultiIndex.from_product([[0], [-0.1, 0.0, 0.1]], names=["epoch", "time"])
    frame = pd.DataFrame({"C1": [0.0, 1.0, 0.0]}, index=index)
    path = tmp_path / "trusted_legacy_epochs.h5"
    metadata = {
        "sfreq": 10.0,
        "tmin": -0.1,
        "tmax": 0.1,
        "baseline": (-0.1, 0.0),
        "stim_ch": ["S1"],
    }
    _write_legacy_epochs_file(path, frame, metadata)

    with pytest.warns(RuntimeWarning, match="Unsafe legacy ERPy pickle metadata"):
        loaded = ep.Epochs.from_hdf(path, allow_unsafe_legacy_pickle=True)

    assert loaded.sfreq == 10.0
    assert loaded.baseline == (-0.1, 0.0)


def test_processed_hdf_roundtrip_uses_safe_json_metadata(tmp_path):
    from types import SimpleNamespace

    import tables

    pipeline = ep.Pipeline(SimpleNamespace())
    frame = pd.DataFrame(
        {"C1": [0.5, 1.5, 2.5]},
        index=pd.Index([0.0, 0.1, 0.2], name="time"),
    )
    frame.attrs.update(sfreq=np.float64(10.0), pipeline_id="pipe-1")
    path = tmp_path / "processed.h5"

    pipeline.save_processed_data(
        frame,
        path,
        session_id="A",
        pipeline_steps=[("artifact_blank", {"width_s": 0.003})],
    )
    loaded = pipeline.load_processed_data(path)

    pd.testing.assert_frame_equal(loaded, frame, check_dtype=True)
    assert loaded.attrs["sfreq"] == 10.0
    assert loaded.attrs["pipeline_id"] == "pipe-1"
    assert loaded.attrs["session_id"] == "A"
    assert loaded.attrs["pipeline_steps"] == [
        ("artifact_blank", {"width_s": 0.003})
    ]
    with tables.open_file(path, mode="r") as handle:
        group = handle.get_node("/data")
        assert hasattr(group, "erpy_metadata_v2")
        assert "metadata" not in group._v_attrs._f_list()
        assert group._v_attrs.erpy_metadata_storage == "zlib+json-v2"
    assert not list(tmp_path.glob("*.tmp.h5"))


def _write_legacy_processed_file(
    path: Path,
    frame: pd.DataFrame,
    metadata: dict,
) -> None:
    """Create the PyTables-attribute representation used before ERPy 1.0."""

    with pd.HDFStore(path, mode="w") as store:
        store.put("data", frame, format="table")
        store.get_storer("data").attrs.metadata = metadata


def test_processed_hdf_rejects_legacy_pickle_metadata_by_default(tmp_path):
    from types import SimpleNamespace

    pipeline = ep.Pipeline(SimpleNamespace())
    frame = pd.DataFrame({"C1": [0.0, 1.0]}, index=[0.0, 0.1])
    path = tmp_path / "legacy_processed.h5"
    _write_legacy_processed_file(
        path,
        frame,
        {"sfreq": 10.0, "pipeline_id": "legacy"},
    )

    with pytest.raises(
        ep.UnsafeLegacyProcessedMetadataError,
        match="blocked by default",
    ):
        pipeline.load_processed_data(path)


def test_processed_hdf_loads_legacy_metadata_only_with_unsafe_opt_in(tmp_path):
    from types import SimpleNamespace

    pipeline = ep.Pipeline(SimpleNamespace())
    frame = pd.DataFrame({"C1": [0.0, 1.0]}, index=[0.0, 0.1])
    path = tmp_path / "trusted_legacy_processed.h5"
    _write_legacy_processed_file(
        path,
        frame,
        {"sfreq": 10.0, "pipeline_id": "legacy"},
    )

    with pytest.warns(
        RuntimeWarning,
        match="Unsafe legacy ERPy processed-data pickle metadata",
    ):
        loaded = pipeline.load_processed_data(
            path,
            allow_unsafe_legacy_pickle=True,
        )

    assert loaded.attrs["sfreq"] == 10.0
    assert loaded.attrs["pipeline_id"] == "legacy"


def test_slurm_wrap_quotes_python_and_script_paths(monkeypatch):
    from types import SimpleNamespace

    import ERPy.utils.jobrunner as jobrunner

    observed = {}

    monkeypatch.setattr(jobrunner.shutil, "which", lambda command: f"/usr/bin/{command}")

    def capture_run(command, **kwargs):
        observed["command"] = command
        return SimpleNamespace(stdout="123\n", stderr="")

    monkeypatch.setattr(jobrunner.subprocess, "run", capture_run)
    runner = jobrunner.JobRunner(backend="slurm", job_name="quoted-paths")
    runner.run(
        "/tmp/study's scripts/run analysis.py",
        python="/tmp/Python Env/bin/python",
    )

    command = observed["command"]
    wrap = command[command.index("--wrap") + 1]
    assert wrap == "'/tmp/Python Env/bin/python' '/tmp/study'\"'\"'s scripts/run analysis.py'"


def test_post_artifact_anchor_detector_flags_onset_spike():
    times = np.round(np.arange(-20, 51) / 1000, 3)
    df = pd.DataFrame(
        {
            "STIM1": np.zeros(len(times)),
            "ADJ1": np.zeros(len(times)),
            "REC1": np.zeros(len(times)),
        },
        index=times,
    )
    df.loc[0.000, ["STIM1", "ADJ1"]] = [70.0, -60.0]
    df.loc[0.001, ["STIM1", "ADJ1"]] = [25.0, -20.0]
    df.loc[0.002, ["STIM1", "ADJ1"]] = [10.0, -8.0]
    df.loc[0.020:0.045, "REC1"] = 4.0 * np.hanning(len(df.loc[0.020:0.045]))

    risk = ep.detect_zero_time_artifact(df, stim_ch=["STIM1"], baseline_window=(-0.02, -0.005))
    assert risk["zero_time_is_artifact"]
    assert risk["artifact_score_z"] >= risk["artifact_z_threshold"]

    anchor = ep.detect_post_artifact_anchor(df, stim_ch=["STIM1"], baseline_window=(-0.02, -0.005))
    assert anchor["onset_is_artifact"]
    assert anchor["artifact_peak_time_s"] == pytest.approx(0.0)
    assert anchor["anchor_time_s"] >= 0.003
    assert anchor["anchor_time_s"] > anchor["artifact_peak_time_s"]


def test_artifact_events_pipeline_epoch_and_detection(tmp_path):
    config_path, _ = _write_fixture(tmp_path)
    dl = ep.DataLoader("P1", config_path=config_path)
    events = ep.detect_events_from_artifacts(
        dl,
        "A",
        "LACC_1_2",
        method="adjacent",
        stim_freq=0.5,
        min_peak_height=10,
        raw_source="raw",
        save_events=True,
    )
    assert len(events) == 4

    pipeline = ep.Pipeline(dl)
    steps = [
        ("reject_bad_channels", {"bad_channels": ["BAD1"], "method": "drop"}),
        ("artifact_blank", {"width_s": 0.005}),
        ("bandpass_filter", {"lowcut": 0.5, "highcut": 40.0, "order": 2}),
    ]
    epochs, processed = pipeline.process_and_epoch(
        "A",
        "LACC_1_2",
        steps,
        event_times=pd.to_datetime(events["times"]),
        tmin=-0.25,
        tmax=0.5,
        baseline=(-0.25, -0.05),
        save_processed=True,
        save_epochs=True,
    )
    assert "BAD1" not in processed.columns
    assert epochs.n_trials() == 4
    assert "RSGC1" in epochs.channels
    zero_report = epochs.zero_time_report()
    assert zero_report["is_zeroed"]
    assert zero_report["max_abs_uv"] <= 1e-9
    assert zero_report["zero_time_strategy"] == ep.POST_ARTIFACT_ZERO
    assert zero_report["median_anchor_s"] > 0
    assert zero_report["common_anchor_s"] > 0
    anchors = epochs.post_artifact_anchor_report()
    assert len(anchors) == epochs.n_trials()
    assert (anchors["anchor_time_s"] > 0).all()
    # Every trial is zero-anchored at the same common time over a short baseline
    # window; that window mean is ~0 per trial, so the trial-averaged waveform is
    # zero at the anchor (the point a single-sample per-trial anchor could not hit).
    times = epochs.times
    common_idx = int(np.argmin(np.abs(times - zero_report["nearest_sample_s"])))
    n_win = max(int(round(zero_report["post_baseline_window_s"] * epochs.sfreq)), 1)
    window_times = times[common_idx : common_idx + n_win]
    for ep_id in epochs.epochs_df.index.get_level_values("epoch").unique():
        trial = epochs.epochs_df.xs(ep_id, level="epoch")
        window_mean = trial.reindex(window_times).mean(axis=0).to_numpy(dtype=float)
        assert np.allclose(window_mean, 0.0, atol=1e-9)
    mean_wave = epochs.get_mean_waveform().reindex(window_times).to_numpy(dtype=float)
    assert np.allclose(np.nanmean(mean_wave, axis=0), 0.0, atol=1e-9)

    det = epochs.detect_erp_all(methods=["keller_zscore", "peak_amplitude", "rms_response"], min_consensus=1)
    assert {"channel", "method", "significant", "peak_amplitude_uv", "consensus_ch"}.issubset(det.columns)
    assert det[det["channel"] == "RSGC1"]["significant"].any()

    cached = pipeline.load_epochs("A", "LACC_1_2", pipeline_steps=steps)
    assert cached.n_trials() == epochs.n_trials()

    corrupted = ep.Epochs(
        epochs.epochs_df.copy(),
        epochs.sfreq,
        epochs.tmin,
        epochs.tmax,
        epochs.baseline,
        epochs.stim_ch,
        epochs.metadata,
    )
    corrupted.epochs_df.loc[(0, slice(None)), "RSGC1"] *= 1000
    report = corrupted.flag_artifacts(zscore_threshold=2.5, saturation_fraction=0.8)
    assert report.table["bad_response"].any()
    cleaned = corrupted.reject_artifacts(report=report, mode="drop_epoch", max_bad_channel_fraction=0.01)
    assert cleaned.n_trials() < corrupted.n_trials()


def test_patient_analyze_returns_auditable_result(tmp_path):
    config_path, _ = _write_fixture(tmp_path)
    patient = ep.Patient("P1", config_path=config_path)
    result = patient.analyze(
        "A",
        "LACC_1_2",
        pipeline=[
            ("reject_bad_channels", {"bad_channels": ["BAD1"], "method": "drop"}),
            ("artifact_blank", {"width_s": 0.005}),
            ("bandpass_filter", {"lowcut": 0.5, "highcut": 40.0, "order": 2}),
        ],
        event_method="adjacent",
        stim_freq=0.5,
        min_peak_height=10,
        tmin=-0.25,
        tmax=0.5,
        baseline=(-0.25, -0.05),
        methods=["keller_zscore", "peak_amplitude", "rms_response"],
        min_consensus=1,
    )

    assert isinstance(result, ep.AnalysisResult)
    assert result.epochs.n_trials() == 4
    assert {"channel", "artifact_qc_pass"}.issubset(result.qc_detections.columns)
    assert not result.artifact_summary.empty
    assert not result.significant.empty
    assert result.metadata["audit_schema_version"] == 1
    assert result.metadata["erpy_version"] == "1.0.0"
    assert result.metadata["erpy_compatibility_id"] == ep.CHECKPOINT_COMPATIBILITY_ID
    assert result.metadata["epoch"]["tmin"] == -0.25
    assert result.metadata["epoch"]["baseline"] == (-0.25, -0.05)
    assert result.metadata["persistent_channel_qc"]["configured"] is True
    assert result.metadata["artifact_qc"]["effective_options"]["zscore_threshold"] == 6.0
    assert result.metadata["detection"]["effective_core"]["methods"] == [
        "keller_zscore",
        "peak_amplitude",
        "rms_response",
    ]
    assert {
        row["method"]
        for row in result.metadata["detection"]["method_parameter_records"]
    } == {"keller_zscore", "peak_amplitude", "rms_response"}

    saved = result.save(tmp_path / "analysis_tables")
    assert saved["detections"].exists()
    assert saved["detections_qc_pass"].exists()
    assert saved["artifact_channel_summary"].exists()
    saved_metadata = json.loads(saved["metadata"].read_text(encoding="utf-8"))
    assert saved_metadata["erpy_compatibility_id"] == ep.CHECKPOINT_COMPATIBILITY_ID
    assert saved_metadata["epoch"]["baseline"] == [-0.25, -0.05]
    assert saved_metadata["artifact_qc"]["hard_artifact_reasons"] == sorted(
        ep.DEFAULT_HARD_ARTIFACT_REASONS
    )


def test_analysis_result_primary_significant_returns_one_native_row_per_channel():
    detections = pd.DataFrame(
        [
            {
                "channel": channel,
                "method": method,
                "primary_qc_pass": channel == "C1",
                "p_joint": 0.01 if method == "crp_energy" else np.nan,
            }
            for channel in ("C1", "C2")
            for method in ("kundu_rolston", "crp_energy", "rms_response")
        ]
    )
    result = ep.AnalysisResult(
        epochs=None,
        detections=detections,
        artifact_report=None,
        artifact_summary=pd.DataFrame(),
        qc_detections=detections,
    )

    primary = result.primary_significant

    assert primary[["channel", "method"]].to_dict("records") == [
        {"channel": "C1", "method": "crp_energy"}
    ]
    assert primary.loc[0, "p_joint"] == pytest.approx(0.01)


def test_batched_erp_detection_matches_crp_and_high_gamma_reference():
    from scipy import signal

    from ERPy.crp import CRPConfig, run_crp
    from ERPy.erp_detection import _crp_significance_from_array, _high_gamma_z_all

    sfreq = 400.0
    times = np.arange(-0.2, 0.3000001, 1.0 / sfreq)
    channels = ["C1", "C2", "C3"]
    rng = np.random.default_rng(2026)
    arr = rng.normal(0.0, 0.3, (8, len(times), len(channels)))
    response = (times >= 0.01) & (times <= 0.15)
    arr[:, response, 0] += 1.5 * np.sin(2 * np.pi * 80.0 * times[response])[None, :]
    index = pd.MultiIndex.from_product(
        [range(arr.shape[0]), times],
        names=["epoch", "time"],
    )
    epochs = ep.Epochs(
        pd.DataFrame(arr.reshape(-1, len(channels)), index=index, columns=channels),
        sfreq=sfreq,
        tmin=float(times[0]),
        tmax=float(times[-1]),
        baseline=(-0.2, -0.03),
    )

    fast_score, fast_p = _crp_significance_from_array(
        arr[:, :, 0],
        times,
        response_window=(0.01, 0.15),
    )
    reference_crp = run_crp(
        epochs,
        "C1",
        CRPConfig(response_window=(0.01, 0.15), baseline_window=(-0.2, -0.03)),
    )
    assert fast_score == pytest.approx(reference_crp.score, rel=1e-10, abs=1e-10)
    assert fast_p == pytest.approx(reference_crp.p_value, rel=1e-10, abs=1e-10)

    scores = _high_gamma_z_all(
        arr,
        times,
        sfreq=sfreq,
        baseline_window=(-0.2, -0.03),
        response_window=(0.01, 0.15),
        channel_chunk_size=2,
    )
    sos = signal.butter(4, [70.0, 150.0], btype="bandpass", fs=sfreq, output="sos")
    filtered = signal.sosfiltfilt(sos, arr[:, :, 0], axis=1)
    envelope = np.abs(signal.hilbert(filtered, axis=1))
    baseline = envelope[:, (times >= -0.2) & (times <= -0.03)].reshape(-1)
    response_values = envelope[:, response].reshape(-1)
    expected = (
        np.nanmean(response_values) - np.nanmean(baseline)
    ) / (np.nanstd(baseline, ddof=1) + 1e-12)
    assert scores[0] == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_bad_channel_auto_flags_flat_channel(tmp_path):
    config_path, _ = _write_fixture(tmp_path)
    dl = ep.DataLoader("P1", config_path=config_path)
    df = dl.load_stim_data("A", "LACC_1_2", source="raw")
    report = ep.detect_bad_channels(df)
    assert "BAD1" in report.bad_channels
    cleaned = ep.reject_bad_channels(df, bad_channels="auto", method="drop")
    assert "BAD1" not in cleaned.columns


def test_sample_window_bounds_are_clamped_to_recording(tmp_path):
    config_path, _ = _write_fixture(tmp_path)
    dl = ep.DataLoader("P1", config_path=config_path)

    full = dl._sample_window_from_stim_values(0.0, 10.0, 100.0, 1000, 0.0)
    after_recording = dl._sample_window_from_stim_values(20.0, 21.0, 100.0, 1000, 0.0)

    assert full[:2] == (0, 1000)
    assert after_recording[:2] == (1000, 1000)


def test_response_artifact_detector_flags_plateau_saturation():
    sfreq = 1000.0
    times = np.arange(-0.1, 0.4, 1 / sfreq)
    rng = np.random.default_rng(18)
    rows = []
    index = []
    for trial in range(6):
        clean = rng.normal(0, 0.2, len(times))
        clipped = rng.normal(0, 0.2, len(times))
        if trial == 2:
            mask = (times >= 0.04) & (times <= 0.09)
            clipped[mask] = 4095.0
        rows.append(np.column_stack([clean, clipped]))
        index.extend((trial, float(t)) for t in times)

    epochs = ep.Epochs(
        pd.DataFrame(
            np.vstack(rows),
            index=pd.MultiIndex.from_tuples(index, names=["epoch", "time"]),
            columns=["clean_contact", "clipped_contact"],
        ),
        sfreq=sfreq,
        tmin=float(times[0]),
        tmax=float(times[-1]),
        baseline=(-0.1, -0.02),
    )

    report = epochs.flag_artifacts(response_window=(0.01, 0.2), zscore_threshold=99.0, saturation_fraction=0.95)
    flagged = report.table[
        (report.table["epoch"].eq(2))
        & (report.table["channel"].eq("clipped_contact"))
        & (report.table["bad_response"])
    ]
    assert not flagged.empty
    assert "plateau_clipping" in flagged.iloc[0]["reason"] or "rail_clipping" in flagged.iloc[0]["reason"]

    summary = report.by_channel()
    clipped_summary = summary[summary["channel"].eq("clipped_contact")].iloc[0]
    clean_summary = summary[summary["channel"].eq("clean_contact")].iloc[0]
    assert clipped_summary["bad_response_fraction"] > 0
    assert clean_summary["bad_response_fraction"] == 0


def test_response_artifact_detector_does_not_treat_local_extrema_as_rails():
    sfreq = 2000.0
    times = np.arange(-0.1, 0.4, 1 / sfreq)
    response = 40.0 * np.cos(2 * np.pi * 1.25 * (times - 0.18))
    rows = []
    index = []
    for trial in range(8):
        rows.append(response[:, None])
        index.extend((trial, float(time)) for time in times)
    epochs = ep.Epochs(
        pd.DataFrame(
            np.vstack(rows),
            index=pd.MultiIndex.from_tuples(index, names=["epoch", "time"]),
            columns=["smooth_response"],
        ),
        sfreq=sfreq,
        tmin=float(times[0]),
        tmax=float(times[-1]),
        baseline=(-0.1, -0.02),
    )

    report = epochs.flag_artifacts(
        response_window=(0.01, 0.35),
        zscore_threshold=99.0,
        ringing_zscore_threshold=99.0,
        rail_fraction=0.02,
        rail_epsilon_fraction=0.01,
    )

    assert (report.table["extreme_dwell_fraction"] > 0.02).all()
    assert not report.table["reason"].str.contains("rail_clipping", na=False).any()
    assert not report.table["bad_response"].any()


def test_response_artifact_detector_only_flags_high_amplitude_outliers():
    sfreq = 1000.0
    times = np.arange(-0.1, 0.3, 1 / sfreq)
    shape = np.exp(-0.5 * ((times - 0.08) / 0.02) ** 2)
    rows = []
    index = []
    for trial in range(10):
        amplitude = 0.1 if trial == 0 else 10.0
        rows.append((amplitude * shape)[:, None])
        index.extend((trial, float(time)) for time in times)
    epochs = ep.Epochs(
        pd.DataFrame(
            np.vstack(rows),
            index=pd.MultiIndex.from_tuples(index, names=["epoch", "time"]),
            columns=["response"],
        ),
        sfreq=sfreq,
        tmin=float(times[0]),
        tmax=float(times[-1]),
        baseline=(-0.1, -0.02),
    )

    report = epochs.flag_artifacts(
        response_window=(0.01, 0.2),
        zscore_threshold=2.0,
        ringing_zscore_threshold=99.0,
    )
    low_trial = report.table[report.table["epoch"].eq(0)].iloc[0]

    assert low_trial["peak_abs_z"] < 0
    assert "amplifier_or_motion_artifact" not in low_trial["reason"]


def test_extreme_trial_relative_amplitude_is_a_hard_artifact():
    sfreq = 1000.0
    times = np.arange(-0.1, 0.3, 1 / sfreq)
    shape = np.exp(-0.5 * ((times - 0.08) / 0.02) ** 2)
    rows = []
    index = []
    for trial in range(30):
        amplitude = 1000.0 if trial == 29 else 10.0 + 0.1 * trial
        rows.append((amplitude * shape)[:, None])
        index.extend((trial, float(time)) for time in times)
    epochs = ep.Epochs(
        pd.DataFrame(
            np.vstack(rows),
            index=pd.MultiIndex.from_tuples(index, names=["epoch", "time"]),
            columns=["response"],
        ),
        sfreq=sfreq,
        tmin=float(times[0]),
        tmax=float(times[-1]),
        baseline=(-0.1, -0.02),
    )

    report = epochs.flag_artifacts(
        response_window=(0.01, 0.2),
        zscore_threshold=6.0,
        extreme_zscore_threshold=12.0,
        ringing_zscore_threshold=99.0,
    )
    outlier = report.table[report.table["epoch"].eq(29)].iloc[0]

    assert outlier["peak_abs_z"] > 12
    assert "extreme_amplitude_outlier" in outlier["reason"]
    assert bool(outlier["hard_artifact_response"])


def test_response_window_boundary_peaks_are_excluded_from_qc():
    sfreq = 1000.0
    times = np.arange(-0.1, 0.251, 1 / sfreq)
    response_mask = (times >= 0.01) & (times <= 0.2)
    drift = np.zeros_like(times)
    drift[response_mask] = np.linspace(0.0, 100.0, response_mask.sum())
    interior = 50.0 * np.exp(-0.5 * ((times - 0.08) / 0.012) ** 2)
    rows = []
    index = []
    for trial in range(5):
        rows.append(np.column_stack([drift, interior]))
        index.extend((trial, float(time)) for time in times)
    epochs = ep.Epochs(
        pd.DataFrame(
            np.vstack(rows),
            index=pd.MultiIndex.from_tuples(index, names=["epoch", "time"]),
            columns=["drift", "interior"],
        ),
        sfreq=sfreq,
        tmin=float(times[0]),
        tmax=float(times[-1]),
        baseline=(-0.1, -0.02),
    )

    detections = epochs.detect_erp_all(
        methods=["peak_amplitude"],
        min_consensus=1,
        response_window=(0.01, 0.2),
    )
    annotated = _annotate_detection_artifact_qc(
        detections,
        pd.DataFrame(),
        max_bad_response_fraction=0.25,
        max_hard_artifact_fraction=0.10,
        min_clean_responses=3,
        exclude_boundary_peaks=True,
    ).set_index("channel")

    assert bool(annotated.loc["drift", "peak_at_response_boundary"])
    assert annotated.loc["drift", "peak_boundary_side"] == "end"
    assert bool(annotated.loc["drift", "artifact_boundary_peak_fail"])
    assert not bool(annotated.loc["drift", "artifact_qc_pass"])
    assert not bool(annotated.loc["interior", "peak_at_response_boundary"])
    assert bool(annotated.loc["interior", "artifact_qc_pass"])


def test_artifact_qc_uses_prevalence_not_any_hard_event():
    detections = pd.DataFrame({"channel": ["C1", "C2"]})
    artifact_summary = pd.DataFrame(
        {
            "channel": ["C1", "C2"],
            "n_epochs": [20, 20],
            "n_bad_response": [1, 6],
            "n_clean_response": [19, 14],
            "bad_response_fraction": [0.05, 0.30],
            "n_hard_artifact_response": [1, 0],
            "hard_artifact_fraction": [0.05, 0.0],
            "artifact_reasons": ["plateau_clipping", "sharp_transient"],
        }
    )

    annotated = _annotate_detection_artifact_qc(
        detections,
        artifact_summary,
        max_bad_response_fraction=0.25,
        max_hard_artifact_fraction=0.10,
        min_clean_responses=10,
    ).set_index("channel")

    assert bool(annotated.loc["C1", "artifact_qc_pass"])
    assert not bool(annotated.loc["C2", "artifact_qc_pass"])


def test_artifact_qc_excludes_values_at_configured_fraction_cutoffs():
    detections = pd.DataFrame({"channel": ["BAD", "HARD"]})
    artifact_summary = pd.DataFrame(
        {
            "channel": ["BAD", "HARD"],
            "n_epochs": [20, 20],
            "n_bad_response": [5, 2],
            "n_clean_response": [15, 18],
            "bad_response_fraction": [0.25, 0.10],
            "n_hard_artifact_response": [0, 2],
            "hard_artifact_fraction": [0.0, 0.10],
            "artifact_reasons": ["sharp_transient", "plateau_clipping"],
        }
    )

    annotated = _annotate_detection_artifact_qc(
        detections,
        artifact_summary,
        max_bad_response_fraction=0.25,
        max_hard_artifact_fraction=0.10,
        min_clean_responses=10,
    ).set_index("channel")

    assert not bool(annotated.loc["BAD", "artifact_qc_pass"])
    assert not bool(annotated.loc["HARD", "artifact_qc_pass"])


def test_robust_common_reference_does_not_spread_extreme_contact():
    rng = np.random.default_rng(41)
    values = rng.normal(0, 0.2, (200, 9))
    frame = pd.DataFrame(values, columns=[f"C{i}" for i in range(8)] + ["BAD"])
    frame.loc[60:120, "BAD"] += 10_000.0

    from ERPy.utils.preproc import common_reference

    robust_frame = common_reference(frame, method="robust_mean", zscore_threshold=8.0)
    mean_frame = common_reference(frame, method="mean")
    good = [f"C{i}" for i in range(8)]

    assert robust_frame.loc[60:120, good].abs().to_numpy().max() < 1.0
    assert mean_frame.loc[60:120, good].abs().to_numpy().min() > 100.0


def test_csv_value_scale_is_applied_once_and_recorded_in_cache(tmp_path):
    config_path, _ = _write_fixture(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw_meta_path = Path(config["rawdata_meta_path"])
    raw_meta = pd.read_csv(raw_meta_path)
    raw_meta["value_scale_to_uv"] = 1e-6
    raw_meta.to_csv(raw_meta_path, index=False)
    source = pd.read_csv(Path(config["rawdata_path"]) / "fixture.csv")

    loader = ep.DataLoader("P1", config_path=config_path)
    first = loader.load_stim_data("A", "LACC_1_2", source="raw", save_csv=True)
    second = loader.load_stim_data("A", "LACC_1_2", source="auto")

    assert first.iloc[0]["LACC1"] == pytest.approx(source.iloc[0]["LACC1"] * 1e-6)
    assert first.attrs["signal_units"] == "uV"
    assert first.attrs["value_scale_to_uv_applied"] == pytest.approx(1e-6)
    np.testing.assert_allclose(first.to_numpy(dtype=float), second.to_numpy(dtype=float))
    assert loader._raw_cache_metadata_path(loader.get_raw_csv_path("A", "LACC_1_2")).exists()


def test_binary_raw_cache_roundtrip_avoids_reloading_source(tmp_path):
    config_path, _ = _write_fixture(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["raw_cache_format"] = "npz"
    config["raw_cache_dtype"] = "float32"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    loader = ep.DataLoader("P1", config_path=config_path)
    first = loader.load_stim_data("A", "LACC_1_2", source="raw", save_csv=True)
    cache_path = loader.get_raw_cache_path("A", "LACC_1_2")
    assert cache_path.suffix == ".npz"
    assert cache_path.is_file()

    loader.clear_memory_cache()
    source_path = Path(config["rawdata_path"]) / "fixture.csv"
    source_path.unlink()
    second = loader.load_stim_data("A", "LACC_1_2", source="auto")

    assert set(second.dtypes.astype(str)) == {"float32"}
    np.testing.assert_allclose(
        first.to_numpy(dtype=np.float32),
        second.to_numpy(dtype=np.float32),
        rtol=0,
        atol=0,
    )


def test_raw_prefetch_cancellation_removes_partial_stage(tmp_path):
    config_path, _ = _write_fixture(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    stage_root = tmp_path / "stage"
    config["raw_staging_path"] = str(stage_root)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    loader = ep.DataLoader("P1", config_path=config_path)
    stop_event = threading.Event()
    stop_event.set()

    with pytest.raises(InterruptedError):
        loader.prefetch_stim_data(
            "A",
            "LACC_1_2",
            raw_file="fixture.csv",
            stop_event=stop_event,
        )

    assert not list(stage_root.rglob("*.partial"))


def test_patient_from_bids_brainvision_fixture(tmp_path):
    bids = tmp_path / "bids"
    ieeg = bids / "sub-01" / "ses-ieeg01" / "ieeg"
    ieeg.mkdir(parents=True)
    stem = "sub-01_ses-ieeg01_task-ccep_run-01"
    sfreq = 200.0
    n_samples = int(8 * sfreq)
    channels = ["LACC1", "LACC2", "LACC3", "RSGC1"]
    rng = np.random.default_rng(9)
    data = rng.normal(0, 0.2, (n_samples, len(channels))).astype("<f4")
    event_seconds = np.array([2.0, 4.0, 6.0])
    for onset in event_seconds:
        i = int(onset * sfreq)
        data[i, 2] += 40
        r0 = i + int(0.04 * sfreq)
        r1 = i + int(0.15 * sfreq)
        data[r0:r1 + 1, 3] += (-4 * np.hanning(r1 - r0 + 1)).astype("<f4")
    (ieeg / f"{stem}_ieeg.eeg").write_bytes(data.astype("<f4").tobytes())
    vhdr = "\n".join(
        [
            "Brain Vision Data Exchange Header File Version 1.0",
            "[Common Infos]",
            f"DataFile={stem}_ieeg.eeg",
            f"MarkerFile={stem}_ieeg.vmrk",
            "DataFormat=BINARY",
            "DataOrientation=MULTIPLEXED",
            f"NumberOfChannels={len(channels)}",
            f"SamplingInterval={1_000_000 / sfreq}",
            "[Binary Infos]",
            "BinaryFormat=IEEE_FLOAT_32",
            "[Channel Infos]",
            *[f"Ch{i+1}={ch},,1,uV" for i, ch in enumerate(channels)],
        ]
    )
    (ieeg / f"{stem}_ieeg.vhdr").write_text(vhdr, encoding="utf-8")
    (ieeg / f"{stem}_events.tsv").write_text(
        "onset\tduration\ttrial_type\telectrical_stimulation_site\tstatus\n"
        + "\n".join(f"{s}\t0.0002\telectrical_stimulation\tLACC1-LACC2\tgood" for s in event_seconds),
        encoding="utf-8",
    )
    (ieeg / "sub-01_ses-ieeg01_electrodes.tsv").write_text(
        "name\tx\ty\tz\tgroup\n"
        + "\n".join(f"{ch}\t0\t0\t0\tDemo" for ch in channels),
        encoding="utf-8",
    )

    patient = ep.Patient.from_bids(str(bids), subject="01", session="ieeg01", task="ccep", output_root=str(tmp_path / "erpy_bids"))
    assert patient.stim_pairs("ieeg01") == ["LACC_1_2"]
    epochs = patient.epoch("ieeg01", "LACC_1_2", pipeline=[("artifact_blank", {"width_s": 0.004})], tmin=-0.2, tmax=0.4, baseline=(-0.2, -0.05), cache=False)
    assert epochs.n_trials() == 3
    det = epochs.detect_erp_all(methods=["keller_zscore", "peak_amplitude"], min_consensus=1)
    assert "RSGC1" in det["channel"].values


def test_patient_from_nwb_fixture(tmp_path):
    pytest.importorskip("pynwb")
    from pynwb import NWBHDF5IO, NWBFile
    from pynwb.ecephys import ElectricalSeries
    from pynwb.file import Subject

    sfreq = 200.0
    n_samples = int(8 * sfreq)
    channels = ["LACC1", "LACC2", "RSGC1"]
    rng = np.random.default_rng(12)
    data = rng.normal(0, 0.15, (n_samples, len(channels))).astype(float)
    for onset in (2.0, 4.0, 6.0):
        i = int(onset * sfreq)
        data[i, 1] += 25
        r0 = i + int(0.04 * sfreq)
        r1 = i + int(0.16 * sfreq)
        data[r0:r1 + 1, 2] += -4 * np.hanning(r1 - r0 + 1)

    nwb = NWBFile(
        session_description="ERPy NWB fixture",
        identifier="ERPY-NWB-FIXTURE",
        session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        session_id="A",
    )
    nwb.subject = Subject(subject_id="P2")
    device = nwb.create_device("clinical_ieeg")
    group = nwb.create_electrode_group("LACC", description="depth lead", location="ACC", device=device)
    nwb.add_electrode_column("label", "clinical contact label")
    for idx, ch in enumerate(channels):
        nwb.add_electrode(
            id=idx,
            x=float(-6 + idx),
            y=float(24 - idx),
            z=float(30 + idx),
            imp=np.nan,
            location="ACC" if ch.startswith("LACC") else "subgenual cingulate",
            filtering="none",
            group=group,
            label=ch,
        )
    region = nwb.create_electrode_table_region(region=[0, 1, 2], description="all contacts")
    nwb.add_acquisition(
        ElectricalSeries(
            name="ElectricalSeries",
            data=data,
            electrodes=region,
            rate=sfreq,
            starting_time=0.0,
            conversion=1e-6,
        )
    )
    nwb.add_trial_column("electrical_stimulation_site", "bipolar stimulation site")
    for onset in (2.0, 4.0, 6.0):
        nwb.add_trial(start_time=onset, stop_time=onset + 0.001, electrical_stimulation_site="LACC1-LACC2")

    nwb_path = tmp_path / "sub-P2_ses-A_ieeg.nwb"
    with NWBHDF5IO(str(nwb_path), "w") as io:
        io.write(nwb)

    patient = ep.Patient.from_nwb(str(nwb_path), output_root=str(tmp_path / "erpy_nwb"), stim_pair="LACC_1_2")
    assert patient.stim_pairs("A") == ["LACC_1_2"]
    event_path = patient.dataloader.get_event_path("A", "LACC_1_2")
    assert event_path.exists()

    loaded = patient.dataloader.load_stim_data("A", "LACC_1_2", source="raw", save_csv=False)
    assert loaded.attrs["sfreq"] == pytest.approx(sfreq)
    assert loaded.columns.tolist() == channels
    assert loaded.index.min() <= 1.0
    assert loaded.index.max() >= 6.0

    epochs = patient.epoch(
        "A",
        "LACC_1_2",
        pipeline=[("artifact_blank", {"width_s": 0.004})],
        tmin=-0.2,
        tmax=0.4,
        baseline=(-0.2, -0.05),
        cache=False,
    )
    assert epochs.n_trials() == 3
    det = epochs.detect_erp_all(methods=["keller_zscore", "peak_amplitude"], min_consensus=1)
    assert "RSGC1" in det["channel"].values


def test_brain_graph_exports_and_zero_padded_labels(tmp_path):
    plotly = pytest.importorskip("plotly")
    assert plotly is not None
    config_path, events = _write_fixture(tmp_path)
    patient = ep.Patient("P1", config_path=str(config_path))
    canonical = ep.create_events_from_timestamps(events)
    event_path = patient.dataloader.get_event_path("A", "LACC_1_2")
    event_path.parent.mkdir(parents=True, exist_ok=True)
    canonical.to_csv(event_path, index=False)
    epochs = patient.epoch("A", "LACC_1_2", pipeline=[("artifact_blank", {"width_s": 0.005})], tmin=-0.2, tmax=0.4, baseline=(-0.2, -0.05), cache=False)
    detections = epochs.detect_erp_all(methods=["keller_zscore", "peak_amplitude"], min_consensus=1)

    from ERPy.viz.networks import (
        edges_from_detections,
        graph_from_edges,
        plot_aggregate_evoked_response_graph,
        plot_evoked_response_graph,
        response_matrix_from_edges,
    )

    padded_meta = pd.DataFrame(
        [
            {"elec_label": "PT01", "anat_label": "occipital", "mni_x": -44, "mni_y": -84, "mni_z": -3},
            {"elec_label": "PT03", "anat_label": "temporal", "mni_x": -58, "mni_y": -56, "mni_z": -12},
        ]
    )
    padded_det = pd.DataFrame(
        [{"channel": "PT03", "peak_amplitude_uv": 12.0, "consensus_ch": True}]
    )
    edges = edges_from_detections(padded_det, stim_pair="PT_1_2", elec_meta=padded_meta)
    assert edges.loc[0, "stim_elec"] == "PT01"
    assert edges.loc[0, "record_elec"] == "PT03"
    multi_source_edges = pd.concat(
        [
            edges,
            edges.assign(stim_pair="PT_3_4", stim_elec="PT03", record_elec="PT01", weight=4.0),
        ],
        ignore_index=True,
    )
    response_matrix = response_matrix_from_edges(multi_source_edges, elec_meta=padded_meta)
    assert response_matrix.index.tolist() == ["PT_1_2", "PT_3_4"]
    assert response_matrix.columns.tolist() == ["PT01", "PT03"]
    assert graph_from_edges(multi_source_edges).number_of_edges() == 2

    fig = plot_evoked_response_graph(
        epochs,
        patient.elec_meta,
        stim_pair="LACC_1_2",
        detections=detections,
        top_n=3,
        frame_step_ms=20,
    )
    assert len(fig.frames) > 1

    aggregate_fig = plot_aggregate_evoked_response_graph(
        [
            {"epochs": epochs, "stim_pair": "LACC_1_2", "detections": detections},
            {"epochs": epochs, "stim_pair": "LACC_2_3", "detections": detections},
        ],
        patient.elec_meta,
        top_n_per_stim=2,
        frame_step_ms=30,
    )
    assert len(aggregate_fig.frames) > 1
    assert aggregate_fig.data[-2].marker.colorbar.title.text == "active metrics"
    assert aggregate_fig.data[-1].marker.symbol == "diamond"
    assert aggregate_fig.data[-1].marker.size <= 11


def test_static_edges_exclude_both_stimulation_contacts_with_padded_labels():
    from ERPy.viz.networks import edges_from_detections

    detections = pd.DataFrame(
        [
            {
                "channel": channel,
                "peak_amplitude_uv": amplitude,
                "consensus_ch": True,
            }
            for channel, amplitude in (
                ("PT1-ref", 90.0),
                ("PT2-ref", 80.0),
                ("PT3-ref", 12.0),
            )
        ]
    )
    metadata = pd.DataFrame(
        [
            {
                "elec_label": channel,
                "anat_label": "test",
                "mni_x": float(index),
                "mni_y": 0.0,
                "mni_z": 0.0,
            }
            for index, channel in enumerate(("PT01", "PT02", "PT03"))
        ]
    )

    edges = edges_from_detections(
        detections,
        stim_pair="PT_1_2",
        elec_meta=metadata,
    )

    assert edges["record_elec"].tolist() == ["PT03"]
    assert edges["source"].tolist() == ["PT01"]


def test_edges_scope_repeated_electrode_labels_and_preserve_acquisitions():
    from ERPy.viz.networks import edges_from_detections

    detections = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "session_id": "A",
                "acquisition_id": "run001",
                "stim_pair": "ST_1_2",
                "channel": "C1",
                "method": "peak_amplitude",
                "consensus_ch": True,
                "artifact_qc_pass": True,
                "peak_amplitude_uv": 10.0,
            },
            {
                "patient_id": "P1",
                "session_id": "A",
                "acquisition_id": "run002",
                "stim_pair": "ST_1_2",
                "channel": "C1",
                "method": "peak_amplitude",
                "consensus_ch": True,
                "artifact_qc_pass": True,
                "peak_amplitude_uv": 20.0,
            },
            {
                "patient_id": "P1",
                "session_id": "B",
                "acquisition_id": "run003",
                "stim_pair": "ST_1_2",
                "channel": "C1",
                "method": "peak_amplitude",
                "consensus_ch": True,
                "artifact_qc_pass": True,
                "peak_amplitude_uv": 30.0,
            },
        ]
    )
    metadata = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "session_id": session,
                "elec_label": label,
                "anat_label": region,
                "mni_x": x,
                "mni_y": 0.0,
                "mni_z": 0.0,
            }
            for session, region, x in (
                ("A", "ACC", -10.0),
                ("B", "PAG", 5.0),
            )
            for label in ("ST1", "C1")
        ]
    )

    edges = edges_from_detections(detections, elec_meta=metadata)

    assert edges["acquisition_id"].tolist() == ["run001", "run002", "run003"]
    assert edges["record_region"].tolist() == ["ACC", "ACC", "PAG"]
    assert edges["record_mni_x"].tolist() == [-10.0, -10.0, 5.0]


def test_coordinate_completion_is_patient_scoped_and_auditable():
    rows = []
    for patient_id, offset in (("P1", 0.0), ("P2", 20.0)):
        for contact in range(1, 6):
            rows.append(
                {
                    "patient_id": patient_id,
                    "session_id": "A",
                    "elec_label": f"L{contact}",
                    "mni_x": np.nan if contact == 5 else offset + contact - 1,
                    "mni_y": np.nan if contact == 5 else float(contact - 1),
                    "mni_z": np.nan if contact == 5 else 0.0,
                }
            )
    rows.extend(
        {
            "patient_id": "P1",
            "session_id": "A",
            "elec_label": f"M{contact}",
            "mni_x": np.nan,
            "mni_y": np.nan,
            "mni_z": np.nan,
        }
        for contact in range(1, 6)
    )

    completed = ep.complete_contact_coordinates(pd.DataFrame(rows))
    endpoints = completed[completed["elec_label"].eq("L5")].set_index(
        "patient_id"
    )

    assert endpoints.loc["P1", "mni_x"] == pytest.approx(4.0)
    assert endpoints.loc["P2", "mni_x"] == pytest.approx(24.0)
    assert endpoints["mni_coordinate_source"].eq(
        "linear_lead_trajectory"
    ).all()
    assert endpoints["mni_coordinate_fit_n_contacts"].eq(4).all()
    assert completed.loc[
        completed["elec_label"].str.startswith("M"),
        ["mni_x", "mni_y", "mni_z"],
    ].isna().all().all()


def test_dynamic_graph_hub_authority_are_nonzero_for_directed_response_graph():
    from ERPy.graph_metrics import compute_dynamic_graph_metrics

    edges = pd.DataFrame(
        [
            {"time_s": 0.01, "time_ms": 10.0, "source": "S1", "target": "A", "weight": 3.0},
            {"time_s": 0.01, "time_ms": 10.0, "source": "S1", "target": "B", "weight": 2.0},
            {"time_s": 0.01, "time_ms": 10.0, "source": "S2", "target": "A", "weight": 1.0},
            {"time_s": 0.01, "time_ms": 10.0, "source": "S2", "target": "C", "weight": 5.0},
        ]
    )
    metrics = compute_dynamic_graph_metrics(edges, metrics=("hub", "authority"))
    hub = metrics[metrics["metric"] == "hub"].set_index("node")["value"]
    authority = metrics[metrics["metric"] == "authority"].set_index("node")["value"]
    assert hub[["S1", "S2"]].max() > 0
    assert authority[["A", "B", "C"]].max() > 0
    assert hub[["A", "B", "C"]].abs().max() == 0
    assert authority[["S1", "S2"]].abs().max() == 0


def test_dynamic_evoked_edges_are_invariant_to_trial_dc_offsets():
    from ERPy.graph_metrics import evoked_response_edges_over_time

    sfreq = 500.0
    times = np.arange(-0.2, 0.352, 1.0 / sfreq)
    response_a = 70.0 * np.exp(-0.5 * ((times - 0.08) / 0.018) ** 2)
    response_b = -45.0 * np.exp(-0.5 * ((times - 0.14) / 0.028) ** 2)
    base = []
    shifted = []
    index = []
    for epoch in range(6):
        trial = np.column_stack(
            [
                np.zeros_like(times),
                response_a + 0.2 * epoch,
                response_b - 0.1 * epoch,
            ]
        )
        offsets = np.array([1000.0, -2500.0, 4100.0]) + epoch * 37.0
        base.append(trial)
        shifted.append(trial + offsets)
        index.extend((epoch, float(time)) for time in times)

    def make_epochs(values):
        frame = pd.DataFrame(
            np.concatenate(values),
            index=pd.MultiIndex.from_tuples(
                index,
                names=["epoch", "time"],
            ),
            columns=["STIM1", "A1", "B1"],
        )
        return ep.Epochs(
            frame,
            sfreq=sfreq,
            tmin=-0.2,
            tmax=0.35,
            baseline=(-0.2, -0.03),
        )

    common = {
        "stim_pair": "STIM1_STIM2",
        "channels": ["A1", "B1"],
    }
    edges = evoked_response_edges_over_time(
        [{**common, "epochs": make_epochs(base)}],
        response_window=(0.01, 0.30),
        frame_step_ms=10.0,
        top_n_total=None,
        baseline_window=(-0.2, -0.03),
    )
    shifted_edges = evoked_response_edges_over_time(
        [{**common, "epochs": make_epochs(shifted)}],
        response_window=(0.01, 0.30),
        frame_step_ms=10.0,
        top_n_total=None,
        baseline_window=(-0.2, -0.03),
    )

    keys = ["time_s", "source", "target"]
    values = ["signed_value", "weight", "peak_magnitude"]
    edges = edges.sort_values(keys).reset_index(drop=True)
    shifted_edges = shifted_edges.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        edges[keys],
        shifted_edges[keys],
    )
    np.testing.assert_allclose(
        edges[values].to_numpy(dtype=float),
        shifted_edges[values].to_numpy(dtype=float),
        rtol=1e-10,
        atol=1e-10,
    )


def test_dynamic_evoked_edges_exclude_both_stimulation_contacts():
    from ERPy.graph_metrics import evoked_response_edges_over_time

    sfreq = 500.0
    times = np.arange(-0.2, 0.202, 1.0 / sfreq)
    response = np.exp(-0.5 * ((times - 0.08) / 0.018) ** 2)
    values = np.column_stack(
        [100.0 * response, 80.0 * response, 20.0 * response]
    )
    frame = pd.DataFrame(
        np.tile(values, (4, 1)),
        index=pd.MultiIndex.from_product(
            [range(4), times],
            names=["epoch", "time"],
        ),
        columns=["STIM1", "STIM2", "A1"],
    )
    epochs = ep.Epochs(
        frame,
        sfreq=sfreq,
        tmin=float(times[0]),
        tmax=float(times[-1]),
        baseline=(-0.2, -0.03),
        stim_ch=["STIM1", "STIM2"],
    )
    metadata = pd.DataFrame(
        [
            {
                "elec_label": channel,
                "anat_label": "test",
                "mni_x": float(index),
                "mni_y": 0.0,
                "mni_z": 0.0,
            }
            for index, channel in enumerate(("STIM1", "STIM2", "A1"))
        ]
    )

    edges = evoked_response_edges_over_time(
        [
            {
                "epochs": epochs,
                "stim_pair": "STIM_1_2",
                "channels": ["STIM1", "STIM2", "A1"],
            }
        ],
        elec_meta=metadata,
        response_window=(0.01, 0.18),
        frame_step_ms=20.0,
        top_n_total=None,
    )

    assert set(edges["target"]) == {"A1"}


def test_mni_coordinate_validation_drops_impossible_contacts():
    pytest.importorskip("nilearn")

    from ERPy.viz.networks import _filter_mni_brain_coordinates, validate_mni_coordinates

    meta = pd.DataFrame(
        [
            {"elec_label": "IN1", "mni_x": 0, "mni_y": -20, "mni_z": 20},
            {"elec_label": "OUT1", "mni_x": 140, "mni_y": 140, "mni_z": 140},
        ]
    )
    audited = validate_mni_coordinates(meta)
    assert audited.set_index("elec_label").loc["IN1", "inside_mni152_brain"]
    assert not audited.set_index("elec_label").loc["OUT1", "inside_mni152_brain"]

    coords = pd.DataFrame(
        [
            {"node": "IN1", "x": 0.0, "y": -20.0, "z": 20.0},
            {"node": "OUT1", "x": 140.0, "y": 140.0, "z": 140.0},
        ]
    )
    with pytest.warns(RuntimeWarning, match="outside the MNI152 brain mask"):
        filtered = _filter_mni_brain_coordinates(coords, outside="drop")
    assert filtered["node"].tolist() == ["IN1"]


def test_crp_visualization_helpers(tmp_path):
    import matplotlib.pyplot as plt

    config_path, events = _write_fixture(tmp_path)
    patient = ep.Patient("P1", config_path=str(config_path))
    event_path = patient.dataloader.get_event_path("A", "LACC_1_2")
    event_path.parent.mkdir(parents=True, exist_ok=True)
    ep.create_events_from_timestamps(events).to_csv(event_path, index=False)
    epochs = patient.epoch("A", "LACC_1_2", pipeline=[("artifact_blank", {"width_s": 0.005})], tmin=-0.2, tmax=0.4, baseline=(-0.2, -0.05), cache=False)

    from ERPy.crp import CRPConfig, compare_crp_across_stim_sites, run_crp_all
    from ERPy.response_metrics import zscore_metric_within_stim
    from ERPy.viz.crp import plot_crp_score_map, plot_crp_site_comparison, plot_crp_summary
    from ERPy.viz.response_metrics import plot_within_stim_zscore_bars, plot_within_stim_zscore_heatmap

    crp_table = run_crp_all(
        epochs,
        config=CRPConfig(response_window=(0.01, 0.3), baseline_window=(-0.2, -0.05), permutation_n=10),
    )
    assert {
        "channel",
        "score",
        "p_value",
        "significant",
        "max_canonical_weight_z",
        "reconstruction_rms_uv",
        "permutation_p_value",
    }.issubset(crp_table.columns)
    comparison = compare_crp_across_stim_sites({"LACC_1_2": crp_table, "LACC_2_3": crp_table})
    assert {"within_stim_z", "baseline_z", "reconstruction_rms_within_stim_z"}.issubset(comparison.columns)
    ax = plot_crp_score_map(crp_table)
    fig = plot_crp_summary(epochs, "RSGC1")
    ax2 = plot_crp_site_comparison(comparison, metric="within_stim_z")
    metric_table = zscore_metric_within_stim(
        pd.DataFrame(
            {
                "stim_pair": ["A_1_2", "A_1_2", "A_1_2", "A_2_3", "A_2_3"],
                "channel": ["C1", "C2", "C3", "C1", "C2"],
                "peak_amplitude_uv": [1.0, 3.0, 5.0, 2.0, 4.0],
            }
        ),
        "peak_amplitude_uv",
        absolute=True,
    )
    assert "peak_amplitude_uv_within_stim_z" in metric_table.columns
    ax3 = plot_within_stim_zscore_heatmap(metric_table, "peak_amplitude_uv")
    ax4 = plot_within_stim_zscore_bars(metric_table, "peak_amplitude_uv", stim_pair="A_1_2")
    assert ax.figure is not None
    assert fig is not None
    assert ax2.figure is not None
    assert ax3.figure is not None
    assert ax4.figure is not None
    plt.close(ax.figure)
    plt.close(fig)
    plt.close(ax2.figure)
    plt.close(ax3.figure)
    plt.close(ax4.figure)
