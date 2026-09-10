"""Behavioral regressions for automatic facade epoch cache invalidation."""
from pathlib import Path
import os

import numpy as np
import pandas as pd
import pytest
import yaml

from ERPy import Patient


def write_events(path, seconds):
    values = pd.Timestamp("2026-01-01") + pd.to_timedelta(seconds, unit="s")
    pd.DataFrame({"times": values}).to_csv(path, index=False)


@pytest.fixture
def project(tmp_path):
    raw = tmp_path / "signal.csv"
    times = np.arange(1000) / 100
    timestamps = pd.Timestamp("2026-01-01") + pd.to_timedelta(times, unit="s")
    pd.DataFrame({"times": timestamps, "A1": np.sin(times), "A2": np.cos(times),
                  "B1": np.ones(len(times))}).to_csv(raw, index=False)
    metadata = {
        "rawdata_meta_path": pd.DataFrame([dict(patient_id="P1", session_id="A", raw_file=raw.name, sampling_freq=100)]),
        "stim_meta_path": pd.DataFrame([dict(patient_id="P1", session_id="A", stim_pair="A_1_2", stim_start=0,
                                             stim_stop=10, stim_freq=0.5)]),
        "elec_meta_path": pd.DataFrame([dict(patient_id="P1", session_id="A", elec_label="B1", bad_channel=False)]),
    }
    config = {"rawdata_path": str(tmp_path), "procdata_path": str(tmp_path / "proc")}
    for key, frame in metadata.items():
        path = tmp_path / f"{key}.csv"
        frame.to_csv(path, index=False)
        config[key] = str(path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    patient = Patient("P1", config_path=str(config_path), verbose=False)
    recording = patient.recording("A")
    events = patient.dataloader.get_event_path("A", "A_1_2")
    write_events(events, [2.0, 4.0, 6.0])
    return recording, raw, events, config_path


def run(recording, **kwargs):
    options = dict(pipeline="raw", tmin=-0.1, tmax=0.2, baseline=None, zero_time=None)
    options.update(kwargs)
    return recording.epoch("A_1_2", **options)


def count_processing(recording, monkeypatch):
    calls = []
    original = recording.pipeline_obj.process_and_epoch

    def counted(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(recording.pipeline_obj, "process_and_epoch", counted)
    return calls


def test_unchanged_inputs_reuse_across_patient_instances(project, monkeypatch):
    recording, _, _, config = project
    first = run(recording)
    assert first.metadata["epoch_cache_identity"] is not None
    reopened = Patient("P1", config_path=str(config), verbose=False).recording("A")
    calls = count_processing(reopened, monkeypatch)
    second = run(reopened)
    assert calls == []
    # Epoch HDF serialization intentionally uses float32 signal storage.
    np.testing.assert_allclose(first.as_array()[0], second.as_array()[0], rtol=1e-6, atol=1e-7)


def test_current_event_contents_replace_cached_trials(project, monkeypatch):
    recording, _, events, _ = project
    assert run(recording).n_trials() == 3
    write_events(events, [2.0, 4.0])
    calls = count_processing(recording, monkeypatch)
    result = run(recording)
    assert result.n_trials() == 2 and len(calls) == 1
    assert run(recording).n_trials() == 2 and len(calls) == 1


def test_malformed_current_events_cannot_reuse_old_epochs(project):
    recording, _, events, _ = project
    run(recording)
    events.write_text("wrong_column\n2\n")
    with pytest.raises(ValueError, match="times"):
        run(recording)


def test_missing_events_are_regenerated_before_cache_lookup(project, monkeypatch):
    recording, _, events, _ = project
    run(recording)
    events.unlink()

    def regenerate(*args, **kwargs):
        write_events(events, [4.0])

    monkeypatch.setattr(recording, "events_for", regenerate)
    assert run(recording).n_trials() == 1


def test_same_size_same_mtime_raw_edit_bypasses_memory_and_window_cache(project):
    recording, raw, _, _ = project
    first = run(recording)
    before = raw.stat()
    raw.write_bytes(raw.read_bytes().replace(b",1.0\n", b",9.0\n"))
    assert raw.stat().st_size == before.st_size
    os.utime(raw, ns=(before.st_atime_ns, before.st_mtime_ns))
    second = run(recording)
    np.testing.assert_allclose(first.epochs_df["B1"], 1)
    np.testing.assert_allclose(second.epochs_df["B1"], 9)


@pytest.mark.parametrize("options", [
    {"tmin": -0.2}, {"tmax": 0.3}, {"baseline": (-0.1, -0.02)},
    {"zero_time": 0.0}, {"processing_margin_s": 0.5},
    {"artifact_anchor_params": {"search_end_s": 0.04}},
    {"pipeline": [("artifact_blank", {"width_s": 0.02})]},
])
def test_changed_processing_settings_recompute_and_are_forwarded(project, monkeypatch, options):
    recording = project[0]
    run(recording)
    calls = count_processing(recording, monkeypatch)
    run(recording, **options)
    assert len(calls) == 1
    if "processing_margin_s" in options:
        assert calls[0]["processing_margin_s"] == options["processing_margin_s"]


def test_changed_effective_scale_changes_actual_signal(project):
    recording = project[0]
    run(recording)
    recording.dataloader.config["raw_value_scale_to_uv"] = 5.0
    result = run(recording)
    np.testing.assert_allclose(result.epochs_df["B1"], 5)


def test_changed_electrode_metadata_updates_channel_selection(project):
    recording = project[0]
    steps = [("reject_bad_channels", {})]
    assert "B1" in run(recording, pipeline=steps).channels
    recording.dataloader.elec_meta.loc[:, "bad_channel"] = True
    assert "B1" not in run(recording, pipeline=steps).channels


def test_unbound_legacy_cache_recomputes(project, monkeypatch):
    recording = project[0]
    result = run(recording)
    path = recording.pipeline_obj.find_epochs_files("A", "A_1_2", [])[0]
    result.metadata.pop("epoch_cache_identity")
    result.epochs_df.loc[:, "B1"] = 99
    result.to_hdf(path)
    calls = count_processing(recording, monkeypatch)
    fresh = run(recording)
    assert len(calls) == 1
    np.testing.assert_allclose(fresh.epochs_df["B1"], 1)


def test_custom_hook_never_reuses_unverifiable_closure(project):
    recording = project[0]
    state = {"offset": 2}
    recording.pipeline_obj.registry["custom"] = lambda df, **kwargs: df + state["offset"]
    first = run(recording, pipeline=[("custom", {})])
    assert first.metadata["epoch_cache_identity"] is None
    state["offset"] = 5
    second = run(recording, pipeline=[("custom", {})])
    np.testing.assert_allclose(first.epochs_df["B1"], 3)
    np.testing.assert_allclose(second.epochs_df["B1"], 6)


def test_cache_only_window_is_identified_and_changed_contents_reloaded(project):
    recording, raw, _, _ = project
    run(recording)
    raw.unlink()
    first = run(recording)
    assert first.metadata["epoch_source_kind"] == "cache"
    path = recording.dataloader.get_raw_cache_path("A", "A_1_2")
    path.write_bytes(path.read_bytes().replace(b",1.0\n", b",7.0\n"))
    second = run(recording)
    np.testing.assert_allclose(second.epochs_df["B1"], 7)


def test_changed_inputs_during_processing_are_not_saved(project, monkeypatch):
    recording, _, events, _ = project
    run(recording)
    path = recording.pipeline_obj.find_epochs_files("A", "A_1_2", [])[0]
    saved = path.read_bytes()
    original = recording.pipeline_obj.process_and_epoch

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        events.write_text("times\n4.0\n")
        return result

    monkeypatch.setattr(recording.pipeline_obj, "process_and_epoch", changed)
    with pytest.raises(RuntimeError, match="changed during processing"):
        run(recording, cache=False)
    assert path.read_bytes() == saved


def test_brainvision_binary_header_and_marker_dependencies(project, monkeypatch):
    recording, raw, events, _ = project
    pd.DataFrame({"times": [2.0, 4.0, 6.0]}).to_csv(events, index=False)
    header = raw.with_suffix(".vhdr")
    binary = raw.with_suffix(".eeg")
    marker = raw.with_suffix(".vmrk")
    np.ones(1000, dtype="<f4").tofile(binary)
    marker.write_text("Brain Vision Data Exchange Marker File\n")
    header.write_text("[Common Infos]\nDataFile=signal.eeg\nMarkerFile=signal.vmrk\nNumberOfChannels=1\nSamplingInterval=10000\nDataOrientation=MULTIPLEXED\n[Binary Infos]\nBinaryFormat=IEEE_FLOAT_32\n[Channel Infos]\nCh1=B1,,1,uV\n")
    recording.dataloader.raw_meta.loc[:, "raw_file"] = header.name
    np.testing.assert_allclose(run(recording).epochs_df["B1"], 1)
    calls = count_processing(recording, monkeypatch)
    np.full(1000, 4, dtype="<f4").tofile(binary)
    np.testing.assert_allclose(run(recording).epochs_df["B1"], 4)
    header.write_text(header.read_text().replace("Ch1=B1,,1,uV", "Ch1=B1,,2,uV"))
    np.testing.assert_allclose(run(recording).epochs_df["B1"], 8)
    marker.write_text(marker.read_text() + "[Marker Infos]\n")
    run(recording)
    assert len(calls) == 3
    binary.unlink()
    with pytest.raises(FileNotFoundError):
        run(recording)


def test_staging_rejects_same_size_old_signal(project, tmp_path):
    recording, raw, _, _ = project
    loader = recording.dataloader
    loader.RAW_STAGING_PATH = tmp_path / "staging"
    staged, _ = loader._stage_raw_file(raw)
    raw.write_bytes(raw.read_bytes().replace(b",1.0\n", b",8.0\n"))
    assert staged.read_bytes() != raw.read_bytes()
    fresh, _ = loader._stage_raw_file(raw)
    assert fresh.read_bytes() == raw.read_bytes()


def test_explicit_acquisition_source_is_forwarded_and_bound(project):
    recording, raw, _, _ = project
    run(recording)
    alternate = raw.with_name("other.csv")
    alternate.write_bytes(raw.read_bytes().replace(b",1.0\n", b",6.0\n"))
    row = recording.dataloader.raw_meta.iloc[0].copy()
    row["raw_file"] = alternate.name
    recording.dataloader.raw_meta = pd.concat([recording.dataloader.raw_meta, row.to_frame().T], ignore_index=True)
    result = run(recording, raw_file=alternate.name)
    np.testing.assert_allclose(result.epochs_df["B1"], 6)


def test_nwb_dependencies_are_explicitly_unverifiable(project):
    from ERPy._epoch_cache import epoch_cache_identity
    recording, raw, events, _ = project
    # No interpretation of this placeholder is needed: every NWB input is
    # conservatively uncached until external HDF5 dependencies are enumerated.
    nwb = raw.with_suffix(".nwb")
    nwb.write_bytes(b"placeholder")
    recording.dataloader.raw_meta.loc[:, "raw_file"] = nwb.name
    identity, source = epoch_cache_identity(recording.pipeline_obj, "A", "A_1_2", [], events.read_bytes(), {})
    assert identity is None and source == "raw"
