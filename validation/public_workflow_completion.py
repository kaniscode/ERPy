"""Complete the current public ds003708 acquisition's BH/QC/final audit.

All 89 source channels and all 17 events for the declared stimulation pair are
accounted for. EKG is an explicit auxiliary-channel exclusion. Raw range data
stay in a local work directory; public outputs contain derived evidence only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import ERPy as ep
from ERPy.io import parse_brainvision_header
from validation.validate_public_spes import electrode_metadata_for_channels

OUTPUT = ROOT / "validation/public_workflow_completion"
SOURCE_PIN = ROOT / "validation/public_ds003708_source_manifest.json"
# Recorded in the prior executed public recipe, independently checked again
# against the explicit S3 object version before this new analysis.
RANGE_SHA256 = "134ec1b0c9b80101596fc38343eb300e4ab0d104a35d8a4ab3b252856a4abb5b"
STIM_CONTACTS = {"LTG1", "LTG2"}
AUXILIARY = {"EKG"}
GATE = dict(max_bad_response_fraction=.25, max_hard_artifact_fraction=.10,
            min_clean_responses=8, exclude_boundary_peaks=True)
PIPELINE = [
    ("reject_bad_channels", {"bad_channels": ["EKG"], "method": "drop"}),
    ("reject_bad_channels", {"bad_channels": "auto", "method": "drop", "detection_params": {"zscore_threshold": 8.0}}),
    ("artifact_blank", {"width_s": .004}),
    ("notch_filter", {"notch_freq": 60., "bw": 2., "harm": True}),
    ("bandpass_filter", {"lowcut": .5, "highcut": 80., "order": 3}),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_file(path: Path, size: int, digest: str) -> bool:
    if not path.exists():
        return False
    if path.stat().st_size != size or sha256(path) != digest:
        raise ValueError(f"Cached public source identity mismatch: {path.name}")
    return True


def fetch_pinned(source: dict, path: Path, *, byte_range=None, digest=None) -> Path:
    """Fail closed on mutable identity, range, size or content disagreement."""
    if not source.get("version_id"):
        raise ValueError("An explicit immutable S3 version is required")
    size = source["bytes"] if byte_range is None else byte_range[1] - byte_range[0]
    expected = digest or source["sha256"]
    if _verified_file(path, size, expected):
        return path
    url = source["url"] + "?versionId=" + quote(source["version_id"], safe="")
    headers = {} if byte_range is None else {"Range": f"bytes={byte_range[0]}-{byte_range[1]-1}"}
    response = requests.get(url, headers=headers, timeout=(30, 180), stream=True)
    response.raise_for_status()
    if response.headers.get("x-amz-version-id") != source["version_id"]:
        raise ValueError("The response does not identify the requested S3 version")
    if byte_range is not None:
        expected_range = f"bytes {byte_range[0]}-{byte_range[1]-1}/{source['bytes']}"
        if response.status_code != 206 or response.headers.get("Content-Range") != expected_range:
            raise ValueError("The response does not match the requested full-channel byte range")
    elif response.status_code != 200:
        raise ValueError("Unexpected metadata response status")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    try:
        with temporary.open("wb") as stream:
            for chunk in response.iter_content(8 * 1024 * 1024):
                stream.write(chunk)
        _verified_file(temporary, size, expected)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
        response.close()
    return path


def independent_bh(p_values: np.ndarray, alpha=.05):
    """Reconstruct finite-family BH with a stable sort and reverse cumulative minimum."""
    values = np.asarray(p_values, dtype=float)
    finite = np.isfinite(values)
    if values.ndim != 1 or ((values[finite] < 0) | (values[finite] > 1)).any():
        raise ValueError("BH input must be a vector of probabilities or unavailable values")
    output = np.full(len(values), np.nan)
    indices = np.flatnonzero(finite)
    order = indices[np.argsort(values[indices], kind="stable")]
    if len(order):
        adjusted = values[order] * len(order) / np.arange(1, len(order) + 1)
        output[order] = np.minimum(np.minimum.accumulate(adjusted[::-1])[::-1], 1.)
    return output, finite & (output <= alpha)


def complete_contact_table(qc: pd.DataFrame, source_channels, epoch_channels, gate=GATE) -> pd.DataFrame:
    """Independently validate BH and the separate gate before exporting the intersection."""
    needed = {"channel", "method", "qc_status", "p_joint", "q_joint", "testing_family_size",
        "primary_significant", "artifact_qc_pass", "primary_qc_pass", "bad_response_fraction",
        "hard_artifact_fraction", "n_clean_response", "artifact_boundary_peak_fail"}
    if not needed.issubset(qc):
        raise ValueError(f"Incomplete current analysis output: {sorted(needed-set(qc))}")
    if len(source_channels) != len(set(source_channels)):
        raise ValueError("Source channel identities must be unique")
    data = qc.loc[qc.method.eq("crp_energy")].copy()
    if data.empty or data.channel.duplicated().any() or not set(data.channel) <= set(source_channels):
        raise ValueError("Primary contact rows must be unique and belong to this source acquisition")
    expected = set(epoch_channels) - STIM_CONTACTS - AUXILIARY
    if not expected <= set(data.channel):
        raise ValueError("A retained non-stimulation channel is missing its primary result")
    for key in ("primary_significant", "artifact_qc_pass", "primary_qc_pass", "artifact_boundary_peak_fail"):
        if not data[key].map(lambda x: isinstance(x, (bool, np.bool_))).all():
            raise ValueError(f"{key} must contain actual Boolean decisions")
    eligible = data.qc_status.eq("pass") & np.isfinite(data.p_joint)
    if (eligible & data.channel.isin(STIM_CONTACTS | AUXILIARY)).any():
        raise ValueError("An excluded source contact entered the finite BH family")
    p = data.p_joint.where(eligible).to_numpy(dtype=float)
    q, rejected = independent_bh(p)
    if not data.testing_family_size.eq(int(eligible.sum())).all():
        raise ValueError("Reported BH denominator differs from the finite eligible family")
    if not np.allclose(data.q_joint.to_numpy(float), q, equal_nan=True, atol=2e-15, rtol=0):
        raise ValueError("Reported BH q-values disagree with an independent reconstruction")
    if not np.array_equal(data.primary_significant.to_numpy(bool), rejected):
        raise ValueError("Reported BH calls disagree with reconstructed q-values")
    if data[["bad_response_fraction", "hard_artifact_fraction", "n_clean_response"]].isna().any().any():
        raise ValueError("Gate inputs are missing")
    passed = ((data.bad_response_fraction < gate["max_bad_response_fraction"])
        & (data.hard_artifact_fraction < gate["max_hard_artifact_fraction"])
        & (data.n_clean_response >= gate["min_clean_responses"])
        & ~data.artifact_boundary_peak_fail)
    if not np.array_equal(passed, data.artifact_qc_pass) or not np.array_equal(rejected & passed, data.primary_qc_pass):
        raise ValueError("The separate contact gate or final intersection is inconsistent")
    indexed = data.set_index("channel")
    rows = []
    for index, channel in enumerate(source_channels, 1):
        native = indexed.loc[channel] if channel in indexed.index else None
        stim, auxiliary = channel in STIM_CONTACTS, channel in AUXILIARY
        if auxiliary: disposition = "auxiliary_EKG_excluded_before_processing"
        elif channel not in epoch_channels: disposition = "not_retained_by_persistent_channel_screen"
        elif stim: disposition = "stimulation_contact_excluded_from_testing"
        elif native is None: raise AssertionError("Missing contact result")
        else: disposition = str(native.qc_status)
        row = dict(source_channel_number=index, channel=channel, is_stimulation_contact=stim,
            is_auxiliary_channel=auxiliary, retained_in_epochs=channel in epoch_channels,
            disposition=disposition, bh_tested=bool(native is not None and native.qc_status == "pass" and np.isfinite(native.p_joint)),
            bh_rejected=bool(native is not None and native.primary_significant),
            qc_gate_available=native is not None,
            qc_gate_pass=bool(native.artifact_qc_pass) if native is not None else None,
            final_positive=bool(native is not None and native.primary_qc_pass))
        for key in ("p_crp", "p_energy", "p_joint", "q_joint", "testing_family_size", "n_trials_total", "n_trials_clean",
                    "bad_response_fraction", "hard_artifact_fraction", "n_clean_response", "artifact_boundary_peak_fail",
                    "artifact_reasons", "random_state_root", "random_state_effective", "random_stream_derivation"):
            row[key] = native.get(key) if native is not None else None
        rows.append(row)
    return pd.DataFrame(rows)


def source_records(paths, base):
    return [dict(path=Path(os.path.relpath(path, base)).as_posix(), bytes=path.stat().st_size, sha256=sha256(path)) for path in paths]


def clean_trial_membership(qc, epoch_ids, source_events):
    """Expand the detector's own retained indices, bound to original event samples."""
    if len(epoch_ids) != len(source_events) or len(epoch_ids) != len(set(epoch_ids)):
        raise ValueError("Epochs must map one-to-one to the selected source events")
    rows = []
    for native in qc.loc[qc.method.eq("crp_energy")].itertuples():
        indices = np.asarray(native.clean_trial_indices)
        if (indices.ndim != 1 or (len(indices) and indices.dtype.kind not in "iu")
                or len(indices) != native.n_trials_clean or native.n_trials_total != len(epoch_ids)
                or len(set(indices)) != len(indices) or ((indices < 0) | (indices >= len(epoch_ids))).any()):
            raise ValueError("The native clean-trial indices do not match the epoch/source contract")
        selected = set(indices)
        for index, (epoch, event) in enumerate(zip(epoch_ids, source_events.itertuples())):
            rows.append(dict(channel=native.channel, epoch_array_index=index, epoch_label=int(epoch),
                source_event_row=int(event.Index), source_sample=int(event.sample_start),
                source_onset_s=float(event.onset), used_for_joint_inference=index in selected))
    return pd.DataFrame(rows)


def verify_evidence(output):
    record = json.loads((output / "manifest.json").read_text())
    for artifact in record["artifacts"]:
        if not _verified_file(output / artifact["path"], artifact["bytes"], artifact["sha256"]):
            raise ValueError(f"Missing artifact: {artifact['path']}")
    print("Public workflow evidence hashes and sizes verified.")


def run(output, work):
    if (output / "manifest.json").exists():
        raise FileExistsError("Completed evidence is immutable; use a new --output directory")
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    pin = json.loads(SOURCE_PIN.read_text())
    objects = pin["objects"]
    source_paths = {}
    for key, name in [("vhdr_url", "source_header.vhdr"), ("events_url", "source_events.tsv"), ("electrodes_url", "source_electrodes.tsv")]:
        source_paths[key] = fetch_pinned(objects[key], work / name)
    header_text = source_paths["vhdr_url"].read_text(encoding="utf-8")
    header = parse_brainvision_header(header_text)
    if header["n_channels"] != 89 or header["sfreq"] != 2048. or "DataOrientation=MULTIPLEXED" not in header_text or "BinaryFormat=IEEE_FLOAT_32" not in header_text:
        raise ValueError("The pinned BrainVision encoding or acquisition dimensions differ")
    channels = list(header["channels"])
    if set(channels) & AUXILIARY != AUXILIARY or not STIM_CONTACTS <= set(channels):
        raise ValueError("Expected stimulation/auxiliary source identities are absent")
    first, stop = pin["window"]["sample_range"]
    byte_range = (first * 89 * 4, stop * 89 * 4)
    binary = fetch_pinned(objects["eeg_url"], work / "signal_range.eeg", byte_range=byte_range, digest=RANGE_SHA256)
    print(f"Verified {binary.stat().st_size:,} source bytes; 89 channels; processing all17 selected events.", flush=True)
    # Only the local dependency filename changes. Absolute sample positions are
    # preserved by placing the range's first sample at its original timestamp.
    local_header = re.sub(r"(?m)^DataFile=.*$", "DataFile=signal_range.eeg", header_text)
    local_header = re.sub(r"(?m)^MarkerFile=.*$", "MarkerFile=", local_header)
    header_path = work / "signal_range.vhdr"
    header_path.write_text(local_header, encoding="utf-8")
    events = pd.read_csv(source_paths["events_url"], sep="\t")
    block = events.loc[events.electrical_stimulation_site.eq(pin["stim_site"]) & events.status.eq("good")].sort_values("onset")
    if len(block) != 17 or not np.allclose(block.onset.to_numpy(float)*2048, block.sample_start, atol=1e-6, rtol=0):
        raise ValueError("The selected event count or exact source sample mapping changed")
    base = pd.Timestamp("2026-01-01")  # Explicit synthetic clock, not an acquisition date.
    start_time = base + pd.to_timedelta(first / 2048, unit="s")
    stop_time = base + pd.to_timedelta((stop-1) / 2048, unit="s")
    patient_id, session_id, stim_pair = "DS003708", "A", "LTG_1_2"
    meta = work / "metadata"; meta.mkdir(exist_ok=True)
    pd.DataFrame([dict(patient_id=patient_id, session_id=session_id, raw_file=header_path.name,
        start_time=start_time.isoformat(), stop_time=stop_time.isoformat(), sampling_freq=2048.)]).to_csv(meta / "raw_metadata.csv", index=False)
    pd.DataFrame([dict(patient_id=patient_id, session_id=session_id, stim_pair=stim_pair,
        stim_start=start_time.isoformat(), stim_stop=stop_time.isoformat(),
        stim_freq=float(1/np.median(np.diff(block.onset))))]).to_csv(meta / "stim_metadata.csv", index=False)
    electrodes = pd.read_csv(source_paths["electrodes_url"], sep="\t")
    electrode_metadata_for_channels(electrodes, channels, patient_id, session_id).to_csv(meta / "electrode_metadata.csv", index=False)
    config = dict(rawdata_path=str(work), procdata_path=str(work / "procdata"),
        rawdata_meta_path=str(meta / "raw_metadata.csv"), stim_meta_path=str(meta / "stim_metadata.csv"),
        elec_meta_path=str(meta / "electrode_metadata.csv"), session_ids={patient_id: [session_id]})
    config_path = work / "config.yaml"; config_path.write_text(yaml.safe_dump(config))
    patient = ep.Patient(patient_id, config_path=str(config_path))
    canonical_events = ep.create_events_from_timestamps(base + pd.to_timedelta(block.onset, unit="s"))
    # Reproduce the current loader's clock construction and its searchsorted
    # epoch alignment. The timestamp rebase must not shift any event sample.
    source_clock = pd.DatetimeIndex(start_time + pd.to_timedelta(np.arange(stop-first) / 2048, unit="s"))
    realized_source_samples = source_clock.searchsorted(pd.to_datetime(canonical_events.times)) + first
    if not np.array_equal(realized_source_samples, block.sample_start.to_numpy(int)):
        raise ValueError("Local clock rebasing shifted a source event sample")
    event_path = patient.dataloader.get_event_path(session_id, stim_pair)
    event_path.parent.mkdir(parents=True, exist_ok=True); canonical_events.to_csv(event_path, index=False)
    package_files = sorted((ROOT / "ERPy").rglob("*.py"))
    implementation_before = {str(path.relative_to(ROOT)): sha256(path) for path in package_files}
    result = patient.recording(session_id).analyze(stim_pair, pipeline=PIPELINE,
        methods=["crp_energy"], artifact_kwargs={"zscore_threshold": 3.}, artifact_reject="nan_response",
        qc_max_bad_response_fraction=.25, qc_max_hard_artifact_fraction=.10,
        qc_min_clean_responses=8, qc_exclude_boundary_peaks=True,
        detect_kwargs={"crp_energy": {"random_state": 42, "return_arrays": True}},
        tmin=-.5, tmax=.6, baseline=(-.5, -.03), cache=False, save_epochs=False, input_source="raw")
    if implementation_before != {str(path.relative_to(ROOT)): sha256(path) for path in package_files}:
        raise RuntimeError("ERPy source changed during the demonstration")
    table = complete_contact_table(result.qc_detections, channels, result.epochs.channels)
    epoch_ids = result.epochs.epochs_df.index.get_level_values("epoch").unique().tolist()
    trial_membership = clean_trial_membership(result.qc_detections, epoch_ids, block)
    paths = result.save(output, prefix="acquisition")
    table.to_csv(output / "contact_decisions.csv", index=False)
    trial_membership.to_csv(output / "contact_clean_trials.csv", index=False)
    pd.DataFrame(dict(epoch_sample_index=np.arange(len(result.epochs.times)),
        epoch_time_s=result.epochs.times)).to_csv(output / "epoch_sample_grid.csv", index=False)
    block.to_csv(output / "selected_source_events.tsv", sep="\t", index=False)
    (output / "source_header.vhdr").write_bytes(source_paths["vhdr_url"].read_bytes())
    eligible = ~table.is_stimulation_contact & ~table.is_auxiliary_channel
    summary = dict(dataset="ds003708", snapshot=pin["snapshot"], source_pair=pin["stim_site"],
        source_channels=len(channels), source_events=len(block), epochs=result.epochs.n_trials(),
        epoch_channels=len(result.epochs.channels), auxiliary_exclusions=sorted(AUXILIARY),
        stimulation_contacts=sorted(STIM_CONTACTS), finite_BH_family=int(table.bh_tested.sum()),
        bh_rejected=int(table.bh_rejected.sum()), qc_gate_available_nonstim=int((eligible & table.qc_gate_available).sum()),
        qc_gate_pass_nonstim=int((eligible & table.qc_gate_pass.eq(True)).sum()),
        bh_and_qc_final=int(table.final_positive.sum()),
        bh_positive_gate_negative=int((table.bh_rejected & ~table.final_positive).sum()),
        retained_nonstim_without_finite_test=int((eligible & table.retained_in_epochs & ~table.bh_tested).sum()),
        independent_BH_gate_intersection_checks="pass", erpy_version=ep.__version__,
        historical_results_changed=False,
        scope="Current-source workflow demonstration on the complete 89-channel, 17-event public stimulation acquisition; no external accuracy or unconditional FDR calibration claim")
    (output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    provenance = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
        package_version=ep.__version__, python=platform.python_version(),
        package_versions={name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "h5py", "tables")},
        source_git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_git_commit_role="Contextual HEAD; manifest hashes identify the executed working tree",
        source_identity=pin, full_range=dict(sample_start=first, sample_stop_exclusive=stop,
            byte_start=byte_range[0], byte_stop_exclusive=byte_range[1], bytes=binary.stat().st_size, sha256=sha256(binary)),
        local_BrainVision_rebase="Binary samples unchanged; only DataFile renamed and unused MarkerFile cleared; first sample timestamp = synthetic clock + source sample_start /2048",
        synthetic_clock="2026-01-01 is an arbitrary software clock, not the actual recording date",
        sample_grid=dict(header_sfreq_hz=2048., epoch_sfreq_hz=float(result.epochs.sfreq),
            samples_per_epoch=len(result.epochs.times), minimum_time_s=float(result.epochs.times.min()),
            maximum_time_s=float(result.epochs.times.max()), source_event_sample_offsets=(realized_source_samples-block.sample_start.to_numpy(int)).tolist(),
            note="The native loader uses nanosecond timestamps; current preprocessing infers a rate from that grid. The exact realized epoch times and detector windows are retained, rather than silently substituting nominal header times."),
        processing=dict(pipeline=PIPELINE, recording_analyze_metadata=paths["metadata"].name,
            event_onset_samples_verified=True, cache=False, historical_epochs_reused=False),
        local_input_artifacts=[dict(name=path.name, bytes=path.stat().st_size, sha256=sha256(path)) for path in [binary, header_path, event_path, *sorted(meta.glob("*.csv"))]],
        source_objects=[dict(key=key, bytes=path.stat().st_size, sha256=sha256(path)) for key,path in source_paths.items()],
        notes="All89 header channels are accounted for. EKG is excluded by its source label before persistent screening; stimulation contacts are excluded from the testing family. No record was selected by its detector result.")
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    artifacts = [*package_files, Path(__file__).resolve(), ROOT / "validation/validate_public_spes.py", SOURCE_PIN,
        ROOT / "tests/test_public_workflow_completion.py", OUTPUT / "README.md",
        *[path for path in sorted(output.iterdir()) if path.is_file() and path.name not in {"README.md", "manifest.json"}]]
    manifest = dict(schema_version=1, purpose="Current-source full acquisition BH, separate QC gate and final intersection demonstration",
        artifacts=source_records(artifacts, output))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    verify_evidence(output)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--work-dir", type=Path, default=Path(tempfile.gettempdir()) / "erpy-public-workflow-completion")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify: verify_evidence(args.output.resolve())
    else: run(args.output.resolve(), args.work_dir.resolve())


if __name__ == "__main__":
    main()
