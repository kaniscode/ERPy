#!/usr/bin/env python3
"""Recover trial-level public ds004774 Mayo epochs using optional pymef.

Reads released good electrical-stimulation events and good ECOG channels.
No labels or detector results enter epoch selection. Voltage conversion is
read explicitly from each MEF channel, rather than assumed from integer data.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def extract_subject(data_root: Path, output: Path, subject: str) -> list[dict]:
    from pymef.mef_session import MefSession
    folder = next((data_root / f"sub-{subject}").glob("ses-*/ieeg"))
    sessions = sorted(folder.glob("*.mefd"))
    if len(sessions) != 1:
        raise ValueError(f"Expected one MEF session for {subject}")
    session_path = sessions[0]
    prefix = session_path.name.removesuffix("_ieeg.mefd")
    channels_path = folder / f"{prefix}_channels.tsv"
    events_path = folder / f"{prefix}_events.tsv"
    channels = pd.read_csv(channels_path, sep="\t")
    events = pd.read_csv(events_path, sep="\t")
    channels = channels.loc[(channels.type == "ECOG") & (channels.status == "good")].copy()
    events = events.loc[(events.trial_type == "electrical_stimulation") & (events.status == "good")].copy()
    labels = channels.name.astype(str).tolist()
    if len(labels) != len(set(labels)):
        raise ValueError("Channel labels are not unique")
    session = MefSession(str(session_path), password="", check_all_passwords=False)
    records = []
    try:
        channel_metadata = session.session_md["time_series_channels"]
        rates = [float(channel_metadata[c]["section_2"]["sampling_frequency"][0]) for c in labels]
        if not rates or not np.allclose(rates, rates[0]):
            raise ValueError("Expected a common sampling rate for retained channels")
        fs = rates[0]
        factors = np.array([channel_metadata[c]["section_2"]["units_conversion_factor"][0] for c in labels])
        units = [bytes(channel_metadata[c]["section_2"]["units_description"][0]).decode().strip('\0') for c in labels]
        if any(u.lower() not in {"microvolts", "microvolt", "uv", "µv"} for u in units):
            raise ValueError(f"Unexpected MEF voltage units: {set(units)}")
        counts = [int(channel_metadata[c]["section_2"]["number_of_samples"][0]) for c in labels]
        discontinuities = [int(channel_metadata[c]["section_2"]["number_of_discontinuities"][0]) for c in labels]
        # The released ieegprep MEF reader and BIDS events use sample-index
        # extraction. Use that convention, not a second absolute-clock mapping.
        starts = [int(channel_metadata[c]["channel_specific_metadata"]["earliest_start_time"][0]) for c in labels]
        if len(set(starts)) != 1:
            raise ValueError("Differing channel starts require explicit event realignment")
        offsets = np.arange(-round(fs), round(.35 * fs) + 1)
        times = offsets / fs
        baseline = (times >= -.5) & (times <= -.02)
        target_dir = output / subject
        target_dir.mkdir(parents=True, exist_ok=True)
        for pair, group in events.groupby("electrical_stimulation_site", sort=True):
            pair = str(pair)
            if len(group) < 5:
                continue
            destination = target_dir / (pair.replace("/", "_") + ".npz")
            if destination.exists():
                with np.load(destination, allow_pickle=False) as saved:
                    if str(saved["subject"]) != subject or str(saved["stimpair"]) != pair:
                        raise ValueError("Cached epoch identity mismatch")
                    records.append(json.loads(str(saved["metadata"])))
                continue
            trials, ranges, onsets = [], [], []
            for onset in group.onset.astype(float):
                center = int(round(onset * fs))
                start, stop = center + int(offsets[0]), center + int(offsets[-1]) + 1
                if start < 0 or stop > min(counts):
                    continue
                values = np.asarray(session.read_ts_channels_sample(labels, [start, stop]), dtype=np.float64).T
                if values.shape != (len(times), len(labels)):
                    raise ValueError("Unexpected decoded epoch shape")
                values *= factors[None, :]
                values -= np.nanmedian(values[baseline], axis=0, keepdims=True)
                trials.append(values.astype(np.float32))
                ranges.append([start, stop])
                onsets.append(onset)
            if not trials:
                continue
            metadata = {"dataset": "ds004774", "dataset_snapshot": "1.0.0", "subject": subject,
                        "site": "Mayo", "stimpair": pair, "sfreq": fs, "n_trials": len(trials),
                        "n_channels": len(labels), "units": "microvolts", "epoch_seconds": [float(times[0]), float(times[-1])],
                        "preprocessing": "No additional filter or rereference; subtract trial median over [-0.5,-0.02]s (released ER-detect config)",
                        "eligibility": "Released status=good electrical-stimulation events and status=good ECOG contacts; at least five events per pair before extraction",
                        "source_events_sha256": hashlib.sha256(events_path.read_bytes()).hexdigest(),
                        "source_channels_sha256": hashlib.sha256(channels_path.read_bytes()).hexdigest(),
                        "sample_ranges": ranges, "onsets_seconds": onsets,
                        "event_alignment": "Round BIDS onset times to sample indices, matching ieegprep's MEF reader; no absolute-clock remapping",
                        "mef_reported_discontinuities": discontinuities,
                        "mef_units_conversion_factors": factors.tolist(), "channel_names": labels}
            temporary = destination.with_suffix(".partial.npz")
            np.savez_compressed(temporary, values=np.stack(trials), times=times, channels=np.array(labels),
                                subject=subject, stimpair=pair, sfreq=fs, metadata=json.dumps(metadata))
            temporary.replace(destination)
            records.append(metadata)
            print(f"{subject} {pair}: {len(trials)} trials, {len(labels)} channels", flush=True)
    finally:
        session.close()
    (output / subject / "epoch_manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    return records


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data_root", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--subject", action="append", required=True)
    a = p.parse_args()
    for subject in a.subject:
        extract_subject(a.data_root, a.output, subject)
