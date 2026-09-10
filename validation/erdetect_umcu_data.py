#!/usr/bin/env python3
"""Independently retrieve ds004774 UMCU trial windows; no detector or label tuning.

Only the public BrainVision storage specification and source metadata are used.
Voltage channels are cached in microvolts; auxiliary nonvoltage channels retain
their declared units. No filtering, referencing, baseline subtraction, artifact
rejection, or selection by response annotation values is applied.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import re
import threading
import time
from urllib.parse import quote, urlencode

import numpy as np
import pandas as pd
import requests
from scipy.io import loadmat

DATASET = "ds004774"
COMMIT = "582149a14a29c29f765d71ed61ba376047cd6686"
S3 = "https://s3.amazonaws.com/openneuro.org/"
GIT = f"https://raw.githubusercontent.com/OpenNeuroDatasets/{DATASET}/{COMMIT}/"
SUBJECTS = ("UMCU20", "UMCU21", "UMCU22", "UMCU23", "UMCU25", "UMCU26", "UMCU59", "UMCU62", "UMCU67")
_LOCAL = threading.local()


def _session() -> requests.Session:
    if not hasattr(_LOCAL, "session"):
        _LOCAL.session = requests.Session()
    return _LOCAL.session


def _request(url: str, *, method: str = "GET", headers: dict | None = None) -> requests.Response:
    for attempt in range(4):
        try:
            response = _session().request(method, url, headers=headers, timeout=(15, 90))
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _blob_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def verified_metadata(relative: str, root: Path, tree: dict) -> tuple[bytes, dict]:
    """Validate ordinary Git blobs or annex SHA256 content before using metadata."""
    entry = tree[relative]
    path = root / "dataset" / relative
    if entry["mode"] == "120000":
        pointer = _request(GIT + quote(relative, safe="/")).content
        if _blob_sha1(pointer) != entry["sha"]:
            raise ValueError(f"Pinned annex pointer hash mismatch: {relative}")
        match = re.search(rb"SHA256E-s(\d+)--([0-9a-f]{64})", pointer)
        if not match:
            raise ValueError(f"Unsupported annex key: {relative}")
        expected_size, expected_sha = int(match[1]), match[2].decode()
        valid = lambda b: len(b) == expected_size and _sha256(b) == expected_sha
    else:
        valid = lambda b: _blob_sha1(b) == entry["sha"]
    data = path.read_bytes() if path.exists() else b""
    if not valid(data):
        data = _request(S3 + DATASET + "/" + quote(relative, safe="/")).content
        if not valid(data):
            raise ValueError(f"Metadata differs from pinned dataset: {relative}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return data, {"path": relative, "bytes": len(data), "sha256": _sha256(data), "git_blob": entry["sha"]}


def parse_header(text: str) -> dict:
    sections: dict[str, dict[str, str]] = {}
    section = ""
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            sections.setdefault(section, {})
        elif "=" in line:
            key, value = line.split("=", 1)
            sections.setdefault(section, {})[key.strip()] = value.strip()
    common = sections["Common Infos"]
    if common["DataFormat"].upper() != "BINARY" or common["DataOrientation"].upper() != "MULTIPLEXED":
        raise ValueError("Only binary multiplexed BrainVision data are supported")
    binary = sections["Binary Infos"]
    if binary.get("UseBigEndianOrder", "NO").upper() in {"YES", "TRUE", "1"}:
        raise ValueError("Big-endian BrainVision input is not supported by this adapter")
    dtypes = {"IEEE_FLOAT_32": "<f4", "INT_16": "<i2", "UINT_16": "<u2", "INT_32": "<i4"}
    dtype = np.dtype(dtypes[binary["BinaryFormat"].upper()])
    channels, resolutions, units = [], [], []
    for index in range(1, int(common["NumberOfChannels"]) + 1):
        fields = sections["Channel Infos"][f"Ch{index}"].split(",")
        channels.append(fields[0].replace(r"\1", ","))
        resolutions.append(float(fields[2]) if len(fields) > 2 and fields[2] else 1.0)
        units.append(fields[3] if len(fields) > 3 else "")
    if len(set(channels)) != len(channels):
        raise ValueError("Duplicate source channel names")
    return {"channels": channels, "resolutions": np.asarray(resolutions), "units": units,
            "dtype": dtype, "sfreq_header": 1e6 / float(common["SamplingInterval"]),
            "data_file": common["DataFile"]}


def canonical_pair(value: str) -> str:
    parts = str(value).strip().split("-")
    if len(parts) != 2 or any(not part for part in parts):
        raise ValueError(f"Unsupported stimulation pair: {value!r}")
    return "-".join(sorted(parts))


def _unit_factor(unit: str) -> float:
    normalized = str(unit).strip().replace("μ", "u").replace("µ", "u").lower()
    try:
        return {"v": 1e6, "mv": 1e3, "uv": 1.0, "nv": 1e-3}[normalized]
    except KeyError as exc:
        raise ValueError(f"Unrecognized signal unit {unit!r}; refusing to guess") from exc


def fetch_window(url: str, first: int, stop: int, n_channels: int, dtype: np.dtype,
                 factors: np.ndarray, object_bytes: int, version_id: str | None) -> tuple[np.ndarray, dict]:
    frame_bytes = n_channels * dtype.itemsize
    start_byte, stop_byte = first * frame_bytes, stop * frame_bytes
    response = _request(url, headers={"Range": f"bytes={start_byte}-{stop_byte - 1}"})
    expected_range = f"bytes {start_byte}-{stop_byte - 1}/{object_bytes}"
    if response.status_code != 206 or response.headers.get("Content-Range") != expected_range:
        raise ValueError("Server did not honor the exact requested raw byte range")
    if version_id is not None and response.headers.get("x-amz-version-id") != version_id:
        raise ValueError("Raw S3 object version changed during retrieval")
    payload = response.content
    if len(payload) != stop_byte - start_byte:
        raise ValueError("Incomplete raw range")
    values = np.frombuffer(payload, dtype=dtype).reshape(stop - first, n_channels)
    values = (values * factors[None, :]).astype(np.float32)
    return values, {"start_sample": first, "stop_sample_exclusive": stop,
                    "start_byte": start_byte, "stop_byte_exclusive": stop_byte,
                    "sha256": _sha256(payload), "bytes": len(payload),
                    "s3_version_id": response.headers.get("x-amz-version-id"),
                    "etag": response.headers.get("ETag")}


def prepare_subject(subject: str, metadata_root: Path, tree: dict, tmin: float, tmax: float) -> dict:
    prefix = f"sub-{subject}/"
    paths = [p for p in tree if p.startswith(prefix)]
    header_path = next(p for p in paths if p.endswith(".vhdr"))
    stem = header_path.removesuffix("_ieeg.vhdr")
    proofs = []
    def read(path: str) -> bytes:
        data, proof = verified_metadata(path, metadata_root, tree)
        proofs.append(proof)
        return data
    header = parse_header(read(header_path).decode("utf-8-sig"))
    info = json.loads(read(stem + "_ieeg.json"))
    nominal_sfreq = float(info["SamplingFrequency"])
    if abs(header["sfreq_header"] - nominal_sfreq) > 0.001:
        raise ValueError("Sampling rates in header and sidecar disagree")
    # Reproduce the archived ER-detect/ieegprep sample grid, including the
    # finite precision of SamplingInterval in the BrainVision header.
    sfreq = float(header["sfreq_header"])
    channel_table = pd.read_csv(io.BytesIO(read(stem + "_channels.tsv")), sep="\t", keep_default_na=False)
    if channel_table["name"].duplicated().any():
        raise ValueError("Duplicate channels.tsv labels")
    lookup = channel_table.set_index("name")
    if any(channel not in lookup.index for channel in header["channels"]):
        raise ValueError("A source channel is absent from channels.tsv")
    ordered = lookup.loc[header["channels"]]
    units = [u or str(ordered.iloc[i]["units"]) for i, u in enumerate(header["units"])]
    output_units, unit_factors = [], []
    for i, unit in enumerate(units):
        try:
            unit_factors.append(_unit_factor(unit))
            output_units.append("uV")
        except ValueError:
            if str(ordered.iloc[i]["type"]).upper() in {"ECOG", "SEEG", "EEG", "DBS"}:
                raise
            # Auxiliary triggers/markers can be dimensionless. Preserve their
            # numbers and units, and never advertise them as microvolts.
            unit_factors.append(1.0)
            output_units.append(str(unit))
    factors = header["resolutions"] * np.asarray(unit_factors)
    source_good = ordered["status"].str.lower().eq("good")
    ecog_good = set(ordered.index[source_good & ordered["type"].str.upper().eq("ECOG")])
    events = pd.read_csv(io.BytesIO(read(stem + "_events.tsv")), sep="\t", keep_default_na=False)
    events["source_event_row"] = np.arange(len(events))
    event_mask = events["trial_type"].str.lower().eq("electrical_stimulation")
    if "status" in events:
        event_mask &= events["status"].str.lower().eq("good")
    events = events.loc[event_mask].copy()
    events["canonical_pair"] = events["electrical_stimulation_site"].map(canonical_pair)
    events["onset"] = pd.to_numeric(events["onset"], errors="raise")
    if not np.isfinite(events["onset"]).all():
        raise ValueError("Nonfinite stimulation event onset")
    # Read label identities only; response values do not enter retrieval selection.
    annotation_paths = sorted(p for p in tree if p.startswith(f"derivatives/annots/sub-{subject}_") and p.endswith(".mat"))
    annotated_pairs: set[str] = set()
    for path in annotation_paths:
        annot = loadmat(io.BytesIO(read(path)), simplify_cells=True)["annot"]
        annotated_pairs.update(canonical_pair(p) for p in np.atleast_1d(annot["stimpairs"]))
    raw_relative = str(Path(header_path).parent / header["data_file"])
    pointer = _request(GIT + quote(raw_relative, safe="/")).content
    if _blob_sha1(pointer) != tree[raw_relative]["sha"]:
        raise ValueError("Raw annex pointer differs from pinned tree")
    match = re.search(rb"SHA256E-s(\d+)--([0-9a-f]{64})", pointer)
    if not match:
        raise ValueError("Raw object lacks SHA256 annex key")
    object_bytes, full_sha = int(match[1]), match[2].decode()
    base_url = S3 + DATASET + "/" + quote(raw_relative, safe="/")
    remote = _request(base_url, method="HEAD")
    if int(remote.headers["Content-Length"]) != object_bytes:
        raise ValueError("Raw S3 length differs from pinned annex key")
    version = remote.headers.get("x-amz-version-id")
    url = base_url + ("?" + urlencode({"versionId": version}) if version else "")
    n_channels = len(header["channels"])
    frame_bytes = header["dtype"].itemsize * n_channels
    # The pinned UMCU25 object ends partway through its final sample frame.
    # Retain complete earlier frames; no partial-frame data are decoded and
    # events needing missing samples are explicitly excluded below.
    trailing_bytes = object_bytes % frame_bytes
    complete_samples = object_bytes // frame_bytes
    sample_offsets = np.arange(round(tmin * sfreq), round(tmax * sfreq) + 1, dtype=int)
    times = sample_offsets / sfreq
    groups, exclusions = [], []
    for pair in sorted(annotated_pairs):
        if any(channel not in ecog_good for channel in pair.split("-")):
            exclusions.append({"stimpair": pair, "reason": "stimulation_contact_not_good_ecog"})
            continue
        rows = events.loc[events["canonical_pair"].eq(pair)].sort_values(["onset", "source_event_row"])
        if rows.empty:
            exclusions.append({"stimpair": pair, "reason": "no_available_good_stimulation_events"})
            continue
        trials = []
        for row in rows.to_dict("records"):
            onset_sample = int(round(float(row["onset"]) * sfreq))
            first, stop = onset_sample + int(sample_offsets[0]), onset_sample + int(sample_offsets[-1]) + 1
            if first < 0 or stop > object_bytes // frame_bytes:
                exclusions.append({"stimpair": pair, "source_event_row": int(row["source_event_row"]),
                                   "reason": "epoch_out_of_source_bounds"})
                continue
            trials.append({"onset": float(row["onset"]), "onset_sample": onset_sample,
                           "source_event_row": int(row["source_event_row"]), "first": first, "stop": stop,
                           "original_stimulation_site": row["electrical_stimulation_site"]})
        if trials:
            groups.append((pair, trials))
    return {"subject": subject, "header": header, "sfreq": sfreq, "times": times,
            "channel_types": ordered["type"].to_numpy(dtype=str), "channel_status": ordered["status"].to_numpy(dtype=str),
            "channel_units": np.asarray(output_units, dtype=str),
            "factors": factors, "url": url, "object_bytes": object_bytes, "version": version,
            "groups": groups, "exclusions": exclusions,
            "provenance": {"dataset": DATASET, "snapshot": "1.0.0", "commit": COMMIT,
                "cache_schema_version": 2, "alignment_policy": "archived_ieegprep_brainvision_header_rate",
                "raw_path": raw_relative, "raw_s3_url": url, "raw_s3_version_id": version,
                "raw_full_sha256_from_annex_not_recomputed": full_sha, "raw_object_bytes": object_bytes,
                "raw_complete_samples": complete_samples, "raw_trailing_partial_frame_bytes_ignored": trailing_bytes,
                "raw_duration_complete_frames_s": complete_samples / sfreq,
                "sidecar_recording_duration_s": info.get("RecordingDuration"),
                "metadata_checksums": proofs, "event_status_available": "status" in events,
                "annotation_files": annotation_paths, "coordinates_available": False,
                "sample_rate_hz": sfreq, "sample_rate_header_hz": header["sfreq_header"],
                "sample_rate_nominal_sidecar_hz": nominal_sfreq,
                "sample_rounding": "nearest integer using Python round; event-relative integer sample offsets",
                "unit": "per-channel: voltage channels uV, auxiliary channels in declared units",
                "output_units": output_units, "source_units": units, "source_resolution": header["resolutions"].tolist(),
                "channel_scale_to_uv": factors.tolist(), "dtype": header["dtype"].str,
                "orientation": "MULTIPLEXED", "epoch_requested_s": [tmin, tmax],
                "epoch_actual_s": [float(times[0]), float(times[-1])], "epoch_inclusive_end": True,
                "transforms": [], "baseline_normalized": False,
                "selection": "available annotation pair identities; good ECoG stimulation contacts; good events when supplied; no response-label values used",
                "recording_channel_policy": "all source channels retained; channel type/status stored for explicit downstream family selection"}}


def cache_pair(prepared: dict, pair: str, trials: list[dict], output_root: Path) -> dict:
    folder = output_root / prepared["subject"]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"sub-{prepared['subject']}__{pair}.npz"
    manifest_path = path.with_suffix(".json")
    provenance = dict(prepared["provenance"], subject=prepared["subject"], stimpair=pair)
    fingerprint = _sha256(json.dumps({"provenance": provenance, "trials": trials}, sort_keys=True).encode())
    reusable_values, reusable_ranges = None, {}
    if path.exists() and manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        existing_hash_valid = _sha256(path.read_bytes()) == existing.get("npz_sha256")
        if existing.get("input_fingerprint") == fingerprint and existing_hash_valid:
            return dict(existing, reused=True)
        same_source = all(existing.get(k) == provenance.get(k) for k in (
            "raw_full_sha256_from_annex_not_recomputed", "raw_s3_version_id",
            "raw_object_bytes", "channel_scale_to_uv", "dtype"))
        if existing_hash_valid and same_source:
            with np.load(path, allow_pickle=False) as old:
                if list(old["channels"]) == prepared["header"]["channels"]:
                    reusable_values = old["trials"]
            if reusable_values is not None:
                for index, record in enumerate(existing["ranges"]):
                    key = (record["first"], record["stop"], record["source_event_row"])
                    reusable_ranges[key] = (index, record)
    arrays, ranges = [], []
    reused_trial_ranges, newly_downloaded_bytes = 0, 0
    for trial in trials:
        key = (trial["first"], trial["stop"], trial["source_event_row"])
        if key in reusable_ranges:
            index, record = reusable_ranges[key]
            values = reusable_values[index]
            reused_trial_ranges += 1
        else:
            values, record = fetch_window(prepared["url"], trial["first"], trial["stop"],
                len(prepared["header"]["channels"]), prepared["header"]["dtype"],
                prepared["factors"], prepared["object_bytes"], prepared["version"])
            newly_downloaded_bytes += record["bytes"]
        arrays.append(values)
        ranges.append(dict(record, **trial))
    values = np.stack(arrays)
    provenance.update(n_trials=len(trials), shape=list(values.shape), input_fingerprint=fingerprint,
                      source_trials_below_erpy_minimum_8=len(trials) < 8, ranges=ranges)
    temporary = path.with_suffix(".npz.partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, trials=values, times=prepared["times"],
            channels=np.asarray(prepared["header"]["channels"], dtype=str),
            channel_types=prepared["channel_types"], channel_status=prepared["channel_status"],
            channel_units=prepared["channel_units"],
            subject=prepared["subject"], stimpair=pair, sfreq=prepared["sfreq"],
            event_onsets=np.asarray([t["onset"] for t in trials]),
            metadata_json=json.dumps(provenance, sort_keys=True))
    temporary.replace(path)
    report = dict(provenance, npz_path=str(path), npz_sha256=_sha256(path.read_bytes()),
                  downloaded_bytes=sum(r["bytes"] for r in ranges),
                  downloaded_bytes_this_run=newly_downloaded_bytes,
                  reused_trial_ranges_this_run=reused_trial_ranges)
    manifest_path.write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", choices=SUBJECTS, default=list(SUBJECTS))
    parser.add_argument("--tmin", type=float, default=-1.0)
    parser.add_argument("--tmax", type=float, default=0.35)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-pairs", type=int, default=0, help="Deterministic smoke-check limit per subject; zero means all")
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    if args.tmin >= args.tmax or args.workers < 1 or args.max_pairs < 0:
        parser.error("Invalid epoch, worker count or pair limit")
    tree_json = json.loads((args.metadata_root / "dataset_tree.json").read_text())
    if tree_json["sha"] != COMMIT or tree_json.get("truncated"):
        raise ValueError("Requires complete pinned dataset tree")
    tree = {x["path"]: x for x in tree_json["tree"] if x["type"] == "blob"}
    args.output_root.mkdir(parents=True, exist_ok=True)
    for subject in args.subjects:
        prepared = prepare_subject(subject, args.metadata_root, tree, args.tmin, args.tmax)
        groups = prepared["groups"][:args.max_pairs] if args.max_pairs else prepared["groups"]
        inventory = {"subject": subject, "groups": [{"stimpair": p, "n_trials": len(t)} for p, t in prepared["groups"]],
                     "exclusions": prepared["exclusions"], "provenance": prepared["provenance"],
                     "run_max_pairs": args.max_pairs, "run_inventory_only": args.inventory_only}
        (args.output_root / f"{subject}_inventory.json").write_text(json.dumps(inventory, indent=2))
        print(json.dumps({"subject": subject, "pairs_available": len(prepared["groups"]), "pairs_scheduled": len(groups),
                          "channels": len(prepared["header"]["channels"]), "times": len(prepared["times"])}), flush=True)
        if args.inventory_only:
            continue
        completed, failures = [], []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(cache_pair, prepared, pair, trials, args.output_root): pair for pair, trials in groups}
            for future in as_completed(futures):
                pair = futures[future]
                try:
                    result = future.result()
                    completed.append({k: result[k] for k in ["stimpair", "shape", "npz_path", "npz_sha256", "downloaded_bytes"]})
                    print(json.dumps({"subject": subject, "stimpair": pair, "shape": result["shape"],
                                      "reused": result.get("reused", False)}), flush=True)
                except Exception as exc:
                    failures.append({"stimpair": pair, "error": repr(exc)})
                    print(json.dumps({"subject": subject, "stimpair": pair, "error": repr(exc)}), flush=True)
        (args.output_root / f"{subject}_retrieval.json").write_text(json.dumps({"completed": completed, "failures": failures}, indent=2))
        if failures:
            raise RuntimeError(f"{subject}: {len(failures)} pair retrievals failed; successful caches are reusable")


if __name__ == "__main__":
    main()
