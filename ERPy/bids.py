from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml


SUPPORTED_RAW_EXTS = (".edf", ".bdf", ".vhdr", ".eeg", ".csv", ".tsv")


@dataclass
class BIDSImportResult:
    """Files and row counts produced by ``import_bids_project``.

    The path fields identify the generated ERPy configuration and metadata
    tables. The count fields report how many raw recordings, stimulation rows,
    and electrode rows were imported for a quick completeness check.
    """

    config_path: Path
    raw_metadata_path: Path
    stim_metadata_path: Path
    electrode_metadata_path: Path
    n_raw_files: int
    n_stim_rows: int
    n_electrodes: int


def normalize_subject(subject: str) -> str:
    return str(subject).replace("sub-", "")


def normalize_session(session: str | None) -> str:
    if session is None:
        return "A"
    return str(session).replace("ses-", "")


def normalize_stim_pair(site: str) -> str:
    text = str(site)
    match = re.match(r"([A-Za-z]+)0?(\d+)[-_]([A-Za-z]+)?0?(\d+)", text)
    if match:
        lead1, a, lead2, b = match.groups()
        lead = lead1 if not lead2 or lead2 == lead1 else f"{lead1}{lead2}"
        return f"{lead}_{int(a)}_{int(b)}"
    match = re.match(r"([A-Za-z]+)(\d+)[- ]+([A-Za-z]+)(\d+)", text)
    if match:
        a_lead, a, b_lead, b = match.groups()
        lead = a_lead if a_lead == b_lead else f"{a_lead}{b_lead}"
        return f"{lead}_{int(a)}_{int(b)}"
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_") or "STIM_1_2"


def bids_sidecar_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def discover_bids_ieeg(
    bids_root: str | Path,
    subject: str | None = None,
    session: str | None = None,
    task: str | None = None,
    derivatives: bool = True,
) -> pd.DataFrame:
    """Discover BIDS iEEG recordings and their available sidecar files.

    The returned table contains one row per supported ``*_ieeg`` recording,
    including parsed subject, session, task, and run entities plus paths to
    matching events, channels, electrodes, and JSON sidecars when present.
    """

    root = Path(bids_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    patterns = []
    sub_part = f"sub-{normalize_subject(subject)}" if subject else "sub-*"
    ses_part = f"ses-{normalize_session(session)}" if session else "ses-*"
    bases = [root]
    if derivatives and (root / "derivatives").exists():
        bases.extend([p for p in (root / "derivatives").iterdir() if p.is_dir()])
    for base in bases:
        patterns.append(base / sub_part / ses_part / "ieeg")
        patterns.append(base / sub_part / "ieeg")
    rows = []
    seen = set()
    for folder in patterns:
        for raw_path in folder.glob("*_ieeg.*"):
            if raw_path.suffix.lower() not in SUPPORTED_RAW_EXTS:
                continue
            if raw_path.suffix.lower() == ".eeg":
                # BrainVision is represented by the .vhdr header when present.
                vhdr = raw_path.with_suffix(".vhdr")
                if vhdr.exists():
                    continue
            if task and f"task-{task}" not in raw_path.name:
                continue
            key = raw_path.resolve()
            if key in seen:
                continue
            seen.add(key)
            ents = parse_bids_entities(raw_path.name)
            events = raw_path.with_name(raw_path.name.replace("_ieeg" + raw_path.suffix, "_events.tsv"))
            channels = raw_path.with_name(raw_path.name.replace("_ieeg" + raw_path.suffix, "_channels.tsv"))
            electrodes = find_electrodes_file(raw_path.parent, ents.get("sub"), ents.get("ses"))
            json_path = raw_path.with_suffix(".json")
            rows.append(
                {
                    "raw_file": raw_path,
                    "events_file": events if events.exists() else None,
                    "channels_file": channels if channels.exists() else None,
                    "electrodes_file": electrodes,
                    "json_file": json_path if json_path.exists() else None,
                    "subject": ents.get("sub"),
                    "session": ents.get("ses") or "A",
                    "task": ents.get("task"),
                    "run": ents.get("run"),
                }
            )
    return pd.DataFrame(rows)


def parse_bids_entities(name: str) -> dict[str, str]:
    entities = {}
    for part in Path(name).stem.split("_"):
        if "-" in part:
            key, value = part.split("-", 1)
            entities[key] = value
    return entities


def find_electrodes_file(folder: Path, subject: str | None, session: str | None) -> Path | None:
    candidates = []
    if subject and session:
        candidates.append(folder / f"sub-{subject}_ses-{session}_electrodes.tsv")
    if subject:
        candidates.append(folder / f"sub-{subject}_electrodes.tsv")
    candidates.extend(folder.glob("*_electrodes.tsv"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def import_bids_project(
    bids_root: str | Path,
    output_root: str | Path,
    subject: str,
    session: str | None = None,
    task: str | None = None,
    patient_id: str | None = None,
    copy_raw: bool = False,
    derivatives: bool = True,
) -> BIDSImportResult:
    """Build an ERPy config/metadata project from BIDS/OpenNeuro iEEG files."""

    import shutil

    bids_root = Path(bids_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    patient_id = patient_id or normalize_subject(subject)
    discovered = discover_bids_ieeg(bids_root, subject=subject, session=session, task=task, derivatives=derivatives)
    if discovered.empty:
        raise FileNotFoundError(f"No BIDS iEEG files found in {bids_root} for subject={subject!r}")

    raw_dir = output_root / "raw"
    proc_dir = output_root / "procdata"
    meta_dir = output_root / "metadata"
    for d in (raw_dir, proc_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)

    raw_rows = []
    stim_rows = []
    event_rows_by_pair: dict[tuple[str, str], list[pd.DataFrame]] = {}
    sessions = set()
    for _, row in discovered.iterrows():
        raw_path = Path(row["raw_file"])
        erpy_session = normalize_session(row["session"])
        sessions.add(erpy_session)
        sidecar = bids_sidecar_json(Path(row["json_file"])) if row["json_file"] else {}
        sfreq = sidecar.get("SamplingFrequency")
        if sfreq is None and raw_path.suffix.lower() == ".vhdr":
            try:
                from .io import parse_brainvision_header

                sfreq = parse_brainvision_header(raw_path.read_text(errors="ignore"))["sfreq"]
            except Exception:
                pass
        erpy_raw_file = raw_path
        if copy_raw:
            erpy_raw_file = raw_dir / raw_path.name
            if not erpy_raw_file.exists():
                shutil.copy2(raw_path, erpy_raw_file)
                if raw_path.suffix.lower() == ".vhdr":
                    for ext in (".eeg", ".vmrk"):
                        side = raw_path.with_suffix(ext)
                        if side.exists():
                            shutil.copy2(side, raw_dir / side.name)
        raw_rows.append(
            {
                "patient_id": patient_id,
                "session_id": erpy_session,
                "raw_file": str(erpy_raw_file if Path(erpy_raw_file).is_absolute() else Path(erpy_raw_file).name),
                "start_time": "",
                "stop_time": "",
                "sampling_freq": sfreq,
                "source_bids_file": str(raw_path),
                "task": row.get("task"),
                "run": row.get("run"),
            }
        )
        events_file = row.get("events_file")
        if events_file is not None and Path(events_file).exists():
            events = pd.read_csv(events_file, sep="\t")
            if "trial_type" in events.columns:
                stim_mask = events["trial_type"].astype(str).str.contains("stim", case=False, na=False)
                if stim_mask.any():
                    events = events[stim_mask].copy()
            if "status" in events.columns:
                events = events[events["status"].fillna("good").astype(str).str.lower() != "bad"].copy()
            if "electrical_stimulation_site" in events.columns and "onset" in events.columns:
                for site, sub in events.dropna(subset=["electrical_stimulation_site", "onset"]).groupby("electrical_stimulation_site"):
                    stim_pair = normalize_stim_pair(site)
                    onset = pd.to_numeric(sub["onset"], errors="coerce").dropna()
                    if onset.empty:
                        continue
                    freq = sub.get("electrical_stimulation_frequency")
                    if freq is not None and pd.to_numeric(freq, errors="coerce").notna().any():
                        stim_freq = float(pd.to_numeric(freq, errors="coerce").dropna().iloc[0])
                    elif len(onset) > 2:
                        stim_freq = float(1.0 / onset.diff().dropna().median())
                    else:
                        stim_freq = ""
                    stim_rows.append(
                        {
                            "patient_id": patient_id,
                            "session_id": erpy_session,
                            "stim_pair": stim_pair,
                            "stim_start": float(onset.min()),
                            "stim_stop": float(onset.max()),
                            "stim_freq": stim_freq,
                            "source_stim_site": site,
                            "source_events_file": str(events_file),
                            "source_raw_file": str(raw_path),
                        }
                    )
                    canonical = pd.DataFrame({"times": onset.to_numpy(dtype=float)})
                    event_rows_by_pair.setdefault((erpy_session, stim_pair), []).append(canonical)

    raw_meta = pd.DataFrame(raw_rows).drop_duplicates()
    stim_meta = pd.DataFrame(stim_rows).drop_duplicates(subset=["patient_id", "session_id", "stim_pair", "stim_start", "stim_stop"])
    elec_meta = build_electrode_metadata(discovered, patient_id)
    raw_meta_path = meta_dir / "raw_metadata.csv"
    stim_meta_path = meta_dir / "stim_metadata.csv"
    elec_meta_path = meta_dir / "electrode_metadata.csv"
    raw_meta.to_csv(raw_meta_path, index=False)
    stim_meta.to_csv(stim_meta_path, index=False)
    elec_meta.to_csv(elec_meta_path, index=False)

    config = {
        "rawdata_path": str(raw_dir if copy_raw else bids_root),
        "procdata_path": str(proc_dir),
        "rawdata_meta_path": str(raw_meta_path),
        "stim_meta_path": str(stim_meta_path),
        "elec_meta_path": str(elec_meta_path),
        "session_ids": {patient_id: sorted(sessions)},
    }
    config_path = output_root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    # Write canonical event files immediately. DataLoader can consume numeric
    # seconds as event times when raw files are also indexed in seconds.
    for (sid, stim_pair), frames in event_rows_by_pair.items():
        out = proc_dir / patient_id / sid / "events" / f"{stim_pair}_events.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(frames, ignore_index=True).drop_duplicates("times").sort_values("times").to_csv(out, index=False)

    return BIDSImportResult(
        config_path=config_path,
        raw_metadata_path=raw_meta_path,
        stim_metadata_path=stim_meta_path,
        electrode_metadata_path=elec_meta_path,
        n_raw_files=len(raw_meta),
        n_stim_rows=len(stim_meta),
        n_electrodes=len(elec_meta),
    )


def build_electrode_metadata(discovered: pd.DataFrame, patient_id: str) -> pd.DataFrame:
    rows = []
    seen = set()
    for _, row in discovered.iterrows():
        path = row.get("electrodes_file")
        session = normalize_session(row.get("session"))
        if path is None or not Path(path).exists():
            channels_file = row.get("channels_file")
            if channels_file is not None and Path(channels_file).exists():
                ch = pd.read_csv(channels_file, sep="\t")
                name_col = "name" if "name" in ch.columns else ch.columns[0]
                for name in ch[name_col].dropna().astype(str):
                    key = (session, name)
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append({"patient_id": patient_id, "session_id": session, "elec_label": name, "anat_label": "", "bad_channel": False})
            continue
        elec = pd.read_csv(path, sep="\t")
        name_col = "name" if "name" in elec.columns else elec.columns[0]
        for _, erow in elec.iterrows():
            name = str(erow[name_col])
            key = (session, name)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "patient_id": patient_id,
                    "session_id": session,
                    "elec_label": name,
                    "anat_label": erow.get("group", erow.get("region", "")),
                    "mni_x": erow.get("x", erow.get("mni_x", "")),
                    "mni_y": erow.get("y", erow.get("mni_y", "")),
                    "mni_z": erow.get("z", erow.get("mni_z", "")),
                    "bad_channel": False,
                    "source_electrodes_file": str(path),
                }
            )
    return pd.DataFrame(rows)
